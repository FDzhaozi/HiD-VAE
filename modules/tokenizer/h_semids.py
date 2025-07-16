import math
import torch
import os # Added for path manipulation

from data.tags_processed import ItemData
from data.tags_processed import SeqData
from data.tags_processed import RecDataset
from data.schemas import SeqBatch
from data.schemas import TokenizedSeqBatch
from data.utils import batch_to
from einops import rearrange
from einops import pack
from modules.utils import eval_mode
from modules.h_rqvae import HRqVae
from typing import List, Dict, Optional, Tuple
from torch import nn
from torch import Tensor
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler

BATCH_SIZE = 16

class HSemanticIdTokenizer(nn.Module):
    """
    使用 HRQVAE 模型将项目特征序列标记化为语义 ID 序列。
    支持标签预测功能。
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        n_layers: int = 3,
        n_cat_feats: int = 18,
        commitment_weight: float = 0.25,
        hrqvae_weights_path: Optional[str] = None,
        hrqvae_codebook_normalize: bool = False,
        hrqvae_sim_vq: bool = False,
        tag_alignment_weight: float = 0.5,
        tag_prediction_weight: float = 0.5,
        tag_class_counts: Optional[List[int]] = None,
        tag_embed_dim: int = 768,
        use_dedup_dim: bool = False,  # 使用去重维度
        use_concatenated_ids: bool = False,  # 新增参数，使用拼接模式
        use_interleaved_ids: bool = False # 新增参数，使用交错模式
    ) -> None:
        super().__init__()

        # 确保use_dedup_dim和use_concatenated_ids不能同时为True
        if use_dedup_dim and use_concatenated_ids:
            raise ValueError("use_dedup_dim和use_concatenated_ids不能同时为True，它们是互斥的")
        if use_dedup_dim and use_interleaved_ids:
            raise ValueError("use_dedup_dim和use_interleaved_ids不能同时为True，它们是互斥的")
        if use_concatenated_ids and use_interleaved_ids:
            raise ValueError("use_concatenated_ids和use_interleaved_ids不能同时为True，它们是互斥的")

        self.hrq_vae = HRqVae(
            input_dim=input_dim,
            embed_dim=output_dim,
            hidden_dims=hidden_dims,
            codebook_size=codebook_size,
            codebook_kmeans_init=False,
            codebook_normalize=hrqvae_codebook_normalize,
            codebook_sim_vq=hrqvae_sim_vq,
            n_layers=n_layers,
            n_cat_features=n_cat_feats,
            commitment_weight=commitment_weight,
            tag_alignment_weight=tag_alignment_weight,
            tag_prediction_weight=tag_prediction_weight,
            tag_class_counts=tag_class_counts,
            tag_embed_dim=tag_embed_dim
        )
        
        if hrqvae_weights_path is not None:
            self.hrq_vae.load_pretrained(hrqvae_weights_path)

        self.hrq_vae.eval()

        self.codebook_size = codebook_size
        self.n_layers = n_layers
        self.use_dedup_dim = use_dedup_dim  # 保存参数
        self.use_concatenated_ids = use_concatenated_ids  # 保存参数
        self.tag_class_counts = tag_class_counts  # 保存标签类别数量
        self.use_interleaved_ids = use_interleaved_ids # 保存参数
        self.reset()
    
    def _get_hits(self, query: Tensor, key: Tensor) -> Tensor:
        return (rearrange(key, "b d -> 1 b d") == rearrange(query, "b d -> b 1 d")).all(axis=-1)
    
    def reset(self):
        self.cached_ids = None
    
    @property
    def sem_ids_dim(self):
        # 根据使用的模式返回不同的维度值
        if self.use_dedup_dim:
            return self.n_layers + 1  # 语义ID层数 + 去重维度
        elif self.use_concatenated_ids and self.tag_class_counts is not None:
            # 注意：当使用拼接模式时，总维度是语义ID层数加上标签层数
            return self.n_layers + len(self.tag_class_counts)  # 语义ID层数 + 标签层数
        elif self.use_interleaved_ids and self.tag_class_counts is not None:
            # 交错模式下，总维度也是语义ID层数加上标签层数
            return self.n_layers + len(self.tag_class_counts)
        else:
            return self.n_layers  # 仅语义ID层数
    
    @torch.no_grad()
    @eval_mode
    def precompute_corpus_ids(self, movie_dataset: ItemData) -> Tensor:
        all_ids_list = [] # 用于收集所有处理后的ID
        
        # sampler = BatchSampler(
        #     SequentialSampler(range(len(movie_dataset))), batch_size=512, drop_last=False
        # )
        # 移除自定义的collate_fn，以便 DataLoader 返回字典或 SeqBatch 对象
        # 让 DataLoader 使用默认的 collate_fn
        dataloader = DataLoader(movie_dataset, batch_size=512, shuffle=False)

        for batch_data in dataloader:
            # 将数据移动到模型设备
            batch_on_device = batch_to(batch_data, self.hrq_vae.device)
            item_features = batch_on_device.x # 假设物品特征在 x 属性中

            # 1. 获取语义ID
            encoded_features = self.hrq_vae.encode(item_features)
            hrqvae_output = self.hrq_vae.get_semantic_ids(encoded_features)
            # semantic_ids 的形状应为 [batch_size, n_layers]
            semantic_ids = hrqvae_output.sem_ids 
            
            current_batch_ids = semantic_ids

            if self.use_concatenated_ids:
                # 2. 获取预测的标签ID
                # predict_tags 输入期望 [batch_size, feature_dim] 或 [batch_size, seq_len, feature_dim]
                # ItemData 通常是 [batch_size, feature_dim]
                predicted_tags_output = self.hrq_vae.predict_tags(item_features)
                predicted_tag_indices = predicted_tags_output['predictions'] # 形状: [batch_size, n_layers_tags]
                
                # 确保预测的标签索引和语义ID的批次大小一致
                if predicted_tag_indices.shape[0] != semantic_ids.shape[0]:
                    raise ValueError(f"语义ID批次大小 ({semantic_ids.shape[0]}) 与预测标签批次大小 ({predicted_tag_indices.shape[0]}) 不匹配")

                # 3. 拼接语义ID和预测的标签ID
                current_batch_ids = torch.cat([semantic_ids, predicted_tag_indices], dim=1)
            elif self.use_interleaved_ids:
                # 2. 获取预测的标签ID
                predicted_tags_output = self.hrq_vae.predict_tags(item_features)
                predicted_tag_indices = predicted_tags_output['predictions'] # 形状: [batch_size, n_layers_tags]

                if predicted_tag_indices.shape[0] != semantic_ids.shape[0]:
                    raise ValueError(f"语义ID批次大小 ({semantic_ids.shape[0]}) 与预测标签批次大小 ({predicted_tag_indices.shape[0]}) 不匹配")

                # 3. 交错拼接语义ID和预测的标签ID
                # semantic_ids: [B, n_layers_sem]
                # predicted_tag_indices: [B, n_layers_tag]
                # 目标: [B, n_layers_sem + n_layers_tag] where elements are [s1, t1, s2, t2, ...]
                
                n_sem = semantic_ids.shape[1]
                n_tag = predicted_tag_indices.shape[1]
                max_len = max(n_sem, n_tag)
                interleaved_ids_list = []
                for i in range(max_len):
                    if i < n_sem:
                        interleaved_ids_list.append(semantic_ids[:, i:i+1])
                    if i < n_tag:
                        interleaved_ids_list.append(predicted_tag_indices[:, i:i+1])
                current_batch_ids = torch.cat(interleaved_ids_list, dim=1)

            all_ids_list.append(current_batch_ids)

        if not all_ids_list:
            # 如果数据集为空，则返回一个空的张量或引发错误
            self.cached_ids = torch.empty(0, self.sem_ids_dim, device=self.hrq_vae.device, dtype=torch.long)
        else:
            # 合并所有批次的ID
            concatenated_ids = torch.cat(all_ids_list, dim=0)
        
            # 如果使用去重维度（注意：当前逻辑与拼接ID互斥，但保留框架）
            if self.use_dedup_dim: 
                dedup_dim_values = []
                # ...去重逻辑（如果需要与拼接ID结合，需要重新设计）...
                # 这里假设在拼接模式下，不去计算去重维度
                # 如果需要，这里需要基于 concatenated_ids 计算去重维度，然后拼接
                # For now, assuming dedup_dim is not used with concatenated_ids as per prior logic
                self.cached_ids = concatenated_ids 
            else:
                self.cached_ids = concatenated_ids
        
        print(f"预计算完成。缓存的ID形状: {self.cached_ids.shape if self.cached_ids is not None else 'None'}")
        if self.cached_ids is not None and self.cached_ids.numel() > 0:
            print(f"缓存ID样本 (前3条):\n{self.cached_ids[:3]}")
        
        return self.cached_ids
    
    @torch.no_grad()
    @eval_mode
    def exists_prefix(self, sem_id_prefix: Tensor) -> Tensor:
        if self.cached_ids is None:
            raise Exception("空缓存中找不到匹配项。")
    
        # 打印维度信息用于调试
        print(f"前缀形状: {sem_id_prefix.shape}, 缓存形状: {self.cached_ids.shape}")
        
        # 获取前缀长度，并且确保不超过缓存的维度
        prefix_length = min(sem_id_prefix.shape[-1], self.cached_ids.shape[-1])
        prefix_cache = self.cached_ids[:, :prefix_length]
        
        # 只使用前缀的前prefix_length个元素
        sem_id_prefix_truncated = sem_id_prefix[..., :prefix_length]
        
        print(f"截断后 - 前缀形状: {sem_id_prefix_truncated.shape}, 缓存前缀形状: {prefix_cache.shape}")
        
        out = torch.zeros(*sem_id_prefix.shape[:-1], dtype=bool, device=sem_id_prefix.device)
        
        # 批量前缀匹配以避免 OOM
        batches = math.ceil(sem_id_prefix.shape[0] // BATCH_SIZE)
        for i in range(batches):
            prefixes = sem_id_prefix_truncated[i*BATCH_SIZE:(i+1)*BATCH_SIZE,...]
            
            # 确保维度匹配
            if prefixes.shape[-1] == prefix_cache.shape[-1]:
                # 标准比较
                matches = (prefixes.unsqueeze(-2) == prefix_cache.unsqueeze(-3)).all(axis=-1).any(axis=-1)
            else:
                # 出现了异常情况，记录更多信息
                print(f"警告: 维度仍然不匹配! prefixes: {prefixes.shape}, prefix_cache: {prefix_cache.shape}")
                
                # 获取最小共同维度
                common_dims = min(prefixes.shape[-1], prefix_cache.shape[-1])
                
                # 使用共同维度比较
                matches = (prefixes[..., :common_dims].unsqueeze(-2) == 
                           prefix_cache[..., :common_dims].unsqueeze(-3)).all(axis=-1).any(axis=-1)
            
            out[i*BATCH_SIZE:(i+1)*BATCH_SIZE,...] = matches
        
        return out
    
    def _tokenize_seq_batch_from_cached(self, ids: Tensor) -> Tensor:
        """
        从缓存的语义ID中获取指定ID的语义表示
        
        参数:
            ids: 物品ID张量，形状为 [batch_size, seq_len]
            
        返回:
            语义ID张量，形状为 [batch_size, seq_len * sem_ids_dim]
            在拼接模式下，仅返回语义ID部分，标签ID会在forward方法中单独处理
        """
        # 确保所有ID都在缓存的范围内
        valid_ids = ids.clone()
        valid_ids[valid_ids >= self.cached_ids.shape[0]] = 0  # 将超出范围的ID替换为0
        
        # 获取语义ID（从缓存中）
        # 注意：在拼接模式下，缓存中只存储了语义ID部分，标签ID会在forward方法中单独处理
        return rearrange(self.cached_ids[valid_ids.flatten(), :], "(b n) d -> b (n d)", n=ids.shape[1])
    
    @torch.no_grad()
    @eval_mode
    def forward(self, batch: SeqBatch) -> TokenizedSeqBatch:
        # 如果缓存为空或批次 ID 超出缓存范围，则使用 HRQVAE 生成语义 ID
        if self.cached_ids is None or batch.ids.max() >= self.cached_ids.shape[0]:
            B, N = batch.ids.shape
            # D = self.sem_ids_dim # 总维度将在下面确定

            # --- 处理当前序列 (sem_ids) ---
            # 1. 获取语义ID
            encoded_x = self.hrq_vae.encode(batch.x)
            hrqvae_output = self.hrq_vae.get_semantic_ids(encoded_x) # batch.x 形状: [B, N, feat_dim]
            # hrqvae_output.sem_ids 的形状应为 [B*N, n_layers]
            actual_semantic_ids_flat = hrqvae_output.sem_ids 
            actual_semantic_ids = rearrange(actual_semantic_ids_flat, '(b n) d -> b n d', b=B, n=N)

            combined_ids_matrix = actual_semantic_ids

            if self.use_concatenated_ids:
                # 2. 获取预测的标签ID
                # predict_tags 输入期望 [B, N, feat_dim], 输出 'predictions' 形状 [B, N, n_tag_layers]
                predicted_tags_output = self.hrq_vae.predict_tags(batch.x)
                predicted_tag_indices = predicted_tags_output['predictions'] # [B, N, n_tag_layers]
                
                # 3. 拼接语义ID和预测的标签ID
                combined_ids_matrix = torch.cat([actual_semantic_ids, predicted_tag_indices], dim=2)
            elif self.use_interleaved_ids:
                # 2. 获取预测的标签ID
                predicted_tags_output = self.hrq_vae.predict_tags(batch.x)
                predicted_tag_indices = predicted_tags_output['predictions'] # [B, N, n_tag_layers]

                # 3. 交错拼接
                # actual_semantic_ids: [B, N, n_layers_sem]
                # predicted_tag_indices: [B, N, n_layers_tag]
                # 目标: [B, N, n_layers_sem + n_layers_tag] with interleaved IDs
                n_sem = actual_semantic_ids.shape[2]
                n_tag = predicted_tag_indices.shape[2]
                max_item_dim = max(n_sem, n_tag)
                interleaved_list_batch = []
                for i in range(max_item_dim):
                    if i < n_sem:
                        interleaved_list_batch.append(actual_semantic_ids[:, :, i:i+1])
                    if i < n_tag:
                        interleaved_list_batch.append(predicted_tag_indices[:, :, i:i+1])
                combined_ids_matrix = torch.cat(interleaved_list_batch, dim=2)

            # 将每个项目的ID序列（长度为D_total）展平并连接
            sem_ids = rearrange(combined_ids_matrix, 'b n d -> b (n d)')
            D_total = combined_ids_matrix.shape[2] # 更新总维度

            # --- 处理未来序列 (sem_ids_fut) ---
            sem_ids_fut = None
            if batch.x_fut is not None:
                # 1. 获取未来项目的语义ID
                # batch.x_fut 形状: [B, feat_dim], .unsqueeze(1) 后为 [B, 1, feat_dim]
                # hrqvae_output_fut.sem_ids 的形状应为 [B*1, n_layers] 即 [B, n_layers]
                encoded_x_fut = self.hrq_vae.encode(batch.x_fut.unsqueeze(1))
                hrqvae_output_fut = self.hrq_vae.get_semantic_ids(encoded_x_fut) 
                actual_semantic_ids_fut_flat = hrqvae_output_fut.sem_ids 
                actual_semantic_ids_fut = rearrange(actual_semantic_ids_fut_flat, '(b n) d -> b (n d)', b=B, n=1) # [B, n_layers]

                combined_ids_matrix_fut = actual_semantic_ids_fut

                if self.use_concatenated_ids:
                    # 2. 获取未来项目的预测标签ID
                    predicted_tags_output_fut = self.hrq_vae.predict_tags(batch.x_fut.unsqueeze(1))
                    predicted_tag_indices_fut = predicted_tags_output_fut['predictions'] # [B, 1, n_tag_layers]
                    predicted_tag_indices_fut = rearrange(predicted_tag_indices_fut, 'b n d -> b (n d)') # [B, n_tag_layers]
                    
                    # 3. 拼接
                    combined_ids_matrix_fut = torch.cat([actual_semantic_ids_fut, predicted_tag_indices_fut], dim=1)
                elif self.use_interleaved_ids:
                    # 2. 获取未来项目的预测标签ID
                    predicted_tags_output_fut = self.hrq_vae.predict_tags(batch.x_fut.unsqueeze(1))
                    predicted_tag_indices_fut = predicted_tags_output_fut['predictions'] # [B, 1, n_tag_layers]
                    predicted_tag_indices_fut = rearrange(predicted_tag_indices_fut, 'b n d -> b (n d)') # [B, n_tag_layers]

                    # 3. 交错拼接
                    # actual_semantic_ids_fut: [B, n_layers_sem]
                    # predicted_tag_indices_fut: [B, n_layers_tag]
                    n_sem_fut = actual_semantic_ids_fut.shape[1]
                    n_tag_fut = predicted_tag_indices_fut.shape[1]
                    max_fut_dim = max(n_sem_fut, n_tag_fut)
                    interleaved_list_fut = []
                    for i in range(max_fut_dim):
                        if i < n_sem_fut:
                            interleaved_list_fut.append(actual_semantic_ids_fut[:, i:i+1])
                        if i < n_tag_fut:
                            interleaved_list_fut.append(predicted_tag_indices_fut[:, i:i+1])
                    combined_ids_matrix_fut = torch.cat(interleaved_list_fut, dim=1)

                sem_ids_fut = combined_ids_matrix_fut
                # D_total_fut 应该是和 D_total 一样的
            
            seq_mask = batch.seq_mask.repeat_interleave(D_total, dim=1) if batch.seq_mask is not None else None
            if seq_mask is not None:
                sem_ids[~seq_mask] = -1
        else:
            # 从缓存中获取语义 ID (缓存中已经是拼接好的了)
            B, N = batch.ids.shape
            D_total = self.cached_ids.shape[-1] # 从缓存获取总维度
            
            sem_ids = self._tokenize_seq_batch_from_cached(batch.ids)
            # _tokenize_seq_batch_from_cached 返回 [B, N * D_total]
            
            seq_mask = batch.seq_mask.repeat_interleave(D_total, dim=1) if batch.seq_mask is not None else None
            if seq_mask is not None:
                sem_ids[~seq_mask] = -1
        
            # 处理未来ID (从缓存中获取)
            sem_ids_fut = self._tokenize_seq_batch_from_cached(batch.ids_fut)
            # _tokenize_seq_batch_from_cached 返回 [B, 1 * D_total] (因为未来ID通常是单个项目)

        # token_type_ids 应该基于 D_total
        token_type_ids = torch.arange(D_total, device=sem_ids.device).repeat(B, N)
        token_type_ids_fut = torch.arange(D_total, device=sem_ids.device).repeat(B, 1)
        
        # 打印shape信息用于调试
        print(f"sem_ids形状: {sem_ids.shape}, sem_ids_fut形状: {sem_ids_fut.shape if sem_ids_fut is not None else 'None'}")
        print(f"token_type_ids形状: {token_type_ids.shape}, token_type_ids_fut形状: {token_type_ids_fut.shape}")
        
        result = TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=sem_ids,
            sem_ids_fut=sem_ids_fut,
            seq_mask=seq_mask,
            token_type_ids=token_type_ids,
            token_type_ids_fut=token_type_ids_fut
        )
        
        # 打印样本的ID结构以进行调试
        # if self.use_concatenated_ids and B > 0 and sem_ids.numel() > 0 and (sem_ids_fut is None or sem_ids_fut.numel() > 0):
        #     sample_idx = 0
        #     print(f"样本ID结构 (索引 {sample_idx}):")
            
        #     n_sem_layers = self.n_layers
        #     # D_total 来自上面的计算

        #     # 打印当前ID - 修改为打印序列中所有项目
        #     if sem_ids is not None and sem_ids.shape[1] > 0:
        #         # sem_ids 已经是 [B, N * D_total]
        #         # N 是原始序列长度, D_total 是每个项目的总ID数
        #         if N > 0: # N is batch.ids.shape[1] (sequence length of items)
        #             print(f"  当前序列 (共 {N} 个项目):")
        #             for item_idx_in_seq in range(N):
        #                 start_idx = item_idx_in_seq * D_total
        #                 end_idx = (item_idx_in_seq + 1) * D_total
                        
        #                 if end_idx > sem_ids.shape[1]:
        #                     continue 
                            
        #                 if sem_ids[sample_idx, start_idx].item() == -1:
        #                     continue

        #                 current_item_ids_flat = sem_ids[sample_idx, start_idx:end_idx]
                        
        #                 if current_item_ids_flat.numel() == D_total:
        #                     current_sem_ids_part = current_item_ids_flat[:n_sem_layers]
        #                     current_tag_ids_part = current_item_ids_flat[n_sem_layers:]
        #                     print(f"    项目 {item_idx_in_seq + 1} - 语义ID: {current_sem_ids_part.tolist()}, 标签ID: {current_tag_ids_part.tolist()}")
        #                 else:
        #                     print(f"    项目 {item_idx_in_seq + 1}: ID数据不完整或被掩码 (预期长度 {D_total}, 实际 {current_item_ids_flat.numel()})")
        #         else:
        #             print("  当前序列为空或格式不正确。")
            
        #     # 打印未来ID
        #     if sem_ids_fut is not None and sem_ids_fut.shape[1] > 0:
        #         # sem_ids_fut 已经是 [B, 1 * D_total] 或 [B, D_total]
        #         # 取第一个（也是唯一一个）未来项目的ID
        #         if D_total > 0 : # Make sure D_total is valid
        #             future_item_ids_flat = sem_ids_fut[sample_idx, :D_total]

        #             future_sem_ids_part = future_item_ids_flat[:n_sem_layers]
        #             future_tag_ids_part = future_item_ids_flat[n_sem_layers:]
        #             print(f"  未来语义ID: {future_sem_ids_part.tolist()}")
        #             print(f"  未来标签ID: {future_tag_ids_part.tolist()}")
        #         else:
        #             print("  未来ID的D_total为0，无法解析。")
            
        #     # 检查数据中的原始标签索引 (如果存在且适用)
        #     if hasattr(batch, 'tags_indices') and batch.tags_indices is not None and self.tag_class_counts is not None and batch.tags_indices.shape[0] > sample_idx:
        #         num_tag_layers_original = len(self.tag_class_counts)
        #         if batch.tags_indices.dim() == 3 and N > 0: # [B, N, n_tag_layers_original]
        #             if batch.tags_indices.shape[2] >= num_tag_layers_original:
        #                 original_tags_for_sample = batch.tags_indices[sample_idx, 0, :num_tag_layers_original]
        #                 print(f"  原始标签索引 (来自数据批次, 第一个项目): {original_tags_for_sample.tolist()}")
        #         elif batch.tags_indices.dim() == 2: # [B, n_tag_layers_original]
        #             if batch.tags_indices.shape[1] >= num_tag_layers_original:
        #                 original_tags_for_sample = batch.tags_indices[sample_idx, :num_tag_layers_original]
        #                 print(f"  原始标签索引 (来自数据批次): {original_tags_for_sample.tolist()}")

        return result
    
    @torch.no_grad()
    @eval_mode
    @torch.no_grad()
    @eval_mode
    def predict_tags(self, batch: SeqBatch) -> Dict[str, Tensor]:
        """
        预测批次中项目的标签，忽略序列中的填充项（-1）
        
        参数:
            batch: 包含项目特征的批次
            
        返回:
            包含预测标签索引和置信度的字典
        """
        # 使用 HRQVAE 的标签预测功能
        print(f"the shape of batch.x is {batch.x.shape}")
        
        # 获取序列掩码，标识哪些位置是有效的（非填充）
        seq_mask = batch.seq_mask if hasattr(batch, 'seq_mask') else None
        
        if seq_mask is not None:
            # 如果有掩码，我们需要只处理有效的项目
            batch_size, seq_len, feat_dim = batch.x.shape
            
            # 创建一个掩码版本的特征张量，将填充位置的特征设为0
            # 这样不会影响预测结果，因为我们会在后面根据掩码过滤结果
            masked_x = batch.x.clone()
            
            # 将掩码扩展到特征维度
            expanded_mask = seq_mask.unsqueeze(-1).expand_as(masked_x)
            
            # 将填充位置的特征设为0
            masked_x = masked_x * expanded_mask
            
            # 使用处理后的特征进行预测
            predictions = self.hrq_vae.predict_tags(masked_x)
            
            # 处理预测结果，将填充位置的预测设为-1
            if 'predictions' in predictions:
                # 获取预测结果
                pred = predictions['predictions']
                # 创建掩码的扩展版本，匹配预测结果的形状
                # 预测结果形状为 [batch_size, seq_len, n_layers]
                expanded_pred_mask = seq_mask.unsqueeze(-1).expand_as(pred)
                # 创建填充值张量 (-1)
                fill_value = torch.full_like(pred, -1)
                # 根据掩码选择预测值或填充值
                predictions['predictions'] = torch.where(expanded_pred_mask, pred, fill_value)
            
            if 'confidences' in predictions:
                # 获取置信度
                conf = predictions['confidences']
                # 创建掩码的扩展版本，匹配置信度的形状
                expanded_conf_mask = seq_mask.unsqueeze(-1).expand_as(conf)
                # 创建填充值张量 (0.0)
                fill_value = torch.zeros_like(conf)
                # 根据掩码选择置信度或填充值
                predictions['confidences'] = torch.where(expanded_conf_mask, conf, fill_value)
        else:
            # 如果没有掩码，直接进行预测
            predictions = self.hrq_vae.predict_tags(batch.x)
        

            
        return predictions
    
    @torch.no_grad()
    @eval_mode
    def tokenize_with_tags(self, batch: SeqBatch) -> Tuple[TokenizedSeqBatch, Dict[str, Tensor]]:
        """
        将批次标记化并预测标签
        
        参数:
            batch: 包含项目特征的批次
            
        返回:
            标记化批次和标签预测结果的元组
        """
        tokenized_batch = self.forward(batch)
        tag_predictions = self.predict_tags(batch)
        return tokenized_batch, tag_predictions
        

if __name__ == "__main__":
    # 参数硬编码，不再使用argparse
    dataset_name_arg = 'beauty' # 'ml-1m' 或 'beauty'
    input_dim_arg = 768
    embed_dim_arg = 32
    hidden_dims_arg = [512, 256, 128]
    codebook_size_arg = 256
    n_cat_feats_arg = 0
    tag_embed_dim_arg = 768
    tag_alignment_weight_arg = 0.5 # 默认值
    tag_prediction_weight_arg = 0.5 # 默认值
    use_dedup_dim_arg = False
    # 新增参数：使用拼接模式
    use_concatenated_ids_arg = True  # 默认开启拼接模式
    # 新增参数：使用交错模式
    use_interleaved_ids_arg = False # 默认关闭交错模式
    # 硬编码模型路径
    hrqvae_weights_path_arg = "out/hrqvae/amazon/hrqvae_AMAZON_20250524_212758/hrqvae_model_ACC0.7658_RQLOSS0.3243_20250525_031159.pt"
    n_layers_arg = 3 # 根据tag_class_counts推断或默认

    print(f"测试数据集: {dataset_name_arg}")
    print(f"模型参数: input_dim={input_dim_arg}, embed_dim={embed_dim_arg}, "
          f"hidden_dims={hidden_dims_arg}, codebook_size={codebook_size_arg}, "
          f"n_cat_feats={n_cat_feats_arg}, tag_embed_dim={tag_embed_dim_arg}, "
          f"use_dedup_dim={use_dedup_dim_arg}, n_layers={n_layers_arg}")
    
    if dataset_name_arg == 'ml-1m':
        dataset_path = "dataset/ml-1m-movie"
        seq_dataset_path = "dataset/ml-1m"
        dataset_enum = RecDataset.ML_1M
        original_tag_class_counts_for_remapping = [18, 7, 20][:n_layers_arg] # Example, adjust if necessary
        # 这些需要与模型训练时一致
        tag_class_counts_arg = [18, 7, 20][:n_layers_arg] # This would be remapped counts if ml-1m used remapping
    else:  # beauty (amazon)
        dataset_path = "dataset/amazon"
        seq_dataset_path = "dataset/amazon"
        dataset_enum = RecDataset.AMAZON
        original_tag_class_counts_for_remapping = [6, 130, 927][:n_layers_arg] # Original counts for Amazon Beauty
        # 这些需要与模型训练时一致, 根据错误信息调整 (these are remapped counts model expects)
        tag_class_counts_arg = [7, 30, 97][:n_layers_arg] # MODIFIED
    
    # 加载数据集
    print(f"加载物品数据集: {dataset_path}")
    dataset = ItemData(dataset_path, dataset=dataset_enum, split="beauty" if dataset_name_arg == 'beauty' else None)

    # --- 对加载的 dataset.tags_indices 执行标签重映射 --- 
    if dataset_name_arg == 'beauty' and hasattr(dataset, 'tags_indices') and dataset.tags_indices is not None:
        print("对 Amazon Beauty 数据集的标签索引执行重映射...")
        # 确定 rare_tags.pt 的路径
        # Предполагаемый save_dir_root во время обучения: out/hrqvae/amazon/
        # Модель сохранена в save_dir_root/run_specific_folder/model.pt
        # Файл rare_tags.pt сохраняется в save_dir_root/special_tags_files/rare_tags.pt
        model_dir = os.path.dirname(hrqvae_weights_path_arg) # .../hrqvae_AMAZON_20250524_212758
        save_dir_root_guess = os.path.dirname(model_dir) # .../amazon
        # Check if this is the second level 'amazon' or the first 'hrqvae/amazon'
        if os.path.basename(save_dir_root_guess) == dataset_name_arg: # e.g., ends with /amazon
             # This seems to be the project-level dataset folder, not the run output base
             # Let's assume save_dir_root was 'out/hrqvae/amazon/' during training
             # which means rare_tags.pt is relative to that. 
             # The model path is 'out/hrqvae/amazon/RUN_FOLDER/model.pt'
             # So, the save_dir_root that created special_tags_files is likely 'out/hrqvae/amazon/'
            rare_tags_base_path = save_dir_root_guess # 'out/hrqvae/amazon/'
        else:
            # Fallback if the above logic is not perfect, assume 'out/hrqvae/amazon/' structure based on model path segments
            path_parts = hrqvae_weights_path_arg.split(os.sep)
            if "hrqvae" in path_parts and "amazon" in path_parts:
                hrqvae_idx = path_parts.index("hrqvae")
                amazon_idx = path_parts.index("amazon")
                if amazon_idx == hrqvae_idx + 1:
                    rare_tags_base_path = os.path.join(*path_parts[:amazon_idx+1])
                else:
                    rare_tags_base_path = "out/hrqvae/amazon" # Default guess
            else:
                rare_tags_base_path = "out/hrqvae/amazon" # Default guess

        rare_tags_path = os.path.join(rare_tags_base_path, "special_tags_files", "rare_tags.pt")
        
        print(f"尝试从以下路径加载稀有标签字典: {rare_tags_path}")

        if not os.path.exists(rare_tags_path):
            print(f"警告: 稀有标签文件 {rare_tags_path} 未找到。跳过标签重映射。如果模型使用重映射标签进行训练，这可能会导致错误。")
        else:
            rare_tags_dict = torch.load(rare_tags_path, map_location=torch.device('cpu'))
            print(f"已加载稀有标签字典: {list(rare_tags_dict.keys())}")

            # `tag_class_counts_arg` is the remapped_tag_class_counts
            # `original_tag_class_counts_for_remapping` is the original
            for i in range(n_layers_arg):
                if i in rare_tags_dict and rare_tags_dict[i].numel() > 0:
                    layer_indices_tensor = dataset.tags_indices[:, i]
                    remapped_num_classes_for_layer = tag_class_counts_arg[i]
                    original_num_classes_for_layer = original_tag_class_counts_for_remapping[i]
                    
                    special_class_id = remapped_num_classes_for_layer - 1
                    
                    id_mapping = torch.full((original_num_classes_for_layer,), -1, dtype=torch.long) # Initialize with -1
                    
                    current_rare_tags = rare_tags_dict[i].long()
                    non_rare_mask = torch.ones(original_num_classes_for_layer, dtype=torch.bool)
                    if current_rare_tags.numel() > 0:
                         # Ensure current_rare_tags are within bounds for non_rare_mask
                        valid_rare_tags = current_rare_tags[current_rare_tags < original_num_classes_for_layer]
                        if len(valid_rare_tags) < len(current_rare_tags):
                            print(f"警告: 层 {i} 的稀有标签索引超出了原始类别数量的界限 ({original_num_classes_for_layer}).")
                        if valid_rare_tags.numel() > 0:
                            non_rare_mask[valid_rare_tags] = False

                    new_current_id = 0
                    for orig_id in range(original_num_classes_for_layer):
                        if non_rare_mask[orig_id]:
                            id_mapping[orig_id] = new_current_id
                            new_current_id += 1
                        else:
                            id_mapping[orig_id] = special_class_id
                    
                    # Verify that new_current_id matches remapped_num_classes_for_layer - 1 (if any non-rare) or 0 (if all rare)
                    if new_current_id > special_class_id and special_class_id != -1 : # special_class_id can be -1 if remapped_num_classes is 0
                        print(f"警告: 层 {i} 重映射后的ID计数 ({new_current_id}) 与期望的不符 ({special_class_id}). 可能所有标签都是稀有的或配置错误。")
                    elif new_current_id == 0 and original_num_classes_for_layer > 0 and not torch.all(~non_rare_mask):
                        # This case means no non-rare items were found, yet not all items were declared rare. This can happen if original_num_classes_for_layer is small. 
                        pass # Allow this, might just mean all are rare and map to special_class_id

                    valid_mask_for_layer = (layer_indices_tensor >= 0) & (layer_indices_tensor < original_num_classes_for_layer)
                    invalid_indices_present = (layer_indices_tensor >= original_num_classes_for_layer).any()
                    if invalid_indices_present:
                        print(f"警告: 层 {i} 的原始标签索引包含超出预期最大值 {original_num_classes_for_layer-1} 的值。这些值将不被映射或可能导致错误。")

                    # Create a temporary tensor for new indices to avoid in-place issues with advanced indexing
                    new_layer_indices = layer_indices_tensor.clone()
                    # Apply mapping only to valid original indices
                    indices_to_map = layer_indices_tensor[valid_mask_for_layer].long()
                    mapped_values = id_mapping[indices_to_map]
                    new_layer_indices[valid_mask_for_layer] = mapped_values
                    dataset.tags_indices[:, i] = new_layer_indices
                    
                    print(f"层 {i} 的 tags_indices 已被重映射。特殊类别 ID: {special_class_id}")
                else:
                    print(f"层 {i} 无需重映射 (rare_tags_dict中无此层或稀有标签为空)。")
            print("标签重映射完成。")
    # --- 标签重映射结束 ---
    
    # 创建标记器
    print("初始化标记器...")
    tokenizer = HSemanticIdTokenizer(
        input_dim=input_dim_arg,
        output_dim=embed_dim_arg,
        hidden_dims=hidden_dims_arg,
        codebook_size=codebook_size_arg,
        n_cat_feats=n_cat_feats_arg,
        n_layers=n_layers_arg, # 确保与模型一致
        hrqvae_weights_path=hrqvae_weights_path_arg,
        hrqvae_codebook_normalize=True, # MODIFIED from False
        hrqvae_sim_vq=False, # 默认值
        tag_alignment_weight=tag_alignment_weight_arg,
        tag_prediction_weight=tag_prediction_weight_arg,
        tag_class_counts=tag_class_counts_arg, # 确保与模型一致
        tag_embed_dim=tag_embed_dim_arg,
        use_dedup_dim=use_dedup_dim_arg,
        use_concatenated_ids=use_concatenated_ids_arg,  # 新增参数
        use_interleaved_ids=use_interleaved_ids_arg # 新增参数
    )
    
    # 预计算语义ID
    print("预计算语义ID...")
    tokenizer.precompute_corpus_ids(dataset)
    
    # 加载序列数据并测试标记器
    print(f"加载序列数据: {seq_dataset_path}")
    seq_data = SeqData(seq_dataset_path, dataset=dataset_enum, split="beauty" if dataset_name_arg == 'beauty' else None)
    
    # 获取一小批数据进行测试
    sample_size = 5
    print(f"获取 {sample_size} 条测试批次...")
    if len(seq_data) < sample_size:
        print(f"警告: 序列数据样本不足 {sample_size} 条，实际获取 {len(seq_data)} 条。")
        sample_size = len(seq_data)

    if sample_size == 0:
        print("错误: 没有可用的序列数据进行测试。")
        exit()
        
    batch = seq_data[:sample_size]

    print("\n测试批次样本 (前5条):")
    for i in range(min(sample_size, 5)):
        print(f"  样本 {i+1}:")
        print(f"    用户ID: {batch.user_ids[i]}")
        print(f"    物品ID: {batch.ids[i]}")
        print(f"    物品特征 (x) 形状: {batch.x[i].shape}")
        if hasattr(batch, 'tags_emb') and batch.tags_emb is not None:
             print(f"    标签嵌入 (tags_emb) 形状: {batch.tags_emb[i].shape if batch.tags_emb is not None else 'N/A'}")
        if hasattr(batch, 'tags_indices') and batch.tags_indices is not None:
            print(f"    标签索引 (tags_indices): {batch.tags_indices[i] if batch.tags_indices is not None else 'N/A'}")

    
    # 使用标记器处理批次
    print("\n使用标记器生成语义ID...")
    tokenized = tokenizer(batch)
    
    print("\n语义ID结果:")
    print(f"用户ID: {tokenized.user_ids}")
    print(f"语义ID形状: {tokenized.sem_ids.shape}")
    print(f"语义ID (前5条): \n{tokenized.sem_ids[:5]}")
    
    # 测试标签预测功能
    print("\n测试标签预测功能...")
    tag_predictions = tokenizer.predict_tags(batch) # predict_tags 内部会处理序列
    
    print("\n标签预测结果:")
    if 'predictions' in tag_predictions and tag_predictions['predictions'] is not None:
        print(f"预测标签索引形状: {tag_predictions['predictions'].shape}")
        print(f"预测标签索引 (前5条): \n{tag_predictions['predictions'][:5]}")
    else:
        print("未找到预测标签索引。")
        
    if 'confidences' in tag_predictions and tag_predictions['confidences'] is not None:
        print(f"预测置信度形状: {tag_predictions['confidences'].shape}")
        print(f"预测置信度 (前5条): \n{tag_predictions['confidences'][:5]}")
    else:
        print("未找到预测置信度。")

    # 打印真实标签以供比较 (如果可用)
    if hasattr(batch, 'tags_indices') and batch.tags_indices is not None:
        print("\n真实标签索引 (供比较, 前5条):")
        print(batch.tags_indices[:5])
    
    print("\n测试完成!")