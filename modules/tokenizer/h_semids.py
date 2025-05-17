import math
import torch

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
        use_dedup_dim: bool = False  # 新增参数，控制是否使用去重维度
    ) -> None:
        super().__init__()

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
        self.reset()
    
    def _get_hits(self, query: Tensor, key: Tensor) -> Tensor:
        return (rearrange(key, "b d -> 1 b d") == rearrange(query, "b d -> b 1 d")).all(axis=-1)
    
    def reset(self):
        self.cached_ids = None
    
    @property
    def sem_ids_dim(self):
        # 根据是否使用去重维度返回不同的维度值
        return self.n_layers + (1 if self.use_dedup_dim else 0)
    
    @torch.no_grad()
    @eval_mode
    def precompute_corpus_ids(self, movie_dataset: ItemData) -> Tensor:
        cached_ids = None
        dedup_dim = []
        sampler = BatchSampler(
            SequentialSampler(range(len(movie_dataset))), batch_size=512, drop_last=False
        )
        dataloader = DataLoader(movie_dataset, sampler=sampler, shuffle=False, collate_fn=lambda batch: batch[0])
        for batch in dataloader:
            batch_ids = self.forward(batch_to(batch, self.hrq_vae.device)).sem_ids
            
            # 如果使用去重维度，则计算重复项
            if self.use_dedup_dim:
                # 检测批次内重复
                is_hit = self._get_hits(batch_ids, batch_ids)
                hits = torch.tril(is_hit, diagonal=-1).sum(axis=-1)
                assert hits.min() >= 0
                
                if cached_ids is None:
                    cached_ids = batch_ids.clone()
                else:
                    # 检测批次-缓存重复
                    is_hit = self._get_hits(batch_ids, cached_ids)
                    hits += is_hit.sum(axis=-1)
                    cached_ids = pack([cached_ids, batch_ids], "* d")[0]
                dedup_dim.append(hits)
            else:
                # 不使用去重维度，直接合并ID
                if cached_ids is None:
                    cached_ids = batch_ids.clone()
                else:
                    cached_ids = pack([cached_ids, batch_ids], "* d")[0]
        
        # 如果使用去重维度，则添加去重列
        if self.use_dedup_dim:
            # 连接新列以去重 ID
            dedup_dim_tensor = pack(dedup_dim, "*")[0]
            self.cached_ids = pack([cached_ids, dedup_dim_tensor], "b *")[0]
        else:
            self.cached_ids = cached_ids
        
        return self.cached_ids
    
    @torch.no_grad()
    @eval_mode
    def exists_prefix(self, sem_id_prefix: Tensor) -> Tensor:
        if self.cached_ids is None:
            raise Exception("空缓存中找不到匹配项。")
    
        prefix_length = sem_id_prefix.shape[-1]
        prefix_cache = self.cached_ids[:, :prefix_length]
        out = torch.zeros(*sem_id_prefix.shape[:-1], dtype=bool, device=sem_id_prefix.device)
        
        # 批量前缀匹配以避免 OOM
        batches = math.ceil(sem_id_prefix.shape[0] // BATCH_SIZE)
        for i in range(batches):
            prefixes = sem_id_prefix[i*BATCH_SIZE:(i+1)*BATCH_SIZE,...]
            matches = (prefixes.unsqueeze(-2) == prefix_cache.unsqueeze(-3)).all(axis=-1).any(axis=-1)
            out[i*BATCH_SIZE:(i+1)*BATCH_SIZE,...] = matches
        
        return out
    
    def _tokenize_seq_batch_from_cached(self, ids: Tensor) -> Tensor:
        return rearrange(self.cached_ids[ids.flatten(), :], "(b n) d -> b (n d)", n=ids.shape[1])
    
    @torch.no_grad()
    @eval_mode
    def forward(self, batch: SeqBatch) -> TokenizedSeqBatch:
        # 如果缓存为空或批次 ID 超出缓存范围，则使用 HRQVAE 生成语义 ID
        if self.cached_ids is None or batch.ids.max() >= self.cached_ids.shape[0]:
            B, N = batch.ids.shape
            
            # 获取标签嵌入和索引（如果有）
            tags_emb = getattr(batch, 'tags_emb', None)
            tags_indices = getattr(batch, 'tags_indices', None)
            
            # 使用 HRQVAE 获取语义 ID
            sem_ids = self.hrq_vae.get_semantic_ids(batch.x, tags_emb, tags_indices).sem_ids
            D = sem_ids.shape[-1]
            seq_mask, sem_ids_fut = None, None
        else:
            # 从缓存中获取语义 ID
            B, N = batch.ids.shape
            _, D = self.cached_ids.shape
            sem_ids = self._tokenize_seq_batch_from_cached(batch.ids)
            seq_mask = batch.seq_mask.repeat_interleave(D, dim=1)
            sem_ids[~seq_mask] = -1
        
            sem_ids_fut = self._tokenize_seq_batch_from_cached(batch.ids_fut)
            
        token_type_ids = torch.arange(D, device=sem_ids.device).repeat(B, N)
        token_type_ids_fut = torch.arange(D, device=sem_ids.device).repeat(B, 1)
        return TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=sem_ids,
            sem_ids_fut=sem_ids_fut,
            seq_mask=seq_mask,
            token_type_ids=token_type_ids,
            token_type_ids_fut=token_type_ids_fut
        )
    
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
    import argparse
    
    parser = argparse.ArgumentParser(description='测试带标签预测功能的语义ID标记器')
    parser.add_argument('--dataset', type=str, default='beauty', choices=['ml-1m', 'beauty'],
                        help='要测试的数据集 (ml-1m 或 beauty)')
    parser.add_argument('--input_dim', type=int, default=768, 
                        help='输入维度 (ml-1m: 18, beauty: 768)')
    parser.add_argument('--embed_dim', type=int, default=32,
                        help='嵌入维度')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256, 128],
                        help='隐藏层维度')
    parser.add_argument('--codebook_size', type=int, default=256,
                        help='码本大小')
    parser.add_argument('--n_cat_feats', type=int, default=0,
                        help='分类特征数量 (ml-1m: 18, beauty: 0)')
    parser.add_argument('--tag_embed_dim', type=int, default=768,
                        help='标签嵌入维度')
    parser.add_argument('--tag_alignment_weight', type=float, default=0.5,
                        help='标签对齐权重')
    parser.add_argument('--tag_prediction_weight', type=float, default=0.5,
                        help='标签预测权重')
    parser.add_argument('--use_dedup_dim', action='store_true', default=False,
                        help='是否使用去重维度')
    parser.add_argument('--no_dedup_dim', dest='use_dedup_dim', action='store_false',
                        help='不使用去重维度')
    args = parser.parse_args()
    

    
    print(f"测试数据集: {args.dataset}")
    print(f"模型参数: input_dim={args.input_dim}, embed_dim={args.embed_dim}, "
          f"hidden_dims={args.hidden_dims}, codebook_size={args.codebook_size}, "
          f"n_cat_feats={args.n_cat_feats}, tag_embed_dim={args.tag_embed_dim}, "
          f"use_dedup_dim={args.use_dedup_dim}")
    
    if args.dataset == 'ml-1m':
        dataset_path = "dataset/ml-1m-movie"
        seq_dataset_path = "dataset/ml-1m"
        dataset_name = RecDataset.ML_1M
        tag_class_counts = [18, 7, 20]  # 示例值，请根据实际情况调整
    else:  # beauty
        dataset_path = "dataset/amazon"
        seq_dataset_path = "dataset/amazon"
        dataset_name = RecDataset.AMAZON
        tag_class_counts = [6, 130, 927]   # 示例值，请根据实际情况调整
    
    # 加载数据集
    print(f"加载物品数据集: {dataset_path}")
    dataset = ItemData(dataset_path, dataset=dataset_name, split="beauty" if args.dataset == 'beauty' else None)
    
    # 创建标记器
    print("初始化标记器...")
    tokenizer = HSemanticIdTokenizer(
        input_dim=args.input_dim,
        output_dim=args.embed_dim,
        hidden_dims=args.hidden_dims,
        codebook_size=args.codebook_size,
        n_cat_feats=args.n_cat_feats,
        n_layers=3,
        hrqvae_weights_path="out/hrqvae/amazon/hrqvae_checkpoint_best.pt",
        hrqvae_codebook_normalize=False,
        hrqvae_sim_vq=False,
        tag_alignment_weight=args.tag_alignment_weight,
        tag_prediction_weight=args.tag_prediction_weight,
        tag_class_counts=tag_class_counts,
        tag_embed_dim=args.tag_embed_dim,
        use_dedup_dim=args.use_dedup_dim  # 传入参数
    )
    
    # 预计算语义ID
    print("预计算语义ID...")
    tokenizer.precompute_corpus_ids(dataset)
    
    # 加载序列数据并测试标记器
    print(f"加载序列数据: {seq_dataset_path}")
    seq_data = SeqData(seq_dataset_path, dataset=dataset_name, split="beauty" if args.dataset == 'beauty' else None)
    
    # 获取一小批数据进行测试
    print("获取测试批次...")
    batch = seq_data[:10]
    # 打印测试批次
    print("\n测试批次:")
    print(f"用户ID: {batch.user_ids}")
    print(f"物品ID: {batch.ids}")
    print(f"下一项物品ID: {batch.ids_fut}")
    
    # 使用标记器处理批次
    print("使用标记器处理批次...")
    tokenized = tokenizer(batch)
    
    # 打印结果
    print("\n标记化结果:")
    print(f"用户ID: {tokenized.user_ids}")
    print(f"语义ID形状: {tokenized.sem_ids.shape}")
    print(f"语义ID: {tokenized.sem_ids}")
    print(f"语义ID形状(未来): {tokenized.sem_ids_fut.shape}")
    print(f"语义ID(未来): {tokenized.sem_ids_fut}")
    print(f"序列掩码: {tokenized.seq_mask}")
    print(f"标记类型ID: {tokenized.token_type_ids}")
    print(f"标记类型ID(未来): {tokenized.token_type_ids_fut}")
    
    # 测试标签预测功能
    print("\n测试标签预测功能...")
    tag_predictions = tokenizer.predict_tags(batch)
    print(f"标签预测结果形状: {len(tag_predictions)}")
    print("标签预测结果:", tag_predictions)
   
    
    # 测试联合标记化和标签预测功能
    print("\n测试联合标记化和标签预测功能...")
    tokenized_with_tags, tag_predictions = tokenizer.tokenize_with_tags(batch)
    print("联合处理结果:")
    print(f"标记化批次形状: {tokenized_with_tags.sem_ids.shape}")
    print(f"标签预测数量: {len(tag_predictions)}")
    
    print("\n测试完成!")
    
    # 交互式调试
    import pdb; pdb.set_trace()