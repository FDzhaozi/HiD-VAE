import torch

from data.schemas import TokenizedSeqBatch
from torch import nn
from torch import Tensor
from typing import NamedTuple


class SemIdEmbeddingBatch(NamedTuple):
    seq: Tensor
    fut: Tensor


class SemIdEmbedder(nn.Module):
    def __init__(self, num_embeddings, sem_ids_dim, embeddings_dim, n_sem_layers=3, use_interleaved_ids: bool = False) -> None:
        super().__init__()
        
        self.sem_ids_dim = sem_ids_dim
        self.num_embeddings = num_embeddings
        self.padding_idx = sem_ids_dim*num_embeddings # This might need adjustment for interleaved
        self.n_sem_layers = n_sem_layers  # 语义ID的层数，用于区分语义ID和标签ID
        self.use_interleaved_ids = use_interleaved_ids # 保存参数
        
        # 增加嵌入表的大小，为可能的额外标签ID预留空间
        # 这将允许我们处理语义ID和标签ID
        max_tag_size = 1000  # 每层标签的最大类别数量（根据实际情况调整）
        self.max_tag_size = max_tag_size
        self.n_tag_layers = sem_ids_dim - n_sem_layers # 标签ID的层数
        
        # 计算总嵌入大小: 语义ID范围 + 标签ID范围 + 填充ID
        # 对于语义ID：num_embeddings * n_sem_layers
        # 对于标签ID：max_tag_size * (sem_ids_dim - n_sem_layers)
        # 额外 +1 用于填充
        # The total_embed_size calculation needs to be robust for all modes.
        # For concatenated and interleaved modes, both semantic and tag IDs are present.
        # For semantic-only mode, only semantic IDs.
        semantic_part_size_total = num_embeddings * n_sem_layers
        tag_part_size_total = max_tag_size * self.n_tag_layers if self.n_tag_layers > 0 else 0
        total_embed_size = semantic_part_size_total + tag_part_size_total + 1

        # Adjust padding_idx based on the largest possible index before padding
        self.padding_idx = total_embed_size -1
        
        self.emb = nn.Embedding(
            num_embeddings=total_embed_size, # Use the calculated total_embed_size
            embedding_dim=embeddings_dim,
            padding_idx=self.padding_idx
        )
    
    def forward(self, batch: TokenizedSeqBatch) -> Tensor:
        # 获取序列中的语义ID和标记类型ID
        sem_ids = batch.sem_ids
        token_type_ids = batch.token_type_ids
        
        # 创建嵌入索引，区分语义ID和标签ID
        emb_indices = torch.zeros_like(sem_ids)
        
        # 确定语义ID和标签ID的分界点
        # 在拼接模式下，前n_sem_layers维度是语义ID，剩余的是标签ID
        # 我们假设sem_ids_dim参数已经包含了全部维度（语义ID+标签ID）

        # 语义ID部分的总大小（用于偏移标签ID索引）
        semantic_part_offset = self.num_embeddings * self.n_sem_layers

        if self.use_interleaved_ids:
            # 交错模式 [s1, t1, s2, t2, ...]
            # token_type_ids: 0, 1, 2, 3, ...
            # 偶数索引是语义ID，奇数索引是标签ID
            for i in range(self.sem_ids_dim): # i is the position in the interleaved sequence
                dim_mask = token_type_ids == i
                dim_ids = sem_ids[dim_mask]

                if i % 2 == 0: # Semantic ID
                    sem_layer_idx = i // 2
                    if sem_layer_idx < self.n_sem_layers:
                        emb_indices[dim_mask] = sem_layer_idx * self.num_embeddings + dim_ids
                    else:
                        # This case should ideally not happen if sem_ids_dim is correctly n_sem_layers + n_tag_layers
                        # and n_sem_layers and n_tag_layers are roughly equal or sem_layer_idx is bounded
                        # print(f"Warning: Semantic layer index {sem_layer_idx} out of bounds {self.n_sem_layers}")
                        emb_indices[dim_mask] = self.padding_idx # Fallback to padding
                else: # Tag ID
                    tag_layer_idx = i // 2
                    if tag_layer_idx < self.n_tag_layers:
                        emb_indices[dim_mask] = semantic_part_offset + tag_layer_idx * self.max_tag_size + dim_ids
                    else:
                        # print(f"Warning: Tag layer index {tag_layer_idx} out of bounds {self.n_tag_layers}")
                        emb_indices[dim_mask] = self.padding_idx # Fallback to padding
        else:
            # 非交错模式 (拼接或仅语义ID)
            # 遍历每个维度的ID
            for i in range(self.sem_ids_dim):
                # 获取当前维度的ID和类型
                dim_mask = token_type_ids == i
                dim_ids = sem_ids[dim_mask]
                
                # 确保ID在有效范围内，防止索引越界
                if dim_ids.numel() > 0:  # 只有在有元素时才处理
                    # 对于语义ID，限制在[0, self.num_embeddings-1]范围内
                    # 对于标签ID，限制在[0, self.max_tag_size-1]范围内
                    if i % 2 == 0 if self.use_interleaved_ids else i < self.n_sem_layers:  # 语义ID
                        dim_ids = torch.clamp(dim_ids, min=0, max=self.num_embeddings-1)
                    else:  # 标签ID
                        dim_ids = torch.clamp(dim_ids, min=0, max=self.max_tag_size-1)
                
                # 生成嵌入索引
                if i < self.n_sem_layers:  # 前n_sem_layers维度是语义ID
                    # 语义ID的索引计算: i * num_embeddings + id
                    emb_indices[dim_mask] = i * self.num_embeddings + dim_ids
                else: # 标签ID (仅在拼接模式下，且 i >= n_sem_layers)
                    # 标签ID的索引计算: (语义ID部分大小) + (i - n_sem_layers) * max_tag_size + id
                    tag_dim_index = i - self.n_sem_layers
                    if tag_dim_index < self.n_tag_layers: # Check if current index is within tag layers
                         emb_indices[dim_mask] = semantic_part_offset + tag_dim_index * self.max_tag_size + dim_ids
                    else:
                        # This can happen if sem_ids_dim for some reason is > n_sem_layers + n_tag_layers (e.g. dedup dim)
                        # Or if it's semantic-only mode and i >= n_sem_layers (but then self.n_tag_layers would be 0)
                        # print(f"Warning: Tag dimension index {tag_dim_index} out of bounds {self.n_tag_layers}")
                        emb_indices[dim_mask] = self.padding_idx # Fallback to padding
        
        # 应用填充掩码
        if hasattr(batch, 'seq_mask') and batch.seq_mask is not None:
            emb_indices[~batch.seq_mask] = self.padding_idx

        # 嵌入当前序列
        seq_embs = self.emb(emb_indices)

        # 处理未来序列（如果有）
        if batch.sem_ids_fut is not None:
            fut_ids = batch.sem_ids_fut
            fut_type_ids = batch.token_type_ids_fut
            
            # 创建未来序列的嵌入索引
            fut_emb_indices = torch.zeros_like(fut_ids)
            
            # 遍历每个维度的ID
            for i in range(min(self.sem_ids_dim, fut_type_ids.shape[-1])):
                # 获取当前维度的ID和类型
                dim_mask = fut_type_ids == i
                dim_ids = fut_ids[dim_mask]
                
                # 确保ID在有效范围内，防止索引越界
                if dim_ids.numel() > 0:  # 只有在有元素时才处理
                    # 对于语义ID，限制在[0, self.num_embeddings-1]范围内
                    # 对于标签ID，限制在[0, self.max_tag_size-1]范围内
                    if i % 2 == 0 if self.use_interleaved_ids else i < self.n_sem_layers:  # 语义ID
                        dim_ids = torch.clamp(dim_ids, min=0, max=self.num_embeddings-1)
                    else:  # 标签ID
                        dim_ids = torch.clamp(dim_ids, min=0, max=self.max_tag_size-1)

                if self.use_interleaved_ids:
                    # 交错模式 [s1, t1, s2, t2, ...]
                    if i % 2 == 0: # Semantic ID
                        sem_layer_idx = i // 2
                        if sem_layer_idx < self.n_sem_layers:
                            fut_emb_indices[dim_mask] = sem_layer_idx * self.num_embeddings + dim_ids
                        else:
                            fut_emb_indices[dim_mask] = self.padding_idx
                    else: # Tag ID
                        tag_layer_idx = i // 2
                        if tag_layer_idx < self.n_tag_layers:
                             fut_emb_indices[dim_mask] = semantic_part_offset + tag_layer_idx * self.max_tag_size + dim_ids
                        else:
                            fut_emb_indices[dim_mask] = self.padding_idx
                else:
                    # 非交错模式 (拼接或仅语义ID)
                    # 生成嵌入索引
                    if i < self.n_sem_layers:  # 前n_sem_layers维度是语义ID
                        # 语义ID的索引计算
                        fut_emb_indices[dim_mask] = i * self.num_embeddings + dim_ids
                    else:
                        # 标签ID的索引计算
                        tag_dim_index = i - self.n_sem_layers
                        if tag_dim_index < self.n_tag_layers:
                             fut_emb_indices[dim_mask] = semantic_part_offset + tag_dim_index * self.max_tag_size + dim_ids
                        else:
                            fut_emb_indices[dim_mask] = self.padding_idx
            
            # 嵌入未来序列
            fut_embs = self.emb(fut_emb_indices)
        else:
            fut_embs = None
        
        return SemIdEmbeddingBatch(seq=seq_embs, fut=fut_embs)
    

class UserIdEmbedder(nn.Module):
    # TODO: Implement hashing trick embedding for user id
    def __init__(self, num_buckets, embedding_dim) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.emb = nn.Embedding(num_buckets, embedding_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        hashed_indices = x % self.num_buckets
        # hashed_indices = torch.tensor([hash(token) % self.num_buckets for token in x], device=x.device)
        return self.emb(hashed_indices)
