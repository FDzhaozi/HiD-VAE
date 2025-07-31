from collections import defaultdict
from einops import rearrange
from torch import Tensor
import torch
import numpy as np


class TopKAccumulator:
    def __init__(self, ks=[1, 5, 10]):
        self.ks = ks
        self.reset()

    def reset(self):
        self.total = 0
        self.metrics = defaultdict(int)

    def accumulate(self, actual: Tensor, top_k: Tensor) -> None:
        B, D = actual.shape
        pos_match = (rearrange(actual, "b d -> b 1 d") == top_k)
        for i in range(D):
            match_found, rank = pos_match[...,:i+1].all(axis=-1).max(axis=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_slice_:{i+1}"] += len(matched_rank[matched_rank < k])
            
            match_found, rank = pos_match[...,i:i+1].all(axis=-1).max(axis=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_pos_{i}"] += len(matched_rank[matched_rank < k])
        self.total += B
        
    def reduce(self) -> dict:
        return {k: v/self.total for k, v in self.metrics.items()}


class NDCGAccumulator:
    """
    计算NDCG (Normalized Discounted Cumulative Gain) 
    
    NDCG衡量排序质量，考虑了项目的相关性和排名位置
    """
    def __init__(self, ks=[1, 5, 10]):
        self.ks = ks
        self.reset()
        
    def reset(self):
        self.total = 0
        self.metrics = defaultdict(float)
    
    def _dcg_at_k(self, relevances, k):
        """计算DCG@k"""
        relevances = relevances[:k]
        gains = 2 ** relevances - 1
        discounts = np.log2(np.arange(2, len(relevances) + 2))
        return np.sum(gains / discounts)
    
    def _ndcg_at_k(self, relevances, k):
        """计算NDCG@k"""
        dcg = self._dcg_at_k(relevances, k)
        # 理想情况下的排序（相关性降序排列）
        ideal_relevances = np.sort(relevances)[::-1]
        idcg = self._dcg_at_k(ideal_relevances, k)
        return dcg / idcg if idcg > 0 else 0.0
    
    def accumulate(self, actual: Tensor, top_k: Tensor) -> None:
        """
        计算NDCG指标
        
        参数:
            actual: 形状为 [B, D] 的张量，表示实际的semantic ID
            top_k: 形状为 [B, K, D] 的张量，表示预测的top-k结果
        """
        B, D = actual.shape
        pos_match = (rearrange(actual, "b d -> b 1 d") == top_k)
        
        for i in range(D):
            # 计算完整semantic ID的NDCG（考虑前i+1位）
            slice_match = pos_match[...,:i+1].all(axis=-1).float()  # [B, K]
            
            for b in range(B):
                relevances = slice_match[b].cpu().numpy()
                for k in self.ks:
                    if k <= len(relevances):
                        ndcg = self._ndcg_at_k(relevances, k)
                        self.metrics[f"ndcg@{k}_slice_:{i+1}"] += ndcg
            
            # 计算单个位置的NDCG
            pos_match_i = pos_match[...,i:i+1].all(axis=-1).float()  # [B, K]
            
            for b in range(B):
                relevances = pos_match_i[b].cpu().numpy()
                for k in self.ks:
                    if k <= len(relevances):
                        ndcg = self._ndcg_at_k(relevances, k)
                        self.metrics[f"ndcg@{k}_pos_{i}"] += ndcg
        
        self.total += B
    
    def reduce(self) -> dict:
        """返回平均NDCG指标"""
        return {k: v/self.total for k, v in self.metrics.items()}
