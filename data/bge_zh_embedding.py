from FlagEmbedding import FlagModel
import torch
from torch.nn.functional import cosine_similarity

# 检查是否有可用的CUDA设备
use_cuda = torch.cuda.is_available()
print(f"CUDA可用: {'是' if use_cuda else '否'}")

# 加载模型（首次运行会自动下载）
# 如果CUDA可用，模型会自动使用GPU
model = FlagModel('BAAI/bge-base-zh-v1.5', 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True)  # 启用FP16加速

# 编码句子
sentences = ["二次元", "动漫"]
embeddings = model.encode(sentences)

print(f"\n'二次元'的向量 (前5维): {embeddings[0][:5]}")
print(f"'动漫'的向量 (前5维): {embeddings[1][:5]}")

# 将numpy数组转换为torch张量
embedding1 = torch.from_numpy(embeddings[0])
embedding2 = torch.from_numpy(embeddings[1])

# 计算余弦相似度
# 需要将一维向量扩展为二维 [1, D]
similarity = cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))

print(f"\n向量维度: {embeddings.shape[1]}")
print(f"“二次元”和“动漫”的余弦相似度: {similarity.item():.4f}")  