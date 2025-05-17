


import gin
import os
# 添加主工作路径方便导入包
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import wandb

import warnings
import logging
from datetime import datetime
import matplotlib.pyplot as plt

from accelerate import Accelerator
from data.processed import ItemData
from data.processed import RecDataset
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import parse_config
from torch.optim import AdamW
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from tqdm import tqdm

import torch._dynamo


# 抑制警告信息
warnings.filterwarnings('ignore')
logging.getLogger("torch._dynamo.convert_frame").setLevel(logging.ERROR)
logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)

# 设置 torch._dynamo 的警告
torch._dynamo.config.verbose = False
torch._dynamo.config.suppress_errors = True

@gin.configurable
def train(
    iterations=50000,
    batch_size=64,
    learning_rate=0.0001,
    weight_decay=0.01,
    dataset_folder="dataset/ml-1m",
    dataset=RecDataset.ML_1M,
    pretrained_rqvae_path=None,
    save_dir_root="out/",
    use_kmeans_init=False,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    do_eval=True,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    eval_every=50000,
    commitment_weight=0.25,
    vae_n_cat_feats=18,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_codebook_mode=QuantizeForwardMode.GUMBEL_SOFTMAX,
    vae_sim_vq=False,
    vae_n_layers=3,
    dataset_split="beauty"
):
    # 设置日志记录
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger("rqvae_view")
    
    # 打印训练参数
    logger.info("=== 训练参数 ===")
    params = locals()
    for key, value in params.items():
        if key != 'logger':
            logger.info(f"  {key}: {value}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载数据集
    logger.info(f"正在加载数据集: {dataset}, 分割: {dataset_split}")
    train_dataset = ItemData(
        root=dataset_folder, 
        dataset=dataset, 
        force_process=force_dataset_process, 
        train_test_split="train" if do_eval else "all", 
        split=dataset_split
    )
    logger.info(f"数据集大小: {len(train_dataset)}")
    
    # 创建数据加载器
    train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size, False)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None, collate_fn=lambda batch: batch)
    
    # 创建模型
    logger.info("=== 创建模型 ===")
    model = RqVae(
        input_dim=vae_input_dim,
        embed_dim=vae_embed_dim,
        hidden_dims=vae_hidden_dims,
        codebook_size=vae_codebook_size,
        codebook_kmeans_init=use_kmeans_init and pretrained_rqvae_path is None,
        codebook_normalize=vae_codebook_normalize,
        codebook_sim_vq=vae_sim_vq,
        codebook_mode=vae_codebook_mode,
        n_layers=vae_n_layers,
        n_cat_features=vae_n_cat_feats,
        commitment_weight=commitment_weight
    )
    model = model.to(device)
    
    # 打印模型结构
    logger.info("=== 模型结构 ===")
    logger.info(f"输入维度: {vae_input_dim}")
    logger.info(f"嵌入维度: {vae_embed_dim}")
    logger.info(f"隐藏层维度: {vae_hidden_dims}")
    logger.info(f"码本大小: {vae_codebook_size}")
    logger.info(f"层数: {vae_n_layers}")
    logger.info(f"类别特征数: {vae_n_cat_feats}")
    
    # 打印编码器结构
    logger.info("=== 编码器结构 ===")
    for name, param in model.encoder.named_parameters():
        logger.info(f"{name}: {param.shape}")
    
    # 打印量化层结构
    logger.info("=== 量化层结构 ===")
    for i, layer in enumerate(model.layers):
        logger.info(f"量化层 {i}:")
        for name, param in layer.named_parameters():
            logger.info(f"  {name}: {param.shape}")
    
    # 打印解码器结构
    logger.info("=== 解码器结构 ===")
    for name, param in model.decoder.named_parameters():
        logger.info(f"{name}: {param.shape}")
    
    # 创建优化器
    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 加载预训练模型（如果有）
    if pretrained_rqvae_path is not None:
        logger.info(f"加载预训练模型: {pretrained_rqvae_path}")
        model.load_pretrained(pretrained_rqvae_path)
    
    # 创建分词器
    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_rqvae_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq
    )
    tokenizer.rq_vae = model
    
    # 运行一轮前向传播
    logger.info("=== 运行前向传播 ===")
    model.eval()
    
    # 获取一个批次的数据
    train_iter = iter(train_dataloader)  # 创建迭代器
    batch = next(train_iter)  # 获取一个批次的数据
    data = batch_to(batch, device)  # 将数据移动到设备
    
    # 打印输入数据形状
    logger.info(f"输入数据形状: {data.x.shape}")
    # 打印输入数据的多个字段的形状
    logger.info("=== 输入数据字段形状 ===")
    for field_name in data._fields:
        field_value = getattr(data, field_name)
        if isinstance(field_value, torch.Tensor):
            logger.info(f"  {field_name}: {field_value.shape}")
        elif field_value is not None:
            logger.info(f"  {field_name}: {type(field_value)}")
    
    # 运行前向传播
    with torch.no_grad():
        t = 0.2  # Gumbel温度
        model_output = model(data, gumbel_t=t)
    
    # 打印模型输出
    logger.info("=== 模型输出 ===")
    logger.info(f"总损失: {model_output.loss.item():.4f}")
    logger.info(f"重构损失: {model_output.reconstruction_loss.item():.4f}")
    logger.info(f"RQVAE损失: {model_output.rqvae_loss.item():.4f}")
    # logger.info(f"嵌入范数 形状: {model_output.embs_norm}")
    logger.info(f"嵌入范数 形状: {model_output.embs_norm.shape}")
    logger.info(f"唯一ID比例: {model_output.p_unique_ids.item():.4f}")
    
    # 获取语义ID
    logger.info("=== 获取语义ID ===")
    quantized = model.get_semantic_ids(data.x, gumbel_t=t)
    logger.info(f"嵌入形状: {quantized.embeddings.shape}")
    logger.info(f"残差形状: {quantized.residuals.shape}")
    logger.info(f"语义ID形状: {quantized.sem_ids.shape}")
    
    # 修复：处理量化损失可能是张量的情况
    if quantized.quantize_loss.numel() > 1:
        # 如果是多元素张量，计算平均值
        logger.info(f"量化损失(平均值): {quantized.quantize_loss.mean().item():.4f}")
        logger.info(f"量化损失形状: {quantized.quantize_loss.shape}")
    else:
        # 如果是标量，直接使用item()
        logger.info(f"量化损失: {quantized.quantize_loss.item():.4f}")
    
    # 打印每层的语义ID分布
    logger.info("=== 语义ID分布 ===")
    for i in range(vae_n_layers):
        layer_ids = quantized.sem_ids[i]
        unique_ids, counts = torch.unique(layer_ids, return_counts=True)
        usage = len(unique_ids) / vae_codebook_size
        logger.info(f"层 {i} 码本使用率: {usage:.4f} ({len(unique_ids)}/{vae_codebook_size})")
        
        # 打印前10个最常用的ID
        sorted_indices = torch.argsort(counts, descending=True)
        top_ids = unique_ids[sorted_indices[:10]]
        top_counts = counts[sorted_indices[:10]]
        logger.info(f"层 {i} 前10个最常用ID: {top_ids.tolist()}")
        logger.info(f"层 {i} 前10个最常用ID计数: {top_counts.tolist()}")
    
    # 计算重构
    logger.info("=== 重构结果 ===")
    x_hat = model.decode(quantized.embeddings.sum(axis=-1))
    mse = torch.nn.functional.mse_loss(x_hat, data.x)
    logger.info(f"重构MSE: {mse.item():.4f}")
    
    logger.info("查看完成")

if __name__ == "__main__":
    parse_config()
    train()
