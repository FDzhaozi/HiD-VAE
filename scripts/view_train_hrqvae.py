import gin
import os
# 添加主工作路径方便导入包
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import warnings
import logging
from datetime import datetime
import matplotlib.pyplot as plt

from accelerate import Accelerator
from data.tags_processed import ItemData
from data.tags_processed import RecDataset
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
from modules.h_rqvae import HRqVae
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
    pretrained_hrqvae_path=None,
    save_dir_root="out/",
    use_kmeans_init=False,
    split_batches=True,
    amp=False,
    do_eval=True,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000,
    eval_every=5000,
    commitment_weight=0.25,
    tag_alignment_weight=0.5,
    tag_prediction_weight=0.5,
    vae_n_cat_feats=18,
    vae_input_dim=768,
    vae_embed_dim=128,
    vae_hidden_dims=[512, 256],
    vae_codebook_size=512,
    vae_codebook_normalize=False,
    vae_codebook_mode=QuantizeForwardMode.GUMBEL_SOFTMAX,
    vae_sim_vq=False,
    vae_n_layers=3,
    dataset_split="beauty",
    tag_class_counts=None,
    tag_embed_dim=768
):
    # 设置日志记录
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger("hrqvae_view")
    
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
    
    # 检查数据集是否包含标签信息
    has_tags = getattr(train_dataset, 'has_tags', False)
    if not has_tags:
        logger.warning("数据集中没有标签信息，将禁用标签对齐和预测功能")
        tag_alignment_weight = 0.0
        tag_prediction_weight = 0.0
    else:
        logger.info("数据集包含标签信息")
        
        # 确保只使用与vae_n_layers匹配的标签层数
        # 检查标签数据的形状
        sample_data = train_dataset[0]
        if hasattr(sample_data, 'tags_emb') and sample_data.tags_emb is not None:
            logger.info(f"sample_data.tags_emb.shape = {sample_data.tags_emb.shape}")
            actual_tag_layers = sample_data.tags_emb.shape[1]
            logger.info(f"数据集标签层数: {actual_tag_layers}")
            
            if actual_tag_layers != vae_n_layers:
                logger.warning(f"标签层数({actual_tag_layers})与模型层数({vae_n_layers})不匹配")
                
                # 直接对整个数据集进行操作
                if actual_tag_layers > vae_n_layers:
                    logger.warning(f"将裁剪数据集标签，只保留前{vae_n_layers}层")
                    # 裁剪标签嵌入
                    if hasattr(train_dataset, 'tags_emb') and train_dataset.tags_emb is not None:
                        train_dataset.tags_emb = train_dataset.tags_emb[:, :vae_n_layers, :]
                    
                    # 裁剪标签索引
                    if hasattr(train_dataset, 'tags_indices') and train_dataset.tags_indices is not None:
                        train_dataset.tags_indices = train_dataset.tags_indices[:, :vae_n_layers]
                    
                    logger.info(f"裁剪后 train_dataset.tags_emb.shape = {train_dataset.tags_emb.shape if hasattr(train_dataset, 'tags_emb') and train_dataset.tags_emb is not None else 'None'}")
                    logger.info(f"裁剪后 train_dataset.tags_indices.shape = {train_dataset.tags_indices.shape if hasattr(train_dataset, 'tags_indices') and train_dataset.tags_indices is not None else 'None'}")
                else:
                    logger.warning(f"模型层数({vae_n_layers})大于标签层数({actual_tag_layers})，将填充数据集标签")
                    # 填充标签嵌入
                    if hasattr(train_dataset, 'tags_emb') and train_dataset.tags_emb is not None:
                        tag_embed_shape = train_dataset.tags_emb.shape
                        padded_tags_emb = torch.zeros((tag_embed_shape[0], vae_n_layers, tag_embed_shape[2]), 
                                                     dtype=train_dataset.tags_emb.dtype)
                        padded_tags_emb[:, :actual_tag_layers, :] = train_dataset.tags_emb
                        train_dataset.tags_emb = padded_tags_emb
                    
                    # 填充标签索引
                    if hasattr(train_dataset, 'tags_indices') and train_dataset.tags_indices is not None:
                        tag_indices_shape = train_dataset.tags_indices.shape
                        padded_tags_indices = torch.ones((tag_indices_shape[0], vae_n_layers), 
                                                        dtype=train_dataset.tags_indices.dtype) * -1
                        padded_tags_indices[:, :actual_tag_layers] = train_dataset.tags_indices
                        train_dataset.tags_indices = padded_tags_indices
                    
                    logger.info(f"填充后 train_dataset.tags_emb.shape = {train_dataset.tags_emb.shape if hasattr(train_dataset, 'tags_emb') and train_dataset.tags_emb is not None else 'None'}")
                    logger.info(f"填充后 train_dataset.tags_indices.shape = {train_dataset.tags_indices.shape if hasattr(train_dataset, 'tags_indices') and train_dataset.tags_indices is not None else 'None'}")
    
    
    
    logger.info(f"最终使用的标签类别数量: {tag_class_counts}")
    
    # 确保标签类别数量与层数匹配
    assert len(tag_class_counts) == vae_n_layers, f"标签类别数量 {len(tag_class_counts)} 与层数 {vae_n_layers} 不匹配"
    
    # 创建数据加载器
    train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size, False)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None, collate_fn=lambda batch: batch)
    
    # 创建模型
    logger.info("=== 创建模型 ===")
    model = HRqVae(
        input_dim=vae_input_dim,
        embed_dim=vae_embed_dim,
        hidden_dims=vae_hidden_dims,
        codebook_size=vae_codebook_size,
        codebook_kmeans_init=use_kmeans_init and pretrained_hrqvae_path is None,
        codebook_normalize=vae_codebook_normalize,
        codebook_sim_vq=vae_sim_vq,
        codebook_mode=vae_codebook_mode,
        n_layers=vae_n_layers,
        n_cat_features=vae_n_cat_feats,
        commitment_weight=commitment_weight,
        tag_alignment_weight=tag_alignment_weight,
        tag_prediction_weight=tag_prediction_weight,
        tag_class_counts=tag_class_counts,
        tag_embed_dim=tag_embed_dim
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
    logger.info(f"标签类别数量: {tag_class_counts}")
    logger.info(f"标签嵌入维度: {tag_embed_dim}")
    logger.info(f"标签对齐权重: {tag_alignment_weight}")
    logger.info(f"标签预测权重: {tag_prediction_weight}")
    
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
    
    # 打印标签预测器结构
    logger.info("=== 标签预测器结构 ===")
    for i, predictor in enumerate(model.tag_predictors):
        logger.info(f"标签预测器 {i}:")
        for name, param in predictor.named_parameters():
            logger.info(f"  {name}: {param.shape}")
    
    # 打印标签投影器结构
    logger.info("=== 标签投影器结构 ===")
    for i, projector in enumerate(model.tag_projectors):
        logger.info(f"标签投影器 {i}:")
        for name, param in projector.named_parameters():
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
    if pretrained_hrqvae_path is not None:
        logger.info(f"加载预训练模型: {pretrained_hrqvae_path}")
        model.load_pretrained(pretrained_hrqvae_path)
    
    # 创建分词器
    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_hrqvae_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq
    )
    tokenizer.rq_vae = model
    
    # 运行一轮前向传播
    logger.info("=== 运行前向传播 ===")
    model.eval()
    
    # 获取一个批次的数据
    train_iter = iter(train_dataloader)
    batch = next(train_iter)
    data = batch_to(batch, device)
    
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
    
    # 检查是否有标签数据
    has_tag_data = hasattr(data, 'tags_emb') and data.tags_emb is not None
    has_tag_indices = hasattr(data, 'tags_indices') and data.tags_indices is not None
    
    if has_tag_data:
        logger.info(f"标签嵌入形状: {data.tags_emb.shape}")
    else:
        logger.info("数据中没有标签嵌入")
    
    if has_tag_indices:
        logger.info(f"标签索引形状: {data.tags_indices.shape}")
    else:
        logger.info("数据中没有标签索引")
    
    # 运行前向传播
    with torch.no_grad():
        t = 0.2  # Gumbel温度
        # 获取语义ID
        logger.info("=== 获取语义ID ===")
        quantized = model.get_semantic_ids(data.x, tags_emb=data.tags_emb, tags_indices=data.tags_indices, gumbel_t=t)
    
    
    logger.info(f"嵌入形状: {quantized.embeddings.shape}")
    logger.info(f"残差形状: {quantized.residuals.shape}")
    logger.info(f"语义ID形状: {quantized.sem_ids.shape}")
    
    # 处理量化损失
    if quantized.quantize_loss.numel() > 1:
        logger.info(f"量化损失(平均值): {quantized.quantize_loss.mean().item():.4f}")
        logger.info(f"量化损失形状: {quantized.quantize_loss.shape}")
    else:
        logger.info(f"量化损失: {quantized.quantize_loss.item():.4f}")
    
    # 处理标签对齐损失
    if quantized.tag_align_loss.numel() > 1:
        logger.info(f"标签对齐损失(平均值): {quantized.tag_align_loss.mean().item():.4f}")
        logger.info(f"标签对齐损失形状: {quantized.tag_align_loss.shape}")
    else:
        logger.info(f"标签对齐损失: {quantized.tag_align_loss.item():.4f}")
    
    # 处理标签预测损失
    if quantized.tag_pred_loss.numel() > 1:
        logger.info(f"标签预测损失(平均值): {quantized.tag_pred_loss.mean().item():.4f}")
        logger.info(f"标签预测损失形状: {quantized.tag_pred_loss.shape}")
    else:
        logger.info(f"标签预测损失: {quantized.tag_pred_loss.item():.4f}")
    
    # 新增：输出每层的标签对齐损失和标签预测损失
    logger.info("=== 每层标签损失详情 ===")
    if hasattr(quantized, 'tag_align_loss_by_layer') and quantized.tag_align_loss_by_layer is not None:
        logger.info("每层标签对齐损失:")
        for i, loss in enumerate(quantized.tag_align_loss_by_layer):
            logger.info(f"  层 {i}: {loss.item():.4f}")
    else:
        logger.info("没有每层标签对齐损失信息")
        
    if hasattr(quantized, 'tag_pred_loss_by_layer') and quantized.tag_pred_loss_by_layer is not None:
        logger.info("每层标签预测损失:")
        for i, loss in enumerate(quantized.tag_pred_loss_by_layer):
            logger.info(f"  层 {i}: {loss.item():.4f}")
    else:
        logger.info("没有每层标签预测损失信息")
        
    if hasattr(quantized, 'tag_pred_accuracy_by_layer') and quantized.tag_pred_accuracy_by_layer is not None:
        logger.info("每层标签预测准确率:")
        for i, acc in enumerate(quantized.tag_pred_accuracy_by_layer):
            logger.info(f"  层 {i}: {acc.item():.4f}")
    else:
        logger.info("没有每层标签预测准确率信息")
    
    # 处理标签预测准确率
    if quantized.tag_pred_accuracy.numel() > 1:
        logger.info(f"标签预测准确率(平均值): {quantized.tag_pred_accuracy.mean().item():.4f}")
        logger.info(f"标签预测准确率形状: {quantized.tag_pred_accuracy.shape}")
    else:
        logger.info(f"标签预测准确率: {quantized.tag_pred_accuracy.item():.4f}")
    
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
    
    # 打印标签预测结果
    logger.info("=== 标签预测结果 ===")
    for i in range(vae_n_layers):
        if has_tag_indices:
            # 获取当前层的标签索引
            logger.info(f"data.tags_indices shape = {data.tags_indices.shape}")
            logger.info(f"data.tags_indices[:, i] shape = {data.tags_indices[:, i].shape}")
            layer_tag_indices = data.tags_indices[:, i]  # 获取当前层的标签索引
            valid_mask = (layer_tag_indices >= 0)
            valid_count = valid_mask.sum().item()
            
            if valid_count > 0:
                logger.info(f"层 {i} 有效标签数量: {valid_count}/{data.x.shape[0]}")
                
                # 获取当前层的残差
                # logger.info(f"quantized   = {quantized}")
                logger.info(f"quantized.residuals shape = {quantized.residuals.shape}")
                layer_residual = quantized.residuals[:,:, i]  # 获取当前层的残差
                logger.info(f"layer_residual shape = {layer_residual.shape}")
                
                # 使用标签预测器进行预测
                with torch.no_grad():
                    # 在评估标签预测部分，需要修改为使用拼接embedding
                    # 大约在第390-410行左右
                    
                    # 修改前:
                    # layer_residual = quantized.residuals[:, :, i]
                    # pred_logits = model.tag_predictors[i](layer_residual)
                    
                    # 修改为:
                    layer_embs = []
                    for j in range(i+1):  # 收集前i+1层的embedding
                        layer_embs.append(quantized.embeddings[:, :, j])
                        
                    # 拼接前i+1层的embedding
                    concat_emb = torch.cat(layer_embs, dim=1)  # [batch_size, (i+1)*embed_dim]
                        
                    # 使用拼接后的embedding进行预测
                    pred_logits = model.tag_predictors[i](concat_emb)
                    pred_indices = torch.argmax(pred_logits, dim=-1)
                
                # 计算准确率
                valid_pred = pred_indices[valid_mask]
                valid_targets = layer_tag_indices[valid_mask]
                accuracy = (valid_pred == valid_targets).float().mean().item()
                
                logger.info(f"层 {i} 标签预测准确率: {accuracy:.4f}")
                
                # 打印前5个样本的预测结果
                num_samples = min(5, valid_count)
                valid_indices = torch.where(valid_mask)[0][:num_samples]
                
                for j, idx in enumerate(valid_indices):
                    logger.info(f"  样本 {j}: 预测={pred_indices[idx].item()}, 真实={layer_tag_indices[idx].item()}")
            else:
                logger.info(f"层 {i} 没有有效标签")
        else:
            logger.info(f"层 {i} 没有标签索引数据")
    
    logger.info("查看完成")

if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='查看HRQVAE模型训练过程')
    parser.add_argument('config', type=str, help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    gin.parse_config_file(args.config)
    
    # 运行训练函数
    train()