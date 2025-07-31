import gin
import os
import torch
import numpy as np
import warnings
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from modules.utils import parse_config
from accelerate import Accelerator
from data.tags_processed import ItemData
from data.tags_processed import RecDataset
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
from modules.h_rqvae import HRqVae
from modules.quantize import QuantizeForwardMode
from modules.tokenizer.h_semids import HSemanticIdTokenizer
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from tqdm import tqdm

import torch._dynamo
import torch.nn.functional as F

# 抑制警告信息
warnings.filterwarnings('ignore')
logging.getLogger("torch._dynamo.convert_frame").setLevel(logging.ERROR)
logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)

# 设置 torch._dynamo 的警告
torch._dynamo.config.verbose = False
torch._dynamo.config.suppress_errors = True

# 新增：计算ID重复率的函数
def calculate_repetition_rate(item_ids: torch.Tensor):
    """
    计算ID重复率
    
    参数:
        item_ids: 形状为 [num_items, id_dim] 的ID张量
        
    返回:
        repetition_rate: 重复率
        num_unique_items: 唯一ID数量
        total_items: 总ID数量
    """
    if item_ids is None or item_ids.nelement() == 0:
        return 0.0, 0, 0
    
    # 使用 PyTorch 的 unique 函数计算唯一行及其计数
    unique_ids, inverse_indices, counts = torch.unique(item_ids, dim=0, return_inverse=True, return_counts=True)
    num_unique_items = unique_ids.shape[0]
    total_items = item_ids.shape[0]
    
    if total_items == 0:
        return 0.0, 0, 0
    
    repetition_rate = 1.0 - (num_unique_items / total_items)
    return repetition_rate, num_unique_items, total_items

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
    use_kmeans_init=True,
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
    tag_embed_dim=768,
    use_focal_loss=True,  # 默认启用焦点损失
    focal_loss_gamma_base=2.0,  # 基础gamma参数
    focal_loss_alpha_base=0.25,  # 基础alpha参数
    rare_tag_threshold=30,  # 稀有标签阈值，出现次数小于此值的标签被视为稀有标签
    # 新增超参数
    dropout_rate=0.3,  # 预测器中的dropout率
    use_batch_norm=True,  # 是否使用BatchNorm
    alignment_temperature=0.1,  # 对比学习温度参数
    predictor_weight_decay=0.02,  # 标签预测器的权重衰减
    layer_specific_lr=False,  # 是否为不同层使用不同的学习率
    # 新增: 防止过拟合的高级策略
    use_label_smoothing=True,  # 是否使用标签平滑
    label_smoothing_alpha=0.1,  # 标签平滑系数
    use_mixup=True,  # 是否使用混合样本策略
    mixup_alpha=0.2,  # 混合系数
    # 新增: 评估策略
    eval_tta=True,  # 测试时增强
    eval_temperature=0.8,  # 预测时的温度参数
    ensemble_predictions=True,  # 是否使用集成预测
    # 新增: 学习率调度器参数
    use_lr_scheduler=True,
    lr_scheduler_type='cosine',  # 'cosine', 'reduce_on_plateau', 'step'
    lr_scheduler_T_max=400000, # For CosineAnnealingLR: Number of iterations for one cycle
    lr_scheduler_eta_min=1e-7, # For CosineAnnealingLR: Minimum learning rate
    lr_scheduler_step_size=100000, # For StepLR: Period of learning rate decay
    lr_scheduler_gamma=0.5, # For StepLR: Multiplicative factor of learning rate decay
    lr_scheduler_factor=0.5, # For ReduceLROnPlateau: Factor by which the learning rate will be reduced
    lr_scheduler_patience=10, # For ReduceLROnPlateau: Number of epochs with no improvement after which learning rate will be reduced
    # 新增: 语义ID唯一性约束参数
    sem_id_uniqueness_weight=0.5,  # 语义ID唯一性约束权重
    sem_id_uniqueness_margin=0.5,  # 语义ID唯一性约束边界值
    # 新增: ID重复率阈值
    id_repetition_threshold=0.03,  # ID重复率阈值，低于此值才保存模型
    # 新增: Tokenizer模式参数
    use_concatenated_ids: bool = True,
    use_interleaved_ids: bool = False,
):
    # 设置日志记录
    # 创建图表保存目录
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_dir_root, f"hrqvae_{dataset.name}_{time_stamp}")
    #log_dir = os.path.join(save_dir_root, "log")
    log_dir = os.path.join(save_dir, "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    plots_dir = os.path.join(log_dir, "plots") 
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"hrqvae_training_{timestamp}.log")
    
    # 配置日志记录器
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[file_handler, console_handler]
    )
    
    logger = logging.getLogger("hrqvae_training")
    
    # 初始化用于绘图的数据收集器
    plot_data = {
        'iterations': [],
        'total_loss': [],
        'reconstruction_loss': [],
        'rqvae_loss': [],
        'tag_align_loss': [],
        'tag_pred_loss': [],
        'tag_pred_accuracy': [],
        'emb_norms': [[] for _ in range(vae_n_layers)],
        'codebook_usage': [[] for _ in range(vae_n_layers)],
        'eval_iterations': [],
        'eval_total_loss': [],
        'eval_reconstruction_loss': [],
        'eval_rqvae_loss': [],
        'eval_tag_align_loss': [],
        'eval_tag_pred_loss': [],
        'eval_tag_pred_accuracy': [],
        'rqvae_entropy': [],
        'max_id_duplicates': []
    }
    
    # 首先创建 accelerator 实例
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )
    
    best_eval_accuracy = 0.0
    
    # 记录训练参数
    if accelerator.is_main_process:
        params = locals()
        logger.info("训练参数:")
        for key, value in params.items():
            if key != 'logger':
                logger.info(f"  {key}: {value}")

    device = accelerator.device

    # 加载带有标签数据的数据集
    if dataset == RecDataset.KUAIRAND:
        # KuaiRand数据集不接受split参数
        train_dataset = ItemData(root=dataset_folder, dataset=dataset, force_process=force_dataset_process, train_test_split="train" if do_eval else "all")
    else:
        # 其他数据集接受split参数
        train_dataset = ItemData(root=dataset_folder, dataset=dataset, force_process=force_dataset_process, train_test_split="train" if do_eval else "all", split=dataset_split)
    
    train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size, False)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None, collate_fn=lambda batch: batch)
    train_dataloader = cycle(train_dataloader)

    if do_eval:
        if dataset == RecDataset.KUAIRAND:
            # KuaiRand数据集不接受split参数
            eval_dataset = ItemData(root=dataset_folder, dataset=dataset, force_process=False, train_test_split="eval")
        else:
            # 其他数据集接受split参数
            eval_dataset = ItemData(root=dataset_folder, dataset=dataset, force_process=False, train_test_split="eval", split=dataset_split)
        
        eval_sampler = BatchSampler(RandomSampler(eval_dataset), batch_size, False)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=None, collate_fn=lambda batch: batch)

    if dataset == RecDataset.KUAIRAND:
        # KuaiRand数据集不接受split参数
        index_dataset = ItemData(root=dataset_folder, dataset=dataset, force_process=False, train_test_split="all") if do_eval else train_dataset
    else:
        # 其他数据集接受split参数
        index_dataset = ItemData(root=dataset_folder, dataset=dataset, force_process=False, train_test_split="all", split=dataset_split) if do_eval else train_dataset
    
    train_dataloader = accelerator.prepare(train_dataloader)

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
                    
        # 如果评估数据集存在，也需要对其进行相同的处理
        if do_eval:
            sample_eval_data = eval_dataset[0]
            if hasattr(sample_eval_data, 'tags_emb') and sample_eval_data.tags_emb is not None:
                logger.info(f"sample_eval_data.tags_emb.shape = {sample_eval_data.tags_emb.shape}")
                actual_eval_tag_layers = sample_eval_data.tags_emb.shape[1]
                logger.info(f"评估数据集标签层数: {actual_eval_tag_layers}")
                
                if actual_eval_tag_layers != vae_n_layers:
                    logger.warning(f"评估数据集标签层数({actual_eval_tag_layers})与模型层数({vae_n_layers})不匹配")
                    
                    if actual_eval_tag_layers > vae_n_layers:
                        logger.warning(f"将裁剪评估数据集标签，只保留前{vae_n_layers}层")
                        # 裁剪标签嵌入
                        if hasattr(eval_dataset, 'tags_emb') and eval_dataset.tags_emb is not None:
                            eval_dataset.tags_emb = eval_dataset.tags_emb[:, :vae_n_layers, :]
                        
                        # 裁剪标签索引
                        if hasattr(eval_dataset, 'tags_indices') and eval_dataset.tags_indices is not None:
                            eval_dataset.tags_indices = eval_dataset.tags_indices[:, :vae_n_layers]
                        
                        logger.info(f"裁剪后 eval_dataset.tags_emb.shape = {eval_dataset.tags_emb.shape if hasattr(eval_dataset, 'tags_emb') and eval_dataset.tags_emb is not None else 'None'}")
                        logger.info(f"裁剪后 eval_dataset.tags_indices.shape = {eval_dataset.tags_indices.shape if hasattr(eval_dataset, 'tags_indices') and eval_dataset.tags_indices is not None else 'None'}")
                    else:
                        logger.warning(f"模型层数({vae_n_layers})大于评估数据集标签层数({actual_eval_tag_layers})，将填充评估数据集标签")
                        # 填充标签嵌入
                        if hasattr(eval_dataset, 'tags_emb') and eval_dataset.tags_emb is not None:
                            tag_embed_shape = eval_dataset.tags_emb.shape
                            padded_tags_emb = torch.zeros((tag_embed_shape[0], vae_n_layers, tag_embed_shape[2]), 
                                                         dtype=eval_dataset.tags_emb.dtype)
                            padded_tags_emb[:, :actual_eval_tag_layers, :] = eval_dataset.tags_emb
                            eval_dataset.tags_emb = padded_tags_emb
                        
                        # 填充标签索引
                        if hasattr(eval_dataset, 'tags_indices') and eval_dataset.tags_indices is not None:
                            tag_indices_shape = eval_dataset.tags_indices.shape
                            padded_tags_indices = torch.ones((tag_indices_shape[0], vae_n_layers), 
                                                            dtype=eval_dataset.tags_indices.dtype) * -1
                            padded_tags_indices[:, :actual_eval_tag_layers] = eval_dataset.tags_indices
                            eval_dataset.tags_indices = padded_tags_indices
                        
                        logger.info(f"填充后 eval_dataset.tags_emb.shape = {eval_dataset.tags_emb.shape if hasattr(eval_dataset, 'tags_emb') and eval_dataset.tags_emb is not None else 'None'}")
                        logger.info(f"填充后 eval_dataset.tags_indices.shape = {eval_dataset.tags_indices.shape if hasattr(eval_dataset, 'tags_indices') and eval_dataset.tags_indices is not None else 'None'}")

    # 如果未提供标签类别数量，使用默认值
    if tag_class_counts is None:
        tag_class_counts = [10, 100, 1000][:vae_n_layers]
    
    # 确保标签类别数量与层数匹配
    assert len(tag_class_counts) == vae_n_layers, f"标签类别数量 {len(tag_class_counts)} 与层数 {vae_n_layers} 不匹配"
    
    # 创建焦点损失参数字典，为每层设置不同的参数
    focal_loss_params = {
        'gamma': focal_loss_gamma_base,
        'alpha': focal_loss_alpha_base,
    }
    
    # 为每层设置不同的gamma参数，层级越高gamma越大，更关注难分类样本
    for i in range(vae_n_layers):
        # 层级越高，gamma越大，更关注难分类样本
        focal_loss_params[f'gamma_{i}'] = focal_loss_gamma_base * (1 + i * 0.5)
        # 层级越高，alpha越小，更关注少数类
        focal_loss_params[f'alpha_{i}'] = max(0.05, focal_loss_alpha_base - i * 0.05)
    
    if accelerator.is_main_process:
        logger.info("焦点损失参数:")
        for key, value in focal_loss_params.items():
            logger.info(f"  {key}: {value}")
    
    # 计算各层标签类别的频率统计，并处理稀有标签
    if has_tags and use_focal_loss:
        logger.info("正在计算标签类别频率统计...")
        class_counts_dict = {}
        rare_tags_dict = {}  # 存储每层的稀有标签ID
        original_tag_class_counts = tag_class_counts.copy()  # 保存原始的标签类别数量
        new_tag_class_counts = []  # 新的标签类别数量
        
        # 遍历训练集，统计每层每个类别的出现次数
        for i in range(vae_n_layers):
            if hasattr(train_dataset, 'tags_indices') and train_dataset.tags_indices is not None:
                # 获取当前层的所有标签索引
                layer_indices = train_dataset.tags_indices[:, i]
                # 只统计有效的标签（不是-1）
                valid_indices = layer_indices[layer_indices >= 0]
                
                if len(valid_indices) > 0:
                    # 统计每个类别的出现次数
                    unique_classes, counts = torch.unique(valid_indices, return_counts=True)
                    
                    # 创建完整的类别计数张量，未出现的类别计数为0
                    full_counts = torch.zeros(original_tag_class_counts[i], dtype=torch.long)
                    full_counts[unique_classes.long()] = counts
                    
                    # 找出稀有标签（出现次数小于阈值）
                    rare_mask = (full_counts > 0) & (full_counts < rare_tag_threshold)
                    rare_tag_ids = torch.nonzero(rare_mask).squeeze(-1)
                    
                    # 记录稀有标签ID
                    rare_tags_dict[i] = rare_tag_ids
                    
                    # 计算非稀有标签的数量
                    non_rare_count = ((full_counts >= rare_tag_threshold) | (full_counts == 0)).sum().item()
                    
                    # 新的标签类别数量 = 非稀有标签数量 + 1（特殊类别）
                    new_tag_class_counts.append(non_rare_count + 1)
                    
                    # 计算类别分布统计信息
                    total_samples = full_counts.sum().item()
                    non_zero_classes = (full_counts > 0).sum().item()
                    rare_classes = rare_tag_ids.numel()
                    max_count = full_counts.max().item()
                    min_count = full_counts[full_counts > 0].min().item() if (full_counts > 0).any() else 0
                    mean_count = full_counts[full_counts > 0].float().mean().item() if (full_counts > 0).any() else 0
                    
                    logger.info(f"层 {i} 标签统计: 总样本数={total_samples}, 非零类别数={non_zero_classes}/{original_tag_class_counts[i]}, "
                               f"稀有类别数={rare_classes}, 最大计数={max_count}, 最小计数={min_count}, 平均计数={mean_count:.2f}")
                    
                    # 存储到字典中（这里只存储非稀有标签的计数）
                    class_counts_dict[i] = full_counts
                else:
                    # 如果没有有效标签，保持原始类别数量
                    new_tag_class_counts.append(original_tag_class_counts[i])
            else:
                # 如果没有标签索引，保持原始类别数量
                new_tag_class_counts.append(original_tag_class_counts[i])
        
        # 导出稀有标签ID到PT文件
        rare_tags_path = os.path.join(save_dir_root+"special_tags_files", "rare_tags.pt")
        # 确保目录存在
        os.makedirs(os.path.dirname(rare_tags_path), exist_ok=True)
        torch.save(rare_tags_dict, rare_tags_path)
        logger.info(f"稀有标签ID已保存到: {rare_tags_path}")
        
        # 更新标签类别数量
        tag_class_counts = new_tag_class_counts
        logger.info(f"更新后的标签类别数量: {tag_class_counts}")
        
        # 重新映射训练集中的标签索引
        if hasattr(train_dataset, 'tags_indices') and train_dataset.tags_indices is not None:
            for i in range(vae_n_layers):
                if i in rare_tags_dict and len(rare_tags_dict[i]) > 0:
                    # 获取当前层的标签索引
                    layer_indices = train_dataset.tags_indices[:, i]
                    
                    # 创建新的映射：稀有标签 -> 特殊类别ID（使用新的类别数量-1作为特殊类别ID）
                    special_class_id = new_tag_class_counts[i] - 1
                    
                    # 创建映射表：原始ID -> 新ID
                    id_mapping = torch.arange(original_tag_class_counts[i], dtype=torch.long)
                    
                    # 计算非稀有标签的新ID（保持顺序但跳过稀有标签）
                    non_rare_ids = torch.ones(original_tag_class_counts[i], dtype=torch.bool)
                    non_rare_ids[rare_tags_dict[i]] = False
                    
                    # 为非稀有标签分配新ID
                    new_ids = torch.cumsum(non_rare_ids, dim=0) - 1
                    id_mapping[non_rare_ids] = new_ids[non_rare_ids]
                    
                    # 为稀有标签分配特殊类别ID
                    id_mapping[rare_tags_dict[i]] = special_class_id
                    
                    # 应用映射到训练集
                    valid_mask = layer_indices >= 0
                    layer_indices[valid_mask] = id_mapping[layer_indices[valid_mask]]
                    train_dataset.tags_indices[:, i] = layer_indices
                    
                    logger.info(f"层 {i} 标签索引已重新映射，特殊类别ID: {special_class_id}")
            
            # 如果存在评估数据集，也需要重新映射
            if do_eval and hasattr(eval_dataset, 'tags_indices') and eval_dataset.tags_indices is not None:
                for i in range(vae_n_layers):
                    if i in rare_tags_dict and len(rare_tags_dict[i]) > 0:
                        # 获取当前层的标签索引
                        layer_indices = eval_dataset.tags_indices[:, i]
                        
                        # 创建新的映射：稀有标签 -> 特殊类别ID
                        special_class_id = new_tag_class_counts[i] - 1
                        
                        # 创建映射表：原始ID -> 新ID
                        id_mapping = torch.arange(original_tag_class_counts[i], dtype=torch.long)
                        
                        # 计算非稀有标签的新ID
                        non_rare_ids = torch.ones(original_tag_class_counts[i], dtype=torch.bool)
                        non_rare_ids[rare_tags_dict[i]] = False
                        
                        # 为非稀有标签分配新ID
                        new_ids = torch.cumsum(non_rare_ids, dim=0) - 1
                        id_mapping[non_rare_ids] = new_ids[non_rare_ids]
                        
                        # 为稀有标签分配特殊类别ID
                        id_mapping[rare_tags_dict[i]] = special_class_id
                        
                        # 应用映射到评估集
                        valid_mask = layer_indices >= 0
                        layer_indices[valid_mask] = id_mapping[layer_indices[valid_mask]]
                        eval_dataset.tags_indices[:, i] = layer_indices
                        
                        logger.info(f"评估集层 {i} 标签索引已重新映射")
        
        logger.info("标签类别频率统计和稀有标签处理完成")
    
        # 将类别频率统计信息传递给模型，以便标签预测损失可以使用类别权重
        class_counts_tensor_dict = {k: v.to(device) for k, v in class_counts_dict.items()}

    # 创建HRQVAE模型
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
        tag_class_counts=tag_class_counts,  # 使用更新后的标签类别数量
        tag_embed_dim=tag_embed_dim,
        use_focal_loss=use_focal_loss,
        focal_loss_params=focal_loss_params,
        # 新增参数
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
        alignment_temperature=alignment_temperature,
        # 新增语义ID唯一性约束参数
        sem_id_uniqueness_weight=sem_id_uniqueness_weight,
        sem_id_uniqueness_margin=sem_id_uniqueness_margin
    )

    # 如果启用了焦点损失并且已经计算类别频率，则更新模型中的类别计数
    if use_focal_loss and 'class_counts_tensor_dict' in locals():
        model.update_class_counts(class_counts_tensor_dict)

    # 设置标签预测损失函数的参数
    if hasattr(model, 'tag_prediction_loss'):
        model.tag_prediction_loss.use_label_smoothing = use_label_smoothing
        model.tag_prediction_loss.label_smoothing_alpha = label_smoothing_alpha
        model.tag_prediction_loss.use_mixup = use_mixup
        model.tag_prediction_loss.mixup_alpha = mixup_alpha

    # 如果使用层特定学习率，为不同组件使用不同的参数组
    if layer_specific_lr:
        # 将模型参数分为多个组，每个组使用不同的学习率和权重衰减
        param_groups = [
            # 编码器和解码器使用基础学习率
            {'params': list(model.encoder.parameters()) + list(model.decoder.parameters()), 
             'lr': learning_rate, 'weight_decay': weight_decay},
            # 量化层使用基础学习率
            {'params': [p for layer in model.layers for p in layer.parameters()], 
             'lr': learning_rate, 'weight_decay': weight_decay},
        ]
        
        # 为每层标签预测器和投影器使用不同的学习率和权重衰减
        for i in range(vae_n_layers):
            # 预测器的学习率随层数增加而略微增加，权重衰减随层数增加而略微减小
            predictor_lr = learning_rate * (1 + i * 0.1) 
            predictor_wd = predictor_weight_decay / (1 + i * 0.2) if predictor_weight_decay > 0 and (1 + i * 0.2) > 0 else predictor_weight_decay

            param_groups.append({
                'params': model.tag_predictors[i].parameters(),
                'lr': predictor_lr,
                'weight_decay': predictor_wd
            })
            
            # 投影器使用相同的策略
            param_groups.append({
                'params': model.tag_projectors[i].parameters(),
                'lr': predictor_lr,
                'weight_decay': predictor_wd
            })
        
        optimizer = AdamW(param_groups)
        
        if accelerator.is_main_process:
            logger.info("使用层特定学习率和权重衰减")
            for i, group in enumerate(param_groups):
                logger.info(f"参数组 {i}: 学习率={group['lr']}, 权重衰减={group['weight_decay']}")
    else:
        # 使用统一的学习率和权重衰减
        optimizer = AdamW(
            params=model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if accelerator.is_main_process:
        logger.info("模型配置:")
        logger.info(f"  input_dim: {vae_input_dim}")
        logger.info(f"  embed_dim: {vae_embed_dim}")
        logger.info(f"  hidden_dims: {vae_hidden_dims}")
        logger.info(f"  codebook_size: {vae_codebook_size}")
        logger.info(f"  n_layers: {vae_n_layers}")
        logger.info(f"  tag_class_counts: {tag_class_counts}")
        logger.info(f"  tag_embed_dim: {tag_embed_dim}")
        logger.info(f"  tag_alignment_weight: {tag_alignment_weight}")
        logger.info(f"  tag_prediction_weight: {tag_prediction_weight}")
        logger.info(f" len(train_dataset): {len(train_dataset)}")

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

    start_iter = 0
    if pretrained_hrqvae_path is not None:
        model.load_pretrained(pretrained_hrqvae_path)
        state = torch.load(pretrained_hrqvae_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_iter = state["iter"]+1
        if accelerator.is_main_process:
            logger.info(f"加载预训练模型: {pretrained_hrqvae_path}, 从迭代 {start_iter} 开始")

    model, optimizer = accelerator.prepare(
        model, optimizer
    )

    # 初始化学习率调度器
    scheduler = None
    if use_lr_scheduler:
        if lr_scheduler_type == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_scheduler_T_max, eta_min=lr_scheduler_eta_min, last_epoch=start_iter-1 if start_iter > 0 else -1)
            if accelerator.is_main_process:
                logger.info(f"使用CosineAnnealingLR调度器: T_max={lr_scheduler_T_max}, eta_min={lr_scheduler_eta_min}")
        elif lr_scheduler_type == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma, last_epoch=start_iter-1 if start_iter > 0 else -1)
            if accelerator.is_main_process:
                logger.info(f"使用StepLR调度器: step_size={lr_scheduler_step_size}, gamma={lr_scheduler_gamma}")
        # ReduceLROnPlateau需要一个指标来监控，通常是验证损失，所以直接在迭代中调用比较复杂
        # For now, we are not supporting ReduceLROnPlateau directly in this script due to its metric dependency.
        # elif lr_scheduler_type == 'reduce_on_plateau':
        #     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_scheduler_factor, patience=lr_scheduler_patience, verbose=True)
        #     if accelerator.is_main_process:
        #         logger.info(f"使用ReduceLROnPlateau调度器: factor={lr_scheduler_factor}, patience={lr_scheduler_patience}")
        else:
            if accelerator.is_main_process:
                logger.warning(f"不支持的学习率调度器类型: {lr_scheduler_type}. 将不使用调度器.")

    if scheduler: # Prepare scheduler with accelerator if it exists
        scheduler = accelerator.prepare(scheduler)

    # 创建语义ID分词器
    tokenizer = HSemanticIdTokenizer(
        input_dim=vae_input_dim,
        output_dim=vae_embed_dim,
        hidden_dims=vae_hidden_dims,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        hrqvae_weights_path=pretrained_hrqvae_path,
        hrqvae_codebook_normalize=vae_codebook_normalize,
        hrqvae_sim_vq=vae_sim_vq,
        tag_alignment_weight=tag_alignment_weight,
        tag_prediction_weight=tag_prediction_weight,
        tag_class_counts=tag_class_counts,
        tag_embed_dim=tag_embed_dim,
        use_concatenated_ids=use_concatenated_ids,
        use_interleaved_ids=use_interleaved_ids,
        commitment_weight=commitment_weight,
    )
    tokenizer.hrq_vae = accelerator.unwrap_model(model)

    with tqdm(initial=start_iter, total=start_iter+1+iterations,
              disable=not accelerator.is_main_process) as pbar:
        losses = [[], [], [], [], [], []]  # 总损失, 重构损失, RQVAE损失, 标签对齐损失, 标签预测损失, 标签预测准确率
        # 新增按层记录的损失
        tag_align_losses_by_layer = [[] for _ in range(vae_n_layers)]
        tag_pred_losses_by_layer = [[] for _ in range(vae_n_layers)]
        tag_pred_accuracies_by_layer = [[] for _ in range(vae_n_layers)]
        
        for iter in range(start_iter, start_iter+1+iterations):
            model.train()
            total_loss = 0
            t = 0.2
            
            if iter == 0 and use_kmeans_init:
                kmeans_init_data = batch_to(train_dataset[torch.arange(min(20000, len(train_dataset)))], device)
                model(kmeans_init_data, t)
                if accelerator.is_main_process:
                    logger.info("完成K-means初始化")

            optimizer.zero_grad()
            for _ in range(gradient_accumulate_every):
                data = next_batch(train_dataloader, device)
                # # 打印输入数据形状
                # logger.info(f"输入数据形状: {data.x.shape}")
                # # 打印输入数据的多个字段的形状
                # logger.info("=== 输入数据字段形状 ===")
                # for field_name in data._fields:
                #     field_value = getattr(data, field_name)
                #     if isinstance(field_value, torch.Tensor):
                #         logger.info(f"  {field_name}: {field_value.shape}")
                #     elif field_value is not None:
                #         logger.info(f"  {field_name}: {type(field_value)}")

                with accelerator.autocast():
                    model_output = model(data, gumbel_t=t)
                    loss = model_output.loss
                    loss = loss / gradient_accumulate_every
                    total_loss += loss
            
            # 反向传播
            accelerator.backward(total_loss)

            losses[0].append(total_loss.cpu().item())
            # 确保在调用.item()前先计算平均值
            losses[1].append(model_output.reconstruction_loss.mean().cpu().item())
            losses[2].append(model_output.rqvae_loss.mean().cpu().item())
            losses[3].append(model_output.tag_align_loss.mean().cpu().item())
            losses[4].append(model_output.tag_pred_loss.mean().cpu().item())
            losses[5].append(model_output.tag_pred_accuracy.mean().cpu().item())
            
            # 记录按层的标签对齐和预测损失
            if hasattr(model_output, 'tag_align_loss_by_layer') and model_output.tag_align_loss_by_layer is not None:
                for i in range(vae_n_layers):
                    if i < len(model_output.tag_align_loss_by_layer):
                        tag_align_losses_by_layer[i].append(model_output.tag_align_loss_by_layer[i].cpu().item())
                    else:
                        tag_align_losses_by_layer[i].append(0.0)
            
            if hasattr(model_output, 'tag_pred_loss_by_layer') and model_output.tag_pred_loss_by_layer is not None:
                for i in range(vae_n_layers):
                    if i < len(model_output.tag_pred_loss_by_layer):
                        tag_pred_losses_by_layer[i].append(model_output.tag_pred_loss_by_layer[i].cpu().item())
                    else:
                        tag_pred_losses_by_layer[i].append(0.0)
            
            if hasattr(model_output, 'tag_pred_accuracy_by_layer') and model_output.tag_pred_accuracy_by_layer is not None:
                for i in range(vae_n_layers):
                    if i < len(model_output.tag_pred_accuracy_by_layer):
                        tag_pred_accuracies_by_layer[i].append(model_output.tag_pred_accuracy_by_layer[i].cpu().item())
                    else:
                        tag_pred_accuracies_by_layer[i].append(0.0)
            
            # 保持滑动窗口大小为1000
            for i in range(len(losses)):
                losses[i] = losses[i][-1000:]
            
            for i in range(vae_n_layers):
                tag_align_losses_by_layer[i] = tag_align_losses_by_layer[i][-1000:]
                tag_pred_losses_by_layer[i] = tag_pred_losses_by_layer[i][-1000:]
                tag_pred_accuracies_by_layer[i] = tag_pred_accuracies_by_layer[i][-1000:]
            
            if iter % 100 == 0:
                print_loss = np.mean(losses[0])
                print_rec_loss = np.mean(losses[1])
                print_vae_loss = np.mean(losses[2])
                print_tag_align_loss = np.mean(losses[3])
                print_tag_pred_loss = np.mean(losses[4])
                print_tag_pred_acc = np.mean(losses[5])

            pbar.set_description(f'loss: {print_loss:.4f}, rl: {print_rec_loss:.4f}, vl: {print_vae_loss:.4f}, tal: {print_tag_align_loss:.4f}, tpl: {print_tag_pred_loss:.4f}, acc: {print_tag_pred_acc:.4f}')

            accelerator.wait_for_everyone()

            optimizer.step()
            
            # 更新学习率 (如果使用了调度器)
            if scheduler and lr_scheduler_type != 'reduce_on_plateau': # ReduceLROnPlateau is typically stepped after validation
                scheduler.step()
            
            accelerator.wait_for_everyone()

            id_diversity_log = {}
            if accelerator.is_main_process:
                # 使用logging记录训练信息
                if iter % 100 == 0:  # 每100次迭代记录一次详细日志
                    # 计算嵌入范数平均值
                    emb_norms_avg = model_output.embs_norm.mean(axis=0)
                    emb_norms_str = ", ".join([f"layer_{i}: {emb_norms_avg[i].cpu().item():.4f}" for i in range(vae_n_layers)])
                    
                    # 收集绘图数据
                    plot_data['iterations'].append(iter)
                    plot_data['total_loss'].append(total_loss.cpu().item())
                    # Fix these lines to use mean() before item()
                    plot_data['reconstruction_loss'].append(model_output.reconstruction_loss.mean().cpu().item())
                    plot_data['rqvae_loss'].append(model_output.rqvae_loss.mean().cpu().item())
                    plot_data['tag_align_loss'].append(model_output.tag_align_loss.mean().cpu().item())
                    plot_data['tag_pred_loss'].append(model_output.tag_pred_loss.mean().cpu().item())
                    plot_data['tag_pred_accuracy'].append(model_output.tag_pred_accuracy.mean().cpu().item())
                    
                    for i in range(vae_n_layers):
                        plot_data['emb_norms'][i].append(emb_norms_avg[i].cpu().item())
                    
                    logger.info(f"迭代 {iter} - 损失: {total_loss.cpu().item():.4f}, "
                               f"重构损失: {model_output.reconstruction_loss.mean().cpu().item():.4f}, "
                               f"RQVAE损失: {model_output.rqvae_loss.mean().cpu().item():.4f}, "
                               f"标签对齐损失: {model_output.tag_align_loss.mean().cpu().item():.4f}, "
                               f"标签预测损失: {model_output.tag_pred_loss.mean().cpu().item():.4f}, "
                               f"标签预测准确率: {model_output.tag_pred_accuracy.mean().cpu().item():.4f}, "
                               f"温度: {t:.4f}, "
                               f"唯一ID比例: {model_output.p_unique_ids.cpu().item():.4f}, "
                               f"嵌入范数: {emb_norms_str}")
                    
                    # 打印每层的标签预测准确率
                    if model_output.tag_pred_accuracy_by_layer is not None:
                        layer_acc_str = ", ".join([f"层{i}: {acc.cpu().item():.4f}" for i, acc in enumerate(model_output.tag_pred_accuracy_by_layer)])
                        logger.info(f"每层标签预测准确率: {layer_acc_str}")
                    
                    # 记录当前学习率
                    current_lrs = [group['lr'] for group in optimizer.param_groups]
                    lr_str = ", ".join([f"{lr:.2e}" for lr in current_lrs])
                    logger.info(f"当前学习率: {lr_str}")
            # 评估阶段
            if do_eval and ((iter+1) % eval_every == 0 or iter+1 == iterations):
                model.eval()
                with tqdm(eval_dataloader, desc=f'Eval {iter+1}', disable=not accelerator.is_main_process) as pbar_eval:
                    eval_losses = [[], [], [], [], [], []]  # 总损失, 重构损失, RQVAE损失, 标签对齐损失, 标签预测损失, 标签预测准确率
                    
                    # 收集验证样本，用于展示预测结果
                    eval_samples = []
                    # 确保在这里初始化预测标签列表
                    predicted_tags_list = []
                    
                    for batch in pbar_eval:
                        data = batch_to(batch, device)
                        with torch.no_grad():
                            eval_model_output = model(data, gumbel_t=t)
                        
                        eval_losses[0].append(eval_model_output.loss.cpu().item())
                        eval_losses[1].append(eval_model_output.reconstruction_loss.mean().cpu().item())
                        eval_losses[2].append(eval_model_output.rqvae_loss.mean().cpu().item())
                        eval_losses[3].append(eval_model_output.tag_align_loss.mean().cpu().item())
                        eval_losses[4].append(eval_model_output.tag_pred_loss.mean().cpu().item())
                        eval_losses[5].append(eval_model_output.tag_pred_accuracy.mean().cpu().item())
                        
                        # 收集样本数据（限制总数）
                        if len(eval_samples) < 100 and hasattr(data, 'tags_indices') and data.tags_indices is not None:
                            # 收集样本的输入特征和真实标签
                            batch_size = data.x.size(0)
                            samples_to_collect = min(batch_size, 100 - len(eval_samples))
                            
                            for i in range(samples_to_collect):
                                # 获取当前样本的特征和真实标签
                                sample_x = data.x[i].cpu()
                                sample_tags = data.tags_indices[i].cpu()
                                
                                # 检查样本是否有至少一个有效标签
                                has_valid_tag = (sample_tags >= 0).any().item()
                                
                                if has_valid_tag:
                                    # 只收集有有效标签的样本信息
                                    eval_samples.append({
                                        'x': sample_x,
                                        'true_tags': sample_tags
                                    })
                    
                    # 记录收集到的总样本数
                    logger.info(f"收集到 {len(eval_samples)} 个有效样本用于评估")
                    
                    eval_losses = np.array(eval_losses).mean(axis=-1)
                    id_diversity_log["eval_total_loss"] = eval_losses[0]
                    id_diversity_log["eval_reconstruction_loss"] = eval_losses[1]
                    id_diversity_log["eval_rqvae_loss"] = eval_losses[2]
                    id_diversity_log["eval_tag_align_loss"] = eval_losses[3]
                    id_diversity_log["eval_tag_pred_loss"] = eval_losses[4]
                    id_diversity_log["eval_tag_pred_accuracy"] = eval_losses[5]
                    
                    if accelerator.is_main_process and len(eval_samples) > 0:
                        # 对收集的样本进行标签预测
                        sample_batch_size = 20  # 每批处理的样本数
                        predicted_tags_list = []
                        
                        for i in range(0, len(eval_samples), sample_batch_size):
                            batch_slice = eval_samples[i:i+sample_batch_size]
                            batch_x = torch.stack([sample['x'] for sample in batch_slice]).to(device)
                            
                            # 使用测试时增强(TTA)进行预测
                            if eval_tta:
                                # 多次前向传播并平均结果
                                n_augmentations = 5  # 增强次数
                                all_layer_predictions = [[] for _ in range(vae_n_layers)]
                                
                                for aug_idx in range(n_augmentations):
                                    # 添加小噪声来创建不同版本的输入
                                    if aug_idx > 0:  # 第一次使用原始输入
                                        noise_scale = 0.02 * aug_idx  # 逐渐增加噪声
                                        noise = torch.randn_like(batch_x) * noise_scale
                                        augmented_x = batch_x + noise
                                    else:
                                        augmented_x = batch_x
                                    
                                    # 对增强后的输入进行预测
                                    with torch.no_grad():
                                        # 获取语义ID和拼接的嵌入
                                        res = model.encode(augmented_x)
                                        embs = []  # 存储每层的嵌入
                                        
                                        # 对每一层进行预测
                                        for layer_idx, layer in enumerate(model.layers):
                                            # 获取量化嵌入
                                            quantized = layer(res, temperature=0.001)
                                            emb = quantized.embeddings
                                            
                                            # 添加当前层的嵌入
                                            embs.append(emb)
                                            
                                            # 拼接前layer_idx+1层的嵌入
                                            concat_emb = torch.cat(embs, dim=-1)
                                            
                                            # 使用对应层的标签预测器预测标签
                                            tag_logits = model.tag_predictors[layer_idx](concat_emb)
                                            
                                            # 调整softmax温度以增加置信度
                                            tag_logits = tag_logits / eval_temperature
                                            tag_probs = F.softmax(tag_logits, dim=-1)
                                            
                                            # 收集这一次增强的预测结果
                                            all_layer_predictions[layer_idx].append(tag_probs)
                                            
                                            # 更新残差
                                            res = res - emb
                                
                                # 对多次增强的预测进行平均
                                ensemble_predictions = []
                                for layer_idx in range(vae_n_layers):
                                    # 确保当前层有预测结果
                                    if len(all_layer_predictions[layer_idx]) > 0:
                                        try:
                                            # 平均每一层的预测概率
                                            avg_probs = torch.stack(all_layer_predictions[layer_idx], dim=0).mean(dim=0)
                                            # 获取最可能的类别
                                            _, pred_indices = torch.max(avg_probs, dim=1)
                                            ensemble_predictions.append(pred_indices)
                                        except Exception as e:
                                            logger.warning(f"层 {layer_idx} 集成预测失败: {str(e)}")
                                            # 创建一个默认预测（全零）
                                            default_pred = torch.zeros(batch_x.size(0), dtype=torch.long, device=device)
                                            ensemble_predictions.append(default_pred)
                                    else:
                                        # 如果没有预测结果，添加一个默认预测
                                        logger.warning(f"层 {layer_idx} 没有预测结果，使用默认预测")
                                        default_pred = torch.zeros(batch_x.size(0), dtype=torch.long, device=device)
                                        ensemble_predictions.append(default_pred)
                                
                                # 确保有足够的预测结果用于堆叠
                                if len(ensemble_predictions) == vae_n_layers:
                                    # 将预测标签转换为张量 [batch_size, n_layers]
                                    batch_predicted_tags = torch.stack(ensemble_predictions, dim=1).cpu()
                                    
                                    # 添加到预测列表
                                    predicted_tags_list.append(batch_predicted_tags)
                                else:
                                    logger.warning(f"集成预测不完整: 预期 {vae_n_layers} 层，实际 {len(ensemble_predictions)} 层")
                            else:
                                # 使用标准预测（无增强）
                                with torch.no_grad():
                                    # 获取语义ID和拼接的嵌入
                                    res = model.encode(batch_x)
                                    predicted_tags = []
                                    
                                    embs = []  # 存储每层的嵌入
                                    
                                    # 对每一层进行预测
                                    for layer_idx, layer in enumerate(model.layers):
                                        # 获取量化嵌入
                                        quantized = layer(res, temperature=0.001)
                                        emb = quantized.embeddings
                                        
                                        # 添加当前层的嵌入
                                        embs.append(emb)
                                        
                                        # 拼接前layer_idx+1层的嵌入
                                        concat_emb = torch.cat(embs, dim=-1)
                                        
                                        # 使用对应层的标签预测器预测标签
                                        tag_logits = model.tag_predictors[layer_idx](concat_emb)
                                        
                                        # 获取预测的标签索引
                                        _, pred_indices = torch.max(tag_logits, dim=1)
                                        predicted_tags.append(pred_indices)
                                        
                                        # 更新残差
                                        res = res - emb
                                    
                                    # 将预测标签转换为张量 [batch_size, n_layers]
                                    batch_predicted_tags = torch.stack(predicted_tags, dim=1).cpu()
                                
                                # 添加到预测列表
                                predicted_tags_list.append(batch_predicted_tags)
                        
                        # 合并所有批次的预测结果
                        # 添加检查确保列表非空
                        if predicted_tags_list:
                            all_predicted_tags = torch.cat(predicted_tags_list, dim=0)
                        else:
                            logger.warning("预测标签列表为空，可能是评估样本没有有效标签或预测失败")
                            # 创建一个空张量作为占位符
                            all_predicted_tags = torch.zeros((0, vae_n_layers), dtype=torch.long)
                        
                        # 计算每层的准确率
                        layer_accuracies = []
                        
                        # 检查是否有有效的预测样本
                        if len(eval_samples) > 0 and all_predicted_tags.size(0) > 0:
                            for layer_idx in range(vae_n_layers):
                                correct = 0
                                total = 0
                                
                                # 遍历所有样本
                                for i, sample in enumerate(eval_samples):
                                    if i >= all_predicted_tags.size(0):
                                        break
                                        
                                    # 确保样本包含当前层的标签
                                    if layer_idx < len(sample['true_tags']):
                                        try:
                                            true_tag = sample['true_tags'][layer_idx].item()
                                            pred_tag = all_predicted_tags[i, layer_idx].item()
                                            
                                            if true_tag >= 0:  # 只统计有效标签
                                                total += 1
                                                if true_tag == pred_tag:
                                                    correct += 1
                                        except Exception as e:
                                            logger.warning(f"计算准确率时出错: {str(e)}")
                                            continue
                                
                                accuracy = correct / max(1, total)  # 避免除零错误
                                layer_accuracies.append(accuracy)
                                logger.info(f"层 {layer_idx} 标签预测准确率: {accuracy:.4f} (正确: {correct}/{total})")
                        else:
                            # 如果没有有效的预测样本，记录警告
                            logger.warning("没有有效的预测样本或预测结果，无法计算准确率")
                            for _ in range(vae_n_layers):
                                layer_accuracies.append(0.0)
                        
                        # 打印样本的预测结果
                        if len(eval_samples) > 0 and all_predicted_tags.size(0) > 0:
                            logger.info(f"\n==== 验证集样本预测结果 (显示 {min(len(eval_samples), all_predicted_tags.size(0))} 条) ====")
                            
                            for i, sample in enumerate(eval_samples):
                                if i >= all_predicted_tags.size(0) or i >= 99:  # 只显示最多100条记录
                                    break
                                
                                true_tags = sample['true_tags']
                                pred_tags = all_predicted_tags[i]
                                
                                # 检查标签形状是否匹配
                                if true_tags.size(0) != pred_tags.size(0):
                                    logger.warning(f"样本 {i+1} 的标签形状不匹配: 真实={true_tags.size()}, 预测={pred_tags.size()}")
                                    continue
                                
                                # 格式化输出
                                sample_str = f"样本 {i+1}:\n"
                                
                                # 添加输入特征（只显示前几个值，避免输出过长）
                                x_sample = sample['x']
                                x_preview = x_sample[:10].numpy()  # 只显示前10个值
                                sample_str += f"  输入特征(前10个值): {x_preview}\n"
                                
                                # 添加每层的真实标签和预测标签
                                for layer_idx in range(min(len(true_tags), len(pred_tags))):
                                    try:
                                        true_tag = true_tags[layer_idx].item()
                                        pred_tag = pred_tags[layer_idx].item()
                                        
                                        # 检查标签是否有效（非负）
                                        if true_tag >= 0:
                                            match_str = "✓" if true_tag == pred_tag else "✗"
                                            sample_str += f"  层 {layer_idx + 1}: 真实标签={true_tag}, 预测标签={pred_tag} {match_str}\n"
                                        else:
                                            sample_str += f"  层 {layer_idx + 1}: 真实标签=无效, 预测标签={pred_tag}\n"
                                    except Exception as e:
                                        logger.warning(f"处理样本 {i+1} 层 {layer_idx} 时出错: {str(e)}")
                                        continue
                                
                                logger.info(sample_str)
                            
                            logger.info("==== 验证集样本预测结果结束 ====\n")
                        else:
                            logger.warning("没有有效的预测样本或预测结果，无法显示预测结果")
                    
                        # 收集评估数据用于绘图
                        plot_data['eval_iterations'].append(iter+1)
                        plot_data['eval_total_loss'].append(eval_losses[0])
                        plot_data['eval_reconstruction_loss'].append(eval_losses[1])
                        plot_data['eval_rqvae_loss'].append(eval_losses[2])
                        plot_data['eval_tag_align_loss'].append(eval_losses[3])
                        plot_data['eval_tag_pred_loss'].append(eval_losses[4])
                        plot_data['eval_tag_pred_accuracy'].append(eval_losses[5])
                        
                        logger.info(f"评估 {iter+1} - 总损失: {eval_losses[0]:.4f}, "
                                   f"重构损失: {eval_losses[1]:.4f}, "
                                   f"RQVAE损失: {eval_losses[2]:.4f}, "
                                   f"标签对齐损失: {eval_losses[3]:.4f}, "
                                   f"标签预测损失: {eval_losses[4]:.4f}, "
                                   f"标签预测准确率: {eval_losses[5]:.4f}")
                    
                        # 打印每层的标签预测准确率
                        if eval_tta:
                            logger.info(f"TTA每层准确率: {', '.join([f'层{i}: {acc:.4f}' for i, acc in enumerate(layer_accuracies)])}")
                        elif eval_model_output.tag_pred_accuracy_by_layer is not None:
                            eval_layer_acc_str = ", ".join([f"层{i}: {acc.cpu().item():.4f}" for i, acc in enumerate(eval_model_output.tag_pred_accuracy_by_layer)])
                            logger.info(f"评估 - 每层标签预测准确率: {eval_layer_acc_str}")

                    # 新的模型保存逻辑：只保存准确率超过70%且语义ID重复率低于阈值的模型
                    current_eval_accuracy = eval_losses[5]  # eval_tag_pred_accuracy
                    current_rqvae_loss = eval_losses[2]   # eval_rqvae_loss

                    # 先计算ID多样性指标，确保sem_repetition_rate变量已定义
                    tokenizer.reset()
                    model.eval()
                    tokenizer.hrq_vae = accelerator.unwrap_model(model) #确保tokenizer用的是最新的模型

                    corpus_ids = tokenizer.precompute_corpus_ids(index_dataset)
                    max_duplicates = corpus_ids[:,-1].max() / corpus_ids.shape[0]
                    
                    # 修复：在这里定义cid变量，使用最后一层的索引
                    cid = vae_n_layers - 1  # 使用最后一层的索引
                    _, counts = torch.unique(corpus_ids[:,cid], dim=0, return_counts=True)
                    p = counts / corpus_ids.shape[0]
                    rqvae_entropy = -(p*torch.log(p)).sum()

                    codebook_usage_info = []
                    for cid in range(vae_n_layers):
                        _, counts = torch.unique(corpus_ids[:,cid], return_counts=True)
                        usage = len(counts) / vae_codebook_size
                        id_diversity_log[f"codebook_usage_{cid}"] = usage
                        codebook_usage_info.append(f"层{cid}: {usage:.4f}")
                        
                        # 收集码本使用率数据用于绘图
                        plot_data['codebook_usage'][cid].append(usage)
                    
                    # 新增：计算并打印语义ID部分的重复率
                    # 只取前vae_n_layers维度（语义ID部分）
                    semantic_ids = corpus_ids[:, :vae_n_layers]
                    sem_repetition_rate, sem_unique_items, sem_total_items = calculate_repetition_rate(semantic_ids)
                    logger.info(f"仅语义ID部分的重复率: {sem_repetition_rate:.4f} ({sem_unique_items} unique / {sem_total_items} total)")

                    plot_data['rqvae_entropy'].append(rqvae_entropy.cpu().item())
                    plot_data['max_id_duplicates'].append(max_duplicates.cpu().item())
                    
                    logger.info(f"ID多样性 {iter+1} - "
                               f"RQVAE熵: {rqvae_entropy.cpu().item():.4f}, "
                               f"最大ID重复: {max_duplicates.cpu().item():.4f}, "
                               f"码本使用率: {', '.join(codebook_usage_info)}")

                    # 现在可以安全地使用sem_repetition_rate变量
                    if current_eval_accuracy > 0.60 and sem_repetition_rate < id_repetition_threshold:
                        logger.info(f"模型准确率和ID重复率均达到阈值！准确率: {current_eval_accuracy:.4f}, RQVAE损失: {current_rqvae_loss:.4f}, 语义ID重复率: {sem_repetition_rate:.4f}")

                        # 保存新的最佳模型
                        model_save_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # 确保保存目录存在 (save_dir 在函数开头定义)
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                        new_best_model_filename = f"hrqvae_model_ACC{current_eval_accuracy:.4f}_RQLOSS{current_rqvae_loss:.4f}_DUPR{sem_repetition_rate:.4f}_{model_save_timestamp}.pt"
                        new_model_path = os.path.join(save_dir, new_best_model_filename)
                        
                        unwrapped_model = accelerator.unwrap_model(model)
                        state = {
                            "iter": iter + 1,
                            "model": unwrapped_model.state_dict(),
                            "model_config": unwrapped_model.config,
                            "optimizer": optimizer.state_dict(),
                            "accuracy": current_eval_accuracy,
                            "rqvae_loss": current_rqvae_loss,
                            "sem_id_repetition_rate": sem_repetition_rate  # 新增：记录语义ID重复率
                        }
                        
                        accelerator.save(state, new_model_path)
                        logger.info(f"模型已保存到: {new_model_path}")
                    else:
                        if current_eval_accuracy <= 0.70:
                            logger.info(f"当前评估准确率 {current_eval_accuracy:.4f} 未达到 0.70 阈值，不保存模型。最佳准确率仍为 {best_eval_accuracy:.4f}")
                        if sem_repetition_rate >= id_repetition_threshold:
                            logger.info(f"当前语义ID重复率 {sem_repetition_rate:.4f} 高于 {id_repetition_threshold} 阈值，不保存模型。")

                # 计算ID多样性指标
                if (iter+1) % eval_every == 0 or iter+1 == iterations:
                    # 这部分代码已经移到上面，避免重复计算
                    pass

            pbar.update(1)
    
    # 训练结束时绘制最终图表
    if accelerator.is_main_process:
        logger.info("训练完成，正在生成训练过程图表...")
        # 绘制所有图表
        plot_all_metrics(plot_data, plots_dir, vae_n_layers)
        logger.info("所有图表已保存到 {}".format(plots_dir))


def plot_all_metrics(plot_data, plots_dir, n_layers):
    """绘制所有指标的整体训练过程图表"""
    
    # 1. 绘制训练损失图
    plt.figure(figsize=(12, 8))
    plt.plot(plot_data['iterations'], plot_data['total_loss'], label='Total Loss')
    plt.plot(plot_data['iterations'], plot_data['reconstruction_loss'], label='Reconstruction Loss')
    plt.plot(plot_data['iterations'], plot_data['rqvae_loss'], label='RQVAE Loss')
    plt.plot(plot_data['iterations'], plot_data['tag_align_loss'], label='Tag Alignment Loss')
    plt.plot(plot_data['iterations'], plot_data['tag_pred_loss'], label='Tag Prediction Loss')
    
    if plot_data['eval_iterations']:
        plt.plot(plot_data['eval_iterations'], plot_data['eval_total_loss'], 'o-', label='Eval Total Loss')
        plt.plot(plot_data['eval_iterations'], plot_data['eval_reconstruction_loss'], 'o-', label='Eval Reconstruction Loss')
        plt.plot(plot_data['eval_iterations'], plot_data['eval_rqvae_loss'], 'o-', label='Eval RQVAE Loss')
        plt.plot(plot_data['eval_iterations'], plot_data['eval_tag_align_loss'], 'o-', label='Eval Tag Alignment Loss')
        plt.plot(plot_data['eval_iterations'], plot_data['eval_tag_pred_loss'], 'o-', label='Eval Tag Prediction Loss')
    
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'losses.png'))
    plt.close()
    
    # 2. 绘制标签预测准确率图
    plt.figure(figsize=(12, 8))
    plt.plot(plot_data['iterations'], plot_data['tag_pred_accuracy'], label='Tag Prediction Accuracy')
    
    if plot_data['eval_iterations']:
        plt.plot(plot_data['eval_iterations'], plot_data['eval_tag_pred_accuracy'], 'o-', label='Eval Tag Prediction Accuracy')
    
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Tag Prediction Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'tag_accuracy.png'))
    plt.close()
    
    # 3. 绘制嵌入范数图
    plt.figure(figsize=(12, 8))
    for i in range(n_layers):
        plt.plot(plot_data['iterations'], plot_data['emb_norms'][i], label=f'Layer {i} Embedding Norm')
    
    plt.xlabel('Iterations')
    plt.ylabel('Embedding Norm')
    plt.title('Embedding Norms by Layer')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'embedding_norms.png'))
    plt.close()
    
    # 4. 绘制码本使用率图
    plt.figure(figsize=(12, 8))
    for i in range(n_layers):
        if plot_data['codebook_usage'][i]:
            eval_iters = plot_data['eval_iterations'][:len(plot_data['codebook_usage'][i])]
            plt.plot(eval_iters, plot_data['codebook_usage'][i], 'o-', label=f'Layer {i} Codebook Usage')
    
    plt.xlabel('Iterations')
    plt.ylabel('Codebook Usage')
    plt.title('Codebook Usage by Layer')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'codebook_usage.png'))
    plt.close()
    
    # 5. 绘制ID多样性指标图
    if plot_data['rqvae_entropy']:
        plt.figure(figsize=(12, 8))
        eval_iters = plot_data['eval_iterations'][:len(plot_data['rqvae_entropy'])]
        plt.plot(eval_iters, plot_data['rqvae_entropy'], 'o-', label='RQVAE Entropy')
        plt.plot(eval_iters, plot_data['max_id_duplicates'], 'o-', label='Max ID Duplicates')
        
        plt.xlabel('Iterations')
        plt.ylabel('Value')
        plt.title('ID Diversity Metrics')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'id_diversity.png'))
        plt.close()


if __name__ == "__main__":
    parse_config()
    train()

    
