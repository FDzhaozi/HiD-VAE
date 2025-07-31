import argparse
import os
import gin
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 4
torch._dynamo.config.dynamic_shapes = False
torch._dynamo.config.optimize_ddp = False

from accelerate import Accelerator
from data.processed import ItemData
from data.processed import RecDataset
from data.processed import SeqData
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
# 导入部分需要添加 NDCGAccumulator
from evaluate.metrics import TopKAccumulator
from evaluate.metrics import NDCGAccumulator
from modules.model import EncoderDecoderRetrievalModel
from modules.scheduler.inv_sqrt import InverseSquareRootScheduler
from modules.tokenizer.semids import SemanticIdTokenizer
# 导入HSemanticIdTokenizer
from modules.tokenizer.h_semids import HSemanticIdTokenizer
from modules.utils import compute_debug_metrics
from modules.utils import parse_config
from huggingface_hub import login
from torch.optim import AdamW
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from tqdm import tqdm

class MetricsTracker:
    def __init__(self, metrics_names):
        self.metrics = {name: [] for name in metrics_names}
        self.iterations = []
    
    def update(self, iteration, **kwargs):
        self.iterations.append(iteration)
        for name, value in kwargs.items():
            if name in self.metrics:
                self.metrics[name].append(value)
    
    def plot_and_save(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        
        for metric_name, values in self.metrics.items():
            if len(values) > 0:
                plt.figure(figsize=(10, 6))
                
                # 确保x和y维度匹配
                iterations = self.iterations[:len(values)]
                
                plt.plot(iterations, values)
                plt.xlabel('Iterations')
                plt.ylabel(metric_name)
                plt.title(f'{metric_name} over Training')
                plt.grid(True)
                plt.savefig(os.path.join(save_path, f'{metric_name}_curve.png'))
                plt.close()

# 定义计算重复率的函数
def calculate_repetition_rate(item_ids: torch.Tensor):
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
    iterations=500000,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=0.01,
    dataset_folder="dataset/ml-1m",
    save_dir_root="out/",
    dataset=RecDataset.ML_1M,
    pretrained_rqvae_path=None,
    pretrained_decoder_path=None,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    partial_eval_every=100,
    full_eval_every=10000,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_sim_vq=False,
    vae_n_cat_feats=18,
    vae_n_layers=3,
    decoder_embed_dim=64,
    dropout_p=0.1,
    attn_heads=8,
    attn_embed_dim=64,
    attn_layers=4,
    dataset_split="beauty",
    push_vae_to_hf=False,
    train_data_subsample=True,
    model_jagged_mode=True,
    vae_hf_model_name="edobotta/rqvae-amazon-beauty",
    # 新增参数，用于选择tokenizer类型
    use_h_tokenizer=False,
    # 新增参数，用于标签预测相关配置
    tag_alignment_weight=0.5,
    tag_prediction_weight=0.5,
    tag_class_counts=None,
    tag_embed_dim=768,
    use_dedup_dim=False,
    # 新增参数，用于拼接模式
    use_concatenated_ids=True,  # 默认开启拼接模式，与use_dedup_dim互斥
    use_interleaved_ids=False, # 新增参数，用于控制交错模式
):  
    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 文件夹带时间戳
    log_dir = os.path.join(save_dir_root, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志记录
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("train_decoder")
    
    # 创建指标跟踪器
    metrics_tracker = MetricsTracker([
        'loss', 'learning_rate', 'eval_loss', 
        'hit@1', 'hit@5', 'hit@10',
        'ndcg@1', 'ndcg@5', 'ndcg@10'
    ])
    
    # 记录训练参数
    params = locals()
    logger.info("Training parameters:")
    for key, value in params.items():
        if key != 'metrics_tracker':
            logger.info(f"  {key}: {value}")
    
    if dataset != RecDataset.AMAZON and dataset != RecDataset.KUAIRAND:
        error_msg = f"Dataset currently not supported: {dataset}."
        logger.error(error_msg)
        raise Exception(error_msg)

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )

    device = accelerator.device
    logger.info(f"Using device: {device}")
    
    item_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        force_process=force_dataset_process,
        split=dataset_split
    )
    train_dataset = SeqData(
        root=dataset_folder, 
        dataset=dataset, 
        is_train=True, 
        subsample=train_data_subsample, 
        split=dataset_split
    )
    eval_dataset = SeqData(
        root=dataset_folder, 
        dataset=dataset, 
        is_train=False, 
        subsample=False, 
        split=dataset_split
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    # 分别打印训练集和测试集的前1条数据
    logger.info("Train dataset sample:")
    for i in range(1):
        logger.info(f"  {train_dataset[i]}")

    logger.info("Eval dataset sample:")
    for i in range(1):
        logger.info(f"  {eval_dataset[i]}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    train_dataloader = cycle(train_dataloader)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    
    train_dataloader, eval_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader
    )

    # 根据参数选择使用哪种tokenizer
    if use_h_tokenizer:
        logger.info("Using HSemanticIdTokenizer with tag prediction capabilities")
        # 确保use_dedup_dim与use_concatenated_ids/use_interleaved_ids互斥
        if use_dedup_dim and (use_concatenated_ids or use_interleaved_ids):
            logger.warning(f"use_dedup_dim ({use_dedup_dim}) 与 use_concatenated_ids ({use_concatenated_ids}) 或 use_interleaved_ids ({use_interleaved_ids}) 互斥。")
            logger.warning("将强制设置 use_dedup_dim=False 以避免HSemanticIdTokenizer初始化错误。")
            use_dedup_dim = False
        
        if use_concatenated_ids:
            logger.info("使用拼接模式: 语义ID和标签ID将被拼接在一起")
        
        # 硬编码正确的tag_class_counts值，确保与实际模型一致
        # 根据错误信息确定的实际值为[7, 30, 97]
        model_tag_class_counts = [7, 30, 97]
        logger.info(f"使用硬编码的标签类别计数: {model_tag_class_counts}，而不是配置文件中的: {tag_class_counts}")
        
        tokenizer = HSemanticIdTokenizer(
            input_dim=vae_input_dim,
            hidden_dims=vae_hidden_dims,
            output_dim=vae_embed_dim,
            codebook_size=vae_codebook_size,
            n_layers=vae_n_layers,
            n_cat_feats=vae_n_cat_feats,
            hrqvae_weights_path=pretrained_rqvae_path,
            hrqvae_codebook_normalize=True,  # 确保这里是True
            hrqvae_sim_vq=vae_sim_vq,
            tag_alignment_weight=tag_alignment_weight,
            tag_prediction_weight=tag_prediction_weight,
            tag_class_counts=model_tag_class_counts,  # 使用硬编码的值
            tag_embed_dim=tag_embed_dim,
            use_dedup_dim=use_dedup_dim,
            use_concatenated_ids=use_concatenated_ids,  # 传递拼接模式参数
            use_interleaved_ids=use_interleaved_ids # 传递交错模式参数
        )
    else:
        logger.info("Using standard SemanticIdTokenizer")
        tokenizer = SemanticIdTokenizer(
            input_dim=vae_input_dim,
            hidden_dims=vae_hidden_dims,
            output_dim=vae_embed_dim,
            codebook_size=vae_codebook_size,
            n_layers=vae_n_layers,
            n_cat_feats=vae_n_cat_feats,
            rqvae_weights_path=pretrained_rqvae_path,
            rqvae_codebook_normalize=vae_codebook_normalize,
            rqvae_sim_vq=vae_sim_vq,
            use_dedup_dim=use_dedup_dim
        )
    
    # 进行预处理
    tokenizer = accelerator.prepare(tokenizer)
    tokenizer.precompute_corpus_ids(item_dataset)
    logger.info("Tokenizer prepared and corpus IDs precomputed")
    
    # 计算和打印ID重复率
    if accelerator.is_main_process: # 只在主进程执行
        cached_item_ids = tokenizer.cached_ids
        if cached_item_ids is not None and cached_item_ids.numel() > 0: # 增加cached_item_ids.numel() > 0判断
            if use_dedup_dim: # 当使用SemanticIdTokenizer且use_dedup_dim=True时
                logger.info("use_dedup_dim is True (通常与 SemanticIdTokenizer 一起使用)。计算原始ID部分和最终ID的重复率。")
                # 原始ID部分 (去掉最后一列的去重维度)
                if cached_item_ids.shape[1] > 1:
                    original_ids_part = cached_item_ids[:, :-1]
                    original_rep_rate, num_unique_orig, total_orig = calculate_repetition_rate(original_ids_part)
                    logger.info(f"  原始ID部分 (前 {original_ids_part.shape[1]} 维) 的重复率: {original_rep_rate:.4f} ({num_unique_orig} unique / {total_orig} total)")
                else:
                    logger.warning("cached_item_ids 维度不足以分离原始ID和去重维度。")

                # 最终ID (包含去重维度)
                final_rep_rate, num_unique_final, total_final = calculate_repetition_rate(cached_item_ids)
                logger.info(f"  最终ID (共 {cached_item_ids.shape[1]} 维, 含去重维度) 的重复率: {final_rep_rate:.4f} ({num_unique_final} unique / {total_final} total)")
            else: # use_dedup_dim is False (当使用HSemanticIdTokenizer, 或 SemanticIdTokenizer且use_dedup_dim=False时)
                logger.info("use_dedup_dim is False. 计算完整ID的重复率 (对于HSemanticIdTokenizer，这可能包含语义ID和标签ID)。")
                rep_rate, num_unique, total = calculate_repetition_rate(cached_item_ids)
                logger.info(f"  完整ID (共 {cached_item_ids.shape[1]} 维) 的重复率: {rep_rate:.4f} ({num_unique} unique / {total} total)")

                # 如果使用的是HSemanticIdTokenizer并且拼接或交错ID，额外计算并打印仅语义ID部分的重复率
                if use_h_tokenizer and (use_concatenated_ids or use_interleaved_ids):
                    num_semantic_layers = tokenizer.n_layers  # 这是 HRQ-VAE 的 n_layers
                    
                    semantic_part_ids = None
                    if num_semantic_layers > 0 and cached_item_ids.shape[1] > 0:
                        if use_concatenated_ids:
                            if cached_item_ids.shape[1] >= num_semantic_layers:
                                semantic_part_ids = cached_item_ids[:, :num_semantic_layers]
                            else:
                                logger.warning(f"拼接模式下，cached_item_ids维度 ({cached_item_ids.shape[1]}) 小于语义层数 ({num_semantic_layers})，无法提取语义部分。")
                        elif use_interleaved_ids:
                            # 提取交错的语义ID：索引为 0, 2, 4, ...
                            semantic_indices = [i * 2 for i in range(num_semantic_layers) if i * 2 < cached_item_ids.shape[1]]
                            if semantic_indices:
                                semantic_part_ids = cached_item_ids[:, semantic_indices]
                            else:
                                logger.warning(f"交错模式下，无法根据语义层数 ({num_semantic_layers}) 和总维度 ({cached_item_ids.shape[1]}) 提取有效的语义ID索引。")
                        
                        if semantic_part_ids is not None and semantic_part_ids.numel() > 0:
                            logger.info(f"  同时，计算仅语义ID部分 (从完整ID中提取的 {semantic_part_ids.shape[1]} 维) 的重复率:")
                            sem_rep_rate, sem_num_unique, sem_total = calculate_repetition_rate(semantic_part_ids)
                            logger.info(f"    仅语义ID部分的重复率: {sem_rep_rate:.4f} ({sem_num_unique} unique / {sem_total} total)")
                        elif num_semantic_layers > 0 : # semantic_part_ids is None or empty but should not be
                            logger.warning(f"  未能成功提取仅语义ID部分进行重复率计算。num_semantic_layers: {num_semantic_layers}")
                    elif num_semantic_layers == 0:
                        logger.info("  num_semantic_layers 为 0，不计算仅语义ID部分的重复率。")

        else:
            logger.warning("tokenizer.cached_ids 为 None 或为空，无法计算重复率。")
    
    if push_vae_to_hf:
        login()
        tokenizer.rq_vae.push_to_hub(vae_hf_model_name)
        logger.info(f"Pushed VAE to HuggingFace: {vae_hf_model_name}")

    # 获取语义ID层数以正确区分语义ID和标签ID
    n_sem_layers = vae_n_layers if hasattr(tokenizer, 'n_layers') else vae_n_layers
    logger.info(f"Using {n_sem_layers} semantic ID layers")

    model = EncoderDecoderRetrievalModel(
        embedding_dim=decoder_embed_dim,
        attn_dim=attn_embed_dim,
        dropout=dropout_p,
        num_heads=attn_heads,
        n_layers=attn_layers,
        num_embeddings=vae_codebook_size,
        inference_verifier_fn=lambda x: tokenizer.exists_prefix(x),
        sem_id_dim=tokenizer.sem_ids_dim,
        max_pos=train_dataset.max_seq_len*tokenizer.sem_ids_dim,
        jagged_mode=model_jagged_mode,
        n_sem_layers=n_sem_layers,  # 传递语义ID层数
        use_interleaved_ids=use_interleaved_ids # 传递交错模式参数
    )
    logger.info(f"Model created with {attn_layers} attention layers, {attn_heads} heads")

    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    lr_scheduler = InverseSquareRootScheduler(
        optimizer=optimizer,
        warmup_steps=10000
    )
    
    start_iter = 0
    if pretrained_decoder_path is not None:
        logger.info(f"Loading pretrained decoder from {pretrained_decoder_path}")
        checkpoint = torch.load(pretrained_decoder_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        start_iter = checkpoint["iter"] + 1
        logger.info(f"Resuming from iteration {start_iter}")

    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )

    # 初始化 NDCGAccumulator 实例
    metrics_accumulator = TopKAccumulator(ks=[1, 5, 10])
    ndcg_accumulator = NDCGAccumulator(ks=[1, 5, 10])
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Device: {device}, Num Parameters: {num_params}")
    
    with tqdm(initial=start_iter, total=start_iter + iterations,
              disable=not accelerator.is_main_process) as pbar:
        for iter in range(iterations):
            model.train()
            total_loss = 0
            optimizer.zero_grad()
            for _ in range(gradient_accumulate_every):
                data = next_batch(train_dataloader, device)
                tokenized_data = tokenizer(data)

                with accelerator.autocast():
                    model_output = model(tokenized_data)
                    loss = model_output.loss / gradient_accumulate_every
                    
                    # 调试信息：检查损失是否有梯度
                    if iter == 0 and _ == 0:
                        logger.info(f"损失值: {loss.item()}")
                        logger.info(f"损失需要梯度: {loss.requires_grad}")
                        logger.info(f"模型参数总数: {sum(p.numel() for p in model.parameters())}")
                        logger.info(f"需要梯度的参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
                        
                    total_loss += loss
                
                if accelerator.is_main_process:
                    train_debug_metrics = compute_debug_metrics(tokenized_data, model_output)

                accelerator.backward(total_loss)
                assert model.sem_id_embedder.emb.weight.grad is not None

            pbar.set_description(f'loss: {total_loss.item():.4f}')

            accelerator.wait_for_everyone()

            optimizer.step()
            lr_scheduler.step()

            accelerator.wait_for_everyone()
            
            # 记录训练指标
            if accelerator.is_main_process:
                current_lr = optimizer.param_groups[0]["lr"]
                loss_value = total_loss.cpu().item()
                logger.info(f"Iteration {iter+1}: loss={loss_value:.4f}, lr={current_lr:.6f}")
                
                metrics_tracker.update(
                    iter+1,
                    loss=loss_value,
                    learning_rate=current_lr,
                    **train_debug_metrics
                )

            if (iter+1) % partial_eval_every == 0:
                model.eval()
                model.enable_generation = False
                eval_losses = []
                
                for batch in eval_dataloader:
                    data = batch_to(batch, device)
                    tokenized_data = tokenizer(data)

                    with torch.no_grad():
                        model_output_eval = model(tokenized_data)
                    
                    eval_losses.append(model_output_eval.loss.detach().cpu().item())
                    
                    if accelerator.is_main_process:
                        eval_debug_metrics = compute_debug_metrics(tokenized_data, model_output_eval, "eval")
                        # 将eval_loss添加到eval_debug_metrics中，而不是在update时重复传递
                        eval_debug_metrics["eval_loss"] = model_output_eval.loss.detach().cpu().item()
                
                if accelerator.is_main_process:
                    avg_eval_loss = np.mean(eval_losses)
                    logger.info(f"Evaluation at iteration {iter+1}: eval_loss={avg_eval_loss:.4f}")
                    
                    # 修复：检查eval_debug_metrics中是否已包含eval_loss，如果有则不再单独传递
                    if "eval_loss" in eval_debug_metrics:
                        metrics_tracker.update(iter+1, **eval_debug_metrics)
                    else:
                        metrics_tracker.update(iter+1, eval_loss=avg_eval_loss, **eval_debug_metrics)

            if (iter+1) % full_eval_every == 0:
                model.eval()
                model.enable_generation = True
                
                # 优化：使用临时变量存储原始配置，而不是直接修改全局配置
                # 创建上下文管理器类来临时禁用torch.compile
                class TempDisableDynamo:
                    def __init__(self):
                        pass
                    
                    def __enter__(self):
                        self.original_cache_size_limit = torch._dynamo.config.cache_size_limit
                        self.original_dynamic_shapes = torch._dynamo.config.dynamic_shapes
                        self.original_optimize_ddp = torch._dynamo.config.optimize_ddp
                        
                        # 临时修改配置
                        torch._dynamo.config.cache_size_limit = 0
                        torch._dynamo.config.dynamic_shapes = False
                        torch._dynamo.config.optimize_ddp = False
                    
                    def __exit__(self, exc_type, exc_val, exc_tb):
                        # 恢复原始配置
                        torch._dynamo.config.cache_size_limit = self.original_cache_size_limit
                        torch._dynamo.config.dynamic_shapes = self.original_dynamic_shapes
                        torch._dynamo.config.optimize_ddp = self.original_optimize_ddp
                
                # 在评估期间临时禁用动态编译
                with TempDisableDynamo():
                    # 在全面评估部分，需要同时使用两个累加器
                    with tqdm(eval_dataloader, desc=f'Eval {iter+1}', disable=not accelerator.is_main_process) as pbar_eval:
                            for batch_idx, batch in enumerate(pbar_eval): # 添加batch_idx用于选择样本
                                try:
                                    data = batch_to(batch, device)
                                    tokenized_data = tokenizer(data)
                
                                    generated = model.generate_next_sem_id(tokenized_data, top_k=True, temperature=1)
                                    actual, top_k = tokenized_data.sem_ids_fut, generated.sem_ids
                
                                    logger.info(f"实际形状: {actual.shape}, 预测形状: {top_k.shape}")
                                    
                                    # 检查拼接模式并更新标签部分
                                    if use_concatenated_ids and hasattr(data, 'tags_indices'):
                                        logger.info("修复拼接模式下的标签问题...")
                                        
                                        # 检查维度不匹配
                                        if actual.size(-1) != top_k.size(-1):
                                            logger.info(f"检测到维度不匹配: actual={actual.size()}, top_k={top_k.size()}")
                                            
                                            # 获取语义ID的层数
                                            num_semantic_layers = n_sem_layers
                                            
                                            # 如果生成的ID包含标签部分但实际ID没有
                                            if top_k.size(-1) > num_semantic_layers and actual.size(-1) == num_semantic_layers:
                                                logger.info("实际ID缺少标签部分，尝试添加...")
                                                
                                                # 检查数据中是否有标签索引
                                                if hasattr(data, 'tags_indices') and data.tags_indices is not None:
                                                    batch_size = actual.size(0)
                                                    
                                                    # 获取标签索引
                                                    tags_indices = data.tags_indices
                                                    logger.info(f"标签索引形状: {tags_indices.shape}")
                                                    
                                                    # 打印原始标签索引和标签类别数量
                                                    logger.info(f"标签类别数量: {tokenizer.tag_class_counts}")
                                                    if batch_size > 0:
                                                        sample_tags = tags_indices[0]
                                                        logger.info(f"样本原始标签索引: {sample_tags.tolist()}")
                                                    
                                                    # 创建新的实际ID张量，包含语义ID和标签ID
                                                    num_tag_layers = min(len(tokenizer.tag_class_counts), tags_indices.shape[1])
                                                    
                                                    # 收集当前的语义ID
                                                    semantic_ids = actual
                                                    
                                                    # 获取当前样本的标签ID
                                                    tag_ids_list = []
                                                    for i in range(num_tag_layers):
                                                        tag_ids = tags_indices[:, i].clone()
                                                        if tokenizer.tag_class_counts is not None and i < len(tokenizer.tag_class_counts):
                                                            special_value = tokenizer.tag_class_counts[i]
                                                            logger.info(f"第{i+1}层标签特殊值: {special_value}")
                                                            
                                                            # 创建替换前后的映射
                                                            orig_tags = tag_ids.clone()
                                                            tag_ids[tag_ids < 0] = special_value
                                                            
                                                            # 打印原始值和替换后的值（第一个样本）
                                                            if batch_size > 0:
                                                                logger.info(f"第{i+1}层标签原始值: {orig_tags[0].item()}")
                                                                logger.info(f"第{i+1}层标签替换后: {tag_ids[0].item()}")
                                                                
                                                        tag_ids_list.append(tag_ids.unsqueeze(1))
                                                    
                                                    # 拼接所有标签ID
                                                    tag_ids = torch.cat(tag_ids_list, dim=1)
                                                    logger.info(f"标签ID形状: {tag_ids.shape}")
                                                    
                                                    # 打印第一个样本的标签ID
                                                    if batch_size > 0:
                                                        logger.info(f"第一个样本的标签ID: {tag_ids[0].tolist()}")
                                                    
                                                    # 拼接语义ID和标签ID
                                                    # 注意：这里我们需要处理不同的维度情况
                                                    if semantic_ids.dim() == 3:  # [batch, k, semantic_dim]
                                                        # 复制标签以匹配候选数量
                                                        expanded_tag_ids = tag_ids.unsqueeze(1).expand(-1, semantic_ids.size(1), -1)
                                                        new_actual = torch.cat([semantic_ids, expanded_tag_ids], dim=2)
                                                        logger.info(f"3D拼接后形状: {new_actual.shape}")
                                                    elif semantic_ids.dim() == 2 and top_k.dim() == 3:  # [batch, semantic_dim] vs [batch, k, total_dim]
                                                        # 使用语义ID的前num_semantic_layers维度
                                                        new_actual = torch.cat([semantic_ids, tag_ids], dim=1)
                                                        logger.info(f"2D拼接后形状: {new_actual.shape}")
                                                        
                                                        # 修改指标计算，在计算时将top_k变为[batch, k, semantic_dim]
                                                        logger.info("指标计算使用共同维度...")
                                                        
                                                        # 打印第一个样本用于检查
                                                        if batch_size > 0:
                                                            logger.info(f"拼接后第一个样本: semantic_ids={semantic_ids[0].tolist()}, tag_ids={tag_ids[0].tolist()}")
                                                            logger.info(f"完整拼接ID: {new_actual[0].tolist()}")
                                                        
                                                        metrics_accumulator.accumulate(actual=new_actual, top_k=top_k)
                                                        ndcg_accumulator.accumulate(actual=new_actual, top_k=top_k)
                                                        continue  # 跳过后面的指标计算
                                                    else:
                                                        # 简单拼接
                                                        new_actual = torch.cat([semantic_ids, tag_ids], dim=1)
                                                        logger.info(f"其他情况拼接后形状: {new_actual.shape}")
                                                    
                                                    logger.info(f"拼接后的实际ID形状: {new_actual.shape}")
                                                    actual = new_actual
                                    
                                    # 处理拼接模式下的维度不匹配问题
                                    if use_concatenated_ids:
                                        # 确保actual和top_k维度一致
                                        if actual.size(-1) != top_k.size(-1):
                                            logger.info(f"检测到维度不匹配: actual={actual.size()}, top_k={top_k.size()}")
                                            
                                            # 获取两者共同的维度
                                            common_dims = min(actual.size(-1), top_k.size(-1))
                                            
                                            # 使用共同维度进行指标计算 - 通常是语义ID部分
                                            metrics_accumulator.accumulate(actual=actual[..., :common_dims], top_k=top_k[..., :common_dims])
                                            ndcg_accumulator.accumulate(actual=actual[..., :common_dims], top_k=top_k[..., :common_dims])
                                        else:
                                            metrics_accumulator.accumulate(actual=actual, top_k=top_k)
                                            ndcg_accumulator.accumulate(actual=actual, top_k=top_k)
                                    else:
                                        metrics_accumulator.accumulate(actual=actual, top_k=top_k)
                                        ndcg_accumulator.accumulate(actual=actual, top_k=top_k)

                                    # 打印随机样本的预测和真实值 (仅在主进程且为评估的第一个批次)
                                    if accelerator.is_main_process and batch_idx == 0:
                                        num_samples_to_print = 3 # 可以调整打印的样本数量
                                        # 确保不超出批次大小
                                        actual_samples_to_print = min(num_samples_to_print, actual.size(0)) 
                                        
                                        # 随机选择样本索引
                                        sample_indices = torch.randperm(actual.size(0))[:actual_samples_to_print]

                                        logger.info(f"--- Sample Predictions at Iteration {iter+1} (Batch {batch_idx}) ---")
                                        # 打印tokenized_data的形状信息
                                        logger.info(f"tokenized_data.sem_ids_fut形状: {tokenized_data.sem_ids_fut.shape}")
                                        logger.info(f"tokenized_data.sem_ids形状: {tokenized_data.sem_ids.shape}")
                                        logger.info(f"实际形状: {actual.shape}, 预测形状: {top_k.shape}")
                                        
                                        # 打印原始数据的标签信息
                                        if hasattr(data, 'tags_indices') and data.tags_indices is not None:
                                            logger.info(f"数据标签索引形状: {data.tags_indices.shape}")
                                            if hasattr(tokenizer, 'tag_class_counts'):
                                                logger.info(f"标签类别数量: {tokenizer.tag_class_counts}")
                                                
                                                # 打印标签层次结构说明
                                                logger.info("标签层次结构说明:")
                                                for i, count in enumerate(tokenizer.tag_class_counts):
                                                    # 确定标签类型，根据层次和类别数量
                                                    if i == 0 and count <= 10:
                                                        tag_type = "类别"
                                                    elif i == 1 and count <= 50:
                                                        tag_type = "子类别"
                                                    else:
                                                        tag_type = "具体标签"
                                                    logger.info(f"  第{i+1}层 ({tag_type}): {count}个类别, 特殊值={count} (用于无效标签)")
                                        
                                        for i in range(actual_samples_to_print):
                                            sample_idx = sample_indices[i].item()
                                            actual_ids_sample = actual[sample_idx]
                                            predicted_ids_sample = top_k[sample_idx]

                                            logger.info(f"  Sample {i+1} (Original Index: {sample_idx}):")
                                            
                                            # 打印原始标签索引
                                            if hasattr(data, 'tags_indices') and data.tags_indices is not None:
                                                orig_tags = data.tags_indices[sample_idx]
                                                logger.info(f"    原始标签索引: {orig_tags.tolist()}")
                                                
                                                # 打印标签映射后的结果
                                                if hasattr(tokenizer, 'tag_class_counts'):
                                                    mapped_tags = []
                                                    for j, tag_idx in enumerate(orig_tags.tolist()):
                                                        if j < len(tokenizer.tag_class_counts) and tag_idx >= 0:
                                                            mapped_tags.append(tag_idx)
                                                        elif j < len(tokenizer.tag_class_counts):
                                                            # 无效标签使用特殊值
                                                            mapped_tags.append(tokenizer.tag_class_counts[j])
                                                        else:
                                                            # 超出范围的标签
                                                            mapped_tags.append(-1)
                                                    logger.info(f"    映射后标签ID: {mapped_tags}")
                                            
                                            # 获取当前序列的信息（如果可用）
                                            if hasattr(tokenized_data, 'sem_ids') and tokenized_data.sem_ids is not None:
                                                current_seq_ids = tokenized_data.sem_ids[sample_idx]
                                                
                                                # 获取序列长度和语义ID维度
                                                seq_len = tokenized_data.seq_mask[sample_idx].sum().item() if hasattr(tokenized_data, 'seq_mask') else current_seq_ids.shape[0]
                                                # n_sem_layers = n_sem_layers  # 从外部获取 (already defined in outer scope)
                                                
                                                # 提取当前序列的ID
                                                # 假设ID格式为 [pos1_sem1, pos1_sem2, ..., pos1_tag1, pos1_tag2, ..., pos2_sem1, ...]
                                                # 我们需要提取第一个位置的所有语义ID和标签ID
                                                if seq_len > 0 and n_sem_layers > 0:
                                                    # 计算每个位置的总ID数
                                                    ids_per_pos = current_seq_ids.shape[0] // seq_len
                                                    
                                                    # 确保位置数量合理
                                                    if ids_per_pos > 0:
                                                        # 提取第一个位置的ID
                                                        first_pos_ids = current_seq_ids[:ids_per_pos]
                                                        
                                                        # 区分语义ID和标签ID
                                                        if ids_per_pos >= n_sem_layers:
                                                            current_sem_ids = first_pos_ids[:n_sem_layers]
                                                            if ids_per_pos > n_sem_layers:
                                                                current_tag_ids = first_pos_ids[n_sem_layers:]
                                                                logger.info(f"    当前序列语义ID: {current_sem_ids.tolist()}")
                                                                logger.info(f"    当前序列标签ID: {current_tag_ids.tolist()}")
                                                            else:
                                                                logger.info(f"    当前序列语义ID: {current_sem_ids.tolist()}")
                                                                logger.info(f"    当前序列标签ID: []")
                                            
                                            if use_concatenated_ids:
                                                # 如果使用拼接ID，分别展示语义ID和标签ID
                                                # 假设语义ID在前，标签ID在后
                                                num_semantic_layers = n_sem_layers # 从外部获取
                                                
                                                # 打印维度信息
                                                logger.info(f"    实际ID维度: {actual_ids_sample.shape}")
                                                logger.info(f"    预测ID维度: {predicted_ids_sample.shape}")
                                                
                                                # 处理未来序列的实际ID
                                                if actual_ids_sample.dim() == 1:
                                                    # 一维张量，可能是由于只有语义ID没有标签ID
                                                    total_dim = actual_ids_sample.shape[0]
                                                    
                                                    # 检查维度是否足够分离语义ID和标签ID
                                                    if total_dim >= num_semantic_layers:
                                                        actual_semantic = actual_ids_sample[:num_semantic_layers]
                                                        # 检查是否有标签部分
                                                        if total_dim > num_semantic_layers:
                                                            actual_tags = actual_ids_sample[num_semantic_layers:]
                                                        else:
                                                            # 仅有语义ID
                                                            actual_tags = torch.tensor([], device=actual_ids_sample.device)
                                                    else:
                                                        logger.warning(f"    警告: 实际ID维度({total_dim})小于语义层数({num_semantic_layers})")
                                                        actual_semantic = actual_ids_sample
                                                        actual_tags = torch.tensor([], device=actual_ids_sample.device)
                                                else:
                                                    # 高维张量，可能是批次内部结构
                                                    logger.warning(f"    警告: 实际ID是高维张量，结构可能不是预期的")
                                                    actual_semantic = actual_ids_sample
                                                    actual_tags = torch.tensor([], device=actual_ids_sample.device)
                                                
                                                # 处理预测ID，区分语义ID和标签ID
                                                if predicted_ids_sample.dim() > 1:
                                                    # 预测ID可能有多个候选，取第一个
                                                    top1_prediction = predicted_ids_sample[0]
                                                    
                                                    # 检查维度是否足够分离语义ID和标签ID
                                                    pred_total_dim = top1_prediction.shape[0]
                                                    if pred_total_dim >= num_semantic_layers:
                                                        predicted_semantic = top1_prediction[:num_semantic_layers]
                                                        if pred_total_dim > num_semantic_layers:
                                                            predicted_tags = top1_prediction[num_semantic_layers:]
                                                        else:
                                                            predicted_tags = torch.tensor([], device=top1_prediction.device)
                                                    else:
                                                        logger.warning(f"    警告: 预测ID维度({pred_total_dim})小于语义层数({num_semantic_layers})")
                                                        predicted_semantic = top1_prediction
                                                        predicted_tags = torch.tensor([], device=top1_prediction.device)
                                                else:
                                                    # 一维预测，直接分割
                                                    pred_total_dim = predicted_ids_sample.shape[0]
                                                    if pred_total_dim >= num_semantic_layers:
                                                        predicted_semantic = predicted_ids_sample[:num_semantic_layers]
                                                        if pred_total_dim > num_semantic_layers:
                                                            predicted_tags = predicted_ids_sample[num_semantic_layers:]
                                                        else:
                                                            predicted_tags = torch.tensor([], device=predicted_ids_sample.device)
                                                    else:
                                                        logger.warning(f"    警告: 预测ID维度({pred_total_dim})小于语义层数({num_semantic_layers})")
                                                        predicted_semantic = predicted_ids_sample
                                                        predicted_tags = torch.tensor([], device=predicted_ids_sample.device)
                                                
                                                # 输出语义ID和标签ID
                                                logger.info(f"    未来语义ID (实际): {actual_semantic.tolist()}")
                                                logger.info(f"    未来标签ID (实际): {actual_tags.tolist()}")
                                                
                                                # 处理多维预测
                                                if predicted_ids_sample.dim() > 1:
                                                    # 打印所有Top-K候选项的语义ID部分
                                                    semantic_predictions = [pred[:num_semantic_layers].tolist() for pred in predicted_ids_sample[:5]]  # 只显示前5个
                                                    logger.info(f"    Top-5 未来语义ID (预测): {semantic_predictions}")
                                                    
                                                    # 如果有足够的维度，打印标签ID部分
                                                    if predicted_ids_sample.shape[1] > num_semantic_layers:
                                                        tag_predictions = [pred[num_semantic_layers:].tolist() for pred in predicted_ids_sample[:5]]
                                                        logger.info(f"    Top-5 未来标签ID (预测): {tag_predictions}")
                                                    else:
                                                        logger.info(f"    未来标签ID (预测): [] (预测ID维度不足)")
                                                else:
                                                    # 一维预测
                                                    logger.info(f"    未来语义ID (预测): {predicted_semantic.tolist()}")
                                                    logger.info(f"    未来标签ID (预测): {predicted_tags.tolist()}")
                                            else:
                                                logger.info(f"    实际ID: {actual_ids_sample.tolist()}")
                                                logger.info(f"    预测ID: {predicted_ids_sample.tolist()}")
                                        logger.info(f"--- End Sample Predictions ---")
                                except Exception as e:
                                    logger.error(f"评估过程中出错: {str(e)}")
                                    import traceback
                                    logger.error(traceback.format_exc())
                                    continue
                
                eval_metrics = metrics_accumulator.reduce()
                ndcg_metrics = ndcg_accumulator.reduce()  # 获取NDCG指标
                
                if accelerator.is_main_process:
                    logger.info(f"Full evaluation at iteration {iter+1}:")
                    for metric_name, metric_value in eval_metrics.items():
                        logger.info(f"  {metric_name}: {metric_value:.4f}")
                    
                    # 记录NDCG指标
                    for metric_name, metric_value in ndcg_metrics.items():
                        logger.info(f"  {metric_name}: {metric_value:.4f}")
                    
                    # 合并两种指标进行更新
                    combined_metrics = {**eval_metrics, **ndcg_metrics}
                    metrics_tracker.update(iter+1, **combined_metrics)
                
                metrics_accumulator.reset()
                ndcg_accumulator.reset()  # 重置NDCG累加器

            if accelerator.is_main_process:
                if (iter+1) % save_model_every == 0 or iter+1 == iterations:
                    state = {
                        "iter": iter,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict()
                    }

                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)

                    checkpoint_path = os.path.join(save_dir_root, f"checkpoint_{iter}.pt")
                    torch.save(state, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            pbar.update(1)
    
    # 训练结束后绘制并保存所有指标图像
    if accelerator.is_main_process:
        plots_dir = os.path.join(log_dir, "plots")
        logger.info(f"Training completed. Saving metric plots to {plots_dir}")
        metrics_tracker.plot_and_save(plots_dir)
        logger.info("All metric plots saved successfully")


if __name__ == "__main__":
    parse_config()
    train()
