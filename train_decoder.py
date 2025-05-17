import argparse
import os
import gin
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

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

# 禁用PyTorch 2.0编译功能，避免dynamo相关警告
torch._dynamo.config.suppress_errors = True
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
    partial_eval_every=1000,
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
    use_dedup_dim=False
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
    
    if dataset != RecDataset.AMAZON:
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
        tokenizer = HSemanticIdTokenizer(
            input_dim=vae_input_dim,
            hidden_dims=vae_hidden_dims,
            output_dim=vae_embed_dim,
            codebook_size=vae_codebook_size,
            n_layers=vae_n_layers,
            n_cat_feats=vae_n_cat_feats,
            hrqvae_weights_path=pretrained_rqvae_path,
            hrqvae_codebook_normalize=vae_codebook_normalize,
            hrqvae_sim_vq=vae_sim_vq,
            tag_alignment_weight=tag_alignment_weight,
            tag_prediction_weight=tag_prediction_weight,
            tag_class_counts=tag_class_counts,
            tag_embed_dim=tag_embed_dim,
            use_dedup_dim=use_dedup_dim
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
    
    tokenizer = accelerator.prepare(tokenizer)
    tokenizer.precompute_corpus_ids(item_dataset)
    logger.info("Tokenizer prepared and corpus IDs precomputed")
    
    if push_vae_to_hf:
        login()
        tokenizer.rq_vae.push_to_hub(vae_hf_model_name)
        logger.info(f"Pushed VAE to HuggingFace: {vae_hf_model_name}")

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
        jagged_mode=model_jagged_mode
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
                # 在全面评估部分，需要同时使用两个累加器
                with tqdm(eval_dataloader, desc=f'Eval {iter+1}', disable=not accelerator.is_main_process) as pbar_eval:
                    for batch in pbar_eval:
                        data = batch_to(batch, device)
                        tokenized_data = tokenizer(data)
                
                        generated = model.generate_next_sem_id(tokenized_data, top_k=True, temperature=1)
                        actual, top_k = tokenized_data.sem_ids_fut, generated.sem_ids
                
                        metrics_accumulator.accumulate(actual=actual, top_k=top_k)
                        ndcg_accumulator.accumulate(actual=actual, top_k=top_k)  # 添加NDCG计算
                
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