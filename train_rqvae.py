import gin
import os
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
    use_kmeans_init=True,
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
    log_dir = os.path.join(save_dir_root, "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建图表保存目录
    plots_dir = os.path.join(log_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # 配置日志记录器
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[file_handler, console_handler]
    )
    
    logger = logging.getLogger("rqvae_training")
    
    # 初始化用于绘图的数据收集器
    plot_data = {
        'iterations': [],
        'total_loss': [],
        'reconstruction_loss': [],
        'rqvae_loss': [],
        'emb_norms': [[] for _ in range(vae_n_layers)],
        'codebook_usage': [[] for _ in range(vae_n_layers)],
        'eval_iterations': [],
        'eval_total_loss': [],
        'eval_reconstruction_loss': [],
        'eval_rqvae_loss': [],
        'rqvae_entropy': [],
        'max_id_duplicates': []
    }
    
    # 首先创建 accelerator 实例
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )
    
    # 记录训练参数
    if accelerator.is_main_process:
        params = locals()
        logger.info("训练参数:")
        for key, value in params.items():
            if key != 'logger':
                logger.info(f"  {key}: {value}")

    device = accelerator.device

    train_dataset = ItemData(root=dataset_folder, dataset=dataset, force_process=force_dataset_process, train_test_split="train" if do_eval else "all", split=dataset_split)
    train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size, False)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None, collate_fn=lambda batch: batch)
    train_dataloader = cycle(train_dataloader)

    if do_eval:
        eval_dataset = ItemData(root=dataset_folder, dataset=dataset, force_process=False, train_test_split="eval", split=dataset_split)
        eval_sampler = BatchSampler(RandomSampler(eval_dataset), batch_size, False)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=None, collate_fn=lambda batch: batch)

    index_dataset = ItemData(root=dataset_folder, dataset=dataset, force_process=False, train_test_split="all", split=dataset_split) if do_eval else train_dataset
    
    train_dataloader = accelerator.prepare(train_dataloader)
    # TODO: Investigate bug with prepare eval_dataloader

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
        logger.info(f" len(train_dataset): {len(train_dataset)}")
        logger.info(f" len(eval_dataset): {len(eval_dataset)}")

    start_iter = 0
    if pretrained_rqvae_path is not None:
        model.load_pretrained(pretrained_rqvae_path)
        state = torch.load(pretrained_rqvae_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_iter = state["iter"]+1
        if accelerator.is_main_process:
            logger.info(f"加载预训练模型: {pretrained_rqvae_path}, 从迭代 {start_iter} 开始")

    model, optimizer = accelerator.prepare(
        model, optimizer
    )

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

    with tqdm(initial=start_iter, total=start_iter+iterations,
              disable=not accelerator.is_main_process) as pbar:
        losses = [[], [], []]
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

                with accelerator.autocast():
                    model_output = model(data, gumbel_t=t)
                    loss = model_output.loss
                    loss = loss / gradient_accumulate_every
                    total_loss += loss

            accelerator.backward(total_loss)

            losses[0].append(total_loss.cpu().item())
            losses[1].append(model_output.reconstruction_loss.cpu().item())
            losses[2].append(model_output.rqvae_loss.cpu().item())
            losses[0] = losses[0][-1000:]  # 滑动平均损失 每次迭代后，每个子列表只保留​​最近的1000条损失记录​​，旧数据被丢弃。
            losses[1] = losses[1][-1000:]
            losses[2] = losses[2][-1000:]
            if iter % 100 == 0:
                print_loss = np.mean(losses[0])
                print_rec_loss = np.mean(losses[1])
                print_vae_loss = np.mean(losses[2])

            pbar.set_description(f'loss: {print_loss:.4f}, rl: {print_rec_loss:.4f}, vl: {print_vae_loss:.4f}')

            accelerator.wait_for_everyone()

            optimizer.step()
            
            accelerator.wait_for_everyone()

            id_diversity_log = {}
            if accelerator.is_main_process:
                # 使用logging替代wandb记录训练信息
                if iter % 100 == 0:  # 每100次迭代记录一次详细日志
                    # 计算嵌入范数平均值
                    emb_norms_avg = model_output.embs_norm.mean(axis=0)
                    emb_norms_str = ", ".join([f"layer_{i}: {emb_norms_avg[i].cpu().item():.4f}" for i in range(vae_n_layers)])
                    
                    # 收集绘图数据
                    plot_data['iterations'].append(iter)
                    plot_data['total_loss'].append(total_loss.cpu().item())
                    plot_data['reconstruction_loss'].append(model_output.reconstruction_loss.cpu().item())
                    plot_data['rqvae_loss'].append(model_output.rqvae_loss.cpu().item())
                    
                    for i in range(vae_n_layers):
                        plot_data['emb_norms'][i].append(emb_norms_avg[i].cpu().item())
                    
                    logger.info(f"迭代 {iter} - 损失: {total_loss.cpu().item():.4f}, "
                               f"重构损失: {model_output.reconstruction_loss.cpu().item():.4f}, "
                               f"RQVAE损失: {model_output.rqvae_loss.cpu().item():.4f}, "
                               f"温度: {t:.4f}, "
                               f"唯一ID比例: {model_output.p_unique_ids.cpu().item():.4f}, "
                               f"嵌入范数: {emb_norms_str}")
                    

            if do_eval and ((iter+1) % eval_every == 0 or iter+1 == iterations):
                model.eval()
                with tqdm(eval_dataloader, desc=f'Eval {iter+1}', disable=not accelerator.is_main_process) as pbar_eval:
                    eval_losses = [[], [], []]
                    for batch in pbar_eval:
                        data = batch_to(batch, device)
                        with torch.no_grad():
                            eval_model_output = model(data, gumbel_t=t)

                        eval_losses[0].append(eval_model_output.loss.cpu().item())
                        eval_losses[1].append(eval_model_output.reconstruction_loss.cpu().item())
                        eval_losses[2].append(eval_model_output.rqvae_loss.cpu().item())
                    
                    eval_losses = np.array(eval_losses).mean(axis=-1)
                    id_diversity_log["eval_total_loss"] = eval_losses[0]
                    id_diversity_log["eval_reconstruction_loss"] = eval_losses[1]
                    id_diversity_log["eval_rqvae_loss"] = eval_losses[2]
                    
                    if accelerator.is_main_process:
                        # 收集评估数据用于绘图
                        plot_data['eval_iterations'].append(iter+1)
                        plot_data['eval_total_loss'].append(eval_losses[0])
                        plot_data['eval_reconstruction_loss'].append(eval_losses[1])
                        plot_data['eval_rqvae_loss'].append(eval_losses[2])
                        
                        logger.info(f"评估 {iter+1} - 总损失: {eval_losses[0]:.4f}, "
                                   f"重构损失: {eval_losses[1]:.4f}, "
                                   f"RQVAE损失: {eval_losses[2]:.4f}")
                    
            if accelerator.is_main_process:
                if (iter+1) % save_model_every == 0 or iter+1 == iterations:
                    state = {
                        "iter": iter,
                        "model": model.state_dict(),
                        "model_config": model.config,
                        "optimizer": optimizer.state_dict()
                    }

                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)

                    save_path = save_dir_root + f"checkpoint_{iter}.pt"
                    torch.save(state, save_path)
                    logger.info(f"保存模型检查点到: {save_path}")
                    
                if (iter+1) % eval_every == 0 or iter+1 == iterations:
                    tokenizer.reset()
                    model.eval()

                    corpus_ids = tokenizer.precompute_corpus_ids(index_dataset)
                    max_duplicates = corpus_ids[:,-1].max() / corpus_ids.shape[0]
                    
                    _, counts = torch.unique(corpus_ids[:,:-1], dim=0, return_counts=True)
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

                    plot_data['rqvae_entropy'].append(rqvae_entropy.cpu().item())
                    plot_data['max_id_duplicates'].append(max_duplicates.cpu().item())
                    
                    logger.info(f"ID多样性 {iter+1} - "
                               f"RQVAE熵: {rqvae_entropy.cpu().item():.4f}, "
                               f"最大ID重复: {max_duplicates.cpu().item():.4f}, "
                               f"码本使用率: {', '.join(codebook_usage_info)}")

            pbar.update(1)
    
    # 训练结束时绘制最终图表
    if accelerator.is_main_process:
        logger.info("训练完成，正在生成训练过程图表...")
        # 绘制所有图表
        plot_all_metrics(plot_data, plots_dir, vae_n_layers)
        logger.info("所有图表已保存到 {}".format(plots_dir))


# 修改绘图函数，只在训练结束时生成整体图表
def plot_all_metrics(plot_data, plots_dir, n_layers):
    """绘制所有指标的整体训练过程图表"""
    
    # 1. 绘制总损失图 (Total Loss)
    plt.figure(figsize=(12, 8))
    plt.plot(plot_data['iterations'], plot_data['total_loss'], label='Training Total Loss')
    
    if plot_data['eval_iterations']:
        plt.plot(plot_data['eval_iterations'], plot_data['eval_total_loss'], 'o-', label='Evaluation Total Loss')
    
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "total_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 绘制重构损失图 (Reconstruction Loss)
    plt.figure(figsize=(12, 8))
    plt.plot(plot_data['iterations'], plot_data['reconstruction_loss'], label='Training Reconstruction Loss')
    
    if plot_data['eval_iterations']:
        plt.plot(plot_data['eval_iterations'], plot_data['eval_reconstruction_loss'], 'o-', label='Evaluation Reconstruction Loss')
    
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Reconstruction Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "reconstruction_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 绘制RQVAE损失图 (RQVAE Loss)
    plt.figure(figsize=(12, 8))
    plt.plot(plot_data['iterations'], plot_data['rqvae_loss'], label='Training RQVAE Loss')
    
    if plot_data['eval_iterations']:
        plt.plot(plot_data['eval_iterations'], plot_data['eval_rqvae_loss'], 'o-', label='Evaluation RQVAE Loss')
    
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('RQVAE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "rqvae_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 绘制嵌入范数图
    plt.figure(figsize=(12, 8))
    for i in range(n_layers):
        plt.plot(plot_data['iterations'], plot_data['emb_norms'][i], label=f'Layer {i}')
    
    plt.xlabel('Iterations')
    plt.ylabel('Embedding Norm')
    plt.title('Embedding Norms by Layer')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "embedding_norms.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 绘制码本使用率图
    if plot_data['codebook_usage'][0]:  # 如果有数据
        plt.figure(figsize=(12, 8))
        for i in range(n_layers):
            plt.plot(plot_data['eval_iterations'], plot_data['codebook_usage'][i], 'o-', label=f'Layer {i}')
        
        plt.xlabel('Iterations')
        plt.ylabel('Codebook Usage Rate')
        plt.title('Codebook Usage Rate by Layer')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "codebook_usage.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. 绘制RQVAE熵图
    if plot_data['rqvae_entropy']:  # 如果有数据
        plt.figure(figsize=(12, 8))
        plt.plot(plot_data['eval_iterations'], plot_data['rqvae_entropy'], 'o-', label='RQVAE Entropy')
        plt.xlabel('Iterations')
        plt.ylabel('Entropy')
        plt.title('RQVAE Entropy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "rqvae_entropy.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 7. 绘制最大ID重复图
    if plot_data['max_id_duplicates']:  # 如果有数据
        plt.figure(figsize=(12, 8))
        plt.plot(plot_data['eval_iterations'], plot_data['max_id_duplicates'], 'o-', label='Max ID Duplicates')
        plt.xlabel('Iterations')
        plt.ylabel('Duplicate Rate')
        plt.title('Maximum ID Duplicates')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "max_id_duplicates.png"), dpi=300, bbox_inches='tight')
        plt.close()



if __name__ == "__main__":
    parse_config()
    train()