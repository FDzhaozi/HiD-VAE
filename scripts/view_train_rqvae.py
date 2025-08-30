import gin
import os
# Add the main working directory to facilitate package imports
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


# Suppress warning messages
warnings.filterwarnings('ignore')
logging.getLogger("torch._dynamo.convert_frame").setLevel(logging.ERROR)
logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)

# Configure torch._dynamo warnings
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
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger("rqvae_view")
    
    # Print training parameters
    logger.info("=== Training Parameters ===")
    params = locals()
    for key, value in params.items():
        if key != 'logger':
            logger.info(f"  {key}: {value}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset}, split: {dataset_split}")
    train_dataset = ItemData(
        root=dataset_folder, 
        dataset=dataset, 
        force_process=force_dataset_process, 
        train_test_split="train" if do_eval else "all", 
        split=dataset_split
    )
    logger.info(f"Dataset size: {len(train_dataset)}")
    
    # Create data loader
    train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size, False)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None, collate_fn=lambda batch: batch)
    
    # Create model
    logger.info("=== Creating Model ===")
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
    
    # Print model structure
    logger.info("=== Model Structure ===")
    logger.info(f"Input dimension: {vae_input_dim}")
    logger.info(f"Embedding dimension: {vae_embed_dim}")
    logger.info(f"Hidden layer dimensions: {vae_hidden_dims}")
    logger.info(f"Codebook size: {vae_codebook_size}")
    logger.info(f"Number of layers: {vae_n_layers}")
    logger.info(f"Number of categorical features: {vae_n_cat_feats}")
    
    # Print encoder structure
    logger.info("=== Encoder Structure ===")
    for name, param in model.encoder.named_parameters():
        logger.info(f"{name}: {param.shape}")
    
    # Print quantizer structure
    logger.info("=== Quantizer Structure ===")
    for i, layer in enumerate(model.layers):
        logger.info(f"Quantizer layer {i}:")
        for name, param in layer.named_parameters():
            logger.info(f"  {name}: {param.shape}")
    
    # Print decoder structure
    logger.info("=== Decoder Structure ===")
    for name, param in model.decoder.named_parameters():
        logger.info(f"{name}: {param.shape}")
    
    # Create optimizer
    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Load pretrained model (if any)
    if pretrained_rqvae_path is not None:
        logger.info(f"Loading pretrained model: {pretrained_rqvae_path}")
        model.load_pretrained(pretrained_rqvae_path)
    
    # Create tokenizer
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
    
    # Run one forward pass
    logger.info("=== Running Forward Pass ===")
    model.eval()
    
    # Get a batch of data
    train_iter = iter(train_dataloader)  # Create an iterator
    batch = next(train_iter)  # Get a batch of data
    data = batch_to(batch, device)  # Move data to the device
    
    # Print input data shape
    logger.info(f"Input data shape: {data.x.shape}")
    # Print the shapes of multiple fields in the input data
    logger.info("=== Input Data Field Shapes ===")
    for field_name in data._fields:
        field_value = getattr(data, field_name)
        if isinstance(field_value, torch.Tensor):
            logger.info(f"  {field_name}: {field_value.shape}")
        elif field_value is not None:
            logger.info(f"  {field_name}: {type(field_value)}")
    
    # Run forward pass
    with torch.no_grad():
        t = 0.2  # Gumbel temperature
        model_output = model(data, gumbel_t=t)
    
    # Print model output
    logger.info("=== Model Output ===")
    logger.info(f"Total loss: {model_output.loss.item():.4f}")
    logger.info(f"Reconstruction loss: {model_output.reconstruction_loss.item():.4f}")
    logger.info(f"RQ-VAE loss: {model_output.rqvae_loss.item():.4f}")
    # logger.info(f"Embedding norm shape: {model_output.embs_norm}")
    logger.info(f"Embedding norm shape: {model_output.embs_norm.shape}")
    logger.info(f"Proportion of unique IDs: {model_output.p_unique_ids.item():.4f}")
    
    # Get semantic IDs
    logger.info("=== Getting Semantic IDs ===")
    quantized = model.get_semantic_ids(data.x, gumbel_t=t)
    logger.info(f"Embeddings shape: {quantized.embeddings.shape}")
    logger.info(f"Residuals shape: {quantized.residuals.shape}")
    logger.info(f"Semantic IDs shape: {quantized.sem_ids.shape}")
    
    # Fix: Handle the case where quantization loss might be a tensor
    if quantized.quantize_loss.numel() > 1:
        # If it's a multi-element tensor, calculate the mean
        logger.info(f"Quantization loss (mean): {quantized.quantize_loss.mean().item():.4f}")
        logger.info(f"Quantization loss shape: {quantized.quantize_loss.shape}")
    else:
        # If it's a scalar, use .item() directly
        logger.info(f"Quantization loss: {quantized.quantize_loss.item():.4f}")
    
    # Print the semantic ID distribution for each layer
    logger.info("=== Semantic ID Distribution ===")
    for i in range(vae_n_layers):
        layer_ids = quantized.sem_ids[i]
        unique_ids, counts = torch.unique(layer_ids, return_counts=True)
        usage = len(unique_ids) / vae_codebook_size
        logger.info(f"Layer {i} codebook usage: {usage:.4f} ({len(unique_ids)}/{vae_codebook_size})")
        
        # Print the top 10 most used IDs
        sorted_indices = torch.argsort(counts, descending=True)
        top_ids = unique_ids[sorted_indices[:10]]
        top_counts = counts[sorted_indices[:10]]
        logger.info(f"Layer {i} top 10 most used IDs: {top_ids.tolist()}")
        logger.info(f"Layer {i} top 10 most used ID counts: {top_counts.tolist()}")
    
    # Calculate reconstruction
    logger.info("=== Reconstruction Result ===")
    x_hat = model.decode(quantized.embeddings.sum(axis=-1))
    mse = torch.nn.functional.mse_loss(x_hat, data.x)
    logger.info(f"Reconstruction MSE: {mse.item():.4f}")
    
    logger.info("Inspection complete")

if __name__ == "__main__":
    parse_config()
    train()
