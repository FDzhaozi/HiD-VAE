import gin
import os
# Add the main working path to facilitate package import
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
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger("hrqvae_view")
    
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
    
    # Check if the dataset contains tag information
    has_tags = getattr(train_dataset, 'has_tags', False)
    if not has_tags:
        logger.warning("No tag information in the dataset, disabling tag alignment and prediction features")
        tag_alignment_weight = 0.0
        tag_prediction_weight = 0.0
    else:
        logger.info("Dataset contains tag information")
        
        # Ensure only the number of tag layers matching vae_n_layers is used
        # Check the shape of the tag data
        sample_data = train_dataset[0]
        if hasattr(sample_data, 'tags_emb') and sample_data.tags_emb is not None:
            logger.info(f"sample_data.tags_emb.shape = {sample_data.tags_emb.shape}")
            actual_tag_layers = sample_data.tags_emb.shape[1]
            logger.info(f"Number of tag layers in dataset: {actual_tag_layers}")
            
            if actual_tag_layers != vae_n_layers:
                logger.warning(f"Number of tag layers ({actual_tag_layers}) does not match the number of model layers ({vae_n_layers})")
                
                # Operate directly on the entire dataset
                if actual_tag_layers > vae_n_layers:
                    logger.warning(f"Trimming dataset tags, keeping only the first {vae_n_layers} layers")
                    # Trim tag embeddings
                    if hasattr(train_dataset, 'tags_emb') and train_dataset.tags_emb is not None:
                        train_dataset.tags_emb = train_dataset.tags_emb[:, :vae_n_layers, :]
                    
                    # Trim tag indices
                    if hasattr(train_dataset, 'tags_indices') and train_dataset.tags_indices is not None:
                        train_dataset.tags_indices = train_dataset.tags_indices[:, :vae_n_layers]
                    
                    logger.info(f"Shape after trimming train_dataset.tags_emb = {train_dataset.tags_emb.shape if hasattr(train_dataset, 'tags_emb') and train_dataset.tags_emb is not None else 'None'}")
                    logger.info(f"Shape after trimming train_dataset.tags_indices = {train_dataset.tags_indices.shape if hasattr(train_dataset, 'tags_indices') and train_dataset.tags_indices is not None else 'None'}")
                else:
                    logger.warning(f"Number of model layers ({vae_n_layers}) is greater than the number of tag layers ({actual_tag_layers}), will pad dataset tags")
                    # Pad tag embeddings
                    if hasattr(train_dataset, 'tags_emb') and train_dataset.tags_emb is not None:
                        tag_embed_shape = train_dataset.tags_emb.shape
                        padded_tags_emb = torch.zeros((tag_embed_shape[0], vae_n_layers, tag_embed_shape[2]), 
                                                      dtype=train_dataset.tags_emb.dtype)
                        padded_tags_emb[:, :actual_tag_layers, :] = train_dataset.tags_emb
                        train_dataset.tags_emb = padded_tags_emb
                    
                    # Pad tag indices
                    if hasattr(train_dataset, 'tags_indices') and train_dataset.tags_indices is not None:
                        tag_indices_shape = train_dataset.tags_indices.shape
                        padded_tags_indices = torch.ones((tag_indices_shape[0], vae_n_layers), 
                                                         dtype=train_dataset.tags_indices.dtype) * -1
                        padded_tags_indices[:, :actual_tag_layers] = train_dataset.tags_indices
                        train_dataset.tags_indices = padded_tags_indices
                    
                    logger.info(f"Shape after padding train_dataset.tags_emb = {train_dataset.tags_emb.shape if hasattr(train_dataset, 'tags_emb') and train_dataset.tags_emb is not None else 'None'}")
                    logger.info(f"Shape after padding train_dataset.tags_indices = {train_dataset.tags_indices.shape if hasattr(train_dataset, 'tags_indices') and train_dataset.tags_indices is not None else 'None'}")
    
    
    
    logger.info(f"Final number of tag classes used: {tag_class_counts}")
    
    # Ensure the number of tag classes matches the number of layers
    assert len(tag_class_counts) == vae_n_layers, f"Number of tag classes {len(tag_class_counts)} does not match number of layers {vae_n_layers}"
    
    # Create data loader
    train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size, False)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None, collate_fn=lambda batch: batch)
    
    # Create model
    logger.info("=== Creating Model ===")
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
    
    # Print model structure
    logger.info("=== Model Structure ===")
    logger.info(f"Input dimension: {vae_input_dim}")
    logger.info(f"Embedding dimension: {vae_embed_dim}")
    logger.info(f"Hidden dimensions: {vae_hidden_dims}")
    logger.info(f"Codebook size: {vae_codebook_size}")
    logger.info(f"Number of layers: {vae_n_layers}")
    logger.info(f"Number of categorical features: {vae_n_cat_feats}")
    logger.info(f"Tag class counts: {tag_class_counts}")
    logger.info(f"Tag embedding dimension: {tag_embed_dim}")
    logger.info(f"Tag alignment weight: {tag_alignment_weight}")
    logger.info(f"Tag prediction weight: {tag_prediction_weight}")
    
    # Print encoder structure
    logger.info("=== Encoder Structure ===")
    for name, param in model.encoder.named_parameters():
        logger.info(f"{name}: {param.shape}")
    
    # Print quantization layer structure
    logger.info("=== Quantization Layer Structure ===")
    for i, layer in enumerate(model.layers):
        logger.info(f"Quantization Layer {i}:")
        for name, param in layer.named_parameters():
            logger.info(f"  {name}: {param.shape}")
    
    # Print tag predictor structure
    logger.info("=== Tag Predictor Structure ===")
    for i, predictor in enumerate(model.tag_predictors):
        logger.info(f"Tag Predictor {i}:")
        for name, param in predictor.named_parameters():
            logger.info(f"  {name}: {param.shape}")
    
    # Print tag projector structure
    logger.info("=== Tag Projector Structure ===")
    for i, projector in enumerate(model.tag_projectors):
        logger.info(f"Tag Projector {i}:")
        for name, param in projector.named_parameters():
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
    if pretrained_hrqvae_path is not None:
        logger.info(f"Loading pretrained model: {pretrained_hrqvae_path}")
        model.load_pretrained(pretrained_hrqvae_path)
    
    # Create tokenizer
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
    
    # Run one forward pass
    logger.info("=== Running Forward Pass ===")
    model.eval()
    
    # Get a batch of data
    train_iter = iter(train_dataloader)
    batch = next(train_iter)
    data = batch_to(batch, device)
    
    # Print input data shape
    logger.info(f"Input data shape: {data.x.shape}")
    # Print shapes of multiple fields in the input data
    logger.info("=== Input Data Field Shapes ===")
    for field_name in data._fields:
        field_value = getattr(data, field_name)
        if isinstance(field_value, torch.Tensor):
            logger.info(f"  {field_name}: {field_value.shape}")
        elif field_value is not None:
            logger.info(f"  {field_name}: {type(field_value)}")
    
    # Check for tag data
    has_tag_data = hasattr(data, 'tags_emb') and data.tags_emb is not None
    has_tag_indices = hasattr(data, 'tags_indices') and data.tags_indices is not None
    
    if has_tag_data:
        logger.info(f"Tag embedding shape: {data.tags_emb.shape}")
    else:
        logger.info("No tag embeddings in data")
    
    if has_tag_indices:
        logger.info(f"Tag indices shape: {data.tags_indices.shape}")
    else:
        logger.info("No tag indices in data")
    
    # Run forward pass
    with torch.no_grad():
        t = 0.2  # Gumbel temperature
        # Get semantic IDs
        logger.info("=== Getting Semantic IDs ===")
        quantized = model.get_semantic_ids(data.x, tags_emb=data.tags_emb, tags_indices=data.tags_indices, gumbel_t=t)
    
    
    logger.info(f"Embeddings shape: {quantized.embeddings.shape}")
    logger.info(f"Residuals shape: {quantized.residuals.shape}")
    logger.info(f"Semantic IDs shape: {quantized.sem_ids.shape}")
    
    # Process quantization loss
    if quantized.quantize_loss.numel() > 1:
        logger.info(f"Quantization loss (mean): {quantized.quantize_loss.mean().item():.4f}")
        logger.info(f"Quantization loss shape: {quantized.quantize_loss.shape}")
    else:
        logger.info(f"Quantization loss: {quantized.quantize_loss.item():.4f}")
    
    # Process tag alignment loss
    if quantized.tag_align_loss.numel() > 1:
        logger.info(f"Tag alignment loss (mean): {quantized.tag_align_loss.mean().item():.4f}")
        logger.info(f"Tag alignment loss shape: {quantized.tag_align_loss.shape}")
    else:
        logger.info(f"Tag alignment loss: {quantized.tag_align_loss.item():.4f}")
    
    # Process tag prediction loss
    if quantized.tag_pred_loss.numel() > 1:
        logger.info(f"Tag prediction loss (mean): {quantized.tag_pred_loss.mean().item():.4f}")
        logger.info(f"Tag prediction loss shape: {quantized.tag_pred_loss.shape}")
    else:
        logger.info(f"Tag prediction loss: {quantized.tag_pred_loss.item():.4f}")
    
    # New: Output tag alignment loss and tag prediction loss for each layer
    logger.info("=== Per-Layer Tag Loss Details ===")
    if hasattr(quantized, 'tag_align_loss_by_layer') and quantized.tag_align_loss_by_layer is not None:
        logger.info("Tag alignment loss per layer:")
        for i, loss in enumerate(quantized.tag_align_loss_by_layer):
            logger.info(f"  Layer {i}: {loss.item():.4f}")
    else:
        logger.info("No per-layer tag alignment loss information")
        
    if hasattr(quantized, 'tag_pred_loss_by_layer') and quantized.tag_pred_loss_by_layer is not None:
        logger.info("Tag prediction loss per layer:")
        for i, loss in enumerate(quantized.tag_pred_loss_by_layer):
            logger.info(f"  Layer {i}: {loss.item():.4f}")
    else:
        logger.info("No per-layer tag prediction loss information")
        
    if hasattr(quantized, 'tag_pred_accuracy_by_layer') and quantized.tag_pred_accuracy_by_layer is not None:
        logger.info("Tag prediction accuracy per layer:")
        for i, acc in enumerate(quantized.tag_pred_accuracy_by_layer):
            logger.info(f"  Layer {i}: {acc.item():.4f}")
    else:
        logger.info("No per-layer tag prediction accuracy information")
    
    # Process tag prediction accuracy
    if quantized.tag_pred_accuracy.numel() > 1:
        logger.info(f"Tag prediction accuracy (mean): {quantized.tag_pred_accuracy.mean().item():.4f}")
        logger.info(f"Tag prediction accuracy shape: {quantized.tag_pred_accuracy.shape}")
    else:
        logger.info(f"Tag prediction accuracy: {quantized.tag_pred_accuracy.item():.4f}")
    
    # Print semantic ID distribution for each layer
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
    
    # Print tag prediction results
    logger.info("=== Tag Prediction Results ===")
    for i in range(vae_n_layers):
        if has_tag_indices:
            # Get tag indices for the current layer
            logger.info(f"data.tags_indices shape = {data.tags_indices.shape}")
            logger.info(f"data.tags_indices[:, i] shape = {data.tags_indices[:, i].shape}")
            layer_tag_indices = data.tags_indices[:, i]  # Get tag indices for the current layer
            valid_mask = (layer_tag_indices >= 0)
            valid_count = valid_mask.sum().item()
            
            if valid_count > 0:
                logger.info(f"Layer {i} number of valid tags: {valid_count}/{data.x.shape[0]}")
                
                # Get residuals for the current layer
                # logger.info(f"quantized  = {quantized}")
                logger.info(f"quantized.residuals shape = {quantized.residuals.shape}")
                layer_residual = quantized.residuals[:,:, i]  # Get residuals for the current layer
                logger.info(f"layer_residual shape = {layer_residual.shape}")
                
                # Predict using the tag predictor
                with torch.no_grad():
                    # In the tag prediction evaluation part, it needs to be modified to use concatenated embeddings
                    # Around lines 390-410
                    
                    # Before modification:
                    # layer_residual = quantized.residuals[:, :, i]
                    # pred_logits = model.tag_predictors[i](layer_residual)
                    
                    # Modified to:
                    layer_embs = []
                    for j in range(i+1):  # Collect embeddings from the first i+1 layers
                        layer_embs.append(quantized.embeddings[:, :, j])
                        
                    # Concatenate embeddings from the first i+1 layers
                    concat_emb = torch.cat(layer_embs, dim=1)  # [batch_size, (i+1)*embed_dim]
                        
                    # Use the concatenated embedding for prediction
                    pred_logits = model.tag_predictors[i](concat_emb)
                    pred_indices = torch.argmax(pred_logits, dim=-1)
                
                # Calculate accuracy
                valid_pred = pred_indices[valid_mask]
                valid_targets = layer_tag_indices[valid_mask]
                accuracy = (valid_pred == valid_targets).float().mean().item()
                
                logger.info(f"Layer {i} tag prediction accuracy: {accuracy:.4f}")
                
                # Print prediction results for the first 5 samples
                num_samples = min(5, valid_count)
                valid_indices = torch.where(valid_mask)[0][:num_samples]
                
                for j, idx in enumerate(valid_indices):
                    logger.info(f"  Sample {j}: Predicted={pred_indices[idx].item()}, True={layer_tag_indices[idx].item()}")
            else:
                logger.info(f"Layer {i} has no valid tags")
        else:
            logger.info(f"Layer {i} has no tag index data")
    
    logger.info("Inspection complete")

if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Inspect the HRQVAE model training process')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    args = parser.parse_args()
    
    # Load configuration
    gin.parse_config_file(args.config)
    
    # Run the training function
    train()
