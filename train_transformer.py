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
# Import NDCGAccumulator
from evaluate.metrics import TopKAccumulator
from evaluate.metrics import NDCGAccumulator
from modules.model import EncoderDecoderRetrievalModel
from modules.scheduler.inv_sqrt import InverseSquareRootScheduler
from modules.tokenizer.semids import SemanticIdTokenizer
# Import HSemanticIdTokenizer
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
                
                # Ensure x and y dimensions match
                iterations = self.iterations[:len(values)]
                
                plt.plot(iterations, values)
                plt.xlabel('Iterations')
                plt.ylabel(metric_name)
                plt.title(f'{metric_name} over Training')
                plt.grid(True)
                plt.savefig(os.path.join(save_path, f'{metric_name}_curve.png'))
                plt.close()

# Function to calculate the repetition rate
def calculate_repetition_rate(item_ids: torch.Tensor):
    if item_ids is None or item_ids.nelement() == 0:
        return 0.0, 0, 0
    # Use PyTorch's unique function to count unique rows
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
    # New parameter to select the tokenizer type
    use_h_tokenizer=False,
    # New parameters for tag prediction configuration
    tag_alignment_weight=0.5,
    tag_prediction_weight=0.5,
    tag_class_counts=None,
    tag_embed_dim=768,
    use_dedup_dim=False,
    # New parameter for concatenation mode
    use_concatenated_ids=True,  # Concatenation mode is enabled by default, mutually exclusive with use_dedup_dim
    use_interleaved_ids=False, # New parameter to control interleaved mode
):  
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Folder with timestamp
    log_dir = os.path.join(save_dir_root, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging
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
    
    # Create metrics tracker
    metrics_tracker = MetricsTracker([
        'loss', 'learning_rate', 'eval_loss', 
        'hit@1', 'hit@5', 'hit@10',
        'ndcg@1', 'ndcg@5', 'ndcg@10'
    ])
    
    # Log training parameters
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
    # Print the first sample from the training and evaluation datasets respectively
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

    # Select tokenizer based on parameters
    if use_h_tokenizer:
        logger.info("Using HSemanticIdTokenizer with tag prediction capabilities")
        # Ensure use_dedup_dim is mutually exclusive with use_concatenated_ids/use_interleaved_ids
        if use_dedup_dim and (use_concatenated_ids or use_interleaved_ids):
            logger.warning(f"use_dedup_dim ({use_dedup_dim}) is mutually exclusive with use_concatenated_ids ({use_concatenated_ids}) or use_interleaved_ids ({use_interleaved_ids}).")
            logger.warning("Forcing use_dedup_dim=False to avoid HSemanticIdTokenizer initialization error.")
            use_dedup_dim = False
        
        if use_concatenated_ids:
            logger.info("Using concatenation mode: Semantic IDs and Tag IDs will be concatenated")
        
        # Hardcode the correct tag_class_counts value to ensure consistency with the actual model
        # The actual value determined from the error message is [7, 30, 97]
        model_tag_class_counts = [7, 30, 97]
        logger.info(f"Using hardcoded tag class counts: {model_tag_class_counts}, instead of the one from the config file: {tag_class_counts}")
        
        tokenizer = HSemanticIdTokenizer(
            input_dim=vae_input_dim,
            hidden_dims=vae_hidden_dims,
            output_dim=vae_embed_dim,
            codebook_size=vae_codebook_size,
            n_layers=vae_n_layers,
            n_cat_feats=vae_n_cat_feats,
            hrqvae_weights_path=pretrained_rqvae_path,
            hrqvae_codebook_normalize=True,  # Ensure this is True
            hrqvae_sim_vq=vae_sim_vq,
            tag_alignment_weight=tag_alignment_weight,
            tag_prediction_weight=tag_prediction_weight,
            tag_class_counts=model_tag_class_counts,  # Use the hardcoded value
            tag_embed_dim=tag_embed_dim,
            use_dedup_dim=use_dedup_dim,
            use_concatenated_ids=use_concatenated_ids,  # Pass the concatenation mode parameter
            use_interleaved_ids=use_interleaved_ids # Pass the interleaved mode parameter
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
    
    # Pre-computation step
    tokenizer = accelerator.prepare(tokenizer)
    tokenizer.precompute_corpus_ids(item_dataset)
    logger.info("Tokenizer prepared and corpus IDs precomputed")
    
    # Calculate and print the ID repetition rate
    if accelerator.is_main_process: # Only execute on the main process
        cached_item_ids = tokenizer.cached_ids
        if cached_item_ids is not None and cached_item_ids.numel() > 0: # Add check for cached_item_ids.numel() > 0
            if use_dedup_dim: # When using SemanticIdTokenizer and use_dedup_dim=True
                logger.info("use_dedup_dim is True (typically used with SemanticIdTokenizer). Calculating repetition rate for original ID part and final IDs.")
                # Original ID part (without the last deduplication dimension)
                if cached_item_ids.shape[1] > 1:
                    original_ids_part = cached_item_ids[:, :-1]
                    original_rep_rate, num_unique_orig, total_orig = calculate_repetition_rate(original_ids_part)
                    logger.info(f"  Repetition rate of original ID part (first {original_ids_part.shape[1]} dims): {original_rep_rate:.4f} ({num_unique_orig} unique / {total_orig} total)")
                else:
                    logger.warning("cached_item_ids dimensions are insufficient to separate original IDs and the deduplication dimension.")

                # Final IDs (including the deduplication dimension)
                final_rep_rate, num_unique_final, total_final = calculate_repetition_rate(cached_item_ids)
                logger.info(f"  Repetition rate of final IDs (total {cached_item_ids.shape[1]} dims, including deduplication dim): {final_rep_rate:.4f} ({num_unique_final} unique / {total_final} total)")
            else: # use_dedup_dim is False (when using HSemanticIdTokenizer, or SemanticIdTokenizer with use_dedup_dim=False)
                logger.info("use_dedup_dim is False. Calculating repetition rate for the full IDs (for HSemanticIdTokenizer, this may include semantic and tag IDs).")
                rep_rate, num_unique, total = calculate_repetition_rate(cached_item_ids)
                logger.info(f"  Repetition rate of full IDs (total {cached_item_ids.shape[1]} dims): {rep_rate:.4f} ({num_unique} unique / {total} total)")

                # If using HSemanticIdTokenizer with concatenated or interleaved IDs, additionally calculate and print the repetition rate for the semantic ID part only
                if use_h_tokenizer and (use_concatenated_ids or use_interleaved_ids):
                    num_semantic_layers = tokenizer.n_layers  # This is the n_layers of the HRQ-VAE
                    
                    semantic_part_ids = None
                    if num_semantic_layers > 0 and cached_item_ids.shape[1] > 0:
                        if use_concatenated_ids:
                            if cached_item_ids.shape[1] >= num_semantic_layers:
                                semantic_part_ids = cached_item_ids[:, :num_semantic_layers]
                            else:
                                logger.warning(f"In concatenation mode, cached_item_ids dimension ({cached_item_ids.shape[1]}) is smaller than the number of semantic layers ({num_semantic_layers}), cannot extract semantic part.")
                        elif use_interleaved_ids:
                            # Extract interleaved semantic IDs: indices are 0, 2, 4, ...
                            semantic_indices = [i * 2 for i in range(num_semantic_layers) if i * 2 < cached_item_ids.shape[1]]
                            if semantic_indices:
                                semantic_part_ids = cached_item_ids[:, semantic_indices]
                            else:
                                logger.warning(f"In interleaved mode, cannot extract valid semantic ID indices based on num_semantic_layers ({num_semantic_layers}) and total dimensions ({cached_item_ids.shape[1]}).")
                        
                        if semantic_part_ids is not None and semantic_part_ids.numel() > 0:
                            logger.info(f"  Additionally, calculating repetition rate for the semantic-only ID part ({semantic_part_ids.shape[1]} dims extracted from full IDs):")
                            sem_rep_rate, sem_num_unique, sem_total = calculate_repetition_rate(semantic_part_ids)
                            logger.info(f"    Repetition rate of semantic-only ID part: {sem_rep_rate:.4f} ({sem_num_unique} unique / {sem_total} total)")
                        elif num_semantic_layers > 0 : # semantic_part_ids is None or empty but should not be
                            logger.warning(f"  Failed to extract semantic-only ID part for repetition rate calculation. num_semantic_layers: {num_semantic_layers}")
                    elif num_semantic_layers == 0:
                        logger.info("  num_semantic_layers is 0, skipping repetition rate calculation for semantic-only ID part.")

        else:
            logger.warning("tokenizer.cached_ids is None or empty, cannot calculate repetition rate.")
    
    if push_vae_to_hf:
        login()
        tokenizer.rq_vae.push_to_hub(vae_hf_model_name)
        logger.info(f"Pushed VAE to HuggingFace: {vae_hf_model_name}")

    # Get the number of semantic ID layers to correctly differentiate between semantic and tag IDs
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
        n_sem_layers=n_sem_layers,  # Pass the number of semantic ID layers
        use_interleaved_ids=use_interleaved_ids # Pass the interleaved mode parameter
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

    # Initialize NDCGAccumulator instance
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
                    
                    # Debug info: check if the loss has a gradient
                    if iter == 0 and _ == 0:
                        logger.info(f"Loss value: {loss.item()}")
                        logger.info(f"Loss requires grad: {loss.requires_grad}")
                        logger.info(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
                        logger.info(f"Parameters with requires_grad: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
                        
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
            
            # Log training metrics
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
                        # Add eval_loss to eval_debug_metrics instead of passing it separately during update
                        eval_debug_metrics["eval_loss"] = model_output_eval.loss.detach().cpu().item()
                
                if accelerator.is_main_process:
                    avg_eval_loss = np.mean(eval_losses)
                    logger.info(f"Evaluation at iteration {iter+1}: eval_loss={avg_eval_loss:.4f}")
                    
                    # Fix: Check if eval_debug_metrics already contains eval_loss, if so, don't pass it separately
                    if "eval_loss" in eval_debug_metrics:
                        metrics_tracker.update(iter+1, **eval_debug_metrics)
                    else:
                        metrics_tracker.update(iter+1, eval_loss=avg_eval_loss, **eval_debug_metrics)

            if (iter+1) % full_eval_every == 0:
                model.eval()
                model.enable_generation = True
                
                # Optimization: Use temporary variables to store original config instead of modifying global config directly
                # Create a context manager class to temporarily disable torch.compile
                class TempDisableDynamo:
                    def __init__(self):
                        pass
                    
                    def __enter__(self):
                        self.original_cache_size_limit = torch._dynamo.config.cache_size_limit
                        self.original_dynamic_shapes = torch._dynamo.config.dynamic_shapes
                        self.original_optimize_ddp = torch._dynamo.config.optimize_ddp
                        
                        # Temporarily modify config
                        torch._dynamo.config.cache_size_limit = 0
                        torch._dynamo.config.dynamic_shapes = False
                        torch._dynamo.config.optimize_ddp = False
                    
                    def __exit__(self, exc_type, exc_val, exc_tb):
                        # Restore original config
                        torch._dynamo.config.cache_size_limit = self.original_cache_size_limit
                        torch._dynamo.config.dynamic_shapes = self.original_dynamic_shapes
                        torch._dynamo.config.optimize_ddp = self.original_optimize_ddp
                
                # Temporarily disable dynamic compilation during evaluation
                with TempDisableDynamo():
                    # In the full evaluation section, both accumulators are needed
                    with tqdm(eval_dataloader, desc=f'Eval {iter+1}', disable=not accelerator.is_main_process) as pbar_eval:
                            for batch_idx, batch in enumerate(pbar_eval): # Add batch_idx for sample selection
                                try:
                                    data = batch_to(batch, device)
                                    tokenized_data = tokenizer(data)
        
                                    generated = model.generate_next_sem_id(tokenized_data, top_k=True, temperature=1)
                                    actual, top_k = tokenized_data.sem_ids_fut, generated.sem_ids
        
                                    logger.info(f"Actual shape: {actual.shape}, Predicted shape: {top_k.shape}")
                                    
                                    # Check for concatenation mode and update the tag part
                                    if use_concatenated_ids and hasattr(data, 'tags_indices'):
                                        logger.info("Fixing tag issue in concatenation mode...")
                                        
                                        # Check for dimension mismatch
                                        if actual.size(-1) != top_k.size(-1):
                                            logger.info(f"Dimension mismatch detected: actual={actual.size()}, top_k={top_k.size()}")
                                            
                                            # Get the number of semantic ID layers
                                            num_semantic_layers = n_sem_layers
                                            
                                            # If the generated IDs include a tag part but the actual IDs do not
                                            if top_k.size(-1) > num_semantic_layers and actual.size(-1) == num_semantic_layers:
                                                logger.info("Actual IDs are missing the tag part, attempting to add it...")
                                                
                                                # Check if tag indices exist in the data
                                                if hasattr(data, 'tags_indices') and data.tags_indices is not None:
                                                    batch_size = actual.size(0)
                                                    
                                                    # Get the tag indices
                                                    tags_indices = data.tags_indices
                                                    logger.info(f"Tag indices shape: {tags_indices.shape}")
                                                    
                                                    # Print original tag indices and number of tag classes
                                                    logger.info(f"Tag class counts: {tokenizer.tag_class_counts}")
                                                    if batch_size > 0:
                                                        sample_tags = tags_indices[0]
                                                        logger.info(f"Sample original tag indices: {sample_tags.tolist()}")
                                                    
                                                    # Create a new actual ID tensor, including semantic and tag IDs
                                                    num_tag_layers = min(len(tokenizer.tag_class_counts), tags_indices.shape[1])
                                                    
                                                    # Collect the current semantic IDs
                                                    semantic_ids = actual
                                                    
                                                    # Get the tag IDs for the current sample
                                                    tag_ids_list = []
                                                    for i in range(num_tag_layers):
                                                        tag_ids = tags_indices[:, i].clone()
                                                        if tokenizer.tag_class_counts is not None and i < len(tokenizer.tag_class_counts):
                                                            special_value = tokenizer.tag_class_counts[i]
                                                            logger.info(f"Special value for tag layer {i+1}: {special_value}")
                                                            
                                                            # Create a mapping for before and after replacement
                                                            orig_tags = tag_ids.clone()
                                                            tag_ids[tag_ids < 0] = special_value
                                                            
                                                            # Print the original and replaced values (for the first sample)
                                                            if batch_size > 0:
                                                                logger.info(f"Original tag value for layer {i+1}: {orig_tags[0].item()}")
                                                                logger.info(f"Replaced tag value for layer {i+1}: {tag_ids[0].item()}")
                                                            
                                                        tag_ids_list.append(tag_ids.unsqueeze(1))
                                                    
                                                    # Concatenate all tag IDs
                                                    tag_ids = torch.cat(tag_ids_list, dim=1)
                                                    logger.info(f"Tag IDs shape: {tag_ids.shape}")
                                                    
                                                    # Print the tag IDs for the first sample
                                                    if batch_size > 0:
                                                        logger.info(f"Tag IDs for the first sample: {tag_ids[0].tolist()}")
                                                    
                                                    # Concatenate semantic IDs and tag IDs
                                                    # Note: We need to handle different dimension cases here
                                                    if semantic_ids.dim() == 3:  # [batch, k, semantic_dim]
                                                        # Duplicate tags to match the number of candidates
                                                        expanded_tag_ids = tag_ids.unsqueeze(1).expand(-1, semantic_ids.size(1), -1)
                                                        new_actual = torch.cat([semantic_ids, expanded_tag_ids], dim=2)
                                                        logger.info(f"Shape after 3D concatenation: {new_actual.shape}")
                                                    elif semantic_ids.dim() == 2 and top_k.dim() == 3:  # [batch, semantic_dim] vs [batch, k, total_dim]
                                                        # Use the first num_semantic_layers dimensions of the semantic ID
                                                        new_actual = torch.cat([semantic_ids, tag_ids], dim=1)
                                                        logger.info(f"Shape after 2D concatenation: {new_actual.shape}")
                                                        
                                                        # Modify metric calculation, changing top_k to [batch, k, semantic_dim]
                                                        logger.info("Using common dimensions for metric calculation...")
                                                        
                                                        # Print the first sample for checking
                                                        if batch_size > 0:
                                                            logger.info(f"First sample after concatenation: semantic_ids={semantic_ids[0].tolist()}, tag_ids={tag_ids[0].tolist()}")
                                                            logger.info(f"Full concatenated ID: {new_actual[0].tolist()}")
                                                        
                                                        metrics_accumulator.accumulate(actual=new_actual, top_k=top_k)
                                                        ndcg_accumulator.accumulate(actual=new_actual, top_k=top_k)
                                                        continue  # Skip the subsequent metric calculation
                                                    else:
                                                        # Simple concatenation
                                                        new_actual = torch.cat([semantic_ids, tag_ids], dim=1)
                                                        logger.info(f"Shape after concatenation in other cases: {new_actual.shape}")
                                                    
                                                    logger.info(f"Shape of actual ID after concatenation: {new_actual.shape}")
                                                    actual = new_actual
                                        
                                        # Handle dimension mismatch issue in concatenation mode
                                        if use_concatenated_ids:
                                            # Ensure actual and top_k dimensions are consistent
                                            if actual.size(-1) != top_k.size(-1):
                                                logger.info(f"Dimension mismatch detected: actual={actual.size()}, top_k={top_k.size()}")
                                                
                                                # Get the common dimensions between the two
                                                common_dims = min(actual.size(-1), top_k.size(-1))
                                                
                                                # Use common dimensions for metric calculation - typically the semantic ID part
                                                metrics_accumulator.accumulate(actual=actual[..., :common_dims], top_k=top_k[..., :common_dims])
                                                ndcg_accumulator.accumulate(actual=actual[..., :common_dims], top_k=top_k[..., :common_dims])
                                            else:
                                                metrics_accumulator.accumulate(actual=actual, top_k=top_k)
                                                ndcg_accumulator.accumulate(actual=actual, top_k=top_k)
                                        else:
                                            metrics_accumulator.accumulate(actual=actual, top_k=top_k)
                                            ndcg_accumulator.accumulate(actual=actual, top_k=top_k)

                                    # Print predictions and ground truth for random samples (only on main process and for the first evaluation batch)
                                    if accelerator.is_main_process and batch_idx == 0:
                                        num_samples_to_print = 3 # The number of samples to print can be adjusted
                                        # Ensure it does not exceed the batch size
                                        actual_samples_to_print = min(num_samples_to_print, actual.size(0)) 
                                        
                                        # Randomly select sample indices
                                        sample_indices = torch.randperm(actual.size(0))[:actual_samples_to_print]

                                        logger.info(f"--- Sample Predictions at Iteration {iter+1} (Batch {batch_idx}) ---")
                                        # Print shape information of tokenized_data
                                        logger.info(f"tokenized_data.sem_ids_fut shape: {tokenized_data.sem_ids_fut.shape}")
                                        logger.info(f"tokenized_data.sem_ids shape: {tokenized_data.sem_ids.shape}")
                                        logger.info(f"Actual shape: {actual.shape}, Predicted shape: {top_k.shape}")
                                        
                                        # Print tag information from the original data
                                        if hasattr(data, 'tags_indices') and data.tags_indices is not None:
                                            logger.info(f"Data tag indices shape: {data.tags_indices.shape}")
                                            if hasattr(tokenizer, 'tag_class_counts'):
                                                logger.info(f"Tag class counts: {tokenizer.tag_class_counts}")
                                                
                                                # Print explanation of the tag hierarchy
                                                logger.info("Tag Hierarchy Explanation:")
                                                for i, count in enumerate(tokenizer.tag_class_counts):
                                                    # Determine tag type based on hierarchy and number of classes
                                                    if i == 0 and count <= 10:
                                                        tag_type = "Category"
                                                    elif i == 1 and count <= 50:
                                                        tag_type = "Sub-category"
                                                    else:
                                                        tag_type = "Specific Tag"
                                                    logger.info(f"  Layer {i+1} ({tag_type}): {count} classes, special_value={count} (for invalid tags)")
                                        
                                        for i in range(actual_samples_to_print):
                                            sample_idx = sample_indices[i].item()
                                            actual_ids_sample = actual[sample_idx]
                                            predicted_ids_sample = top_k[sample_idx]

                                            logger.info(f"  Sample {i+1} (Original Index: {sample_idx}):")
                                            
                                            # Print original tag indices
                                            if hasattr(data, 'tags_indices') and data.tags_indices is not None:
                                                orig_tags = data.tags_indices[sample_idx]
                                                logger.info(f"    Original tag indices: {orig_tags.tolist()}")
                                                
                                                # Print the result after tag mapping
                                                if hasattr(tokenizer, 'tag_class_counts'):
                                                    mapped_tags = []
                                                    for j, tag_idx in enumerate(orig_tags.tolist()):
                                                        if j < len(tokenizer.tag_class_counts) and tag_idx >= 0:
                                                            mapped_tags.append(tag_idx)
                                                        elif j < len(tokenizer.tag_class_counts):
                                                            # Use special value for invalid tags
                                                            mapped_tags.append(tokenizer.tag_class_counts[j])
                                                        else:
                                                            # Out-of-range tags
                                                            mapped_tags.append(-1)
                                                    logger.info(f"    Mapped tag IDs: {mapped_tags}")
                                            
                                            # Get information about the current sequence (if available)
                                            if hasattr(tokenized_data, 'sem_ids') and tokenized_data.sem_ids is not None:
                                                current_seq_ids = tokenized_data.sem_ids[sample_idx]
                                                
                                                # Get sequence length and semantic ID dimension
                                                seq_len = tokenized_data.seq_mask[sample_idx].sum().item() if hasattr(tokenized_data, 'seq_mask') else current_seq_ids.shape[0]
                                                
                                                # Extract IDs of the current sequence
                                                # Assuming ID format is [pos1_sem1, pos1_sem2, ..., pos1_tag1, pos1_tag2, ..., pos2_sem1, ...]
                                                # We need to extract all semantic and tag IDs from the first position
                                                if seq_len > 0 and n_sem_layers > 0:
                                                    # Calculate the total number of IDs per position
                                                    ids_per_pos = current_seq_ids.shape[0] // seq_len
                                                    
                                                    # Ensure the number of positions is reasonable
                                                    if ids_per_pos > 0:
                                                        # Extract IDs from the first position
                                                        first_pos_ids = current_seq_ids[:ids_per_pos]
                                                        
                                                        # Differentiate between semantic IDs and tag IDs
                                                        if ids_per_pos >= n_sem_layers:
                                                            current_sem_ids = first_pos_ids[:n_sem_layers]
                                                            if ids_per_pos > n_sem_layers:
                                                                current_tag_ids = first_pos_ids[n_sem_layers:]
                                                                logger.info(f"    Current sequence semantic IDs: {current_sem_ids.tolist()}")
                                                                logger.info(f"    Current sequence tag IDs: {current_tag_ids.tolist()}")
                                                            else:
                                                                logger.info(f"    Current sequence semantic IDs: {current_sem_ids.tolist()}")
                                                                logger.info(f"    Current sequence tag IDs: []")
                                            
                                            if use_concatenated_ids:
                                                # If using concatenated IDs, show semantic and tag IDs separately
                                                # Assuming semantic IDs come first, followed by tag IDs
                                                num_semantic_layers = n_sem_layers 
                                                
                                                # Print dimension information
                                                logger.info(f"    Actual ID dimension: {actual_ids_sample.shape}")
                                                logger.info(f"    Predicted ID dimension: {predicted_ids_sample.shape}")
                                                
                                                # Process the actual IDs of the future sequence
                                                if actual_ids_sample.dim() == 1:
                                                    # 1D tensor, possibly because there are only semantic IDs and no tag IDs
                                                    total_dim = actual_ids_sample.shape[0]
                                                    
                                                    # Check if dimensions are sufficient to separate semantic and tag IDs
                                                    if total_dim >= num_semantic_layers:
                                                        actual_semantic = actual_ids_sample[:num_semantic_layers]
                                                        # Check if there is a tag part
                                                        if total_dim > num_semantic_layers:
                                                            actual_tags = actual_ids_sample[num_semantic_layers:]
                                                        else:
                                                            # Only semantic IDs
                                                            actual_tags = torch.tensor([], device=actual_ids_sample.device)
                                                    else:
                                                        logger.warning(f"    Warning: Actual ID dimension ({total_dim}) is smaller than the number of semantic layers ({num_semantic_layers})")
                                                        actual_semantic = actual_ids_sample
                                                        actual_tags = torch.tensor([], device=actual_ids_sample.device)
                                                else:
                                                    # High-dimensional tensor, possibly an internal batch structure
                                                    logger.warning(f"    Warning: Actual ID is a high-dimensional tensor, structure might not be as expected")
                                                    actual_semantic = actual_ids_sample
                                                    actual_tags = torch.tensor([], device=actual_ids_sample.device)
                                                
                                                # Process predicted IDs, differentiating between semantic and tag IDs
                                                if predicted_ids_sample.dim() > 1:
                                                    # Predicted IDs may have multiple candidates, take the first one
                                                    top1_prediction = predicted_ids_sample[0]
                                                    
                                                    # Check if dimensions are sufficient to separate semantic and tag IDs
                                                    pred_total_dim = top1_prediction.shape[0]
                                                    if pred_total_dim >= num_semantic_layers:
                                                        predicted_semantic = top1_prediction[:num_semantic_layers]
                                                        if pred_total_dim > num_semantic_layers:
                                                            predicted_tags = top1_prediction[num_semantic_layers:]
                                                        else:
                                                            predicted_tags = torch.tensor([], device=top1_prediction.device)
                                                    else:
                                                        logger.warning(f"    Warning: Predicted ID dimension ({pred_total_dim}) is smaller than the number of semantic layers ({num_semantic_layers})")
                                                        predicted_semantic = top1_prediction
                                                        predicted_tags = torch.tensor([], device=top1_prediction.device)
                                                else:
                                                    # 1D prediction, split directly
                                                    pred_total_dim = predicted_ids_sample.shape[0]
                                                    if pred_total_dim >= num_semantic_layers:
                                                        predicted_semantic = predicted_ids_sample[:num_semantic_layers]
                                                        if pred_total_dim > num_semantic_layers:
                                                            predicted_tags = predicted_ids_sample[num_semantic_layers:]
                                                        else:
                                                            predicted_tags = torch.tensor([], device=predicted_ids_sample.device)
                                                    else:
                                                        logger.warning(f"    Warning: Predicted ID dimension ({pred_total_dim}) is smaller than the number of semantic layers ({num_semantic_layers})")
                                                        predicted_semantic = predicted_ids_sample
                                                        predicted_tags = torch.tensor([], device=predicted_ids_sample.device)
                                                
                                                # Output semantic IDs and tag IDs
                                                logger.info(f"    Future Semantic IDs (Actual): {actual_semantic.tolist()}")
                                                logger.info(f"    Future Tag IDs (Actual): {actual_tags.tolist()}")
                                                
                                                # Handle multi-dimensional predictions
                                                if predicted_ids_sample.dim() > 1:
                                                    # Print the semantic ID part of all Top-K candidates
                                                    semantic_predictions = [pred[:num_semantic_layers].tolist() for pred in predicted_ids_sample[:5]]  # Show top 5 only
                                                    logger.info(f"    Top-5 Future Semantic IDs (Predicted): {semantic_predictions}")
                                                    
                                                    # If dimensions are sufficient, print the tag ID part
                                                    if predicted_ids_sample.shape[1] > num_semantic_layers:
                                                        tag_predictions = [pred[num_semantic_layers:].tolist() for pred in predicted_ids_sample[:5]]
                                                        logger.info(f"    Top-5 Future Tag IDs (Predicted): {tag_predictions}")
                                                    else:
                                                        logger.info(f"    Future Tag IDs (Predicted): [] (Predicted ID dimension is insufficient)")
                                                else:
                                                    # 1D prediction
                                                    logger.info(f"    Future Semantic IDs (Predicted): {predicted_semantic.tolist()}")
                                                    logger.info(f"    Future Tag IDs (Predicted): {predicted_tags.tolist()}")
                                            else:
                                                logger.info(f"    Actual IDs: {actual_ids_sample.tolist()}")
                                                logger.info(f"    Predicted IDs: {predicted_ids_sample.tolist()}")
                                        logger.info(f"--- End Sample Predictions ---")
                                except Exception as e:
                                    logger.error(f"Error during evaluation: {str(e)}")
                                    import traceback
                                    logger.error(traceback.format_exc())
                                    continue
                
                eval_metrics = metrics_accumulator.reduce()
                ndcg_metrics = ndcg_accumulator.reduce()  # Get NDCG metrics
                
                if accelerator.is_main_process:
                    logger.info(f"Full evaluation at iteration {iter+1}:")
                    for metric_name, metric_value in eval_metrics.items():
                        logger.info(f"  {metric_name}: {metric_value:.4f}")
                    
                    # Log NDCG metrics
                    for metric_name, metric_value in ndcg_metrics.items():
                        logger.info(f"  {metric_name}: {metric_value:.4f}")
                    
                    # Merge both types of metrics for update
                    combined_metrics = {**eval_metrics, **ndcg_metrics}
                    metrics_tracker.update(iter+1, **combined_metrics)
                
                metrics_accumulator.reset()
                ndcg_accumulator.reset()  # Reset NDCG accumulator

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
    
    # After training, plot and save all metric graphs
    if accelerator.is_main_process:
        plots_dir = os.path.join(log_dir, "plots")
        logger.info(f"Training completed. Saving metric plots to {plots_dir}")
        metrics_tracker.plot_and_save(plots_dir)
        logger.info("All metric plots saved successfully")


if __name__ == "__main__":
    parse_config()
    train()
