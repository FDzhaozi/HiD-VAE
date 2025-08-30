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

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger("torch._dynamo.convert_frame").setLevel(logging.ERROR)
logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)

# Configure torch._dynamo warnings
torch._dynamo.config.verbose = False
torch._dynamo.config.suppress_errors = True

# New: Function to calculate ID repetition rate
def calculate_repetition_rate(item_ids: torch.Tensor):
    """
    Calculate the repetition rate of IDs.
    
    Args:
        item_ids: A tensor of IDs with shape [num_items, id_dim].
        
    Returns:
        repetition_rate: The repetition rate.
        num_unique_items: The number of unique IDs.
        total_items: The total number of IDs.
    """
    if item_ids is None or item_ids.nelement() == 0:
        return 0.0, 0, 0
    
    # Use PyTorch's unique function to find unique rows and their counts
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
    use_focal_loss=True,  # Enable focal loss by default
    focal_loss_gamma_base=2.0,  # Base gamma parameter
    focal_loss_alpha_base=0.25,  # Base alpha parameter
    rare_tag_threshold=30,  # Tags with counts below this are considered rare
    # New hyperparameters
    dropout_rate=0.3,  # Dropout rate in the predictor
    use_batch_norm=True,  # Whether to use BatchNorm
    alignment_temperature=0.1,  # Contrastive learning temperature parameter
    predictor_weight_decay=0.02,  # Weight decay for the tag predictor
    layer_specific_lr=False,  # Whether to use different learning rates for different layers
    # New: Advanced strategies to prevent overfitting
    use_label_smoothing=True,  # Whether to use label smoothing
    label_smoothing_alpha=0.1,  # Label smoothing alpha
    use_mixup=True,  # Whether to use mixup
    mixup_alpha=0.2,  # Mixup alpha
    # New: Evaluation strategies
    eval_tta=True,  # Test-Time Augmentation (TTA)
    eval_temperature=0.8,  # Temperature parameter for prediction
    ensemble_predictions=True,  # Whether to use ensemble predictions
    # New: Learning rate scheduler parameters
    use_lr_scheduler=True,
    lr_scheduler_type='cosine',  # 'cosine', 'reduce_on_plateau', 'step'
    lr_scheduler_T_max=400000, # For CosineAnnealingLR: Number of iterations for one cycle
    lr_scheduler_eta_min=1e-7, # For CosineAnnealingLR: Minimum learning rate
    lr_scheduler_step_size=100000, # For StepLR: Period of learning rate decay
    lr_scheduler_gamma=0.5, # For StepLR: Multiplicative factor of learning rate decay
    lr_scheduler_factor=0.5, # For ReduceLROnPlateau: Factor by which the learning rate will be reduced
    lr_scheduler_patience=10, # For ReduceLROnPlateau: Number of epochs with no improvement after which learning rate will be reduced
    # New: Semantic ID uniqueness constraint parameters
    sem_id_uniqueness_weight=0.5,  # Weight for the semantic ID uniqueness constraint
    sem_id_uniqueness_margin=0.5,  # Margin for the semantic ID uniqueness constraint
    # New: ID repetition rate threshold
    id_repetition_threshold=0.03,  # ID repetition rate threshold, model is saved only if below this value
    # New: Tokenizer mode parameters
    use_concatenated_ids: bool = True,
    use_interleaved_ids: bool = False,
):
    # Set up logging
    # Create directory to save plots
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_dir_root, f"hrqvae_{dataset.name}_{time_stamp}")
    log_dir = os.path.join(save_dir, "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    plots_dir = os.path.join(log_dir, "plots") 
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"hrqvae_training_{timestamp}.log")
    
    # Configure logger
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[file_handler, console_handler]
    )
    
    logger = logging.getLogger("hrqvae_training")
    
    # Initialize data collectors for plotting
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
    
    # First, create an accelerator instance
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )
    
    best_eval_accuracy = 0.0
    
    # Log training parameters
    if accelerator.is_main_process:
        params = locals()
        logger.info("Training parameters:")
        for key, value in params.items():
            if key != 'logger':
                logger.info(f"  {key}: {value}")

    device = accelerator.device

    # Load the dataset with tag data
    if dataset == RecDataset.KUAIRAND:
        # KuaiRand dataset does not accept the split parameter
        train_dataset = ItemData(root=dataset_folder, dataset=dataset, force_process=force_dataset_process, train_test_split="train" if do_eval else "all")
    else:
        # Other datasets accept the split parameter
        train_dataset = ItemData(root=dataset_folder, dataset=dataset, force_process=force_dataset_process, train_test_split="train" if do_eval else "all", split=dataset_split)
    
    train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size, False)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None, collate_fn=lambda batch: batch)
    train_dataloader = cycle(train_dataloader)

    if do_eval:
        if dataset == RecDataset.KUAIRAND:
            # KuaiRand dataset does not accept the split parameter
            eval_dataset = ItemData(root=dataset_folder, dataset=dataset, force_process=False, train_test_split="eval")
        else:
            # Other datasets accept the split parameter
            eval_dataset = ItemData(root=dataset_folder, dataset=dataset, force_process=False, train_test_split="eval", split=dataset_split)
        
        eval_sampler = BatchSampler(RandomSampler(eval_dataset), batch_size, False)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=None, collate_fn=lambda batch: batch)

    if dataset == RecDataset.KUAIRAND:
        # KuaiRand dataset does not accept the split parameter
        index_dataset = ItemData(root=dataset_folder, dataset=dataset, force_process=False, train_test_split="all") if do_eval else train_dataset
    else:
        # Other datasets accept the split parameter
        index_dataset = ItemData(root=dataset_folder, dataset=dataset, force_process=False, train_test_split="all", split=dataset_split) if do_eval else train_dataset
    
    train_dataloader = accelerator.prepare(train_dataloader)

    # Check if the dataset contains tag information
    has_tags = getattr(train_dataset, 'has_tags', False)
    if not has_tags:
        logger.warning("Dataset does not contain tag information. Disabling tag alignment and prediction.")
        tag_alignment_weight = 0.0
        tag_prediction_weight = 0.0
    else:
        logger.info("Dataset contains tag information.")
        
        # Ensure only the number of tag layers matching vae_n_layers is used
        # Check the shape of the tag data
        sample_data = train_dataset[0]
        if hasattr(sample_data, 'tags_emb') and sample_data.tags_emb is not None:
            logger.info(f"sample_data.tags_emb.shape = {sample_data.tags_emb.shape}")
            actual_tag_layers = sample_data.tags_emb.shape[1]
            logger.info(f"Number of tag layers in dataset: {actual_tag_layers}")
            
            if actual_tag_layers != vae_n_layers:
                logger.warning(f"Number of tag layers ({actual_tag_layers}) does not match number of model layers ({vae_n_layers})")
                
                # Operate directly on the entire dataset
                if actual_tag_layers > vae_n_layers:
                    logger.warning(f"Truncating dataset tags to keep only the first {vae_n_layers} layers")
                    # Truncate tag embeddings
                    if hasattr(train_dataset, 'tags_emb') and train_dataset.tags_emb is not None:
                        train_dataset.tags_emb = train_dataset.tags_emb[:, :vae_n_layers, :]
                    
                    # Truncate tag indices
                    if hasattr(train_dataset, 'tags_indices') and train_dataset.tags_indices is not None:
                        train_dataset.tags_indices = train_dataset.tags_indices[:, :vae_n_layers]
                    
                    logger.info(f"Shape of train_dataset.tags_emb after truncation = {train_dataset.tags_emb.shape if hasattr(train_dataset, 'tags_emb') and train_dataset.tags_emb is not None else 'None'}")
                    logger.info(f"Shape of train_dataset.tags_indices after truncation = {train_dataset.tags_indices.shape if hasattr(train_dataset, 'tags_indices') and train_dataset.tags_indices is not None else 'None'}")
                else:
                    logger.warning(f"Number of model layers ({vae_n_layers}) is greater than number of tag layers ({actual_tag_layers}). Padding dataset tags.")
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
                    
                    logger.info(f"Shape of train_dataset.tags_emb after padding = {train_dataset.tags_emb.shape if hasattr(train_dataset, 'tags_emb') and train_dataset.tags_emb is not None else 'None'}")
                    logger.info(f"Shape of train_dataset.tags_indices after padding = {train_dataset.tags_indices.shape if hasattr(train_dataset, 'tags_indices') and train_dataset.tags_indices is not None else 'None'}")
                    
        # If an evaluation dataset exists, it also needs the same processing
        if do_eval:
            sample_eval_data = eval_dataset[0]
            if hasattr(sample_eval_data, 'tags_emb') and sample_eval_data.tags_emb is not None:
                logger.info(f"sample_eval_data.tags_emb.shape = {sample_eval_data.tags_emb.shape}")
                actual_eval_tag_layers = sample_eval_data.tags_emb.shape[1]
                logger.info(f"Number of tag layers in eval dataset: {actual_eval_tag_layers}")
                
                if actual_eval_tag_layers != vae_n_layers:
                    logger.warning(f"Number of tag layers in eval dataset ({actual_eval_tag_layers}) does not match number of model layers ({vae_n_layers})")
                    
                    if actual_eval_tag_layers > vae_n_layers:
                        logger.warning(f"Truncating eval dataset tags to keep only the first {vae_n_layers} layers")
                        # Truncate tag embeddings
                        if hasattr(eval_dataset, 'tags_emb') and eval_dataset.tags_emb is not None:
                            eval_dataset.tags_emb = eval_dataset.tags_emb[:, :vae_n_layers, :]
                        
                        # Truncate tag indices
                        if hasattr(eval_dataset, 'tags_indices') and eval_dataset.tags_indices is not None:
                            eval_dataset.tags_indices = eval_dataset.tags_indices[:, :vae_n_layers]
                        
                        logger.info(f"Shape of eval_dataset.tags_emb after truncation = {eval_dataset.tags_emb.shape if hasattr(eval_dataset, 'tags_emb') and eval_dataset.tags_emb is not None else 'None'}")
                        logger.info(f"Shape of eval_dataset.tags_indices after truncation = {eval_dataset.tags_indices.shape if hasattr(eval_dataset, 'tags_indices') and eval_dataset.tags_indices is not None else 'None'}")
                    else:
                        logger.warning(f"Number of model layers ({vae_n_layers}) is greater than number of tag layers in eval dataset ({actual_eval_tag_layers}). Padding eval dataset tags.")
                        # Pad tag embeddings
                        if hasattr(eval_dataset, 'tags_emb') and eval_dataset.tags_emb is not None:
                            tag_embed_shape = eval_dataset.tags_emb.shape
                            padded_tags_emb = torch.zeros((tag_embed_shape[0], vae_n_layers, tag_embed_shape[2]), 
                                                          dtype=eval_dataset.tags_emb.dtype)
                            padded_tags_emb[:, :actual_eval_tag_layers, :] = eval_dataset.tags_emb
                            eval_dataset.tags_emb = padded_tags_emb
                        
                        # Pad tag indices
                        if hasattr(eval_dataset, 'tags_indices') and eval_dataset.tags_indices is not None:
                            tag_indices_shape = eval_dataset.tags_indices.shape
                            padded_tags_indices = torch.ones((tag_indices_shape[0], vae_n_layers), 
                                                            dtype=eval_dataset.tags_indices.dtype) * -1
                            padded_tags_indices[:, :actual_eval_tag_layers] = eval_dataset.tags_indices
                            eval_dataset.tags_indices = padded_tags_indices
                        
                        logger.info(f"Shape of eval_dataset.tags_emb after padding = {eval_dataset.tags_emb.shape if hasattr(eval_dataset, 'tags_emb') and eval_dataset.tags_emb is not None else 'None'}")
                        logger.info(f"Shape of eval_dataset.tags_indices after padding = {eval_dataset.tags_indices.shape if hasattr(eval_dataset, 'tags_indices') and eval_dataset.tags_indices is not None else 'None'}")

    # If tag_class_counts is not provided, use default values
    if tag_class_counts is None:
        tag_class_counts = [10, 100, 1000][:vae_n_layers]
    
    # Ensure the number of tag classes matches the number of layers
    assert len(tag_class_counts) == vae_n_layers, f"Number of tag classes {len(tag_class_counts)} does not match number of layers {vae_n_layers}"
    
    # Create a dictionary for focal loss parameters, setting different values for each layer
    focal_loss_params = {
        'gamma': focal_loss_gamma_base,
        'alpha': focal_loss_alpha_base,
    }
    
    # Set different gamma for each layer; higher layers get a larger gamma to focus more on hard-to-classify samples
    for i in range(vae_n_layers):
        # Higher layers get a larger gamma to focus more on hard-to-classify samples
        focal_loss_params[f'gamma_{i}'] = focal_loss_gamma_base * (1 + i * 0.5)
        # Higher layers get a smaller alpha to focus more on minority classes
        focal_loss_params[f'alpha_{i}'] = max(0.05, focal_loss_alpha_base - i * 0.05)
    
    if accelerator.is_main_process:
        logger.info("Focal loss parameters:")
        for key, value in focal_loss_params.items():
            logger.info(f"  {key}: {value}")
    
    # Calculate frequency statistics for tag classes in each layer and handle rare tags
    if has_tags and use_focal_loss:
        logger.info("Calculating tag class frequency statistics...")
        class_counts_dict = {}
        rare_tags_dict = {}  # Store rare tag IDs for each layer
        original_tag_class_counts = tag_class_counts.copy()  # Save original number of tag classes
        new_tag_class_counts = []  # New number of tag classes
        
        # Iterate through the training set to count occurrences of each class in each layer
        for i in range(vae_n_layers):
            if hasattr(train_dataset, 'tags_indices') and train_dataset.tags_indices is not None:
                # Get all tag indices for the current layer
                layer_indices = train_dataset.tags_indices[:, i]
                # Only count valid tags (not -1)
                valid_indices = layer_indices[layer_indices >= 0]
                
                if len(valid_indices) > 0:
                    # Count occurrences for each class
                    unique_classes, counts = torch.unique(valid_indices, return_counts=True)
                    
                    # Create a full class count tensor, with counts of 0 for classes that do not appear
                    full_counts = torch.zeros(original_tag_class_counts[i], dtype=torch.long)
                    full_counts[unique_classes.long()] = counts
                    
                    # Find rare tags (occurrence count is less than the threshold)
                    rare_mask = (full_counts > 0) & (full_counts < rare_tag_threshold)
                    rare_tag_ids = torch.nonzero(rare_mask).squeeze(-1)
                    
                    # Record rare tag IDs
                    rare_tags_dict[i] = rare_tag_ids
                    
                    # Calculate the number of non-rare tags
                    non_rare_count = ((full_counts >= rare_tag_threshold) | (full_counts == 0)).sum().item()
                    
                    # New number of tag classes = number of non-rare tags + 1 (for the special class)
                    new_tag_class_counts.append(non_rare_count + 1)
                    
                    # Calculate class distribution statistics
                    total_samples = full_counts.sum().item()
                    non_zero_classes = (full_counts > 0).sum().item()
                    rare_classes = rare_tag_ids.numel()
                    max_count = full_counts.max().item()
                    min_count = full_counts[full_counts > 0].min().item() if (full_counts > 0).any() else 0
                    mean_count = full_counts[full_counts > 0].float().mean().item() if (full_counts > 0).any() else 0
                    
                    logger.info(f"Layer {i} Tag Statistics: Total samples={total_samples}, Non-zero classes={non_zero_classes}/{original_tag_class_counts[i]}, "
                                f"Rare classes={rare_classes}, Max count={max_count}, Min count={min_count}, Mean count={mean_count:.2f}")
                    
                    # Store in dictionary
                    class_counts_dict[i] = full_counts
                else:
                    # If there are no valid tags, keep the original number of classes
                    new_tag_class_counts.append(original_tag_class_counts[i])
            else:
                # If there are no tag indices, keep the original number of classes
                new_tag_class_counts.append(original_tag_class_counts[i])
        
        # Export rare tag IDs to a .pt file
        rare_tags_path = os.path.join(save_dir_root+"special_tags_files", "rare_tags.pt")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(rare_tags_path), exist_ok=True)
        torch.save(rare_tags_dict, rare_tags_path)
        logger.info(f"Rare tag IDs have been saved to: {rare_tags_path}")
        
        # Update the number of tag classes
        tag_class_counts = new_tag_class_counts
        logger.info(f"Updated number of tag classes: {tag_class_counts}")
        
        # Remap tag indices in the training set
        if hasattr(train_dataset, 'tags_indices') and train_dataset.tags_indices is not None:
            for i in range(vae_n_layers):
                if i in rare_tags_dict and len(rare_tags_dict[i]) > 0:
                    # Get all tag indices for the current layer
                    layer_indices = train_dataset.tags_indices[:, i]
                    
                    # Create a new mapping: rare tags -> special class ID (using new_class_count - 1 as the special ID)
                    special_class_id = new_tag_class_counts[i] - 1
                    
                    # Create mapping table: original ID -> new ID
                    id_mapping = torch.arange(original_tag_class_counts[i], dtype=torch.long)
                    
                    # Calculate new IDs for non-rare tags (maintaining order but skipping rare tags)
                    non_rare_ids = torch.ones(original_tag_class_counts[i], dtype=torch.bool)
                    non_rare_ids[rare_tags_dict[i]] = False
                    
                    # Assign new IDs to non-rare tags
                    new_ids = torch.cumsum(non_rare_ids, dim=0) - 1
                    id_mapping[non_rare_ids] = new_ids[non_rare_ids]
                    
                    # Assign the special class ID to rare tags
                    id_mapping[rare_tags_dict[i]] = special_class_id
                    
                    # Apply mapping to the training set
                    valid_mask = layer_indices >= 0
                    layer_indices[valid_mask] = id_mapping[layer_indices[valid_mask]]
                    train_dataset.tags_indices[:, i] = layer_indices
                    
                    logger.info(f"Layer {i} tag indices have been remapped. Special class ID: {special_class_id}")
            
            # If an evaluation dataset exists, it also needs to be remapped
            if do_eval and hasattr(eval_dataset, 'tags_indices') and eval_dataset.tags_indices is not None:
                for i in range(vae_n_layers):
                    if i in rare_tags_dict and len(rare_tags_dict[i]) > 0:
                        # Get all tag indices for the current layer
                        layer_indices = eval_dataset.tags_indices[:, i]
                        
                        # Create a new mapping: rare tags -> special class ID
                        special_class_id = new_tag_class_counts[i] - 1
                        
                        # Create mapping table: original ID -> new ID
                        id_mapping = torch.arange(original_tag_class_counts[i], dtype=torch.long)
                        
                        # Calculate new IDs for non-rare tags
                        non_rare_ids = torch.ones(original_tag_class_counts[i], dtype=torch.bool)
                        non_rare_ids[rare_tags_dict[i]] = False
                        
                        # Assign new IDs to non-rare tags
                        new_ids = torch.cumsum(non_rare_ids, dim=0) - 1
                        id_mapping[non_rare_ids] = new_ids[non_rare_ids]
                        
                        # Assign the special class ID to rare tags
                        id_mapping[rare_tags_dict[i]] = special_class_id
                        
                        # Apply mapping to the evaluation set
                        valid_mask = layer_indices >= 0
                        layer_indices[valid_mask] = id_mapping[layer_indices[valid_mask]]
                        eval_dataset.tags_indices[:, i] = layer_indices
                        
                        logger.info(f"Eval set layer {i} tag indices have been remapped")
        
        logger.info("Tag class frequency statistics and rare tag handling complete.")
    
        # Pass class frequency statistics to the model so that class weights can be used in the tag prediction loss
        class_counts_tensor_dict = {k: v.to(device) for k, v in class_counts_dict.items()}

    # Create HRqVae model
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
        tag_class_counts=tag_class_counts,  # Use updated tag class counts
        tag_embed_dim=tag_embed_dim,
        use_focal_loss=use_focal_loss,
        focal_loss_params=focal_loss_params,
        # New parameters
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
        alignment_temperature=alignment_temperature,
        # New semantic ID uniqueness constraint parameters
        sem_id_uniqueness_weight=sem_id_uniqueness_weight,
        sem_id_uniqueness_margin=sem_id_uniqueness_margin
    )

    # If focal loss is enabled and class frequencies have been calculated, update the class counts in the model
    if use_focal_loss and 'class_counts_tensor_dict' in locals():
        model.update_class_counts(class_counts_tensor_dict)

    # Set parameters for the tag prediction loss function
    if hasattr(model, 'tag_prediction_loss'):
        model.tag_prediction_loss.use_label_smoothing = use_label_smoothing
        model.tag_prediction_loss.label_smoothing_alpha = label_smoothing_alpha
        model.tag_prediction_loss.use_mixup = use_mixup
        model.tag_prediction_loss.mixup_alpha = mixup_alpha

    # If using layer-specific learning rates, use different parameter groups for different components
    if layer_specific_lr:
        # Divide model parameters into groups, each with a different learning rate and weight decay
        param_groups = [
            # Encoder and decoder use the base learning rate
            {'params': list(model.encoder.parameters()) + list(model.decoder.parameters()), 
             'lr': learning_rate, 'weight_decay': weight_decay},
            # Quantization layers use the base learning rate
            {'params': [p for layer in model.layers for p in layer.parameters()], 
             'lr': learning_rate, 'weight_decay': weight_decay},
        ]
        
        # Use different learning rates and weight decays for each layer's tag predictor and projector
        for i in range(vae_n_layers):
            # Predictor's learning rate slightly increases with layer number, while weight decay slightly decreases
            predictor_lr = learning_rate * (1 + i * 0.1) 
            predictor_wd = predictor_weight_decay / (1 + i * 0.2) if predictor_weight_decay > 0 and (1 + i * 0.2) > 0 else predictor_weight_decay

            param_groups.append({
                'params': model.tag_predictors[i].parameters(),
                'lr': predictor_lr,
                'weight_decay': predictor_wd
            })
            
            # Projector uses the same strategy
            param_groups.append({
                'params': model.tag_projectors[i].parameters(),
                'lr': predictor_lr,
                'weight_decay': predictor_wd
            })
        
        optimizer = AdamW(param_groups)
        
        if accelerator.is_main_process:
            logger.info("Using layer-specific learning rate and weight decay")
            for i, group in enumerate(param_groups):
                logger.info(f"Parameter group {i}: lr={group['lr']}, weight_decay={group['weight_decay']}")
    else:
        # Use a uniform learning rate and weight decay
        optimizer = AdamW(
            params=model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if accelerator.is_main_process:
        logger.info("Model configuration:")
        logger.info(f"  input_dim: {vae_input_dim}")
        logger.info(f"  embed_dim: {vae_embed_dim}")
        logger.info(f"  hidden_dims: {vae_hidden_dims}")
        logger.info(f"  codebook_size: {vae_codebook_size}")
        logger.info(f"  n_layers: {vae_n_layers}")
        logger.info(f"  tag_class_counts: {tag_class_counts}")
        logger.info(f"  tag_embed_dim: {tag_embed_dim}")
        logger.info(f"  tag_alignment_weight: {tag_alignment_weight}")
        logger.info(f"  tag_prediction_weight: {tag_prediction_weight}")
        logger.info(f"  len(train_dataset): {len(train_dataset)}")

    # Print encoder structure
    logger.info("=== Encoder Structure ===")
    for name, param in model.encoder.named_parameters():
        logger.info(f"{name}: {param.shape}")
    
    # Print quantization layers structure
    logger.info("=== Quantization Layers Structure ===")
    for i, layer in enumerate(model.layers):
        logger.info(f"Quantization Layer {i}:")
        for name, param in layer.named_parameters():
            logger.info(f"  {name}: {param.shape}")
    
    # Print tag predictors structure
    logger.info("=== Tag Predictors Structure ===")
    for i, predictor in enumerate(model.tag_predictors):
        logger.info(f"Tag Predictor {i}:")
        for name, param in predictor.named_parameters():
            logger.info(f"  {name}: {param.shape}")
    
    # Print tag projectors structure
    logger.info("=== Tag Projectors Structure ===")
    for i, projector in enumerate(model.tag_projectors):
        logger.info(f"Tag Projector {i}:")
        for name, param in projector.named_parameters():
            logger.info(f"  {name}: {param.shape}")
    
    # Print decoder structure
    logger.info("=== Decoder Structure ===")
    for name, param in model.decoder.named_parameters():
        logger.info(f"{name}: {param.shape}")

    start_iter = 0
    if pretrained_hrqvae_path is not None:
        model.load_pretrained(pretrained_hrqvae_path)
        state = torch.load(pretrained_hrqvae_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_iter = state["iter"]+1
        if accelerator.is_main_process:
            logger.info(f"Loading pretrained model: {pretrained_hrqvae_path}, starting from iteration {start_iter}")

    model, optimizer = accelerator.prepare(
        model, optimizer
    )

    # Initialize learning rate scheduler
    scheduler = None
    if use_lr_scheduler:
        if lr_scheduler_type == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_scheduler_T_max, eta_min=lr_scheduler_eta_min, last_epoch=start_iter-1 if start_iter > 0 else -1)
            if accelerator.is_main_process:
                logger.info(f"Using CosineAnnealingLR scheduler: T_max={lr_scheduler_T_max}, eta_min={lr_scheduler_eta_min}")
        elif lr_scheduler_type == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma, last_epoch=start_iter-1 if start_iter > 0 else -1)
            if accelerator.is_main_process:
                logger.info(f"Using StepLR scheduler: step_size={lr_scheduler_step_size}, gamma={lr_scheduler_gamma}")
        # ReduceLROnPlateau needs a metric to monitor, usually validation loss, so it's complex to call directly in the loop.
        # For now, we are not supporting ReduceLROnPlateau directly in this script due to its metric dependency.
        # elif lr_scheduler_type == 'reduce_on_plateau':
        #     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_scheduler_factor, patience=lr_scheduler_patience, verbose=True)
        #     if accelerator.is_main_process:
        #         logger.info(f"Using ReduceLROnPlateau scheduler: factor={lr_scheduler_factor}, patience={lr_scheduler_patience}")
        else:
            if accelerator.is_main_process:
                logger.warning(f"Unsupported learning rate scheduler type: {lr_scheduler_type}. Not using a scheduler.")

    if scheduler: # Prepare scheduler with accelerator if it exists
        scheduler = accelerator.prepare(scheduler)

    # Create HSemanticIdTokenizer
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
        losses = [[], [], [], [], [], []]  # total_loss, recon_loss, rqvae_loss, tag_align_loss, tag_pred_loss, tag_pred_accuracy
        # Added per-layer loss tracking
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
                    logger.info("K-means initialization complete")

            optimizer.zero_grad()
            for _ in range(gradient_accumulate_every):
                data = next_batch(train_dataloader, device)

                with accelerator.autocast():
                    model_output = model(data, gumbel_t=t)
                    loss = model_output.loss
                    loss = loss / gradient_accumulate_every
                    total_loss += loss
            
            # Backward pass
            accelerator.backward(total_loss)

            losses[0].append(total_loss.cpu().item())
            # Ensure to call .mean() before .item()
            losses[1].append(model_output.reconstruction_loss.mean().cpu().item())
            losses[2].append(model_output.rqvae_loss.mean().cpu().item())
            losses[3].append(model_output.tag_align_loss.mean().cpu().item())
            losses[4].append(model_output.tag_pred_loss.mean().cpu().item())
            losses[5].append(model_output.tag_pred_accuracy.mean().cpu().item())
            
            # Record per-layer tag alignment and prediction losses
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
            
            # Maintain a sliding window of size 1000
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
            
            # Update learning rate (if scheduler is used)
            if scheduler and lr_scheduler_type != 'reduce_on_plateau': # ReduceLROnPlateau is typically stepped after validation
                scheduler.step()
            
            accelerator.wait_for_everyone()

            id_diversity_log = {}
            if accelerator.is_main_process:
                # Log training information using logging
                if iter % 100 == 0:  # Log detailed info every 100 iterations
                    # Calculate the average of embedding norms
                    emb_norms_avg = model_output.embs_norm.mean(axis=0)
                    emb_norms_str = ", ".join([f"layer_{i}: {emb_norms_avg[i].cpu().item():.4f}" for i in range(vae_n_layers)])
                    
                    # Collect data for plotting
                    plot_data['iterations'].append(iter)
                    plot_data['total_loss'].append(total_loss.cpu().item())
                    plot_data['reconstruction_loss'].append(model_output.reconstruction_loss.mean().cpu().item())
                    plot_data['rqvae_loss'].append(model_output.rqvae_loss.mean().cpu().item())
                    plot_data['tag_align_loss'].append(model_output.tag_align_loss.mean().cpu().item())
                    plot_data['tag_pred_loss'].append(model_output.tag_pred_loss.mean().cpu().item())
                    plot_data['tag_pred_accuracy'].append(model_output.tag_pred_accuracy.mean().cpu().item())
                    
                    for i in range(vae_n_layers):
                        plot_data['emb_norms'][i].append(emb_norms_avg[i].cpu().item())
                    
                    logger.info(f"Iteration {iter} - Loss: {total_loss.cpu().item():.4f}, "
                                f"Reconstruction Loss: {model_output.reconstruction_loss.mean().cpu().item():.4f}, "
                                f"RQVAE Loss: {model_output.rqvae_loss.mean().cpu().item():.4f}, "
                                f"Tag Alignment Loss: {model_output.tag_align_loss.mean().cpu().item():.4f}, "
                                f"Tag Prediction Loss: {model_output.tag_pred_loss.mean().cpu().item():.4f}, "
                                f"Tag Prediction Accuracy: {model_output.tag_pred_accuracy.mean().cpu().item():.4f}, "
                                f"Temperature: {t:.4f}, "
                                f"Unique ID Ratio: {model_output.p_unique_ids.cpu().item():.4f}, "
                                f"Embedding Norms: {emb_norms_str}")
                    
                    # Print per-layer tag prediction accuracy
                    if model_output.tag_pred_accuracy_by_layer is not None:
                        layer_acc_str = ", ".join([f"Layer {i}: {acc.cpu().item():.4f}" for i, acc in enumerate(model_output.tag_pred_accuracy_by_layer)])
                        logger.info(f"Per-layer Tag Prediction Accuracy: {layer_acc_str}")
                    
                    # Log the current learning rate
                    current_lrs = [group['lr'] for group in optimizer.param_groups]
                    lr_str = ", ".join([f"{lr:.2e}" for lr in current_lrs])
                    logger.info(f"Current Learning Rate(s): {lr_str}")
            # Evaluation phase
            if do_eval and ((iter+1) % eval_every == 0 or iter+1 == iterations):
                model.eval()
                with tqdm(eval_dataloader, desc=f'Eval {iter+1}', disable=not accelerator.is_main_process) as pbar_eval:
                    eval_losses = [[], [], [], [], [], []]  # total, recon, rqvae, tag_align, tag_pred, tag_acc
                    
                    # Collect evaluation samples to display prediction results
                    eval_samples = []
                    # Ensure the predicted tags list is initialized here
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
                        
                        # Collect sample data (limit total number)
                        if len(eval_samples) < 100 and hasattr(data, 'tags_indices') and data.tags_indices is not None:
                            # Collect input features and true labels of the sample
                            batch_size = data.x.size(0)
                            samples_to_collect = min(batch_size, 100 - len(eval_samples))
                            
                            for i in range(samples_to_collect):
                                # Get features and true labels for the current sample
                                sample_x = data.x[i].cpu()
                                sample_tags = data.tags_indices[i].cpu()
                                
                                # Check if the sample has at least one valid tag
                                has_valid_tag = (sample_tags >= 0).any().item()
                                
                                if has_valid_tag:
                                    # Only collect information for samples with valid tags
                                    eval_samples.append({
                                        'x': sample_x,
                                        'true_tags': sample_tags
                                    })
                    
                    # Log the total number of collected samples
                    logger.info(f"Collected {len(eval_samples)} valid samples for evaluation")
                    
                    eval_losses = np.array(eval_losses).mean(axis=-1)
                    id_diversity_log["eval_total_loss"] = eval_losses[0]
                    id_diversity_log["eval_reconstruction_loss"] = eval_losses[1]
                    id_diversity_log["eval_rqvae_loss"] = eval_losses[2]
                    id_diversity_log["eval_tag_align_loss"] = eval_losses[3]
                    id_diversity_log["eval_tag_pred_loss"] = eval_losses[4]
                    id_diversity_log["eval_tag_pred_accuracy"] = eval_losses[5]
                    
                    if accelerator.is_main_process and len(eval_samples) > 0:
                        # Perform tag prediction on the collected samples
                        sample_batch_size = 20  # Number of samples to process per batch
                        predicted_tags_list = []
                        
                        for i in range(0, len(eval_samples), sample_batch_size):
                            batch_slice = eval_samples[i:i+sample_batch_size]
                            batch_x = torch.stack([sample['x'] for sample in batch_slice]).to(device)
                            
                            # Predict using Test-Time Augmentation (TTA)
                            if eval_tta:
                                # Multiple forward passes and average the results
                                n_augmentations = 5  # Number of augmentations
                                all_layer_predictions = [[] for _ in range(vae_n_layers)]
                                
                                for aug_idx in range(n_augmentations):
                                    # Add small noise to create different versions of the input
                                    if aug_idx > 0:  # First time use the original input
                                        noise_scale = 0.02 * aug_idx  # Gradually increase noise
                                        noise = torch.randn_like(batch_x) * noise_scale
                                        augmented_x = batch_x + noise
                                    else:
                                        augmented_x = batch_x
                                    
                                    # Predict on the augmented input
                                    with torch.no_grad():
                                        # Get semantic IDs and concatenated embeddings
                                        res = model.encode(augmented_x)
                                        embs = []  # Store embeddings for each layer
                                        
                                        # Predict for each layer
                                        for layer_idx, layer in enumerate(model.layers):
                                            # Get quantized embeddings
                                            quantized = layer(res, temperature=0.001)
                                            emb = quantized.embeddings
                                            
                                            # Add the current layer's embedding
                                            embs.append(emb)
                                            
                                            # Concatenate embeddings from the first layer_idx+1 layers
                                            concat_emb = torch.cat(embs, dim=-1)
                                            
                                            # Use the corresponding layer's tag predictor
                                            tag_logits = model.tag_predictors[layer_idx](concat_emb)
                                            
                                            # Adjust softmax temperature to increase confidence
                                            tag_logits = tag_logits / eval_temperature
                                            tag_probs = F.softmax(tag_logits, dim=-1)
                                            
                                            # Collect prediction results for this augmentation
                                            all_layer_predictions[layer_idx].append(tag_probs)
                                            
                                            # Update the residual
                                            res = res - emb
                                
                                # Average predictions from multiple augmentations
                                ensemble_predictions = []
                                for layer_idx in range(vae_n_layers):
                                    # Ensure the current layer has prediction results
                                    if len(all_layer_predictions[layer_idx]) > 0:
                                        try:
                                            # Average the prediction probabilities for each layer
                                            avg_probs = torch.stack(all_layer_predictions[layer_idx], dim=0).mean(dim=0)
                                            # Get the most likely class
                                            _, pred_indices = torch.max(avg_probs, dim=1)
                                            ensemble_predictions.append(pred_indices)
                                        except Exception as e:
                                            logger.warning(f"Layer {layer_idx} ensemble prediction failed: {str(e)}")
                                            # Create a default prediction (all zeros)
                                            default_pred = torch.zeros(batch_x.size(0), dtype=torch.long, device=device)
                                            ensemble_predictions.append(default_pred)
                                    else:
                                        # If there are no prediction results, add a default prediction
                                        logger.warning(f"Layer {layer_idx} has no predictions, using default.")
                                        default_pred = torch.zeros(batch_x.size(0), dtype=torch.long, device=device)
                                        ensemble_predictions.append(default_pred)
                                
                                # Ensure there are enough prediction results for stacking
                                if len(ensemble_predictions) == vae_n_layers:
                                    # Convert predicted tags to a tensor [batch_size, n_layers]
                                    batch_predicted_tags = torch.stack(ensemble_predictions, dim=1).cpu()
                                    
                                    # Add to the prediction list
                                    predicted_tags_list.append(batch_predicted_tags)
                                else:
                                    logger.warning(f"Incomplete ensemble predictions: expected {vae_n_layers} layers, got {len(ensemble_predictions)}")
                            else:
                                # Use standard prediction (no augmentation)
                                with torch.no_grad():
                                    # Get semantic IDs and concatenated embeddings
                                    res = model.encode(batch_x)
                                    predicted_tags = []
                                    
                                    embs = []  # Store embeddings for each layer
                                    
                                    # Predict for each layer
                                    for layer_idx, layer in enumerate(model.layers):
                                        # Get quantized embeddings
                                        quantized = layer(res, temperature=0.001)
                                        emb = quantized.embeddings
                                        
                                        # Add the current layer's embedding
                                        embs.append(emb)
                                        
                                        # Concatenate embeddings from the first layer_idx+1 layers
                                        concat_emb = torch.cat(embs, dim=-1)
                                        
                                        # Use the corresponding layer's tag predictor
                                        tag_logits = model.tag_predictors[layer_idx](concat_emb)
                                        
                                        # Get the predicted tag indices
                                        _, pred_indices = torch.max(tag_logits, dim=1)
                                        predicted_tags.append(pred_indices)
                                        
                                        # Update the residual
                                        res = res - emb
                                    
                                    # Convert predicted tags to a tensor [batch_size, n_layers]
                                    batch_predicted_tags = torch.stack(predicted_tags, dim=1).cpu()
                                
                                # Add to the prediction list
                                predicted_tags_list.append(batch_predicted_tags)
                        
                        # Concatenate prediction results from all batches
                        # Add a check to ensure the list is not empty
                        if predicted_tags_list:
                            all_predicted_tags = torch.cat(predicted_tags_list, dim=0)
                        else:
                            logger.warning("Predicted tags list is empty. This may be due to no valid evaluation samples or prediction failures.")
                            # Create an empty tensor as a placeholder
                            all_predicted_tags = torch.zeros((0, vae_n_layers), dtype=torch.long)
                        
                        # Calculate accuracy for each layer
                        layer_accuracies = []
                        
                        # Check if there are valid prediction samples
                        if len(eval_samples) > 0 and all_predicted_tags.size(0) > 0:
                            for layer_idx in range(vae_n_layers):
                                correct = 0
                                total = 0
                                
                                # Iterate through all samples
                                for i, sample in enumerate(eval_samples):
                                    if i >= all_predicted_tags.size(0):
                                        break
                                    
                                    # Ensure the sample contains a tag for the current layer
                                    if layer_idx < len(sample['true_tags']):
                                        try:
                                            true_tag = sample['true_tags'][layer_idx].item()
                                            pred_tag = all_predicted_tags[i, layer_idx].item()
                                            
                                            if true_tag >= 0:  # Only count valid tags
                                                total += 1
                                                if true_tag == pred_tag:
                                                    correct += 1
                                        except Exception as e:
                                            logger.warning(f"Error calculating accuracy: {str(e)}")
                                            continue
                                
                                accuracy = correct / max(1, total)  # Avoid division by zero
                                layer_accuracies.append(accuracy)
                                logger.info(f"Layer {layer_idx} Tag Prediction Accuracy: {accuracy:.4f} (Correct: {correct}/{total})")
                        else:
                            # If there are no valid prediction samples, log a warning
                            logger.warning("No valid prediction samples or results, cannot calculate accuracy.")
                            for _ in range(vae_n_layers):
                                layer_accuracies.append(0.0)
                        
                        # Print prediction results for samples
                        if len(eval_samples) > 0 and all_predicted_tags.size(0) > 0:
                            logger.info(f"\n==== Evaluation Set Sample Predictions (Showing {min(len(eval_samples), all_predicted_tags.size(0))} records) ====")
                            
                            for i, sample in enumerate(eval_samples):
                                if i >= all_predicted_tags.size(0) or i >= 99:  # Only display up to 100 records
                                    break
                                
                                true_tags = sample['true_tags']
                                pred_tags = all_predicted_tags[i]
                                
                                # Check if tag shapes match
                                if true_tags.size(0) != pred_tags.size(0):
                                    logger.warning(f"Shape mismatch for sample {i+1}: True={true_tags.size()}, Pred={pred_tags.size()}")
                                    continue
                                
                                # Format the output
                                sample_str = f"Sample {i+1}:\n"
                                
                                # Add input features (display only the first few values to avoid long output)
                                x_sample = sample['x']
                                x_preview = x_sample[:10].numpy()  # Show only the first 10 values
                                sample_str += f"  Input Features (first 10 values): {x_preview}\n"
                                
                                # Add true and predicted labels for each layer
                                for layer_idx in range(min(len(true_tags), len(pred_tags))):
                                    try:
                                        true_tag = true_tags[layer_idx].item()
                                        pred_tag = pred_tags[layer_idx].item()
                                        
                                        # Check if the tag is valid (non-negative)
                                        if true_tag >= 0:
                                            match_str = "✓" if true_tag == pred_tag else "✗"
                                            sample_str += f"  Layer {layer_idx + 1}: True Tag={true_tag}, Predicted Tag={pred_tag} {match_str}\n"
                                        else:
                                            sample_str += f"  Layer {layer_idx + 1}: True Tag=Invalid, Predicted Tag={pred_tag}\n"
                                    except Exception as e:
                                        logger.warning(f"Error processing sample {i+1}, layer {layer_idx}: {str(e)}")
                                        continue
                                
                                logger.info(sample_str)
                            
                            logger.info("==== End of Evaluation Set Sample Predictions ====\n")
                        else:
                            logger.warning("No valid prediction samples or results to display.")
                    
                    # Collect evaluation data for plotting
                    plot_data['eval_iterations'].append(iter+1)
                    plot_data['eval_total_loss'].append(eval_losses[0])
                    plot_data['eval_reconstruction_loss'].append(eval_losses[1])
                    plot_data['eval_rqvae_loss'].append(eval_losses[2])
                    plot_data['eval_tag_align_loss'].append(eval_losses[3])
                    plot_data['eval_tag_pred_loss'].append(eval_losses[4])
                    plot_data['eval_tag_pred_accuracy'].append(eval_losses[5])
                    
                    logger.info(f"Evaluation {iter+1} - Total Loss: {eval_losses[0]:.4f}, "
                                f"Reconstruction Loss: {eval_losses[1]:.4f}, "
                                f"RQVAE Loss: {eval_losses[2]:.4f}, "
                                f"Tag Alignment Loss: {eval_losses[3]:.4f}, "
                                f"Tag Prediction Loss: {eval_losses[4]:.4f}, "
                                f"Tag Prediction Accuracy: {eval_losses[5]:.4f}")
                    
                    # Print per-layer tag prediction accuracy
                    if eval_tta:
                        logger.info(f"TTA Per-layer Accuracy: {', '.join([f'Layer {i}: {acc:.4f}' for i, acc in enumerate(layer_accuracies)])}")
                    elif eval_model_output.tag_pred_accuracy_by_layer is not None:
                        eval_layer_acc_str = ", ".join([f"Layer {i}: {acc.cpu().item():.4f}" for i, acc in enumerate(eval_model_output.tag_pred_accuracy_by_layer)])
                        logger.info(f"Evaluation - Per-layer Tag Prediction Accuracy: {eval_layer_acc_str}")

                    # New model saving logic: only save models with accuracy > 0.60 and semantic ID repetition rate below the threshold
                    current_eval_accuracy = eval_losses[5]  # eval_tag_pred_accuracy
                    current_rqvae_loss = eval_losses[2]   # eval_rqvae_loss

                    # First calculate ID diversity metrics to ensure the sem_repetition_rate variable is defined
                    tokenizer.reset()
                    model.eval()
                    tokenizer.hrq_vae = accelerator.unwrap_model(model) # Ensure tokenizer uses the latest model

                    corpus_ids = tokenizer.precompute_corpus_ids(index_dataset)
                    max_duplicates = corpus_ids[:,-1].max() / corpus_ids.shape[0]
                    
                    # Fix: Define the cid variable here, using the index of the last layer
                    cid = vae_n_layers - 1  # Use the index of the last layer
                    _, counts = torch.unique(corpus_ids[:,cid], dim=0, return_counts=True)
                    p = counts / corpus_ids.shape[0]
                    rqvae_entropy = -(p*torch.log(p)).sum()

                    codebook_usage_info = []
                    for cid in range(vae_n_layers):
                        _, counts = torch.unique(corpus_ids[:,cid], return_counts=True)
                        usage = len(counts) / vae_codebook_size
                        id_diversity_log[f"codebook_usage_{cid}"] = usage
                        codebook_usage_info.append(f"Layer {cid}: {usage:.4f}")
                        
                        # Collect codebook usage data for plotting
                        plot_data['codebook_usage'][cid].append(usage)
                    
                    # New: Calculate and print the repetition rate for the semantic ID part
                    # Take only the first vae_n_layers dimensions (the semantic ID part)
                    semantic_ids = corpus_ids[:, :vae_n_layers]
                    sem_repetition_rate, sem_unique_items, sem_total_items = calculate_repetition_rate(semantic_ids)
                    logger.info(f"Repetition rate of semantic IDs only: {sem_repetition_rate:.4f} ({sem_unique_items} unique / {sem_total_items} total)")

                    plot_data['rqvae_entropy'].append(rqvae_entropy.cpu().item())
                    plot_data['max_id_duplicates'].append(max_duplicates.cpu().item())
                    
                    logger.info(f"ID Diversity {iter+1} - "
                                f"RQVAE Entropy: {rqvae_entropy.cpu().item():.4f}, "
                                f"Max ID Duplicates: {max_duplicates.cpu().item():.4f}, "
                                f"Codebook Usage: {', '.join(codebook_usage_info)}")

                    # Now it is safe to use the sem_repetition_rate variable
                    if current_eval_accuracy > 0.60 and sem_repetition_rate < id_repetition_threshold:
                        logger.info(f"Model accuracy and ID repetition rate have met the thresholds! "
                                    f"Accuracy: {current_eval_accuracy:.4f}, RQVAE Loss: {current_rqvae_loss:.4f}, "
                                    f"Semantic ID Repetition Rate: {sem_repetition_rate:.4f}")

                        # Save the new best model
                        model_save_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Ensure the save directory exists (save_dir is defined at the beginning of the function)
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
                            "sem_id_repetition_rate": sem_repetition_rate  # New: Log semantic ID repetition rate
                        }
                        
                        accelerator.save(state, new_model_path)
                        logger.info(f"Model saved to: {new_model_path}")
                    else:
                        if current_eval_accuracy <= 0.60:
                            logger.info(f"Current eval accuracy {current_eval_accuracy:.4f} did not reach the 0.60 threshold. Not saving model. Best accuracy remains {best_eval_accuracy:.4f}")
                        if sem_repetition_rate >= id_repetition_threshold:
                            logger.info(f"Current semantic ID repetition rate {sem_repetition_rate:.4f} is higher than the {id_repetition_threshold} threshold. Not saving model.")

            pbar.update(1)
    
    # Plot final charts at the end of training
    if accelerator.is_main_process:
        logger.info("Training complete. Generating training process charts...")
        # Plot all charts
        plot_all_metrics(plot_data, plots_dir, vae_n_layers)
        logger.info(f"All charts have been saved to {plots_dir}")


def plot_all_metrics(plot_data, plots_dir, n_layers):
    """Plots the overall training progress for all metrics."""
    
    # 1. Plot training losses
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
    
    # 2. Plot tag prediction accuracy
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
    
    # 3. Plot embedding norms
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
    
    # 4. Plot codebook usage
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
    
    # 5. Plot ID diversity metrics
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
