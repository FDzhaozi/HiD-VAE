import torch
import torch.nn.functional as F

from data.schemas import SeqBatch, HRqVaeOutput, HRqVaeComputedLosses
from einops import rearrange
from functools import cached_property
from modules.encoder import MLP
from modules.loss import CategoricalReconstuctionLoss
from modules.loss import ReconstructionLoss
from modules.loss import QuantizeLoss
from modules.loss import TagAlignmentLoss
from modules.loss import TagPredictionLoss
from modules.normalize import l2norm
from modules.quantize import Quantize
from modules.quantize import QuantizeForwardMode
from huggingface_hub import PyTorchModelHubMixin
from typing import List, Dict, Optional
from torch import nn
from torch import Tensor

torch.set_float32_matmul_precision('high')


# Add a new loss function class to calculate the semantic ID uniqueness constraint loss
class SemanticIdUniquenessLoss(nn.Module):
    """
    Calculates the semantic ID uniqueness constraint loss, pushing the semantic IDs of different items apart.
    """
    def __init__(self, margin: float = 0.5, weight: float = 1.0):
        """
        Initializes the semantic ID uniqueness constraint loss.
        
        Args:
            margin: The minimum distance threshold between semantic IDs.
            weight: The weight of the loss.
        """
        super().__init__()
        self.margin = margin
        self.weight = weight
    
    def forward(self, sem_ids: Tensor, encoded_features: Tensor) -> Tensor:
        """
        Calculates the uniqueness loss for semantic IDs within a batch.
        
        Args:
            sem_ids: A tensor of semantic IDs with shape [batch_size, n_layers].
            encoded_features: The encoder output with shape [batch_size, embed_dim].
            
        Returns:
            The uniqueness constraint loss.
        """
        batch_size, n_layers = sem_ids.shape
        
        # If the batch size is too small, do not calculate the loss.
        if batch_size <= 1:
            return torch.tensor(0.0, device=sem_ids.device)
        
        # Find pairs with identical semantic IDs.
        # Expand to [batch_size, 1, n_layers] and [1, batch_size, n_layers].
        id1 = sem_ids.unsqueeze(1)
        id2 = sem_ids.unsqueeze(0)
        
        # Check if all layers are equal.
        id_eq = (id1 == id2).all(dim=-1)
        
        # Create a diagonal mask to exclude self-comparison.
        diag_mask = ~torch.eye(batch_size, device=sem_ids.device, dtype=torch.bool)
        
        # Find pairs of identical IDs (excluding self).
        identical_pairs_mask = id_eq & diag_mask
        
        # If there are no identical ID pairs, return zero loss.
        if not identical_pairs_mask.any():
            return torch.tensor(0.0, device=sem_ids.device)
        
        # Get the indices of the identical ID pairs.
        idx_a, idx_b = torch.where(identical_pairs_mask)
        
        # To avoid duplicate calculations, only consider pairs where i < j.
        unique_pairs_mask = idx_a < idx_b
        idx_a = idx_a[unique_pairs_mask]
        idx_b = idx_b[unique_pairs_mask]

        if len(idx_a) == 0:
            return torch.tensor(0.0, device=sem_ids.device)
            
        # Get the encoded features for these pairs.
        features_a = encoded_features[idx_a]
        features_b = encoded_features[idx_b]
        
        # Normalize features to calculate cosine similarity.
        features_a_norm = F.normalize(features_a, p=2, dim=-1)
        features_b_norm = F.normalize(features_b, p=2, dim=-1)

        # Calculate cosine similarity.
        cosine_sim = (features_a_norm * features_b_norm).sum(dim=-1)
        
        # Calculate the loss: we want to push these features apart, so we apply a penalty when the similarity is higher than the margin.
        # Loss = max(0, cosine_sim - margin).
        loss = F.relu(cosine_sim - self.margin)

        # Average the loss over all conflicting pairs and multiply by the weight.
        uniqueness_loss = self.weight * loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=sem_ids.device)

        return uniqueness_loss


class TagPredictor(nn.Module):
    """
    Tag predictor for predicting the tag index for each layer.
    """
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        layer_idx: int = 0
    ) -> None:
        super().__init__()
        
        # If hidden_dim is not specified, use twice the embed_dim.
        if hidden_dim is None:
            hidden_dim = embed_dim * 2
        
        # Adjust the dropout rate based on the layer index; deeper layers have a higher dropout rate.
        # However, don't make it too high to avoid excessive information loss.
        dropout_rate = min(0.55, dropout_rate + layer_idx * 0.075)
        
        # Self-attention mechanism - helps the model focus on important parts of the input features.
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.Sigmoid()
        )
        
        # Build a deeper and wider classifier network with residual connections.
        # Part 1: Feature Extraction.
        self.feature_extractor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Intermediate layer dimension.
        mid_dim = int(hidden_dim * 0.9)
        
        # Part 2: Residual Block.
        self.residual_block1 = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mid_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim) if use_batch_norm else nn.Identity(),
        )
        
        # Part 3: Residual Block 2.
        self.residual_block2 = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mid_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim) if use_batch_norm else nn.Identity(),
        )
        
        # Output layer.
        classifier_mid_dim = mid_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, classifier_mid_dim),
            nn.LayerNorm(classifier_mid_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_mid_dim, classifier_mid_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(classifier_mid_dim // 2, num_classes)
        )
        
        # Label smoothing regularization parameter.
        self.label_smoothing = 0.1 if layer_idx > 0 else 0.05
        
        # Feature normalization.
        self.apply_norm = layer_idx > 0  # Apply normalization to deeper layers.
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input embeddings with shape [batch_size, embed_dim].
            
        Returns:
            logits: Predicted logits with shape [batch_size, num_classes].
        """
        # Apply self-attention mechanism.
        attention_weights = self.attention(x)
        x_attended = x * attention_weights
        
        # Feature normalization (optional).
        if self.apply_norm:
            x_attended = F.normalize(x_attended, p=2, dim=-1)
        
        # Feature extraction.
        features = self.feature_extractor(x_attended)
        
        # Apply residual blocks.
        res1 = self.residual_block1(features)
        features = features + res1
        
        res2 = self.residual_block2(features)
        features = features + res2
        
        # Classification.
        logits = self.classifier(features)
        
        return logits


class HRqVae(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        codebook_kmeans_init: bool = True,
        codebook_normalize: bool = False,
        codebook_sim_vq: bool = False,
        codebook_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,
        n_layers: int = 3,
        commitment_weight: float = 0.25,
        n_cat_features: int = 18,
        tag_alignment_weight: float = 0.5,
        tag_prediction_weight: float = 0.5,
        tag_class_counts: Optional[List[int]] = None,
        tag_embed_dim: int = 768,  # Tag embedding dimension
        use_focal_loss: bool = False,  # Whether to use focal loss
        focal_loss_params: Optional[Dict] = None,  # Focal loss parameters
        dropout_rate: float = 0.2,  # New: Dropout rate
        use_batch_norm: bool = True,  # New: Whether to use BatchNorm
        alignment_temperature: float = 0.1,  # New: Contrastive learning temperature parameter
        # New: Semantic ID uniqueness constraint parameters
        sem_id_uniqueness_weight: float = 0.5,  # Semantic ID uniqueness constraint weight
        sem_id_uniqueness_margin: float = 0.5,  # Semantic ID uniqueness constraint margin
    ) -> None:
        self._config = locals()
        
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.n_cat_feats = n_cat_features
        self.tag_alignment_weight = tag_alignment_weight
        self.tag_prediction_weight = tag_prediction_weight
        self.tag_embed_dim = tag_embed_dim  # Save the tag embedding dimension
        self.use_focal_loss = use_focal_loss  # Whether to use focal loss
        self.focal_loss_params = focal_loss_params or {'gamma': 2.0}  # Focal loss parameters
        self.dropout_rate = dropout_rate  # New: Dropout rate
        self.use_batch_norm = use_batch_norm  # New: Whether to use BatchNorm
        self.alignment_temperature = alignment_temperature  # New: Contrastive learning temperature parameter
        self.sem_id_uniqueness_weight = sem_id_uniqueness_weight  # New: Semantic ID uniqueness constraint weight
        
        # If tag_class_counts is not provided, use default values
        if tag_class_counts is None:
            # Default number of tag classes for each layer, can be adjusted based on actual needs
            self.tag_class_counts = [10, 100, 1000][:n_layers]
        else:
            self.tag_class_counts = tag_class_counts[:n_layers]
        
        # Ensure the number of tag classes matches the number of layers
        assert len(self.tag_class_counts) == n_layers, f"Number of tag classes {len(self.tag_class_counts)} does not match number of layers {n_layers}"

        # Quantization layers
        self.layers = nn.ModuleList(modules=[
            Quantize(
                embed_dim=embed_dim,
                n_embed=codebook_size,
                forward_mode=codebook_mode,
                do_kmeans_init=codebook_kmeans_init,
                codebook_normalize=i == 0 and codebook_normalize,
                sim_vq=codebook_sim_vq,
                commitment_weight=commitment_weight
            ) for i in range(n_layers)
        ])

        # Tag predictors
        # New: Concatenated embedding dimension for each layer
        self.concat_embed_dims = [(embed_dim * (i + 1)) for i in range(n_layers)]

        # Tag predictors (input dimension becomes the concatenated dimension)
        # Compatibility handling: The tag_class_counts of the weights to be loaded may differ from the current configuration
        # Based on error messages, the stored values are [7, 30, 97]. If the current configuration is different, use the stored values.
        # This ensures that the weights can be loaded successfully.
        self._stored_tag_class_counts = None
        self.tag_predictors = nn.ModuleList(modules=[
            TagPredictor(
                embed_dim=self.concat_embed_dims[i],
                num_classes=self.tag_class_counts[i],
                hidden_dim=hidden_dims[0] // 2 * (i + 1),  # Adjust hidden dimension based on layer index
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                layer_idx=i
            ) for i in range(n_layers)
        ])
        
        # Tag embedding projection layers (output dimension becomes the concatenated dimension)
        self.tag_projectors = nn.ModuleList(modules=[
            nn.Sequential(
                nn.Linear(tag_embed_dim, hidden_dims[0]),
                nn.BatchNorm1d(hidden_dims[0]) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dims[0], self.concat_embed_dims[i]),
                nn.LayerNorm(self.concat_embed_dims[i]) if codebook_normalize else nn.Identity()
            ) for i in range(n_layers)
        ])
        
        # Encoder and Decoder
        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            out_dim=embed_dim,
            normalize=codebook_normalize
        )

        self.decoder = MLP(
            input_dim=embed_dim,
            hidden_dims=hidden_dims[-1::-1],
            out_dim=input_dim,
            normalize=True
        )

        # Loss functions
        self.reconstruction_loss = (
            CategoricalReconstuctionLoss(n_cat_features) if n_cat_features != 0
            else ReconstructionLoss()
        )
        self.tag_alignment_loss = TagAlignmentLoss(
            alignment_weight=tag_alignment_weight,
            temperature=alignment_temperature
        )
        
        # Initialize tag prediction loss, with support for focal loss
        self.tag_prediction_loss = TagPredictionLoss(
            use_focal_loss=use_focal_loss,
            focal_params=focal_loss_params,
            class_counts=None  # Will be updated during the training process
        )
        
        # Add semantic ID uniqueness constraint loss
        self.sem_id_uniqueness_loss = SemanticIdUniquenessLoss(
            margin=sem_id_uniqueness_margin,
            weight=sem_id_uniqueness_weight
        )
        
        # For storing class frequency statistics
        self.register_buffer('class_freq_counts', None)

    @cached_property
    def config(self) -> dict:
        return self._config
    
    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device
    
    def load_pretrained(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        
        # Check for mismatches between the weight file and the current model structure.
        model_dict = self.state_dict()
        pretrained_dict = state["model"]
        
        # Check for mismatches related to the tag predictors.
        tag_predictor_mismatch = False
        tag_class_counts_from_weights = []
        
        for i in range(self.n_layers):
            weight_key = f"tag_predictors.{i}.classifier.7.weight"
            if weight_key in pretrained_dict:
                pretrained_shape = pretrained_dict[weight_key].shape
                current_shape = model_dict[weight_key].shape if weight_key in model_dict else None
                
                if current_shape is not None and pretrained_shape[0] != current_shape[0]:
                    tag_predictor_mismatch = True
                    tag_class_counts_from_weights.append(pretrained_shape[0])
                else:
                    # If weights for the corresponding layer are not found or the dimensions match, keep the current number of classes.
                    tag_class_counts_from_weights.append(self.tag_class_counts[i])
        
        # Check for unexpected keys related to tag_projectors.
        tag_projector_mismatch = False
        for i in range(self.n_layers):
            projector_key = f"tag_projectors.{i}.5.weight"
            if projector_key in pretrained_dict and projector_key not in model_dict:
                tag_projector_mismatch = True
                break
        
        # If there is a tag predictor mismatch, recreate the tag predictors to match the weight file.
        if tag_predictor_mismatch and len(tag_class_counts_from_weights) == self.n_layers:
            print(f"Tag predictor mismatch detected. Adjusting number of classes from {self.tag_class_counts} to {tag_class_counts_from_weights}")
            self._stored_tag_class_counts = self.tag_class_counts  # Store the original values.
            self.tag_class_counts = tag_class_counts_from_weights  # Update with values from the weight file.
            
            # Recreate the tag predictors.
            self.tag_predictors = nn.ModuleList(modules=[
                TagPredictor(
                    embed_dim=self.concat_embed_dims[i],
                    num_classes=self.tag_class_counts[i],
                    hidden_dim=self._config.get('hidden_dims', [512, 256, 128])[0] // 2 * (i + 1),
                    dropout_rate=self._config.get('dropout_rate', 0.2),
                    use_batch_norm=self._config.get('use_batch_norm', True),
                    layer_idx=i
                ) for i in range(self.n_layers)
            ])
        
        # If there is a tag_projectors mismatch, recreate them to match the weight file.
        if tag_projector_mismatch:
            print(f"Tag projector mismatch detected. Adjusting structure to match the weight file.")
            # Recreate tag_projectors, including the LayerNorm layer.
            self.tag_projectors = nn.ModuleList(modules=[
                nn.Sequential(
                    nn.Linear(self.tag_embed_dim, self._config.get('hidden_dims', [512, 256, 128])[0]),
                    nn.BatchNorm1d(self._config.get('hidden_dims', [512, 256, 128])[0]) if self._config.get('use_batch_norm', True) else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(self._config.get('dropout_rate', 0.2)),
                    nn.Linear(self._config.get('hidden_dims', [512, 256, 128])[0], self.concat_embed_dims[i]),
                    nn.LayerNorm(self.concat_embed_dims[i])  # Add LayerNorm layer, regardless of the codebook_normalize setting.
                ) for i in range(self.n_layers)
            ])
        
        # Handle remaining mismatched keys.
        # Filter out keys that are in the pretrained model but not in the current model.
        pretrained_dict_filtered = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        # If the number of keys in the filtered dictionary is less than the original, print a warning.
        if len(pretrained_dict_filtered) < len(pretrained_dict):
            skipped_keys = set(pretrained_dict.keys()) - set(pretrained_dict_filtered.keys())
            print(f"Warning: The following keys were skipped during loading as they do not exist in the current model structure: {skipped_keys}")
        
        try:
            # Try to load the filtered weights.
            model_dict.update(pretrained_dict_filtered)
            self.load_state_dict(model_dict)
            print(f"---Loaded HRQVAE Iter {state['iter']}---")
        except Exception as e:
            # If it still fails, try loading with strict=False.
            print(f"Standard loading failed, trying to load with strict=False: {str(e)}")
            self.load_state_dict(pretrained_dict, strict=False)
            print(f"---Loaded HRQVAE Iter {state['iter']} (strict=False)---")
        
        # If there were mismatches, print warnings.
        if tag_predictor_mismatch:
            print(f"Warning: Tag predictors were automatically adjusted to match the weight file. Original class counts: {self._stored_tag_class_counts}, Adjusted: {self.tag_class_counts}")
        if tag_projector_mismatch:
            print(f"Warning: Tag projectors were automatically adjusted to match the weight file.")

    def encode(self, x: Tensor) -> Tensor:
        # Ensure input data is float32 to match model weights.
        x = x.float()
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def get_semantic_ids(
        self,
        encoded_x: Tensor,
        tags_emb: Optional[Tensor] = None,
        tags_indices: Optional[Tensor] = None,
        gumbel_t: float = 0.001
    ) -> HRqVaeOutput:
        """
        Get semantic IDs and compute related losses.
        
        Args:
            encoded_x: Encoded input features.
            tags_emb: Tag embeddings, shape [batch_size, n_layers, tag_embed_dim].
            tags_indices: Tag indices, shape [batch_size, n_layers].
            gumbel_t: Gumbel softmax temperature parameter.
            
        Returns:
            HRqVaeOutput containing embeddings, residuals, semantic IDs, and various losses.
        """
        res = encoded_x
        
        # Initialize losses.
        quantize_loss = torch.tensor(0.0, device=encoded_x.device)  # Modification: Initialize as a tensor.
        tag_align_loss = torch.tensor(0.0, device=encoded_x.device)
        tag_pred_loss = torch.tensor(0.0, device=encoded_x.device)
        tag_pred_accuracy = torch.tensor(0.0, device=encoded_x.device)
        
        embs, residuals, sem_ids = [], [], []
        
        # Add lists to collect per-layer losses.
        tag_align_loss_by_layer = []
        tag_pred_loss_by_layer = []
        tag_pred_accuracy_by_layer = []
        
        for i, layer in enumerate(self.layers):
            residuals.append(res)
            quantized = layer(res, temperature=gumbel_t)
            quantize_loss = quantize_loss + quantized.loss  # Modification: Ensure losses are accumulated.
            emb, id = quantized.embeddings, quantized.ids
            
            # Add the embedding and id of the current layer.
            embs.append(emb)
            sem_ids.append(id)
            
            # Concatenate the embeddings of the first i+1 layers.
            concat_emb = torch.cat(embs, dim=-1)  # [batch, (i+1)*embed_dim]
            
            # If tag embeddings and indices are provided, calculate tag alignment and prediction losses.
            if tags_emb is not None and tags_indices is not None:
                # Get the tag embedding and index for the current layer.
                current_tag_emb = tags_emb[:, i]
                current_tag_idx = tags_indices[:, i]
                
                # Use the projection layer to project the tag embedding to the same dimension as the concatenated codebook embedding.
                projected_tag_emb = self.tag_projectors[i](current_tag_emb)
                
                # Calculate tag alignment loss.
                align_loss = self.tag_alignment_loss(concat_emb, projected_tag_emb, i)
                tag_align_loss += align_loss.mean()
                tag_align_loss_by_layer.append(align_loss.mean())
                
                # Predict tag indices.
                tag_logits = self.tag_predictors[i](concat_emb)
                pred_loss, pred_accuracy = self.tag_prediction_loss(tag_logits, current_tag_idx)
                
                tag_pred_loss += pred_loss
                tag_pred_accuracy += pred_accuracy
                tag_pred_loss_by_layer.append(pred_loss)
                tag_pred_accuracy_by_layer.append(pred_accuracy)
            
            # Update the residual.
            res = res - emb
        
        # If no tag data is provided, set tag-related losses to 0.
        if tags_emb is None or tags_indices is None:
            tag_align_loss = torch.tensor(0.0, device=encoded_x.device)
            tag_pred_loss = torch.tensor(0.0, device=encoded_x.device)
            tag_pred_accuracy = torch.tensor(0.0, device=encoded_x.device)
        else:
            # Calculate average loss and accuracy.
            tag_align_loss = tag_align_loss / self.n_layers
            tag_pred_loss = tag_pred_loss / self.n_layers
            tag_pred_accuracy = tag_pred_accuracy / self.n_layers
            
            # Convert per-layer losses and accuracies to tensors.
            tag_align_loss_by_layer = torch.stack(tag_align_loss_by_layer) if tag_align_loss_by_layer else None
            tag_pred_loss_by_layer = torch.stack(tag_pred_loss_by_layer) if tag_pred_loss_by_layer else None
            tag_pred_accuracy_by_layer = torch.stack(tag_pred_accuracy_by_layer) if tag_pred_accuracy_by_layer else None

        # Return the result.
        return HRqVaeOutput(
            embeddings=rearrange(embs, "b h d -> h d b"),
            residuals=rearrange(residuals, "b h d -> h d b"),
            sem_ids=rearrange(sem_ids, "b d -> d b"),
            quantize_loss=quantize_loss,
            tag_align_loss=tag_align_loss,
            tag_pred_loss=tag_pred_loss,
            tag_pred_accuracy=tag_pred_accuracy,
            # Add the following three new attributes.
            tag_align_loss_by_layer=tag_align_loss_by_layer,
            tag_pred_loss_by_layer=tag_pred_loss_by_layer,
            tag_pred_accuracy_by_layer=tag_pred_accuracy_by_layer
        )

    @torch.compile(mode="reduce-overhead")
    def forward(self, batch: SeqBatch, gumbel_t: float = 1.0) -> HRqVaeComputedLosses:
        x = batch.x
        
        # Get tag embeddings and indices (if available).
        tags_emb = getattr(batch, 'tags_emb', None)
        tags_indices = getattr(batch, 'tags_indices', None)
        
        # Ensure input data is float32 to match model weights.
        x = x.float()
        if tags_emb is not None:
            tags_emb = tags_emb.float()
        
        # Encode input features.
        encoded_features = self.encode(x)

        # Get semantic IDs and related losses.
        quantized = self.get_semantic_ids(encoded_features, tags_emb, tags_indices, gumbel_t)
        
        
        embs, residuals = quantized.embeddings, quantized.residuals
        # Decode, the input is the sum of embeddings from all layers, the output is the reconstructed feature.
        x_hat = self.decode(embs.sum(axis=-1))  
        # print(f"x_hat shape: {x_hat.shape}")
        # Fix: Handle concatenation of categorical features.
        x_hat = torch.cat([l2norm(x_hat[...,:-self.n_cat_feats]), x_hat[...,-self.n_cat_feats:]], axis=-1)
        # print(f"x_hat shape: {x_hat.shape}")
        # Calculate reconstruction loss.
        reconstuction_loss = self.reconstruction_loss(x_hat, x)
        # print(f"Reconstruction loss: {reconstuction_loss.mean().item():.4f}")
        
        # Calculate total loss.
        rqvae_loss = quantized.quantize_loss
        tag_align_loss = quantized.tag_align_loss
        tag_pred_loss = quantized.tag_pred_loss
        tag_pred_accuracy = quantized.tag_pred_accuracy
        # print(f"RQVAE loss: {rqvae_loss.mean().item():.4f}")
        # print(f"Tag alignment loss: {tag_align_loss.mean().item():.4f}")
        # print(f"Tag prediction loss: {tag_pred_loss.mean().item():.4f}")
        # print(f"Tag prediction accuracy: {tag_pred_accuracy.mean().item():.4f}")
        
        # New: Calculate semantic ID uniqueness constraint loss.
        # Fix: Correctly handle tensor shape transformation.
        # The shape of quantized.sem_ids is [n_layers, batch_size].
        # We need to convert it to [batch_size, n_layers].
        sem_ids_tensor = quantized.sem_ids.transpose(0, 1)  # Direct transpose, the result is [batch_size, n_layers].
        sem_id_uniqueness_loss = self.sem_id_uniqueness_loss(sem_ids_tensor, encoded_features)
        
        # Total loss = Reconstruction loss + RQVAE loss + Tag alignment loss + Tag prediction loss + Semantic ID uniqueness constraint loss.
        loss = (
            reconstuction_loss.mean() +  
            rqvae_loss.mean() +  
            self.tag_alignment_weight * tag_align_loss +  
            self.tag_prediction_weight * tag_pred_loss +
            self.sem_id_uniqueness_weight * sem_id_uniqueness_loss  # New: Semantic ID uniqueness constraint loss.
        )
        # print(f"Total loss: {loss.item():.4f}")

        with torch.no_grad():
            # Calculate debug ID statistics.
            embs_norm = embs.norm(dim=1)
            p_unique_ids = (~torch.triu(
                (rearrange(quantized.sem_ids, "b d -> b 1 d") == rearrange(quantized.sem_ids, "b d -> 1 b d")).all(axis=-1), diagonal=1)
            ).all(axis=1).sum() / quantized.sem_ids.shape[0]

        
        # Directly use the per-layer losses from quantized.
        tag_align_loss_by_layer = quantized.tag_align_loss_by_layer
        tag_pred_loss_by_layer = quantized.tag_pred_loss_by_layer
        tag_pred_accuracy_by_layer = quantized.tag_pred_accuracy_by_layer
        
        # Add the semantic ID uniqueness constraint loss to the return result.
        return HRqVaeComputedLosses(
            loss=loss,
            reconstruction_loss=reconstuction_loss,
            rqvae_loss=rqvae_loss,
            tag_align_loss=tag_align_loss,
            tag_pred_loss=tag_pred_loss,
            tag_pred_accuracy=tag_pred_accuracy,
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
            # New: per-layer losses and accuracy.
            tag_align_loss_by_layer=tag_align_loss_by_layer,
            tag_pred_loss_by_layer=tag_pred_loss_by_layer,
            tag_pred_accuracy_by_layer=tag_pred_accuracy_by_layer,
            # New: semantic ID uniqueness constraint loss.
            sem_id_uniqueness_loss=sem_id_uniqueness_loss
        )
    
    def predict_tags(self, x: Tensor, gumbel_t: float = 0.001) -> Dict[str, Tensor]:
        """
        Predicts the tag indices corresponding to the input features.
        
        Args:
            x: Input features, can be shape [batch_size, feature_dim] or [batch_size, seq_len, feature_dim].
            gumbel_t: Gumbel softmax temperature parameter.
            
        Returns:
            A dictionary containing the predicted tag indices and their confidences.
        """
        # Check input dimensions and handle sequence data.
        original_shape = x.shape
        if len(original_shape) == 3:
            # Input is sequence data [batch_size, seq_len, feature_dim].
            batch_size, seq_len, feature_dim = original_shape
            # Flatten the sequence to [batch_size*seq_len, feature_dim].
            x = x.reshape(-1, feature_dim)
            is_sequence = True
        else:
            # Input is already [batch_size, feature_dim].
            is_sequence = False
            
        # Encode input features.
        res = self.encode(x)
        print(f"Shape of encoded features: {res.shape}")
        
        tag_predictions = []
        tag_confidences = []
        embs = []  # Store embeddings for each layer for concatenation.
        
        # Predict for each layer.
        for i, layer in enumerate(self.layers):
            # Get quantized embeddings.
            quantized = layer(res, temperature=gumbel_t)
            emb = quantized.embeddings
            
            # Add current layer's embedding.
            embs.append(emb)
            
            # Concatenate embeddings of the first i+1 layers.
            concat_emb = torch.cat(embs, dim=-1)
            
            # Use the concatenated embedding to predict tags.
            tag_logits = self.tag_predictors[i](concat_emb)
            tag_probs = torch.softmax(tag_logits, dim=-1)
            
            # Get the most likely tag index and its confidence.
            confidence, prediction = torch.max(tag_probs, dim=-1)
            
            tag_predictions.append(prediction)
            tag_confidences.append(confidence)
            
            # Update the residual.
            res = res - emb
        
        # Reshape the prediction results back to the original sequence shape (if the input was a sequence).
        if is_sequence:
            tag_predictions = [pred.reshape(batch_size, seq_len) for pred in tag_predictions]
            tag_confidences = [conf.reshape(batch_size, seq_len) for conf in tag_confidences]
        
        return {
            "predictions": torch.stack(tag_predictions, dim=-1),  # For sequence: [batch_size, seq_len, n_layers].
            "confidences": torch.stack(tag_confidences, dim=-1)   # For sequence: [batch_size, seq_len, n_layers].
        }

    def update_class_counts(self, class_counts_dict):
        """
        Updates the class frequency counts.
        
        Args:
            class_counts_dict: A dictionary containing class counts for each layer.
        """
        # Convert the dictionary to a module dictionary instead of direct assignment.
        for layer_idx, counts in class_counts_dict.items():
            # Ensure counts is a tensor.
            if not isinstance(counts, torch.Tensor):
                counts = torch.tensor(counts, device=self.device)
            # Use register_buffer to dynamically register, or update an existing buffer.
            self.register_buffer(f'class_freq_counts_{layer_idx}', counts)
        
        # Store the list of layer indices for later use.
        self.class_freq_layers = list(class_counts_dict.keys())
