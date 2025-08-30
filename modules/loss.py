from torch import nn
from torch import Tensor
import torch
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        return ((x_hat - x)**2).sum(axis=-1)


class CategoricalReconstructionLoss(nn.Module):
    def __init__(self, n_cat_feats: int) -> None:
        super().__init__()
        self.reconstruction_loss = ReconstructionLoss()
        self.n_cat_feats = n_cat_feats

    def forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        reconstr = self.reconstruction_loss(
            x_hat[:, :-self.n_cat_feats],
            x[:, :-self.n_cat_feats]
        )
        if self.n_cat_feats > 0:
            cat_reconstr = nn.functional.binary_cross_entropy_with_logits(
                x_hat[:, -self.n_cat_feats:],
                x[:, -self.n_cat_feats:],
                reduction='none'
            ).sum(axis=-1)
            reconstr += cat_reconstr
        return reconstr


class QuantizeLoss(nn.Module):
    def __init__(self, commitment_weight: float = 1.0) -> None:
        super().__init__()
        self.commitment_weight = commitment_weight

    def forward(self, query: Tensor, value: Tensor) -> Tensor:
        emb_loss = ((query.detach() - value)**2).sum(axis=[-1])
        query_loss = ((query - value.detach())**2).sum(axis=[-1])
        return emb_loss + self.commitment_weight * query_loss


# Modified Tag Alignment Loss function using InfoNCE contrastive learning loss
class TagAlignmentLoss(nn.Module):
    def __init__(self, alignment_weight: float = 1.0, temperature: float = 0.1) -> None:
        super().__init__()
        self.alignment_weight = alignment_weight
        self.temperature = temperature

    def forward(self, codebook_emb: Tensor, tag_emb: Tensor, layer_idx: int) -> Tensor:
        """
        Computes the contrastive learning loss (InfoNCE) between codebook embeddings and tag embeddings.

        Args:
            codebook_emb: Codebook embeddings of shape [batch_size, embed_dim].
            tag_emb: Tag embeddings of shape [batch_size, embed_dim].
            layer_idx: The index of the current layer, used to adjust alignment strategy for different layers.
        """
        batch_size = codebook_emb.size(0)

        # Normalize the embedding vectors
        codebook_emb_norm = F.normalize(codebook_emb, p=2, dim=-1)
        tag_emb_norm = F.normalize(tag_emb, p=2, dim=-1)

        # Calculate dot products
        logits = torch.matmul(codebook_emb_norm, tag_emb_norm.transpose(0, 1)) / self.temperature

        # The diagonal elements are positive samples
        positive_logits = torch.diag(logits)

        # InfoNCE loss calculation
        labels = torch.arange(batch_size, device=codebook_emb.device)
        loss = F.cross_entropy(logits, labels)

        # Adjust weight based on layer index
        layer_weight = 1.0 / ((layer_idx * 0.5) + 1)  # Modified: Adjusted layer weighting strategy

        # Total loss = InfoNCE loss. A previously included potentially negative term has been removed.
        total_loss = loss * self.alignment_weight * layer_weight

        return total_loss


# Improved Tag Prediction Loss function with more flexible focal loss parameter adjustment
class TagPredictionLoss(nn.Module):
    def __init__(self, use_focal_loss: bool = False, focal_params: dict = None, class_counts: dict = None) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.focal_params = focal_params or {'gamma': 2.0, 'alpha': 0.25}
        self.class_counts = class_counts  # Frequency statistics for each class in each layer

        # Add label smoothing parameters
        self.use_label_smoothing = True
        self.label_smoothing_alpha = 0.1

        # Additional strategy to mitigate overfitting
        self.use_mixup = True
        self.mixup_alpha = 0.2  # Controls the degree of mixing

        # Weight adjuster for learning rate decay
        self.weight_scheduler = None

    def forward(self, pred_logits: Tensor, target_indices: Tensor, layer_idx: int = 0) -> tuple:
        """
        Computes the tag prediction loss and accuracy, with support for focal loss.

        Args:
            pred_logits: Predicted logits of shape [batch_size, num_classes].
            target_indices: Target tag indices of shape [batch_size].
            layer_idx: The index of the current layer, used to get focal loss parameters for that layer.

        Returns:
            loss: Cross-entropy loss or focal loss.
            accuracy: Prediction accuracy.
        """
        # Ensure target indices are valid (not -1)
        valid_mask = (target_indices >= 0)

        if valid_mask.sum() == 0:
            # If there are no valid targets, return zero loss and accuracy
            return torch.tensor(0.0, device=pred_logits.device), torch.tensor(0.0, device=pred_logits.device)

        # Only compute loss for valid targets
        valid_logits = pred_logits[valid_mask]
        valid_targets = target_indices[valid_mask]

        # Calculate accuracy
        pred_indices = torch.argmax(valid_logits, dim=-1)
        accuracy = (pred_indices == valid_targets).float().mean()

        # Get the predicted probability distribution from the model
        probs = F.softmax(valid_logits, dim=-1)

        # Apply Mixup strategy - randomly mix samples to enhance generalization
        if self.use_mixup and valid_logits.size(0) > 1:
            # Only use mixup during the training phase
            if valid_logits.requires_grad:
                batch_size = valid_logits.size(0)
                # Create a randomly permuted index
                indices = torch.randperm(batch_size, device=valid_logits.device)
                # Generate mixup weights
                lam = torch.distributions.Beta(torch.tensor(self.mixup_alpha),
                                               torch.tensor(self.mixup_alpha)).sample().to(valid_logits.device)

                # Mix logits and targets
                mixed_logits = lam * valid_logits + (1 - lam) * valid_logits[indices]
                valid_logits = mixed_logits

                # Save original and mixed targets for loss calculation
                targets_a, targets_b = valid_targets, valid_targets[indices]

        # Choose different loss calculation methods based on whether focal loss is used
        if self.use_focal_loss:
            # Get focal loss parameters for the current layer, using higher gamma for deeper layers
            gamma = self.focal_params.get(f'gamma_{layer_idx}', self.focal_params.get('gamma', 2.0)) * (1 + 0.35 * layer_idx)

            # Smaller class balancing factor to avoid over-compensation
            alpha = max(0.08, self.focal_params.get(f'alpha_{layer_idx}', self.focal_params.get('alpha', 0.25)) - 0.06 * layer_idx)

            # If class frequency statistics are available, use dynamic weights
            if self.class_counts is not None and layer_idx in self.class_counts:
                class_counts = self.class_counts[layer_idx]
                if isinstance(class_counts, torch.Tensor) and class_counts.numel() > 0:
                    # Calculate class weights: lower frequency means higher weight
                    class_freq = class_counts.float() / class_counts.sum()
                    # Prevent division by zero
                    class_freq = torch.clamp(class_freq, min=1e-6)
                    # Weights are inversely proportional to frequency, emphasizing rare classes but avoiding extreme weights
                    weights = 1.0 / torch.sqrt(class_freq)
                    # Normalize weights but reduce extreme values
                    weights = torch.clamp(weights / weights.mean(), min=0.5, max=3.0)
                    # Ensure the weight tensor is on the same device as the logits
                    weights = weights.to(valid_logits.device)

                    # Use class weights to compute focal loss, while applying label smoothing
                    if self.use_mixup and valid_logits.requires_grad and valid_logits.size(0) > 1:
                        # If mixup is used, calculate loss for both sets of targets
                        loss_a = self._focal_loss_with_weights_and_smoothing(valid_logits, targets_a, gamma, weights)
                        loss_b = self._focal_loss_with_weights_and_smoothing(valid_logits, targets_b, gamma, weights)
                        loss = lam * loss_a + (1 - lam) * loss_b
                    else:
                        loss = self._focal_loss_with_weights_and_smoothing(valid_logits, valid_targets, gamma, weights)
                else:
                    # No valid class statistics, use standard focal loss
                    if self.use_mixup and valid_logits.requires_grad and valid_logits.size(0) > 1:
                        loss_a = self._focal_loss_with_smoothing(valid_logits, targets_a, gamma, alpha)
                        loss_b = self._focal_loss_with_smoothing(valid_logits, targets_b, gamma, alpha)
                        loss = lam * loss_a + (1 - lam) * loss_b
                    else:
                        loss = self._focal_loss_with_smoothing(valid_logits, valid_targets, gamma, alpha)
            else:
                # Use standard focal loss
                if self.use_mixup and valid_logits.requires_grad and valid_logits.size(0) > 1:
                    loss_a = self._focal_loss_with_smoothing(valid_logits, targets_a, gamma, alpha)
                    loss_b = self._focal_loss_with_smoothing(valid_logits, targets_b, gamma, alpha)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    loss = self._focal_loss_with_smoothing(valid_logits, valid_targets, gamma, alpha)
        else:
            # Use standard cross-entropy loss, but add label smoothing and regularization
            label_smoothing = min(0.25, 0.05 + layer_idx * 0.06)  # Use more smoothing for deeper layers

            # Apply L2 regularization
            weight_decay = 0.01 * (1 + layer_idx * 0.5)  # Increases with layer number
            l2_reg = torch.tensor(0.0, device=valid_logits.device)
            for param in pred_logits.parameters() if hasattr(pred_logits, 'parameters') else []:
                l2_reg += torch.norm(param, 2)

            # Use cross-entropy loss and apply label smoothing
            if self.use_mixup and valid_logits.requires_grad and valid_logits.size(0) > 1:
                loss_a = F.cross_entropy(valid_logits, targets_a, reduction='mean', label_smoothing=label_smoothing)
                loss_b = F.cross_entropy(valid_logits, targets_b, reduction='mean', label_smoothing=label_smoothing)
                ce_loss = lam * loss_a + (1 - lam) * loss_b
            else:
                ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction='mean', label_smoothing=label_smoothing)

            # Add an extra KL divergence regularization term to prevent overfitting
            uniform = torch.ones_like(probs) / probs.size(-1)
            kl_div = F.kl_div(torch.log(probs + 1e-8), uniform, reduction='batchmean') * 0.05

            # Combine losses
            loss = ce_loss + weight_decay * l2_reg + kl_div

        return loss, accuracy

    def _focal_loss_with_smoothing(self, logits: Tensor, targets: Tensor, gamma: float = 2.0, alpha: float = 0.25) -> Tensor:
        """
        Focal loss with label smoothing.

        Args:
            logits: Predicted logits.
            targets: Target class indices.
            gamma: Focusing parameter to reduce the weight of easy-to-classify samples.
            alpha: Balancing parameter to handle class imbalance.
        """
        num_classes = logits.size(-1)
        batch_size = targets.size(0)

        # Create one-hot encoding
        one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)

        # Apply label smoothing
        if self.use_label_smoothing and logits.requires_grad:
            # Adjust smoothing based on the number of classes - more classes, more smoothing
            class_factor = min(0.3, 0.05 * (num_classes / 100))  # Class count influence factor
            smoothing = min(0.25, self.label_smoothing_alpha + gamma * 0.015 + class_factor) # Adjust gamma influence factor
            one_hot = one_hot * (1 - smoothing) + smoothing / num_classes

        # Calculate focal loss
        probs = F.softmax(logits, dim=-1)
        pt = (one_hot * probs).sum(dim=1)
        focal_weight = alpha * ((1 - pt) ** gamma)

        # Calculate cross-entropy loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(one_hot * log_probs, dim=1)

        # Apply focal weight
        focal_loss = focal_weight * loss

        return focal_loss.mean()

    def _focal_loss_with_weights_and_smoothing(self, logits: Tensor, targets: Tensor, gamma: float = 2.0, class_weights: Tensor = None) -> Tensor:
        """
        Focal loss with label smoothing and class weights.

        Args:
            logits: Predicted logits.
            targets: Target class indices.
            gamma: Focusing parameter to reduce the weight of easy-to-classify samples.
            class_weights: Weight for each class.
        """
        num_classes = logits.size(-1)
        batch_size = targets.size(0)

        # Create one-hot encoding
        one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)

        # Apply label smoothing - use stronger smoothing for layers with a large number of classes
        if self.use_label_smoothing and logits.requires_grad:
            # Adjust smoothing based on the number of classes - more classes, more smoothing
            class_factor = min(0.3, 0.05 * (num_classes / 100))  # Class count influence factor
            smoothing = min(0.25, self.label_smoothing_alpha + gamma * 0.015 + class_factor) # Adjust gamma influence factor
            one_hot = one_hot * (1 - smoothing) + smoothing / num_classes

        # Get the weight for each sample's corresponding class
        sample_weights = class_weights[targets] if class_weights is not None else torch.ones_like(targets, dtype=torch.float)

        # Calculate focal loss
        probs = F.softmax(logits, dim=-1)
        pt = (one_hot * probs).sum(dim=1)

        # Adjust gamma value for cases with many classes
        # The more classes, the larger the gamma, focusing more on hard-to-classify samples
        adjusted_gamma = gamma * (1.0 + 0.25 * min(1.0, num_classes / 250)) # Adjust num_classes influence factor
        focal_weight = ((1 - pt) ** adjusted_gamma)

        # Apply sample weights and focal weights
        weighted_focal = sample_weights * focal_weight

        # Calculate cross-entropy loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(one_hot * log_probs, dim=1)

        # Apply weighted focal loss
        focal_loss = weighted_focal * loss

        # Add an extra regularization term to prevent overfitting
        # Add stronger regularization for layers with a large number of classes
        if num_classes > 100 and logits.requires_grad:
            # Encourage a more uniform prediction distribution to avoid overconfidence
            uniform = torch.ones_like(probs) / num_classes
            kl_div = F.kl_div(torch.log(probs + 1e-8), uniform, reduction='batchmean')
            reg_weight = min(0.12, 0.015 * (num_classes / 100))  # Adjust weight based on the number of classes
            return focal_loss.mean() + reg_weight * kl_div

        return focal_loss.mean()
