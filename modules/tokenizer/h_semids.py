import math
import torch
import os # Added for path manipulation

from data.tags_processed import ItemData
from data.tags_processed import SeqData
from data.tags_processed import RecDataset
from data.schemas import SeqBatch
from data.schemas import TokenizedSeqBatch
from data.utils import batch_to
from einops import rearrange
from einops import pack
from modules.utils import eval_mode
from modules.h_rqvae import HRqVae
from typing import List, Dict, Optional, Tuple
from torch import nn
from torch import Tensor
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler

BATCH_SIZE = 16

class HSemanticIdTokenizer(nn.Module):
    """
    Tokenizes a sequence of item features into a sequence of semantic IDs using the HRQVAE model.
    Supports tag prediction functionality.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        n_layers: int = 3,
        n_cat_feats: int = 18,
        commitment_weight: float = 0.25,
        hrqvae_weights_path: Optional[str] = None,
        hrqvae_codebook_normalize: bool = False,
        hrqvae_sim_vq: bool = False,
        tag_alignment_weight: float = 0.5,
        tag_prediction_weight: float = 0.5,
        tag_class_counts: Optional[List[int]] = None,
        tag_embed_dim: int = 768,
        use_dedup_dim: bool = False,  # Use deduplication dimension
        use_concatenated_ids: bool = False,  # New parameter, use concatenated mode
        use_interleaved_ids: bool = False # New parameter, use interleaved mode
    ) -> None:
        super().__init__()

        # Ensure that use_dedup_dim, use_concatenated_ids, and use_interleaved_ids are mutually exclusive
        if use_dedup_dim and use_concatenated_ids:
            raise ValueError("use_dedup_dim and use_concatenated_ids cannot be True at the same time, they are mutually exclusive")
        if use_dedup_dim and use_interleaved_ids:
            raise ValueError("use_dedup_dim and use_interleaved_ids cannot be True at the same time, they are mutually exclusive")
        if use_concatenated_ids and use_interleaved_ids:
            raise ValueError("use_concatenated_ids and use_interleaved_ids cannot be True at the same time, they are mutually exclusive")

        self.hrq_vae = HRqVae(
            input_dim=input_dim,
            embed_dim=output_dim,
            hidden_dims=hidden_dims,
            codebook_size=codebook_size,
            codebook_kmeans_init=False,
            codebook_normalize=hrqvae_codebook_normalize,
            codebook_sim_vq=hrqvae_sim_vq,
            n_layers=n_layers,
            n_cat_features=n_cat_feats,
            commitment_weight=commitment_weight,
            tag_alignment_weight=tag_alignment_weight,
            tag_prediction_weight=tag_prediction_weight,
            tag_class_counts=tag_class_counts,
            tag_embed_dim=tag_embed_dim
        )
        
        if hrqvae_weights_path is not None:
            self.hrq_vae.load_pretrained(hrqvae_weights_path)

        self.hrq_vae.eval()

        self.codebook_size = codebook_size
        self.n_layers = n_layers
        self.use_dedup_dim = use_dedup_dim  # Save parameter
        self.use_concatenated_ids = use_concatenated_ids  # Save parameter
        self.tag_class_counts = tag_class_counts  # Save tag class counts
        self.use_interleaved_ids = use_interleaved_ids # Save parameter
        self.reset()
    
    def _get_hits(self, query: Tensor, key: Tensor) -> Tensor:
        return (rearrange(key, "b d -> 1 b d") == rearrange(query, "b d -> b 1 d")).all(axis=-1)
    
    def reset(self):
        self.cached_ids = None
    
    @property
    def sem_ids_dim(self):
        # Return different dimension values based on the mode used
        if self.use_dedup_dim:
            return self.n_layers + 1  # Number of semantic ID layers + deduplication dimension
        elif self.use_concatenated_ids and self.tag_class_counts is not None:
            # Note: When using concatenated mode, the total dimension is the number of semantic ID layers plus the number of tag layers
            return self.n_layers + len(self.tag_class_counts)  # Number of semantic ID layers + number of tag layers
        elif self.use_interleaved_ids and self.tag_class_counts is not None:
            # In interleaved mode, the total dimension is also the number of semantic ID layers plus the number of tag layers
            return self.n_layers + len(self.tag_class_counts)
        else:
            return self.n_layers  # Only the number of semantic ID layers
    
    @torch.no_grad()
    @eval_mode
    def precompute_corpus_ids(self, movie_dataset: ItemData) -> Tensor:
        all_ids_list = [] # Used to collect all processed IDs
        
        # sampler = BatchSampler(
        #     SequentialSampler(range(len(movie_dataset))), batch_size=512, drop_last=False
        # )
        # Remove custom collate_fn so that DataLoader returns a dictionary or SeqBatch object
        # Let DataLoader use the default collate_fn
        dataloader = DataLoader(movie_dataset, batch_size=512, shuffle=False)

        for batch_data in dataloader:
            # Move data to the model's device
            batch_on_device = batch_to(batch_data, self.hrq_vae.device)
            item_features = batch_on_device.x # Assume item features are in the x attribute

            # 1. Get semantic IDs
            encoded_features = self.hrq_vae.encode(item_features)
            hrqvae_output = self.hrq_vae.get_semantic_ids(encoded_features)
            # The shape of semantic_ids should be [batch_size, n_layers]
            semantic_ids = hrqvae_output.sem_ids 
            
            current_batch_ids = semantic_ids

            if self.use_concatenated_ids:
                # 2. Get predicted tag IDs
                # predict_tags expects input of shape [batch_size, feature_dim] or [batch_size, seq_len, feature_dim]
                # ItemData is usually [batch_size, feature_dim]
                predicted_tags_output = self.hrq_vae.predict_tags(item_features)
                predicted_tag_indices = predicted_tags_output['predictions'] # Shape: [batch_size, n_layers_tags]
                
                # Ensure the batch size of predicted tag indices and semantic IDs are consistent
                if predicted_tag_indices.shape[0] != semantic_ids.shape[0]:
                    raise ValueError(f"Semantic ID batch size ({semantic_ids.shape[0]}) does not match predicted tag batch size ({predicted_tag_indices.shape[0]})")

                # 3. Concatenate semantic IDs and predicted tag IDs
                current_batch_ids = torch.cat([semantic_ids, predicted_tag_indices], dim=1)
            elif self.use_interleaved_ids:
                # 2. Get predicted tag IDs
                predicted_tags_output = self.hrq_vae.predict_tags(item_features)
                predicted_tag_indices = predicted_tags_output['predictions'] # Shape: [batch_size, n_layers_tags]

                if predicted_tag_indices.shape[0] != semantic_ids.shape[0]:
                    raise ValueError(f"Semantic ID batch size ({semantic_ids.shape[0]}) does not match predicted tag batch size ({predicted_tag_indices.shape[0]})")

                # 3. Interleave semantic IDs and predicted tag IDs
                # semantic_ids: [B, n_layers_sem]
                # predicted_tag_indices: [B, n_layers_tag]
                # Target: [B, n_layers_sem + n_layers_tag] where elements are [s1, t1, s2, t2, ...]
                
                n_sem = semantic_ids.shape[1]
                n_tag = predicted_tag_indices.shape[1]
                max_len = max(n_sem, n_tag)
                interleaved_ids_list = []
                for i in range(max_len):
                    if i < n_sem:
                        interleaved_ids_list.append(semantic_ids[:, i:i+1])
                    if i < n_tag:
                        interleaved_ids_list.append(predicted_tag_indices[:, i:i+1])
                current_batch_ids = torch.cat(interleaved_ids_list, dim=1)

            all_ids_list.append(current_batch_ids)

        if not all_ids_list:
            # If the dataset is empty, return an empty tensor or raise an error
            self.cached_ids = torch.empty(0, self.sem_ids_dim, device=self.hrq_vae.device, dtype=torch.long)
        else:
            # Concatenate IDs from all batches
            concatenated_ids = torch.cat(all_ids_list, dim=0)
        
            # If using deduplication dimension (note: current logic is mutually exclusive with concatenated IDs, but the framework is kept)
            if self.use_dedup_dim: 
                dedup_dim_values = []
                # ...deduplication logic (needs redesign if it is to be combined with concatenated IDs)...
                # Here we assume that the deduplication dimension is not calculated in concatenated mode
                # If needed, the deduplication dimension should be calculated based on concatenated_ids and then concatenated
                # For now, assuming dedup_dim is not used with concatenated_ids as per prior logic
                self.cached_ids = concatenated_ids 
            else:
                self.cached_ids = concatenated_ids
        
        print(f"Precomputation finished. Cached IDs shape: {self.cached_ids.shape if self.cached_ids is not None else 'None'}")
        if self.cached_ids is not None and self.cached_ids.numel() > 0:
            print(f"Cached ID samples (first 3):\n{self.cached_ids[:3]}")
        
        return self.cached_ids
    
    @torch.no_grad()
    @eval_mode
    def exists_prefix(self, sem_id_prefix: Tensor) -> Tensor:
        if self.cached_ids is None:
            raise Exception("No match found in empty cache.")
    
        # Print dimension info for debugging
        print(f"Prefix shape: {sem_id_prefix.shape}, Cache shape: {self.cached_ids.shape}")
        
        # Get prefix length and ensure it does not exceed the cache's dimension
        prefix_length = min(sem_id_prefix.shape[-1], self.cached_ids.shape[-1])
        prefix_cache = self.cached_ids[:, :prefix_length]
        
        # Only use the first prefix_length elements of the prefix
        sem_id_prefix_truncated = sem_id_prefix[..., :prefix_length]
        
        print(f"After truncation - Prefix shape: {sem_id_prefix_truncated.shape}, Cache prefix shape: {prefix_cache.shape}")
        
        out = torch.zeros(*sem_id_prefix.shape[:-1], dtype=bool, device=sem_id_prefix.device)
        
        # Batch prefix matching to avoid OOM
        batches = math.ceil(sem_id_prefix.shape[0] // BATCH_SIZE)
        for i in range(batches):
            prefixes = sem_id_prefix_truncated[i*BATCH_SIZE:(i+1)*BATCH_SIZE,...]
            
            # Ensure dimensions match
            if prefixes.shape[-1] == prefix_cache.shape[-1]:
                # Standard comparison
                matches = (prefixes.unsqueeze(-2) == prefix_cache.unsqueeze(-3)).all(axis=-1).any(axis=-1)
            else:
                # An exception occurred, logging more information
                print(f"Warning: Dimensions still do not match! prefixes: {prefixes.shape}, prefix_cache: {prefix_cache.shape}")
                
                # Get the minimum common dimension
                common_dims = min(prefixes.shape[-1], prefix_cache.shape[-1])
                
                # Compare using the common dimension
                matches = (prefixes[..., :common_dims].unsqueeze(-2) == 
                           prefix_cache[..., :common_dims].unsqueeze(-3)).all(axis=-1).any(axis=-1)
            
            out[i*BATCH_SIZE:(i+1)*BATCH_SIZE,...] = matches
        
        return out
    
    def _tokenize_seq_batch_from_cached(self, ids: Tensor) -> Tensor:
        """
        Gets the semantic representation for specified IDs from the cached semantic IDs.
        
        Args:
            ids: Item ID tensor of shape [batch_size, seq_len].
            
        Returns:
            Semantic ID tensor of shape [batch_size, seq_len * sem_ids_dim].
            In concatenated mode, only the semantic ID part is returned; tag IDs will be handled separately in the forward method.
        """
        # Ensure all IDs are within the cache's range
        valid_ids = ids.clone()
        valid_ids[valid_ids >= self.cached_ids.shape[0]] = 0  # Replace out-of-range IDs with 0
        
        # Get semantic IDs (from cache)
        # Note: In concatenated mode, the cache only stores the semantic ID part; tag IDs will be handled separately in the forward method.
        return rearrange(self.cached_ids[valid_ids.flatten(), :], "(b n) d -> b (n d)", n=ids.shape[1])
    
    @torch.no_grad()
    @eval_mode
    def forward(self, batch: SeqBatch) -> TokenizedSeqBatch:
        # If the cache is empty or batch IDs are out of the cache's range, generate semantic IDs using HRQVAE
        if self.cached_ids is None or batch.ids.max() >= self.cached_ids.shape[0]:
            B, N = batch.ids.shape
            # D = self.sem_ids_dim # Total dimension will be determined below

            # --- Process current sequence (sem_ids) ---
            # 1. Get semantic IDs
            encoded_x = self.hrq_vae.encode(batch.x)
            hrqvae_output = self.hrq_vae.get_semantic_ids(encoded_x) # batch.x shape: [B, N, feat_dim]
            # The shape of hrqvae_output.sem_ids should be [B*N, n_layers]
            actual_semantic_ids_flat = hrqvae_output.sem_ids 
            actual_semantic_ids = rearrange(actual_semantic_ids_flat, '(b n) d -> b n d', b=B, n=N)

            combined_ids_matrix = actual_semantic_ids

            if self.use_concatenated_ids:
                # 2. Get predicted tag IDs
                # predict_tags expects input of shape [B, N, feat_dim], outputs 'predictions' of shape [B, N, n_tag_layers]
                predicted_tags_output = self.hrq_vae.predict_tags(batch.x)
                predicted_tag_indices = predicted_tags_output['predictions'] # [B, N, n_tag_layers]
                
                # 3. Concatenate semantic IDs and predicted tag IDs
                combined_ids_matrix = torch.cat([actual_semantic_ids, predicted_tag_indices], dim=2)
            elif self.use_interleaved_ids:
                # 2. Get predicted tag IDs
                predicted_tags_output = self.hrq_vae.predict_tags(batch.x)
                predicted_tag_indices = predicted_tags_output['predictions'] # [B, N, n_tag_layers]

                # 3. Interleave
                # actual_semantic_ids: [B, N, n_layers_sem]
                # predicted_tag_indices: [B, N, n_layers_tag]
                # Target: [B, N, n_layers_sem + n_layers_tag] with interleaved IDs
                n_sem = actual_semantic_ids.shape[2]
                n_tag = predicted_tag_indices.shape[2]
                max_item_dim = max(n_sem, n_tag)
                interleaved_list_batch = []
                for i in range(max_item_dim):
                    if i < n_sem:
                        interleaved_list_batch.append(actual_semantic_ids[:, :, i:i+1])
                    if i < n_tag:
                        interleaved_list_batch.append(predicted_tag_indices[:, :, i:i+1])
                combined_ids_matrix = torch.cat(interleaved_list_batch, dim=2)

            # Flatten and concatenate the ID sequence (of length D_total) for each item
            sem_ids = rearrange(combined_ids_matrix, 'b n d -> b (n d)')
            D_total = combined_ids_matrix.shape[2] # Update total dimension

            # --- Process future sequence (sem_ids_fut) ---
            sem_ids_fut = None
            if batch.x_fut is not None:
                # 1. Get semantic IDs for future items
                # batch.x_fut shape: [B, feat_dim], after .unsqueeze(1) it becomes [B, 1, feat_dim]
                # The shape of hrqvae_output_fut.sem_ids should be [B*1, n_layers], i.e., [B, n_layers]
                encoded_x_fut = self.hrq_vae.encode(batch.x_fut.unsqueeze(1))
                hrqvae_output_fut = self.hrq_vae.get_semantic_ids(encoded_x_fut) 
                actual_semantic_ids_fut_flat = hrqvae_output_fut.sem_ids 
                actual_semantic_ids_fut = rearrange(actual_semantic_ids_fut_flat, '(b n) d -> b (n d)', b=B, n=1) # [B, n_layers]

                combined_ids_matrix_fut = actual_semantic_ids_fut

                if self.use_concatenated_ids:
                    # 2. Get predicted tag IDs for future items
                    predicted_tags_output_fut = self.hrq_vae.predict_tags(batch.x_fut.unsqueeze(1))
                    predicted_tag_indices_fut = predicted_tags_output_fut['predictions'] # [B, 1, n_tag_layers]
                    predicted_tag_indices_fut = rearrange(predicted_tag_indices_fut, 'b n d -> b (n d)') # [B, n_tag_layers]
                    
                    # 3. Concatenate
                    combined_ids_matrix_fut = torch.cat([actual_semantic_ids_fut, predicted_tag_indices_fut], dim=1)
                elif self.use_interleaved_ids:
                    # 2. Get predicted tag IDs for future items
                    predicted_tags_output_fut = self.hrq_vae.predict_tags(batch.x_fut.unsqueeze(1))
                    predicted_tag_indices_fut = predicted_tags_output_fut['predictions'] # [B, 1, n_tag_layers]
                    predicted_tag_indices_fut = rearrange(predicted_tag_indices_fut, 'b n d -> b (n d)') # [B, n_tag_layers]

                    # 3. Interleave
                    # actual_semantic_ids_fut: [B, n_layers_sem]
                    # predicted_tag_indices_fut: [B, n_layers_tag]
                    n_sem_fut = actual_semantic_ids_fut.shape[1]
                    n_tag_fut = predicted_tag_indices_fut.shape[1]
                    max_fut_dim = max(n_sem_fut, n_tag_fut)
                    interleaved_list_fut = []
                    for i in range(max_fut_dim):
                        if i < n_sem_fut:
                            interleaved_list_fut.append(actual_semantic_ids_fut[:, i:i+1])
                        if i < n_tag_fut:
                            interleaved_list_fut.append(predicted_tag_indices_fut[:, i:i+1])
                    combined_ids_matrix_fut = torch.cat(interleaved_list_fut, dim=1)

                sem_ids_fut = combined_ids_matrix_fut
                # D_total_fut should be the same as D_total
            
            seq_mask = batch.seq_mask.repeat_interleave(D_total, dim=1) if batch.seq_mask is not None else None
            if seq_mask is not None:
                sem_ids[~seq_mask] = -1
        else:
            # Get semantic IDs from cache (they are already concatenated in the cache)
            B, N = batch.ids.shape
            D_total = self.cached_ids.shape[-1] # Get total dimension from cache
            
            sem_ids = self._tokenize_seq_batch_from_cached(batch.ids)
            # _tokenize_seq_batch_from_cached returns [B, N * D_total]
            
            seq_mask = batch.seq_mask.repeat_interleave(D_total, dim=1) if batch.seq_mask is not None else None
            if seq_mask is not None:
                sem_ids[~seq_mask] = -1
        
            # Process future IDs (get from cache)
            sem_ids_fut = self._tokenize_seq_batch_from_cached(batch.ids_fut)
            # _tokenize_seq_batch_from_cached returns [B, 1 * D_total] (because future IDs are usually for a single item)

        # token_type_ids should be based on D_total
        token_type_ids = torch.arange(D_total, device=sem_ids.device).repeat(B, N)
        token_type_ids_fut = torch.arange(D_total, device=sem_ids.device).repeat(B, 1)
        
        # Print shape info for debugging
        print(f"sem_ids shape: {sem_ids.shape}, sem_ids_fut shape: {sem_ids_fut.shape if sem_ids_fut is not None else 'None'}")
        print(f"token_type_ids shape: {token_type_ids.shape}, token_type_ids_fut shape: {token_type_ids_fut.shape}")
        
        result = TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=sem_ids,
            sem_ids_fut=sem_ids_fut,
            seq_mask=seq_mask,
            token_type_ids=token_type_ids,
            token_type_ids_fut=token_type_ids_fut
        )
        
        # # Print sample ID structure for debugging
        # if self.use_concatenated_ids and B > 0 and sem_ids.numel() > 0 and (sem_ids_fut is None or sem_ids_fut.numel() > 0):
        #     sample_idx = 0
        #     print(f"Sample ID structure (index {sample_idx}):")
            
        #     n_sem_layers = self.n_layers
        #     # D_total is calculated above

        #     # Print current IDs - modified to print all items in the sequence
        #     if sem_ids is not None and sem_ids.shape[1] > 0:
        #         # sem_ids is already [B, N * D_total]
        #         # N is the original sequence length, D_total is the total number of IDs per item
        #         if N > 0: # N is batch.ids.shape[1] (sequence length of items)
        #             print(f"  Current sequence ({N} items total):")
        #             for item_idx_in_seq in range(N):
        #                 start_idx = item_idx_in_seq * D_total
        #                 end_idx = (item_idx_in_seq + 1) * D_total
                        
        #                 if end_idx > sem_ids.shape[1]:
        #                     continue 
                        
        #                 if sem_ids[sample_idx, start_idx].item() == -1:
        #                     continue

        #                 current_item_ids_flat = sem_ids[sample_idx, start_idx:end_idx]
                        
        #                 if current_item_ids_flat.numel() == D_total:
        #                     current_sem_ids_part = current_item_ids_flat[:n_sem_layers]
        #                     current_tag_ids_part = current_item_ids_flat[n_sem_layers:]
        #                     print(f"    Item {item_idx_in_seq + 1} - Semantic IDs: {current_sem_ids_part.tolist()}, Tag IDs: {current_tag_ids_part.tolist()}")
        #                 else:
        #                     print(f"    Item {item_idx_in_seq + 1}: ID data is incomplete or masked (expected length {D_total}, actual {current_item_ids_flat.numel()})")
        #         else:
        #             print("  Current sequence is empty or incorrectly formatted.")
            
        #     # Print future IDs
        #     if sem_ids_fut is not None and sem_ids_fut.shape[1] > 0:
        #         # sem_ids_fut is already [B, 1 * D_total] or [B, D_total]
        #         # Take the IDs of the first (and only) future item
        #         if D_total > 0 : # Make sure D_total is valid
        #             future_item_ids_flat = sem_ids_fut[sample_idx, :D_total]

        #             future_sem_ids_part = future_item_ids_flat[:n_sem_layers]
        #             future_tag_ids_part = future_item_ids_flat[n_sem_layers:]
        #             print(f"  Future Semantic IDs: {future_sem_ids_part.tolist()}")
        #             print(f"  Future Tag IDs: {future_tag_ids_part.tolist()}")
        #         else:
        #             print("  D_total for future IDs is 0, cannot parse.")
            
        #     # Check original tag indices in the data (if available and applicable)
        #     if hasattr(batch, 'tags_indices') and batch.tags_indices is not None and self.tag_class_counts is not None and batch.tags_indices.shape[0] > sample_idx:
        #         num_tag_layers_original = len(self.tag_class_counts)
        #         if batch.tags_indices.dim() == 3 and N > 0: # [B, N, n_tag_layers_original]
        #             if batch.tags_indices.shape[2] >= num_tag_layers_original:
        #                 original_tags_for_sample = batch.tags_indices[sample_idx, 0, :num_tag_layers_original]
        #                 print(f"  Original tag indices (from data batch, first item): {original_tags_for_sample.tolist()}")
        #         elif batch.tags_indices.dim() == 2: # [B, n_tag_layers_original]
        #             if batch.tags_indices.shape[1] >= num_tag_layers_original:
        #                 original_tags_for_sample = batch.tags_indices[sample_idx, :num_tag_layers_original]
        #                 print(f"  Original tag indices (from data batch): {original_tags_for_sample.tolist()}")

        return result
    
    @torch.no_grad()
    @eval_mode
    def predict_tags(self, batch: SeqBatch) -> Dict[str, Tensor]:
        """
        Predicts tags for items in a batch, ignoring padded items (-1) in the sequence.
        
        Args:
            batch: A batch containing item features.
            
        Returns:
            A dictionary containing predicted tag indices and confidences.
        """
        # Use the tag prediction functionality of HRQVAE
        print(f"the shape of batch.x is {batch.x.shape}")
        
        # Get the sequence mask, which identifies valid (non-padded) positions
        seq_mask = batch.seq_mask if hasattr(batch, 'seq_mask') else None
        
        if seq_mask is not None:
            # If there is a mask, we only need to process the valid items
            batch_size, seq_len, feat_dim = batch.x.shape
            
            # Create a masked version of the feature tensor, setting features of padded positions to 0
            # This will not affect the prediction results, as we will filter the results based on the mask later
            masked_x = batch.x.clone()
            
            # Expand the mask to the feature dimension
            expanded_mask = seq_mask.unsqueeze(-1).expand_as(masked_x)
            
            # Set features of padded positions to 0
            masked_x = masked_x * expanded_mask
            
            # Use the processed features for prediction
            predictions = self.hrq_vae.predict_tags(masked_x)
            
            # Process the prediction results, setting predictions for padded positions to -1
            if 'predictions' in predictions:
                # Get prediction results
                pred = predictions['predictions']
                # Create an expanded version of the mask to match the shape of the prediction results
                # Prediction result shape is [batch_size, seq_len, n_layers]
                expanded_pred_mask = seq_mask.unsqueeze(-1).expand_as(pred)
                # Create a fill value tensor (-1)
                fill_value = torch.full_like(pred, -1)
                # Select prediction values or fill values based on the mask
                predictions['predictions'] = torch.where(expanded_pred_mask, pred, fill_value)
            
            if 'confidences' in predictions:
                # Get confidences
                conf = predictions['confidences']
                # Create an expanded version of the mask to match the shape of the confidences
                expanded_conf_mask = seq_mask.unsqueeze(-1).expand_as(conf)
                # Create a fill value tensor (0.0)
                fill_value = torch.zeros_like(conf)
                # Select confidences or fill values based on the mask
                predictions['confidences'] = torch.where(expanded_conf_mask, conf, fill_value)
        else:
            # If there is no mask, predict directly
            predictions = self.hrq_vae.predict_tags(batch.x)
        

        
        return predictions
    
    @torch.no_grad()
    @eval_mode
    def tokenize_with_tags(self, batch: SeqBatch) -> Tuple[TokenizedSeqBatch, Dict[str, Tensor]]:
        """
        Tokenizes the batch and predicts tags.
        
        Args:
            batch: A batch containing item features.
            
        Returns:
            A tuple of the tokenized batch and tag prediction results.
        """
        tokenized_batch = self.forward(batch)
        tag_predictions = self.predict_tags(batch)
        return tokenized_batch, tag_predictions
        

if __name__ == "__main__":
    # Hardcode parameters, no longer using argparse
    dataset_name_arg = 'beauty' # 'ml-1m' or 'beauty'
    input_dim_arg = 768
    embed_dim_arg = 32
    hidden_dims_arg = [512, 256, 128]
    codebook_size_arg = 256
    n_cat_feats_arg = 0
    tag_embed_dim_arg = 768
    tag_alignment_weight_arg = 0.5 # Default value
    tag_prediction_weight_arg = 0.5 # Default value
    use_dedup_dim_arg = False
    # New parameter: use concatenated mode
    use_concatenated_ids_arg = True  # Concatenated mode enabled by default
    # New parameter: use interleaved mode
    use_interleaved_ids_arg = False # Interleaved mode disabled by default
    # Hardcode model path
    hrqvae_weights_path_arg = "out/hrqvae/amazon/hrqvae_AMAZON_20250524_212758/hrqvae_model_ACC0.7658_RQLOSS0.3243_20250525_031159.pt"
    n_layers_arg = 3 # Inferred from tag_class_counts or default

    print(f"Testing on dataset: {dataset_name_arg}")
    print(f"Model parameters: input_dim={input_dim_arg}, embed_dim={embed_dim_arg}, "
          f"hidden_dims={hidden_dims_arg}, codebook_size={codebook_size_arg}, "
          f"n_cat_feats={n_cat_feats_arg}, tag_embed_dim={tag_embed_dim_arg}, "
          f"use_dedup_dim={use_dedup_dim_arg}, n_layers={n_layers_arg}")
    
    if dataset_name_arg == 'ml-1m':
        dataset_path = "dataset/ml-1m-movie"
        seq_dataset_path = "dataset/ml-1m"
        dataset_enum = RecDataset.ML_1M
        original_tag_class_counts_for_remapping = [18, 7, 20][:n_layers_arg] # Example, adjust if necessary
        # These need to be consistent with the model training
        tag_class_counts_arg = [18, 7, 20][:n_layers_arg] # This would be remapped counts if ml-1m used remapping
    else:  # beauty (amazon)
        dataset_path = "dataset/amazon"
        seq_dataset_path = "dataset/amazon"
        dataset_enum = RecDataset.AMAZON
        original_tag_class_counts_for_remapping = [6, 130, 927][:n_layers_arg] # Original counts for Amazon Beauty
        # These need to be consistent with model training, adjusted based on error messages (these are remapped counts model expects)
        tag_class_counts_arg = [7, 30, 97][:n_layers_arg] # MODIFIED
    
    # Load dataset
    print(f"Loading item dataset: {dataset_path}")
    dataset = ItemData(dataset_path, dataset=dataset_enum, split="beauty" if dataset_name_arg == 'beauty' else None)

    # --- Perform tag remapping on the loaded dataset.tags_indices --- 
    if dataset_name_arg == 'beauty' and hasattr(dataset, 'tags_indices') and dataset.tags_indices is not None:
        print("Performing tag index remapping for the Amazon Beauty dataset...")
        # Determine the path for rare_tags.pt
        # Assumed save_dir_root during training: out/hrqvae/amazon/
        # Model is saved in save_dir_root/run_specific_folder/model.pt
        # The rare_tags.pt file is saved in save_dir_root/special_tags_files/rare_tags.pt
        model_dir = os.path.dirname(hrqvae_weights_path_arg) # .../hrqvae_AMAZON_20250524_212758
        save_dir_root_guess = os.path.dirname(model_dir) # .../amazon
        # Check if this is the second level 'amazon' or the first 'hrqvae/amazon'
        if os.path.basename(save_dir_root_guess) == dataset_name_arg: # e.g., ends with /amazon
              # This seems to be the project-level dataset folder, not the run output base
              # Let's assume save_dir_root was 'out/hrqvae/amazon/' during training
              # which means rare_tags.pt is relative to that. 
              # The model path is 'out/hrqvae/amazon/RUN_FOLDER/model.pt'
              # So, the save_dir_root that created special_tags_files is likely 'out/hrqvae/amazon/'
              rare_tags_base_path = save_dir_root_guess # 'out/hrqvae/amazon/'
        else:
            # Fallback if the above logic is not perfect, assume 'out/hrqvae/amazon/' structure based on model path segments
            path_parts = hrqvae_weights_path_arg.split(os.sep)
            if "hrqvae" in path_parts and "amazon" in path_parts:
                hrqvae_idx = path_parts.index("hrqvae")
                amazon_idx = path_parts.index("amazon")
                if amazon_idx == hrqvae_idx + 1:
                    rare_tags_base_path = os.path.join(*path_parts[:amazon_idx+1])
                else:
                    rare_tags_base_path = "out/hrqvae/amazon" # Default guess
            else:
                rare_tags_base_path = "out/hrqvae/amazon" # Default guess

        rare_tags_path = os.path.join(rare_tags_base_path, "special_tags_files", "rare_tags.pt")
        
        print(f"Attempting to load rare tags dictionary from: {rare_tags_path}")

        if not os.path.exists(rare_tags_path):
            print(f"Warning: Rare tags file {rare_tags_path} not found. Skipping tag remapping. This may cause errors if the model was trained with remapped tags.")
        else:
            rare_tags_dict = torch.load(rare_tags_path, map_location=torch.device('cpu'))
            print(f"Loaded rare tags dictionary: {list(rare_tags_dict.keys())}")

            # `tag_class_counts_arg` is the remapped_tag_class_counts
            # `original_tag_class_counts_for_remapping` is the original
            for i in range(n_layers_arg):
                if i in rare_tags_dict and rare_tags_dict[i].numel() > 0:
                    layer_indices_tensor = dataset.tags_indices[:, i]
                    remapped_num_classes_for_layer = tag_class_counts_arg[i]
                    original_num_classes_for_layer = original_tag_class_counts_for_remapping[i]
                    
                    special_class_id = remapped_num_classes_for_layer - 1
                    
                    id_mapping = torch.full((original_num_classes_for_layer,), -1, dtype=torch.long) # Initialize with -1
                    
                    current_rare_tags = rare_tags_dict[i].long()
                    non_rare_mask = torch.ones(original_num_classes_for_layer, dtype=torch.bool)
                    if current_rare_tags.numel() > 0:
                        # Ensure current_rare_tags are within bounds for non_rare_mask
                        valid_rare_tags = current_rare_tags[current_rare_tags < original_num_classes_for_layer]
                        if len(valid_rare_tags) < len(current_rare_tags):
                            print(f"Warning: Rare tag indices for layer {i} are out of bounds for the original number of classes ({original_num_classes_for_layer}).")
                        if valid_rare_tags.numel() > 0:
                            non_rare_mask[valid_rare_tags] = False

                    new_current_id = 0
                    for orig_id in range(original_num_classes_for_layer):
                        if non_rare_mask[orig_id]:
                            id_mapping[orig_id] = new_current_id
                            new_current_id += 1
                        else:
                            id_mapping[orig_id] = special_class_id
                    
                    # Verify that new_current_id matches remapped_num_classes_for_layer - 1 (if any non-rare) or 0 (if all rare)
                    if new_current_id > special_class_id and special_class_id != -1 : # special_class_id can be -1 if remapped_num_classes is 0
                        print(f"Warning: Remapped ID count for layer {i} ({new_current_id}) does not match expectation ({special_class_id}). It's possible all tags are rare or there's a configuration error.")
                    elif new_current_id == 0 and original_num_classes_for_layer > 0 and not torch.all(~non_rare_mask):
                        # This case means no non-rare items were found, yet not all items were declared rare. This can happen if original_num_classes_for_layer is small. 
                        pass # Allow this, might just mean all are rare and map to special_class_id

                    valid_mask_for_layer = (layer_indices_tensor >= 0) & (layer_indices_tensor < original_num_classes_for_layer)
                    invalid_indices_present = (layer_indices_tensor >= original_num_classes_for_layer).any()
                    if invalid_indices_present:
                        print(f"Warning: Original tag indices for layer {i} contain values exceeding the expected maximum of {original_num_classes_for_layer-1}. These values will not be mapped and may cause errors.")

                    # Create a temporary tensor for new indices to avoid in-place issues with advanced indexing
                    new_layer_indices = layer_indices_tensor.clone()
                    # Apply mapping only to valid original indices
                    indices_to_map = layer_indices_tensor[valid_mask_for_layer].long()
                    mapped_values = id_mapping[indices_to_map]
                    new_layer_indices[valid_mask_for_layer] = mapped_values
                    dataset.tags_indices[:, i] = new_layer_indices
                    
                    print(f"tags_indices for layer {i} have been remapped. Special class ID: {special_class_id}")
                else:
                    print(f"Layer {i} does not require remapping (layer not in rare_tags_dict or rare tags are empty).")
            print("Tag remapping complete.")
    # --- End of tag remapping ---
    
    # Create tokenizer
    print("Initializing tokenizer...")
    tokenizer = HSemanticIdTokenizer(
        input_dim=input_dim_arg,
        output_dim=embed_dim_arg,
        hidden_dims=hidden_dims_arg,
        codebook_size=codebook_size_arg,
        n_cat_feats=n_cat_feats_arg,
        n_layers=n_layers_arg, # Ensure consistency with the model
        hrqvae_weights_path=hrqvae_weights_path_arg,
        hrqvae_codebook_normalize=True, # MODIFIED from False
        hrqvae_sim_vq=False, # Default value
        tag_alignment_weight=tag_alignment_weight_arg,
        tag_prediction_weight=tag_prediction_weight_arg,
        tag_class_counts=tag_class_counts_arg, # Ensure consistency with the model
        tag_embed_dim=tag_embed_dim_arg,
        use_dedup_dim=use_dedup_dim_arg,
        use_concatenated_ids=use_concatenated_ids_arg,  # New parameter
        use_interleaved_ids=use_interleaved_ids_arg # New parameter
    )
    
    # Precompute semantic IDs
    print("Precomputing semantic IDs...")
    tokenizer.precompute_corpus_ids(dataset)
    
    # Load sequence data and test the tokenizer
    print(f"Loading sequence data: {seq_dataset_path}")
    seq_data = SeqData(seq_dataset_path, dataset=dataset_enum, split="beauty" if dataset_name_arg == 'beauty' else None)
    
    # Get a small batch of data for testing
    sample_size = 5
    print(f"Getting a test batch of size {sample_size}...")
    if len(seq_data) < sample_size:
        print(f"Warning: Not enough sequence data samples for {sample_size}, getting {len(seq_data)} instead.")
        sample_size = len(seq_data)

    if sample_size == 0:
        print("Error: No sequence data available for testing.")
        exit()
        
    batch = seq_data[:sample_size]

    print("\nTest batch samples (first 5):")
    for i in range(min(sample_size, 5)):
        print(f"  Sample {i+1}:")
        print(f"    User ID: {batch.user_ids[i]}")
        print(f"    Item IDs: {batch.ids[i]}")
        print(f"    Item features (x) shape: {batch.x[i].shape}")
        if hasattr(batch, 'tags_emb') and batch.tags_emb is not None:
              print(f"    Tag embeddings (tags_emb) shape: {batch.tags_emb[i].shape if batch.tags_emb is not None else 'N/A'}")
        if hasattr(batch, 'tags_indices') and batch.tags_indices is not None:
            print(f"    Tag indices (tags_indices): {batch.tags_indices[i] if batch.tags_indices is not None else 'N/A'}")

    
    # Process the batch with the tokenizer
    print("\nGenerating semantic IDs with tokenizer...")
    tokenized = tokenizer(batch)
    
    print("\nSemantic ID results:")
    print(f"User IDs: {tokenized.user_ids}")
    print(f"Semantic IDs shape: {tokenized.sem_ids.shape}")
    print(f"Semantic IDs (first 5): \n{tokenized.sem_ids[:5]}")
    
    # Test tag prediction functionality
    print("\nTesting tag prediction functionality...")
    tag_predictions = tokenizer.predict_tags(batch) # predict_tags handles sequences internally
    
    print("\nTag prediction results:")
    if 'predictions' in tag_predictions and tag_predictions['predictions'] is not None:
        print(f"Predicted tag indices shape: {tag_predictions['predictions'].shape}")
        print(f"Predicted tag indices (first 5): \n{tag_predictions['predictions'][:5]}")
    else:
        print("Predicted tag indices not found.")
        
    if 'confidences' in tag_predictions and tag_predictions['confidences'] is not None:
        print(f"Prediction confidences shape: {tag_predictions['confidences'].shape}")
        print(f"Prediction confidences (first 5): \n{tag_predictions['confidences'][:5]}")
    else:
        print("Prediction confidences not found.")

    # Print ground truth labels for comparison (if available)
    if hasattr(batch, 'tags_indices') and batch.tags_indices is not None:
        print("\nGround truth tag indices (for comparison, first 5):")
        print(batch.tags_indices[:5])
    
    print("\nTesting complete!")
