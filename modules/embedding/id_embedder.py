import torch

from data.schemas import TokenizedSeqBatch
from torch import nn
from torch import Tensor
from typing import NamedTuple


class SemIdEmbeddingBatch(NamedTuple):
    """
    A batch of embedded sequences, containing the main sequence and a future sequence.
    
    Attributes:
        seq (Tensor): The embedded main sequence.
        fut (Tensor): The embedded future sequence, which can be None.
    """
    seq: Tensor
    fut: Tensor


class SemIdEmbedder(nn.Module):
    """
    Embeds semantic IDs and optional tag IDs into a dense vector representation.
    
    This module can handle two modes of ID organization:
    1.  Concatenated: Semantic IDs come first, followed by tag IDs.
    2.  Interleaved: Semantic IDs and tag IDs are interleaved.
    
    It uses a single large embedding table, partitioning it for different ID layers
    and types (semantic vs. tag) to create unique embeddings for each.
    """
    def __init__(self, num_embeddings, sem_ids_dim, embeddings_dim, n_sem_layers=3, use_interleaved_ids: bool = False) -> None:
        """
        Initializes the SemIdEmbedder module.

        Args:
            num_embeddings (int): The number of unique semantic IDs per layer.
            sem_ids_dim (int): The total number of dimensions for semantic and tag IDs combined.
            embeddings_dim (int): The dimensionality of the output embeddings.
            n_sem_layers (int): The number of layers dedicated to semantic IDs.
            use_interleaved_ids (bool): If True, assumes IDs are interleaved (semantic, tag, semantic, tag, ...).
                                        Otherwise, assumes they are concatenated (all semantic, then all tag).
        """
        super().__init__()
        
        self.sem_ids_dim = sem_ids_dim
        self.num_embeddings = num_embeddings
        self.n_sem_layers = n_sem_layers  # Number of semantic ID layers
        self.use_interleaved_ids = use_interleaved_ids # Save the parameter
        
        # This will allow us to handle both semantic IDs and tag IDs by reserving space in the embedding table.
        max_tag_size = 1000  # Maximum number of categories per tag layer (adjust as needed)
        self.max_tag_size = max_tag_size
        self.n_tag_layers = sem_ids_dim - n_sem_layers # Number of tag ID layers
        
        # Calculate the total embedding size needed.
        # This needs to be robust for all modes (concatenated, interleaved, semantic-only).
        semantic_part_size_total = num_embeddings * n_sem_layers
        tag_part_size_total = max_tag_size * self.n_tag_layers if self.n_tag_layers > 0 else 0
        
        # Total size = semantic ID range + tag ID range + 1 for padding
        total_embed_size = semantic_part_size_total + tag_part_size_total + 1

        # Adjust padding_idx to be the last index in the embedding table.
        self.padding_idx = total_embed_size - 1
        
        self.emb = nn.Embedding(
            num_embeddings=total_embed_size, # Use the calculated total size
            embedding_dim=embeddings_dim,
            padding_idx=self.padding_idx
        )
    
    def forward(self, batch: TokenizedSeqBatch) -> Tensor:
        """
        Forward pass for embedding a batch of tokenized sequences.

        Args:
            batch (TokenizedSeqBatch): The input batch containing semantic IDs and token type IDs.

        Returns:
            SemIdEmbeddingBatch: A NamedTuple containing the embedded 'seq' and 'fut' tensors.
        """
        # Get the semantic IDs and token type IDs from the sequence
        sem_ids = batch.sem_ids
        token_type_ids = batch.token_type_ids
        
        # Create a tensor to hold embedding indices, distinguishing between semantic and tag IDs
        emb_indices = torch.zeros_like(sem_ids)
        
        # Total size of the semantic ID part, used to offset tag ID indices
        semantic_part_offset = self.num_embeddings * self.n_sem_layers

        if self.use_interleaved_ids:
            # Interleaved mode [s1, t1, s2, t2, ...]
            # Even indices are semantic IDs, odd indices are tag IDs
            for i in range(self.sem_ids_dim): # i is the position in the interleaved sequence
                dim_mask = token_type_ids == i
                dim_ids = sem_ids[dim_mask]

                if i % 2 == 0: # Semantic ID
                    sem_layer_idx = i // 2
                    if sem_layer_idx < self.n_sem_layers:
                        emb_indices[dim_mask] = sem_layer_idx * self.num_embeddings + dim_ids
                    else:
                        # This case should not happen if inputs are configured correctly.
                        # print(f"Warning: Semantic layer index {sem_layer_idx} is out of bounds for n_sem_layers {self.n_sem_layers}")
                        emb_indices[dim_mask] = self.padding_idx # Fallback to padding
                else: # Tag ID
                    tag_layer_idx = i // 2
                    if tag_layer_idx < self.n_tag_layers:
                        emb_indices[dim_mask] = semantic_part_offset + tag_layer_idx * self.max_tag_size + dim_ids
                    else:
                        # print(f"Warning: Tag layer index {tag_layer_idx} is out of bounds for n_tag_layers {self.n_tag_layers}")
                        emb_indices[dim_mask] = self.padding_idx # Fallback to padding
        else:
            # Non-interleaved mode (concatenated or semantic IDs only)
            # Iterate through the IDs of each dimension
            for i in range(self.sem_ids_dim):
                # Get the IDs and types for the current dimension
                dim_mask = token_type_ids == i
                dim_ids = sem_ids[dim_mask]
                
                # Ensure IDs are within a valid range to prevent out-of-bounds errors
                if dim_ids.numel() > 0:  # Only process if there are elements
                    # Check if it's a semantic or tag ID based on its position
                    is_semantic = i < self.n_sem_layers
                    if is_semantic:
                        # For semantic IDs, clamp to the range [0, self.num_embeddings-1]
                        dim_ids = torch.clamp(dim_ids, min=0, max=self.num_embeddings-1)
                    else:
                        # For tag IDs, clamp to the range [0, self.max_tag_size-1]
                        dim_ids = torch.clamp(dim_ids, min=0, max=self.max_tag_size-1)
                
                # Generate embedding indices
                if i < self.n_sem_layers:  # The first n_sem_layers dimensions are semantic IDs
                    # Index calculation for semantic IDs: layer_index * vocab_size + id
                    emb_indices[dim_mask] = i * self.num_embeddings + dim_ids
                else: # Tag ID (only in concatenated mode and when i >= n_sem_layers)
                    tag_dim_index = i - self.n_sem_layers
                    if tag_dim_index < self.n_tag_layers:
                         # Index calculation for tag IDs: offset + layer_index * vocab_size + id
                         emb_indices[dim_mask] = semantic_part_offset + tag_dim_index * self.max_tag_size + dim_ids
                    else:
                        # This can happen if sem_ids_dim is misconfigured.
                        # print(f"Warning: Tag dimension index {tag_dim_index} is out of bounds for n_tag_layers {self.n_tag_layers}")
                        emb_indices[dim_mask] = self.padding_idx # Fallback to padding
        
        # Apply padding mask
        if hasattr(batch, 'seq_mask') and batch.seq_mask is not None:
            emb_indices[~batch.seq_mask] = self.padding_idx

        # Embed the current sequence
        seq_embs = self.emb(emb_indices)

        # Process the future sequence (if it exists)
        if batch.sem_ids_fut is not None:
            fut_ids = batch.sem_ids_fut
            fut_type_ids = batch.token_type_ids_fut
            
            # Create embedding indices for the future sequence
            fut_emb_indices = torch.zeros_like(fut_ids)
            
            # Iterate through the IDs of each dimension
            for i in range(min(self.sem_ids_dim, fut_ids.shape[-1])):
                # Get the IDs and types for the current dimension
                dim_mask = fut_type_ids == i
                dim_ids = fut_ids[dim_mask]
                
                # Ensure IDs are within a valid range to prevent out-of-bounds errors
                if dim_ids.numel() > 0:  # Only process if there are elements
                    is_semantic = (i % 2 == 0) if self.use_interleaved_ids else (i < self.n_sem_layers)
                    if is_semantic:
                        # For semantic IDs, clamp to the range [0, self.num_embeddings-1]
                        dim_ids = torch.clamp(dim_ids, min=0, max=self.num_embeddings-1)
                    else:
                        # For tag IDs, clamp to the range [0, self.max_tag_size-1]
                        dim_ids = torch.clamp(dim_ids, min=0, max=self.max_tag_size-1)

                if self.use_interleaved_ids:
                    # Interleaved mode [s1, t1, s2, t2, ...]
                    if i % 2 == 0: # Semantic ID
                        sem_layer_idx = i // 2
                        if sem_layer_idx < self.n_sem_layers:
                            fut_emb_indices[dim_mask] = sem_layer_idx * self.num_embeddings + dim_ids
                        else:
                            fut_emb_indices[dim_mask] = self.padding_idx
                    else: # Tag ID
                        tag_layer_idx = i // 2
                        if tag_layer_idx < self.n_tag_layers:
                            fut_emb_indices[dim_mask] = semantic_part_offset + tag_layer_idx * self.max_tag_size + dim_ids
                        else:
                            fut_emb_indices[dim_mask] = self.padding_idx
                else:
                    # Non-interleaved mode (concatenated or semantic IDs only)
                    if i < self.n_sem_layers:  # The first n_sem_layers dimensions are semantic IDs
                        # Index calculation for semantic IDs
                        fut_emb_indices[dim_mask] = i * self.num_embeddings + dim_ids
                    else:
                        # Index calculation for tag IDs
                        tag_dim_index = i - self.n_sem_layers
                        if tag_dim_index < self.n_tag_layers:
                            fut_emb_indices[dim_mask] = semantic_part_offset + tag_dim_index * self.max_tag_size + dim_ids
                        else:
                            fut_emb_indices[dim_mask] = self.padding_idx
            
            # Embed the future sequence
            fut_embs = self.emb(fut_emb_indices)
        else:
            fut_embs = None
        
        return SemIdEmbeddingBatch(seq=seq_embs, fut=fut_embs)
    

class UserIdEmbedder(nn.Module):
    """
    Embeds user IDs into a dense vector representation using the hashing trick.

    This is useful for handling a very large or unbounded vocabulary of user IDs
    by mapping them to a fixed number of buckets.
    """
    # TODO: Further explore and implement more sophisticated hashing trick embeddings.
    def __init__(self, num_buckets, embedding_dim) -> None:
        """
        Initializes the UserIdEmbedder.

        Args:
            num_buckets (int): The number of hash buckets to use.
            embedding_dim (int): The dimensionality of the output embeddings.
        """
        super().__init__()
        self.num_buckets = num_buckets
        self.emb = nn.Embedding(num_buckets, embedding_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for embedding user IDs.

        Args:
            x (Tensor): A tensor of user IDs.

        Returns:
            Tensor: The resulting embeddings.
        """
        # Use the modulo operator as a simple hashing function
        hashed_indices = x % self.num_buckets
        # For non-integer inputs, a more robust hash function might be needed:
        # hashed_indices = torch.tensor([hash(token) % self.num_buckets for token in x], device=x.device)
        return self.emb(hashed_indices)
