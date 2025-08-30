from typing import NamedTuple
from torch import Tensor

FUT_SUFFIX = "_fut"


class SeqBatch(NamedTuple):
    user_ids: Tensor
    ids: Tensor
    ids_fut: Tensor
    x: Tensor
    x_fut: Tensor
    seq_mask: Tensor


class TaggedSeqBatch(NamedTuple):
    user_ids: Tensor
    ids: Tensor
    ids_fut: Tensor
    x: Tensor
    x_fut: Tensor
    seq_mask: Tensor
    tags_emb: Tensor
    tags_indices: Tensor


class TokenizedSeqBatch(NamedTuple):
    user_ids: Tensor
    sem_ids: Tensor
    sem_ids_fut: Tensor
    seq_mask: Tensor
    token_type_ids: Tensor
    token_type_ids_fut: Tensor


class TaggedTokenizedSeqBatch(NamedTuple):
    user_ids: Tensor
    sem_ids: Tensor
    sem_ids_fut: Tensor
    seq_mask: Tensor
    token_type_ids: Tensor
    token_type_ids_fut: Tensor
    tags_emb: Tensor
    tags_indices: Tensor


# class HRqVaeOutput(NamedTuple):
#     embeddings: Tensor
#     residuals: Tensor
#     sem_ids: Tensor
#     quantize_loss: Tensor
#     tag_align_loss: Tensor
#     tag_pred_loss: Tensor
#     tag_pred_accuracy: Tensor


class HRqVaeComputedLosses(NamedTuple):
    loss: Tensor
    reconstruction_loss: Tensor
    rqvae_loss: Tensor
    tag_align_loss: Tensor
    tag_pred_loss: Tensor
    tag_pred_accuracy: Tensor
    embs_norm: Tensor
    p_unique_ids: Tensor
    tag_align_loss_by_layer: Tensor = None
    tag_pred_loss_by_layer: Tensor = None
    tag_pred_accuracy_by_layer: Tensor = None
    sem_id_uniqueness_loss: Tensor = None


from typing import Optional, List, Dict, Any, Union

class HRqVaeOutput:
    def __init__(
        self,
        embeddings: Tensor,
        residuals: Tensor,
        sem_ids: Tensor,
        quantize_loss: Tensor,
        tag_align_loss: Tensor,
        tag_pred_loss: Tensor,
        tag_pred_accuracy: Tensor,
        tag_align_loss_by_layer: Optional[Tensor] = None,
        tag_pred_loss_by_layer: Optional[Tensor] = None,
        tag_pred_accuracy_by_layer: Optional[Tensor] = None,
    ) -> None:
        self.embeddings = embeddings
        self.residuals = residuals
        self.sem_ids = sem_ids
        self.quantize_loss = quantize_loss
        self.tag_align_loss = tag_align_loss
        self.tag_pred_loss = tag_pred_loss
        self.tag_pred_accuracy = tag_pred_accuracy
        self.tag_align_loss_by_layer = tag_align_loss_by_layer
        self.tag_pred_loss_by_layer = tag_pred_loss_by_layer
        self.tag_pred_accuracy_by_layer = tag_pred_accuracy_by_layer
