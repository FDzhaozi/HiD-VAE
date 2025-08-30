from data.schemas import SeqBatch, TaggedSeqBatch

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


def batch_to(batch, device):
    if isinstance(batch, SeqBatch):
        # 处理普通SeqBatch
        return SeqBatch(
            user_ids=batch.user_ids.to(device),
            ids=batch.ids.to(device),
            ids_fut=batch.ids_fut.to(device),
            x=batch.x.to(device),
            x_fut=batch.x_fut.to(device),
            seq_mask=batch.seq_mask.to(device)
        )
    elif isinstance(batch, TaggedSeqBatch):
        return TaggedSeqBatch(
            user_ids=batch.user_ids.to(device),
            ids=batch.ids.to(device),
            ids_fut=batch.ids_fut.to(device),
            x=batch.x.to(device),
            x_fut=batch.x_fut.to(device),
            seq_mask=batch.seq_mask.to(device),
            tags_emb=batch.tags_emb.to(device),
            tags_indices=batch.tags_indices.to(device)
        )
    else:
        return batch.to(device)


def next_batch(dataloader, device):
    batch = next(dataloader)
    return batch_to(batch, device)
