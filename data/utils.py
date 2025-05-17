from data.schemas import SeqBatch, TaggedSeqBatch

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


def batch_to(batch, device):
    """
    将批次数据移动到指定设备
    
    参数:
        batch: 批次数据，可以是SeqBatch或TaggedSeqBatch类型
        device: 目标设备
        
    返回:
        移动到目标设备的批次数据
    """
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
        # 处理带标签的TaggedSeqBatch
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
        # 如果不是命名元组，直接移动到设备
        return batch.to(device)


def next_batch(dataloader, device):
    batch = next(dataloader)
    return batch_to(batch, device)