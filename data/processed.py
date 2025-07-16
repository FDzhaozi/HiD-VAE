import gin
import os
import random
import torch

from data.amazon import AmazonReviews
from data.ml1m import RawMovieLens1M
from data.ml32m import RawMovieLens32M
from data.load_kuairand import KuaiRandItemData
from data.load_kuairand import KuaiRandDataset
from data.schemas import SeqBatch
from enum import Enum
from torch import Tensor
from torch.utils.data import Dataset
from typing import Optional

PROCESSED_MOVIE_LENS_SUFFIX = "/processed/data.pt"


@gin.constants_from_enum
class RecDataset(Enum):
    AMAZON = 1
    ML_1M = 2
    ML_32M = 3
    KUAIRAND = 4


DATASET_NAME_TO_RAW_DATASET = {
    RecDataset.AMAZON: AmazonReviews,
    RecDataset.ML_1M: RawMovieLens1M,
    RecDataset.ML_32M: RawMovieLens32M,
    RecDataset.KUAIRAND: KuaiRandItemData
}


DATASET_NAME_TO_MAX_SEQ_LEN = {
    RecDataset.AMAZON: 20,
    RecDataset.ML_1M: 200,
    RecDataset.ML_32M: 200,
    RecDataset.KUAIRAND: 50  # 为KuaiRand设置合适的最大序列长度
}


class ItemData(Dataset):
    def __init__(
        self,
        root: str,
        *args,
        force_process: bool = False,
        dataset: RecDataset = RecDataset.ML_1M,
        train_test_split: str = "all",
        **kwargs
    ) -> None:
        self.root = root
        self.train_test_split = train_test_split

        if dataset == RecDataset.ML_1M:
            self.data = RawMovieLens1M(root, force_process=force_process)
        elif dataset == RecDataset.ML_32M:
            self.data = RawMovieLens32M(root, force_process=force_process)
        elif dataset == RecDataset.AMAZON:
            self.data = AmazonReviews(root, force_reload=force_process, split=kwargs.get("split", None))
        elif dataset == RecDataset.KUAIRAND:
            self.data = KuaiRandItemData(root, force_process=force_process, train_test_split=train_test_split, **kwargs)
        else:
            raise ValueError(f"Unknown dataset {dataset}")

        # KuaiRand数据集已经在KuaiRandItemData中处理好了，不需要额外处理
        if dataset == RecDataset.KUAIRAND:
            self.item_data = self.data.item_data
            self.item_text = self.data.item_text
            return

        processed_data_path = self.data.processed_paths[0]
        print(f"processed_data_path: {processed_data_path}")
        if not os.path.exists(processed_data_path) or force_process:
            self.data.process(max_seq_len=DATASET_NAME_TO_MAX_SEQ_LEN[dataset])
        
        if train_test_split == "train":
            filt = self.data.data["item"]["is_train"]
        elif train_test_split == "eval":
            filt = ~self.data.data["item"]["is_train"]
        elif train_test_split == "all":
            filt = torch.ones_like(self.data.data["item"]["x"][:,0], dtype=bool)

        self.item_data = self.data.data["item"]["x"][filt]
        self.item_text = self.data.data["item"]["text"][filt]

    def __len__(self):
        return self.item_data.shape[0]

    def __getitem__(self, idx):
        item_ids = torch.tensor(idx).unsqueeze(0) if not isinstance(idx, torch.Tensor) else idx
        x = self.item_data[idx, :768]
        # print(f"idx: {idx}")
        # print(f"len of idx: {len(idx)}")
        # print(f"type of idx: {type(idx)}")
        return SeqBatch(
            # 用户ID设为-1（表示未使用用户信息）
            user_ids=-1 * torch.ones_like(item_ids.squeeze(0)),
            ids=item_ids,
            # 未来商品ID设为-1（非序列任务）
            ids_fut=-1 * torch.ones_like(item_ids.squeeze(0)),
            x=x,
            # 未来商品特征设为-1
            x_fut=-1 * torch.ones_like(item_ids.squeeze(0)),
            seq_mask=torch.ones_like(item_ids, dtype=bool)
        )


class SeqData(Dataset):
    def __init__(
        self,
        root: str,
        *args,
        is_train: bool = True,
        subsample: bool = False,
        force_process: bool = False,
        dataset: RecDataset = RecDataset.ML_1M,
        **kwargs
    ) -> None:
        
        assert (not subsample) or is_train, "Can only subsample on training split."

        raw_dataset_class = DATASET_NAME_TO_RAW_DATASET[dataset]
        max_seq_len = DATASET_NAME_TO_MAX_SEQ_LEN[dataset]
        self._max_seq_len = max_seq_len
        self.split = "train" if is_train else "test"
        self.subsample = subsample
        
        # 特殊处理KuaiRand数据集
        if dataset == RecDataset.KUAIRAND:
            # KuaiRand数据集使用不同的加载方式
            raw_data = raw_dataset_class(
                root=root, 
                train_test_split=self.split, 
                force_process=force_process,
                **kwargs
            )
            
            # 直接使用KuaiRand的数据
            self.item_data = raw_data.item_data
            
            # 创建简单的序列数据结构
            # 对于KuaiRand，我们创建一个简单的序列数据结构，每个用户只有一个物品
            num_items = len(raw_data)
            
            # 创建用户ID (简单地使用索引)
            user_ids = torch.arange(num_items)
            
            # 创建物品序列 (每个用户只有一个物品)
            item_ids = torch.arange(num_items).unsqueeze(1)  # [num_items, 1]
            
            # 创建未来物品ID (简单地使用下一个物品)
            future_item_ids = torch.arange(1, num_items+1) % num_items
            
            self.sequence_data = {
                "userId": user_ids,
                "itemId": item_ids,
                "itemId_fut": future_item_ids
            }
            
        else:
            # 原始处理逻辑，用于其他数据集
            raw_data = raw_dataset_class(root=root, *args, **kwargs)

            processed_data_path = raw_data.processed_paths[0]
            if not os.path.exists(processed_data_path) or force_process:
                raw_data.process(max_seq_len=max_seq_len)

            self.sequence_data = raw_data.data[("user", "rated", "item")]["history"][self.split]

            if not self.subsample:
                self.sequence_data["itemId"] = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(l[-max_seq_len:]) for l in self.sequence_data["itemId"]],
                    batch_first=True,
                    padding_value=-1
                )

            self.item_data = raw_data.data["item"]["x"]
    
    
    @property
    def max_seq_len(self):
        return self._max_seq_len

    def __len__(self):
        return self.sequence_data["userId"].shape[0]
  
    def __getitem__(self, idx):
        user_ids = self.sequence_data["userId"][idx]
        
        if self.subsample:
            # 检查sequence_data["itemId"][idx]是否为张量
            if isinstance(self.sequence_data["itemId"][idx], torch.Tensor) and self.sequence_data["itemId"][idx].dim() == 1:
                seq = self.sequence_data["itemId"][idx].tolist() + [self.sequence_data["itemId_fut"][idx].item()]
            else:
                seq = self.sequence_data["itemId"][idx] + self.sequence_data["itemId_fut"][idx].tolist()
            
            # 确保序列长度足够进行子采样
            if len(seq) >= 3:
                start_idx = random.randint(0, max(0, len(seq)-3))
                end_idx = random.randint(start_idx+3, start_idx+self.max_seq_len+1)
                sample = seq[start_idx:end_idx]
            else:
                sample = seq
                
            item_ids = torch.tensor(sample[:-1] + [-1] * (self.max_seq_len - len(sample[:-1])))
            item_ids_fut = torch.tensor([sample[-1]])

        else:
            item_ids = self.sequence_data["itemId"][idx]
            item_ids_fut = self.sequence_data["itemId_fut"][idx]
        
        assert (item_ids >= -1).all(), "Invalid movie id found"
        
        # 确保item_ids和item_ids_fut是正确的形状
        if isinstance(item_ids, torch.Tensor) and item_ids.dim() == 0:
            item_ids = item_ids.unsqueeze(0)
        
        if isinstance(item_ids_fut, torch.Tensor) and item_ids_fut.dim() == 0:
            item_ids_fut = item_ids_fut.unsqueeze(0)
            
        # 确保索引在有效范围内
        valid_item_ids = torch.clamp(item_ids, min=-1, max=len(self.item_data)-1)
        valid_item_ids_fut = torch.clamp(item_ids_fut, min=-1, max=len(self.item_data)-1)
        
        # 获取特征
        x = torch.zeros((valid_item_ids.shape[0], 768), dtype=self.item_data.dtype, device=self.item_data.device)
        for i, item_id in enumerate(valid_item_ids):
            if item_id >= 0:
                x[i] = self.item_data[item_id, :768]
            else:
                x[i] = -1
                
        x_fut = torch.zeros((valid_item_ids_fut.shape[0], 768), dtype=self.item_data.dtype, device=self.item_data.device)
        for i, item_id in enumerate(valid_item_ids_fut):
            if item_id >= 0:
                x_fut[i] = self.item_data[item_id, :768]
            else:
                x_fut[i] = -1

        return SeqBatch(
            user_ids=user_ids,
            ids=item_ids,
            ids_fut=item_ids_fut,
            x=x,
            x_fut=x_fut,
            seq_mask=(item_ids >= 0)
        )


if __name__ == "__main__":
    dataset = ItemData("dataset/amazon", dataset=RecDataset.AMAZON, split="beauty", force_process=True)
    dataset[0]
    import pdb; pdb.set_trace()
