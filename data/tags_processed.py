import gin
import os
import random
import torch
import torch.nn.functional as F

from data.tags_amazon import AmazonReviews
from data.ml1m import RawMovieLens1M
from data.ml32m import RawMovieLens32M
from data.tags_kuairand import KuaiRand
from data.schemas import SeqBatch, TaggedSeqBatch  # 正确导入TaggedSeqBatch
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
    RecDataset.KUAIRAND: KuaiRand
}


DATASET_NAME_TO_MAX_SEQ_LEN = {
    RecDataset.AMAZON: 20,
    RecDataset.ML_1M: 200,
    RecDataset.ML_32M: 200,
    RecDataset.KUAIRAND: 20
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

        print(f"root: {root}")
        print(f"dataset: {dataset}")
        
        raw_dataset_class = DATASET_NAME_TO_RAW_DATASET[dataset]
        max_seq_len = DATASET_NAME_TO_MAX_SEQ_LEN[dataset]

        raw_data = raw_dataset_class(root=root, *args, **kwargs)
        
        processed_data_path = raw_data.processed_paths[0]
        print(f"processed_data_path: {processed_data_path}")
        if not os.path.exists(processed_data_path) or force_process:
            raw_data.process(max_seq_len=max_seq_len)
        
        # 新增: 打印训练和验证集的物品比例
        if "is_train" in raw_data.data["item"]:
            is_train_flags = raw_data.data["item"]["is_train"]
            num_total_items = len(is_train_flags)
            if num_total_items > 0:
                num_train_items = is_train_flags.sum().item()
                num_eval_items = num_total_items - num_train_items
                train_ratio = num_train_items / num_total_items
                eval_ratio = num_eval_items / num_total_items
                print(f"物品数据划分比例: 总数={num_total_items}, "
                      f"训练集={num_train_items} ({train_ratio:.2%}), "
                      f"验证集={num_eval_items} ({eval_ratio:.2%})")

        if train_test_split == "train":
            filt = raw_data.data["item"]["is_train"]
        elif train_test_split == "eval":
            filt = ~raw_data.data["item"]["is_train"]
        elif train_test_split == "all":
            filt = torch.ones_like(raw_data.data["item"]["x"][:,0], dtype=bool)

        self.item_data = raw_data.data["item"]["x"][filt]
        
        # 增加对item_text加载的鲁棒性
        if "text" in raw_data.data["item"]:
            try:
                self.item_text = raw_data.data["item"]["text"][filt]
            except (TypeError, IndexError) as e:
                print(f"警告: 无法加载 item_text: {e}。将跳过此字段。")
                self.item_text = None
        else:
            self.item_text = None
        
        # 加载新增的标签嵌入和标签索引数据
        if "tags_emb" in raw_data.data["item"] and "tags_indices" in raw_data.data["item"]:
            self.tags_emb = raw_data.data["item"]["tags_emb"][filt]
            self.tags_indices = raw_data.data["item"]["tags_indices"][filt]
            self.has_tags = True
        else:
            self.has_tags = False
            print("警告: 数据集中没有找到标签嵌入或标签索引数据")

    def __len__(self):
        return self.item_data.shape[0]

    def __getitem__(self, idx):
        item_ids = torch.tensor(idx).unsqueeze(0) if not isinstance(idx, torch.Tensor) else idx
        x = self.item_data[idx, :768]
        
        # 构建基本的批次数据
        batch_data = {
            "user_ids": -1 * torch.ones_like(item_ids.squeeze(0)),
            "ids": item_ids,
            "ids_fut": -1 * torch.ones_like(item_ids.squeeze(0)),
            "x": x,
            "x_fut": -1 * torch.ones_like(item_ids.squeeze(0)),
            "seq_mask": torch.ones_like(item_ids, dtype=bool)
        }
        
        # 如果有标签数据，则使用TaggedSeqBatch
        if self.has_tags:
            # 形状
            # print(f"self.tags_emb.shape: {self.tags_emb.shape}")
            # print(f"self.tags_indices.shape: {self.tags_indices.shape}")
            # print(f"idx: {idx}")
            # print(f"type of idx: {type(idx)}")
            
            # 修改处理逻辑以支持列表类型的idx
            if isinstance(idx, torch.Tensor):
                tags_emb = self.tags_emb[idx]
                tags_indices = self.tags_indices[idx]
            elif isinstance(idx, list):
                # 处理列表类型的idx
                tags_emb = self.tags_emb[idx]
                tags_indices = self.tags_indices[idx]
            else:
                # 处理整数类型的idx
                tags_emb = self.tags_emb[idx:idx+1]
                tags_indices = self.tags_indices[idx:idx+1]
            
            return TaggedSeqBatch(
                user_ids=batch_data["user_ids"],
                ids=batch_data["ids"],
                ids_fut=batch_data["ids_fut"],
                x=batch_data["x"],
                x_fut=batch_data["x_fut"],
                seq_mask=batch_data["seq_mask"],
                tags_emb=tags_emb,
                tags_indices=tags_indices
            )
        else:
            # 如果没有标签数据，则使用普通的SeqBatch
            return SeqBatch(**batch_data)


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

        raw_data = raw_dataset_class(root=root, *args, **kwargs)

        processed_data_path = raw_data.processed_paths[0]
        if not os.path.exists(processed_data_path) or force_process:
            raw_data.process(max_seq_len=max_seq_len)

        split = "train" if is_train else "test"
        self.subsample = subsample
        self.sequence_data = raw_data.data[("user", "rated", "item")]["history"][split]

        if not self.subsample:
            self.sequence_data["itemId"] = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(l[-max_seq_len:]) for l in self.sequence_data["itemId"]],
                batch_first=True,
                padding_value=-1
            )

        self._max_seq_len = max_seq_len
        self.item_data = raw_data.data["item"]["x"]
        
        # 加载新增的标签嵌入和标签索引数据
        if "tags_emb" in raw_data.data["item"] and "tags_indices" in raw_data.data["item"]:
            self.tags_emb = raw_data.data["item"]["tags_emb"]
            self.tags_indices = raw_data.data["item"]["tags_indices"]
            self.has_tags = True
        else:
            self.has_tags = False
            print("警告: 数据集中没有找到标签嵌入或标签索引数据")
            
        self.split = split
    
    @property
    def max_seq_len(self):
        return self._max_seq_len

    def __len__(self):
        return self.sequence_data["userId"].shape[0]
  
    def __getitem__(self, idx):
        user_ids = self.sequence_data["userId"][idx]
        
        if self.subsample:
            seq = self.sequence_data["itemId"][idx] + self.sequence_data["itemId_fut"][idx].tolist()
            start_idx = random.randint(0, max(0, len(seq)-3))
            end_idx = random.randint(start_idx+3, start_idx+self.max_seq_len+1)
            sample = seq[start_idx:end_idx]
            
            item_ids = torch.tensor(sample[:-1] + [-1] * (self.max_seq_len - len(sample[:-1])))
            item_ids_fut = torch.tensor([sample[-1]])

        else:
            item_ids = self.sequence_data["itemId"][idx]
            item_ids_fut = self.sequence_data["itemId_fut"][idx]
        
        assert (item_ids >= -1).all(), "Invalid movie id found"
        x = self.item_data[item_ids, :768]
        x[item_ids == -1] = -1

        x_fut = self.item_data[item_ids_fut, :768]
        x_fut[item_ids_fut == -1] = -1
        
        # 构建基本的批次数据
        batch_data = {
            "user_ids": user_ids,
            "ids": item_ids,
            "ids_fut": item_ids_fut,
            "x": x,
            "x_fut": x_fut,
            "seq_mask": (item_ids >= 0)
        }
        
        # 如果有标签数据，则使用TaggedSeqBatch
        if self.has_tags:
            # 获取序列中每个商品的标签嵌入和索引
            tags_emb = self.tags_emb[item_ids]
            tags_indices = self.tags_indices[item_ids]
            # 对于无效的商品ID，将标签数据设为-1
            tags_emb[item_ids == -1] = -1
            tags_indices[item_ids == -1] = -1
            
            # 获取未来商品的标签嵌入和索引
            tags_emb_fut = self.tags_emb[item_ids_fut]
            tags_indices_fut = self.tags_indices[item_ids_fut]
            # 对于无效的未来商品ID，将标签数据设为-1
            tags_emb_fut[item_ids_fut == -1] = -1
            tags_indices_fut[item_ids_fut == -1] = -1
            
            return TaggedSeqBatch(
                user_ids=batch_data["user_ids"],
                ids=batch_data["ids"],
                ids_fut=batch_data["ids_fut"],
                x=batch_data["x"],
                x_fut=batch_data["x_fut"],
                seq_mask=batch_data["seq_mask"],
                tags_emb=tags_emb,
                tags_indices=tags_indices
            )
        else:
            # 如果没有标签数据，则使用普通的SeqBatch
            return SeqBatch(**batch_data)


if __name__ == "__main__":
    dataset = ItemData("dataset/amazon", dataset=RecDataset.AMAZON, split="beauty", force_process=True)
    sample = dataset[0]
    print(f"样本数据形状: {sample.x.shape}")
    if hasattr(sample, 'tags_emb') and sample.tags_emb is not None:
        print(f"标签嵌入形状: {sample.tags_emb.shape}")
        print(f"标签索引形状: {sample.tags_indices.shape}")
    import pdb; pdb.set_trace()
