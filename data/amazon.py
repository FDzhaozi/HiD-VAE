import gzip
import json
import numpy as np
import os
import os.path as osp
import pandas as pd
import polars as pl
import torch

from collections import defaultdict
from data.preprocessing import PreprocessingMixin
from torch_geometric.data import download_google_url
from torch_geometric.data import extract_zip
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from torch_geometric.io import fs
from typing import Callable
from typing import List
from typing import Optional


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


class AmazonReviews(InMemoryDataset, PreprocessingMixin):
    gdrive_id = "1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G"
    gdrive_filename = "P5_data.zip"

    def __init__(
        self,
        root: str,
        split: str,  # 'beauty', 'sports', 'toys'
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.split = split
        super(AmazonReviews, self).__init__(
            root, transform, pre_transform, force_reload
        )
        self.load(self.processed_paths[0], data_cls=HeteroData)
    
    @property
    def raw_file_names(self) -> List[str]:
        return [self.split]
    
    @property
    def processed_file_names(self) -> str:
        #return f'data_{self.split}.pt'
        return f'title_data_beauty_5tags.pt'
    
    def download(self) -> None:
        path = download_google_url(self.gdrive_id, self.root, self.gdrive_filename)
        extract_zip(path, self.root)
        os.remove(path)
        folder = osp.join(self.root, 'data')
        fs.rm(self.raw_dir)
        os.rename(folder, self.raw_dir)
    
    def _remap_ids(self, x):
        return x - 1

    def train_test_split(self, max_seq_len=20):
        splits = ["train", "eval", "test"]
        sequences = {sp: defaultdict(list) for sp in splits}
        user_ids = []
        with open(os.path.join(self.raw_dir, self.split, "sequential_data.txt"), "r") as f:
            for line in f:
                parsed_line = list(map(int, line.strip().split()))
                user_ids.append(parsed_line[0])
                items = [self._remap_ids(id) for id in parsed_line[1:]]
                
                # We keep the whole sequence without padding. Allows flexible training-time subsampling.
                train_items = items[:-2]
                sequences["train"]["itemId"].append(train_items)
                sequences["train"]["itemId_fut"].append(items[-2])
                
                eval_items = items[-(max_seq_len+2):-2]
                sequences["eval"]["itemId"].append(eval_items + [-1] * (max_seq_len - len(eval_items)))
                sequences["eval"]["itemId_fut"].append(items[-2])
                
                test_items = items[-(max_seq_len+1):-1]
                sequences["test"]["itemId"].append(test_items + [-1] * (max_seq_len - len(test_items)))
                sequences["test"]["itemId_fut"].append(items[-1])
        
        for sp in splits:
            sequences[sp]["userId"] = user_ids
            sequences[sp] = pl.from_dict(sequences[sp])
        return sequences
    
    def process(self, max_seq_len=20) -> None:
        print("\n=== 开始处理Amazon数据集 ===")
        data = HeteroData()

        print(f"\n正在加载 {self.split} 数据集的映射文件...")
        with open(os.path.join(self.raw_dir, self.split, "datamaps.json"), 'r') as f:
            data_maps = json.load(f)    
        print(f"商品ID映射数量: {len(data_maps['item2id'])}")

        print("\n构建用户序列...")
        sequences = self.train_test_split(max_seq_len=max_seq_len)
        print(f"训练集序列数: {len(sequences['train'])}")
        print(f"验证集序列数: {len(sequences['eval'])}")
        print(f"测试集序列数: {len(sequences['test'])}")
        print("\n序列示例:")
        print(f"用户ID: {sequences['train']['userId'][0]}")
        print(f"商品序列: {sequences['train']['itemId'][0][:5]}... (展示前5个)")
        print(f"目标商品: {sequences['train']['itemId_fut'][0]}")

        data["user", "rated", "item"].history = {
            k: self._df_to_tensor_dict(v, ["itemId"])
            for k, v in sequences.items() 
        }
        
        print("\n处理商品特征...")
        asin2id = pd.DataFrame([{"asin": k, "id": self._remap_ids(int(v))} for k, v in data_maps["item2id"].items()])
        print(f"商品ASIN到ID的映射数量: {len(asin2id)}")

        print("\n加载商品元数据...")
        item_data = (
            pd.DataFrame([
                meta for meta in
                parse(path=os.path.join(self.raw_dir, self.split, "meta.json.gz"))
            ])
            .merge(asin2id, on="asin")
            .sort_values(by="id")
            .fillna({"brand": "Unknown"})
        )
        print(f"商品总数: {len(item_data)}")
        print("\n商品数据示例:")
        print(item_data.iloc[0][["title", "brand", "categories", "price"]].to_dict())

        print("\n构建商品文本描述...")
        sentences = item_data.apply(
            lambda row:
                "Title: " +
                str(row["title"]) + "; " +
                "Brand: " +
                str(row["brand"]) + "; " +
                "Categories: " +
                str(row["categories"][0]) + "; " + 
                "Price: " +
                str(row["price"]) + "; ",
            axis=1
        )
        print("\n文本描述示例:")
        print(sentences.iloc[0])
        
        print("\n开始文本编码...")
        item_emb = self._encode_text_feature(sentences)
        print(f"文本特征维度: {item_emb.shape}")
        print(f"编码示例(前5维): {item_emb[0,:5]}")
        
        data['item'].x = item_emb
        data['item'].text = np.array(sentences)

        print("\n划分训练集和测试集...")
        gen = torch.Generator()
        gen.manual_seed(42)
        data['item'].is_train = torch.rand(item_emb.shape[0], generator=gen) > 0.05
        print(f"训练集商品数: {data['item'].is_train.sum().item()}")
        print(f"测试集商品数: {(~data['item'].is_train).sum().item()}")

        print("\n保存处理后的数据...")
        self.save([data], self.processed_paths[0])
        print("=== 数据处理完成 ===\n")
        



if __name__ == "__main__":
    dataset = AmazonReviews(root="dataset/amazon", split="beauty", force_reload=True)
