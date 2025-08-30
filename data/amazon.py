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
        print("\n=== Starting Amazon dataset processing ===")
        data = HeteroData()

        print(f"\nLoading {self.split} dataset mapping files...")
        with open(os.path.join(self.raw_dir, self.split, "datamaps.json"), 'r') as f:
            data_maps = json.load(f)    
        print(f"Number of item ID mappings: {len(data_maps['item2id'])}")

        print("\nBuilding user sequences...")
        sequences = self.train_test_split(max_seq_len=max_seq_len)
        print(f"Training sequences count: {len(sequences['train'])}")
        print(f"Validation sequences count: {len(sequences['eval'])}")
        print(f"Test sequences count: {len(sequences['test'])}")
        print("\nSequence example:")
        print(f"User ID: {sequences['train']['userId'][0]}")
        print(f"Item sequence: {sequences['train']['itemId'][0][:5]}... (first 5 items)")
        print(f"Target item: {sequences['train']['itemId_fut'][0]}")

        data["user", "rated", "item"].history = {
            k: self._df_to_tensor_dict(v, ["itemId"])
            for k, v in sequences.items() 
        }
        
        print("\nProcessing item features...")
        asin2id = pd.DataFrame([{"asin": k, "id": self._remap_ids(int(v))} for k, v in data_maps["item2id"].items()])
        print(f"ASIN to ID mappings count: {len(asin2id)}")

        print("\nLoading item metadata...")
        item_data = (
            pd.DataFrame([
                meta for meta in
                parse(path=os.path.join(self.raw_dir, self.split, "meta.json.gz"))
            ])
            .merge(asin2id, on="asin")
            .sort_values(by="id")
            .fillna({"brand": "Unknown"})
        )
        print(f"Total items: {len(item_data)}")
        print("\nItem data example:")
        print(item_data.iloc[0][["title", "brand", "categories", "price"]].to_dict())

        print("\nConstructing item text descriptions...")
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
        print("\nText description example:")
        print(sentences.iloc[0])
        
        print("\nStarting text encoding...")
        item_emb = self._encode_text_feature(sentences)
        print(f"Text feature dimensions: {item_emb.shape}")
        print(f"Encoded example (first 5 dims): {item_emb[0,:5]}")
        
        data['item'].x = item_emb
        data['item'].text = np.array(sentences)

        print("\nSplitting train and test sets...")
        gen = torch.Generator()
        gen.manual_seed(42)
        data['item'].is_train = torch.rand(item_emb.shape[0], generator=gen) > 0.05
        print(f"Training items count: {data['item'].is_train.sum().item()}")
        print(f"Test items count: {(~data['item'].is_train).sum().item()}")

        print("\nSaving processed data...")
        self.save([data], self.processed_paths[0])
        print("=== Dataset processing completed ===\n")


if __name__ == "__main__":
    dataset = AmazonReviews(root="dataset/amazon", split="beauty", force_reload=True)
