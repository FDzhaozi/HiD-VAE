import gzip
import json
import numpy as np
import os
import os.path as osp
import pandas as pd
import polars as pl
import torch
import psutil
import gc
import nltk
from nltk.corpus import stopwords

# Add support for numpy._core.multiarray._reconstruct
from torch.serialization import add_safe_globals
try:
    from numpy._core.multiarray import _reconstruct
    add_safe_globals([_reconstruct])
except ImportError:
    # If direct import fails, try registering via string
    add_safe_globals(['numpy._core.multiarray._reconstruct'])

from collections import defaultdict
from data.tags_preprocessing import PreprocessingMixin
from torch_geometric.data import download_google_url
from torch_geometric.data import extract_zip
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from torch_geometric.io import fs
from typing import Callable
from typing import List
from typing import Optional
from sentence_transformers import SentenceTransformer

# Download NLTK stopwords
# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('stopwords')

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
        self.force_reload = force_reload  # Save the force_reload parameter
        
        print(f"\nInitializing AmazonReviews dataset: split={split}, force_reload={force_reload}")
        
        # If force_reload is True, delete the processed file first
        if force_reload:
            processed_file_path = osp.join(osp.join(root, 'processed'), f'title_data_{split}_5tags.pt')
            if osp.exists(processed_file_path):
                print(f"Force reloading: Deleting existing processed file {processed_file_path}")
                os.remove(processed_file_path)
                print("File deleted. Reprocessing data.")
        
        super(AmazonReviews, self).__init__(
            root, transform, pre_transform, force_reload
        )
        
        # Modify torch_geometric's loading behavior to ensure weights_only=False
        # Save the original function
        original_torch_load = torch.load
        
        # Create a wrapper function to force weights_only=False
        def patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        # Temporarily replace torch.load
        torch.load = patched_torch_load
        
        try:
            # Try to load the data
            print("Attempting to load data with weights_only=False...")
            self.load(self.processed_paths[0], data_cls=HeteroData)
            print("Data loaded successfully!")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
        finally:
            # Restore the original function
            torch.load = original_torch_load
    
    @property
    def raw_file_names(self) -> List[str]:
        return [self.split]
    
    @property
    def processed_file_names(self) -> str:
        return f'title_data_{self.split}_5tags.pt'
    
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
        def show_memory_usage():
            process = psutil.Process()
            print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
        print("\n=== Start processing Amazon dataset ===")
        show_memory_usage()
        data = HeteroData()

        print(f"\nLoading mapping files for {self.split} dataset...")
        with open(os.path.join(self.raw_dir, self.split, "datamaps.json"), 'r') as f:
            data_maps = json.load(f)    
        print(f"Number of item ID mappings: {len(data_maps['item2id'])}")

        print("\nBuilding user sequences...")
        sequences = self.train_test_split(max_seq_len=max_seq_len)
        print(f"Number of training sequences: {len(sequences['train'])}")
        print(f"Number of evaluation sequences: {len(sequences['eval'])}")
        print(f"Number of test sequences: {len(sequences['test'])}")
        print("\nSequence example:")
        print(f"User ID: {sequences['train']['userId'][0]}")
        print(f"Item sequence: {sequences['train']['itemId'][0][:5]}... (showing first 5)")
        print(f"Target item: {sequences['train']['itemId_fut'][0]}")

        data["user", "rated", "item"].history = {
            k: self._df_to_tensor_dict(v, ["itemId"])
            for k, v in sequences.items() 
        }
        
        print("\nProcessing item features...")
        asin2id = pd.DataFrame([{"asin": k, "id": self._remap_ids(int(v))} for k, v in data_maps["item2id"].items()])
        print(f"Number of item ASIN to ID mappings: {len(asin2id)}")

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
        print(f"Total number of items: {len(item_data)}")
        print("\nItem data example:")
        print(item_data.iloc[0][["title", "brand", "categories", "price"]].to_dict())

        # Flatten category list
        def flatten_categories(categories):
            """Flattens a nested list of categories into a single list."""
            flattened = []
            for cat in categories:
                if isinstance(cat, list):
                    flattened.extend(cat)
                else:
                    flattened.append(cat)
            return list(dict.fromkeys(flattened))  # Remove duplicates
        
        print("\nProcessing item categories...")
        item_data['flat_categories'] = item_data['categories'].apply(flatten_categories)
        
        # Show category processing example
        print("\nCategory flattening example:")
        sample_idx = 0
        print(f"Original categories: {item_data.iloc[sample_idx]['categories']}")
        print(f"Flattened: {item_data.iloc[sample_idx]['flat_categories']}")
        
        # Process categories to ensure each item has 5 tags
        def process_categories_to_five_tags(row):
            import re
            import random
            
            # Get flattened categories
            cats = row['flat_categories']
            
            # Remove the first tag (if it exists)
            if len(cats) > 0:
                cats = cats[1:]
            
            # Get stopwords
            stop_words = set(stopwords.words('english'))
            
            # If less than 5 categories, extract words from the title to supplement
            if len(cats) < 5:
                # Extract words from the title
                title_words = re.findall(r'\b[A-Za-z]{3,}\b', str(row['title']))
                # Remove stopwords, duplicates, and words already in categories
                title_words = [w for w in title_words if w.lower() not in stop_words and 
                                w.lower() not in [c.lower() for c in cats]]
                
                # If title words are not enough, add the brand
                if len(title_words) + len(cats) < 5 and row['brand'] != "Unknown":
                    if row['brand'].lower() not in [c.lower() for c in cats]:
                        title_words.append(row['brand'])
                
                # Randomly select enough words to make up 5
                random.seed(42 + row['id'])  # Ensure reproducible results
                needed = 5 - len(cats)
                
                selected_words = []
                while len(selected_words) < needed:
                    if len(title_words) > 0:
                        word = random.choice(title_words)
                        title_words.remove(word)  # Avoid re-selecting the same word
                        if word not in selected_words and word.strip() != "":
                            selected_words.append(word)
                    else:
                        # If words are still not enough, use generic tags
                        tag_idx = len(selected_words) + 1
                        selected_words.append(f"GenericTag{tag_idx}")
                
                # Merge categories and selected words
                five_tags = cats + selected_words
            else:
                # If there are more than 5 categories, keep the first 4 and merge the rest into the 5th
                if len(cats) > 5:
                    five_tags = cats[:4] + [" ".join(cats[4:])]
                else:
                    five_tags = cats
            
            # Ensure no empty tags
            five_tags = [tag if tag.strip() != "" else f"GenericTag{i+1}" for i, tag in enumerate(five_tags)]
            
            # Ensure there are exactly 5 tags
            while len(five_tags) < 5:
                five_tags.append(f"GenericTag{len(five_tags)+1}")
                
            return five_tags
        
        item_data['five_tags'] = item_data.apply(process_categories_to_five_tags, axis=1)
        
        # Show examples of the 5 processed tags
        print("\nProcessed 5-tags example:")
        for i in range(3):  # Show 3 examples
            print(f"Item {i}:")
            print(f"  Number of original categories: {len(item_data.iloc[i]['flat_categories'])}")
            print(f"  Processed 5 tags: {item_data.iloc[i]['five_tags']}")
        
        # Create tag index map
        print("\nCreating tag index map...")
        all_tags = []
        for tags in item_data['five_tags']:
            all_tags.extend(tags)
        
        # Create a list of unique tags
        unique_tags = sorted(list(set(all_tags)))
        tag_to_idx = {tag: idx for idx, tag in enumerate(unique_tags)}
        idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
        
        print(f"Number of unique tags: {len(unique_tags)}")
        print(f"First 10 tag examples: {unique_tags[:10]}")
        
        # Convert tags to indices
        def tags_to_indices(tags_list):
            return [tag_to_idx[tag] for tag in tags_list]
        
        item_data['tags_indices'] = item_data['five_tags'].apply(tags_to_indices)
        
        print("\nTag index example:")
        for i in range(3):
            print(f"Item {i}:")
            print(f"  Tags: {item_data.iloc[i]['five_tags']}")
            print(f"  Indices: {item_data.iloc[i]['tags_indices']}")
        
        # Count the number of unique IDs for each tag layer
        print("\nCounting unique IDs for each tag layer...")
        for layer in range(5):  # 5 layers of tags
            # Extract all tags for the current layer
            layer_tags = item_data['five_tags'].apply(lambda x: x[layer] if layer < len(x) else None).dropna().tolist()
            unique_layer_tags = set(layer_tags)
            
            # Extract all tag IDs for the current layer
            layer_ids = item_data['tags_indices'].apply(lambda x: x[layer] if layer < len(x) else None).dropna().tolist()
            unique_layer_ids = set(layer_ids)
            
            print(f"Layer {layer+1} tags:")
            print(f"  Number of unique tags: {len(unique_layer_tags)}")
            print(f"  Number of unique tag IDs: {len(unique_layer_ids)}")
            print(f"  Total number of tags: {len(layer_tags)}")  # Add total tag count
            print(f"  Top 5 tag examples: {list(unique_layer_tags)[:5]}")
            
            # If the number of tags is large, show the distribution
            if len(unique_layer_tags) > 10:
                from collections import Counter
                tag_counts = Counter(layer_tags)
                most_common = tag_counts.most_common(5)
                print(f"  5 most common tags: {most_common}")
                
                # Add tag distribution statistics
                total_items = len(layer_tags)
                top5_count = sum(count for _, count in most_common)
                print(f"  Top 5 tags coverage: {top5_count/total_items*100:.2f}%")
        
        print("\nBuilding item text descriptions...")
        sentences = item_data.apply(
            lambda row:
                "Title: " +
                str(row["title"]) + "; " +
                "Brand: " +
                str(row["brand"]) + "; " +
                "Price: " +
                str(row["price"]) + "; " ,
            axis=1
        ).tolist()  # Convert directly to a list

        print("\nText description example:")
        print(sentences[0])
        
        print("\nStarting text encoding...")
        show_memory_usage()
        item_emb = self._encode_text_feature_batched(sentences, batch_size=32)  # Reduce batch size
        gc.collect()  # Manually trigger garbage collection
        torch.cuda.empty_cache()
        show_memory_usage()
        print(f"Text feature dimension: {item_emb.shape}")
        print(f"Encoding example (first 5 dims): {item_emb[0,:5]}")
        
        # Encode the 5 tags separately - optimize memory usage
        print("\nStarting tag encoding...")
        tags_embs = []
        model = SentenceTransformer('sentence-transformers/sentence-t5-xl')  # Load the model only once
        
        for i in range(5):
            print(f"\nProcessing tag {i+1}...")
            tag_sentences = item_data['five_tags'].apply(lambda x: x[i] if i < len(x) else "").tolist()
            
            # Process tag encoding in batches, clearing memory after each batch
            batch_size = 16  # Smaller batch size
            tag_emb = self._encode_text_feature_batched(tag_sentences, model=model, batch_size=batch_size)
            tags_embs.append(tag_emb)
            
            # More aggressive memory cleaning
            del tag_sentences
            gc.collect()
            torch.cuda.empty_cache()
            show_memory_usage()
        
        # Stack the 5 tag embeddings into a single tensor
        tags_emb_tensor = torch.stack(tags_embs, dim=1)  # [n_items, 5, emb_dim]
        print(f"\nCombined tag feature dimension: {tags_emb_tensor.shape}")
        
        # Clean up variables that are no longer needed
        del tags_embs, model
        gc.collect()
        torch.cuda.empty_cache()
        
        data['item'].x = item_emb
        data['item'].text = np.array(sentences)
        data['item'].tags_emb = tags_emb_tensor
        data['item'].tags = np.array(item_data['five_tags'].tolist())
        data['item'].tags_indices = torch.tensor(item_data['tags_indices'].tolist(), dtype=torch.long)
        
        # Save the tag index map
        tag_index_dict = {
            'tag_to_idx': tag_to_idx,
            'idx_to_tag': idx_to_tag
        }
        
        # Get the processed file path and save the tag index map in the same directory
        processed_dir = os.path.dirname(self.processed_paths[0])
        tag_index_path = os.path.join(processed_dir, f'tag_index_{self.split}.pt')
        torch.save(tag_index_dict, tag_index_path)
        print(f"\nTag index map saved to: {tag_index_path}")

        print("\nSplitting into training and test sets...")
        gen = torch.Generator()
        gen.manual_seed(42)
        data['item'].is_train = torch.rand(item_emb.shape[0], generator=gen) > 0.05
        print(f"Number of training items: {data['item'].is_train.sum().item()}")
        print(f"Number of test items: {(~data['item'].is_train).sum().item()}")

        print("\nSaving processed data...")
        # Modify torch.save behavior to ensure compatibility
        original_torch_save = torch.save
        def patched_torch_save(*args, **kwargs):
            kwargs['_use_new_zipfile_serialization'] = False
            return original_torch_save(*args, **kwargs)
        
        # Temporarily replace torch.save
        torch.save = patched_torch_save
        
        try:
            self.save([data], self.processed_paths[0])
            print("Data saved successfully!")
        except Exception as e:
            print(f"Error saving data: {str(e)}")
        finally:
            # Restore the original function
            torch.save = original_torch_save
            
        print("=== Data processing complete ===\n")
        
    @staticmethod
    def _encode_text_feature_batched(text_feat, model=None, batch_size=32):
        """Process text encoding in batches to optimize memory usage"""
        if model is None:
            model = SentenceTransformer('sentence-transformers/sentence-t5-xl')
        
        total_samples = len(text_feat)
        embeddings_list = []
        
        # Convert Series to list
        if isinstance(text_feat, pd.Series):
            text_feat = text_feat.tolist()
        
        for i in range(0, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            batch_text = text_feat[i:batch_end]  # Use list slicing directly
            print(f"\rProcessing progress: {batch_end}/{total_samples} ({batch_end/total_samples*100:.1f}%)", end="")
            
            # Ensure memory cleanup
            gc.collect()
            torch.cuda.empty_cache()
            
            batch_embeddings = model.encode(
                sentences=batch_text,
                show_progress_bar=False,
                convert_to_tensor=True
            ).cpu()
            
            embeddings_list.append(batch_embeddings)
            
            # Immediately delete variables that are no longer needed
            del batch_text, batch_embeddings
            
            # Manually clean up memory
            gc.collect()
            torch.cuda.empty_cache()
        
        print()  # Newline
        
        # Concatenate embeddings from all batches
        result = torch.cat(embeddings_list, dim=0)
        
        # Clean up list
        del embeddings_list
        gc.collect()
        
        return result


if __name__ == "__main__":
    # Reprocess already downloaded files and specify a new path
    dataset = AmazonReviews(root="dataset/amazon", split="sports", force_reload=True)
