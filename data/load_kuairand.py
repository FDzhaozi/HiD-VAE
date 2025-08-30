import gin
import os
import torch
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from enum import Enum
from typing import Optional
from torch_geometric.data import HeteroData
from data.schemas import SeqBatch, TaggedSeqBatch

# Add KuaiRand to the RecDataset enum
@gin.constants_from_enum
class KuaiRandDataset(Enum):
    KUAIRAND = 4

class KuaiRandItemData(Dataset):
    """
    Loads item data from the KuaiRand dataset in a way that is compatible with the ItemData class.
    """
    
    def __init__(
        self,
        root: str,
        *args,
        force_process: bool = False,
        dataset: KuaiRandDataset = KuaiRandDataset.KUAIRAND,
        train_test_split: str = "all",
        data_file: str = None,
        **kwargs
    ) -> None:
        """
        Initializes the KuaiRand item data loader.
        
        Args:
            root: The root directory of the dataset, e.g., 'dataset/kuairand'.
            force_process: Whether to force reprocessing of the data (has no effect here, for compatibility with ItemData).
            dataset: The type of dataset (fixed to KUAIRAND here).
            train_test_split: Specifies the data split to load, can be "train", "eval", or "all".
            data_file: The specific data file to load; uses a default path if not specified.
        """
        print(f"root: {root}")
        print(f"dataset: {dataset}")
        
        # Determine the data file path
        if data_file is None:
            data_file = "title_data_kuairand_5tags.pt"
        
        data_path = os.path.join(root, "processed", data_file)
        # Set the processed_paths attribute
        self.processed_paths = [data_path]
        
        print(f"data_path: {data_path}")
        
        # Load the data
        try:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")
                
            self.data = torch.load(data_path, map_location='cpu')
            print("✓ Data loaded successfully!")
            
            # Ensure item features are of type float32
            if 'x' in self.data['item']:
                self.data['item']['x'] = self.data['item']['x'].float()
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
        
        # Apply train/eval filtering
        if train_test_split == "train":
            if 'is_train' in self.data['item']:
                filt = self.data['item']['is_train']
            else:
                # If 'is_train' field is not present, use 80% of the data for training
                total_items = len(self.data['item']['x'])
                train_size = int(0.8 * total_items)
                filt = torch.zeros(total_items, dtype=torch.bool)
                filt[:train_size] = True
                print("Note: Using default 80-20 split for training and evaluation sets.")
        elif train_test_split == "eval":
            if 'is_train' in self.data['item']:
                filt = ~self.data['item']['is_train']
            else:
                total_items = len(self.data['item']['x'])
                train_size = int(0.8 * total_items)
                filt = torch.zeros(total_items, dtype=torch.bool)
                filt[train_size:] = True
        else:  # "all"
            filt = torch.ones(len(self.data['item']['x']), dtype=torch.bool)
        
        # Store the required data fields and ensure they are float32
        self.item_data = self.data['item']['x'][filt].float()
        
        # Process text data
        if 'text' in self.data['item']:
            text_data = self.data['item']['text']
            if isinstance(text_data, (list, tuple)):
                # If it's a list, convert to a tensor first
                text_data = torch.tensor([1 if t else 0 for t in text_data])
            self.item_text = text_data[filt]
        else:
            self.item_text = None
            print("Warning: Text field not found in the dataset.")
        
        # Extract tag data
        self.has_tags = False
        
        # Check for the new tag format (tags_emb_l1, tags_emb_l2, tags_emb_l3)
        if all(f'tags_emb_l{i}' in self.data['item'] for i in range(1, 4)):
            self.tags_emb = torch.stack([
                self.data['item'][f'tags_emb_l{i}'][filt].float() for i in range(1, 4)
            ], dim=1)
            self.has_tags = True
        # Check for the old tag format (tags_emb)
        elif 'tags_emb' in self.data['item']:
            self.tags_emb = self.data['item']['tags_emb'][filt].float()
            self.has_tags = True
        else:
            self.tags_emb = None
            print("Warning: Tag embedding field not found in the dataset.")
        
        if 'tags_indices' in self.data['item']:
            self.tags_indices = self.data['item']['tags_indices'][filt]
            self.has_tags = True
        else:
            self.tags_indices = None
            print("Warning: Tag indices field not found in the dataset.")
            
        if 'tags' in self.data['item']:
            tags_data = self.data['item']['tags']
            if isinstance(tags_data, (list, tuple)):
                # If it's a list, convert to a tensor first
                tags_data = torch.tensor([1 if t else 0 for t in tags_data])
            self.tags = tags_data[filt]
            self.has_tags = True
        else:
            self.tags = None
            print("Warning: Tag text field not found in the dataset.")

    def __len__(self):
        return self.item_data.shape[0]

    def __getitem__(self, idx):
        """Returns item data for a given index in a format compatible with ItemData."""
        item_ids = torch.tensor(idx).unsqueeze(0) if not isinstance(idx, torch.Tensor) else idx
        x = self.item_data[idx, :768]  # Maintain feature dimension consistent with other datasets
        
        # Construct the basic batch data
        batch_data = {
            "user_ids": -1 * torch.ones_like(item_ids.squeeze(0)),
            "ids": item_ids,
            "ids_fut": -1 * torch.ones_like(item_ids.squeeze(0)),
            "x": x,
            "x_fut": -1 * torch.ones_like(item_ids.squeeze(0)),
            "seq_mask": torch.ones_like(item_ids, dtype=bool)
        }
        
        # If tag data exists, use TaggedSeqBatch
        if self.has_tags:
            # Get tag data
            if isinstance(idx, torch.Tensor):
                tags_emb = self.tags_emb[idx]
                tags_indices = self.tags_indices[idx]
            elif isinstance(idx, list):
                # Handle list-type index
                tags_emb = self.tags_emb[idx]
                tags_indices = self.tags_indices[idx]
            else:
                # Handle integer-type index
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
            # If no tag data exists, use the regular SeqBatch
            return SeqBatch(**batch_data)


def analyze_tag_distribution(dataset):
    """Analyzes the tag distribution and generates statistics."""
    
    print("\n===== Tag Distribution Analysis =====")
    
    if not dataset.has_tags:
        print("Error: No tag information in the dataset, cannot perform analysis.")
        return
    
    # Check tag indices
    if dataset.tags_indices is not None:
        # Calculate the number of non -1 tags in each level
        non_empty_tags = (dataset.tags_indices != -1).sum(dim=0)
        total_items = len(dataset)
        
        print(f"There are {total_items} items in the dataset.")
        print(f"Tag coverage per level:")
        
        for level in range(dataset.tags_indices.shape[1]):
            coverage = non_empty_tags[level].item() / total_items * 100
            print(f"  - Level {level+1} tags: {non_empty_tags[level].item()} items have tags ({coverage:.2f}%)")
        
        # Calculate the number of non-empty tags per item
        tags_per_item = (dataset.tags_indices != -1).sum(dim=1)
        avg_tags = tags_per_item.float().mean().item()
        
        print(f"\nOn average, each item has {avg_tags:.2f} non-empty tags.")
        
        # Statistics on the distribution of the number of tags per item
        tag_counts = Counter(tags_per_item.tolist())
        print("\nDistribution of number of tags per item:")
        for count in sorted(tag_counts.keys()):
            percentage = tag_counts[count] / total_items * 100
            print(f"  - {count} tags: {tag_counts[count]} items ({percentage:.2f}%)")
        
        # Analyze the tag distribution for each level
        print("\nDistribution of tag values per level:")
        for level in range(dataset.tags_indices.shape[1]):
            # Get all non -1 tags for the current level
            level_tags = dataset.tags_indices[:, level]
            valid_tags = level_tags[level_tags != -1]
            
            if len(valid_tags) > 0:
                unique_tags = torch.unique(valid_tags)
                print(f"  - Level {level+1} tags: {len(unique_tags)} unique tag values.")
                
                # Statistics for the TOP 10 most common tags
                tag_counts = Counter(valid_tags.tolist())
                print(f"    TOP 10 most common tags:")
                for tag, count in tag_counts.most_common(10):
                    percentage = count / len(valid_tags) * 100
                    print(f"       - Tag ID {tag}: {count} occurrences ({percentage:.2f}%)")
    
    # If tag text is available, perform text analysis
    if dataset.tags is not None:
        print("\nTag Text Analysis:")
        for level in range(dataset.tags.shape[1]):
            # Count non-empty tag texts
            level_tags = [tag for tag in dataset.tags[:, level] if tag != '']
            if level_tags:
                tag_counts = Counter(level_tags)
                print(f"  - Level {level+1} tags: {len(tag_counts)} unique tag texts.")
                
                # Statistics for the TOP 10 most common tag texts
                print(f"    TOP 10 most common tag texts:")
                for tag, count in tag_counts.most_common(10):
                    percentage = count / len(level_tags) * 100
                    print(f"       - '{tag}': {count} occurrences ({percentage:.2f}%)")

def plot_tag_distribution(dataset, save_dir=None):
    """Plots charts for the tag distribution."""
    
    if not dataset.has_tags or dataset.tags_indices is None:
        print("Error: No tag information in the dataset, cannot plot charts.")
        return
    
    # Create save directory
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Calculate the number of non-empty tags per item
    tags_per_item = (dataset.tags_indices != -1).sum(dim=1).tolist()
    
    # 1. Plot the distribution of the number of tags
    plt.figure(figsize=(10, 6))
    plt.hist(tags_per_item, bins=range(5), alpha=0.7, rwidth=0.8)
    plt.xlabel('Number of Tags per Item')
    plt.ylabel('Number of Items')
    plt.title('Distribution of Number of Tags per Item')
    plt.xticks(range(4))
    plt.grid(axis='y', alpha=0.3)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'tags_per_item.png'), dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {os.path.join(save_dir, 'tags_per_item.png')}")
    else:
        plt.show()
    plt.close()
    
    # 2. Plot tag coverage per level
    coverage = [(dataset.tags_indices[:, i] != -1).sum().item() / len(dataset) * 100 for i in range(dataset.tags_indices.shape[1])]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(coverage) + 1), coverage, alpha=0.7)
    plt.xlabel('Tag Level')
    plt.ylabel('Coverage (%)')
    plt.title('Tag Coverage per Level')
    plt.xticks(range(1, len(coverage) + 1))
    plt.ylim(0, 100)
    
    for i, v in enumerate(coverage):
        plt.text(i + 1, v + 1, f'{v:.1f}%', ha='center')
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'tag_level_coverage.png'), dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {os.path.join(save_dir, 'tag_level_coverage.png')}")
    else:
        plt.show()
    plt.close()
    
    # 3. Plot TOP 10 tag distribution (for each level)
    for level in range(dataset.tags_indices.shape[1]):
        # Get all non -1 tags for the current level
        level_tags = dataset.tags_indices[:, level]
        valid_tags = level_tags[level_tags != -1]
        
        if len(valid_tags) > 0:
            tag_counts = Counter(valid_tags.tolist())
            top_tags = tag_counts.most_common(10)
            
            # Plot TOP 10 tag distribution
            plt.figure(figsize=(12, 6))
            
            labels = [f'ID {tag}' for tag, _ in top_tags]
            values = [count for _, count in top_tags]
            
            plt.bar(labels, values, alpha=0.7)
            plt.xlabel('Tag ID')
            plt.ylabel('Occurrences')
            plt.title(f'Level {level+1} TOP 10 Tag Distribution')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'top_tags_level_{level+1}.png'), dpi=300, bbox_inches='tight')
                print(f"Chart saved to: {os.path.join(save_dir, f'top_tags_level_{level+1}.png')}")
            else:
                plt.show()
            plt.close()

# Entry point for running the script independently
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='KuaiRand Dataset Analysis Tool')
    parser.add_argument('--data_path', type=str, default='dataset/kuairand',
                        help='Root directory of the KuaiRand dataset')
    parser.add_argument('--data_file', type=str, default='kuairand_data_minimal_interactions30000.pt',
                        help='The data file to load')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'eval', 'all'],
                        help='The dataset split to analyze')
    parser.add_argument('--plot', action='store_true', help='Whether to plot and save charts')
    parser.add_argument('--plot_dir', type=str, default='plots/kuairand_analysis',
                        help='Directory to save the plots')
    
    args = parser.parse_args()
    
    # Load the dataset
    dataset = KuaiRandItemData(
        root=args.data_path, 
        train_test_split=args.split,
        data_file=args.data_file
    )
    
    # Print basic dataset information
    print("\n===== KuaiRand Dataset Basic Information =====")
    print(f"Data path: {os.path.join(args.data_path, 'processed', args.data_file)}")
    print(f"Data split: {args.split}")
    print(f"Number of items: {len(dataset)}")
    
    if dataset.item_data is not None:
        print(f"Feature dimension: {dataset.item_data.shape[1]}")
    
    if dataset.has_tags:
        if dataset.tags_emb is not None:
            print(f"Tag embedding shape: {dataset.tags_emb.shape}")
        if dataset.tags_indices is not None:
            print(f"Tag indices shape: {dataset.tags_indices.shape}")
        if dataset.tags is not None:
            print(f"Tag text shape: {dataset.tags.shape}")
    
    # Analyze tag distribution
    analyze_tag_distribution(dataset)
    
    # Plot charts
    if args.plot:
        plot_tag_distribution(dataset, args.plot_dir)
