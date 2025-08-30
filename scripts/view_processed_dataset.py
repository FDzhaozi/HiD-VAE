import os
import json
import gzip
import pickle
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.serialization import add_safe_globals
from numpy.core.multiarray import _reconstruct

# Add safe global variables for torch loading
add_safe_globals([_reconstruct])

def parse_gz(path):
    """Parse a gzip file."""
    with gzip.open(path, 'r') as g:
        for l in g:
            yield eval(l)

# Define paths
root_path = Path(__file__).parent.parent
raw_path = root_path / "dataset" / "amazon" / "raw" / "beauty"
processed_path = root_path / "dataset" / "amazon" / "processed"
processed_path_kuairand = root_path / "dataset" / "kuairand" / "processed"

class BeautyDatasetViewer:
    def __init__(self, raw_path=raw_path, processed_path=processed_path):
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)
        self.processed_path_kuairand = Path(processed_path_kuairand)
    
    def view_raw_files(self):
        """Display the list of raw data files and their sizes."""
        print("\n=== Raw Data Files ===")
        for file in self.raw_path.glob("*"):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"{file.name}: {size_mb:.2f}MB")
    
    def view_datamaps(self, n_samples=5):
        """View the ID mapping file."""
        print("\n=== Data Mapping File (datamaps.json) ===")
        with open(self.raw_path / "datamaps.json", 'r') as f:
            data_maps = json.load(f)
        
        print("\nItem ID Mapping Examples:")
        items = list(data_maps["item2id"].items())[:n_samples]
        for asin, id in items:
            print(f"ASIN: {asin} -> ID: {id}")
            
        print(f"\nTotal items: {len(data_maps['item2id'])}")
    
    def view_meta(self, n_samples=5):
        """View item metadata."""
        print("\n=== Item Metadata (meta.json.gz) ===")
        items = []
        category_counts = {}  # To count the number of items for different category counts
        category_examples = {}  # To store an example item for each category count
        total_items = 0
        
        def flatten_categories(categories):
            """Flatten a nested list of categories into a single list."""
            flattened = []
            for cat in categories:
                if isinstance(cat, list):
                    flattened.extend(cat)
                else:
                    flattened.append(cat)
            return list(dict.fromkeys(flattened))  # Deduplicate
        
        # First pass: collect all data and statistics
        for item in parse_gz(self.raw_path / "meta.json.gz"):
            total_items += 1
            
            # Flatten and deduplicate categories
            categories = flatten_categories(item.get('categories', []))
            num_categories = len(categories)
            
            # Update the item's categories to the flattened list
            item['categories'] = categories
            
            category_counts[num_categories] = category_counts.get(num_categories, 0) + 1
            
            # Save the first example for each category count
            if num_categories not in category_examples:
                category_examples[num_categories] = item
            
            # Save sample data
            if len(items) < n_samples:
                items.append(item)
        
        # Display sample data
        print("\nItem Metadata Examples:")
        for item in items:
            print("\n---")
            print(f"ASIN: {item.get('asin')}")
            print(f"Title: {item.get('title')}")
            print(f"Brand: {item.get('brand')}")
            print(f"Categories (flattened): {item.get('categories', [])}")
            print(f"Price: {item.get('price')}")
        
        # Display category statistics
        print("\nCategory Statistics:")
        print(f"Total items: {total_items}")
        if category_counts:
            min_categories = min(category_counts.keys())
            max_categories = max(category_counts.keys())
            print(f"Min number of categories: {min_categories}")
            print(f"Max number of categories: {max_categories}")
            
            print("\nDistribution of category counts and examples:")
            total_counted = 0
            for num_cats in sorted(category_counts.keys()):
                count = category_counts[num_cats]
                total_counted += count
                print(f"\nItems with {num_cats} categories: {count} total")
                
                # Display an example item for this category count
                example = category_examples[num_cats]
                print("Example item:")
                print(f"  ASIN: {example.get('asin')}")
                print(f"  Title: {example.get('title')}")
                print(f"  Brand: {example.get('brand')}")
                print(f"  Categories: {example.get('categories', [])}")
                print(f"  Price: {example.get('price')}")
            
            print(f"\nTotal counted: {total_counted}")
            if total_counted != total_items:
                print(f"Warning: Total counted does not match total items! Difference: {total_items - total_counted}")
    
    def view_sequential_data(self, n_samples=5):
        """View sequential data."""
        print("\n=== User Sequence Data (sequential_data.txt) ===")
        with open(self.raw_path / "sequential_data.txt", "r") as f:
            sequences = []
            for i, line in enumerate(f):
                if i < n_samples:
                    sequences.append(list(map(int, line.strip().split())))
                else:
                    break
        
        print("\nUser Sequence Examples:")
        for seq in sequences:
            print(f"User ID: {seq[0]}, Item Sequence: {seq[1:5]}... (showing first 4 items)")
    
    def view_processed_data(self, n_samples=5):
        """View processed data."""
        print("\n=== Processed Data (data.pt) ===")
        try:
            # Try to load data using weights_only=False
            #loaded_data = torch.load(self.processed_path / "title_data_beauty_5tags.pt", weights_only=False)
            loaded_data = torch.load(self.processed_path_kuairand / "title_data_kuairand_5tags.pt", weights_only=False)
            
            # Handle the case where the data is a list
            if isinstance(loaded_data, list):
                print(f"Original data type: {type(loaded_data)}")
                if len(loaded_data) > 0:
                    data = loaded_data[0]  # Take the first element
                    print(f"Extracted data type: {type(data)}")
                else:
                    print("Error: The loaded data list is empty")
                    return
            # If the data is a tuple, take the first element
            elif isinstance(loaded_data, tuple):
                print(f"Original data type: {type(loaded_data)}")
                data = loaded_data[0]
                print(f"Extracted data type: {type(data)}")
            else:
                data = loaded_data
            
            print("\nData Structure:")
            print(f"Data type: {type(data)}")
            print("Data keys:")
            for key in data.keys():
                print(f"- {key}")
                if key == 'item':
                    print("  Sub-keys and examples:")
                    for subkey in data[key].keys():
                        print(f"    - {subkey}:")
                        if isinstance(data[key][subkey], torch.Tensor):
                            print(f"      Shape: {data[key][subkey].shape}")
                            # For tags_indices, show more different samples
                            if subkey == 'tags_indices' or subkey == 'raw_tags_indices':
                                print(f"      Sample 1: {data[key][subkey][0]}")
                                print(f"      Sample 2: {data[key][subkey][1]}")
                                print(f"      Sample 3: {data[key][subkey][2]}")
                                print(f"      Sample 4: {data[key][subkey][3]}")
                                print(f"      Sample 5: {data[key][subkey][4]}")
                                
                                # New: Calculate and display the number of categories for each tag layer
                                print("\n      === Tag Category Statistics per Layer ===")
                                tags_indices = data[key][subkey]
                                n_layers = tags_indices.shape[1]
                                for layer in range(n_layers):
                                    # Get the number of unique tags in the current layer
                                    unique_tags = torch.unique(tags_indices[:, layer])
                                    print(f"        Layer {layer+1} tag categories: {len(unique_tags)}")
                                    # Display the min and max tag indices
                                    if len(unique_tags) > 0:
                                        print(f"        Layer {layer+1} tag index range: {unique_tags.min().item()} to {unique_tags.max().item()}")
                                    
                                    # Display tag distribution
                                    tag_counts = {}
                                    for tag in tags_indices[:, layer]:
                                        tag_val = tag.item()
                                        tag_counts[tag_val] = tag_counts.get(tag_val, 0) + 1
                                    
                                    # Display the top 5 most frequent tags
                                    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
                                    print(f"        Layer {layer+1} most frequent tags: ", end="")
                                    for tag_val, count in sorted_tags[:5]:
                                        print(f"{tag_val}({count} times) ", end="")
                                    print()
                                
                                print("      === End of Tag Category Statistics ===\n")
                            else:
                                # Take only the first 5 elements for vectors
                                print(f"      Example 1: {data[key][subkey][:5]}")
                        elif subkey == 'tags_mapping_dicts' or subkey == 'tags_reverse_mapping_dicts':
                            # For mapping dictionaries, show the size and some examples for each layer's dict
                            print(f"      Type: {type(data[key][subkey])}")
                            print(f"      Number of layers: {len(data[key][subkey])}")
                            for i, layer_dict in enumerate(data[key][subkey]):
                                print(f"        Layer {i+1} mapping dict size: {len(layer_dict)}")
                                # Show the first 3 mapping relations
                                items = list(layer_dict.items())[:3]
                                print(f"        Layer {i+1} mapping examples: {items}")
                        else:
                            print(f"      Type: {type(data[key][subkey])}")
                            # For boolean types, print the value directly without trying to get length
                            if isinstance(data[key][subkey], bool):
                                print(f"      Value: {data[key][subkey]}")
                            else:
                                print(f"      Length: {len(data[key][subkey])}")
                                print(f"      Example 1: {data[key][subkey][0]}")
                                print(f"      Example 2: {data[key][subkey][1]}")
                elif isinstance(data[key], dict):
                    print("  Sub-keys:")
                    for subkey in data[key].keys():
                        print(f"    - {subkey}")
                        if isinstance(data[key][subkey], dict):
                            print("      Inner keys:")
                            for inner_key in data[key][subkey].keys():
                                print(f"        - {inner_key}")

            # Try to access user interaction data
            user_item_keys = [k for k in data.keys() if isinstance(k, tuple) and 'user' in k[0] and 'item' in k[2]]
            if user_item_keys:
                key = user_item_keys[0]
                if 'history' in data[key]:
                    print("\nSequence Splitting Examples (leave-one-out strategy):")
                    history_data = data[key]['history']
                    
                    # Get full sequence examples for a few users
                    sample_users = history_data['train']['userId'][:n_samples]
                    for user_idx in range(len(sample_users)):
                        user_id = sample_users[user_idx]
                        if isinstance(user_id, torch.Tensor):
                            user_id = user_id.item()
                        print(f"\nSequence for User {user_id}:")
                        
                        # Training sequence
                        train_seq = history_data['train']['itemId'][user_idx]
                        train_target = history_data['train']['itemId_fut'][user_idx]
                        if isinstance(train_seq, list):
                            valid_train = [x for x in train_seq if x >= 0]
                        else:  # tensor
                            valid_train = train_seq[train_seq >= 0].tolist()
                        print(f"Train: {valid_train} -> ", end="")
                        print(train_target.item() if isinstance(train_target, torch.Tensor) else train_target)
                        
                        # Validation sequence
                        eval_seq = history_data['eval']['itemId'][user_idx]
                        eval_target = history_data['eval']['itemId_fut'][user_idx]
                        if isinstance(eval_seq, list):
                            valid_eval = [x for x in eval_seq if x >= 0]
                        else:  # tensor
                            valid_eval = eval_seq[eval_seq >= 0].tolist()
                        print(f"Validation: {valid_eval} -> ", end="")
                        print(eval_target.item() if isinstance(eval_target, torch.Tensor) else eval_target)
                        
                        # Test sequence
                        test_seq = history_data['test']['itemId'][user_idx]
                        test_target = history_data['test']['itemId_fut'][user_idx]
                        if isinstance(test_seq, list):
                            valid_test = [x for x in test_seq if x >= 0]
                        else:  # tensor
                            valid_test = test_seq[test_seq >= 0].tolist()
                        print(f"Test: {valid_test} -> ", end="")
                        print(test_target.item() if isinstance(test_target, torch.Tensor) else test_target)
                    
                    print("\nDataset Statistics:")
                    print(f"Total users: {len(set([u.item() if isinstance(u, torch.Tensor) else u for u in history_data['train']['userId']]))}")
                    
                    # Calculate and print the total number of interactions
                    total_interactions = 0
                    # Calculate training set interactions
                    train_interactions = sum([(seq >= 0).sum().item() if isinstance(seq, torch.Tensor) else len([x for x in seq if x >= 0]) for seq in history_data['train']['itemId']])
                    # Calculate validation set interactions
                    eval_interactions = sum([(seq >= 0).sum().item() if isinstance(seq, torch.Tensor) else len([x for x in seq if x >= 0]) for seq in history_data['eval']['itemId']])
                    # Calculate test set interactions
                    test_interactions = sum([(seq >= 0).sum().item() if isinstance(seq, torch.Tensor) else len([x for x in seq if x >= 0]) for seq in history_data['test']['itemId']])
                    # Calculate target interactions (future items)
                    future_interactions = len(history_data['train']['itemId_fut']) + len(history_data['eval']['itemId_fut']) + len(history_data['test']['itemId_fut'])
                    
                    total_interactions = train_interactions + eval_interactions + test_interactions + future_interactions
                    print(f"Total interactions: {total_interactions}")
                    print(f"  - Train interactions: {train_interactions}")
                    print(f"  - Validation interactions: {eval_interactions}")
                    print(f"  - Test interactions: {test_interactions}")
                    print(f"  - Target interactions: {future_interactions}")
                    
                    # Calculate sequence length statistics
                    if isinstance(history_data['train']['itemId'][0], list):
                        train_lengths = [len([x for x in seq if x >= 0]) for seq in history_data['train']['itemId']]
                    else:
                        train_lengths = [(seq >= 0).sum().item() for seq in history_data['train']['itemId']]
                    
                    print("\nSequence Length Statistics:")
                    print(f"Min sequence length: {min(train_lengths)}")
                    print(f"Max sequence length: {max(train_lengths)}")
                    print(f"Avg sequence length: {sum(train_lengths) / len(train_lengths):.2f}")

        except Exception as e:
            print(f"Error loading processed data: {str(e)}")
            print("\nHint: Please ensure the data processing step has been run and the data file exists in the correct location.")
            import traceback
            print("\nDetailed error information:")
            print(traceback.format_exc())

    def remap_tags_indices(self):
        """
        Remaps the tags_indices so that the tag indices for each layer start from 0.
        The original indices are saved as raw_tags_indices.
        """
        print("\n=== Remapping Tag Indices ===")
        try:
            # Load the processed data
            #data_path = self.processed_path / "title_data_sports_5tags.pt"
            data_path = self.processed_path_kuairand / "title_data_kuairand_5tags.pt"
            print(f"Loading data from: {data_path}")
            loaded_data = torch.load(data_path, weights_only=False)
            
            # Record the original data format and content
            original_format = None  # Record the original data format
            original_data = loaded_data  # Save the complete original data
            
            if isinstance(loaded_data, list):
                print("Data is in list format")
                original_format = "list"
                if len(loaded_data) > 0:
                    data = loaded_data[0]
                else:
                    print("Error: The loaded data list is empty")
                    return
            # If the data is a tuple, process the first element
            elif isinstance(loaded_data, tuple):
                print(f"Data is in tuple format with length: {len(loaded_data)}")
                original_format = "tuple"
                data = loaded_data[0]  # Process only the first element
                # Print the type of each element in the tuple
                for i, item in enumerate(loaded_data):
                    print(f"Type of element {i+1} in tuple: {type(item)}")
            else:
                print("Data is in dictionary format")
                original_format = "dict"
                data = loaded_data
            
            # Check the data structure
            if 'item' not in data or 'tags_indices' not in data['item']:
                print("Error: Data does not contain 'item.tags_indices' field")
                print(f"Fields contained in the 'item' key: {list(data['item'].keys())}")
                return
            
            # Get the original tag indices
            original_tags_indices = data['item']['tags_indices']
            print(f"Original tags_indices shape: {original_tags_indices.shape}")
            
            # Check if it has already been remapped
            if 'raw_tags_indices' in data['item']:
                print("Warning: The data has already been remapped.")
                choice = input("Do you want to continue remapping? (y/n): ")
                if choice.lower() != 'y':
                    print("Operation cancelled")
                    return
                print("Continuing with remapping...")
            
            # Save the original tag indices
            data['item']['raw_tags_indices'] = original_tags_indices.clone()
            
            # Remap the tags for each layer
            remapped_indices = torch.zeros_like(original_tags_indices, dtype=torch.long)
            mapping_dicts = []  # Store the mapping dictionary for each layer
            reverse_mapping_dicts = []  # Store the reverse mapping dictionary
            
            for layer in range(original_tags_indices.shape[1]):  # Iterate through each layer
                # Get all tag indices for the current layer
                layer_indices = original_tags_indices[:, layer]
                
                # Get unique indices and sort them
                unique_indices = torch.unique(layer_indices).tolist()
                unique_indices.sort()  # Ensure sorting to make the mapping more stable
                
                # Create mapping dictionary: original index -> new index (starting from 0)
                mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}
                mapping_dicts.append(mapping)
                
                # Create reverse mapping dictionary: new index -> original index
                reverse_mapping = {new_idx: old_idx for old_idx, new_idx in mapping.items()}
                reverse_mapping_dicts.append(reverse_mapping)
                
                # Apply the mapping
                for i in range(len(layer_indices)):
                    remapped_indices[i, layer] = mapping[layer_indices[i].item()]
            
            # Update the data
            data['item']['tags_indices'] = remapped_indices
            
            # Save the mapping dictionaries to restore original indices later
            data['item']['tags_mapping_dicts'] = mapping_dicts
            data['item']['tags_reverse_mapping_dicts'] = reverse_mapping_dicts
            
            # Display remapping results
            print("\nRemapping complete!")
            print(f"New tags_indices shape: {data['item']['tags_indices'].shape}")
            
            # Display the number of unique tags for each layer
            print("\nNumber of unique tags per layer:")
            for layer in range(original_tags_indices.shape[1]):
                original_unique = len(torch.unique(original_tags_indices[:, layer]))
                new_unique = len(torch.unique(remapped_indices[:, layer]))
                print(f"Layer {layer+1}: Original unique tags {original_unique}, Remapped unique tags {new_unique}")
                
                # Display some mapping examples
                print(f"  Mapping examples (first 3):")
                items = list(mapping_dicts[layer].items())[:3]
                for old_idx, new_idx in items:
                    print(f"    Original index {old_idx} -> New index {new_idx}")
            
            # Verify recoverability
            print("\nVerifying the recoverability of the remapping:")
            # Randomly select a few samples for verification
            sample_indices = torch.randint(0, original_tags_indices.shape[0], (5,))
            for idx in sample_indices:
                idx = idx.item()
                print(f"\nSample {idx}:")
                for layer in range(original_tags_indices.shape[1]):
                    original = original_tags_indices[idx, layer].item()
                    remapped = remapped_indices[idx, layer].item()
                    recovered = reverse_mapping_dicts[layer][remapped]
                    print(f"  Layer {layer+1}: Original {original} -> Remapped {remapped} -> Recovered {recovered}")
                    assert original == recovered, f"Recovery failed: {original} != {recovered}"
            
            # Save the updated data according to the original format
            print("\nSaving the updated data...")
            if original_format == "list":
                torch.save([data], data_path)
            elif original_format == "tuple":
                # If it's a tuple, keep the original tuple length and other elements unchanged
                if len(original_data) == 1:
                    torch.save((data,), data_path)
                elif len(original_data) == 2:
                    torch.save((data, original_data[1]), data_path)
                elif len(original_data) == 3:
                    torch.save((data, original_data[1], original_data[2]), data_path)
                else:
                    # Create a new tuple with the modified data as the first element
                    new_data = (data,) + original_data[1:]
                    torch.save(new_data, data_path)
            else:  # dict
                torch.save(data, data_path)
            print(f"Data saved to: {data_path}")
            
            # Display a before-and-after comparison
            print("\nComparison of tag indices before and after remapping (first 5 samples):")
            for i in range(5):
                print(f"\nSample {i}:")
                print(f"  Original indices: {data['item']['raw_tags_indices'][i]}")
                print(f"  Remapped indices: {data['item']['tags_indices'][i]}")
            
        except Exception as e:
            print(f"Error during tag index remapping: {str(e)}")
            import traceback
            print("\nDetailed error information:")
            print(traceback.format_exc())

def main():
    viewer = BeautyDatasetViewer()
    
    while True:
        print("\n=== Amazon Beauty Dataset Viewer ===")
        print("1. View list of raw data files")
        print("2. View data mapping file (datamaps.json)")
        print("3. View item metadata (meta.json.gz)")
        print("4. View user sequence data (sequential_data.txt)")
        print("5. View processed data (data.pt)")
        print("6. Remap tag indices")
        
        print("0. Exit")
        
        choice = input("\nPlease select an option (0-6): ")
        
        if choice == '0':
            break
        elif choice == '1':
            viewer.view_raw_files()
        elif choice == '2':
            viewer.view_datamaps()
        elif choice == '3':
            viewer.view_meta()
        elif choice == '4':
            viewer.view_sequential_data()
        elif choice == '5':
            viewer.view_processed_data()
        elif choice == '6':
            viewer.remap_tags_indices()
        else:
            print("Invalid choice, please try again")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
