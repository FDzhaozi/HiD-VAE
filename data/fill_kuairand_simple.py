import os
import torch
import numpy as np
import random
from tqdm import tqdm
from collections import Counter
from copy import deepcopy
import time
import logging
import gc
from torch.serialization import add_safe_globals
from numpy.core.multiarray import _reconstruct

# Add safe globals for torch loading
add_safe_globals([_reconstruct])

# Import necessary functions from fill_kuairand.py
from data.fill_kuairand import cosine_similarity, check_completion_progress


def build_tag_hierarchy(data):
    """
    Iterate through the dataset to build a parent-child relationship graph for tag hierarchies.
    """
    print("Building tag hierarchy graph...")
    item_data = data['item']
    tags_indices = item_data['tags_indices']

    l1_to_l2 = {}
    l2_to_l3 = {}

    for i in tqdm(range(tags_indices.shape[0]), desc="Analyzing hierarchical relationships"):
        l1_id, l2_id, l3_id = tags_indices[i, 0].item(), tags_indices[i, 1].item(), tags_indices[i, 2].item()

        # L1 -> L2
        if l1_id != -1 and l2_id != -1:
            if l1_id not in l1_to_l2:
                l1_to_l2[l1_id] = set()
            l1_to_l2[l1_id].add(l2_id)

        # L2 -> L3
        if l2_id != -1 and l3_id != -1:
            if l2_id not in l2_to_l3:
                l2_to_l3[l2_id] = set()
            l2_to_l3[l2_id].add(l3_id)

    # Convert sets to lists for easier access later
    for k in l1_to_l2:
        l1_to_l2[k] = list(l1_to_l2[k])
    for k in l2_to_l3:
        l2_to_l3[k] = list(l2_to_l3[k])

    print(f"Hierarchy construction complete. L1->L2 relations: {len(l1_to_l2)}, L2->L3 relations: {len(l2_to_l3)}")

    return {"l1_to_l2": l1_to_l2, "l2_to_l3": l2_to_l3}


def load_kuairand_dataset(data_path, seed=42):
    """
    Loads the KuaiRand dataset and returns the data object.

    Args:
        data_path: Path to the data file.
        seed: Random seed for reproducibility.

    Returns:
        The data object.
    """
    print(f"Loading data from {data_path}...")

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Check if the file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    try:
        # Load the dataset, explicitly setting weights_only=False to handle potential PyTorch version issues.
        loaded_data = torch.load(data_path, map_location='cpu', weights_only=False)
        print("✓ Data loaded successfully!")

        # Handle cases where the loaded data is a list or tuple
        if isinstance(loaded_data, list):
            print(f"Original data type: {type(loaded_data)}")
            if len(loaded_data) > 0:
                data = loaded_data[0]  # Take the first element
                print(f"Extracted data type: {type(data)}")
            else:
                print("Error: Loaded data list is empty")
                return None
        # If the data is a tuple, take the first element
        elif isinstance(loaded_data, tuple):
            print(f"Original data type: {type(loaded_data)}")
            data = loaded_data[0]
            print(f"Extracted data type: {type(data)}")
        else:
            data = loaded_data

        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        print(traceback.format_exc())
        raise


def create_tag_pools(data):
    """
    Create tag pools for each level, including tag text and corresponding embedding vectors.

    Args:
        data: The data object, containing item node information.

    Returns:
        A list of three tag pools, one for each level, where each pool contains tag text and its embedding.
    """
    print("Creating tag pools...")

    if 'item' not in data:
        raise ValueError("Dataset does not contain 'item' node.")

    item_data = data['item']

    if 'tags_indices' not in item_data:
        raise ValueError("Dataset does not contain tag index information.")

    tags_indices = item_data['tags_indices']

    # Get tag text data
    if 'tags' not in item_data or item_data['tags'] is None:
        raise ValueError("Dataset does not contain tag text information.")

    # Create tag pools
    tag_pools = []

    for level in range(tags_indices.shape[1]):
        # Get all valid tag indices for the current level
        valid_indices = tags_indices[:, level]
        valid_indices = valid_indices[valid_indices != -1]
        unique_indices = torch.unique(valid_indices)

        # Create the pool for this level
        level_pool = {}

        for idx in unique_indices:
            # Find all items that have this tag
            items_with_tag = (tags_indices[:, level] == idx).nonzero(as_tuple=True)[0]

            if len(items_with_tag) == 0:
                continue

            # Get the tag text from a sample item
            sample_item = items_with_tag[0].item()
            tag_text = item_data['tags'][sample_item][level]

            # Get the tag embedding
            if 'tags_emb' in item_data and item_data['tags_emb'] is not None:
                # Average the embeddings of all items with this tag
                tag_embs = item_data['tags_emb'][items_with_tag, level]
                tag_emb = torch.mean(tag_embs, dim=0)
            else:
                # If no dedicated tag embedding, use the feature vector of a sample item
                tag_emb = item_data['x'][sample_item]

            # Add to the tag pool
            if tag_text and tag_text != '':
                level_pool[idx.item()] = {
                    'text': tag_text,
                    'embedding': tag_emb,
                    'count': len(items_with_tag)
                }

        tag_pools.append(level_pool)
        print(f"Level {level+1} tag pool created, containing {len(level_pool)} unique tags.")

    return tag_pools


def retrieve_most_similar_tag(context_embedding, tag_pool, candidate_ids=None):
    """
    Retrieve the most similar tag based on a context embedding.

    Args:
        context_embedding: The context embedding vector (e.g., item title vector, or a combination with parent tags).
        tag_pool: The tag pool containing tag IDs, text, and embeddings.
        candidate_ids: (Optional) A list of candidate tag IDs to constrain the search space.

    Returns:
        The ID, text, similarity score, and embedding of the most similar tag.
    """
    max_similarity = -1
    best_tag_id = None
    best_tag_text = None
    best_tag_embedding = None

    search_space = candidate_ids if candidate_ids is not None else tag_pool.keys()

    if not search_space:
        return None, None, -1, None

    for tag_id in search_space:
        tag_info = tag_pool.get(tag_id)
        if not tag_info:
            continue

        tag_embedding = tag_info['embedding']
        similarity = cosine_similarity(context_embedding, tag_embedding)

        if similarity > max_similarity:
            max_similarity = similarity
            best_tag_id = tag_id
            best_tag_text = tag_info['text']
            best_tag_embedding = tag_embedding

    return best_tag_id, best_tag_text, max_similarity.item(), best_tag_embedding


def complete_tags_hierarchically(data, item_idx, tag_pools, tag_hierarchy):
    """
    Complete missing tags using hierarchical constraints and sequential filling.
    Includes a fallback mechanism: when no candidates are found under hierarchical constraints,
    it falls back to searching the entire level.

    Args:
        data: The data object.
        item_idx: The index of the item to complete.
        tag_pools: A list of tag pools for the three levels.
        tag_hierarchy: The tag hierarchy relationship graph.

    Returns:
        A dictionary with the completion results.
    """
    item_data = data['item']
    original_indices = item_data['tags_indices'][item_idx].clone()  # Clone to track changes

    # Get the item's feature vector
    item_embedding = item_data['x'][item_idx]

    completion_result = {
        "status": "pending",
        "completed_tags": {},
        "reasoning": {}  # Record the reason for each level separately
    }

    # Store the vector information of each level's tag to build context
    level_embeddings = {}
    if 'tags_emb' in item_data:
        for i in range(3):
            if original_indices[i] != -1:
                level_embeddings[i] = item_data['tags_emb'][item_idx, i]

    # --- Fill sequentially ---

    # Fill Level 1
    if original_indices[0] == -1:
        tag_id, tag_text, _, tag_embedding = retrieve_most_similar_tag(item_embedding, tag_pools[0])
        if tag_id is not None:
            original_indices[0] = tag_id
            level_embeddings[0] = tag_embedding
            completion_result["completed_tags"][0] = {"id": tag_id, "name": tag_text, "embedding": tag_embedding}
            completion_result["reasoning"][0] = "Global search based on item feature vector"

    # Fill Level 2
    if original_indices[1] == -1 and original_indices[0] != -1:
        l1_id = original_indices[0].item()
        candidate_l2_ids = tag_hierarchy['l1_to_l2'].get(l1_id)
        reason = "Search based on hierarchical constraint"

        # Fallback: If no candidates under the constraint, search the entire L2
        if not candidate_l2_ids:
            candidate_l2_ids = None  # Passing None indicates a global search
            reason = "No candidates under hierarchy, falling back to global search"

        # Build context embedding
        l1_embedding = level_embeddings.get(0, item_embedding)  # If L1 was just filled, use its vector
        context_embedding = 0.6 * l1_embedding + 0.4 * item_embedding

        tag_id, tag_text, _, tag_embedding = retrieve_most_similar_tag(context_embedding, tag_pools[1], candidate_l2_ids)
        if tag_id is not None:
            original_indices[1] = tag_id
            level_embeddings[1] = tag_embedding
            completion_result["completed_tags"][1] = {"id": tag_id, "name": tag_text, "embedding": tag_embedding}
            completion_result["reasoning"][1] = reason

    # Fill Level 3
    if original_indices[2] == -1 and original_indices[1] != -1:
        l2_id = original_indices[1].item()
        candidate_l3_ids = tag_hierarchy['l2_to_l3'].get(l2_id)
        reason = "Search based on hierarchical constraint"

        # Fallback mechanism
        if not candidate_l3_ids:
            candidate_l3_ids = None
            reason = "No candidates under hierarchy, falling back to global search"

        # Build context embedding
        l1_embedding = level_embeddings.get(0, item_embedding)
        l2_embedding = level_embeddings.get(1, item_embedding)
        context_embedding = 0.5 * l2_embedding + 0.3 * l1_embedding + 0.2 * item_embedding

        tag_id, tag_text, _, tag_embedding = retrieve_most_similar_tag(context_embedding, tag_pools[2], candidate_l3_ids)
        if tag_id is not None:
            original_indices[2] = tag_id
            level_embeddings[2] = tag_embedding
            completion_result["completed_tags"][2] = {"id": tag_id, "name": tag_text, "embedding": tag_embedding}
            completion_result["reasoning"][2] = reason

    if completion_result["completed_tags"]:
        completion_result["status"] = "success"
    else:
        # Check if there are still missing tags
        if torch.any(original_indices == -1):
            completion_result["status"] = "failed"
            completion_result["message"] = "Could not fill a level (e.g., L1 was missing or L1 completion failed)"
        else:
            completion_result["status"] = "complete"
            completion_result["message"] = "All tags were already complete, no completion needed"

    return completion_result


def simple_complete_tags(data, tag_pools, tag_hierarchy, batch_size=100, save_path=None, start_idx=0):
    """
    Batch complete tags using similarity and save to a new data file.

    Args:
        data: The data object.
        tag_pools: A list of tag pools for the three levels.
        tag_hierarchy: The tag hierarchy relationship graph.
        batch_size: The number of items to process in each batch.
        save_path: The path to save the completed data.
        start_idx: The starting index for processing, for resuming.

    Returns:
        The updated data object.
    """
    # Create a copy of the data to avoid modifying the original
    new_data = deepcopy(data)

    item_data = new_data['item']
    num_items = item_data['x'].shape[0]

    # Find items missing at least one tag level
    incomplete_items = []
    for i in range(num_items):
        tags_indices = item_data['tags_indices'][i]
        if torch.any(tags_indices == -1):
            incomplete_items.append(i)

    # Filter out items before the start index
    incomplete_items = [idx for idx in incomplete_items if idx >= start_idx]

    total_incomplete = len(incomplete_items)
    print(f"Found {total_incomplete} items with missing tags. Starting processing from index {start_idx}.")

    # Find items with empty titles
    empty_title_items = []
    if 'text' in item_data and item_data['text'] is not None:
        for i in range(num_items):
            if not item_data['text'][i] or item_data['text'][i].strip() == '':
                empty_title_items.append(i)
        print(f"Found {len(empty_title_items)} items with empty titles.")

    # Calculate number of batches
    num_batches = (total_incomplete + batch_size - 1) // batch_size

    # Dictionary to store processing statistics
    stats = {
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "empty_titles_filled": 0,
        "errors": []
    }

    # Batch processing
    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min((batch_num + 1) * batch_size, total_incomplete)
        batch_indices = incomplete_items[batch_start:batch_end]

        print(f"\nProcessing batch {batch_num + 1}/{num_batches} (Item indices from {batch_indices[0]} to {batch_indices[-1]})")

        # Create a progress bar for the batch
        pbar = tqdm(batch_indices, desc=f"Batch {batch_num+1}")

        # Iterate over each item in the batch
        for idx in pbar:
            # Check if completion is still needed
            tags_indices = item_data['tags_indices'][idx]
            if not torch.any(tags_indices == -1):
                stats["skipped"] += 1
                continue

            # Use the new hierarchical filling function
            completion_result = complete_tags_hierarchically(new_data, idx, tag_pools, tag_hierarchy)
            stats["processed"] += 1

            # If completion is successful, update the data
            if completion_result["status"] == "success" and "completed_tags" in completion_result:
                # Update tags
                for level, tag_info in completion_result["completed_tags"].items():
                    level = int(level)
                    if tag_info["id"] is not None:
                        # Update tag text
                        item_data['tags'][idx][level] = tag_info["name"]
                        # Update tag index
                        item_data['tags_indices'][idx, level] = tag_info["id"]
                        # Update tag embedding (if available)
                        if tag_info["embedding"] is not None and 'tags_emb' in item_data:
                            item_data['tags_emb'][idx, level] = tag_info["embedding"]

                # Check if it's an item with an empty title; if so, use the completed tags as the new title
                if 'text' in item_data and idx in empty_title_items:
                    # Get all tags (both original and newly completed)
                    all_tags = item_data['tags'][idx]
                    # Filter out empty tags
                    valid_tags = [tag for tag in all_tags if tag and tag.strip() != '']

                    if valid_tags:
                        # Use the combination of tags as the new title
                        new_title = " ".join(valid_tags)
                        item_data['text'][idx] = new_title
                        stats["empty_titles_filled"] += 1
                        pbar.set_postfix({"Filled Title": new_title[:20] + ('...' if len(new_title) > 20 else '')})

                stats["successful"] += 1
            else:
                stats["failed"] += 1
                error_msg = completion_result.get("message", "Unknown error")
                stats["errors"].append((idx, error_msg))

            # Save data every 50 processed items
            if stats["processed"] % 50 == 0 and save_path:
                temp_save_path = f"{save_path}_temp"
                torch.save(new_data, temp_save_path)
                print(f"\nSaved temporary data to {temp_save_path}")

        # Save data after each batch
        if save_path:
            torch.save(new_data, save_path)
            print(f"\nSaved batch {batch_num+1} data to {save_path}")

    # Save final data
    if save_path:
        torch.save(new_data, save_path)
        print(f"\nSaved final data to {save_path}")

    # Print statistics
    print("\n===== Processing Statistics =====")
    print(f"Total items processed: {stats['processed']}")
    print(f"Successfully completed: {stats['successful']}")
    print(f"Failed to complete: {stats['failed']}")
    print(f"Skipped (already complete): {stats['skipped']}")
    print(f"Empty titles filled: {stats['empty_titles_filled']}")

    if stats["errors"]:
        print(f"\nFirst 10 errors:")
        for idx, error in stats["errors"][:10]:
            print(f"  - Item index {idx}: {error}")

    return new_data


def view_tags(data, num_samples=100, seed=42):
    """
    Randomly sample items and print their tag information.

    Args:
        data: The data object.
        num_samples: The number of samples to draw.
        seed: The random seed.
    """
    # Set random seed
    random.seed(seed)

    item_data = data['item']
    num_items = item_data['x'].shape[0]

    # Random sampling
    sample_indices = random.sample(range(num_items), min(num_samples, num_items))

    print(f"\n===== Randomly Sampled Information for {len(sample_indices)} Items =====")

    # Calculate tag completeness statistics for the sample
    complete_count = 0
    level_complete_counts = [0, 0, 0]

    for i, idx in enumerate(sample_indices):
        title = item_data['text'][idx] if 'text' in item_data and item_data['text'] is not None else "Unknown Title"
        tags = item_data['tags'][idx]
        tags_indices = item_data['tags_indices'][idx]

        # Check tag completeness
        is_complete = True
        for level in range(len(tags)):
            if tags_indices[level] != -1 and tags[level] != '':
                level_complete_counts[level] += 1
            else:
                is_complete = False

        if is_complete:
            complete_count += 1

        print(f"\nSample {i+1} (Index {idx}):")
        print(f"  Title: {title[:50]}{'...' if len(title) > 50 else ''}")
        print(f"  Tags:")
        for level in range(len(tags)):
            tag_status = "Valid" if tags_indices[level] != -1 and tags[level] != '' else "Missing"
            tag_text = tags[level] if tags_indices[level] != -1 and tags[level] != '' else "None"
            print(f"    - Level {level+1}: {tag_text} (ID: {tags_indices[level].item()}, Status: {tag_status})")

    # Print statistics for the sample
    print("\n===== Tag Statistics for Sample =====")
    print(f"Items with complete tags: {complete_count}/{len(sample_indices)} ({complete_count/len(sample_indices)*100:.2f}%)")
    for level in range(3):
        print(f"Items with complete Level {level+1} tag: {level_complete_counts[level]}/{len(sample_indices)} ({level_complete_counts[level]/len(sample_indices)*100:.2f}%)")


def analyze_tag_distribution(data):
    """
    Analyze the distribution of tags.

    Args:
        data: The data object.
    """
    print("\n===== Tag Distribution Analysis =====")

    item_data = data['item']
    tags_indices = item_data['tags_indices']
    num_items = tags_indices.shape[0]
    num_levels = tags_indices.shape[1]

    print(f"Total number of items: {num_items}")
    print(f"Number of tag levels: {num_levels}")

    # Analyze the distribution for each level
    for level in range(num_levels):
        level_indices = tags_indices[:, level]

        # Calculate number of valid tags
        valid_count = (level_indices != -1).sum().item()
        valid_percentage = valid_count / num_items * 100

        print(f"\nLevel {level+1} Tag Statistics:")
        print(f"  Number of valid tags: {valid_count}/{num_items} ({valid_percentage:.2f}%)")

        # Count unique tags
        unique_indices = torch.unique(level_indices)
        print(f"  Number of unique tags: {len(unique_indices)}")

        # Calculate tag frequency
        if valid_count > 0:
            # Exclude -1 (missing value)
            valid_indices = level_indices[level_indices != -1]

            # Count occurrences of each tag
            tag_counts = {}
            for idx in valid_indices:
                idx_val = idx.item()
                tag_counts[idx_val] = tag_counts.get(idx_val, 0) + 1

            # Sort tags by frequency
            sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)

            # Display the top 10 most frequent tags
            print(f"  Top 10 most frequent tags:")
            for i, (tag_id, count) in enumerate(sorted_tags[:10]):
                if i < 10:
                    # Get tag text (if available)
                    tag_text = "Unknown"
                    for j in range(num_items):
                        if tags_indices[j, level].item() == tag_id:
                            tag_text = item_data['tags'][j][level]
                            break

                    print(f"    {i+1}. ID: {tag_id}, Text: {tag_text}, Occurrences: {count}, Percentage: {count/num_items*100:.2f}%")

    # Analyze tag completeness
    complete_items = 0
    level_complete_counts = [0] * num_levels

    for i in range(num_items):
        item_indices = tags_indices[i]

        # Check if each level is valid
        is_complete = True
        for level in range(num_levels):
            if item_indices[level] != -1:
                level_complete_counts[level] += 1
            else:
                is_complete = False

        if is_complete:
            complete_items += 1

    print("\nTag Completeness Statistics:")
    print(f"Items with complete tags: {complete_items}/{num_items} ({complete_items/num_items*100:.2f}%)")

    for level in range(num_levels):
        print(f"Items with a valid Level {level+1} tag: {level_complete_counts[level]}/{num_items} ({level_complete_counts[level]/num_items*100:.2f}%)")


def check_completion_progress(data, save_path=None):
    """
    Check the tag completion progress and find the index of the first item with a missing tag.

    Args:
        data: The data object.
        save_path: (Optional) The path where completed data is saved.

    Returns:
        The index of the first incomplete item, or None if all are complete.
    """
    print("\n===== Checking Tag Completion Progress =====")

    if 'item' not in data:
        raise ValueError("Dataset does not contain 'item' node.")

    item_data = data['item']

    if 'tags_indices' not in item_data:
        raise ValueError("Dataset does not contain tag index information.")

    tags_indices = item_data['tags_indices']
    num_items = tags_indices.shape[0]
    num_levels = tags_indices.shape[1]

    # Tally completeness for each level
    level_complete_counts = [0] * num_levels
    incomplete_items = []

    for i in range(num_items):
        item_indices = tags_indices[i]

        has_missing = False
        for level in range(num_levels):
            if item_indices[level] != -1:
                level_complete_counts[level] += 1
            else:
                has_missing = True

        if has_missing:
            incomplete_items.append(i)

    # Print statistics
    print(f"Total number of items: {num_items}")
    print(f"Number of tag levels: {num_levels}")

    for level in range(num_levels):
        complete_percentage = level_complete_counts[level] / num_items * 100
        print(f"Level {level+1} tag completeness: {level_complete_counts[level]}/{num_items} ({complete_percentage:.2f}%)")

    total_incomplete = len(incomplete_items)
    complete_percentage = (num_items - total_incomplete) / num_items * 100
    print(f"Fully complete items: {num_items - total_incomplete}/{num_items} ({complete_percentage:.2f}%)")

    # Find the first item with an incomplete tag
    first_incomplete_idx = None
    if incomplete_items:
        first_incomplete_idx = incomplete_items[0]
        print(f"\nFirst item with an incomplete tag is at index: {first_incomplete_idx}")

        # Print this item's info
        tags = item_data['tags'][first_incomplete_idx]
        tags_indices_item = item_data['tags_indices'][first_incomplete_idx]

        print("\nTag information for this item:")
        for level in range(num_levels):
            tag_status = "Valid" if tags_indices_item[level] != -1 else "Missing"
            tag_text = tags[level] if tags_indices_item[level] != -1 else "None"
            print(f"  Level {level+1}: {tag_text} (ID: {tags_indices_item[level].item()}, Status: {tag_status})")

        # Provide command to resume completion if a save path is given
        if save_path:
            print(f"\nTo continue completing tags, use the following command:")
            print(f"python -m data.fill_kuairand_simple --data_path {save_path} --start_idx {first_incomplete_idx}")
    else:
        print("\nAll items have complete tags!")

    return first_incomplete_idx


def fill_empty_titles(data, save_path=None):
    """
    Fill empty titles for items using their existing tags as the new title.

    Args:
        data: The data object.
        save_path: The path to save the updated data.

    Returns:
        The updated data object.
    """
    print("\n===== Starting to Fill Empty Titles =====")

    # Create a copy to avoid modifying the original data
    new_data = deepcopy(data)
    item_data = new_data['item']

    if 'text' not in item_data or item_data['text'] is None:
        print("No 'text' field in data, cannot fill titles.")
        return new_data

    # Find items with empty titles
    empty_title_items = []
    for i in range(len(item_data['text'])):
        if not item_data['text'][i] or item_data['text'][i].strip() == '':
            empty_title_items.append(i)

    print(f"Found {len(empty_title_items)} items with empty titles.")

    if not empty_title_items:
        print("No empty titles to fill.")
        return new_data

    # Fill empty titles
    filled_count = 0
    for idx in tqdm(empty_title_items, desc="Filling empty titles"):
        # Get all tags for the item
        if 'tags' in item_data and idx < len(item_data['tags']):
            tags = item_data['tags'][idx]

            # Filter out empty tags
            valid_tags = [tag for tag in tags if tag and tag.strip() != '']

            if valid_tags:
                # Use the combination of tags as the new title
                new_title = " ".join(valid_tags)
                item_data['text'][idx] = new_title
                filled_count += 1

    print(f"\nSuccessfully filled {filled_count} empty titles.")

    # Save the updated data
    if save_path:
        torch.save(new_data, save_path)
        print(f"Saved updated data to: {save_path}")

    return new_data


def main():
    """Main function"""
    # Set default paths
    default_data_path = os.path.join('dataset', 'kuairand', 'processed', 'title_data_kuairand_5tags.pt')
    default_save_path = os.path.join('dataset', 'kuairand', 'processed', 'title_data_kuairand_5tags_completed.pt')

    # Set up command-line argument parser
    import argparse
    parser = argparse.ArgumentParser(description='KuaiRand Dataset Simple Tag Completion Tool')
    parser.add_argument('--data_path', type=str, default=default_data_path,
                        help=f'Path to the KuaiRand dataset file (default: {default_data_path})')
    parser.add_argument('--save_path', type=str, default=default_save_path,
                        help=f'Path to save the completed data (default: {default_save_path})')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for processing (default: 100)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting item index for resuming processing (default: 0)')
    parser.add_argument('--check_progress', action='store_true',
                        help='Check tag completion progress and find the first incomplete item')
    parser.add_argument('--view_tags', action='store_true',
                        help='Randomly sample and print item tag information')
    parser.add_argument('--analyze_tags', action='store_true',
                        help='Analyze the tag distribution')
    parser.add_argument('--fill_titles', action='store_true',
                        help='Only fill empty titles, do not complete tags')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of random samples to view (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset
    print(f"Loading dataset: {args.data_path}")
    data = load_kuairand_dataset(args.data_path, args.seed)

    if args.view_tags:
        # View tag information
        view_tags(data, args.num_samples, args.seed)
    elif args.analyze_tags:
        # Analyze tag distribution
        analyze_tag_distribution(data)
    elif args.check_progress:
        # Check tag completion progress
        first_incomplete_idx = check_completion_progress(data, args.save_path)
        if first_incomplete_idx is not None:
            print(f"\nTo continue completing tags, use the following command:")
            print(f"python -m data.fill_kuairand_simple --data_path {args.save_path} --start_idx {first_incomplete_idx}")
    elif args.fill_titles:
        # Only fill empty titles
        fill_empty_titles(data, args.save_path)
    else:
        # Create tag pools
        tag_pools = create_tag_pools(data)

        # Build tag hierarchy relationships
        tag_hierarchy = build_tag_hierarchy(data)

        # Perform simple batch tag completion and save
        print(f"Starting tag completion with hierarchical constraints. Data will be saved to: {args.save_path}")
        simple_complete_tags(
            data,
            tag_pools,
            tag_hierarchy,
            args.batch_size,
            args.save_path,
            args.start_idx
        )

    print("\nProcessing finished!")


if __name__ == "__main__":
    main()

    # --- USAGE EXAMPLES ---
    
    # 1. Analyze tag distribution:
    # ```
    # python -m data.fill_kuairand_simple --analyze_tags
    # ```

    # 2. Randomly sample items and view their tags:
    # ```
    # python -m data.fill_kuairand_simple --view_tags --num_samples 20
    # ```

    # 3. Check tag completion progress:
    # ```
    # python -m data.fill_kuairand_simple --check_progress --data_path path/to/your/data_completed.pt
    # ```

    # 4. Execute tag completion:
    # ```
    # python -m data.fill_kuairand_simple
    # ```
    
    # 5. Only fill empty titles:
    # ```
    # python -m data.fill_kuairand_simple --fill_titles
    # ```
