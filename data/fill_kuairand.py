python
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from torch_geometric.data import HeteroData
from typing import List, Dict, Tuple, Optional, Union
import logging
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from data.chat_with_llm import query_llm, parse_llm_response, batch_query_llm, get_model_stats


def load_kuairand_dataset(data_path, seed=42):
    """
    Loads the KuaiRand dataset and returns a HeteroData object.

    Args:
        data_path: Path to the data file.
        seed: Random seed for reproducibility.

    Returns:
        A HeteroData object.
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
        # Load the dataset, explicitly setting weights_only=False to handle the default behavior change in PyTorch 2.6
        data = torch.load(data_path, map_location='cpu', weights_only=False)
        print("✓ Data loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        # Attempting to load with an alternative method
        try:
            print("Attempting to load with an alternative method...")
            # Add numpy._core.multiarray._reconstruct to safe globals
            import numpy
            torch.serialization.add_safe_globals(['_reconstruct'])
            data = torch.load(data_path, map_location='cpu', weights_only=False)
            print("✓ Successfully loaded using the alternative method!")
            return data
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            raise


def print_dataset_structure(data):
    """
    Prints the structure information of the dataset.

    Args:
        data: A HeteroData object.
    """
    print("\n===== Dataset Structure =====")

    # Iterate over all node types
    for node_type in data.node_types:
        print(f"\nNode Type: {node_type}")
        node_data = data[node_type]

        for key, value in node_data.items():
            if isinstance(value, torch.Tensor):
                shape_str = list(value.shape)
                print(f"  - {key}: {shape_str}, dtype: {value.dtype}")
            elif isinstance(value, list):
                print(f"  - {key}: List, length: {len(value)}")
            elif isinstance(value, dict):
                print(f"  - {key}: Dict, keys: {list(value.keys())}")
            elif isinstance(value, bool):
                print(f"  - {key}: Boolean, value: {value}")
            else:
                print(f"  - {key}: {type(value)}")

    # Iterate over all edge types
    for edge_type in data.edge_types:
        print(f"\nEdge Type: {edge_type}")
        edge_data = data[edge_type[0], edge_type[1], edge_type[2]]

        for key, value in edge_data.items():
            if isinstance(value, torch.Tensor):
                shape_str = list(value.shape)
                print(f"  - {key}: {shape_str}, dtype: {value.dtype}")
            elif isinstance(value, dict):
                print(f"  - {key}: Dict")
                for k, v in value.items():
                    print(f"    - {k}:")
                    for subk, subv in v.items():
                        if isinstance(subv, torch.Tensor):
                            shape_str = list(subv.shape)
                            print(f"      - {subk}: {shape_str}, dtype: {subv.dtype}")
                        elif isinstance(subv, list):
                            print(f"      - {subk}: List, length: {len(subv)}")
                        else:
                            print(f"      - {subk}: {type(subv)}")
            else:
                print(f"  - {key}: {type(value)}")


def sample_and_print_values(data, num_samples=2):
    """
    Randomly samples and prints specific values for various attributes.

    Args:
        data: A HeteroData object.
        num_samples: The number of samples for each attribute.
    """
    print("\n===== Randomly Sampled Attribute Values =====")

    # Sample attributes of 'item' nodes
    if 'item' in data.node_types:
        print("\nItem Node Attribute Samples:")
        item_data = data['item']
        num_items = item_data.x.shape[0]

        # Randomly select sample indices
        sample_indices = random.sample(range(num_items), min(num_samples, num_items))

        for i, idx in enumerate(sample_indices):
            print(f"\nSample {i+1} (Index {idx}):")

            # Print text features
            if hasattr(item_data, 'text') and item_data.text is not None:
                text = item_data.text[idx] if idx < len(item_data.text) else "N/A"
                print(f"  - Text: {text}")

            # Print feature vector
            if hasattr(item_data, 'x') and item_data.x is not None:
                features = item_data.x[idx]
                print(f"  - First 5 features: {features[:5].tolist()}")

            # Print tag text
            if hasattr(item_data, 'tags') and item_data.tags is not None:
                tags = item_data.tags[idx]
                print(f"  - Tags Text: {tags.tolist() if hasattr(tags, 'tolist') else tags}")

            # Print tag indices
            if hasattr(item_data, 'tags_indices') and item_data.tags_indices is not None:
                tag_indices = item_data.tags_indices[idx]
                print(f"  - Tag Indices: {tag_indices.tolist()}")

            # Print tag embeddings
            if hasattr(item_data, 'tags_emb') and item_data.tags_emb is not None:
                tag_emb = item_data.tags_emb[idx]
                if len(tag_emb.shape) > 1:
                    for j in range(tag_emb.shape[0]):
                        print(f"  - Level {j+1} tag embedding (first 3 dims): {tag_emb[j][:3].tolist()}")
                else:
                    print(f"  - Tag embedding (first 3 dims): {tag_emb[:3].tolist()}")

            # Handle the three levels of tag embeddings separately
            for l in range(1, 4):
                if hasattr(item_data, f'tags_emb_l{l}') and getattr(item_data, f'tags_emb_l{l}') is not None:
                    tag_emb = getattr(item_data, f'tags_emb_l{l}')[idx]
                    print(f"  - Level {l} tag embedding (first 3 dims): {tag_emb[:3].tolist()}")

            # Print train/test split information
            if hasattr(item_data, 'is_train') and item_data.is_train is not None:
                is_train = item_data.is_train[idx].item()
                print(f"  - Is Train: {is_train}")

    # Sample user-item interaction sequences
    if ('user', 'rated', 'item') in data.edge_types:
        print("\nUser-Item Interaction Sequence Samples:")

        history = data['user', 'rated', 'item'].history

        # Randomly sample training sequences
        if 'train' in history and 'itemId' in history['train']:
            train_sequences = history['train']['itemId']
            train_targets = history['train']['itemId_fut']

            num_train_seqs = len(train_sequences)
            train_sample_indices = random.sample(range(num_train_seqs), min(num_samples, num_train_seqs))

            for i, idx in enumerate(train_sample_indices):
                print(f"\nTraining Sample {i+1} (Index {idx}):")

                if isinstance(train_sequences, list):
                    sequence = train_sequences[idx]
                    # Filter out padding value -1
                    sequence = [item for item in sequence if item != -1]
                    print(f"  - Input Sequence: {sequence}")
                else:
                    sequence = train_sequences[idx]
                    # Filter out padding value -1
                    sequence = sequence[sequence != -1].tolist()
                    print(f"  - Input Sequence: {sequence}")

                target = train_targets[idx].item() if isinstance(train_targets[idx], torch.Tensor) else train_targets[idx]
                print(f"  - Target Item: {target}")

        # Randomly sample validation sequences
        if 'eval' in history and 'itemId' in history['eval']:
            eval_sequences = history['eval']['itemId']
            eval_targets = history['eval']['itemId_fut']

            num_eval_seqs = len(eval_sequences)
            eval_sample_indices = random.sample(range(num_eval_seqs), min(num_samples, num_eval_seqs))

            for i, idx in enumerate(eval_sample_indices):
                print(f"\nValidation Sample {i+1} (Index {idx}):")

                sequence = eval_sequences[idx]
                # Filter out padding value -1
                sequence = sequence[sequence != -1].tolist()
                print(f"  - Input Sequence: {sequence}")

                target = eval_targets[idx].item() if isinstance(eval_targets[idx], torch.Tensor) else eval_targets[idx]
                print(f"  - Target Item: {target}")


def analyze_tag_completeness(data):
    """
    Analyzes the completeness statistics of tags.

    Args:
        data: A HeteroData object.
    """
    print("\n===== Tag Completeness Analysis =====")

    if 'item' not in data.node_types or not hasattr(data['item'], 'tags_indices'):
        print("Error: Tag information or tag indices not available in the dataset")
        return

    tags_indices = data['item'].tags_indices
    num_items = tags_indices.shape[0]
    num_levels = tags_indices.shape[1]

    print(f"Tags have {num_levels} levels, total items: {num_items}")

    # Calculate tag existence statistics for each level
    level_stats = []
    for level in range(num_levels):
        # Number of valid tags (index is not -1)
        valid_count = (tags_indices[:, level] != -1).sum().item()
        coverage = valid_count / num_items * 100
        level_stats.append((valid_count, coverage))

        print(f"Level {level+1} Tags:")
        print(f"  - Valid tags: {valid_count} / {num_items} ({coverage:.2f}%)")
        print(f"  - Missing tags: {num_items - valid_count} ({100 - coverage:.2f}%)")

    # Calculate multi-level tag completeness
    tag_completeness = torch.sum(tags_indices != -1, dim=1)
    completeness_counter = Counter(tag_completeness.tolist())

    print("\nMulti-level Tag Completeness Statistics:")
    for level_count in range(num_levels + 1):
        count = completeness_counter.get(level_count, 0)
        percentage = (count / num_items) * 100
        print(f"  - Items with {level_count} levels of tags: {count} ({percentage:.2f}%)")

    # Analyze tag combination patterns
    print("\nTag Combination Pattern Analysis:")
    # Create a binary mask to represent tag existence patterns
    # e.g., [1,0,1] means level 1 and 3 tags exist, but level 2 is missing
    patterns = []
    for i in range(num_items):
        pattern = tuple((tags_indices[i] != -1).tolist())
        patterns.append(pattern)

    pattern_counter = Counter(patterns)

    for pattern, count in pattern_counter.most_common():
        percentage = (count / num_items) * 100
        pattern_str = ['Present' if p else 'Absent' for p in pattern]
        print(f"  - Pattern [{', '.join(pattern_str)}]: {count} items ({percentage:.2f}%)")

    # Plot tag completeness visualization
    plt.figure(figsize=(10, 6))
    levels = [f"Level {i+1}" for i in range(num_levels)]
    coverages = [stats[1] for stats in level_stats]

    plt.bar(levels, coverages, color='skyblue')
    plt.ylim(0, 100)
    plt.xlabel('Tag Level')
    plt.ylabel('Coverage (%)')
    plt.title('Tag Level Coverage in KuaiRand Dataset')

    # Add value labels
    for i, coverage in enumerate(coverages):
        plt.text(i, coverage + 1, f'{coverage:.1f}%', ha='center')

    # Save the plot
    save_dir = 'plots'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kuairand_tag_coverage.png'), dpi=300)
    print(f"\nTag coverage plot saved to: {os.path.join(save_dir, 'kuairand_tag_coverage.png')}")
    plt.close()

    # Plot tag completeness distribution
    plt.figure(figsize=(10, 6))
    counts = [completeness_counter.get(i, 0) for i in range(num_levels + 1)]
    percentages = [(count / num_items) * 100 for count in counts]

    plt.bar(range(num_levels + 1), percentages, color='lightgreen')
    plt.xticks(range(num_levels + 1))
    plt.xlabel('Number of Tag Levels')
    plt.ylabel('Percentage of Items (%)')
    plt.title('Item Tag Completeness Distribution in KuaiRand Dataset')

    # Add value labels
    for i, percentage in enumerate(percentages):
        if percentage > 0:
            plt.text(i, percentage + 0.5, f'{percentage:.1f}%', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kuairand_tag_completeness.png'), dpi=300)
    print(f"Tag completeness distribution plot saved to: {os.path.join(save_dir, 'kuairand_tag_completeness.png')}")
    plt.close()


def create_tag_pools(data):
    """
    Creates tag pools for each level, including tag text and corresponding embeddings.

    Args:
        data: A HeteroData object.

    Returns:
        A list of tag pools for the three levels, each containing tag text and embeddings.
    """
    print("Creating tag pools...")

    if 'item' not in data.node_types or not hasattr(data['item'], 'tags_indices'):
        raise ValueError("Tag information or tag indices not available in the dataset")

    item_data = data['item']
    tags_indices = item_data.tags_indices

    # Get tag text data
    if not hasattr(item_data, 'tags') or item_data.tags is None:
        raise ValueError("Tag text information not available in the dataset")

    # Create tag pools
    tag_pools = []

    for level in range(tags_indices.shape[1]):
        # Get all valid tag indices
        valid_indices = tags_indices[:, level]
        valid_indices = valid_indices[valid_indices != -1]
        unique_indices = torch.unique(valid_indices)

        # Create tag pool for the current level
        level_pool = {}

        for idx in unique_indices:
            # Find all items with this tag
            items_with_tag = (tags_indices[:, level] == idx).nonzero(as_tuple=True)[0]

            if len(items_with_tag) == 0:
                continue

            # Get the tag text
            sample_item = items_with_tag[0].item()
            tag_text = item_data.tags[sample_item, level]

            # Get the tag embedding
            if hasattr(item_data, f'tags_emb_l{level+1}'):
                # Get embeddings of all items with this tag and take the average
                tag_embs = getattr(item_data, f'tags_emb_l{level+1}')[items_with_tag]
                tag_emb = torch.mean(tag_embs, dim=0)
            else:
                # If no specific tag embedding is available, use the feature vector of a sample item
                tag_emb = item_data.x[sample_item]

            # Add to the tag pool
            if tag_text and tag_text != '':
                level_pool[idx.item()] = {
                    'text': tag_text,
                    'embedding': tag_emb,
                    'count': len(items_with_tag)
                }

        tag_pools.append(level_pool)
        print(f"Level {level+1} tag pool created with {len(level_pool)} unique tags")

    return tag_pools

def cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two vectors."""
    dot_product = torch.sum(vec1 * vec2)
    norm1 = torch.sqrt(torch.sum(vec1 * vec1))
    norm2 = torch.sqrt(torch.sum(vec2 * vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def retrieve_candidate_tags(title_embedding, tag_pool, top_k=6):
    """
    Retrieves the most similar candidate tags based on a title embedding.

    Args:
        title_embedding: The embedding vector of the title.
        tag_pool: The tag pool, containing tag ID, text, and embedding.
        top_k: The number of candidates to return.

    Returns:
        The top_k most similar candidate tags.
    """
    similarities = []

    for tag_id, tag_info in tag_pool.items():
        tag_embedding = tag_info['embedding']
        similarity = cosine_similarity(title_embedding, tag_embedding)
        similarities.append((tag_id, tag_info['text'], similarity.item(), tag_info['count']))

    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[2], reverse=True)

    # Return the top_k candidates
    return similarities[:top_k]

def generate_optimized_prompt(data, item_idx, tag_pools, top_k=6):
    """
    Generates a structured prompt based on video title and existing tags
    to facilitate tag extraction from LLM responses.

    Args:
        data: A HeteroData object.
        item_idx: The index of the item.
        tag_pools: A list of tag pools for the three levels.
        top_k: Number of candidates to retrieve for each missing level.

    Returns:
        The prompt string.
    """
    item_data = data['item']

    # Get video title
    title = item_data.text[item_idx] if hasattr(item_data, 'text') and item_data.text is not None else "Unknown Title"

    # Get video tags
    tags = item_data.tags[item_idx]
    tags_indices = item_data.tags_indices[item_idx]

    # Get video feature vector
    item_embedding = item_data.x[item_idx]

    # Check each level for missing tags
    missing_levels = []
    for level in range(len(tags)):
        if tags_indices[level] == -1 or tags[level] == '':
            missing_levels.append(level)

    # If no tags are missing, no prompt is needed
    if not missing_levels:
        return f"Video '{title}' already has complete three-level tags: {tags[0]}, {tags[1]}, {tags[2]}. No completion needed."

    # Construct a more structured prompt, requiring the LLM to return in JSON format
    prompt = f"""Please help me choose the most appropriate category tags for a short video.

## Video Information
- Title: {title}

## Current Tag Status
"""

    for level in range(len(tags)):
        level_name = f"Level {level+1}"
        tag_text = tags[level] if tags[level] != '' else "Missing"
        prompt += f"- {level_name} Tag: {tag_text}\n"

    prompt += "\n## Task Description\n"
    prompt += "For each missing tag level, please select the most suitable tag from the candidates below. You should infer the video's content based on its title and existing tags and choose the most relevant candidate.\n\n"

    # Retrieve candidates for each missing level
    for level in missing_levels:
        candidates = retrieve_candidate_tags(item_embedding, tag_pools[level], top_k)

        prompt += f"## Level {level+1} Tag Candidates\n"
        for i, (tag_id, tag_text, similarity, count) in enumerate(candidates):
            similarity_percent = similarity * 100
            # Do not show IDs in the prompt, let the large model focus only on the tag text
            prompt += f"{i+1}. {tag_text} (Similarity: {similarity_percent:.2f}%, Occurrences: {count})\n"
        prompt += "\n"

    # Modify the format requirements
    prompt += """## Required Output Format
Please return your selection in JSON format. The JSON object must contain the following fields:
json
{
  "reasoning": "A brief explanation of why you chose these tags",
  "completed_tags": {
    "level_0": "Selected name for the Level 1 tag",
    "level_1": "Selected name for the Level 2 tag",
    "level_2": "Selected name for the Level 3 tag"
  }
}


Please ensure your response is in valid JSON format and only fill in the missing tag levels (do not include existing tags).
Level 1 corresponds to 'level\_0', Level 2 to 'level\_1', and Level 3 to 'level\_2'.
For example, if the Level 3 tag is missing, you only need to include 'level\_2' in the completed\_tags object.
"""


return prompt


def call\_llm\_for\_tag\_completion(prompt, temperature=0.3):
"""Calls the LLM to get tag completion results."""
system\_prompt = "You are an expert video tag classifier. Your task is to select the most appropriate category tags for a video based on its title and existing tags. Your response must strictly follow the required JSON format."


response = query_llm(
    prompt=prompt,
    system_prompt=system_prompt,
    temperature=temperature,
    max_tokens=800
)

# Parse the response
parsed = parse_llm_response(response, expected_format="json")
return response, parsed


def find\_tag\_id\_by\_name(tag\_pools, level, tag\_name):
"""
Finds the corresponding ID for a tag name in the tag pool.


Args:
    tag_pools: A list of tag pools.
    level: The tag level.
    tag_name: The name of the tag.

Returns:
    The tag ID and its embedding, or (None, None) if not found.
"""
for tag_id, tag_info in tag_pools[level].items():
    if tag_info['text'] == tag_name:
        return tag_id, tag_info['embedding']
return None, None


def complete\_tags\_with\_llm(data, item\_idx, tag\_pools, max\_retries=2):
"""
Completes missing tags using an LLM.


Args:
    data: A HeteroData object.
    item_idx: The index of the item.
    tag_pools: A list of tag pools for the three levels.
    max_retries: Maximum number of retries.

Returns:
    A dictionary with the completion results, including completed tags, IDs, embeddings, and the raw response.
"""
# Get the video's tag status
item_data = data['item']
tags = item_data.tags[item_idx]
tags_indices = item_data.tags_indices[item_idx]

# Check for any missing tags
missing_levels = []
for level in range(len(tags)):
    if tags_indices[level] == -1 or tags[level] == '':
        missing_levels.append(level)

if not missing_levels:
    return {"status": "complete", "message": "All tags are already complete, no action needed"}

# Generate the prompt
prompt = generate_optimized_prompt(data, item_idx, tag_pools)

# LLM call with a retry mechanism
retry_count = 0
while retry_count <= max_retries:
    try:
        # Call the LLM
        raw_response, llm_result = call_llm_for_tag_completion(prompt)

        if not llm_result["success"]:
            if retry_count < max_retries:
                retry_count += 1
                continue
            return {
                "status": "error",
                "message": f"LLM call failed: {llm_result.get('error', 'Unknown error')}",
                "raw_response": raw_response
            }

        # Extract completed tags
        try:
            parsed_data = llm_result["parsed_data"]
            completion_result = {
                "status": "success",
                "reasoning": parsed_data.get("reasoning", "No reasoning provided"),
                "completed_tags": {},
                "raw_response": raw_response,
                "parsed_response": llm_result
            }

            # Extract completed tags for each level
            completed_tags_dict = parsed_data.get("completed_tags", {})

            # Handle various possible key formats: 'level_X', 'X', integer X
            for level in missing_levels:
                # Try various possible key formats
                level_keys = [f"level_{level}", str(level), level]
                found_key = None

                # Check each possible key format
                for key in level_keys:
                    if key in completed_tags_dict:
                        found_key = key
                        break

                if found_key is not None:
                    tag_name = completed_tags_dict[found_key]
                    # If the value is a dict, try to get the 'name' field
                    if isinstance(tag_name, dict) and "name" in tag_name:
                        tag_name = tag_name["name"]

                    # Find the corresponding ID and embedding from the tag pool
                    tag_id, tag_embedding = find_tag_id_by_name(tag_pools, level, tag_name)

                    if tag_id is not None:
                        completion_result["completed_tags"][level] = {
                            "id": tag_id,
                            "name": tag_name,
                            "embedding": tag_embedding
                        }
                    else:
                        # If the corresponding tag is not found in the tag pool
                        completion_result["completed_tags"][level] = {
                            "id": None,
                            "name": tag_name,
                            "embedding": None,
                            "error": "Could not find the corresponding tag in the tag pool"
                        }

            # Check if at least one tag was effectively completed
            if not completion_result["completed_tags"] and retry_count < max_retries:
                retry_count += 1
                continue

            return completion_result
        except Exception as e:
            if retry_count < max_retries:
                retry_count += 1
                continue
            return {
                "status": "error",
                "message": f"Failed to parse LLM response: {str(e)}",
                "raw_response": raw_response,
                "parsed_response": llm_result
            }
    except Exception as e:
        if retry_count < max_retries:
            retry_count += 1
            continue
        return {
            "status": "error",
            "message": f"An exception occurred: {str(e)}"
        }


def analyze\_and\_complete\_tags(data, tag\_pools, num\_samples=5, seed=None):
"""
Analyzes current tags and completes missing ones using an LLM for random samples.


Args:
    data: A HeteroData object.
    tag_pools: A list of tag pools for the three levels.
    num_samples: The number of samples.
    seed: Random seed for reproducibility.
"""
if seed is not None:
    random.seed(seed)

item_data = data['item']
num_items = item_data.x.shape[0]

# Find items missing at least one level of tags
incomplete_items = []
for i in range(num_items):
    tags_indices = item_data.tags_indices[i]
    if torch.any(tags_indices == -1):
        incomplete_items.append(i)

print(f"Found {len(incomplete_items)} items with missing tags")

# Randomly select samples
sample_indices = random.sample(incomplete_items, min(num_samples, len(incomplete_items)))

# Analyze and complete tags for each sample
for i, idx in enumerate(sample_indices):
    print(f"\n\n===== Sample {i+1} =====")

    # Print original data
    title = item_data.text[idx] if hasattr(item_data, 'text') and item_data.text is not None else "Unknown Title"
    tags = item_data.tags[idx]
    tags_indices = item_data.tags_indices[idx]

    print(f"Video Index: {idx}")
    print(f"Video Title: {title}")
    print(f"Current Tags: {tags.tolist() if hasattr(tags, 'tolist') else tags}")
    print(f"Tag Indices: {tags_indices.tolist()}")

    # Generate the prompt
    prompt = generate_optimized_prompt(data, idx, tag_pools)
    print("\n----- Generated Prompt -----")
    print(prompt)

    print("\n----- Calling LLM for Tag Completion -----")
    completion_result = complete_tags_with_llm(data, idx, tag_pools)

    # Show raw LLM response
    print("\nRaw LLM Response:")
    if "raw_response" in completion_result and completion_result["raw_response"] and "text" in completion_result["raw_response"]:
        print(completion_result["raw_response"]["text"])
    else:
        print("Could not retrieve raw response")

    # Show parsed result
    print("\nParsed Result:")
    # Create a simplified version of the result, removing large objects like embeddings for printing
    simplified_result = {k: v for k, v in completion_result.items() if k not in ["raw_response", "parsed_response"]}

    # Print key-value mappings to help understand the parsing process
    if "parsed_response" in completion_result and completion_result["parsed_response"].get("success", False):
        completed_tags_dict = completion_result["parsed_response"]["parsed_data"].get("completed_tags", {})
        if completed_tags_dict:
            print("\nParsed Key-Value Mappings:")
            for key, value in completed_tags_dict.items():
                print(f"  - Key: '{key}' -> Value: '{value}'")

    # Print the parsed result
    if "completed_tags" in simplified_result:
        for level, tag_info in simplified_result["completed_tags"].items():
            if "embedding" in tag_info:
                # Only keep the shape information of the embedding
                embedding = tag_info["embedding"]
                if embedding is not None:
                    tag_info["embedding"] = f"Tensor(shape={list(embedding.shape)})"
                else:
                    tag_info["embedding"] = None

    print(json.dumps(simplified_result, ensure_ascii=False, indent=2))

    # If completion was successful, show the updated tag status
    if completion_result["status"] == "success":
        print("\n----- Completed Tags -----")
        completed_tags = list(tags)
        completed_indices = list(tags_indices.tolist())

        for level, tag_info in completion_result["completed_tags"].items():
            level = int(level)
            if tag_info["id"] is not None:
                completed_tags[level] = tag_info["name"]
                completed_indices[level] = tag_info["id"]

        print(f"Tags after completion: {completed_tags}")
        print(f"Indices after completion: {completed_indices}")

    # Add a delay to avoid frequent API calls
    if i < num_samples - 1:
        print("Waiting for 2 seconds before continuing...")
        time.sleep(2)


def generate\_prompts\_for\_samples(data, tag\_pools, num\_samples=5, seed=None):
"""
Generates prompts for completing tags for random samples.


Args:
    data: A HeteroData object.
    tag_pools: A list of tag pools for the three levels.
    num_samples: The number of samples.
    seed: Random seed for reproducibility.
"""
if seed is not None:
    random.seed(seed)

item_data = data['item']
num_items = item_data.x.shape[0]

# Find items missing at least one level of tags
incomplete_items = []
for i in range(num_items):
    tags_indices = item_data.tags_indices[i]
    if torch.any(tags_indices == -1):
        incomplete_items.append(i)

print(f"Found {len(incomplete_items)} items with missing tags")

# Randomly select samples
sample_indices = random.sample(incomplete_items, min(num_samples, len(incomplete_items)))

# Generate a prompt for each sample
for i, idx in enumerate(sample_indices):
    print(f"\n\n===== Sample {i+1} =====")

    # Print original data
    title = item_data.text[idx] if hasattr(item_data, 'text') and item_data.text is not None else "Unknown Title"
    tags = item_data.tags[idx]
    tags_indices = item_data.tags_indices[idx]

    print(f"Video Index: {idx}")
    print(f"Video Title: {title}")
    print(f"Current Tags: {tags.tolist() if hasattr(tags, 'tolist') else tags}")
    print(f"Tag Indices: {tags_indices.tolist()}")

    # Generate the prompt
    prompt = generate_optimized_prompt(data, idx, tag_pools)
    print("\n----- Generated Prompt -----")
    print(prompt)


def batch\_complete\_tags(data, tag\_pools, batch\_size=100, save\_path=None, start\_idx=0):
"""
Batch completes tags and saves them to a new data file.


Args:
    data: A HeteroData object.
    tag_pools: A list of tag pools for the three levels.
    batch_size: Number of items to process in each batch.
    save_path: Path to save the completed data.
    start_idx: Starting index, for resuming an interrupted process.

Returns:
    The updated HeteroData object.
"""
# Create a copy of the data to avoid modifying the original
from copy import deepcopy
new_data = deepcopy(data)

item_data = new_data['item']
num_items = item_data.x.shape[0]

# Find items missing at least one level of tags
incomplete_items = []
for i in range(num_items):
    tags_indices = item_data.tags_indices[i]
    if torch.any(tags_indices == -1):
        incomplete_items.append(i)

# Filter out items before the start index
incomplete_items = [idx for idx in incomplete_items if idx >= start_idx]

total_incomplete = len(incomplete_items)
print(f"Found {total_incomplete} items with missing tags. Starting processing from index {start_idx}.")

# Create a progress bar for each batch
num_batches = (total_incomplete + batch_size - 1) // batch_size

# To store statistics for each batch result
stats = {
    "processed": 0,
    "successful": 0,
    "failed": 0,
    "skipped": 0,
    "errors": []
}

# Batch processing
for batch_num in range(num_batches):
    batch_start = batch_num * batch_size
    batch_end = min((batch_num + 1) * batch_size, total_incomplete)
    batch_indices = incomplete_items[batch_start:batch_end]

    print(f"\nProcessing batch {batch_num + 1}/{num_batches} (Item indices from {batch_indices[0]} to {batch_indices[-1]})")

    # Iterate through each item in the batch
    for i, idx in enumerate(tqdm(batch_indices, desc=f"Batch {batch_num+1}")):
        # Check if completion is still needed
        tags_indices = item_data.tags_indices[idx]
        missing_levels = [level for level in range(tags_indices.shape[0]) if tags_indices[level] == -1]

        if not missing_levels:
            stats["skipped"] += 1
            continue

        # Complete tags
        completion_result = complete_tags_with_llm(new_data, idx, tag_pools)
        stats["processed"] += 1

        # If completion is successful, update the data
        if completion_result["status"] == "success" and "completed_tags" in completion_result:
            # Update tags
            for level, tag_info in completion_result["completed_tags"].items():
                level = int(level)
                if tag_info["id"] is not None:
                    # Update tag text
                    item_data.tags[idx, level] = tag_info["name"]
                    # Update tag indices
                    item_data.tags_indices[idx, level] = tag_info["id"]
                    # Update tag embedding (if available)
                    if tag_info["embedding"] is not None and hasattr(item_data, f'tags_emb_l{level+1}'):
                        getattr(item_data, f'tags_emb_l{level+1}')[idx] = tag_info["embedding"]

            stats["successful"] += 1
        else:
            stats["failed"] += 1
            error_msg = completion_result.get("message", "Unknown error")
            stats["errors"].append((idx, error_msg))
            print(f"\nFailed to process item index {idx}: {error_msg}")

        # Save data after every 10 items
        if i % 10 == 9 and save_path:
            temp_save_path = f"{save_path}_temp"
            torch.save(new_data, temp_save_path)
            print(f"\nTemporary data saved to {temp_save_path}")

    # Save data after each batch
    if save_path:
        torch.save(new_data, save_path)
        print(f"\nData for batch {batch_num+1} saved to {save_path}")

# Save final data
if save_path:
    torch.save(new_data, save_path)
    print(f"\nFinal data saved to {save_path}")

# Print statistics
print("\n===== Processing Statistics =====")
print(f"Total items processed: {stats['processed']}")
print(f"Successfully completed: {stats['successful']}")
print(f"Failed: {stats['failed']}")
print(f"Skipped (tags already complete): {stats['skipped']}")

if stats["errors"]:
    print(f"\nFirst 10 errors:")
    for idx, error in stats["errors"][:10]:
        print(f"  - Item Index {idx}: {error}")

return new_data


def parallel\_complete\_tags(data, tag\_pools, batch\_size=50, save\_path=None, start\_idx=0, max\_workers=8, parallel\_batch\_size=10):
"""
Batch completes tags in parallel and saves them to a new data file.


Args:
    data: A HeteroData object.
    tag_pools: A list of tag pools for the three levels.
    batch_size: Number of items to process in each major batch.
    save_path: Path to save the completed data.
    start_idx: Starting index, for resuming an interrupted process.
    max_workers: Maximum number of parallel workers.
    parallel_batch_size: How many items to process in one parallel batch.

Returns:
    The updated HeteroData object.
"""
# Create a copy of the data to avoid modifying the original
from copy import deepcopy
new_data = deepcopy(data)

item_data = new_data['item']
num_items = item_data.x.shape[0]

# Find items missing at least one level of tags
incomplete_items = []
for i in range(num_items):
    tags_indices = item_data.tags_indices[i]
    if torch.any(tags_indices == -1):
        incomplete_items.append(i)

# Filter out items before the start index
incomplete_items = [idx for idx in incomplete_items if idx >= start_idx]

total_incomplete = len(incomplete_items)
print(f"Found {total_incomplete} items with missing tags. Starting processing from index {start_idx}.")

# Calculate number of major batches
num_batches = (total_incomplete + batch_size - 1) // batch_size

# To store statistics
stats = {
    "processed": 0,
    "successful": 0,
    "failed": 0,
    "skipped": 0,
    "errors": []
}

# Batch processing
for batch_num in range(num_batches):
    batch_start = batch_num * batch_size
    batch_end = min((batch_num + 1) * batch_size, total_incomplete)
    batch_indices = incomplete_items[batch_start:batch_end]

    print(f"\nProcessing batch {batch_num + 1}/{num_batches} (Item indices from {batch_indices[0]} to {batch_indices[-1]})")

    # Split into multiple parallel batches
    for i in range(0, len(batch_indices), parallel_batch_size):
        parallel_indices = batch_indices[i:i+parallel_batch_size]
        pbar = tqdm(total=len(parallel_indices), desc=f"Batch {batch_num+1} parallel group {i//parallel_batch_size+1}/{(len(batch_indices)+parallel_batch_size-1)//parallel_batch_size}")

        # Prepare prompts for each parallel batch
        prompts_data = []
        for idx in parallel_indices:
            # Check if completion is still needed
            tags_indices = item_data.tags_indices[idx]
            missing_levels = [level for level in range(tags_indices.shape[0]) if tags_indices[level] == -1]

            if not missing_levels:
                stats["skipped"] += 1
                pbar.update(1)
                continue

            # Generate the prompt
            prompt = generate_optimized_prompt(new_data, idx, tag_pools)
            system_prompt = "You are an expert video tag classifier. Your task is to select the most appropriate category tags for a video based on its title and existing tags. Your response must strictly follow the required JSON format."

            prompts_data.append((idx, prompt, system_prompt, missing_levels))

        # If there are no prompts to process, continue to the next batch
        if not prompts_data:
            continue

        # Extract the list of prompts for parallel processing
        indices = [data[0] for data in prompts_data]
        prompts = [(data[1], data[2]) for data in prompts_data]
        missing_levels_map = {data[0]: data[3] for data in prompts_data}

        # Directly assign a different model to each request
        from data.chat_with_llm import AVAILABLE_MODELS
        models = [AVAILABLE_MODELS[j % len(AVAILABLE_MODELS)] for j in range(len(prompts))]

        # Create parallel tasks
        def process_prompt(idx, prompt_tuple, model_name):
            prompt, system_prompt = prompt_tuple
            try:
                # Using specified model
                # print(f"Item index {indices[idx]} is using model: {model_name}")

                # Call the LLM
                result = query_llm(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model_name,
                    max_tokens=800,
                    temperature=0.3,
                    max_retries=3
                )

                # Parse the response
                parsed_result = parse_llm_response(result, expected_format="json")
                return idx, result, parsed_result
            except Exception as e:
                print(f"Error processing item index {indices[idx]}: {str(e)}")
                return idx, {"success": False, "error": str(e)}, {"success": False, "error": str(e)}

        # Execute tasks in parallel
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_prompt, i, prompt_tuple, models[i])
                       for i, prompt_tuple in enumerate(prompts)]

            for future in as_completed(futures):
                results.append(future.result())

        # Process results
        for idx_in_batch, result, parsed_result in sorted(results, key=lambda x: x[0]):
            item_idx = indices[idx_in_batch]
            stats["processed"] += 1

            # Build the complete result
            completion_result = {
                "status": "success" if parsed_result.get("success", False) else "error",
                "raw_response": result,
                "parsed_response": parsed_result,
                "completed_tags": {}
            }

            if completion_result["status"] == "success":
                # Extract completed tags
                try:
                    parsed_data = parsed_result.get("parsed_data", {})
                    completion_result["reasoning"] = parsed_data.get("reasoning", "No reasoning provided")

                    # Extract completed tags for each level
                    completed_tags_dict = parsed_data.get("completed_tags", {})

                    # Handle various possible key formats: 'level_X', 'X', integer X
                    for level in missing_levels_map[item_idx]:
                        # Try various possible key formats
                        level_keys = [f"level_{level}", str(level), level]
                        found_key = None

                        # Check each possible key format
                        for key in level_keys:
                            if str(key) in completed_tags_dict:
                                found_key = str(key)
                                break

                        if found_key is not None:
                            tag_name = completed_tags_dict[found_key]
                            # If the value is a dict, try to get the 'name' field
                            if isinstance(tag_name, dict) and "name" in tag_name:
                                tag_name = tag_name["name"]

                            # Find the corresponding ID and embedding from the tag pool
                            tag_id, tag_embedding = find_tag_id_by_name(tag_pools, level, tag_name)

                            if tag_id is not None:
                                completion_result["completed_tags"][level] = {
                                    "id": tag_id,
                                    "name": tag_name,
                                    "embedding": tag_embedding
                                }
                            else:
                                # If the corresponding tag is not found in the tag pool
                                completion_result["completed_tags"][level] = {
                                    "id": None,
                                    "name": tag_name,
                                    "embedding": None,
                                    "error": "Could not find the corresponding tag in the tag pool"
                                }

                    # If at least one tag was effectively completed
                    if completion_result["completed_tags"]:
                        # Update tags
                        for level, tag_info in completion_result["completed_tags"].items():
                            level = int(level)
                            if tag_info["id"] is not None:
                                # Update tag text
                                item_data.tags[item_idx, level] = tag_info["name"]
                                # Update tag indices
                                item_data.tags_indices[item_idx, level] = tag_info["id"]
                                # Update tag embedding (if available)
                                if tag_info["embedding"] is not None and hasattr(item_data, f'tags_emb_l{level+1}'):
                                    getattr(item_data, f'tags_emb_l{level+1}')[item_idx] = tag_info["embedding"]

                        stats["successful"] += 1
                    else:
                        stats["failed"] += 1
                        stats["errors"].append((item_idx, "Could not find any matching tags in the tag pool"))
                except Exception as e:
                    stats["failed"] += 1
                    error_msg = f"Failed to parse LLM response: {str(e)}"
                    stats["errors"].append((item_idx, error_msg))
            else:
                stats["failed"] += 1
                error_msg = completion_result.get("message", result.get("error", "Unknown error"))
                stats["errors"].append((item_idx, error_msg))

            # Update progress bar
            pbar.update(1)

        pbar.close()

        # Save temporary data after each parallel batch
        if save_path:
            temp_save_path = f"{save_path}_temp"
            torch.save(new_data, temp_save_path)
            print(f"\nTemporary data saved to {temp_save_path}")

            # Print model usage statistics
            model_stats = get_model_stats()
            print("\nModel Usage Statistics:")
            for model, stat in sorted(model_stats.items(), key=lambda x: x[1]['usage'], reverse=True):
                if stat['usage'] > 0:
                    print(f"{model}: Usage={stat['usage']}, Errors={stat['errors']}, Error Rate={stat['error_rate']}")

    # Save data after each major batch
    if save_path:
        torch.save(new_data, save_path)
        print(f"\nData for batch {batch_num+1} saved to {save_path}")

# Save final data
if save_path:
    torch.save(new_data, save_path)
    print(f"\nFinal data saved to {save_path}")

# Print statistics
print("\n===== Processing Statistics =====")
print(f"Total items processed: {stats['processed']}")
print(f"Successfully completed: {stats['successful']}")
print(f"Failed: {stats['failed']}")
print(f"Skipped (tags already complete): {stats['skipped']}")

if stats["errors"]:
    print(f"\nFirst 10 errors:")
    for idx, error in stats["errors"][:10]:
        print(f"  - Item Index {idx}: {error}")

return new_data


def check\_completion\_progress(data, save\_path=None):
"""
Checks the tag completion progress of the dataset and finds the index
of the first item with incomplete tags.


Args:
    data: A HeteroData object.
    save_path: Path to save the check results (optional).

Returns:
    The index of the first item with incomplete tags.
"""
item_data = data['item']
num_items = item_data.x.shape[0]

print(f"\n===== Tag Completion Progress Check =====")
print(f"Total items: {num_items}")

# Find the index of the first item with incomplete tags
first_incomplete_idx = None
incomplete_count = 0

for i in range(num_items):
    tags_indices = item_data.tags_indices[i]
    if torch.any(tags_indices == -1):
        if first_incomplete_idx is None:
            first_incomplete_idx = i
        incomplete_count += 1

if first_incomplete_idx is not None:
    print(f"Found the first incomplete item at index: {first_incomplete_idx}")
    print(f"Total incomplete items: {incomplete_count} ({incomplete_count/num_items*100:.2f}%)")
    print(f"Completed up to item index: {first_incomplete_idx} ({first_incomplete_idx/num_items*100:.2f}%)")

    # Print samples before and after the first incomplete item
    print("\n----- 10 Samples Before First Incomplete Item -----")
    start_idx = max(0, first_incomplete_idx - 10)
    for i in range(start_idx, first_incomplete_idx):
        title = item_data.text[i] if hasattr(item_data, 'text') and item_data.text is not None else "Unknown Title"
        tags = item_data.tags[i]
        tags_indices = item_data.tags_indices[i]
        print(f"Index {i}: Title: {title[:30]}... | Tags: {tags.tolist()} | Tag Indices: {tags_indices.tolist()}")

    print("\n----- First Incomplete Item -----")
    title = item_data.text[first_incomplete_idx] if hasattr(item_data, 'text') and item_data.text is not None else "Unknown Title"
    tags = item_data.tags[first_incomplete_idx]
    tags_indices = item_data.tags_indices[first_incomplete_idx]
    print(f"Index {first_incomplete_idx}: Title: {title} | Tags: {tags.tolist()} | Tag Indices: {tags_indices.tolist()}")

    print("\n----- 10 Samples After First Incomplete Item -----")
    end_idx = min(num_items, first_incomplete_idx + 11)
    for i in range(first_incomplete_idx + 1, end_idx):
        title = item_data.text[i] if hasattr(item_data, 'text') and item_data.text is not None else "Unknown Title"
        tags = item_data.tags[i]
        tags_indices = item_data.tags_indices[i]
        print(f"Index {i}: Title: {title[:30]}... | Tags: {tags.tolist()} | Tag Indices: {tags_indices.tolist()}")
else:
    print("All item tags are complete!")

return first_incomplete_idx


def main():
"""Main function."""
\# Set default data path
default\_data\_path = os.path.join('dataset', 'kuairand', 'processed', 'kuairand\_data\_minimal\_interactions30000.pt')


# Get command-line arguments
import argparse
parser = argparse.ArgumentParser(description='KuaiRand Dataset Analysis and Tag Completion Tool')
parser.add_argument('--data_path', type=str, default=default_data_path,
                    help=f'Path to the KuaiRand dataset file (default: {default_data_path})')
parser.add_argument('--samples', type=int, default=2,
                    help='Number of random samples to display for each attribute (default: 2)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42)')
parser.add_argument('--generate_prompts', action='store_true',
                    help='Generate prompts for tag completion')
parser.add_argument('--prompt_samples', type=int, default=5,
                    help='Number of samples to generate prompts for (default: 5)')
parser.add_argument('--complete_tags', action='store_true',
                    help='Complete tags for a few samples using LLM')
parser.add_argument('--batch_complete', action='store_true',
                    help='Batch complete all missing tags sequentially')
parser.add_argument('--parallel_complete', action='store_true',
                    help='Batch complete all missing tags in parallel')
parser.add_argument('--batch_size', type=int, default=100,
                    help='Batch size for batch processing (default: 100)')
parser.add_argument('--parallel_batch_size', type=int, default=10,
                    help='Parallel batch size, how many items to process in one parallel batch (default: 10)')
parser.add_argument('--max_workers', type=int, default=8,
                    help='Maximum number of parallel workers (default: 8)')
parser.add_argument('--save_path', type=str, default=None,
                    help='Path to save the completed data')
parser.add_argument('--start_idx', type=int, default=0,
                    help='Starting item index for processing, for resuming (default: 0)')
parser.add_argument('--check_progress', action='store_true',
                    help='Check tag completion progress and find the first incomplete item index')

args = parser.parse_args()

# Load dataset
data = load_kuairand_dataset(args.data_path, args.seed)

if args.check_progress:
    # Check tag completion progress
    first_incomplete_idx = check_completion_progress(data)
    if first_incomplete_idx is not None:
        print(f"\nTo continue tag completion, use the following command:")
        print(f"python -m data.fill_kuairand --parallel_complete --data_path {args.data_path} --start_idx {first_incomplete_idx}")
elif args.parallel_complete:
    # Default save path
    if not args.save_path:
        save_dir = os.path.dirname(args.data_path)
        filename = os.path.basename(args.data_path)
        name, ext = os.path.splitext(filename)
        args.save_path = os.path.join(save_dir, f"{name}_completed{ext}")

    print(f"Will save the completed data using parallel processing to: {args.save_path}")

    # Create tag pools
    tag_pools = create_tag_pools(data)

    # Batch complete tags in parallel and save
    parallel_complete_tags(
        data,
        tag_pools,
        args.batch_size,
        args.save_path,
        args.start_idx,
        args.max_workers,
        args.parallel_batch_size
    )
elif args.batch_complete:
    # Default save path
    if not args.save_path:
        save_dir = os.path.dirname(args.data_path)
        filename = os.path.basename(args.data_path)
        name, ext = os.path.splitext(filename)
        args.save_path = os.path.join(save_dir, f"{name}_completed{ext}")

    print(f"Will save the completed data to: {args.save_path}")

    # Create tag pools
    tag_pools = create_tag_pools(data)

    # Batch complete tags and save
    batch_complete_tags(data, tag_pools, args.batch_size, args.save_path, args.start_idx)
elif args.complete_tags:
    # Create tag pools
    tag_pools = create_tag_pools(data)

    # Analyze and complete tags for samples
    analyze_and_complete_tags(data, tag_pools, args.prompt_samples, args.seed)
elif args.generate_prompts:
    # Create tag pools
    tag_pools = create_tag_pools(data)

    # Generate sample prompts
    generate_prompts_for_samples(data, tag_pools, args.prompt_samples, args.seed)
else:
    # Print dataset structure
    print_dataset_structure(data)

    # Randomly sample and print attribute values
    sample_and_print_values(data, args.samples)

    # Analyze tag completeness
    analyze_tag_completeness(data)

print("\nProcess finished!")

if __name__ == "__main__":
    main()
