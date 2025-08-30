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
from tqdm import tqdm
from collections import defaultdict
from data.tags_preprocessing import PreprocessingMixin
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from typing import Callable, List, Optional
from FlagEmbedding import FlagModel


class KuaiRandBeautyFormat(InMemoryDataset, PreprocessingMixin):
    """
    Processes the KuaiRand dataset to match the format of the Amazon Beauty dataset
    and constructs sequential data using the leave-one-out method.
    """
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        encode_features: bool = False,
        max_users: int = None,
        max_videos: int = None,
        max_seq_len: int = 50,
        random_seed: int = 42,
        min_seq_len: int = 5,
        min_user_interactions: int = 30,
        sliding_window: bool = False,
        window_step: int = 1,
        low_memory: bool = False,  # Add low memory mode option
    ) -> None:
        self.encode_features = encode_features
        self.max_users = max_users
        self.max_videos = max_videos
        self.max_seq_len = max_seq_len
        self.random_seed = random_seed
        self.min_seq_len = min_seq_len
        self.min_user_interactions = min_user_interactions
        self.sliding_window = sliding_window
        self.window_step = window_step
        self.bge_model_name = 'BAAI/bge-base-zh-v1.5'
        self.force_reload = force_reload
        self.low_memory = low_memory  # Low memory mode flag
        
        # Set random seeds for reproducibility
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
        
        # Check for available CUDA device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        super(KuaiRandBeautyFormat, self).__init__(
            root, transform, pre_transform, force_reload
        )
        
        try:
            print(f"\nAttempting to load data: {self.processed_paths[0]}")
            self.load(self.processed_paths[0], data_cls=HeteroData)
            print("Data loaded successfully!")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            print("Reprocessing data...")
            self.process()
            self.load(self.processed_paths[0], data_cls=HeteroData)
    
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "raw", "data")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "processed")
    
    @property
    def interim_dir(self) -> str:
        """Directory to store intermediate processed files"""
        interim_dir = os.path.join(self.processed_dir, "interim")
        os.makedirs(interim_dir, exist_ok=True)
        return interim_dir

    @property
    def raw_file_names(self) -> list:
        return [
            "log_standard_4_08_to_4_21_1k.csv",
            "log_standard_4_22_to_5_08_1k.csv",
            "log_random_4_22_to_5_08_1k.csv",
            "video_features_basic_1k.csv",
            "kuairand_video_categories.csv",
            "kuairand_video_captions.csv"
        ]

    @property
    def processed_file_names(self) -> str:
        prefix = "kuairand_beauty_format"
        suffix = "_encoded" if self.encode_features else "_raw"
        
        extra_info = []
        if self.max_users is not None:
            extra_info.append(f"users{self.max_users}")
        if self.max_videos is not None:
            extra_info.append(f"videos{self.max_videos}")
        if self.random_seed != 42:
            extra_info.append(f"seed{self.random_seed}")
        if self.low_memory:
            extra_info.append("lowmem")
            
        if extra_info:
            return f'{prefix}_{suffix}_{"_".join(extra_info)}.pt'
        return f'{prefix}_{suffix}.pt'

    def download(self):
        print(f"Data files should be manually placed in the '{self.raw_dir}' directory.")
    
    def _get_interim_path(self, basename: str) -> str:
        """Gets the path for an intermediate file"""
        return os.path.join(self.interim_dir, basename)

    def _remap_ids(self, x):
        """ID remapping function to be consistent with the Amazon Beauty dataset"""
        return x - 1
    
    def _load_csv_in_chunks(self, file_path, usecols=None, dtype=None, chunk_size=100000, **kwargs):
        """Loads large CSV files in chunks to reduce memory usage"""
        chunks = []
        for chunk in tqdm(pd.read_csv(file_path, usecols=usecols, dtype=dtype, chunksize=chunk_size, **kwargs),
                          desc=f"Loading {os.path.basename(file_path)} in chunks"):
            chunks.append(chunk)
            if len(chunks) % 5 == 0:
                gc.collect()
        return pd.concat(chunks, ignore_index=True)
    
    def _process_video_tags(self, videos_categories_df):
        """Processes video tags"""
        print("Processing video tags...")
        video_tags = defaultdict(lambda: {1: None, 2: None, 3: None})
        
        for _, row in tqdm(videos_categories_df.iterrows(), total=len(videos_categories_df), desc="Processing video tags"):
            vid = row['final_video_id']
            for l_num, l_name in enumerate(['first', 'second', 'third'], 1):
                if video_tags[vid][l_num] is None and pd.notna(row[f'{l_name}_level_category_id']):
                    try:
                        tag_id = str(int(float(row[f'{l_name}_level_category_id'])))
                        tag_name = row[f'{l_name}_level_category_name']
                        if tag_name != 'UNKNOWN':
                            video_tags[vid][l_num] = (tag_name, tag_id)
                    except (ValueError, TypeError):
                        continue
        
        return video_tags
    
    def _is_embedding_cache_valid(self, cache_path, num_embeddings, embedding_dim=768):
        """Checks if the embedding cache file exists and has the correct size"""
        if not os.path.exists(cache_path):
            return False
        expected_size = num_embeddings * embedding_dim * np.dtype(np.float16).itemsize
        return os.path.getsize(cache_path) == expected_size
    
    def _load_embedding_safely(self, cache_path, num_embeddings, embedding_dim=768):
        """Safely loads the embedding file"""
        if not self._is_embedding_cache_valid(cache_path, num_embeddings, embedding_dim):
            if os.path.exists(cache_path):
                print(f"Warning: Invalid cache file detected, deleting: {os.path.basename(cache_path)}")
                os.remove(cache_path)
            return None
        try:
            embedding_data = np.memmap(cache_path, dtype=np.float16, mode='r', shape=(num_embeddings, embedding_dim))
            return torch.from_numpy(embedding_data.copy())
        except Exception as e:
            print(f"Warning: Could not load embedding file {os.path.basename(cache_path)}: {e}")
            if os.path.exists(cache_path):
                os.remove(cache_path)
            return None
    
    def _encode_to_memmap(self, texts, cache_path, model, batch_size=16):
        """Encodes texts and saves them directly to a memory-mapped array on disk"""
        embedding_dim = 768
        
        if self._is_embedding_cache_valid(cache_path, len(texts), embedding_dim) and not self.force_reload:
            print(f"Found complete and valid embedding cache, skipping encoding: {os.path.basename(cache_path)}")
            return
        
        print(f"Starting encoding and saving directly to: {os.path.basename(cache_path)}")
        
        tmp_cache_path = cache_path + ".tmp"
        if os.path.exists(tmp_cache_path):
            os.remove(tmp_cache_path)
            
        try:
            memmap_array = np.memmap(tmp_cache_path, dtype=np.float16, mode='w+', shape=(len(texts), embedding_dim))
            
            with torch.no_grad():
                for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", unit="batch"):
                    start_idx = i
                    end_idx = min(i + batch_size, len(texts))
                    batch_texts = texts[start_idx:end_idx]
                    
                    batch_embeddings = model.encode(batch_texts)
                    memmap_array[start_idx:end_idx, :] = batch_embeddings
                    
                    if i % (batch_size * 100) == 0:
                        memmap_array.flush()
                        gc.collect()

            memmap_array.flush()
            del memmap_array
            gc.collect()
            
            if os.path.exists(cache_path):
                os.remove(cache_path)
            os.rename(tmp_cache_path, cache_path)
            print(f"Embeddings successfully saved to: {os.path.basename(cache_path)}")

        except Exception as e:
            print(f"Error during encoding process: {e}")
            if os.path.exists(tmp_cache_path):
                os.remove(tmp_cache_path)
            raise
    
    def train_test_split(self, logs_df, video_info_df, max_seq_len=50):
        """Constructs training, validation, and test sets using the leave-one-out method, consistent with the Amazon Beauty dataset"""
        print("\nConstructing sequential data using leave-one-out method...")
        
        # Video IDs have already been filtered in process(), so we use them directly here
        video_ids = video_info_df['video_id'].unique()
        all_video_id_map = {orig_id: self._remap_ids(new_id+1) for new_id, orig_id in enumerate(video_ids)}
        
        # Apply mapping to the log data
        logs_df['video_id'] = logs_df['video_id'].map(all_video_id_map)
        # Remove interactions that are null after mapping (i.e., video not in the high-quality list)
        logs_df.dropna(subset=['video_id'], inplace=True)
        logs_df['video_id'] = logs_df['video_id'].astype(int)

        # In the current implementation, video_id_map is the same as all_video_id_map because filtering and sampling have already been done
        video_id_map = all_video_id_map
        
        # Get user ID mapping
        user_ids = logs_df['user_id'].unique()
        user_id_map = {orig_id: new_id for new_id, orig_id in enumerate(user_ids)}
        logs_df['user_id'] = logs_df['user_id'].map(user_id_map)
        
        # Group by user and sort by time
        logs_df = logs_df.sort_values(['user_id', 'time_ms'])
        user_groups = logs_df.groupby('user_id')
        
        splits = ["train", "eval", "test"]
        sequences = {sp: defaultdict(list) for sp in splits}
        
        # Construct sequences for each user
        for user_id, group in tqdm(user_groups, desc="Building user sequences"):
            items = group['video_id'].tolist()
            
            # Skip users with sequences that are too short
            if len(items) < self.min_seq_len:
                continue
            
            if not self.sliding_window:
                # Standard leave-one-out: generate only one training, one validation, and one test sample per user
                
                # Training set: use items from -(max_seq_len+2) to -3 as history, and the second to last item as the target
                train_history = items[-(max_seq_len+2):-2]
                train_items = train_history + [-1] * (max_seq_len - len(train_history))
                sequences["train"]["userId"].append(user_id)
                sequences["train"]["itemId"].append(train_items)
                sequences["train"]["itemId_fut"].append(items[-2])
                
                # Validation set: same as training set, use items from -(max_seq_len+2) to -3 as history, and the second to last item as the target
                eval_items = items[-(max_seq_len+2):-2]
                eval_items_padded = eval_items + [-1] * (max_seq_len - len(eval_items))
                sequences["eval"]["userId"].append(user_id)
                sequences["eval"]["itemId"].append(eval_items_padded)
                sequences["eval"]["itemId_fut"].append(items[-2])
                
                # Test set: use items from -(max_seq_len+1) to -2 as history, and the last item as the target
                test_items = items[-(max_seq_len+1):-1]
                sequences["test"]["userId"].append(user_id)
                sequences["test"]["itemId"].append(test_items + [-1] * (max_seq_len - len(test_items)))
                sequences["test"]["itemId_fut"].append(items[-1])
            
            else:
                # Sliding window method: generate multiple training, validation, and test samples for each user
                # This increases the number of sequences while maintaining the leave-one-out characteristic
                
                # Ensure the sequence is long enough to generate at least one sample
                if len(items) < max_seq_len + 3:
                    # If the sequence is not long enough, generate only one sample
                    # Training set
                    train_history = items[:-2]
                    train_items = train_history + [-1] * (max_seq_len - len(train_history))
                    sequences["train"]["userId"].append(user_id)
                    sequences["train"]["itemId"].append(train_items)
                    sequences["train"]["itemId_fut"].append(items[-2])
                    
                    # Validation set
                    eval_items = items[:-2]
                    eval_items_padded = eval_items + [-1] * (max_seq_len - len(eval_items))
                    sequences["eval"]["userId"].append(user_id)
                    sequences["eval"]["itemId"].append(eval_items_padded)
                    sequences["eval"]["itemId_fut"].append(items[-2])
                    
                    # Test set
                    test_items = items[:-1]
                    sequences["test"]["userId"].append(user_id)
                    sequences["test"]["itemId"].append(test_items + [-1] * (max_seq_len - len(test_items)))
                    sequences["test"]["itemId_fut"].append(items[-1])
                    continue
                
                # Calculate how many samples can be generated
                num_windows = max(1, (len(items) - max_seq_len - 2) // self.window_step + 1)
                
                # Debugging information
                if user_id < 5:  # Only print information for the first 5 users
                    print(f"User {user_id} sequence length: {len(items)}, will generate {num_windows} sample windows")
                
                # Generate multiple training and validation samples (using the same construction method)
                for i in range(num_windows):
                    start_idx = i * self.window_step
                    end_idx = start_idx + max_seq_len
                    
                    # Ensure we don't exceed the sequence length
                    if end_idx + 2 > len(items):
                        if user_id < 5:
                            print(f"  Window {i+1}/{num_windows} exceeds sequence length, skipping")
                        break
                    
                    # Training set: use the items in the current window as history, and the first item after the window as the target
                    train_history = items[start_idx:end_idx]
                    train_items = train_history + [-1] * (max_seq_len - len(train_history))
                    sequences["train"]["userId"].append(user_id)
                    sequences["train"]["itemId"].append(train_items)
                    sequences["train"]["itemId_fut"].append(items[end_idx])
                    
                    # Validation set: same as training set
                    sequences["eval"]["userId"].append(user_id)
                    sequences["eval"]["itemId"].append(train_items)
                    sequences["eval"]["itemId_fut"].append(items[end_idx])
                    
                    # Test set: use the current window plus the target item as history, and the second item after the window as the target
                    if end_idx + 2 <= len(items):
                        test_history = items[start_idx:end_idx+1]
                        test_items = test_history + [-1] * (max_seq_len - len(test_history))
                        sequences["test"]["userId"].append(user_id)
                        sequences["test"]["itemId"].append(test_items)
                        sequences["test"]["itemId_fut"].append(items[end_idx+1])
        
        # Print sequence count information
        print(f"\nSequence generation status with sliding window:")
        for sp in splits:
            print(f"{sp} set sequence count: {len(sequences[sp]['userId'])}")
        
        for sp in splits:
            sequences[sp]["userId"] = np.array(sequences[sp]["userId"])
            
            # Keep all sets in list format to avoid issues with ragged arrays
            if sp == "train" or self.sliding_window:
                # Ensure sequences remain as list type
                pass  # No conversion, keep as a list of lists
            else:
                # If not in sliding window mode, validation and test sets can be converted to numpy arrays
                try:
                    sequences[sp]["itemId"] = np.array(sequences[sp]["itemId"])
                except ValueError:
                    print(f"Warning: Sequence lengths in {sp} set are inconsistent, keeping as a list of lists")
            
            sequences[sp]["itemId_fut"] = np.array(sequences[sp]["itemId_fut"])
            sequences[sp] = pl.from_dict(sequences[sp])
        
        print(f"Training set sequence count: {len(sequences['train'])}")
        print(f"Validation set sequence count: {len(sequences['eval'])}")
        print(f"Test set sequence count: {len(sequences['test'])}")
        
        return sequences, video_id_map, all_video_id_map
    
    def process(self) -> None:
        print("\n=== Start processing KuaiRand dataset (Beauty format) ===")
        
        def show_memory_usage():
            process = psutil.Process()
            print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
        show_memory_usage()
        data = HeteroData()

        # Step 1: Load and filter interaction logs based on user activity
        print("\n--- Step 1: Load and filter interactions of active users ---")
        log_cols = ['user_id', 'video_id', 'time_ms', 'is_click']
        log_dtypes = {'user_id': 'int64', 'video_id': 'int64', 'time_ms': 'int64', 'is_click': 'int8'}
        log_files = ["log_standard_4_08_to_4_21_1k.csv", "log_standard_4_22_to_5_08_1k.csv", "log_random_4_22_to_5_08_1k.csv"]
        log_df_list = [self._load_csv_in_chunks(os.path.join(self.raw_dir, f), usecols=log_cols, dtype=log_dtypes) for f in log_files]
        logs_df = pd.concat(log_df_list, ignore_index=True)
        logs_df = logs_df[logs_df['is_click'] == 1]
        print(f"  - Loaded {len(logs_df)} click interactions")

        # Identify active users
        user_counts = logs_df['user_id'].value_counts()
        active_users = user_counts[user_counts >= self.min_user_interactions].index
        print(f"  - Found {len(active_users)} active users (interactions >= {self.min_user_interactions})")

        # Filter based on max_users
        if self.max_users and len(active_users) > self.max_users:
            selected_users = np.random.choice(active_users, self.max_users, replace=False)
            print(f"  - Randomly selected {len(selected_users)} users based on max_users={self.max_users}")
        else:
            selected_users = active_users.tolist()
        
        logs_df = logs_df[logs_df['user_id'].isin(selected_users)]
        print(f"  - After filtering, {len(logs_df)} interactions from {len(selected_users)} users remain")

        # Get the video pool from the interactions of active users
        video_id_pool = set(logs_df['video_id'].unique())
        print(f"  - These active users interacted with {len(video_id_pool)} unique videos")
        
        # Step 2: Filter high-quality videos from the video pool
        print("\n--- Step 2: Filter high-quality videos from the video pool ---")
        # Load all relevant video metadata
        basic_cols = ['video_id', 'video_duration']
        videos_basic_df = self._load_csv_in_chunks(os.path.join(self.raw_dir, "video_features_basic_1k.csv"), usecols=basic_cols, dtype={'video_id': 'int64'})
        caption_cols = ['final_video_id', 'caption']
        videos_captions_df = self._load_csv_in_chunks(os.path.join(self.raw_dir, "kuairand_video_captions.csv"), usecols=caption_cols, dtype={'final_video_id': 'int64', 'caption': 'str'}, encoding='utf-8').rename(columns={'final_video_id': 'video_id'})
        category_cols = ['final_video_id', 'first_level_category_name', 'second_level_category_name', 'third_level_category_name']
        videos_categories_df = self._load_csv_in_chunks(os.path.join(self.raw_dir, "kuairand_video_categories.csv"), usecols=category_cols, dtype={'final_video_id': 'int64'}, encoding='utf-8').rename(columns={'final_video_id': 'video_id'})

        # Merge and filter
        video_info_df = pd.merge(videos_basic_df, videos_captions_df, on='video_id', how='left')
        video_info_df = pd.merge(video_info_df, videos_categories_df, on='video_id', how='left')
        print(f"  - Loaded metadata for all videos, preparing to filter...")
        
        # First, only keep videos that are in the active user interaction pool
        video_info_df = video_info_df[video_info_df['video_id'].isin(video_id_pool)].copy()
        print(f"  - Remaining after filtering for user-interacted videos: {len(video_info_df)}")

        # Perform quality filtering
        video_info_df['caption'] = video_info_df['caption'].fillna('')
        video_info_df = video_info_df[video_info_df['caption'].str.strip() != ''].copy()
        print(f"  - Remaining after removing empty captions: {len(video_info_df)}")
        
        for l in range(1, 4):
            level_map = {1: 'first', 2: 'second', 3: 'third'}
            col_name = f'{level_map[l]}_level_category_name'
            video_info_df[f'level_{l}_tag_text'] = video_info_df[col_name].fillna('').astype(str)

        def count_tags(row):
            return sum(1 for l in range(1, 4) if row[f'level_{l}_tag_text'] and row[f'level_{l}_tag_text'] != 'UNKNOWN')
        
        video_info_df['num_tags'] = video_info_df.apply(count_tags, axis=1)
        video_info_df = video_info_df[video_info_df['num_tags'] >= 2]
        print(f"  - Remaining after removing videos with fewer than 2 tags: {len(video_info_df)}")

        # Step 3: (Optional) Apply max_videos for stratified sampling
        if self.max_videos and len(video_info_df) > self.max_videos:
            print(f"\n--- Step 3: Stratified sampling of high-quality videos (max_videos={self.max_videos}) ---")
            video_info_df = video_info_df.groupby('level_1_tag_text', group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(1, int(self.max_videos * len(x) / len(video_info_df)))), random_state=self.random_seed)
            ).reset_index(drop=True)
            print(f"  - Final number of videos after stratified sampling: {len(video_info_df)}")

        # Step 4: Final log filtering to ensure data consistency
        print("\n--- Step 4: Construct final sequences ---")
        final_video_ids = set(video_info_df['video_id'].unique())
        logs_df = logs_df[logs_df['video_id'].isin(final_video_ids)]
        
        # Filter users again to ensure they still have long enough sequences with the final video set
        user_counts = logs_df['user_id'].value_counts()
        final_users = user_counts[user_counts >= self.min_seq_len].index
        logs_df = logs_df[logs_df['user_id'].isin(final_users)]
        
        print(f"  - Final number of users for sequence construction: {len(final_users)}")
        print(f"  - Final number of interactions: {len(logs_df)}")
        print(f"  - Final number of videos: {len(final_video_ids)}")

        # Step 5: Construct user sequences
        sequences, video_id_map, _ = self.train_test_split(
            logs_df, video_info_df, max_seq_len=self.max_seq_len
        )
        del logs_df; gc.collect()

        # Add sequence data to the HeteroData object
        data["user", "rated", "item"].history = {
            k: self._df_to_tensor_dict(v, ["itemId"])
            for k, v in sequences.items() 
        }
        
        # Update video_info with mapped IDs
        video_info_df['item_id'] = video_info_df['video_id'].map(video_id_map)
        video_info_df.dropna(subset=['item_id'], inplace=True)
        video_info_df['item_id'] = video_info_df['item_id'].astype(int)
        video_info_df = video_info_df.sort_values('item_id').reset_index(drop=True)
        video_info = video_info_df
        print(f"\nFinal number of videos for feature construction: {len(video_info)}")

        # 4. Process video tags and features (based on the filtered video_info)
        if self.encode_features:
            print("\nProcessing video category tags...")
            
            # Tags are already in video_info, use them directly
            for l in range(1, 4):
                video_info[f'level_{l}_tag_text'] = video_info[f'level_{l}_tag_text'].replace('UNKNOWN', '')

            # Create tag index map
            tag_to_idx_per_level = []
            for l in range(1, 4):
                unique_tags = video_info[f'level_{l}_tag_text'].unique()
                unique_tags = sorted([t for t in unique_tags if t != ""])
                tag_to_idx_per_level.append({tag: i for i, tag in enumerate(unique_tags)})
            
            # Generate tag indices for videos
            tags_indices = []
            for _, row in video_info.iterrows():
                tag_idx_1 = tag_to_idx_per_level[0].get(row['level_1_tag_text'], -1)
                tag_idx_2 = tag_to_idx_per_level[1].get(row['level_2_tag_text'], -1)
                tag_idx_3 = tag_to_idx_per_level[2].get(row['level_3_tag_text'], -1)
                tags_indices.append([tag_idx_1, tag_idx_2, tag_idx_3])
            
            video_info['tags_indices'] = tags_indices
            
            # Save tag index map
            tag_index_dict = {
                'tag_to_idx': tag_to_idx_per_level,
                'tag_class_counts': [len(m) for m in tag_to_idx_per_level]
            }
            
            tag_index_path = os.path.join(self.processed_dir, 'kuairand_tag_index.pt')
            torch.save(tag_index_dict, tag_index_path)
            print(f"Tag index map saved to: {tag_index_path}")
            
            # Record the number of tag classes
            tag_class_counts = [len(m) for m in tag_to_idx_per_level]
            print(f"Number of tag classes: {tag_class_counts}")
    
        # 5. Construct text descriptions
        print("\nConstructing video text descriptions...")
        sentences = video_info['caption'].tolist()
        print(f"Will save text information for all {len(sentences)} videos")
        print(f"Example text description: {sentences[0] if sentences else 'N/A'}")
        
        # 6. Encode text features
        if self.encode_features:
            print("\nStarting text encoding...")
            show_memory_usage()
            
            # Load BGE model
            bge_model = FlagModel(
                self.bge_model_name, 
                query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：", 
                use_fp16=True if self.device=='cuda' else False
            )
            
            # Encode video captions - use batch processing to reduce memory usage
            text_feat_path = self._get_interim_path("kuairand_text_features.mmp")
            
            # Check if cache is valid
            if not self._is_embedding_cache_valid(text_feat_path, len(video_info)) or self.force_reload:
                print(f"  - Batch processing video caption encoding for {len(video_info)} captions...")
                
                # Create temporary file
                tmp_cache_path = text_feat_path + ".tmp"
                if os.path.exists(tmp_cache_path):
                    os.remove(tmp_cache_path)
                
                # Create memory-mapped array
                memmap_array = np.memmap(tmp_cache_path, dtype=np.float16, mode='w+', shape=(len(video_info), 768))
                
                # Batch processing
                batch_size = 16
                with torch.no_grad():
                    for i in tqdm(range(0, len(video_info), batch_size), desc="Encoding captions", unit="batch"):
                        start_idx = i
                        end_idx = min(i + batch_size, len(video_info))
                        batch_texts = video_info['caption'].iloc[start_idx:end_idx].tolist()
                        
                        batch_embeddings = bge_model.encode(batch_texts)
                        memmap_array[start_idx:end_idx, :] = batch_embeddings
                        
                        if i % (batch_size * 100) == 0:
                            memmap_array.flush()
                            gc.collect()
                
                memmap_array.flush()
                del memmap_array
                gc.collect()
                
                # Rename on success
                if os.path.exists(text_feat_path):
                    os.remove(text_feat_path)
                os.rename(tmp_cache_path, text_feat_path)
            
            # Load encoded features
            item_emb = self._load_embedding_safely(text_feat_path, len(video_info))
            
            # Encode tags
            if hasattr(video_info, 'level_1_tag_text'):
                tags_embs = []
                for i in range(1, 4):
                    tag_texts = video_info[f'level_{i}_tag_text'].tolist()
                    tag_feat_path = self._get_interim_path(f"kuairand_tag_{i}_features.mmp")
                    self._encode_to_memmap(tag_texts, tag_feat_path, bge_model, batch_size=16)
                    tag_emb = self._load_embedding_safely(tag_feat_path, len(tag_texts))
                    tags_embs.append(tag_emb)
                
                # Concatenate tag embeddings
                tags_emb_tensor = torch.stack(tags_embs, dim=1)
                data['item'].tags_emb = tags_emb_tensor
                
                # When max_videos is set, ensure all tags are saved
                if self.low_memory and self.max_videos is None:
                    # Only use samples when video count is not specified and low memory mode is enabled
                    print("  - Low memory mode: saving only 100 tag samples")
                    data['item'].tags_samples = video_info[[f'level_{l}_tag_text' for l in range(1, 4)]].iloc[:100].values
                else:
                    # Save tags for all videos
                    print(f"  - Saving tag information for all {len(video_info)} videos")
                    data['item'].tags = np.array(video_info[[f'level_{l}_tag_text' for l in range(1, 4)]].values)
                
                data['item'].tags_indices = torch.tensor(video_info['tags_indices'].tolist(), dtype=torch.long)
            
            # Release the model
            del bge_model
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            # If not encoding, create placeholder embeddings
            print("\nSkipping text encoding, creating placeholder embeddings...")
            item_emb = torch.zeros((len(video_info), 768), dtype=torch.float32)
        
        # 7. Save data
        data['item'].x = item_emb
        
        # Always save all texts and tags because we have already sampled the videos
        print(f"  - Saving text information for all {len(sentences)} videos")
        data['item'].text = sentences
        data['item'].text_available = True
        
        if self.encode_features:
            print(f"  - Saving tag information for all {len(video_info)} videos")
            data['item'].tags = np.array(video_info[[f'level_{l}_tag_text' for l in range(1, 4)]].values)
            data['item'].tags_indices = torch.tensor(video_info['tags_indices'].tolist(), dtype=torch.long)

        # 8. Split into training and test sets
        print("\nSplitting into training and test sets...")
        gen = torch.Generator()
        gen.manual_seed(42)
        data['item'].is_train = torch.rand(item_emb.shape[0], generator=gen) > 0.05
        print(f"Number of items in training set: {data['item'].is_train.sum().item()}")
        print(f"Number of items in test set: {(~data['item'].is_train).sum().item()}")
        
        # 9. Save the processed data
        print("\nSaving processed data...")
        self.save([data], self.processed_paths[0])
        print("=== Data processing complete ===\n")


if __name__ == "__main__":
    # Example usage
    dataset = KuaiRandBeautyFormat(
        root="dataset/kuairand",
        force_reload=True,
        encode_features=True,  # Set to True to perform text and tag encoding
        max_users=30000,       # Adjust user pool to fine-tune the number of sequences
        max_videos=30000,      # Control the number of items to be around 30,000
        max_seq_len=20,        # Set maximum sequence length
        random_seed=42,
        min_seq_len=5,         # Maintain minimum sequence length requirement
        min_user_interactions=25, # Slightly increase the user activity threshold
        sliding_window=True,   # Use sliding window to generate more sequences
        window_step=4,         # **Important**: Adjust the step size to fine-tune the number of sequences
        low_memory=False,      # Since extensive filtering is performed, it is recommended to disable low memory mode to save all information
    )
    
    # Check dataset information
    print(f"\nDataset Information:")
    print(f"Number of items: {dataset.data['item'].x.shape[0]}")
    print(f"Number of training sequences: {len(dataset.data['user', 'rated', 'item'].history['train']['userId'])}")
    print(f"Number of validation sequences: {len(dataset.data['user', 'rated', 'item'].history['eval']['userId'])}")
    print(f"Number of test sequences: {len(dataset.data['user', 'rated', 'item'].history['test']['userId'])}")
