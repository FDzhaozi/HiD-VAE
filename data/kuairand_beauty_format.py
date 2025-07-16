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
    处理KuaiRand数据集，使其格式与Amazon Beauty数据集一致，并使用留一法构建序列数据。
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
        low_memory: bool = False,  # 添加低内存模式选项
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
        self.low_memory = low_memory  # 低内存模式标志
        
        # 设置随机种子以确保可复现性
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
        
        # 检查是否有可用的CUDA设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"将使用设备: {self.device}")
        
        super(KuaiRandBeautyFormat, self).__init__(
            root, transform, pre_transform, force_reload
        )
        
        try:
            print(f"\n尝试加载数据: {self.processed_paths[0]}")
            self.load(self.processed_paths[0], data_cls=HeteroData)
            print("数据加载成功!")
        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            print("将重新处理数据...")
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
        """用于存储中间已处理文件的目录"""
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
        print(f"数据文件应手动放置在 '{self.raw_dir}' 目录下。")
    
    def _get_interim_path(self, basename: str) -> str:
        """获取中间文件的路径"""
        return os.path.join(self.interim_dir, basename)

    def _remap_ids(self, x):
        """与Amazon Beauty数据集保持一致的ID重映射函数"""
        return x - 1
    
    def _load_csv_in_chunks(self, file_path, usecols=None, dtype=None, chunk_size=100000, **kwargs):
        """分块加载大CSV文件以减少内存使用"""
        chunks = []
        for chunk in tqdm(pd.read_csv(file_path, usecols=usecols, dtype=dtype, chunksize=chunk_size, **kwargs),
                          desc=f"分块加载 {os.path.basename(file_path)}"):
            chunks.append(chunk)
            if len(chunks) % 5 == 0:
                gc.collect()
        return pd.concat(chunks, ignore_index=True)
    
    def _process_video_tags(self, videos_categories_df):
        """处理视频标签"""
        print("处理视频标签...")
        video_tags = defaultdict(lambda: {1: None, 2: None, 3: None})
        
        for _, row in tqdm(videos_categories_df.iterrows(), total=len(videos_categories_df), desc="处理视频标签"):
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
        """检查嵌入缓存文件是否存在且大小是否正确"""
        if not os.path.exists(cache_path):
            return False
        expected_size = num_embeddings * embedding_dim * np.dtype(np.float16).itemsize
        return os.path.getsize(cache_path) == expected_size
    
    def _load_embedding_safely(self, cache_path, num_embeddings, embedding_dim=768):
        """安全地加载嵌入文件"""
        if not self._is_embedding_cache_valid(cache_path, num_embeddings, embedding_dim):
            if os.path.exists(cache_path):
                print(f"警告: 检测到无效的缓存文件，将删除: {os.path.basename(cache_path)}")
                os.remove(cache_path)
            return None
        try:
            embedding_data = np.memmap(cache_path, dtype=np.float16, mode='r', shape=(num_embeddings, embedding_dim))
            return torch.from_numpy(embedding_data.copy())
        except Exception as e:
            print(f"警告: 无法加载嵌入文件 {os.path.basename(cache_path)}: {e}")
            if os.path.exists(cache_path):
                os.remove(cache_path)
            return None
    
    def _encode_to_memmap(self, texts, cache_path, model, batch_size=16):
        """将文本编码并直接保存到磁盘上的内存映射数组中"""
        embedding_dim = 768
        
        if self._is_embedding_cache_valid(cache_path, len(texts), embedding_dim) and not self.force_reload:
            print(f"检测到完整有效的嵌入缓存，跳过编码: {os.path.basename(cache_path)}")
            return
        
        print(f"开始编码并直接保存至: {os.path.basename(cache_path)}")
        
        tmp_cache_path = cache_path + ".tmp"
        if os.path.exists(tmp_cache_path):
            os.remove(tmp_cache_path)
            
        try:
            memmap_array = np.memmap(tmp_cache_path, dtype=np.float16, mode='w+', shape=(len(texts), embedding_dim))
            
            with torch.no_grad():
                for i in tqdm(range(0, len(texts), batch_size), desc="编码中", unit="batch"):
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
            print(f"嵌入已成功保存至: {os.path.basename(cache_path)}")

        except Exception as e:
            print(f"编码过程中发生错误: {e}")
            if os.path.exists(tmp_cache_path):
                os.remove(tmp_cache_path)
            raise
    
    def train_test_split(self, logs_df, video_info_df, max_seq_len=50):
        """使用留一法构建训练、验证和测试集，与Amazon Beauty数据集保持一致"""
        print("\n使用留一法构建序列数据...")
        
        # 视频ID已在process中被过滤，这里直接使用
        video_ids = video_info_df['video_id'].unique()
        all_video_id_map = {orig_id: self._remap_ids(new_id+1) for new_id, orig_id in enumerate(video_ids)}
        
        # 应用映射到日志数据
        logs_df['video_id'] = logs_df['video_id'].map(all_video_id_map)
        # 移除映射后为空的交互 (即视频不在高质量列表中)
        logs_df.dropna(subset=['video_id'], inplace=True)
        logs_df['video_id'] = logs_df['video_id'].astype(int)

        # 在当前实现中，video_id_map与all_video_id_map是相同的，因为筛选和采样已在之前完成
        video_id_map = all_video_id_map
        
        # 获取用户ID映射
        user_ids = logs_df['user_id'].unique()
        user_id_map = {orig_id: new_id for new_id, orig_id in enumerate(user_ids)}
        logs_df['user_id'] = logs_df['user_id'].map(user_id_map)
        
        # 按用户分组并按时间排序
        logs_df = logs_df.sort_values(['user_id', 'time_ms'])
        user_groups = logs_df.groupby('user_id')
        
        splits = ["train", "eval", "test"]
        sequences = {sp: defaultdict(list) for sp in splits}
        
        # 为每个用户构建序列
        for user_id, group in tqdm(user_groups, desc="构建用户序列"):
            items = group['video_id'].tolist()
            
            # 跳过序列太短的用户
            if len(items) < self.min_seq_len:
                continue
            
            if not self.sliding_window:
                # 标准留一法：每个用户只生成一个训练样本、一个验证样本和一个测试样本
                
                # 训练集：使用倒数第(max_seq_len+2)到倒数第3个商品作为历史，倒数第二个商品作为目标
                train_history = items[-(max_seq_len+2):-2]
                train_items = train_history + [-1] * (max_seq_len - len(train_history))
                sequences["train"]["userId"].append(user_id)
                sequences["train"]["itemId"].append(train_items)
                sequences["train"]["itemId_fut"].append(items[-2])
                
                # 验证集：与训练集相同，使用倒数第(max_seq_len+2)到倒数第3个商品作为历史，倒数第二个商品作为目标
                eval_items = items[-(max_seq_len+2):-2]
                eval_items_padded = eval_items + [-1] * (max_seq_len - len(eval_items))
                sequences["eval"]["userId"].append(user_id)
                sequences["eval"]["itemId"].append(eval_items_padded)
                sequences["eval"]["itemId_fut"].append(items[-2])
                
                # 测试集：使用倒数第(max_seq_len+1)到倒数第2个商品作为历史，最后一个商品作为目标
                test_items = items[-(max_seq_len+1):-1]
                sequences["test"]["userId"].append(user_id)
                sequences["test"]["itemId"].append(test_items + [-1] * (max_seq_len - len(test_items)))
                sequences["test"]["itemId_fut"].append(items[-1])
            
            else:
                # 滑动窗口法：为每个用户生成多个训练样本、验证样本和测试样本
                # 这样可以增加序列数量，同时保持留一法的特性
                
                # 确保序列足够长，可以生成至少一个样本
                if len(items) < max_seq_len + 3:
                    # 如果序列不够长，只生成一个样本
                    # 训练集
                    train_history = items[:-2]
                    train_items = train_history + [-1] * (max_seq_len - len(train_history))
                    sequences["train"]["userId"].append(user_id)
                    sequences["train"]["itemId"].append(train_items)
                    sequences["train"]["itemId_fut"].append(items[-2])
                    
                    # 验证集
                    eval_items = items[:-2]
                    eval_items_padded = eval_items + [-1] * (max_seq_len - len(eval_items))
                    sequences["eval"]["userId"].append(user_id)
                    sequences["eval"]["itemId"].append(eval_items_padded)
                    sequences["eval"]["itemId_fut"].append(items[-2])
                    
                    # 测试集
                    test_items = items[:-1]
                    sequences["test"]["userId"].append(user_id)
                    sequences["test"]["itemId"].append(test_items + [-1] * (max_seq_len - len(test_items)))
                    sequences["test"]["itemId_fut"].append(items[-1])
                    continue
                
                # 计算可以生成多少个样本
                num_windows = max(1, (len(items) - max_seq_len - 2) // self.window_step + 1)
                
                # 调试信息
                if user_id < 5:  # 只打印前5个用户的信息
                    print(f"用户 {user_id} 序列长度: {len(items)}, 将生成 {num_windows} 个样本窗口")
                
                # 生成多个训练和验证样本（使用相同的构建方式）
                for i in range(num_windows):
                    start_idx = i * self.window_step
                    end_idx = start_idx + max_seq_len
                    
                    # 确保不超过序列长度
                    if end_idx + 2 > len(items):
                        if user_id < 5:
                            print(f"  窗口 {i+1}/{num_windows} 超出序列长度，跳过")
                        break
                    
                    # 训练集：使用当前窗口的商品作为历史，窗口后的第一个商品作为目标
                    train_history = items[start_idx:end_idx]
                    train_items = train_history + [-1] * (max_seq_len - len(train_history))
                    sequences["train"]["userId"].append(user_id)
                    sequences["train"]["itemId"].append(train_items)
                    sequences["train"]["itemId_fut"].append(items[end_idx])
                    
                    # 验证集：与训练集相同
                    sequences["eval"]["userId"].append(user_id)
                    sequences["eval"]["itemId"].append(train_items)
                    sequences["eval"]["itemId_fut"].append(items[end_idx])
                    
                    # 测试集：使用当前窗口加上目标商品作为历史，窗口后的第二个商品作为目标
                    if end_idx + 2 <= len(items):
                        test_history = items[start_idx:end_idx+1]
                        test_items = test_history + [-1] * (max_seq_len - len(test_history))
                        sequences["test"]["userId"].append(user_id)
                        sequences["test"]["itemId"].append(test_items)
                        sequences["test"]["itemId_fut"].append(items[end_idx+1])
        
        # 打印序列数量信息
        print(f"\n滑动窗口生成序列情况:")
        for sp in splits:
            print(f"{sp}集序列数: {len(sequences[sp]['userId'])}")
        
        for sp in splits:
            sequences[sp]["userId"] = np.array(sequences[sp]["userId"])
            
            # 所有集合都保持列表格式，避免不规则数组的问题
            if sp == "train" or self.sliding_window:
                # 确保序列保持列表类型
                pass  # 不做转换，保持原始列表
            else:
                # 如果不是滑动窗口模式，验证集和测试集可以转为numpy数组
                try:
                    sequences[sp]["itemId"] = np.array(sequences[sp]["itemId"])
                except ValueError:
                    print(f"警告: {sp}集的序列长度不一致，保持列表格式")
                
            sequences[sp]["itemId_fut"] = np.array(sequences[sp]["itemId_fut"])
            sequences[sp] = pl.from_dict(sequences[sp])
        
        print(f"训练集序列数: {len(sequences['train'])}")
        print(f"验证集序列数: {len(sequences['eval'])}")
        print(f"测试集序列数: {len(sequences['test'])}")
        
        return sequences, video_id_map, all_video_id_map
    
    def process(self) -> None:
        print("\n=== 开始处理KuaiRand数据集 (Beauty格式) ===")
        
        def show_memory_usage():
            process = psutil.Process()
            print(f"内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
        show_memory_usage()
        data = HeteroData()

        # 步骤 1: 加载并基于用户活跃度筛选交互日志
        print("\n--- 步骤 1: 加载并筛选活跃用户的交互 ---")
        log_cols = ['user_id', 'video_id', 'time_ms', 'is_click']
        log_dtypes = {'user_id': 'int64', 'video_id': 'int64', 'time_ms': 'int64', 'is_click': 'int8'}
        log_files = ["log_standard_4_08_to_4_21_1k.csv", "log_standard_4_22_to_5_08_1k.csv", "log_random_4_22_to_5_08_1k.csv"]
        log_df_list = [self._load_csv_in_chunks(os.path.join(self.raw_dir, f), usecols=log_cols, dtype=log_dtypes) for f in log_files]
        logs_df = pd.concat(log_df_list, ignore_index=True)
        logs_df = logs_df[logs_df['is_click'] == 1]
        print(f"  - 加载了 {len(logs_df)} 条点击交互")

        # 识别活跃用户
        user_counts = logs_df['user_id'].value_counts()
        active_users = user_counts[user_counts >= self.min_user_interactions].index
        print(f"  - 发现 {len(active_users)} 名活跃用户 (交互次数 >= {self.min_user_interactions})")

        # 根据 max_users 筛选
        if self.max_users and len(active_users) > self.max_users:
            selected_users = np.random.choice(active_users, self.max_users, replace=False)
            print(f"  - 已根据 max_users={self.max_users} 随机选择 {len(selected_users)} 名用户")
        else:
            selected_users = active_users.tolist()
        
        logs_df = logs_df[logs_df['user_id'].isin(selected_users)]
        print(f"  - 筛选后保留 {len(selected_users)} 名用户的 {len(logs_df)} 条交互")

        # 从活跃用户的交互中获取视频池
        video_id_pool = set(logs_df['video_id'].unique())
        print(f"  - 这些活跃用户共交互了 {len(video_id_pool)} 个独立视频")
        
        # 步骤 2: 从视频池中筛选高质量视频
        print("\n--- 步骤 2: 从视频池中筛选高质量视频 ---")
        # 加载所有相关的视频元数据
        basic_cols = ['video_id', 'video_duration']
        videos_basic_df = self._load_csv_in_chunks(os.path.join(self.raw_dir, "video_features_basic_1k.csv"), usecols=basic_cols, dtype={'video_id': 'int64'})
        caption_cols = ['final_video_id', 'caption']
        videos_captions_df = self._load_csv_in_chunks(os.path.join(self.raw_dir, "kuairand_video_captions.csv"), usecols=caption_cols, dtype={'final_video_id': 'int64', 'caption': 'str'}, encoding='utf-8').rename(columns={'final_video_id': 'video_id'})
        category_cols = ['final_video_id', 'first_level_category_name', 'second_level_category_name', 'third_level_category_name']
        videos_categories_df = self._load_csv_in_chunks(os.path.join(self.raw_dir, "kuairand_video_categories.csv"), usecols=category_cols, dtype={'final_video_id': 'int64'}, encoding='utf-8').rename(columns={'final_video_id': 'video_id'})

        # 合并并筛选
        video_info_df = pd.merge(videos_basic_df, videos_captions_df, on='video_id', how='left')
        video_info_df = pd.merge(video_info_df, videos_categories_df, on='video_id', how='left')
        print(f"  - 已加载全部视频的元数据，准备筛选...")
        
        # 首先只保留在活跃用户交互池中的视频
        video_info_df = video_info_df[video_info_df['video_id'].isin(video_id_pool)].copy()
        print(f"  - 筛选用户交互过的视频后剩余: {len(video_info_df)}")

        # 进行质量筛选
        video_info_df['caption'] = video_info_df['caption'].fillna('')
        video_info_df = video_info_df[video_info_df['caption'].str.strip() != ''].copy()
        print(f"  - 移除空标题后剩余: {len(video_info_df)}")
        
        for l in range(1, 4):
            level_map = {1: 'first', 2: 'second', 3: 'third'}
            col_name = f'{level_map[l]}_level_category_name'
            video_info_df[f'level_{l}_tag_text'] = video_info_df[col_name].fillna('').astype(str)

        def count_tags(row):
            return sum(1 for l in range(1, 4) if row[f'level_{l}_tag_text'] and row[f'level_{l}_tag_text'] != 'UNKNOWN')
        
        video_info_df['num_tags'] = video_info_df.apply(count_tags, axis=1)
        video_info_df = video_info_df[video_info_df['num_tags'] >= 2]
        print(f"  - 移除少于2个标签的视频后剩余: {len(video_info_df)}")

        # 步骤 3: (可选) 应用 max_videos 进行分层采样
        if self.max_videos and len(video_info_df) > self.max_videos:
            print(f"\n--- 步骤 3: 对高质量视频进行分层采样 (max_videos={self.max_videos}) ---")
            video_info_df = video_info_df.groupby('level_1_tag_text', group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(1, int(self.max_videos * len(x) / len(video_info_df)))), random_state=self.random_seed)
            ).reset_index(drop=True)
            print(f"  - 分层采样后最终视频数量: {len(video_info_df)}")

        # 步骤 4: 最后一次过滤日志，确保数据完全一致
        print("\n--- 步骤 4: 构建最终序列 ---")
        final_video_ids = set(video_info_df['video_id'].unique())
        logs_df = logs_df[logs_df['video_id'].isin(final_video_ids)]
        
        # 再次过滤用户，确保他们在最终的视频集中仍有足够长的序列
        user_counts = logs_df['user_id'].value_counts()
        final_users = user_counts[user_counts >= self.min_seq_len].index
        logs_df = logs_df[logs_df['user_id'].isin(final_users)]
        
        print(f"  - 最终用于构建序列的用户数: {len(final_users)}")
        print(f"  - 最终交互数: {len(logs_df)}")
        print(f"  - 最终视频数: {len(final_video_ids)}")

        # 步骤 5: 构建用户序列
        sequences, video_id_map, _ = self.train_test_split(
            logs_df, video_info_df, max_seq_len=self.max_seq_len
        )
        del logs_df; gc.collect()

        # 将序列数据添加到HeteroData对象
        data["user", "rated", "item"].history = {
            k: self._df_to_tensor_dict(v, ["itemId"])
            for k, v in sequences.items() 
        }
        
        # 使用映射后的ID更新 video_info
        video_info_df['item_id'] = video_info_df['video_id'].map(video_id_map)
        video_info_df.dropna(subset=['item_id'], inplace=True)
        video_info_df['item_id'] = video_info_df['item_id'].astype(int)
        video_info_df = video_info_df.sort_values('item_id').reset_index(drop=True)
        video_info = video_info_df
        print(f"\n最终用于构建特征的视频数量: {len(video_info)}")

        # 4. 处理视频标签和特征 (基于已筛选的 video_info)
        if self.encode_features:
            print("\n处理视频分类标签...")
            
            # 标签已在video_info中，直接使用
            for l in range(1, 4):
                video_info[f'level_{l}_tag_text'] = video_info[f'level_{l}_tag_text'].replace('UNKNOWN', '')

            # 创建标签索引表
            tag_to_idx_per_level = []
            for l in range(1, 4):
                unique_tags = video_info[f'level_{l}_tag_text'].unique()
                unique_tags = sorted([t for t in unique_tags if t != ""])
                tag_to_idx_per_level.append({tag: i for i, tag in enumerate(unique_tags)})
            
            # 为视频生成标签索引
            tags_indices = []
            for _, row in video_info.iterrows():
                tag_idx_1 = tag_to_idx_per_level[0].get(row['level_1_tag_text'], -1)
                tag_idx_2 = tag_to_idx_per_level[1].get(row['level_2_tag_text'], -1)
                tag_idx_3 = tag_to_idx_per_level[2].get(row['level_3_tag_text'], -1)
                tags_indices.append([tag_idx_1, tag_idx_2, tag_idx_3])
            
            video_info['tags_indices'] = tags_indices
            
            # 保存标签索引表
            tag_index_dict = {
                'tag_to_idx': tag_to_idx_per_level,
                'tag_class_counts': [len(m) for m in tag_to_idx_per_level]
            }
            
            tag_index_path = os.path.join(self.processed_dir, 'kuairand_tag_index.pt')
            torch.save(tag_index_dict, tag_index_path)
            print(f"标签索引表已保存至: {tag_index_path}")
            
            # 记录标签类别数量
            tag_class_counts = [len(m) for m in tag_to_idx_per_level]
            print(f"标签类别数量: {tag_class_counts}")
        
        # 5. 构建文本描述
        print("\n构建视频文本描述...")
        sentences = video_info['caption'].tolist()
        print(f"将保存所有{len(sentences)}个视频的文本信息")
        print(f"文本描述示例: {sentences[0] if sentences else 'N/A'}")
        
        # 6. 编码文本特征
        if self.encode_features:
            print("\n开始文本编码...")
            show_memory_usage()
            
            # 加载BGE模型
            bge_model = FlagModel(
                self.bge_model_name, 
                query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：", 
                use_fp16=True if self.device=='cuda' else False
            )
            
            # 编码视频标题 - 使用分批处理减少内存使用
            text_feat_path = self._get_interim_path("kuairand_text_features.mmp")
            
            # 检查缓存是否有效
            if not self._is_embedding_cache_valid(text_feat_path, len(video_info)) or self.force_reload:
                print(f"    - 分批处理视频标题编码，共{len(video_info)}个标题...")
                
                # 创建临时文件
                tmp_cache_path = text_feat_path + ".tmp"
                if os.path.exists(tmp_cache_path):
                    os.remove(tmp_cache_path)
                
                # 创建内存映射数组
                memmap_array = np.memmap(tmp_cache_path, dtype=np.float16, mode='w+', shape=(len(video_info), 768))
                
                # 分批处理
                batch_size = 16
                with torch.no_grad():
                    for i in tqdm(range(0, len(video_info), batch_size), desc="编码标题中", unit="batch"):
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
                
                # 成功后重命名
                if os.path.exists(text_feat_path):
                    os.remove(text_feat_path)
                os.rename(tmp_cache_path, text_feat_path)
            
            # 加载编码后的特征
            item_emb = self._load_embedding_safely(text_feat_path, len(video_info))
            
            # 编码标签
            if hasattr(video_info, 'level_1_tag_text'):
                tags_embs = []
                for i in range(1, 4):
                    tag_texts = video_info[f'level_{i}_tag_text'].tolist()
                    tag_feat_path = self._get_interim_path(f"kuairand_tag_{i}_features.mmp")
                    self._encode_to_memmap(tag_texts, tag_feat_path, bge_model, batch_size=16)
                    tag_emb = self._load_embedding_safely(tag_feat_path, len(tag_texts))
                    tags_embs.append(tag_emb)
                
                # 合并标签嵌入
                tags_emb_tensor = torch.stack(tags_embs, dim=1)
                data['item'].tags_emb = tags_emb_tensor
                
                # 当设置了max_videos时，确保保存所有标签
                if self.low_memory and self.max_videos is None:
                    # 只在未指定视频数量且启用低内存模式时使用样本
                    print("    - 低内存模式：只保存100个标签样本")
                    data['item'].tags_samples = video_info[[f'level_{l}_tag_text' for l in range(1, 4)]].iloc[:100].values
                else:
                    # 保存所有视频的标签
                    print(f"    - 保存所有{len(video_info)}个视频的标签信息")
                    data['item'].tags = np.array(video_info[[f'level_{l}_tag_text' for l in range(1, 4)]].values)
                
                data['item'].tags_indices = torch.tensor(video_info['tags_indices'].tolist(), dtype=torch.long)
            
            # 释放模型
            del bge_model
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            # 如果不编码，创建占位符嵌入
            print("\n跳过文本编码，创建占位符嵌入...")
            item_emb = torch.zeros((len(video_info), 768), dtype=torch.float32)
        
        # 7. 保存数据
        data['item'].x = item_emb
        
        # 总是保存所有文本和标签，因为我们已经对视频进行了采样
        print(f"    - 保存所有{len(sentences)}个视频的文本信息")
        data['item'].text = sentences
        data['item'].text_available = True
        
        if self.encode_features:
            print(f"    - 保存所有{len(video_info)}个视频的标签信息")
            data['item'].tags = np.array(video_info[[f'level_{l}_tag_text' for l in range(1, 4)]].values)
            data['item'].tags_indices = torch.tensor(video_info['tags_indices'].tolist(), dtype=torch.long)

        # 8. 划分训练集和测试集
        print("\n划分训练集和测试集...")
        gen = torch.Generator()
        gen.manual_seed(42)
        data['item'].is_train = torch.rand(item_emb.shape[0], generator=gen) > 0.05
        print(f"训练集商品数: {data['item'].is_train.sum().item()}")
        print(f"测试集商品数: {(~data['item'].is_train).sum().item()}")
        
        # 9. 保存处理后的数据
        print("\n保存处理后的数据...")
        self.save([data], self.processed_paths[0])
        print("=== 数据处理完成 ===\n")


if __name__ == "__main__":
    # 示例用法
    dataset = KuaiRandBeautyFormat(
        root="dataset/kuairand",
        force_reload=True,
        encode_features=True,  # 设置为True则进行文本和标签编码
        max_users=30000,         # 调整用户池以微调序列数
        max_videos=30000,        # 将物品数量控制在3万左右
        max_seq_len=20,         # 设置最大序列长度
        random_seed=42,
        min_seq_len=5,           # 保持最小序列长度要求
        min_user_interactions=25, # 略微提高用户活跃度门槛
        sliding_window=True,    # 使用滑动窗口生成更多序列
        window_step=4,          # **重要**：调整步长以微调序列数
        low_memory=False,       # 由于进行了大量筛选，建议关闭低内存模式以保存所有信息
    )
    
    # 查看数据集信息
    print(f"\n数据集信息:")
    print(f"项目数量: {dataset.data['item'].x.shape[0]}")
    print(f"训练序列数: {len(dataset.data['user', 'rated', 'item'].history['train']['userId'])}")
    print(f"验证序列数: {len(dataset.data['user', 'rated', 'item'].history['eval']['userId'])}")
    print(f"测试序列数: {len(dataset.data['user', 'rated', 'item'].history['test']['userId'])}")