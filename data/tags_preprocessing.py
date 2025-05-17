import numpy as np
import pandas as pd
import polars as pl
import torch
from data.schemas import FUT_SUFFIX
from einops import rearrange
from sentence_transformers import SentenceTransformer
from typing import List



# 1. 数据预处理
# ​**_process_genres**：
#
# 处理电影类型（genres）数据。
# 如果 one_hot=True，直接返回原始数据；否则，将电影类型转换为索引列表。
# ​**_remove_low_occurrence**：
#
# 移除出现次数较少的用户或电影。
# 通过 groupby 统计每个用户或电影的评分次数，并过滤掉评分次数少于 5 的记录。
# ​**_encode_text_feature**：
#
# 使用 SentenceTransformer 对文本特征进行编码。
# 默认使用 sentence-t5-xl 模型。

# ​2. 滑动窗口生成用户历史记录
# ​**_rolling_window**：
# 对每个用户的评分记录应用滑动窗口。
# 使用 np.lib.stride_tricks.as_strided 实现高效的滑动窗口计算。
# 将窗口内的特征数据转换为 torch.Tensor。

# ​3. 数据集划分
# ​**_ordered_train_test_split**：
# 根据时间戳将数据集划分为训练集和测试集。
# 使用分位数（如 train_split=0.8）确定划分阈值。

# ​4. 数据格式转换
# ​**_df_to_tensor_dict**：
# 将 polars.DataFrame 转换为 torch.Tensor 字典。
# 处理特征数据和未来特征（如 movieId_fut、rating_fut）。

# ​5. 生成用户历史记录
# ​**_generate_user_history**：
# 主方法，整合上述步骤，生成用户历史记录。
# 具体流程：
# 按用户分组，并按时间戳排序。
# 使用滑动窗口生成用户历史记录。
# 填充不足窗口大小的记录。
# 划分训练集和测试集。
# 将数据转换为 torch.Tensor 字典。


class PreprocessingMixin:
    @staticmethod
    def _process_genres(genres, one_hot=True):
        if one_hot:
            return genres

        max_genres = genres.sum(axis=1).max()
        idx_list = []
        for i in range(genres.shape[0]):
            idxs = np.where(genres[i, :] == 1)[0] + 1
            missing = max_genres - len(idxs)
            if missing > 0:
                idxs = np.array(list(idxs) + missing * [0])
            idx_list.append(idxs)
        out = np.stack(idx_list)
        return out

    @staticmethod
    def _remove_low_occurrence(source_df, target_df, index_col):
        if isinstance(index_col, str):
            index_col = [index_col]
        out = target_df.copy()
        for col in index_col:
            count = source_df.groupby(col).agg(ratingCnt=("rating", "count"))
            high_occ = count[count["ratingCnt"] >= 5]
            out = out.merge(high_occ, on=col).drop(columns=["ratingCnt"])
        return out

    @staticmethod
    def _encode_text_feature(text_feat, model=None):
        if model is None:
            model = SentenceTransformer('sentence-transformers/sentence-t5-xl')
        embeddings = model.encode(sentences=text_feat, show_progress_bar=True, convert_to_tensor=True).cpu()
        return embeddings
    
    @staticmethod
    def _rolling_window(group, features, window_size=200, stride=1):
        assert group["userId"].nunique() == 1, "Found data for too many users"
        
        if len(group) < window_size:
            window_size = len(group)
            stride = 1
        n_windows = (len(group)+1-window_size)//stride
        feats = group[features].to_numpy().T
        windows = np.lib.stride_tricks.as_strided(
            feats,
            shape=(len(features), n_windows, window_size),
            strides=(feats.strides[0], 8*stride, 8*1)
        )
        feat_seqs = np.split(windows, len(features), axis=0)
        rolling_df = pd.DataFrame({
            name: pd.Series(
                np.split(feat_seqs[i].squeeze(0), n_windows, 0)
            ).map(torch.tensor) for i, name in enumerate(features)
        })
        return rolling_df
    
    @staticmethod
    def _ordered_train_test_split(df, on, train_split=0.8):
        threshold = df.select(pl.quantile(on, train_split)).item()
        return df.with_columns(is_train=pl.col(on) <= threshold)
    
    @staticmethod
    def _df_to_tensor_dict(df, features):
        out = {
            feat: torch.from_numpy(
                rearrange(
                    df.select(feat).to_numpy().squeeze().tolist(), "b d -> b d"
                )
            ) if df.select(pl.col(feat).list.len().max() == pl.col(feat).list.len().min()).item()
            else df.get_column("itemId").to_list()
            for feat in features
        }
        fut_out = {
            feat + FUT_SUFFIX: torch.from_numpy(
                df.select(feat + FUT_SUFFIX).to_numpy()
            ) for feat in features
        }
        out.update(fut_out)
        out["userId"] = torch.from_numpy(df.select("userId").to_numpy())
        
        # 如果数据中包含标签嵌入，也添加到输出中
        if "tags_emb" in df.columns:
            out["tags_emb"] = torch.from_numpy(df.select("tags_emb").to_numpy())
            out["tags_emb_fut"] = torch.from_numpy(df.select("tags_emb_fut").to_numpy())
        
        # 如果数据中包含标签索引，也添加到输出中
        if "tags_indices" in df.columns:
            out["tags_indices"] = torch.from_numpy(df.select("tags_indices").to_numpy())
            out["tags_indices_fut"] = torch.from_numpy(df.select("tags_indices_fut").to_numpy())
        
        return out


    @staticmethod
    def _generate_user_history(
        ratings_df,
        features: List[str] = ["movieId", "rating"],
        window_size: int = 200,
        stride: int = 1,
        train_split: float = 0.8,
    ) -> torch.Tensor:
        
        if isinstance(ratings_df, pd.DataFrame):
            ratings_df = pl.from_pandas(ratings_df)

        grouped_by_user = (ratings_df
            .sort("userId", "timestamp")
            .group_by_dynamic(
                index_column=pl.int_range(pl.len()),
                every=f"{stride}i",
                period=f"{window_size}i",
                by="userId")
            .agg(
                *(pl.col(feat) for feat in features),
                seq_len=pl.col(features[0]).len(),
                max_timestamp=pl.max("timestamp")
            )
        )
        
        max_seq_len = grouped_by_user.select(pl.col("seq_len").max()).item()
        split_grouped_by_user = PreprocessingMixin._ordered_train_test_split(grouped_by_user, "max_timestamp", 0.8)
        padded_history = (split_grouped_by_user
            .with_columns(pad_len=max_seq_len-pl.col("seq_len"))
            .filter(pl.col("is_train").or_(pl.col("seq_len") > 1))
            .select(
                pl.col("userId"),
                pl.col("max_timestamp"),
                pl.col("is_train"),
                *(pl.when(pl.col("is_train"))
                    .then(
                        pl.col(feat).list.concat(
                            pl.lit(-1, dtype=pl.Int64).repeat_by(pl.col("pad_len"))
                        ).list.to_array(max_seq_len)
                    ).otherwise(
                        pl.col(feat).list.slice(0, pl.col("seq_len")-1).list.concat(
                            pl.lit(-1, dtype=pl.Int64).repeat_by(pl.col("pad_len")+1)
                        ).list.to_array(max_seq_len)
                    )
                    for feat in features
                ),
                *(pl.when(pl.col("is_train"))
                    .then(
                        pl.lit(-1, dtype=pl.Int64)
                    )
                    .otherwise(
                        pl.col(feat).list.get(-1)
                    ).alias(feat + FUT_SUFFIX)
                    for feat in features
                )
            )
        )
        
        out = {}
        out["train"] = PreprocessingMixin._df_to_tensor_dict(
            padded_history.filter(pl.col("is_train")),
            features
        )
        out["eval"] = PreprocessingMixin._df_to_tensor_dict(
            padded_history.filter(pl.col("is_train").not_()),
            features
        )
        
        return out

