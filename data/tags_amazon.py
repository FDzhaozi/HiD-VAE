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

# 添加对numpy._core.multiarray._reconstruct的支持
from torch.serialization import add_safe_globals
try:
    from numpy._core.multiarray import _reconstruct
    add_safe_globals([_reconstruct])
except ImportError:
    # 如果无法直接导入，可以尝试通过字符串注册
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

# 下载NLTK停用词
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
        self.force_reload = force_reload  # 保存force_reload参数
        
        print(f"\n初始化AmazonReviews数据集: split={split}, force_reload={force_reload}")
        
        # 如果强制重新加载，先删除已存在的处理文件
        if force_reload:
            processed_file_path = osp.join(osp.join(root, 'processed'), f'title_data_{split}_5tags.pt')
            if osp.exists(processed_file_path):
                print(f"强制重新加载: 删除已存在的处理文件 {processed_file_path}")
                os.remove(processed_file_path)
                print("文件已删除，将重新处理数据")
        
        super(AmazonReviews, self).__init__(
            root, transform, pre_transform, force_reload
        )
        
        # 修改torch_geometric的加载行为，确保使用weights_only=False
        # 保存原始函数
        original_torch_load = torch.load
        
        # 创建一个包装函数，强制使用weights_only=False
        def patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        # 临时替换torch.load
        torch.load = patched_torch_load
        
        try:
            # 尝试加载数据
            print("尝试加载数据，使用weights_only=False...")
            self.load(self.processed_paths[0], data_cls=HeteroData)
            print("数据加载成功!")
        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
        finally:
            # 恢复原始函数
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
            print(f"内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
        print("\n=== 开始处理Amazon数据集 ===")
        show_memory_usage()
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

        # 展平类别列表
        def flatten_categories(categories):
            """将嵌套的类别列表展平为单一列表"""
            flattened = []
            for cat in categories:
                if isinstance(cat, list):
                    flattened.extend(cat)
                else:
                    flattened.append(cat)
            return list(dict.fromkeys(flattened))  # 去重
        
        print("\n处理商品类别...")
        item_data['flat_categories'] = item_data['categories'].apply(flatten_categories)
        
        # 显示类别处理示例
        print("\n类别展平示例:")
        sample_idx = 0
        print(f"原始类别: {item_data.iloc[sample_idx]['categories']}")
        print(f"展平后: {item_data.iloc[sample_idx]['flat_categories']}")
        
        # 处理类别，确保每个商品有5个类别标签
        def process_categories_to_five_tags(row):
            import re
            import random
            
            # 获取展平后的类别
            cats = row['flat_categories']
            
            # 删除第一个标签（如果存在）
            if len(cats) > 0:
                cats = cats[1:]
            
            # 获取停用词
            stop_words = set(stopwords.words('english'))
            
            # 如果类别不足5个，从标题中提取单词补充
            if len(cats) < 5:
                # 从标题中提取单词
                title_words = re.findall(r'\b[A-Za-z]{3,}\b', str(row['title']))
                # 去除停用词、去除重复并排除已在类别中的单词
                title_words = [w for w in title_words if w.lower() not in stop_words and 
                              w.lower() not in [c.lower() for c in cats]]
                
                # 如果标题词不够，添加品牌
                if len(title_words) + len(cats) < 5 and row['brand'] != "Unknown":
                    if row['brand'].lower() not in [c.lower() for c in cats]:
                        title_words.append(row['brand'])
                
                # 随机选择足够的单词补充到5个
                random.seed(42 + row['id'])  # 确保结果可重现
                needed = 5 - len(cats)
                
                selected_words = []
                while len(selected_words) < needed:
                    if len(title_words) > 0:
                        word = random.choice(title_words)
                        title_words.remove(word)  # 避免重复选择
                        if word not in selected_words and word.strip() != "":
                            selected_words.append(word)
                    else:
                        # 如果单词不够，使用通用标签
                        tag_idx = len(selected_words) + 1
                        selected_words.append(f"GenericTag{tag_idx}")
                
                # 合并类别和选择的单词
                five_tags = cats + selected_words
            else:
                # 如果类别超过5个，保留前4个，将剩余的合并为第5个
                if len(cats) > 5:
                    five_tags = cats[:4] + [" ".join(cats[4:])]
                else:
                    five_tags = cats
            
            # 确保没有空标签
            five_tags = [tag if tag.strip() != "" else f"GenericTag{i+1}" for i, tag in enumerate(five_tags)]
            
            # 确保有5个标签
            while len(five_tags) < 5:
                five_tags.append(f"GenericTag{len(five_tags)+1}")
                
            return five_tags
        
        item_data['five_tags'] = item_data.apply(process_categories_to_five_tags, axis=1)
        
        # 显示处理后的5个标签示例
        print("\n处理后的5个标签示例:")
        for i in range(3):  # 显示3个示例
            print(f"商品 {i}:")
            print(f"  原始类别数: {len(item_data.iloc[i]['flat_categories'])}")
            print(f"  处理后的5个标签: {item_data.iloc[i]['five_tags']}")
        
        # 创建标签索引表
        print("\n创建标签索引表...")
        all_tags = []
        for tags in item_data['five_tags']:
            all_tags.extend(tags)
        
        # 创建唯一标签列表
        unique_tags = sorted(list(set(all_tags)))
        tag_to_idx = {tag: idx for idx, tag in enumerate(unique_tags)}
        idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
        
        print(f"唯一标签数量: {len(unique_tags)}")
        print(f"前10个标签示例: {unique_tags[:10]}")
        
        # 将标签转换为索引
        def tags_to_indices(tags_list):
            return [tag_to_idx[tag] for tag in tags_list]
        
        item_data['tags_indices'] = item_data['five_tags'].apply(tags_to_indices)
        
        print("\n标签索引示例:")
        for i in range(3):
            print(f"商品 {i}:")
            print(f"  标签: {item_data.iloc[i]['five_tags']}")
            print(f"  索引: {item_data.iloc[i]['tags_indices']}")
        
        # 统计每一层标签的唯一ID数量
        print("\n统计每一层标签的唯一ID数量...")
        for layer in range(5):  # 5个层级的标签
            # 提取当前层的所有标签
            layer_tags = item_data['five_tags'].apply(lambda x: x[layer] if layer < len(x) else None).dropna().tolist()
            unique_layer_tags = set(layer_tags)
            
            # 提取当前层的所有标签ID
            layer_ids = item_data['tags_indices'].apply(lambda x: x[layer] if layer < len(x) else None).dropna().tolist()
            unique_layer_ids = set(layer_ids)
            
            print(f"第{layer+1}层标签:")
            print(f"  唯一标签数量: {len(unique_layer_tags)}")
            print(f"  唯一标签ID数量: {len(unique_layer_ids)}")
            print(f"  标签总量: {len(layer_tags)}")  # 添加标签总量统计
            print(f"  前5个标签示例: {list(unique_layer_tags)[:5]}")
            
            # 如果标签数量较多，显示分布情况
            if len(unique_layer_tags) > 10:
                from collections import Counter
                tag_counts = Counter(layer_tags)
                most_common = tag_counts.most_common(5)
                print(f"  出现最多的5个标签: {most_common}")
                
                # 添加标签分布统计
                total_items = len(layer_tags)
                top5_count = sum(count for _, count in most_common)
                print(f"  前5个标签覆盖率: {top5_count/total_items*100:.2f}%")
        
        print("\n构建商品文本描述...")
        sentences = item_data.apply(
            lambda row:
                "Title: " +
                str(row["title"]) + "; " +
                "Brand: " +
                str(row["brand"]) + "; " +
                "Price: " +
                str(row["price"]) + "; " ,
            axis=1
        ).tolist()  # 直接转换为列表

        print("\n文本描述示例:")
        print(sentences[0])
        
        print("\n开始文本编码...")
        show_memory_usage()
        item_emb = self._encode_text_feature_batched(sentences, batch_size=32)  # 减小批大小
        gc.collect()  # 手动触发垃圾回收
        torch.cuda.empty_cache()
        show_memory_usage()
        print(f"文本特征维度: {item_emb.shape}")
        print(f"编码示例(前5维): {item_emb[0,:5]}")
        
        # 对5个标签分别进行编码 - 优化内存使用
        print("\n开始标签编码...")
        tags_embs = []
        model = SentenceTransformer('sentence-transformers/sentence-t5-xl')  # 只加载一次模型
        
        for i in range(5):
            print(f"\n处理标签{i+1}...")
            tag_sentences = item_data['five_tags'].apply(lambda x: x[i] if i < len(x) else "").tolist()
            
            # 分批处理标签编码，每批处理后清理内存
            batch_size = 16  # 更小的批大小
            tag_emb = self._encode_text_feature_batched(tag_sentences, model=model, batch_size=batch_size)
            tags_embs.append(tag_emb)
            
            # 更积极地清理内存
            del tag_sentences
            gc.collect()
            torch.cuda.empty_cache()
            show_memory_usage()
        
        # 将5个标签的编码合并为一个张量
        tags_emb_tensor = torch.stack(tags_embs, dim=1)  # [n_items, 5, emb_dim]
        print(f"\n合并后的标签特征维度: {tags_emb_tensor.shape}")
        
        # 清理不再需要的变量
        del tags_embs, model
        gc.collect()
        torch.cuda.empty_cache()
        
        data['item'].x = item_emb
        data['item'].text = np.array(sentences)
        data['item'].tags_emb = tags_emb_tensor
        data['item'].tags = np.array(item_data['five_tags'].tolist())
        data['item'].tags_indices = torch.tensor(item_data['tags_indices'].tolist(), dtype=torch.long)
        
        # 保存标签索引表
        tag_index_dict = {
            'tag_to_idx': tag_to_idx,
            'idx_to_tag': idx_to_tag
        }
        
        # 获取处理后的文件路径，并在同一目录下保存标签索引表
        processed_dir = os.path.dirname(self.processed_paths[0])
        tag_index_path = os.path.join(processed_dir, f'tag_index_{self.split}.pt')
        torch.save(tag_index_dict, tag_index_path)
        print(f"\n标签索引表已保存至: {tag_index_path}")

        print("\n划分训练集和测试集...")
        gen = torch.Generator()
        gen.manual_seed(42)
        data['item'].is_train = torch.rand(item_emb.shape[0], generator=gen) > 0.05
        print(f"训练集商品数: {data['item'].is_train.sum().item()}")
        print(f"测试集商品数: {(~data['item'].is_train).sum().item()}")

        print("\n保存处理后的数据...")
        # 修改torch.save的行为，确保兼容性
        original_torch_save = torch.save
        def patched_torch_save(*args, **kwargs):
            kwargs['_use_new_zipfile_serialization'] = False
            return original_torch_save(*args, **kwargs)
        
        # 临时替换torch.save
        torch.save = patched_torch_save
        
        try:
            self.save([data], self.processed_paths[0])
            print("数据保存成功!")
        except Exception as e:
            print(f"保存数据时出错: {str(e)}")
        finally:
            # 恢复原始函数
            torch.save = original_torch_save
            
        print("=== 数据处理完成 ===\n")
        
    @staticmethod
    def _encode_text_feature_batched(text_feat, model=None, batch_size=32):
        """分批处理文本编码，优化内存使用"""
        if model is None:
            model = SentenceTransformer('sentence-transformers/sentence-t5-xl')
        
        total_samples = len(text_feat)
        embeddings_list = []
        
        # 将 Series 转换为列表
        if isinstance(text_feat, pd.Series):
            text_feat = text_feat.tolist()
        
        for i in range(0, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            batch_text = text_feat[i:batch_end]  # 直接使用列表索引
            print(f"\r处理进度: {batch_end}/{total_samples} ({batch_end/total_samples*100:.1f}%)", end="")
            
            # 确保内存清理
            gc.collect()
            torch.cuda.empty_cache()
            
            batch_embeddings = model.encode(
                sentences=batch_text,
                show_progress_bar=False,
                convert_to_tensor=True
            ).cpu()
            
            embeddings_list.append(batch_embeddings)
            
            # 立即删除不再需要的变量
            del batch_text, batch_embeddings
            
            # 手动清理内存
            gc.collect()
            torch.cuda.empty_cache()
        
        print()  # 换行
        
        # 合并所有批次的嵌入
        result = torch.cat(embeddings_list, dim=0)
        
        # 清理列表
        del embeddings_list
        gc.collect()
        
        return result


if __name__ == "__main__":
    # 重新处理已经下载的文件，并指定一个新的路径
    dataset = AmazonReviews(root="dataset/amazon", split="sports", force_reload=True)
   

