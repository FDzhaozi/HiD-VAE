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

# 将KuaiRand添加到RecDataset枚举
@gin.constants_from_enum
class KuaiRandDataset(Enum):
    KUAIRAND = 4

class KuaiRandItemData(Dataset):
    """
    加载KuaiRand数据集的商品信息，与ItemData类兼容的方式
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
        初始化KuaiRand商品数据加载器
        
        参数:
            root: 数据集根目录, e.g., 'dataset/kuairand'
            force_process: 是否强制重新处理数据 (在这里不起作用，仅为与ItemData兼容)
            dataset: 数据集类型 (这里固定为KUAIRAND)
            train_test_split: 指定要加载的数据分割，可选值为"train"、"eval"或"all"
            data_file: 指定要加载的数据文件名，不指定则使用默认路径
        """
        print(f"root: {root}")
        print(f"dataset: {dataset}")
        
        # 确定数据文件路径
        if data_file is None:
            data_file = "title_data_kuairand_5tags.pt"
        
        data_path = os.path.join(root, "processed", data_file)
        # 设置processed_paths属性
        self.processed_paths = [data_path]
        
        print(f"data_path: {data_path}")
        
        # 加载数据
        try:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"数据文件不存在: {data_path}")
                
            self.data = torch.load(data_path, map_location='cpu')
            print("✓ 数据加载成功!")
            
            # 确保item特征是float32类型
            if 'x' in self.data['item']:
                self.data['item']['x'] = self.data['item']['x'].float()
            
        except Exception as e:
            print(f"加载数据时出错: {e}")
            raise
        
        # 应用训练/评估过滤
        if train_test_split == "train":
            if 'is_train' in self.data['item']:
                filt = self.data['item']['is_train']
            else:
                # 如果没有is_train字段，使用80%的数据作为训练集
                total_items = len(self.data['item']['x'])
                train_size = int(0.8 * total_items)
                filt = torch.zeros(total_items, dtype=torch.bool)
                filt[:train_size] = True
                print("注意: 使用默认80-20分割作为训练集和评估集")
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
        
        # 保存所需的数据字段，并确保是float32类型
        self.item_data = self.data['item']['x'][filt].float()
        
        # 处理文本数据
        if 'text' in self.data['item']:
            text_data = self.data['item']['text']
            if isinstance(text_data, (list, tuple)):
                # 如果是列表，先转换为张量
                text_data = torch.tensor([1 if t else 0 for t in text_data])
            self.item_text = text_data[filt]
        else:
            self.item_text = None
            print("警告: 数据集中没有找到文本字段")
        
        # 提取标签数据
        self.has_tags = False
        
        # 检查新的标签格式(tags_emb_l1, tags_emb_l2, tags_emb_l3)
        if all(f'tags_emb_l{i}' in self.data['item'] for i in range(1, 4)):
            self.tags_emb = torch.stack([
                self.data['item'][f'tags_emb_l{i}'][filt].float() for i in range(1, 4)
            ], dim=1)
            self.has_tags = True
        # 检查旧的标签格式(tags_emb)
        elif 'tags_emb' in self.data['item']:
            self.tags_emb = self.data['item']['tags_emb'][filt].float()
            self.has_tags = True
        else:
            self.tags_emb = None
            print("警告: 数据集中没有找到标签嵌入字段")
        
        if 'tags_indices' in self.data['item']:
            self.tags_indices = self.data['item']['tags_indices'][filt]
            self.has_tags = True
        else:
            self.tags_indices = None
            print("警告: 数据集中没有找到标签索引字段")
            
        if 'tags' in self.data['item']:
            tags_data = self.data['item']['tags']
            if isinstance(tags_data, (list, tuple)):
                # 如果是列表，先转换为张量
                tags_data = torch.tensor([1 if t else 0 for t in tags_data])
            self.tags = tags_data[filt]
            self.has_tags = True
        else:
            self.tags = None
            print("警告: 数据集中没有找到标签文本字段")

    def __len__(self):
        return self.item_data.shape[0]

    def __getitem__(self, idx):
        """返回指定索引的商品数据，与ItemData类兼容的格式"""
        item_ids = torch.tensor(idx).unsqueeze(0) if not isinstance(idx, torch.Tensor) else idx
        x = self.item_data[idx, :768]  # 保持与其他数据集一致的特征维度
        
        # 构建基本的批次数据
        batch_data = {
            "user_ids": -1 * torch.ones_like(item_ids.squeeze(0)),
            "ids": item_ids,
            "ids_fut": -1 * torch.ones_like(item_ids.squeeze(0)),
            "x": x,
            "x_fut": -1 * torch.ones_like(item_ids.squeeze(0)),
            "seq_mask": torch.ones_like(item_ids, dtype=bool)
        }
        
        # 如果有标签数据，则使用TaggedSeqBatch
        if self.has_tags:
            # 获取标签数据
            if isinstance(idx, torch.Tensor):
                tags_emb = self.tags_emb[idx]
                tags_indices = self.tags_indices[idx]
            elif isinstance(idx, list):
                # 处理列表类型的idx
                tags_emb = self.tags_emb[idx]
                tags_indices = self.tags_indices[idx]
            else:
                # 处理整数类型的idx
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
            # 如果没有标签数据，则使用普通的SeqBatch
            return SeqBatch(**batch_data)


def analyze_tag_distribution(dataset):
    """分析标签分布并生成统计信息"""
    
    print("\n===== 标签分布分析 =====")
    
    if not dataset.has_tags:
        print("错误: 数据集中没有标签信息，无法进行分析")
        return
    
    # 检查标签索引
    if dataset.tags_indices is not None:
        # 计算每个级别中非-1的标签数量
        non_empty_tags = (dataset.tags_indices != -1).sum(dim=0)
        total_items = len(dataset)
        
        print(f"数据集中共有 {total_items} 个商品")
        print(f"各级别标签覆盖率:")
        
        for level in range(dataset.tags_indices.shape[1]):
            coverage = non_empty_tags[level].item() / total_items * 100
            print(f"  - 第{level+1}级标签: {non_empty_tags[level].item()} 个商品有标签 ({coverage:.2f}%)")
        
        # 计算每个商品的非空标签数量
        tags_per_item = (dataset.tags_indices != -1).sum(dim=1)
        avg_tags = tags_per_item.float().mean().item()
        
        print(f"\n平均每个商品有 {avg_tags:.2f} 个非空标签")
        
        # 统计每个商品拥有的标签数量分布
        tag_counts = Counter(tags_per_item.tolist())
        print("\n商品标签数量分布:")
        for count in sorted(tag_counts.keys()):
            percentage = tag_counts[count] / total_items * 100
            print(f"  - {count}个标签: {tag_counts[count]} 个商品 ({percentage:.2f}%)")
        
        # 分析每个级别的标签分布
        print("\n各级别标签值分布:")
        for level in range(dataset.tags_indices.shape[1]):
            # 获取当前级别的所有非-1标签
            level_tags = dataset.tags_indices[:, level]
            valid_tags = level_tags[level_tags != -1]
            
            if len(valid_tags) > 0:
                unique_tags = torch.unique(valid_tags)
                print(f"  - 第{level+1}级标签: {len(unique_tags)} 个不同的标签值")
                
                # 统计TOP 10最常见的标签
                tag_counts = Counter(valid_tags.tolist())
                print(f"    TOP 10最常见的标签:")
                for tag, count in tag_counts.most_common(10):
                    percentage = count / len(valid_tags) * 100
                    print(f"      - 标签ID {tag}: {count} 次出现 ({percentage:.2f}%)")
    
    # 如果有标签文本，进行文本分析
    if dataset.tags is not None:
        print("\n标签文本分析:")
        for level in range(dataset.tags.shape[1]):
            # 统计非空标签文本
            level_tags = [tag for tag in dataset.tags[:, level] if tag != '']
            if level_tags:
                tag_counts = Counter(level_tags)
                print(f"  - 第{level+1}级标签: {len(tag_counts)} 个不同的标签文本")
                
                # 统计TOP 10最常见的标签文本
                print(f"    TOP 10最常见的标签文本:")
                for tag, count in tag_counts.most_common(10):
                    percentage = count / len(level_tags) * 100
                    print(f"      - '{tag}': {count} 次出现 ({percentage:.2f}%)")

def plot_tag_distribution(dataset, save_dir=None):
    """绘制标签分布图表"""
    
    if not dataset.has_tags or dataset.tags_indices is None:
        print("错误: 数据集中没有标签信息，无法绘制图表")
        return
    
    # 创建保存目录
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 计算每个商品的非空标签数量
    tags_per_item = (dataset.tags_indices != -1).sum(dim=1).tolist()
    
    # 1. 绘制标签数量分布图
    plt.figure(figsize=(10, 6))
    plt.hist(tags_per_item, bins=range(5), alpha=0.7, rwidth=0.8)
    plt.xlabel('每个商品的标签数量')
    plt.ylabel('商品数量')
    plt.title('商品标签数量分布')
    plt.xticks(range(4))
    plt.grid(axis='y', alpha=0.3)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'tags_per_item.png'), dpi=300, bbox_inches='tight')
        print(f"已保存图表: {os.path.join(save_dir, 'tags_per_item.png')}")
    else:
        plt.show()
    plt.close()
    
    # 2. 绘制各级别标签覆盖率
    coverage = [(dataset.tags_indices[:, i] != -1).sum().item() / len(dataset) * 100 for i in range(dataset.tags_indices.shape[1])]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(coverage) + 1), coverage, alpha=0.7)
    plt.xlabel('标签级别')
    plt.ylabel('覆盖率 (%)')
    plt.title('各级别标签覆盖率')
    plt.xticks(range(1, len(coverage) + 1))
    plt.ylim(0, 100)
    
    for i, v in enumerate(coverage):
        plt.text(i + 1, v + 1, f'{v:.1f}%', ha='center')
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'tag_level_coverage.png'), dpi=300, bbox_inches='tight')
        print(f"已保存图表: {os.path.join(save_dir, 'tag_level_coverage.png')}")
    else:
        plt.show()
    plt.close()
    
    # 3. 绘制TOP 10标签分布 (对每个级别)
    for level in range(dataset.tags_indices.shape[1]):
        # 获取当前级别的所有非-1标签
        level_tags = dataset.tags_indices[:, level]
        valid_tags = level_tags[level_tags != -1]
        
        if len(valid_tags) > 0:
            tag_counts = Counter(valid_tags.tolist())
            top_tags = tag_counts.most_common(10)
            
            # 绘制TOP 10标签分布
            plt.figure(figsize=(12, 6))
            
            labels = [f'ID {tag}' for tag, _ in top_tags]
            values = [count for _, count in top_tags]
            
            plt.bar(labels, values, alpha=0.7)
            plt.xlabel('标签ID')
            plt.ylabel('出现次数')
            plt.title(f'第{level+1}级TOP 10标签分布')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'top_tags_level_{level+1}.png'), dpi=300, bbox_inches='tight')
                print(f"已保存图表: {os.path.join(save_dir, f'top_tags_level_{level+1}.png')}")
            else:
                plt.show()
            plt.close()

# 独立运行脚本时的入口点
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='KuaiRand数据集分析工具')
    parser.add_argument('--data_path', type=str, default='dataset/kuairand',
                        help='KuaiRand数据集根目录')
    parser.add_argument('--data_file', type=str, default='kuairand_data_minimal_interactions30000.pt',
                        help='要加载的数据文件名')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'eval', 'all'],
                        help='要分析的数据集分割')
    parser.add_argument('--plot', action='store_true', help='是否绘制并保存图表')
    parser.add_argument('--plot_dir', type=str, default='plots/kuairand_analysis',
                        help='图表保存目录')
    
    args = parser.parse_args()
    
    # 加载数据集
    dataset = KuaiRandItemData(
        root=args.data_path, 
        train_test_split=args.split,
        data_file=args.data_file
    )
    
    # 打印数据集基本信息
    print("\n===== KuaiRand数据集基本信息 =====")
    print(f"数据路径: {os.path.join(args.data_path, 'processed', args.data_file)}")
    print(f"数据分割: {args.split}")
    print(f"商品数量: {len(dataset)}")
    
    if dataset.item_data is not None:
        print(f"特征维度: {dataset.item_data.shape[1]}")
    
    if dataset.has_tags:
        if dataset.tags_emb is not None:
            print(f"标签嵌入形状: {dataset.tags_emb.shape}")
        if dataset.tags_indices is not None:
            print(f"标签索引形状: {dataset.tags_indices.shape}")
        if dataset.tags is not None:
            print(f"标签文本形状: {dataset.tags.shape}")
    
    # 分析标签分布
    analyze_tag_distribution(dataset)
    
    # 绘制图表
    if args.plot:
        plot_tag_distribution(dataset, args.plot_dir)
