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

# 添加安全的全局变量
add_safe_globals([_reconstruct])

def parse_gz(path):
    """解析gzip文件"""
    with gzip.open(path, 'r') as g:
        for l in g:
            yield eval(l)
root_path = Path(__file__).parent.parent
raw_path = root_path / "dataset" / "amazon" / "raw" / "beauty"
processed_path = root_path / "dataset" / "amazon" / "processed"
class BeautyDatasetViewer:
    def __init__(self, raw_path=raw_path, processed_path=processed_path):
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)
    
    def view_raw_files(self):
        """显示原始数据文件列表及其大小"""
        print("\n=== 原始数据文件 ===")
        for file in self.raw_path.glob("*"):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"{file.name}: {size_mb:.2f}MB")
    
    def view_datamaps(self, n_samples=5):
        """查看ID映射文件"""
        print("\n=== 数据映射文件 (datamaps.json) ===")
        with open(self.raw_path / "datamaps.json", 'r') as f:
            data_maps = json.load(f)
        
        print("\n商品ID映射示例:")
        items = list(data_maps["item2id"].items())[:n_samples]
        for asin, id in items:
            print(f"ASIN: {asin} -> ID: {id}")
            
        print(f"\n总商品数: {len(data_maps['item2id'])}")
    
    def view_meta(self, n_samples=5):
        """查看商品元数据"""
        print("\n=== 商品元数据 (meta.json.gz) ===")
        items = []
        category_counts = {}  # 用于统计不同类别数量的商品数
        category_examples = {}  # 用于存储每种类别数量的示例商品
        total_items = 0
        
        def flatten_categories(categories):
            """将嵌套的类别列表展平为单一列表"""
            flattened = []
            for cat in categories:
                if isinstance(cat, list):
                    flattened.extend(cat)
                else:
                    flattened.append(cat)
            return list(dict.fromkeys(flattened))  # 去重
        
        # 第一次遍历：收集所有数据和统计信息
        for item in parse_gz(self.raw_path / "meta.json.gz"):
            total_items += 1
            
            # 展平并去重类别
            categories = flatten_categories(item.get('categories', []))
            num_categories = len(categories)
            
            # 更新商品的类别为展平后的列表
            item['categories'] = categories
            
            category_counts[num_categories] = category_counts.get(num_categories, 0) + 1
            
            # 保存每种类别数量的第一个示例
            if num_categories not in category_examples:
                category_examples[num_categories] = item
            
            # 保存样例数据
            if len(items) < n_samples:
                items.append(item)
        
        # 显示样例数据
        print("\n商品元数据示例:")
        for item in items:
            print("\n---")
            print(f"ASIN: {item.get('asin')}")
            print(f"标题: {item.get('title')}")
            print(f"品牌: {item.get('brand')}")
            print(f"类别 (展平后): {item.get('categories', [])}")
            print(f"价格: {item.get('price')}")
        
        # 显示类别统计信息
        print("\n类别统计信息:")
        print(f"商品总数: {total_items}")
        if category_counts:
            min_categories = min(category_counts.keys())
            max_categories = max(category_counts.keys())
            print(f"最少类别数: {min_categories}")
            print(f"最多类别数: {max_categories}")
            
            print("\n各类别数量分布及示例:")
            total_counted = 0
            for num_cats in sorted(category_counts.keys()):
                count = category_counts[num_cats]
                total_counted += count
                print(f"\n类别个数({num_cats}): 共{count}个商品")
                
                # 显示该类别数量的示例商品
                example = category_examples[num_cats]
                print("示例商品:")
                print(f"  ASIN: {example.get('asin')}")
                print(f"  标题: {example.get('title')}")
                print(f"  品牌: {example.get('brand')}")
                print(f"  类别: {example.get('categories', [])}")
                print(f"  价格: {example.get('price')}")
            
            print(f"\n统计总数: {total_counted}")
            if total_counted != total_items:
                print(f"警告：统计总数与商品总数不匹配！差异：{total_items - total_counted}")
    
    def view_sequential_data(self, n_samples=5):
        """查看序列数据"""
        print("\n=== 用户序列数据 (sequential_data.txt) ===")
        with open(self.raw_path / "sequential_data.txt", "r") as f:
            sequences = []
            for i, line in enumerate(f):
                if i < n_samples:
                    sequences.append(list(map(int, line.strip().split())))
                else:
                    break
        
        print("\n用户序列示例:")
        for seq in sequences:
            print(f"用户ID: {seq[0]}, 商品序列: {seq[1:5]}... (展示前4个商品)")
    
    def view_processed_data(self, n_samples=5):
        """查看处理后的数据"""
        print("\n=== 处理后的数据 (data.pt) ===")
        try:
            # 尝试使用 weights_only=False 加载数据
            loaded_data = torch.load(self.processed_path / "title_data_beauty_5tags.pt", weights_only=False)
            
            # 处理数据可能是列表的情况
            if isinstance(loaded_data, list):
                print(f"原始数据类型: {type(loaded_data)}")
                if len(loaded_data) > 0:
                    data = loaded_data[0]  # 取第一个元素
                    print(f"提取后的数据类型: {type(data)}")
                else:
                    print("错误: 加载的数据列表为空")
                    return
            # 如果数据是元组，取第一个元素
            elif isinstance(loaded_data, tuple):
                print(f"原始数据类型: {type(loaded_data)}")
                data = loaded_data[0]
                print(f"提取后的数据类型: {type(data)}")
            else:
                data = loaded_data
            
            print("\n数据结构:")
            print(f"数据类型: {type(data)}")
            print("数据键值:")
            for key in data.keys():
                print(f"- {key}")
                if key == 'item':
                    print("  子键值及示例:")
                    for subkey in data[key].keys():
                        print(f"    - {subkey}:")
                        if isinstance(data[key][subkey], torch.Tensor):
                            print(f"      形状: {data[key][subkey].shape}")
                            # 对于tags_indices，显示更多不同的样本
                            if subkey == 'tags_indices' or subkey == 'raw_tags_indices':
                                print(f"      样本1: {data[key][subkey][0]}")
                                print(f"      样本2: {data[key][subkey][1]}")
                                print(f"      样本3: {data[key][subkey][2]}")
                                print(f"      样本4: {data[key][subkey][3]}")
                                print(f"      样本5: {data[key][subkey][4]}")
                            else:
                                # 向量只取前5个元素
                                print(f"      示例1: {data[key][subkey][:5]}")
                        elif subkey == 'tags_mapping_dicts' or subkey == 'tags_reverse_mapping_dicts':
                            # 对于映射字典，显示每层的字典大小和部分示例
                            print(f"      类型: {type(data[key][subkey])}")
                            print(f"      层数: {len(data[key][subkey])}")
                            for i, layer_dict in enumerate(data[key][subkey]):
                                print(f"      第{i+1}层映射字典大小: {len(layer_dict)}")
                                # 显示前3个映射关系
                                items = list(layer_dict.items())[:3]
                                print(f"      第{i+1}层映射示例: {items}")
                        else:
                            print(f"      类型: {type(data[key][subkey])}")
                            print(f"      长度: {len(data[key][subkey])}")
                            print(f"      示例1: {data[key][subkey][0]}")
                            print(f"      示例2: {data[key][subkey][1]}")
                elif isinstance(data[key], dict):
                    print("  子键值:")
                    for subkey in data[key].keys():
                        print(f"    - {subkey}")
                        if isinstance(data[key][subkey], dict):
                            print("      内部键值:")
                            for inner_key in data[key][subkey].keys():
                                print(f"        - {inner_key}")

            # 尝试访问用户交互数据
            user_item_keys = [k for k in data.keys() if isinstance(k, tuple) and 'user' in k[0] and 'item' in k[2]]
            if user_item_keys:
                key = user_item_keys[0]
                if 'history' in data[key]:
                    print("\n序列划分示例 (leave-one-out策略):")
                    history_data = data[key]['history']
                    
                    # 获取几个用户的完整序列示例
                    sample_users = history_data['train']['userId'][:n_samples]
                    for user_idx in range(len(sample_users)):
                        user_id = sample_users[user_idx]
                        if isinstance(user_id, torch.Tensor):
                            user_id = user_id.item()
                        print(f"\n用户 {user_id} 的序列:")
                        
                        # 训练序列
                        train_seq = history_data['train']['itemId'][user_idx]
                        train_target = history_data['train']['itemId_fut'][user_idx]
                        if isinstance(train_seq, list):
                            valid_train = [x for x in train_seq if x >= 0]
                        else:  # tensor
                            valid_train = train_seq[train_seq >= 0].tolist()
                        print(f"训练: {valid_train} -> ", end="")
                        print(train_target.item() if isinstance(train_target, torch.Tensor) else train_target)
                        
                        # 验证序列
                        eval_seq = history_data['eval']['itemId'][user_idx]
                        eval_target = history_data['eval']['itemId_fut'][user_idx]
                        if isinstance(eval_seq, list):
                            valid_eval = [x for x in eval_seq if x >= 0]
                        else:  # tensor
                            valid_eval = eval_seq[eval_seq >= 0].tolist()
                        print(f"验证: {valid_eval} -> ", end="")
                        print(eval_target.item() if isinstance(eval_target, torch.Tensor) else eval_target)
                        
                        # 测试序列
                        test_seq = history_data['test']['itemId'][user_idx]
                        test_target = history_data['test']['itemId_fut'][user_idx]
                        if isinstance(test_seq, list):
                            valid_test = [x for x in test_seq if x >= 0]
                        else:  # tensor
                            valid_test = test_seq[test_seq >= 0].tolist()
                        print(f"测试: {valid_test} -> ", end="")
                        print(test_target.item() if isinstance(test_target, torch.Tensor) else test_target)
                    
                    print("\n数据集统计:")
                    print(f"总用户数: {len(set([u.item() if isinstance(u, torch.Tensor) else u for u in history_data['train']['userId']]))}")
                    
                    # 计算序列长度统计
                    if isinstance(history_data['train']['itemId'][0], list):
                        train_lengths = [len([x for x in seq if x >= 0]) for seq in history_data['train']['itemId']]
                    else:
                        train_lengths = [(seq >= 0).sum().item() for seq in history_data['train']['itemId']]
                    
                    print("\n序列长度统计:")
                    print(f"最短序列长度: {min(train_lengths)}")
                    print(f"最长序列长度: {max(train_lengths)}")
                    print(f"平均序列长度: {sum(train_lengths) / len(train_lengths):.2f}")

        except Exception as e:
            print(f"加载处理后的数据时出错: {str(e)}")
            print("\n提示: 请确保已经运行过数据处理步骤，并且数据文件存在于正确的位置。")
            import traceback
            print("\n详细错误信息:")
            print(traceback.format_exc())

    def remap_tags_indices(self):
        """
        对tags_indices进行二次映射，使每一层的标签索引从0开始
        并将原始索引保存为raw_tags_indices
        """
        print("\n=== 重映射标签索引 ===")
        try:
            # 加载处理后的数据
            data_path = self.processed_path / "title_data_beauty_5tags.pt"
            print(f"正在加载数据: {data_path}")
            loaded_data = torch.load(data_path, weights_only=False)
            
            # 记录原始数据格式和内容
            original_format = None  # 记录原始数据格式
            original_data = loaded_data  # 保存完整的原始数据
            
            if isinstance(loaded_data, list):
                print("数据是列表格式")
                original_format = "list"
                if len(loaded_data) > 0:
                    data = loaded_data[0]
                else:
                    print("错误: 加载的数据列表为空")
                    return
            # 如果数据是元组，取第一个元素进行处理
            elif isinstance(loaded_data, tuple):
                print(f"数据是元组格式，长度为: {len(loaded_data)}")
                original_format = "tuple"
                data = loaded_data[0]  # 只处理第一个元素
                # 打印元组中每个元素的类型
                for i, item in enumerate(loaded_data):
                    print(f"元组中第 {i+1} 个元素的类型: {type(item)}")
            else:
                print("数据是字典格式")
                original_format = "dict"
                data = loaded_data
            
            # 检查数据结构
            if 'item' not in data or 'tags_indices' not in data['item']:
                print("错误: 数据中不包含item.tags_indices字段")
                print(f"item键中包含的字段: {list(data['item'].keys())}")
                return
            
            # 获取原始标签索引
            original_tags_indices = data['item']['tags_indices']
            print(f"原始tags_indices形状: {original_tags_indices.shape}")
            
            # 检查是否已经重映射过
            if 'raw_tags_indices' in data['item']:
                print("警告: 数据已经被重映射过。")
                choice = input("是否继续重映射? (y/n): ")
                if choice.lower() != 'y':
                    print("操作已取消")
                    return
                print("继续重映射...")
            
            # 保存原始标签索引
            data['item']['raw_tags_indices'] = original_tags_indices.clone()
            
            # 对每一层标签进行重映射
            remapped_indices = torch.zeros_like(original_tags_indices, dtype=torch.long)
            mapping_dicts = []  # 存储每一层的映射字典
            reverse_mapping_dicts = []  # 存储反向映射字典
            
            for layer in range(original_tags_indices.shape[1]):  # 遍历每一层
                # 获取当前层的所有标签索引
                layer_indices = original_tags_indices[:, layer]
                
                # 获取唯一索引并排序
                unique_indices = torch.unique(layer_indices).tolist()
                unique_indices.sort()  # 确保排序，使映射更加稳定
                
                # 创建映射字典: 原始索引 -> 新索引(从0开始)
                mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}
                mapping_dicts.append(mapping)
                
                # 创建反向映射字典: 新索引 -> 原始索引
                reverse_mapping = {new_idx: old_idx for old_idx, new_idx in mapping.items()}
                reverse_mapping_dicts.append(reverse_mapping)
                
                # 应用映射
                for i in range(len(layer_indices)):
                    remapped_indices[i, layer] = mapping[layer_indices[i].item()]
            
            # 更新数据
            data['item']['tags_indices'] = remapped_indices
            
            # 保存映射字典，用于后续恢复原始索引
            data['item']['tags_mapping_dicts'] = mapping_dicts
            data['item']['tags_reverse_mapping_dicts'] = reverse_mapping_dicts
            
            # 显示重映射结果
            print("\n重映射完成!")
            print(f"新的tags_indices形状: {data['item']['tags_indices'].shape}")
            
            # 显示每一层的唯一标签数量
            print("\n每一层标签的唯一数量:")
            for layer in range(original_tags_indices.shape[1]):
                original_unique = len(torch.unique(original_tags_indices[:, layer]))
                new_unique = len(torch.unique(remapped_indices[:, layer]))
                print(f"第{layer+1}层: 原始唯一标签数 {original_unique}, 重映射后唯一标签数 {new_unique}")
                
                # 显示部分映射关系
                print(f"  映射示例(前3个):")
                items = list(mapping_dicts[layer].items())[:3]
                for old_idx, new_idx in items:
                    print(f"    原始索引 {old_idx} -> 新索引 {new_idx}")
            
            # 验证可恢复性
            print("\n验证重映射的可恢复性:")
            # 随机选择几个样本进行验证
            sample_indices = torch.randint(0, original_tags_indices.shape[0], (5,))
            for idx in sample_indices:
                idx = idx.item()
                print(f"\n样本 {idx}:")
                for layer in range(original_tags_indices.shape[1]):
                    original = original_tags_indices[idx, layer].item()
                    remapped = remapped_indices[idx, layer].item()
                    recovered = reverse_mapping_dicts[layer][remapped]
                    print(f"  第{layer+1}层: 原始索引 {original} -> 重映射索引 {remapped} -> 恢复索引 {recovered}")
                    assert original == recovered, f"恢复失败: {original} != {recovered}"
            
            # 保存更新后的数据，根据原始格式保存
            print("\n正在保存更新后的数据...")
            if original_format == "list":
                torch.save([data], data_path)
            elif original_format == "tuple":
                # 如果是元组，保持原始元组的长度和其他元素不变
                if len(original_data) == 1:
                    torch.save((data,), data_path)
                elif len(original_data) == 2:
                    torch.save((data, original_data[1]), data_path)
                elif len(original_data) == 3:
                    torch.save((data, original_data[1], original_data[2]), data_path)
                else:
                    # 创建一个新的元组，第一个元素是修改后的数据，其余元素保持不变
                    new_data = (data,) + original_data[1:]
                    torch.save(new_data, data_path)
            else:  # dict
                torch.save(data, data_path)
            print(f"数据已保存到: {data_path}")
            
            # 显示重映射前后的对比
            print("\n重映射前后的标签索引对比(前5个样本):")
            for i in range(5):
                print(f"\n样本 {i}:")
                print(f"  原始索引: {data['item']['raw_tags_indices'][i]}")
                print(f"  重映射后: {data['item']['tags_indices'][i]}")
            
        except Exception as e:
            print(f"重映射标签索引时出错: {str(e)}")
            import traceback
            print("\n详细错误信息:")
            print(traceback.format_exc())

def main():
    viewer = BeautyDatasetViewer()
    
    while True:
        print("\n=== Amazon Beauty 数据集查看器 ===")
        print("1. 查看原始数据文件列表")
        print("2. 查看数据映射文件 (datamaps.json)")
        print("3. 查看商品元数据 (meta.json.gz)")
        print("4. 查看用户序列数据 (sequential_data.txt)")
        print("5. 查看处理后的数据 (data.pt)")
        print("6. 重映射标签索引")
        
        print("0. 退出")
        
        choice = input("\n请选择要查看的内容 (0-6): ")
        
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
            print("无效的选择，请重试")
        
        input("\n按回车键继续...")

if __name__ == "__main__":
    main()