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

# 添加安全的全局变量
add_safe_globals([_reconstruct])

# 从fill_kuairand.py导入需要的函数
from data.fill_kuairand import cosine_similarity, check_completion_progress


def build_tag_hierarchy(data):
    """
    遍历数据集，构建标签层级之间的父子关系图。
    """
    print("正在构建标签层级关系图...")
    item_data = data['item']
    tags_indices = item_data['tags_indices']
    
    l1_to_l2 = {}
    l2_to_l3 = {}
    
    for i in tqdm(range(tags_indices.shape[0]), desc="分析层级关系"):
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
            
    # 将set转换为list，方便后续使用
    for k in l1_to_l2:
        l1_to_l2[k] = list(l1_to_l2[k])
    for k in l2_to_l3:
        l2_to_l3[k] = list(l2_to_l3[k])
        
    print(f"层级关系构建完成。 L1->L2 关系数: {len(l1_to_l2)}, L2->L3 关系数: {len(l2_to_l3)}")
    
    return {"l1_to_l2": l1_to_l2, "l2_to_l3": l2_to_l3}


def load_kuairand_dataset(data_path, seed=42):
    """
    加载KuaiRand数据集并返回数据对象
    
    参数:
        data_path: 数据文件路径
        seed: 随机种子，用于确保可重现性
    
    返回:
        数据对象
    """
    print(f"正在从 {data_path} 加载数据...")
    
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    try:
        # 加载数据集，显式设置weights_only=False来处理PyTorch 2.6的默认行为变化
        loaded_data = torch.load(data_path, map_location='cpu', weights_only=False)
        print("✓ 数据加载成功!")
        
        # 处理数据可能是列表或元组的情况
        if isinstance(loaded_data, list):
            print(f"原始数据类型: {type(loaded_data)}")
            if len(loaded_data) > 0:
                data = loaded_data[0]  # 取第一个元素
                print(f"提取后的数据类型: {type(data)}")
            else:
                print("错误: 加载的数据列表为空")
                return None
        # 如果数据是元组，取第一个元素
        elif isinstance(loaded_data, tuple):
            print(f"原始数据类型: {type(loaded_data)}")
            data = loaded_data[0]
            print(f"提取后的数据类型: {type(data)}")
        else:
            data = loaded_data
            
        return data
    except Exception as e:
        print(f"加载数据时出错: {e}")
        import traceback
        print(traceback.format_exc())
        raise


def create_tag_pools(data):
    """
    为每个层级创建标签池，包括标签文本和对应的嵌入向量
    
    参数:
        data: 数据对象，包含item节点信息
    
    返回:
        三个层级的标签池，每个池包含标签文本和对应的嵌入向量
    """
    print("正在创建标签池...")
    
    if 'item' not in data:
        raise ValueError("数据集中没有item节点")
    
    item_data = data['item']
    
    if 'tags_indices' not in item_data:
        raise ValueError("数据集中没有标签索引信息")
    
    tags_indices = item_data['tags_indices']
    
    # 获取标签文本数据
    if 'tags' not in item_data or item_data['tags'] is None:
        raise ValueError("数据集中没有标签文本信息")
    
    # 创建标签池
    tag_pools = []
    
    for level in range(tags_indices.shape[1]):
        # 获取所有有效的标签索引
        valid_indices = tags_indices[:, level]
        valid_indices = valid_indices[valid_indices != -1]
        unique_indices = torch.unique(valid_indices)
        
        # 创建标签池
        level_pool = {}
        
        for idx in unique_indices:
            # 找到具有该标签的所有商品
            items_with_tag = (tags_indices[:, level] == idx).nonzero(as_tuple=True)[0]
            
            if len(items_with_tag) == 0:
                continue
            
            # 获取标签文本
            sample_item = items_with_tag[0].item()
            tag_text = item_data['tags'][sample_item][level]
            
            # 获取标签嵌入
            if 'tags_emb' in item_data and item_data['tags_emb'] is not None:
                # 获取所有具有该标签的商品的嵌入向量，并取平均
                tag_embs = item_data['tags_emb'][items_with_tag, level]
                tag_emb = torch.mean(tag_embs, dim=0)
            else:
                # 如果没有专门的标签嵌入，使用样本商品的特征向量
                tag_emb = item_data['x'][sample_item]
            
            # 添加到标签池
            if tag_text and tag_text != '':
                level_pool[idx.item()] = {
                    'text': tag_text,
                    'embedding': tag_emb,
                    'count': len(items_with_tag)
                }
        
        tag_pools.append(level_pool)
        print(f"第 {level+1} 级标签池创建完成，包含 {len(level_pool)} 个不同标签")
    
    return tag_pools


def retrieve_most_similar_tag(context_embedding, tag_pool, candidate_ids=None):
    """
    根据上下文嵌入检索最相似的标签。
    
    参数:
        context_embedding: 上下文嵌入向量 (可以是视频标题向量，或和父标签向量的组合)
        tag_pool: 标签池，包含标签ID、文本和嵌入
        candidate_ids: (可选) 一个包含候选标签ID的列表，用于约束搜索范围
    
    返回:
        最相似的标签ID、文本、相似度和嵌入向量
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
    使用层级约束和顺序填充来补全缺失的标签。
    包含回退机制：当层级约束找不到候选时，回退到在整个层级中搜索。
    
    参数:
        data: 数据对象
        item_idx: 商品索引
        tag_pools: 三个层级的标签池
        tag_hierarchy: 标签层级关系图
        
    返回:
        补全结果字典
    """
    item_data = data['item']
    original_indices = item_data['tags_indices'][item_idx].clone() # 克隆以跟踪变化
    
    # 获取视频特征向量
    item_embedding = item_data['x'][item_idx]
    
    completion_result = {
        "status": "pending",
        "补全标签": {},
        "选择理由": {} # 每层单独记录理由
    }
    
    # 存储各级标签的向量信息，用于构建上下文
    level_embeddings = {}
    if 'tags_emb' in item_data:
        for i in range(3):
            if original_indices[i] != -1:
                level_embeddings[i] = item_data['tags_emb'][item_idx, i]

    # --- 按顺序填充 ---
    
    # 填充第1级
    if original_indices[0] == -1:
        tag_id, tag_text, _, tag_embedding = retrieve_most_similar_tag(item_embedding, tag_pools[0])
        if tag_id is not None:
            original_indices[0] = tag_id
            level_embeddings[0] = tag_embedding
            completion_result["补全标签"][0] = {"id": tag_id, "name": tag_text, "embedding": tag_embedding}
            completion_result["选择理由"][0] = "基于视频标题向量全局搜索"

    # 填充第2级
    if original_indices[1] == -1 and original_indices[0] != -1:
        l1_id = original_indices[0].item()
        candidate_l2_ids = tag_hierarchy['l1_to_l2'].get(l1_id)
        reason = "基于层级约束搜索"

        # 回退机制：如果层级约束下没有候选，则在整个L2中搜索
        if not candidate_l2_ids:
            candidate_l2_ids = None # 传入None表示全局搜索
            reason = "层级约束无候选，回退到全局搜索"
        
        # 构建上下文向量
        l1_embedding = level_embeddings.get(0, item_embedding) # 如果L1刚填充，用其向量
        context_embedding = 0.6 * l1_embedding + 0.4 * item_embedding
        
        tag_id, tag_text, _, tag_embedding = retrieve_most_similar_tag(context_embedding, tag_pools[1], candidate_l2_ids)
        if tag_id is not None:
            original_indices[1] = tag_id
            level_embeddings[1] = tag_embedding
            completion_result["补全标签"][1] = {"id": tag_id, "name": tag_text, "embedding": tag_embedding}
            completion_result["选择理由"][1] = reason

    # 填充第3级
    if original_indices[2] == -1 and original_indices[1] != -1:
        l2_id = original_indices[1].item()
        candidate_l3_ids = tag_hierarchy['l2_to_l3'].get(l2_id)
        reason = "基于层级约束搜索"

        # 回退机制
        if not candidate_l3_ids:
            candidate_l3_ids = None
            reason = "层级约束无候选，回退到全局搜索"
            
        # 构建上下文向量
        l1_embedding = level_embeddings.get(0, item_embedding)
        l2_embedding = level_embeddings.get(1, item_embedding)
        context_embedding = 0.5 * l2_embedding + 0.3 * l1_embedding + 0.2 * item_embedding
        
        tag_id, tag_text, _, tag_embedding = retrieve_most_similar_tag(context_embedding, tag_pools[2], candidate_l3_ids)
        if tag_id is not None:
            original_indices[2] = tag_id
            level_embeddings[2] = tag_embedding
            completion_result["补全标签"][2] = {"id": tag_id, "name": tag_text, "embedding": tag_embedding}
            completion_result["选择理由"][2] = reason
    
    if completion_result["补全标签"]:
        completion_result["status"] = "success"
    else:
        # 检查是否还有缺失
        if torch.any(original_indices == -1):
             completion_result["status"] = "failed"
             completion_result["message"] = "存在无法填充的层级（例如L1缺失或L1补全失败）"
        else:
             completion_result["status"] = "complete"
             completion_result["message"] = "所有标签已完整，无需补全"

    return completion_result


def simple_complete_tags(data, tag_pools, tag_hierarchy, batch_size=100, save_path=None, start_idx=0):
    """
    使用相似度批量补全标签并保存到新的数据文件
    
    参数:
        data: 数据对象
        tag_pools: 三个层级的标签池
        tag_hierarchy: 标签层级关系图
        batch_size: 每批处理的商品数量
        save_path: 保存补全后数据的路径
        start_idx: 起始索引，用于断点续传
    
    返回:
        更新后的数据对象
    """
    # 创建数据的副本，避免修改原始数据
    new_data = deepcopy(data)
    
    item_data = new_data['item']
    num_items = item_data['x'].shape[0]
    
    # 找出缺少至少一层标签的商品
    incomplete_items = []
    for i in range(num_items):
        tags_indices = item_data['tags_indices'][i]
        if torch.any(tags_indices == -1):
            incomplete_items.append(i)
    
    # 过滤掉起始索引之前的项
    incomplete_items = [idx for idx in incomplete_items if idx >= start_idx]
    
    total_incomplete = len(incomplete_items)
    print(f"找到 {total_incomplete} 个缺失标签的商品，从索引 {start_idx} 开始处理")
    
    # 找出空标题的视频
    empty_title_items = []
    if 'text' in item_data and item_data['text'] is not None:
        for i in range(num_items):
            if not item_data['text'][i] or item_data['text'][i].strip() == '':
                empty_title_items.append(i)
        print(f"找到 {len(empty_title_items)} 个空标题的视频")
    
    # 为每个批次创建进度条
    num_batches = (total_incomplete + batch_size - 1) // batch_size
    
    # 用于保存每批次结果的统计信息
    stats = {
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "empty_titles_filled": 0,
        "errors": []
    }
    
    # 批量处理
    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min((batch_num + 1) * batch_size, total_incomplete)
        batch_indices = incomplete_items[batch_start:batch_end]
        
        print(f"\n处理批次 {batch_num + 1}/{num_batches} (商品索引 {batch_indices[0]} 至 {batch_indices[-1]})")
        
        # 创建进度条
        pbar = tqdm(batch_indices, desc=f"批次 {batch_num+1}")
        
        # 遍历批次中的每个商品
        for idx in pbar:
            # 检查是否仍然需要补全
            tags_indices = item_data['tags_indices'][idx]
            if not torch.any(tags_indices == -1):
                stats["skipped"] += 1
                continue
            
            # 使用新的分层填充函数
            completion_result = complete_tags_hierarchically(new_data, idx, tag_pools, tag_hierarchy)
            stats["processed"] += 1
            
            # 如果补全成功，更新数据
            if completion_result["status"] == "success" and "补全标签" in completion_result:
                # 更新标签
                for level, tag_info in completion_result["补全标签"].items():
                    level = int(level)
                    if tag_info["id"] is not None:
                        # 更新标签文本
                        item_data['tags'][idx][level] = tag_info["name"]
                        # 更新标签索引
                        item_data['tags_indices'][idx, level] = tag_info["id"]
                        # 更新标签嵌入（如果可用）
                        if tag_info["embedding"] is not None and 'tags_emb' in item_data:
                            item_data['tags_emb'][idx, level] = tag_info["embedding"]
                
                # 检查是否为空标题视频，如果是则用补全后的标签作为新标题
                if 'text' in item_data and idx in empty_title_items:
                    # 获取所有标签（包括原有的和新补全的）
                    all_tags = item_data['tags'][idx]
                    # 过滤掉空标签
                    valid_tags = [tag for tag in all_tags if tag and tag.strip() != '']
                    
                    if valid_tags:
                        # 使用标签组合作为新标题
                        new_title = " ".join(valid_tags)
                        item_data['text'][idx] = new_title
                        stats["empty_titles_filled"] += 1
                        pbar.set_postfix({"填充标题": new_title[:20] + ('...' if len(new_title) > 20 else '')})
                
                stats["successful"] += 1
            else:
                stats["failed"] += 1
                error_msg = completion_result.get("message", "未知错误")
                stats["errors"].append((idx, error_msg))
            
            # 每完成50个商品保存一次数据
            if stats["processed"] % 50 == 0 and save_path:
                temp_save_path = f"{save_path}_temp"
                torch.save(new_data, temp_save_path)
                print(f"\n已保存临时数据到 {temp_save_path}")
        
        # 每完成一个批次保存一次数据
        if save_path:
            torch.save(new_data, save_path)
            print(f"\n已保存批次 {batch_num+1} 的数据到 {save_path}")
    
    # 保存最终数据
    if save_path:
        torch.save(new_data, save_path)
        print(f"\n已保存最终数据到 {save_path}")
    
    # 打印统计信息
    print("\n===== 处理统计 =====")
    print(f"总处理商品数: {stats['processed']}")
    print(f"成功补全数: {stats['successful']}")
    print(f"失败数: {stats['failed']}")
    print(f"跳过数（已有完整标签）: {stats['skipped']}")
    print(f"填充空标题数: {stats['empty_titles_filled']}")
    
    if stats["errors"]:
        print(f"\n前10个错误:")
        for idx, error in stats["errors"][:10]:
            print(f"  - 商品索引 {idx}: {error}")
    
    return new_data


def view_tags(data, num_samples=100, seed=42):
    """
    随机抽取视频信息并打印标签信息
    
    参数:
        data: 数据对象
        num_samples: 要抽取的样本数量
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    
    item_data = data['item']
    num_items = item_data['x'].shape[0]
    
    # 随机抽样
    sample_indices = random.sample(range(num_items), min(num_samples, num_items))
    
    print(f"\n===== 随机抽取的 {len(sample_indices)} 条视频信息 =====")
    
    # 计算标签完整度统计
    complete_count = 0
    level_complete_counts = [0, 0, 0]
    
    for i, idx in enumerate(sample_indices):
        title = item_data['text'][idx] if 'text' in item_data and item_data['text'] is not None else "未知标题"
        tags = item_data['tags'][idx]
        tags_indices = item_data['tags_indices'][idx]
        
        # 检查标签完整度
        is_complete = True
        for level in range(len(tags)):
            if tags_indices[level] != -1 and tags[level] != '':
                level_complete_counts[level] += 1
            else:
                is_complete = False
        
        if is_complete:
            complete_count += 1
        
        print(f"\n样本 {i+1} (索引 {idx}):")
        print(f"  标题: {title[:50]}{'...' if len(title) > 50 else ''}")
        print(f"  标签:")
        for level in range(len(tags)):
            tag_status = "有效" if tags_indices[level] != -1 and tags[level] != '' else "缺失"
            tag_text = tags[level] if tags_indices[level] != -1 and tags[level] != '' else "无"
            print(f"    - 第{level+1}级: {tag_text} (ID: {tags_indices[level].item()}, 状态: {tag_status})")
    
    # 打印统计信息
    print("\n===== 标签统计信息 =====")
    print(f"完整标签的视频数量: {complete_count}/{len(sample_indices)} ({complete_count/len(sample_indices)*100:.2f}%)")
    for level in range(3):
        print(f"第{level+1}级标签完整的视频数量: {level_complete_counts[level]}/{len(sample_indices)} ({level_complete_counts[level]/len(sample_indices)*100:.2f}%)")


def analyze_tag_distribution(data):
    """
    分析标签分布情况
    
    参数:
        data: 数据对象
    """
    print("\n===== 标签分布分析 =====")
    
    item_data = data['item']
    tags_indices = item_data['tags_indices']
    num_items = tags_indices.shape[0]
    num_levels = tags_indices.shape[1]
    
    print(f"商品总数: {num_items}")
    print(f"标签层级数: {num_levels}")
    
    # 分析每个层级的标签分布
    for level in range(num_levels):
        level_indices = tags_indices[:, level]
        
        # 计算有效标签数量
        valid_count = (level_indices != -1).sum().item()
        valid_percentage = valid_count / num_items * 100
        
        print(f"\n第{level+1}层标签统计:")
        print(f"  有效标签数量: {valid_count}/{num_items} ({valid_percentage:.2f}%)")
        
        # 统计唯一标签数量
        unique_indices = torch.unique(level_indices)
        print(f"  唯一标签数量: {len(unique_indices)}")
        
        # 统计标签出现频率
        if valid_count > 0:
            # 排除-1（缺失值）
            valid_indices = level_indices[level_indices != -1]
            
            # 计算每个标签的出现次数
            tag_counts = {}
            for idx in valid_indices:
                idx_val = idx.item()
                tag_counts[idx_val] = tag_counts.get(idx_val, 0) + 1
            
            # 按出现频率排序
            sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
            
            # 显示出现频率最高的前10个标签
            print(f"  出现频率最高的前10个标签:")
            for i, (tag_id, count) in enumerate(sorted_tags[:10]):
                if i < 10:
                    # 获取标签文本（如果可用）
                    tag_text = "未知"
                    for j in range(num_items):
                        if tags_indices[j, level].item() == tag_id:
                            tag_text = item_data['tags'][j][level]
                            break
                    
                    print(f"    {i+1}. ID: {tag_id}, 文本: {tag_text}, 出现次数: {count}, 占比: {count/num_items*100:.2f}%")
    
    # 分析标签完整度
    complete_items = 0
    level_complete_counts = [0] * num_levels
    
    for i in range(num_items):
        item_indices = tags_indices[i]
        
        # 检查每个层级是否有效
        is_complete = True
        for level in range(num_levels):
            if item_indices[level] != -1:
                level_complete_counts[level] += 1
            else:
                is_complete = False
        
        if is_complete:
            complete_items += 1
    
    print("\n标签完整度统计:")
    print(f"完整标签的商品数量: {complete_items}/{num_items} ({complete_items/num_items*100:.2f}%)")
    
    for level in range(num_levels):
        print(f"第{level+1}层标签完整的商品数量: {level_complete_counts[level]}/{num_items} ({level_complete_counts[level]/num_items*100:.2f}%)")


def check_completion_progress(data, save_path=None):
    """
    检查标签补全进度，找出第一个有残缺标签的索引
    
    参数:
        data: 数据对象
        save_path: 保存补全后数据的路径（可选）
    
    返回:
        第一个有残缺标签的索引，如果所有标签都完整则返回None
    """
    print("\n===== 检查标签补全进度 =====")
    
    if 'item' not in data:
        raise ValueError("数据集中没有item节点")
    
    item_data = data['item']
    
    if 'tags_indices' not in item_data:
        raise ValueError("数据集中没有标签索引信息")
    
    tags_indices = item_data['tags_indices']
    num_items = tags_indices.shape[0]
    num_levels = tags_indices.shape[1]
    
    # 统计每层标签的完整情况
    level_complete_counts = [0] * num_levels
    incomplete_items = []
    
    for i in range(num_items):
        item_indices = tags_indices[i]
        
        # 检查每个层级是否有效
        has_missing = False
        for level in range(num_levels):
            if item_indices[level] != -1:
                level_complete_counts[level] += 1
            else:
                has_missing = True
        
        if has_missing:
            incomplete_items.append(i)
    
    # 打印统计信息
    print(f"商品总数: {num_items}")
    print(f"标签层级数: {num_levels}")
    
    for level in range(num_levels):
        complete_percentage = level_complete_counts[level] / num_items * 100
        print(f"第{level+1}层标签完整度: {level_complete_counts[level]}/{num_items} ({complete_percentage:.2f}%)")
    
    total_incomplete = len(incomplete_items)
    complete_percentage = (num_items - total_incomplete) / num_items * 100
    print(f"完全完整的商品数量: {num_items - total_incomplete}/{num_items} ({complete_percentage:.2f}%)")
    
    # 找出第一个有残缺标签的索引
    first_incomplete_idx = None
    if incomplete_items:
        first_incomplete_idx = incomplete_items[0]
        print(f"\n第一个有残缺标签的商品索引: {first_incomplete_idx}")
        
        # 打印这个商品的信息
        tags = item_data['tags'][first_incomplete_idx]
        tags_indices = item_data['tags_indices'][first_incomplete_idx]
        
        print("\n该商品的标签信息:")
        for level in range(num_levels):
            tag_status = "有效" if tags_indices[level] != -1 else "缺失"
            tag_text = tags[level] if tags_indices[level] != -1 else "无"
            print(f"  第{level+1}层: {tag_text} (ID: {tags_indices[level].item()}, 状态: {tag_status})")
        
        # 如果提供了保存路径，给出继续补全的命令
        if save_path:
            print(f"\n要继续补全标签，请使用以下命令:")
            print(f"python -m data.fill_kuairand_simple --data_path {save_path} --start_idx {first_incomplete_idx}")
    else:
        print("\n所有商品的标签都已完整！")
    
    return first_incomplete_idx


def fill_empty_titles(data, save_path=None):
    """
    为空标题的视频填充标题，使用已有的标签作为新标题
    
    参数:
        data: 数据对象
        save_path: 保存补全后数据的路径
    
    返回:
        更新后的数据对象
    """
    print("\n===== 开始填充空标题 =====")
    
    # 创建数据的副本，避免修改原始数据
    new_data = deepcopy(data)
    item_data = new_data['item']
    
    if 'text' not in item_data or item_data['text'] is None:
        print("数据中没有文本字段，无法填充标题")
        return new_data
    
    # 找出空标题的视频
    empty_title_items = []
    for i in range(len(item_data['text'])):
        if not item_data['text'][i] or item_data['text'][i].strip() == '':
            empty_title_items.append(i)
    
    print(f"找到 {len(empty_title_items)} 个空标题的视频")
    
    if not empty_title_items:
        print("没有空标题需要填充")
        return new_data
    
    # 填充空标题
    filled_count = 0
    for idx in tqdm(empty_title_items, desc="填充空标题"):
        # 获取视频的所有标签
        if 'tags' in item_data and idx < len(item_data['tags']):
            tags = item_data['tags'][idx]
            
            # 过滤掉空标签
            valid_tags = [tag for tag in tags if tag and tag.strip() != '']
            
            if valid_tags:
                # 使用标签组合作为新标题
                new_title = " ".join(valid_tags)
                item_data['text'][idx] = new_title
                filled_count += 1
    
    print(f"\n成功填充 {filled_count} 个空标题")
    
    # 保存更新后的数据
    if save_path:
        torch.save(new_data, save_path)
        print(f"已保存更新后的数据到: {save_path}")
    
    return new_data


def main():
    """主函数"""
    # 设置默认数据路径
    default_data_path = os.path.join('dataset', 'kuairand', 'processed', 'title_data_kuairand_5tags.pt')
    default_save_path = os.path.join('dataset', 'kuairand', 'processed', 'title_data_kuairand_5tags_completed.pt')
    
    # 获取命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='KuaiRand 数据集简单标签补全工具')
    parser.add_argument('--data_path', type=str, default=default_data_path,
                        help=f'KuaiRand 数据集文件路径 (默认: {default_data_path})')
    # parser.add_argument('--data_path', type=str, default=default_save_path,
    #                     help=f'KuaiRand 数据集文件路径 (默认: {default_data_path})')
    parser.add_argument('--save_path', type=str, default=default_save_path,
                        help=f'保存补全后数据的路径 (默认: {default_save_path})')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='批量处理的批次大小 (默认: 100)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='起始处理的商品索引，用于断点续传 (默认: 0)')
    parser.add_argument('--check_progress', action='store_true',
                        help='检查标签补全进度，找出第一个有残缺标签的索引')
    parser.add_argument('--view_tags', action='store_true',
                        help='随机抽取视频信息并打印标签')
    parser.add_argument('--analyze_tags', action='store_true',
                        help='分析标签分布情况')
    parser.add_argument('--fill_titles', action='store_true',
                        help='仅填充空标题，不进行标签补全')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='随机抽取的样本数量 (默认: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 加载数据集
    print(f"正在加载数据集: {args.data_path}")
    data = load_kuairand_dataset(args.data_path, args.seed)
    
    if args.view_tags:
        # 查看标签信息
        view_tags(data, args.num_samples, args.seed)
    elif args.analyze_tags:
        # 分析标签分布
        analyze_tag_distribution(data)
    elif args.check_progress:
        # 检查标签补全进度
        first_incomplete_idx = check_completion_progress(data, args.save_path)
        if first_incomplete_idx is not None:
            print(f"\n要继续补全标签，请使用以下命令:")
            print(f"python -m data.fill_kuairand_simple --data_path {args.save_path} --start_idx {first_incomplete_idx}")
    elif args.fill_titles:
        # 仅填充空标题
        fill_empty_titles(data, args.save_path)
    else:
        # 创建标签池
        tag_pools = create_tag_pools(data)
        
        # 构建标签层级关系
        tag_hierarchy = build_tag_hierarchy(data)
        
        # 简单批量补全标签并保存
        print(f"将使用层级约束的相似度补全方法保存数据到: {args.save_path}")
        simple_complete_tags(
            data, 
            tag_pools, 
            tag_hierarchy,
            args.batch_size, 
            args.save_path, 
            args.start_idx
        )
    
    print("\n处理完成！")


if __name__ == "__main__":
    main() 

    # 1. 查看标签分布情况：
    # ```
    # python -m data.fill_kuairand_simple --analyze_tags
    # ```

    # 2. 随机抽取视频信息并查看标签：
    # ```
    # python -m data.fill_kuairand_simple --view_tags --num_samples 20
    # ```

    # 3. 检查标签补全进度：
    # ```
    # python -m data.fill_kuairand_simple --check_progress
    # ```

    # 4. 执行标签补全：
    # ```
    # python -m data.fill_kuairand_simple

    # 仅填充空标题
    # python -m data.fill_kuairand_simple --fill_titles
