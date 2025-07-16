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
    加载KuaiRand数据集并返回HeteroData对象
    
    参数:
        data_path: 数据文件路径
        seed: 随机种子，用于确保可重现性
    
    返回:
        HeteroData对象
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
        data = torch.load(data_path, map_location='cpu', weights_only=False)
        print("✓ 数据加载成功!")
        return data
    except Exception as e:
        print(f"加载数据时出错: {e}")
        # 尝试使用旧版本的加载方式
        try:
            print("尝试使用替代方法加载...")
            # 添加numpy._core.multiarray._reconstruct到安全全局变量
            import numpy
            torch.serialization.add_safe_globals(['_reconstruct'])
            data = torch.load(data_path, map_location='cpu', weights_only=False)
            print("✓ 使用替代方法加载成功!")
            return data
        except Exception as e2:
            print(f"替代方法也失败: {e2}")
            raise


def print_dataset_structure(data):
    """
    打印数据集的结构信息
    
    参数:
        data: HeteroData对象
    """
    print("\n===== 数据集结构 =====")
    
    # 遍历所有节点类型
    for node_type in data.node_types:
        print(f"\n节点类型: {node_type}")
        node_data = data[node_type]
        
        for key, value in node_data.items():
            if isinstance(value, torch.Tensor):
                shape_str = list(value.shape)
                print(f"  - {key}: {shape_str}, 类型: {value.dtype}")
            elif isinstance(value, list):
                print(f"  - {key}: 列表, 长度: {len(value)}")
            elif isinstance(value, dict):
                print(f"  - {key}: 字典, 键: {list(value.keys())}")
            elif isinstance(value, bool):
                print(f"  - {key}: 布尔值, 值: {value}")
            else:
                print(f"  - {key}: {type(value)}")
    
    # 遍历所有边类型
    for edge_type in data.edge_types:
        print(f"\n边类型: {edge_type}")
        edge_data = data[edge_type[0], edge_type[1], edge_type[2]]
        
        for key, value in edge_data.items():
            if isinstance(value, torch.Tensor):
                shape_str = list(value.shape)
                print(f"  - {key}: {shape_str}, 类型: {value.dtype}")
            elif isinstance(value, dict):
                print(f"  - {key}: 字典")
                for k, v in value.items():
                    print(f"    - {k}:")
                    for subk, subv in v.items():
                        if isinstance(subv, torch.Tensor):
                            shape_str = list(subv.shape)
                            print(f"      - {subk}: {shape_str}, 类型: {subv.dtype}")
                        elif isinstance(subv, list):
                            print(f"      - {subk}: 列表, 长度: {len(subv)}")
                        else:
                            print(f"      - {subk}: {type(subv)}")
            else:
                print(f"  - {key}: {type(value)}")


def sample_and_print_values(data, num_samples=2):
    """
    随机抽样并打印各个属性的具体值
    
    参数:
        data: HeteroData对象
        num_samples: 每个属性要抽样的数量
    """
    print("\n===== 随机抽样属性值 =====")
    
    # 抽样item节点的属性
    if 'item' in data.node_types:
        print("\n商品(item)节点属性样本:")
        item_data = data['item']
        num_items = item_data.x.shape[0]
        
        # 随机选择样本索引
        sample_indices = random.sample(range(num_items), min(num_samples, num_items))
        
        for i, idx in enumerate(sample_indices):
            print(f"\n样本 {i+1} (索引 {idx}):")
            
            # 打印文本特征
            if hasattr(item_data, 'text') and item_data.text is not None:
                text = item_data.text[idx] if idx < len(item_data.text) else "不可用"
                print(f"  - 文本: {text}")
                
            # 打印特征向量
            if hasattr(item_data, 'x') and item_data.x is not None:
                features = item_data.x[idx]
                print(f"  - 特征前5维: {features[:5].tolist()}")
                
            # 打印标签文本
            if hasattr(item_data, 'tags') and item_data.tags is not None:
                tags = item_data.tags[idx]
                print(f"  - 标签文本: {tags.tolist() if hasattr(tags, 'tolist') else tags}")
                
            # 打印标签索引
            if hasattr(item_data, 'tags_indices') and item_data.tags_indices is not None:
                tag_indices = item_data.tags_indices[idx]
                print(f"  - 标签索引: {tag_indices.tolist()}")
                
            # 打印标签嵌入
            if hasattr(item_data, 'tags_emb') and item_data.tags_emb is not None:
                tag_emb = item_data.tags_emb[idx]
                if len(tag_emb.shape) > 1:
                    for j in range(tag_emb.shape[0]):
                        print(f"  - 第{j+1}级标签嵌入前3维: {tag_emb[j][:3].tolist()}")
                else:
                    print(f"  - 标签嵌入前3维: {tag_emb[:3].tolist()}")
            
            # 单独处理三个层级的标签嵌入
            for l in range(1, 4):
                if hasattr(item_data, f'tags_emb_l{l}') and getattr(item_data, f'tags_emb_l{l}') is not None:
                    tag_emb = getattr(item_data, f'tags_emb_l{l}')[idx]
                    print(f"  - 第{l}级标签嵌入前3维: {tag_emb[:3].tolist()}")
            
            # 打印训练/测试分割信息
            if hasattr(item_data, 'is_train') and item_data.is_train is not None:
                is_train = item_data.is_train[idx].item()
                print(f"  - 是否为训练集: {is_train}")
    
    # 抽样用户-商品交互序列
    if ('user', 'rated', 'item') in data.edge_types:
        print("\n用户-商品交互序列样本:")
        
        history = data['user', 'rated', 'item'].history
        
        # 随机抽样训练集序列
        if 'train' in history and 'itemId' in history['train']:
            train_sequences = history['train']['itemId']
            train_targets = history['train']['itemId_fut']
            
            num_train_seqs = len(train_sequences)
            train_sample_indices = random.sample(range(num_train_seqs), min(num_samples, num_train_seqs))
            
            for i, idx in enumerate(train_sample_indices):
                print(f"\n训练样本 {i+1} (索引 {idx}):")
                
                if isinstance(train_sequences, list):
                    sequence = train_sequences[idx]
                    # 过滤掉填充的-1
                    sequence = [item for item in sequence if item != -1]
                    print(f"  - 输入序列: {sequence}")
                else:
                    sequence = train_sequences[idx]
                    # 过滤掉填充的-1
                    sequence = sequence[sequence != -1].tolist()
                    print(f"  - 输入序列: {sequence}")
                
                target = train_targets[idx].item() if isinstance(train_targets[idx], torch.Tensor) else train_targets[idx]
                print(f"  - 目标商品: {target}")
        
        # 随机抽样验证集序列
        if 'eval' in history and 'itemId' in history['eval']:
            eval_sequences = history['eval']['itemId']
            eval_targets = history['eval']['itemId_fut']
            
            num_eval_seqs = len(eval_sequences)
            eval_sample_indices = random.sample(range(num_eval_seqs), min(num_samples, num_eval_seqs))
            
            for i, idx in enumerate(eval_sample_indices):
                print(f"\n验证样本 {i+1} (索引 {idx}):")
                
                sequence = eval_sequences[idx]
                # 过滤掉填充的-1
                sequence = sequence[sequence != -1].tolist()
                print(f"  - 输入序列: {sequence}")
                
                target = eval_targets[idx].item() if isinstance(eval_targets[idx], torch.Tensor) else eval_targets[idx]
                print(f"  - 目标商品: {target}")


def analyze_tag_completeness(data):
    """
    分析标签的完整度统计
    
    参数:
        data: HeteroData对象
    """
    print("\n===== 标签完整度分析 =====")
    
    if 'item' not in data.node_types or not hasattr(data['item'], 'tags_indices'):
        print("错误: 数据集中没有标签信息或标签索引不可用")
        return
    
    tags_indices = data['item'].tags_indices
    num_items = tags_indices.shape[0]
    num_levels = tags_indices.shape[1]
    
    print(f"标签共有 {num_levels} 个层级，商品总数: {num_items}")
    
    # 统计每个层级的标签存在情况
    level_stats = []
    for level in range(num_levels):
        # 有效标签的数量 (索引不为-1)
        valid_count = (tags_indices[:, level] != -1).sum().item()
        coverage = valid_count / num_items * 100
        level_stats.append((valid_count, coverage))
        
        print(f"第 {level+1} 级标签:")
        print(f"  - 有效标签数: {valid_count} / {num_items} ({coverage:.2f}%)")
        print(f"  - 缺失标签数: {num_items - valid_count} ({100 - coverage:.2f}%)")
    
    # 统计多级标签的完整度
    tag_completeness = torch.sum(tags_indices != -1, dim=1)
    completeness_counter = Counter(tag_completeness.tolist())
    
    print("\n多级标签完整度统计:")
    for level_count in range(num_levels + 1):
        count = completeness_counter.get(level_count, 0)
        percentage = (count / num_items) * 100
        print(f"  - 有 {level_count} 级标签的商品数: {count} ({percentage:.2f}%)")
    
    # 分析标签组合模式
    print("\n标签组合模式分析:")
    # 创建一个二进制掩码来表示标签存在的模式
    # 例如，[1,0,1]表示第1级和第3级标签存在，第2级标签缺失
    patterns = []
    for i in range(num_items):
        pattern = tuple((tags_indices[i] != -1).tolist())
        patterns.append(pattern)
    
    pattern_counter = Counter(patterns)
    
    for pattern, count in pattern_counter.most_common():
        percentage = (count / num_items) * 100
        pattern_str = ['有' if p else '无' for p in pattern]
        print(f"  - 模式 [{', '.join(pattern_str)}]: {count} 个商品 ({percentage:.2f}%)")
    
    # 绘制标签完整度可视化
    plt.figure(figsize=(10, 6))
    levels = [f"第{i+1}级" for i in range(num_levels)]
    coverages = [stats[1] for stats in level_stats]
    
    plt.bar(levels, coverages, color='skyblue')
    plt.ylim(0, 100)
    plt.xlabel('标签层级')
    plt.ylabel('覆盖率 (%)')
    plt.title('KuaiRand 数据集标签层级覆盖率')
    
    # 添加数值标签
    for i, coverage in enumerate(coverages):
        plt.text(i, coverage + 1, f'{coverage:.1f}%', ha='center')
    
    # 保存图表
    save_dir = 'plots'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kuairand_tag_coverage.png'), dpi=300)
    print(f"\n已保存标签覆盖率图表到: {os.path.join(save_dir, 'kuairand_tag_coverage.png')}")
    plt.close()
    
    # 绘制标签完整度分布
    plt.figure(figsize=(10, 6))
    counts = [completeness_counter.get(i, 0) for i in range(num_levels + 1)]
    percentages = [(count / num_items) * 100 for count in counts]
    
    plt.bar(range(num_levels + 1), percentages, color='lightgreen')
    plt.xticks(range(num_levels + 1))
    plt.xlabel('标签层级数量')
    plt.ylabel('商品比例 (%)')
    plt.title('KuaiRand 数据集商品标签完整度分布')
    
    # 添加数值标签
    for i, percentage in enumerate(percentages):
        if percentage > 0:
            plt.text(i, percentage + 0.5, f'{percentage:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kuairand_tag_completeness.png'), dpi=300)
    print(f"已保存标签完整度分布图表到: {os.path.join(save_dir, 'kuairand_tag_completeness.png')}")
    plt.close()


def create_tag_pools(data):
    """
    为每个层级创建标签池，包括标签文本和对应的嵌入向量
    
    参数:
        data: HeteroData对象
    
    返回:
        三个层级的标签池，每个池包含标签文本和对应的嵌入向量
    """
    print("正在创建标签池...")
    
    if 'item' not in data.node_types or not hasattr(data['item'], 'tags_indices'):
        raise ValueError("数据集中没有标签信息或标签索引不可用")
    
    item_data = data['item']
    tags_indices = item_data.tags_indices
    
    # 获取标签文本数据
    if not hasattr(item_data, 'tags') or item_data.tags is None:
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
            tag_text = item_data.tags[sample_item, level]
            
            # 获取标签嵌入
            if hasattr(item_data, f'tags_emb_l{level+1}'):
                # 获取所有具有该标签的商品的嵌入向量，并取平均
                tag_embs = getattr(item_data, f'tags_emb_l{level+1}')[items_with_tag]
                tag_emb = torch.mean(tag_embs, dim=0)
            else:
                # 如果没有专门的标签嵌入，使用样本商品的特征向量
                tag_emb = item_data.x[sample_item]
            
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

def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = torch.sum(vec1 * vec2)
    norm1 = torch.sqrt(torch.sum(vec1 * vec1))
    norm2 = torch.sqrt(torch.sum(vec2 * vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def retrieve_candidate_tags(title_embedding, tag_pool, top_k=6):
    """
    根据标题嵌入检索最相似的标签候选项
    
    参数:
        title_embedding: 标题的嵌入向量
        tag_pool: 标签池，包含标签ID、文本和嵌入
        top_k: 返回的候选项数量
    
    返回:
        top_k个最相似的标签候选项
    """
    similarities = []
    
    for tag_id, tag_info in tag_pool.items():
        tag_embedding = tag_info['embedding']
        similarity = cosine_similarity(title_embedding, tag_embedding)
        similarities.append((tag_id, tag_info['text'], similarity.item(), tag_info['count']))
    
    # 按相似度降序排序
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    # 返回top_k个候选项
    return similarities[:top_k]

def generate_optimized_prompt(data, item_idx, tag_pools, top_k=6):
    """
    根据视频标题和已有标签，生成结构化的提示词，方便从LLM回复中提取标签
    
    参数:
        data: HeteroData对象
        item_idx: 商品索引
        tag_pools: 三个层级的标签池
        top_k: 为每个缺失层级检索的候选项数量
    
    返回:
        提示词字符串
    """
    item_data = data['item']
    
    # 获取视频标题
    title = item_data.text[item_idx] if hasattr(item_data, 'text') and item_data.text is not None else "未知标题"
    
    # 获取视频标签
    tags = item_data.tags[item_idx]
    tags_indices = item_data.tags_indices[item_idx]
    
    # 获取视频特征向量
    item_embedding = item_data.x[item_idx]
    
    # 检查每个层级是否缺失标签
    missing_levels = []
    for level in range(len(tags)):
        if tags_indices[level] == -1 or tags[level] == '':
            missing_levels.append(level)
    
    # 如果没有缺失标签，不需要生成提示词
    if not missing_levels:
        return f"视频「{title}」已有完整的三层标签：{tags[0]}、{tags[1]}、{tags[2]}，无需补全。"
    
    # 构建更结构化的提示词，要求LLM以JSON格式返回
    prompt = f"""请帮我为一个短视频选择最合适的分类标签。

## 视频信息
- 标题: {title}

## 当前标签情况
"""
    
    for level in range(len(tags)):
        level_name = f"第{level+1}级"
        tag_text = tags[level] if tags[level] != '' else "缺失"
        prompt += f"- {level_name}标签: {tag_text}\n"
    
    prompt += "\n## 任务说明\n"
    prompt += "请为缺失的标签层级从下面的候选标签中选择一个最合适的标签。需要根据视频标题和已有标签来判断视频内容，并从候选标签中选择最相关的一个。\n\n"
    
    # 为每个缺失的层级检索候选项
    for level in missing_levels:
        candidates = retrieve_candidate_tags(item_embedding, tag_pools[level], top_k)
        
        prompt += f"## 第{level+1}级标签候选项\n"
        for i, (tag_id, tag_text, similarity, count) in enumerate(candidates):
            similarity_percent = similarity * 100
            # 不再在提示词中显示ID，让大模型只关注标签文本
            prompt += f"{i+1}. {tag_text} (相似度: {similarity_percent:.2f}%, 出现次数: {count})\n"
        prompt += "\n"
    
    # 修改格式要求，明确说明层级X是"层级0"、"层级1"或"层级2"
    prompt += """## 输出格式要求
请以JSON格式返回你的选择，必须包含以下字段:
```json
{
  "选择理由": "你为什么选择这些标签的简短解释",
  "补全标签": {
    "层级0": "选择的第1级标签名称",  
    "层级1": "选择的第2级标签名称",
    "层级2": "选择的第3级标签名称"
  }
}
```

请确保你的回答是有效的JSON格式，只填写缺失的标签层级（不要填写已有标签）。
第1级对应"层级0"，第2级对应"层级1"，第3级对应"层级2"。
例如，如果缺少第3级标签，你只需要在补全标签中包含"层级2"。
"""
    
    return prompt

def call_llm_for_tag_completion(prompt, temperature=0.3):
    """调用LLM获取标签补全结果"""
    system_prompt = "你是一个专业的视频标签分类助手，擅长根据视频标题和已有标签为视频选择最合适的分类标签。你的回答必须严格按照要求的JSON格式返回。"
    
    response = query_llm(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=800
    )
    
    # 解析响应
    parsed = parse_llm_response(response, expected_format="json")
    return response, parsed

def find_tag_id_by_name(tag_pools, level, tag_name):
    """
    根据标签名称在标签池中查找对应的ID
    
    参数:
        tag_pools: 标签池
        level: 标签层级
        tag_name: 标签名称
    
    返回:
        标签ID和对应的embedding，如果未找到则返回None
    """
    for tag_id, tag_info in tag_pools[level].items():
        if tag_info['text'] == tag_name:
            return tag_id, tag_info['embedding']
    return None, None

def complete_tags_with_llm(data, item_idx, tag_pools, max_retries=2):
    """
    使用LLM补全缺失的标签
    
    参数:
        data: HeteroData对象
        item_idx: 商品索引
        tag_pools: 三个层级的标签池
        max_retries: 最大重试次数
    
    返回:
        补全结果字典，包含补全的标签、ID和嵌入向量，以及原始响应
    """
    # 获取视频标签状态
    item_data = data['item']
    tags = item_data.tags[item_idx]
    tags_indices = item_data.tags_indices[item_idx]
    
    # 检查是否有缺失标签
    missing_levels = []
    for level in range(len(tags)):
        if tags_indices[level] == -1 or tags[level] == '':
            missing_levels.append(level)
    
    if not missing_levels:
        return {"status": "complete", "message": "所有标签已完整，无需补全"}
    
    # 生成提示词
    prompt = generate_optimized_prompt(data, item_idx, tag_pools)
    
    # 带重试机制的LLM调用
    retry_count = 0
    while retry_count <= max_retries:
        try:
            # 调用LLM
            raw_response, llm_result = call_llm_for_tag_completion(prompt)
            
            if not llm_result["success"]:
                if retry_count < max_retries:
                    retry_count += 1
                    continue
                return {
                    "status": "error", 
                    "message": f"LLM调用失败: {llm_result.get('error', '未知错误')}",
                    "raw_response": raw_response
                }
            
            # 提取补全标签
            try:
                parsed_data = llm_result["parsed_data"]
                completion_result = {
                    "status": "success",
                    "选择理由": parsed_data.get("选择理由", "无理由提供"),
                    "补全标签": {},
                    "raw_response": raw_response,
                    "parsed_response": llm_result
                }
                
                # 提取每个层级的补全标签
                completed_tags = parsed_data.get("补全标签", {})
                
                # 处理多种可能的键格式："层级X"、"X"、数字X
                for level in missing_levels:
                    # 尝试多种可能的键格式
                    level_keys = [f"层级{level}", str(level), level]
                    found_key = None
                    
                    # 检查每种可能的键格式
                    for key in level_keys:
                        if key in completed_tags:
                            found_key = key
                            break
                    
                    if found_key is not None:
                        tag_name = completed_tags[found_key]
                        # 如果值是字典，尝试获取其中的name字段
                        if isinstance(tag_name, dict) and "name" in tag_name:
                            tag_name = tag_name["name"]
                            
                        # 从标签池中查找对应的ID和embedding
                        tag_id, tag_embedding = find_tag_id_by_name(tag_pools, level, tag_name)
                        
                        if tag_id is not None:
                            completion_result["补全标签"][level] = {
                                "id": tag_id,
                                "name": tag_name,
                                "embedding": tag_embedding
                            }
                        else:
                            # 如果在标签池中找不到对应的标签
                            completion_result["补全标签"][level] = {
                                "id": None,
                                "name": tag_name,
                                "embedding": None,
                                "error": "无法在标签池中找到对应标签"
                            }
                
                # 检查是否有效地补全了至少一个标签
                if not completion_result["补全标签"] and retry_count < max_retries:
                    retry_count += 1
                    continue
                    
                return completion_result
            except Exception as e:
                if retry_count < max_retries:
                    retry_count += 1
                    continue
                return {
                    "status": "error", 
                    "message": f"解析LLM响应失败: {str(e)}",
                    "raw_response": raw_response,
                    "parsed_response": llm_result
                }
        except Exception as e:
            if retry_count < max_retries:
                retry_count += 1
                continue
            return {
                "status": "error",
                "message": f"发生异常: {str(e)}"
            }

def analyze_and_complete_tags(data, tag_pools, num_samples=5, seed=None):
    """
    为随机样本分析当前标签并使用LLM补全缺失标签
    
    参数:
        data: HeteroData对象
        tag_pools: 三个层级的标签池
        num_samples: 样本数量
        seed: 随机种子，用于确保可重现性
    """
    if seed is not None:
        random.seed(seed)
    
    item_data = data['item']
    num_items = item_data.x.shape[0]
    
    # 找出缺少至少一层标签的商品
    incomplete_items = []
    for i in range(num_items):
        tags_indices = item_data.tags_indices[i]
        if torch.any(tags_indices == -1):
            incomplete_items.append(i)
    
    print(f"找到 {len(incomplete_items)} 个缺少标签的商品")
    
    # 随机选择样本
    sample_indices = random.sample(incomplete_items, min(num_samples, len(incomplete_items)))
    
    # 为每个样本分析并补全标签
    for i, idx in enumerate(sample_indices):
        print(f"\n\n===== 样本 {i+1} =====")
        
        # 打印原始数据
        title = item_data.text[idx] if hasattr(item_data, 'text') and item_data.text is not None else "未知标题"
        tags = item_data.tags[idx]
        tags_indices = item_data.tags_indices[idx]
        
        print(f"视频索引: {idx}")
        print(f"视频标题: {title}")
        print(f"当前标签: {tags.tolist() if hasattr(tags, 'tolist') else tags}")
        print(f"标签索引: {tags_indices.tolist()}")
        
        # 生成提示词
        prompt = generate_optimized_prompt(data, idx, tag_pools)
        print("\n----- 生成的提示词 -----")
        print(prompt)
        
        print("\n----- 调用LLM补全标签 -----")
        completion_result = complete_tags_with_llm(data, idx, tag_pools)
        
        # 显示原始LLM响应
        print("\n大模型原始回复:")
        if "raw_response" in completion_result and completion_result["raw_response"] and "text" in completion_result["raw_response"]:
            print(completion_result["raw_response"]["text"])
        else:
            print("未能获取原始回复")
        
        # 显示解析后的结果
        print("\n解析后的结果:")
        # 创建一个简化版的结果，去掉embedding等大对象以便于打印
        simplified_result = {k: v for k, v in completion_result.items() if k not in ["raw_response", "parsed_response"]}
        
        # 打印键值对映射情况，帮助理解解析过程
        if "parsed_response" in completion_result and completion_result["parsed_response"].get("success", False):
            completed_tags = completion_result["parsed_response"]["parsed_data"].get("补全标签", {})
            if completed_tags:
                print("\n解析的键值映射:")
                for key, value in completed_tags.items():
                    print(f"  - 键: '{key}' -> 值: '{value}'")
        
        # 打印解析后的结果
        if "补全标签" in simplified_result:
            for level, tag_info in simplified_result["补全标签"].items():
                if "embedding" in tag_info:
                    # 只保留embedding的形状信息
                    embedding = tag_info["embedding"]
                    if embedding is not None:
                        tag_info["embedding"] = f"Tensor(shape={list(embedding.shape)})"
                    else:
                        tag_info["embedding"] = None
        
        print(json.dumps(simplified_result, ensure_ascii=False, indent=2))
        
        # 如果成功补全，显示补全后的标签状态
        if completion_result["status"] == "success":
            print("\n----- 补全后的标签 -----")
            completed_tags = list(tags)
            completed_indices = list(tags_indices.tolist())
            
            for level, tag_info in completion_result["补全标签"].items():
                level = int(level)
                if tag_info["id"] is not None:
                    completed_tags[level] = tag_info["name"]
                    completed_indices[level] = tag_info["id"]
            
            print(f"补全后标签: {completed_tags}")
            print(f"补全后索引: {completed_indices}")
        
        # 添加延迟，避免API调用过于频繁
        if i < num_samples - 1:
            print("等待2秒后继续...")
            time.sleep(2)

def generate_prompts_for_samples(data, tag_pools, num_samples=5, seed=None):
    """
    为随机样本生成补全标签的提示词
    
    参数:
        data: HeteroData对象
        tag_pools: 三个层级的标签池
        num_samples: 样本数量
        seed: 随机种子，用于确保可重现性
    """
    if seed is not None:
        random.seed(seed)
    
    item_data = data['item']
    num_items = item_data.x.shape[0]
    
    # 找出缺少至少一层标签的商品
    incomplete_items = []
    for i in range(num_items):
        tags_indices = item_data.tags_indices[i]
        if torch.any(tags_indices == -1):
            incomplete_items.append(i)
    
    print(f"找到 {len(incomplete_items)} 个缺少标签的商品")
    
    # 随机选择样本
    sample_indices = random.sample(incomplete_items, min(num_samples, len(incomplete_items)))
    
    # 为每个样本生成提示词
    for i, idx in enumerate(sample_indices):
        print(f"\n\n===== 样本 {i+1} =====")
        
        # 打印原始数据
        title = item_data.text[idx] if hasattr(item_data, 'text') and item_data.text is not None else "未知标题"
        tags = item_data.tags[idx]
        tags_indices = item_data.tags_indices[idx]
        
        print(f"视频索引: {idx}")
        print(f"视频标题: {title}")
        print(f"当前标签: {tags.tolist() if hasattr(tags, 'tolist') else tags}")
        print(f"标签索引: {tags_indices.tolist()}")
        
        # 生成提示词
        prompt = generate_optimized_prompt(data, idx, tag_pools)
        print("\n----- 生成的提示词 -----")
        print(prompt)

def batch_complete_tags(data, tag_pools, batch_size=100, save_path=None, start_idx=0):
    """
    批量补全标签并保存到新的数据文件
    
    参数:
        data: HeteroData对象
        tag_pools: 三个层级的标签池
        batch_size: 每批处理的商品数量
        save_path: 保存补全后数据的路径
        start_idx: 起始索引，用于断点续传
    
    返回:
        更新后的HeteroData对象
    """
    # 创建数据的副本，避免修改原始数据
    from copy import deepcopy
    new_data = deepcopy(data)
    
    item_data = new_data['item']
    num_items = item_data.x.shape[0]
    
    # 找出缺少至少一层标签的商品
    incomplete_items = []
    for i in range(num_items):
        tags_indices = item_data.tags_indices[i]
        if torch.any(tags_indices == -1):
            incomplete_items.append(i)
    
    # 过滤掉起始索引之前的项
    incomplete_items = [idx for idx in incomplete_items if idx >= start_idx]
    
    total_incomplete = len(incomplete_items)
    print(f"找到 {total_incomplete} 个缺失标签的商品，从索引 {start_idx} 开始处理")
    
    # 为每个批次创建进度条
    num_batches = (total_incomplete + batch_size - 1) // batch_size
    
    # 用于保存每批次结果的统计信息
    stats = {
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "errors": []
    }
    
    # 批量处理
    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min((batch_num + 1) * batch_size, total_incomplete)
        batch_indices = incomplete_items[batch_start:batch_end]
        
        print(f"\n处理批次 {batch_num + 1}/{num_batches} (商品索引 {batch_indices[0]} 至 {batch_indices[-1]})")
        
        # 遍历批次中的每个商品
        for i, idx in enumerate(tqdm(batch_indices, desc=f"批次 {batch_num+1}")):
            # 检查是否仍然需要补全
            tags_indices = item_data.tags_indices[idx]
            missing_levels = [level for level in range(tags_indices.shape[0]) if tags_indices[level] == -1]
            
            if not missing_levels:
                stats["skipped"] += 1
                continue
            
            # 补全标签
            completion_result = complete_tags_with_llm(new_data, idx, tag_pools)
            stats["processed"] += 1
            
            # 如果补全成功，更新数据
            if completion_result["status"] == "success" and "补全标签" in completion_result:
                # 更新标签
                for level, tag_info in completion_result["补全标签"].items():
                    level = int(level)
                    if tag_info["id"] is not None:
                        # 更新标签文本
                        item_data.tags[idx, level] = tag_info["name"]
                        # 更新标签索引
                        item_data.tags_indices[idx, level] = tag_info["id"]
                        # 更新标签嵌入（如果可用）
                        if tag_info["embedding"] is not None and hasattr(item_data, f'tags_emb_l{level+1}'):
                            getattr(item_data, f'tags_emb_l{level+1}')[idx] = tag_info["embedding"]
                
                stats["successful"] += 1
            else:
                stats["failed"] += 1
                error_msg = completion_result.get("message", "未知错误")
                stats["errors"].append((idx, error_msg))
                print(f"\n处理商品索引 {idx} 时失败: {error_msg}")
            
            # 每完成10个商品保存一次数据
            if i % 10 == 9 and save_path:
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
    
    if stats["errors"]:
        print(f"\n前10个错误:")
        for idx, error in stats["errors"][:10]:
            print(f"  - 商品索引 {idx}: {error}")
    
    return new_data

def parallel_complete_tags(data, tag_pools, batch_size=50, save_path=None, start_idx=0, max_workers=8, parallel_batch_size=10):
    """
    使用并行处理方式批量补全标签并保存到新的数据文件
    
    参数:
        data: HeteroData对象
        tag_pools: 三个层级的标签池
        batch_size: 每批处理的商品数量
        save_path: 保存补全后数据的路径
        start_idx: 起始索引，用于断点续传
        max_workers: 最大并行工作线程数
        parallel_batch_size: 并行批处理大小，一次并行处理多少个商品
    
    返回:
        更新后的HeteroData对象
    """
    # 创建数据的副本，避免修改原始数据
    from copy import deepcopy
    new_data = deepcopy(data)
    
    item_data = new_data['item']
    num_items = item_data.x.shape[0]
    
    # 找出缺少至少一层标签的商品
    incomplete_items = []
    for i in range(num_items):
        tags_indices = item_data.tags_indices[i]
        if torch.any(tags_indices == -1):
            incomplete_items.append(i)
    
    # 过滤掉起始索引之前的项
    incomplete_items = [idx for idx in incomplete_items if idx >= start_idx]
    
    total_incomplete = len(incomplete_items)
    print(f"找到 {total_incomplete} 个缺失标签的商品，从索引 {start_idx} 开始处理")
    
    # 为每个批次创建进度条
    num_batches = (total_incomplete + batch_size - 1) // batch_size
    
    # 用于保存每批次结果的统计信息
    stats = {
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "errors": []
    }
    
    # 批量处理
    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min((batch_num + 1) * batch_size, total_incomplete)
        batch_indices = incomplete_items[batch_start:batch_end]
        
        print(f"\n处理批次 {batch_num + 1}/{num_batches} (商品索引 {batch_indices[0]} 至 {batch_indices[-1]})")
        
        # 分割成多个并行批次
        for i in range(0, len(batch_indices), parallel_batch_size):
            parallel_indices = batch_indices[i:i+parallel_batch_size]
            pbar = tqdm(total=len(parallel_indices), desc=f"批次 {batch_num+1} 并行组 {i//parallel_batch_size+1}/{(len(batch_indices)+parallel_batch_size-1)//parallel_batch_size}")
            
            # 为每个并行批次准备提示词
            prompts_data = []
            for idx in parallel_indices:
                # 检查是否仍然需要补全
                tags_indices = item_data.tags_indices[idx]
                missing_levels = [level for level in range(tags_indices.shape[0]) if tags_indices[level] == -1]
                
                if not missing_levels:
                    stats["skipped"] += 1
                    pbar.update(1)
                    continue
                
                # 生成提示词
                prompt = generate_optimized_prompt(new_data, idx, tag_pools)
                system_prompt = "你是一个专业的视频标签分类助手，擅长根据视频标题和已有标签为视频选择最合适的分类标签。你的回答必须严格按照要求的JSON格式返回。"
                
                prompts_data.append((idx, prompt, system_prompt, missing_levels))
            
            # 如果没有需要处理的提示词，继续下一个批次
            if not prompts_data:
                continue
            
            # 提取提示词列表进行并行处理
            indices = [data[0] for data in prompts_data]
            prompts = [(data[1], data[2]) for data in prompts_data]
            missing_levels_map = {data[0]: data[3] for data in prompts_data}
            
            # 直接为每个请求分配不同的模型
            from data.chat_with_llm import AVAILABLE_MODELS
            models = [AVAILABLE_MODELS[j % len(AVAILABLE_MODELS)] for j in range(len(prompts))]
            
            # 创建并行任务
            def process_prompt(idx, prompt_tuple, model_name):
                prompt, system_prompt = prompt_tuple
                try:
                    # 使用指定模型
                    print(f"商品索引 {indices[idx]} 使用模型: {model_name}")
                    
                    # 调用LLM
                    result = query_llm(
                        prompt=prompt, 
                        system_prompt=system_prompt, 
                        model=model_name,
                        max_tokens=800, 
                        temperature=0.3, 
                        max_retries=3
                    )
                    
                    # 解析响应
                    parsed_result = parse_llm_response(result, expected_format="json")
                    return idx, result, parsed_result
                except Exception as e:
                    print(f"处理商品索引 {indices[idx]} 时出错: {str(e)}")
                    return idx, {"success": False, "error": str(e)}, {"success": False, "error": str(e)}
            
            # 并行执行任务
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_prompt, i, prompt_tuple, models[i]) 
                          for i, prompt_tuple in enumerate(prompts)]
                
                for future in as_completed(futures):
                    results.append(future.result())
            
            # 处理结果
            for idx, result, parsed_result in sorted(results, key=lambda x: x[0]):
                item_idx = indices[idx]
                stats["processed"] += 1
                
                # 构建完整的补全结果
                completion_result = {
                    "status": "success" if parsed_result.get("success", False) else "error",
                    "raw_response": result,
                    "parsed_response": parsed_result,
                    "补全标签": {}
                }
                
                if completion_result["status"] == "success":
                    # 提取补全标签
                    try:
                        parsed_data = parsed_result.get("parsed_data", {})
                        completion_result["选择理由"] = parsed_data.get("选择理由", "无理由提供")
                        
                        # 提取每个层级的补全标签
                        completed_tags = parsed_data.get("补全标签", {})
                        
                        # 处理多种可能的键格式："层级X"、"X"、数字X
                        for level in missing_levels_map[item_idx]:
                            # 尝试多种可能的键格式
                            level_keys = [f"层级{level}", str(level), level]
                            found_key = None
                            
                            # 检查每种可能的键格式
                            for key in level_keys:
                                if str(key) in completed_tags:
                                    found_key = key
                                    break
                            
                            if found_key is not None:
                                tag_name = completed_tags[str(found_key)]
                                # 如果值是字典，尝试获取其中的name字段
                                if isinstance(tag_name, dict) and "name" in tag_name:
                                    tag_name = tag_name["name"]
                                    
                                # 从标签池中查找对应的ID和embedding
                                tag_id, tag_embedding = find_tag_id_by_name(tag_pools, level, tag_name)
                                
                                if tag_id is not None:
                                    completion_result["补全标签"][level] = {
                                        "id": tag_id,
                                        "name": tag_name,
                                        "embedding": tag_embedding
                                    }
                                else:
                                    # 如果在标签池中找不到对应的标签
                                    completion_result["补全标签"][level] = {
                                        "id": None,
                                        "name": tag_name,
                                        "embedding": None,
                                        "error": "无法在标签池中找到对应标签"
                                    }
                        
                        # 如果有效地补全了至少一个标签
                        if completion_result["补全标签"]:
                            # 更新标签
                            for level, tag_info in completion_result["补全标签"].items():
                                level = int(level)
                                if tag_info["id"] is not None:
                                    # 更新标签文本
                                    item_data.tags[item_idx, level] = tag_info["name"]
                                    # 更新标签索引
                                    item_data.tags_indices[item_idx, level] = tag_info["id"]
                                    # 更新标签嵌入（如果可用）
                                    if tag_info["embedding"] is not None and hasattr(item_data, f'tags_emb_l{level+1}'):
                                        getattr(item_data, f'tags_emb_l{level+1}')[item_idx] = tag_info["embedding"]
                            
                            stats["successful"] += 1
                        else:
                            stats["failed"] += 1
                            stats["errors"].append((item_idx, "未能从标签池中找到任何匹配的标签"))
                    except Exception as e:
                        stats["failed"] += 1
                        error_msg = f"解析LLM响应失败: {str(e)}"
                        stats["errors"].append((item_idx, error_msg))
                else:
                    stats["failed"] += 1
                    error_msg = completion_result.get("message", result.get("error", "未知错误"))
                    stats["errors"].append((item_idx, error_msg))
                
                # 更新进度条
                pbar.update(1)
            
            pbar.close()
            
            # 每完成一个并行批次保存一次临时数据
            if save_path:
                temp_save_path = f"{save_path}_temp"
                torch.save(new_data, temp_save_path)
                print(f"\n已保存临时数据到 {temp_save_path}")
                
                # 打印模型使用统计
                model_stats = get_model_stats()
                print("\n模型使用统计:")
                for model, stat in sorted(model_stats.items(), key=lambda x: x[1]['usage'], reverse=True):
                    if stat['usage'] > 0:
                        print(f"{model}: 使用={stat['usage']}, 错误={stat['errors']}, 错误率={stat['error_rate']}")
        
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
    
    if stats["errors"]:
        print(f"\n前10个错误:")
        for idx, error in stats["errors"][:10]:
            print(f"  - 商品索引 {idx}: {error}")
    
    return new_data

def check_completion_progress(data, save_path=None):
    """
    检查数据集标签补全的进度，找出第一个有残缺标签的索引
    
    参数:
        data: HeteroData对象
        save_path: 保存检查结果的路径（可选）
    
    返回:
        第一个有残缺标签的索引
    """
    item_data = data['item']
    num_items = item_data.x.shape[0]
    
    print(f"\n===== 标签补全进度检查 =====")
    print(f"总商品数: {num_items}")
    
    # 找出第一个有残缺标签的商品索引
    first_incomplete_idx = None
    incomplete_count = 0
    
    for i in range(num_items):
        tags_indices = item_data.tags_indices[i]
        if torch.any(tags_indices == -1):
            if first_incomplete_idx is None:
                first_incomplete_idx = i
            incomplete_count += 1
    
    if first_incomplete_idx is not None:
        print(f"发现第一个残缺标签的商品索引: {first_incomplete_idx}")
        print(f"共有 {incomplete_count} 个商品标签不完整 ({incomplete_count/num_items*100:.2f}%)")
        print(f"已完成 {first_incomplete_idx} 个商品的标签补全 ({first_incomplete_idx/num_items*100:.2f}%)")
        
        # 打印第一个残缺标签商品前后的样本
        print("\n----- 第一个残缺标签商品前10个样本 -----")
        start_idx = max(0, first_incomplete_idx - 10)
        for i in range(start_idx, first_incomplete_idx):
            title = item_data.text[i] if hasattr(item_data, 'text') and item_data.text is not None else "未知标题"
            tags = item_data.tags[i]
            tags_indices = item_data.tags_indices[i]
            print(f"索引 {i}: 标题: {title[:30]}... | 标签: {tags.tolist()} | 标签索引: {tags_indices.tolist()}")
        
        print("\n----- 第一个残缺标签商品 -----")
        title = item_data.text[first_incomplete_idx] if hasattr(item_data, 'text') and item_data.text is not None else "未知标题"
        tags = item_data.tags[first_incomplete_idx]
        tags_indices = item_data.tags_indices[first_incomplete_idx]
        print(f"索引 {first_incomplete_idx}: 标题: {title} | 标签: {tags.tolist()} | 标签索引: {tags_indices.tolist()}")
        
        print("\n----- 第一个残缺标签商品后10个样本 -----")
        end_idx = min(num_items, first_incomplete_idx + 11)
        for i in range(first_incomplete_idx + 1, end_idx):
            title = item_data.text[i] if hasattr(item_data, 'text') and item_data.text is not None else "未知标题"
            tags = item_data.tags[i]
            tags_indices = item_data.tags_indices[i]
            print(f"索引 {i}: 标题: {title[:30]}... | 标签: {tags.tolist()} | 标签索引: {tags_indices.tolist()}")
    else:
        print("所有商品的标签都已补全完整！")
    
    return first_incomplete_idx

def main():
    """主函数"""
    # 设置默认数据路径
    default_data_path = os.path.join('dataset', 'kuairand', 'processed', 'kuairand_data_minimal_interactions30000.pt')
    
    # 获取命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='KuaiRand 数据集分析工具')
    parser.add_argument('--data_path', type=str, default=default_data_path,
                        help=f'KuaiRand 数据集文件路径 (默认: {default_data_path})')
    parser.add_argument('--samples', type=int, default=2,
                        help='每种属性要显示的随机样本数量 (默认: 2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--generate_prompts', action='store_true',
                        help='生成补全标签的提示词')
    parser.add_argument('--prompt_samples', type=int, default=5,
                        help='生成提示词的样本数量 (默认: 5)')
    parser.add_argument('--complete_tags', action='store_true',
                        help='使用LLM补全标签')
    parser.add_argument('--batch_complete', action='store_true',
                        help='批量补全所有缺失标签')
    parser.add_argument('--parallel_complete', action='store_true',
                        help='使用并行处理批量补全所有缺失标签')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='批量处理的批次大小 (默认: 100)')
    parser.add_argument('--parallel_batch_size', type=int, default=10,
                        help='并行批处理大小，一次并行处理多少个商品 (默认: 10)')
    parser.add_argument('--max_workers', type=int, default=8,
                        help='最大并行工作线程数 (默认: 8)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='保存补全后数据的路径')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='起始处理的商品索引，用于断点续传 (默认: 0)')
    parser.add_argument('--check_progress', action='store_true',
                        help='检查标签补全进度，找出第一个有残缺标签的索引')
    
    args = parser.parse_args()
    
    # 加载数据集
    data = load_kuairand_dataset(args.data_path, args.seed)
    
    if args.check_progress:
        # 检查标签补全进度
        first_incomplete_idx = check_completion_progress(data)
        if first_incomplete_idx is not None:
            print(f"\n要继续补全标签，请使用以下命令:")
            print(f"python -m data.fill_kuairand --parallel_complete --data_path {args.data_path} --start_idx {first_incomplete_idx}")
    elif args.parallel_complete:
        # 默认保存路径
        if not args.save_path:
            save_dir = os.path.dirname(args.data_path)
            filename = os.path.basename(args.data_path)
            name, ext = os.path.splitext(filename)
            args.save_path = os.path.join(save_dir, f"{name}_completed{ext}")
        
        print(f"将使用并行处理保存补全后的数据到: {args.save_path}")
        
        # 创建标签池
        tag_pools = create_tag_pools(data)
        
        # 并行批量补全标签并保存
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
        # 默认保存路径
        if not args.save_path:
            save_dir = os.path.dirname(args.data_path)
            filename = os.path.basename(args.data_path)
            name, ext = os.path.splitext(filename)
            args.save_path = os.path.join(save_dir, f"{name}_completed{ext}")
        
        print(f"将保存补全后的数据到: {args.save_path}")
        
        # 创建标签池
        tag_pools = create_tag_pools(data)
        
        # 批量补全标签并保存
        batch_complete_tags(data, tag_pools, args.batch_size, args.save_path, args.start_idx)
    elif args.complete_tags:
        # 创建标签池
        tag_pools = create_tag_pools(data)
        
        # 分析并补全标签
        analyze_and_complete_tags(data, tag_pools, args.prompt_samples, args.seed)
    elif args.generate_prompts:
        # 创建标签池
        tag_pools = create_tag_pools(data)
        
        # 生成提示词样本
        generate_prompts_for_samples(data, tag_pools, args.prompt_samples, args.seed)
    else:
        # 打印数据集结构
        print_dataset_structure(data)
        
        # 随机抽样并打印属性值
        sample_and_print_values(data, args.samples)
        
        # 分析标签完整度
        analyze_tag_completeness(data)
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()
