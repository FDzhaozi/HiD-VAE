import os
from openai import OpenAI
import json
from typing import Optional, Dict, Any, List, Tuple
import time
import threading
import queue
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# 可用模型列表
AVAILABLE_MODELS = [
    "qwen3-30b-a3b",
    "qwen3-14b",
    "qwen3-8b",
    "qwen3-4b",
    "qwen3-1.7b",
    "qwen-vl-max-2025-04-08",
    "qwen-vl-max-2025-01-25",
    "qwen-vl-max-2025-04-02",
    "qwen-vl-max-1230"
]

# 需要流式模式的模型
STREAM_MODELS = [
    "qwq-plus",
    "qwq-plus-latest",
    "qwq-plus-2025-03-05"
]

# 不支持HTTP调用的模型（不会被使用）
UNSUPPORTED_MODELS = [
    "qvq-max",
    "qvq-max-latest",
    "qvq-max-2025-05-15",
    "qvq-max-2025-03-25"
]

# 将流式模型添加到可用模型列表
AVAILABLE_MODELS.extend(STREAM_MODELS)

# 模型特殊配置
MODEL_CONFIG = {
    "qwen3-30b-a3b": {"enable_thinking": False},
    "qwen3-14b": {"enable_thinking": False},
    "qwen3-8b": {"enable_thinking": False},
    "qwen3-4b": {"enable_thinking": False},
    "qwen3-1.7b": {"enable_thinking": False},
    "qwen-vl-max-2025-04-08": {"enable_thinking": False},
    "qwen-vl-max-2025-01-25": {"enable_thinking": False},
    "qwen-vl-max-2025-04-02": {"enable_thinking": False},
    "qwen-vl-max-1230": {"enable_thinking": False},
    "qwq-plus": {"enable_thinking": False},
    "qwq-plus-latest": {"enable_thinking": False},
    "qwq-plus-2025-03-05": {"enable_thinking": False},
}

# 模型使用统计和限流控制
class ModelRateLimiter:
    def __init__(self, cooldown_period=2.0):
        self.model_last_used = {model: 0 for model in AVAILABLE_MODELS}
        self.model_usage_count = {model: 0 for model in AVAILABLE_MODELS}
        self.model_error_count = {model: 0 for model in AVAILABLE_MODELS}
        self.cooldown_period = cooldown_period
        self.lock = threading.Lock()
        self.round_robin_index = 0  # 添加轮询索引
    
    def get_available_model(self) -> str:
        """返回当前可用且负载最低的模型，确保所有模型都有机会被使用"""
        with self.lock:
            current_time = time.time()
            
            # 优先选择使用次数最少的模型（强制平衡）
            if random.random() < 0.7:  # 70%概率优先考虑使用次数最少的模型
                # 按使用次数排序
                models_by_usage = sorted(AVAILABLE_MODELS, key=lambda m: self.model_usage_count.get(m, 0))
                
                # 从使用次数最少的模型中选择一个随机模型（前3个中选择）
                candidates = models_by_usage[:min(3, len(models_by_usage))]
                return random.choice(candidates)
            
            # 30%概率使用轮询方式选择模型
            self.round_robin_index = (self.round_robin_index + 1) % len(AVAILABLE_MODELS)
            return AVAILABLE_MODELS[self.round_robin_index]
    
    def mark_model_used(self, model: str, success: bool = True):
        """标记模型已被使用"""
        with self.lock:
            self.model_last_used[model] = time.time()
            self.model_usage_count[model] = self.model_usage_count.get(model, 0) + 1
            if not success:
                self.model_error_count[model] = self.model_error_count.get(model, 0) + 1
    
    def get_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取模型使用统计"""
        with self.lock:
            stats = {}
            for model in AVAILABLE_MODELS:
                usage = self.model_usage_count.get(model, 0)
                errors = self.model_error_count.get(model, 0)
                error_rate = errors / max(usage, 1) * 100
                stats[model] = {
                    "usage": usage,
                    "errors": errors,
                    "error_rate": f"{error_rate:.2f}%",
                    "last_used": self.model_last_used.get(model, 0)
                }
            return stats

# 创建全局模型限流器实例
model_limiter = ModelRateLimiter()

def query_llm(
    prompt: str,
    system_prompt: str = "",
    model: str = None,  # 如果为None，会自动选择模型
    api_key: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    retry_delay: float = 3.0,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    调用大语言模型API获取响应
    
    参数:
        prompt: 用户提示词
        system_prompt: 系统提示词
        model: 使用的模型名称，如果为None则自动选择
        api_key: API密钥（如果为None，则使用环境变量或默认值）
        max_tokens: 生成的最大token数
        temperature: 生成温度，值越低越确定性
        retry_delay: API调用失败时的重试延迟(秒)
        max_retries: 最大重试次数
    
    返回:
        包含模型响应的字典
    """
    # 如果未提供模型，自动选择一个
    if model is None:
        model = model_limiter.get_available_model()
        print(f"自动选择模型: {model}")
    
    # 如果指定的模型不支持，则切换到支持的模型
    if model in UNSUPPORTED_MODELS:
        model = random.choice(AVAILABLE_MODELS)
        print(f"指定的模型不支持HTTP调用，切换到: {model}")
    
    # 如果未提供API密钥，使用默认值
    if api_key is None:
        api_key = "sk-20ce4658d02f4f10a8d1e3e3efc35ddf"  # 默认使用百炼API Key
    
    # 初始化客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 添加重试机制
    retries = 0
    used_models = set()  # 跟踪已使用过的模型
    while retries <= max_retries:
        try:
            # 调用API前标记模型使用
            model_limiter.mark_model_used(model)
            used_models.add(model)
            
            # 准备API调用参数
            api_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            # 添加模型特定的参数
            if model in MODEL_CONFIG:
                # 对于qwen模型，使用extra_body传递特殊参数
                extra_params = MODEL_CONFIG.get(model, {})
                if extra_params:
                    api_params["extra_body"] = extra_params
            
            # 对于需要流式模式的模型，启用stream
            if model in STREAM_MODELS:
                api_params["stream"] = True
                # 调用流式API
                stream_response = client.chat.completions.create(**api_params)
                
                # 收集流式响应
                collected_content = []
                for chunk in stream_response:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        collected_content.append(chunk.choices[0].delta.content)
                
                response_text = "".join(collected_content)
                
                # 构造类似非流式响应的结构
                return {
                    "success": True,
                    "text": response_text,
                    "model": model,
                    "full_response": {"stream": True, "content": response_text}
                }
            else:
                # 调用非流式API
                completion = client.chat.completions.create(**api_params)
                
                # 从响应中提取文本内容
                response_text = completion.choices[0].message.content
                
                return {
                    "success": True,
                    "text": response_text,
                    "model": model,
                    "full_response": completion.model_dump() if hasattr(completion, "model_dump") else None
                }
        except Exception as e:
            # 标记模型调用失败
            model_limiter.mark_model_used(model, success=False)
            
            retries += 1
            if retries <= max_retries:
                # 如果是速率限制相关错误，等待更长时间
                wait_time = retry_delay
                if "rate" in str(e).lower() or "limit" in str(e).lower() or "too many" in str(e).lower():
                    wait_time = retry_delay * 2
                
                print(f"API调用失败 (模型: {model}, 尝试 {retries}/{max_retries}): {str(e)}")
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                
                # 如果重试，获取一个新的模型，确保不重复使用失败的模型
                new_model = model_limiter.get_available_model()
                while new_model in used_models and len(used_models) < len(AVAILABLE_MODELS):
                    new_model = model_limiter.get_available_model()
                
                model = new_model
                print(f"切换到模型: {model}")
            else:
                print(f"调用LLM API失败，已达到最大重试次数: {str(e)}")
                return {
                    "success": False,
                    "error": str(e),
                    "text": None,
                    "model": model
                }

def parse_llm_response(response: Dict[str, Any], expected_format: str = "json") -> Dict[str, Any]:
    """
    解析LLM的响应，提取所需数据
    
    参数:
        response: LLM响应字典
        expected_format: 期望的返回格式，默认为json
    
    返回:
        解析后的数据字典
    """
    if not response.get("success", False):
        return {"success": False, "error": response.get("error", "Unknown error")}
    
    text = response.get("text", "")
    
    # 尝试解析JSON格式
    if expected_format.lower() == "json":
        try:
            # 尝试从文本中提取JSON部分
            json_str = text
            # 如果文本包含代码块
            if "```json" in text and "```" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                json_str = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                json_str = text[start:end].strip()
                
            parsed_data = json.loads(json_str)
            return {
                "success": True,
                "parsed_data": parsed_data,
                "original_text": text,
                "model": response.get("model")
            }
        except json.JSONDecodeError:
            # 如果解析JSON失败，返回原始文本
            return {
                "success": False,
                "error": "Failed to parse JSON from response",
                "original_text": text,
                "model": response.get("model")
            }
    
    # 默认返回原始文本
    return {
        "success": True,
        "parsed_data": text,
        "original_text": text,
        "model": response.get("model")
    }

# 并行处理多个提示词的函数
def batch_query_llm(prompts: List[Tuple[str, str]], max_workers=None, **kwargs) -> List[Dict[str, Any]]:
    """
    并行处理多个提示词
    
    参数:
        prompts: 提示词列表，每个元素是(prompt, system_prompt)元组
        max_workers: 最大工作线程数，默认为None（使用线程池默认值）
        **kwargs: 传递给query_llm的其他参数
    
    返回:
        响应列表，与输入prompts顺序对应
    """
    results = [None] * len(prompts)
    
    def process_prompt(idx, prompt_tuple):
        prompt, system_prompt = prompt_tuple
        try:
            # 为每个请求选择不同的模型以实现并行
            # 确保不同线程使用不同模型，并避免使用不支持的模型
            model = AVAILABLE_MODELS[idx % len(AVAILABLE_MODELS)]
            print(f"请求 {idx} 使用模型: {model}")
            
            result = query_llm(prompt=prompt, system_prompt=system_prompt, model=model, **kwargs)
            return idx, result
        except Exception as e:
            return idx, {
                "success": False,
                "error": f"处理提示词时出错: {str(e)}",
                "text": None
            }
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_prompt, i, prompt_tuple) for i, prompt_tuple in enumerate(prompts)]
        
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
    
    return results

# 获取模型使用统计
def get_model_stats() -> Dict[str, Dict[str, Any]]:
    """获取模型使用统计"""
    return model_limiter.get_model_stats()

# 测试函数（直接运行此文件时使用）
if __name__ == "__main__":
    # 创建更多测试请求以测试所有模型
    test_prompts = [
        ("我爱你", "你是一个有用的助手"),
        ("讲个笑话", "你是一个幽默的助手"),
        ("解释量子力学", "你是一个科学解释者"),
        ("写一首诗", "你是一个诗人"),
        ("如何做炒饭", "你是一个厨师"),
        ("解释神经网络", "你是一个人工智能专家"),
        ("讲一个历史故事", "你是一个历史学家"),
        ("设计一个游戏", "你是一个游戏设计师"),
        ("描述一幅画", "你是一个艺术评论家"),
        ("解释区块链", "你是一个技术专家"),
        ("写一个故事大纲", "你是一个作家"),
        ("介绍一本经典书籍", "你是一个文学评论家"),
        ("解释相对论", "你是一个物理学家"),
        ("如何学习编程", "你是一个编程教师"),
        ("推荐一部电影", "你是一个电影评论家"),
        ("分析当前经济形势", "你是一个经济学家")
    ]
    
    # 确保测试任务数量足够多，至少覆盖所有模型
    while len(test_prompts) < len(AVAILABLE_MODELS):
        test_prompts.append((f"测试提示词{len(test_prompts)}", "你是一个助手"))
    
    print("开始并行测试...")
    results = batch_query_llm(test_prompts, max_tokens=100, temperature=0.7)
    
    for i, result in enumerate(results):
        print(f"\n--- 测试 {i+1} ---")
        print(f"提示词: {test_prompts[i][0]}")
        print(f"使用模型: {result.get('model', '未知')}")
        print(f"响应: {result.get('text', '无响应')[:100]}...")
    
    # 打印模型统计
    print("\n模型使用统计:")
    stats = get_model_stats()
    for model, stat in sorted(stats.items(), key=lambda x: x[1]['usage'], reverse=True):
        print(f"{model}: 使用次数={stat['usage']}, 错误次数={stat['errors']}, 错误率={stat['error_rate']}")