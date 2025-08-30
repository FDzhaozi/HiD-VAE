import os
from openai import OpenAI
import json
from typing import Optional, Dict, Any, List, Tuple
import time
import threading
import queue
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# List of available models
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

# Models requiring streaming mode
STREAM_MODELS = [
    "qwq-plus",
    "qwq-plus-latest",
    "qwq-plus-2025-03-05"
]

# Unsupported models (won't be used)
UNSUPPORTED_MODELS = [
    "qvq-max",
    "qvq-max-latest",
    "qvq-max-2025-05-15",
    "qvq-max-2025-03-25"
]

# Add streaming models to available models list
AVAILABLE_MODELS.extend(STREAM_MODELS)

# Model-specific configurations
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

# Model usage statistics and rate limiting control
class ModelRateLimiter:
    def __init__(self, cooldown_period=2.0):
        self.model_last_used = {model: 0 for model in AVAILABLE_MODELS}
        self.model_usage_count = {model: 0 for model in AVAILABLE_MODELS}
        self.model_error_count = {model: 0 for model in AVAILABLE_MODELS}
        self.cooldown_period = cooldown_period
        self.lock = threading.Lock()
        self.round_robin_index = 0  # Round-robin index
    
    def get_available_model(self) -> str:
        """Return currently available model with lowest load, ensuring all models get used"""
        with self.lock:
            current_time = time.time()
            
            # Prioritize least-used models (70% chance)
            if random.random() < 0.7:
                models_by_usage = sorted(AVAILABLE_MODELS, key=lambda m: self.model_usage_count.get(m, 0))
                candidates = models_by_usage[:min(3, len(models_by_usage))]
                return random.choice(candidates)
            
            # 30% chance to use round-robin selection
            self.round_robin_index = (self.round_robin_index + 1) % len(AVAILABLE_MODELS)
            return AVAILABLE_MODELS[self.round_robin_index]
    
    def mark_model_used(self, model: str, success: bool = True):
        """Mark a model as used"""
        with self.lock:
            self.model_last_used[model] = time.time()
            self.model_usage_count[model] = self.model_usage_count.get(model, 0) + 1
            if not success:
                self.model_error_count[model] = self.model_error_count.get(model, 0) + 1
    
    def get_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get model usage statistics"""
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

# Global model rate limiter instance
model_limiter = ModelRateLimiter()

def query_llm(
    prompt: str,
    system_prompt: str = "",
    model: str = None,  # If None, auto-selects model
    api_key: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    retry_delay: float = 3.0,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Call LLM API to get response
    
    Args:
        prompt: User prompt
        system_prompt: System prompt
        model: Model name (None for auto-selection)
        api_key: API key (None uses environment/default)
        max_tokens: Maximum tokens to generate
        temperature: Generation temperature (lower=more deterministic)
        retry_delay: Retry delay in seconds on failure
        max_retries: Maximum retry attempts
    
    Returns:
        Dictionary containing model response
    """
    # Auto-select model if none provided
    if model is None:
        model = model_limiter.get_available_model()
        print(f"Auto-selected model: {model}")
    
    # Switch to supported model if specified one is unsupported
    if model in UNSUPPORTED_MODELS:
        model = random.choice(AVAILABLE_MODELS)
        print(f"Specified model doesn't support HTTP calls, switching to: {model}")
    
    # Use default API key if none provided
    if api_key is None:
        api_key = ""  # Default API key
    
    # Initialize client
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # Retry mechanism
    retries = 0
    used_models = set()  # Track used models
    while retries <= max_retries:
        try:
            # Mark model as used before API call
            model_limiter.mark_model_used(model)
            used_models.add(model)
            
            # Prepare API parameters
            api_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            # Add model-specific parameters
            if model in MODEL_CONFIG:
                extra_params = MODEL_CONFIG.get(model, {})
                if extra_params:
                    api_params["extra_body"] = extra_params
            
            # Handle streaming models
            if model in STREAM_MODELS:
                api_params["stream"] = True
                # Call streaming API
                stream_response = client.chat.completions.create(**api_params)
                
                # Collect streaming response
                collected_content = []
                for chunk in stream_response:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        collected_content.append(chunk.choices[0].delta.content)
                
                response_text = "".join(collected_content)
                
                # Construct response similar to non-streaming
                return {
                    "success": True,
                    "text": response_text,
                    "model": model,
                    "full_response": {"stream": True, "content": response_text}
                }
            else:
                # Call non-streaming API
                completion = client.chat.completions.create(**api_params)
                
                # Extract text from response
                response_text = completion.choices[0].message.content
                
                return {
                    "success": True,
                    "text": response_text,
                    "model": model,
                    "full_response": completion.model_dump() if hasattr(completion, "model_dump") else None
                }
        except Exception as e:
            # Mark failed model call
            model_limiter.mark_model_used(model, success=False)
            
            retries += 1
            if retries <= max_retries:
                # Longer wait for rate limit errors
                wait_time = retry_delay
                if "rate" in str(e).lower() or "limit" in str(e).lower() or "too many" in str(e).lower():
                    wait_time = retry_delay * 2
                
                print(f"API call failed (model: {model}, attempt {retries}/{max_retries}): {str(e)}")
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                
                # Get new model for retry, avoiding previously failed ones
                new_model = model_limiter.get_available_model()
                while new_model in used_models and len(used_models) < len(AVAILABLE_MODELS):
                    new_model = model_limiter.get_available_model()
                
                model = new_model
                print(f"Switching to model: {model}")
            else:
                print(f"LLM API call failed after max retries: {str(e)}")
                return {
                    "success": False,
                    "error": str(e),
                    "text": None,
                    "model": model
                }

def parse_llm_response(response: Dict[str, Any], expected_format: str = "json") -> Dict[str, Any]:
    """
    Parse LLM response to extract required data
    
    Args:
        response: LLM response dictionary
        expected_format: Expected return format (default: json)
    
    Returns:
        Parsed data dictionary
    """
    if not response.get("success", False):
        return {"success": False, "error": response.get("error", "Unknown error")}
    
    text = response.get("text", "")
    
    # Try parsing JSON format
    if expected_format.lower() == "json":
        try:
            # Extract JSON from text if in code block
            json_str = text
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
            # Return raw text if JSON parsing fails
            return {
                "success": False,
                "error": "Failed to parse JSON from response",
                "original_text": text,
                "model": response.get("model")
            }
    
    # Default return raw text
    return {
        "success": True,
        "parsed_data": text,
        "original_text": text,
        "model": response.get("model")
    }

# Function for parallel processing of multiple prompts
def batch_query_llm(prompts: List[Tuple[str, str]], max_workers=None, **kwargs) -> List[Dict[str, Any]]:
    """
    Process multiple prompts in parallel
    
    Args:
        prompts: List of prompt tuples (prompt, system_prompt)
        max_workers: Max worker threads (None for default)
        **kwargs: Additional arguments for query_llm
    
    Returns:
        List of responses in same order as input prompts
    """
    results = [None] * len(prompts)
    
    def process_prompt(idx, prompt_tuple):
        prompt, system_prompt = prompt_tuple
        try:
            # Assign different models to each request for parallelism
            model = AVAILABLE_MODELS[idx % len(AVAILABLE_MODELS)]
            print(f"Request {idx} using model: {model}")
            
            result = query_llm(prompt=prompt, system_prompt=system_prompt, model=model, **kwargs)
            return idx, result
        except Exception as e:
            return idx, {
                "success": False,
                "error": f"Error processing prompt: {str(e)}",
                "text": None
            }
    
    # Parallel processing with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_prompt, i, prompt_tuple) for i, prompt_tuple in enumerate(prompts)]
        
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
    
    return results

# Get model usage statistics
def get_model_stats() -> Dict[str, Dict[str, Any]]:
    """Get model usage statistics"""
    return model_limiter.get_model_stats()

# Test function (when run directly)
if __name__ == "__main__":
    # Create test requests to test all models
    test_prompts = [
        ("Tell a historical story", "You are a historian"),
        ("Design a game", "You are a game designer"),
        ("Describe a painting", "You are an art critic")
    ]
    
    # Ensure enough test prompts to cover all models
    while len(test_prompts) < len(AVAILABLE_MODELS):
        test_prompts.append((f"Test prompt {len(test_prompts)}", "You are an assistant"))
    
    print("Starting parallel tests...")
    results = batch_query_llm(test_prompts, max_tokens=100, temperature=0.7)
    
    for i, result in enumerate(results):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {test_prompts[i][0]}")
        print(f"Model used: {result.get('model', 'Unknown')}")
        print(f"Response: {result.get('text', 'No response')[:100]}...")
    
    # Print model statistics
    print("\nModel usage statistics:")
    stats = get_model_stats()
    for model, stat in sorted(stats.items(), key=lambda x: x[1]['usage'], reverse=True):
        print(f"{model}: Usage={stat['usage']}, Errors={stat['errors']}, Error rate={stat['error_rate']}")
