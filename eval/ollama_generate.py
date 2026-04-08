import json
import os
import requests
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import copy

# Add project root to sys.path for cache access
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cache.llm_cache import cache_lookup, cache_store

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test.jsonl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "zero_shot_responses.jsonl")
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:26b"
MAX_WORKERS = 4

file_lock = threading.Lock()

def convert_to_ollama_tools(bfcl_functions: List[Dict]) -> List[Dict]:
    ollama_tools = []
    for func in bfcl_functions:
        # Deep copy to avoid modifying original
        f = copy.deepcopy(func)
        if "parameters" in f and f["parameters"].get("type") == "dict":
            f["parameters"]["type"] = "object"
        
        ollama_tools.append({
            "type": "function",
            "function": f
        })
    return ollama_tools

def format_messages(entry: Dict) -> List[Dict]:
    # For models with native tool calling, we don't necessarily need tools in system prompt
    # but we can keep it for better steerability if needed.
    # However, Ollama's tool calling works best if we don't duplicate tool definitions in system prompt
    # unless the model is specifically trained for it. Gemma-4 is.
    
    if isinstance(entry["question"][0], list):
        q_messages = entry["question"][0]
    else:
        q_messages = entry["question"]
        
    return q_messages

def generate_ollama(messages: List[Dict], tools: List[Dict], model: str = MODEL) -> Optional[Dict]:
    params = {
        "temperature": 0,
        "num_predict": 1024,
    }
    
    # Cache key includes tools now
    prompt_payload = {
        "messages": messages,
        "tools": tools
    }
    prompt_str = json.dumps(prompt_payload, sort_keys=True)
    
    cached = cache_lookup("ollama_fc", model, prompt_str, params)
    if cached:
        return cached
    
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
        "options": params
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        cache_store("ollama_fc", model, prompt_str, params, result)
        return result
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None

def process_entry(entry: Dict):
    messages = format_messages(entry)
    tools = convert_to_ollama_tools(entry["function"])
    
    result_json = generate_ollama(messages, tools)
    if result_json is not None:
        message = result_json.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])
        
        # If tool_calls exist, we format them as a JSON string in the response field
        # so that the ast_evaluator can extract them.
        if tool_calls:
            # Convert Ollama tool_calls to a more standard format for our evaluator
            # Ollama: [{"function": {"name": "...", "arguments": {...}}}]
            # Our evaluator handles this.
            response_content = json.dumps(tool_calls, ensure_ascii=False)
        else:
            response_content = content
            
        output = {
            "id": entry["id"],
            "category": entry["category"],
            "response": response_content,
            "thinking": message.get("thinking", "")
        }
        with file_lock:
            with open(OUTPUT_PATH, "a") as f:
                f.write(json.dumps(output, ensure_ascii=False) + "\n")
        return True
    return False

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data path {DATA_PATH} does not exist.")
        return

    with open(DATA_PATH, "r") as f:
        entries = [json.loads(line) for line in f]
    
    print(f"Total entries: {len(entries)}")
    
    processed_ids = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["id"])
                except:
                    pass
    
    print(f"Already processed: {len(processed_ids)}")
    
    to_process = [e for entry in entries if (e := entry)["id"] not in processed_ids]
    
    if not to_process:
        print("All entries already processed.")
        return

    print(f"Processing {len(to_process)} entries with {MAX_WORKERS} workers...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_entry, entry): entry for entry in to_process}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Entry processing failed: {e}")

if __name__ == "__main__":
    main()
