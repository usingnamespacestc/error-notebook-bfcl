import json
import os
import random
from typing import List, Dict, Optional
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eval.ollama_generate import format_messages, convert_to_ollama_tools, generate_ollama
from scoring.format_utils import build_fewshot_messages, merge_tools

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test.jsonl")
MODEL = "gemma4:26b"
MAX_WORKERS = 4 # Increased now that vLLM is not running

file_lock = threading.Lock()

def evaluate_subset(subset: List[Dict], output_path: str):
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data path {DATA_PATH} does not exist.")
        return

    with open(DATA_PATH, "r") as f:
        test_entries = [json.loads(line) for line in f]
    
    print(f"Total test entries: {len(test_entries)}")
    
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["id"])
                except:
                    pass
    
    print(f"Already processed: {len(processed_ids)}")
    to_process = [e for e in test_entries if e["id"] not in processed_ids]
    
    if not to_process:
        print("All entries already processed.")
        return

    def process_entry(entry: Dict):
        # Construct messages with few-shot
        messages = []
        
        # Only use target's own tool definitions (no merging with few-shot tools)
        tools = convert_to_ollama_tools(entry["function"])

        # Add few-shot examples using proper tool_calls format
        messages.extend(build_fewshot_messages(subset))

        # Add target question
        if isinstance(entry["question"][0], list):
            target_q = entry["question"][0][0]["content"]
        else:
            target_q = entry["question"][0]["content"]
        messages.append({"role": "user", "content": target_q})
        
        # Generate
        result_json = generate_ollama(messages, tools)
        if result_json:
            message = result_json.get("message", {})
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])
            
            if tool_calls:
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
                with open(output_path, "a") as f:
                    f.write(json.dumps(output, ensure_ascii=False) + "\n")
            return True
        return False

    print(f"Processing {len(to_process)} entries with {MAX_WORKERS} workers...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_entry, entry): entry for entry in to_process}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Entry failed: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, required=True, help="Path to subset JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL file")
    args = parser.parse_args()
    
    with open(args.subset, "r") as f:
        subset_data = json.load(f)
        
    evaluate_subset(subset_data, args.output)
