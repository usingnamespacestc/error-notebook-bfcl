import json
import os
import random
from typing import List, Dict
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eval.ollama_generate import format_messages, convert_to_ollama_tools, generate_ollama
from scoring.format_utils import build_fewshot_messages, merge_tools

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test.jsonl")
POOL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pool.jsonl")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "random_kshot")
MODEL = "gemma4:26b"
K = 5
NUM_RUNS = 4
SEED = 42
MAX_WORKERS = 4

def run_random_baseline():
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    with open(POOL_PATH, "r") as f:
        pool = [json.loads(line) for line in f]
        
    with open(DATA_PATH, "r") as f:
        test_entries = [json.loads(line) for line in f]
        
    random.seed(SEED)

    for run_idx in range(NUM_RUNS):
        print(f"Starting Random Run {run_idx}...")
        few_shot_subset = random.sample(pool, K)

        # Save the subset for reproducibility
        subset_path = os.path.join(RESULT_DIR, f"run_{run_idx}_subset.json")
        with open(subset_path, "w") as f:
            json.dump([ex["id"] for ex in few_shot_subset], f)

        output_path = os.path.join(RESULT_DIR, f"run_{run_idx}_responses.jsonl")

        processed_ids = set()
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                for line in f:
                    try:
                        processed_ids.add(json.loads(line)["id"])
                    except:
                        pass

        # Build few-shot messages with proper tool_calls format
        fewshot_messages = build_fewshot_messages(few_shot_subset)
        file_lock = threading.Lock()
        to_process = [e for e in test_entries if e["id"] not in processed_ids]

        if not to_process:
            print(f"Run {run_idx}: all entries already processed.")
            continue

        def process_entry(entry):
            all_tool_defs = merge_tools(entry["function"], few_shot_subset)
            tools = convert_to_ollama_tools(all_tool_defs)

            messages = list(fewshot_messages)
            if isinstance(entry["question"][0], list):
                target_q = entry["question"][0][0]["content"]
            else:
                target_q = entry["question"][0]["content"]
            messages.append({"role": "user", "content": target_q})

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

        print(f"Run {run_idx}: processing {len(to_process)} entries with {MAX_WORKERS} workers...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_entry, e): e for e in to_process}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Run {run_idx}"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Entry failed: {e}")

if __name__ == "__main__":
    run_random_baseline()
