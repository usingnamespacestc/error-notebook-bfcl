"""
Run Ollama inference on pool.jsonl to collect model predictions.
These predictions will be classified into correct/incorrect for error notebook selection.
"""
import json
import os
import sys
from typing import Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eval.ollama_generate import format_messages, convert_to_ollama_tools, generate_ollama
from cache.llm_cache import cache_lookup

POOL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pool.jsonl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "pool_predictions.jsonl")
MODEL = "gemma4:26b"
MAX_WORKERS = 4

file_lock = threading.Lock()


def process_entry(entry: Dict) -> bool:
    messages = format_messages(entry)
    tools = convert_to_ollama_tools(entry["function"])

    result_json = generate_ollama(messages, tools, model=MODEL)
    if result_json is not None:
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
        }
        with file_lock:
            with open(OUTPUT_PATH, "a") as f:
                f.write(json.dumps(output, ensure_ascii=False) + "\n")
        return True
    return False


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(POOL_PATH, "r") as f:
        entries = [json.loads(line) for line in f]

    print(f"Total pool entries: {len(entries)}")

    # Resume support
    processed_ids = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["id"])
                except:
                    pass

    print(f"Already processed: {len(processed_ids)}")
    to_process = [e for e in entries if e["id"] not in processed_ids]

    if not to_process:
        print("All pool entries already processed.")
        return

    print(f"Processing {len(to_process)} entries with {MAX_WORKERS} workers...")
    success = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_entry, e): e for e in to_process}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                if future.result():
                    success += 1
                else:
                    fail += 1
            except Exception as e:
                fail += 1
                print(f"Error: {e}")

    print(f"Done. Success: {success}, Failed: {fail}")


if __name__ == "__main__":
    main()
