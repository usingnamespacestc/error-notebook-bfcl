"""
Run Volcano Engine API inference on pool.jsonl to collect model predictions.
Much faster than local Ollama. Predictions are used for error classification.
"""
import json
import os
import sys
import time
import argparse
from typing import List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eval.ollama_generate import convert_to_ollama_tools, format_messages

POOL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pool.jsonl")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

API_BASE = "https://ark.cn-beijing.volces.com/api/coding/v3"
API_KEY = os.environ.get("VOLCENGINE_API_KEY", "")
MODEL = "ark-code-latest"
MAX_WORKERS = 8

file_lock = threading.Lock()


def call_volcengine(messages: List[Dict], tools: List[Dict], model: str = MODEL) -> Dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    openai_tools = []
    for t in tools:
        openai_tools.append({
            "type": "function",
            "function": t["function"] if "function" in t else t,
        })

    payload = {
        "model": model,
        "messages": messages,
        "tools": openai_tools if openai_tools else None,
        "temperature": 0,
        "max_tokens": 1024,
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    for attempt in range(3):
        try:
            resp = requests.post(
                f"{API_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == 2:
                print(f"API error after 3 attempts: {e}")
                return None
            time.sleep(2 ** attempt)
    return None


def process_entry(entry: Dict, output_path: str, model: str = MODEL) -> bool:
    messages = format_messages(entry)
    tools = convert_to_ollama_tools(entry["function"])

    result = call_volcengine(messages, tools, model=model)
    if result is None:
        return False

    choices = result.get("choices", [])
    if not choices:
        return False

    message = choices[0].get("message", {})
    content = message.get("content", "") or ""
    tool_calls = message.get("tool_calls", [])

    if tool_calls:
        formatted_tc = []
        for tc in tool_calls:
            func = tc.get("function", {})
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    pass
            formatted_tc.append({
                "function": {
                    "name": func.get("name", ""),
                    "arguments": args,
                }
            })
        response_content = json.dumps(formatted_tc, ensure_ascii=False)
    else:
        response_content = content

    output = {
        "id": entry["id"],
        "category": entry["category"],
        "response": response_content,
        "model": result.get("model", model),
    }

    with file_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL, help="Model name")
    args = parser.parse_args()
    model = args.model

    model_tag = model.lower().replace(" ", "_").replace(".", "-")
    os.makedirs(RESULT_DIR, exist_ok=True)
    output_path = os.path.join(RESULT_DIR, f"volcengine_{model_tag}_pool_predictions.jsonl")

    with open(POOL_PATH, "r") as f:
        entries = [json.loads(line) for line in f]

    print(f"Total pool entries: {len(entries)}")

    # Resume support
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
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

    # Quick API test
    print(f"Testing API connection with model={model}...")
    test_result = call_volcengine(
        [{"role": "user", "content": "hi"}], [], model=model
    )
    if test_result is None:
        print("API connection failed!")
        return
    actual_model = test_result.get("model", "unknown")
    print(f"API OK, actual model: {actual_model}")

    print(f"Processing {len(to_process)} entries with {MAX_WORKERS} workers...")
    success = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_entry, e, output_path, model): e for e in to_process}
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
