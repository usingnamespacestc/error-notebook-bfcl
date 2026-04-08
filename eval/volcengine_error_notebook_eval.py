"""
Error Notebook evaluation via Volcano Engine API.
Uses k correct examples + k error-correction examples as few-shot context.
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
from scoring.format_utils import convert_bfcl_to_ollama_tc

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test.jsonl")
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


def build_correct_fewshot(examples: List[Dict]) -> List[Dict]:
    """Build standard few-shot messages from correct examples."""
    messages = []
    for ex in examples:
        if isinstance(ex["question"][0], list):
            q = ex["question"][0][0]["content"]
        else:
            q = ex["question"][0]["content"]
        messages.append({"role": "user", "content": q})

        tool_calls = convert_bfcl_to_ollama_tc(ex["ground_truth"])
        # OpenAI format tool_calls for volcengine
        openai_tc = []
        for tc in tool_calls:
            openai_tc.append({
                "id": f"call_{len(openai_tc)}",
                "type": "function",
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": json.dumps(tc["function"]["arguments"], ensure_ascii=False),
                }
            })
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": openai_tc,
        })
    return messages


def build_error_correction_fewshot(error_examples: List[Dict]) -> List[Dict]:
    """Build error-correction messages in OpenAI chat format."""
    messages = []
    for ex in error_examples:
        # User question
        if isinstance(ex["question"][0], list):
            q = ex["question"][0][0]["content"]
        else:
            q = ex["question"][0]["content"]
        messages.append({"role": "user", "content": q})

        # Assistant gives WRONG answer
        wrong_output = ex.get("model_wrong_output", [])
        if wrong_output:
            wrong_tc = []
            for call in wrong_output:
                if isinstance(call, dict) and len(call) == 1:
                    func_name = list(call.keys())[0]
                    args = call[func_name]
                    wrong_tc.append({
                        "id": f"call_wrong_{len(wrong_tc)}",
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": json.dumps(args, ensure_ascii=False) if not isinstance(args, str) else args,
                        }
                    })
            if wrong_tc:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": wrong_tc,
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(wrong_output, ensure_ascii=False),
                })
        else:
            messages.append({
                "role": "assistant",
                "content": "(no valid tool call produced)",
            })

        # User correction
        correct_tc = convert_bfcl_to_ollama_tc(ex["ground_truth"])
        error_detail = ex.get("error_detail", ex.get("error", []))
        error_desc = "; ".join(error_detail) if error_detail else "incorrect parameters"
        correct_calls_text = json.dumps(correct_tc, ensure_ascii=False)
        correction_msg = (
            f"That tool call has an error: {error_desc}. "
            f"The correct call should be:\n{correct_calls_text}"
        )
        messages.append({"role": "user", "content": correction_msg})

        # Assistant corrects with proper tool call
        correct_openai_tc = []
        for tc in correct_tc:
            correct_openai_tc.append({
                "id": f"call_correct_{len(correct_openai_tc)}",
                "type": "function",
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": json.dumps(tc["function"]["arguments"], ensure_ascii=False),
                }
            })
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": correct_openai_tc,
        })

    return messages


def process_entry(entry: Dict, fewshot_messages: List[Dict],
                  output_path: str, model: str) -> bool:
    messages = list(fewshot_messages)

    # Add target question
    if isinstance(entry["question"][0], list):
        target_q = entry["question"][0][0]["content"]
    else:
        target_q = entry["question"][0]["content"]
    messages.append({"role": "user", "content": target_q})

    # Only target's own tools
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
    parser.add_argument("--subset", type=str, required=True,
                        help="Path to error_notebook_subset.json")
    parser.add_argument("--model", type=str, default=MODEL, help="Model name")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--interleaved", action="store_true",
                        help="Interleave error and correct examples (error->correct->error->...)")
    args = parser.parse_args()
    model = args.model

    # Load subset
    with open(args.subset, "r") as f:
        subset = json.load(f)

    correct_examples = subset["correct_examples"]
    error_examples = subset["error_examples"]

    print(f"Correct examples: {len(correct_examples)}")
    print(f"Error examples: {len(error_examples)}")
    print(f"Interleaved: {args.interleaved}")

    # Build few-shot messages
    fewshot_messages = [{
        "role": "system",
        "content": (
            "You are a helpful assistant that calls tools accurately. "
            "Pay close attention to parameter types and values. "
            "Below are some examples of correct and incorrect tool calls for reference."
        )
    }]

    if args.interleaved and correct_examples and error_examples:
        # Interleave: error->correct->error->correct->... ending with error correction
        # so the last thing before target is a corrected answer (positive momentum)
        # 正→错→正→错→...→正→错→[目标问题]
        # Last example before target is an error correction (ends on corrected answer)
        max_len = max(len(correct_examples), len(error_examples))
        for i in range(max_len):
            if i < len(correct_examples):
                fewshot_messages.extend(build_correct_fewshot([correct_examples[i]]))
            if i < len(error_examples):
                fewshot_messages.extend(build_error_correction_fewshot([error_examples[i]]))
    else:
        fewshot_messages.extend(build_correct_fewshot(correct_examples))
        fewshot_messages.extend(build_error_correction_fewshot(error_examples))

    print(f"Total few-shot messages: {len(fewshot_messages)}")

    # Output path
    model_tag = model.lower().replace(" ", "_").replace(".", "-")
    output_path = args.output or os.path.join(
        RESULT_DIR, f"volcengine_{model_tag}_error_notebook_responses.jsonl"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load test data
    with open(DATA_PATH, "r") as f:
        entries = [json.loads(line) for line in f]

    print(f"Total test entries: {len(entries)}")

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
        print("All entries already processed.")
        return

    # Quick API test
    print(f"Testing API with model={model}...")
    test_result = call_volcengine([{"role": "user", "content": "hi"}], [], model=model)
    if test_result is None:
        print("API connection failed!")
        return
    print(f"API OK, actual model: {test_result.get('model', 'unknown')}")

    print(f"Processing {len(to_process)} entries with {MAX_WORKERS} workers...")
    success = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_entry, e, fewshot_messages, output_path, model): e
            for e in to_process
        }
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
