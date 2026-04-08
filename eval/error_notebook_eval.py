"""
Error Notebook evaluation: uses k correct examples + k error-correction examples as few-shot.

Prompt structure:
  [System: You are a helpful assistant that calls tools accurately.]

  --- Correct examples (standard few-shot) ---
  User: <question>
  Assistant: <correct tool_call>

  --- Error-correction examples ---
  User: <question>
  Assistant: <wrong tool_call>  (the model's actual mistake)
  User: "That's incorrect. The correct call should be: ..."
  Assistant: <correct tool_call>

  --- Target question ---
  User: <target question>

Only the target's own tool definitions are provided (no merging from examples).
"""
import json
import os
import sys
from typing import List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eval.ollama_generate import convert_to_ollama_tools, generate_ollama, format_messages
from scoring.format_utils import convert_bfcl_to_ollama_tc

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test.jsonl")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL = "gemma4:26b"
MAX_WORKERS = 4

file_lock = threading.Lock()


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
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls
        })
    return messages


def build_error_correction_fewshot(error_examples: List[Dict]) -> List[Dict]:
    """
    Build error-correction messages.
    Each example shows: question → wrong answer → correction feedback → correct answer.
    """
    messages = []
    for ex in error_examples:
        # User question
        if isinstance(ex["question"][0], list):
            q = ex["question"][0][0]["content"]
        else:
            q = ex["question"][0]["content"]
        messages.append({"role": "user", "content": q})

        # Assistant gives WRONG answer (what the model actually produced)
        wrong_output = ex.get("model_wrong_output", [])
        if wrong_output:
            # Format as tool_calls if possible
            wrong_tc = []
            for call in wrong_output:
                if isinstance(call, dict) and len(call) == 1:
                    func_name = list(call.keys())[0]
                    args = call[func_name]
                    wrong_tc.append({
                        "function": {
                            "name": func_name,
                            "arguments": args
                        }
                    })
            if wrong_tc:
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": wrong_tc
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(wrong_output, ensure_ascii=False)
                })
        else:
            messages.append({
                "role": "assistant",
                "content": "(no valid tool call produced)"
            })

        # User correction feedback
        correct_tc = convert_bfcl_to_ollama_tc(ex["ground_truth"])
        error_detail = ex.get("error_detail", ex.get("error", []))
        error_desc = "; ".join(error_detail) if error_detail else "incorrect parameters"

        correction_msg = (
            f"That tool call has an error: {error_desc}. "
            f"The correct call should be:"
        )
        # Include the correct call in the correction message as text
        correct_calls_text = json.dumps(correct_tc, ensure_ascii=False, indent=None)
        correction_msg += f"\n{correct_calls_text}"

        messages.append({"role": "user", "content": correction_msg})

        # Assistant acknowledges with correct tool call
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": correct_tc
        })

    return messages


def process_entry(entry: Dict, fewshot_messages: List[Dict], output_path: str) -> bool:
    """Process a single test entry with error notebook few-shot context."""
    messages = list(fewshot_messages)  # copy

    # Add target question
    if isinstance(entry["question"][0], list):
        target_q = entry["question"][0][0]["content"]
    else:
        target_q = entry["question"][0]["content"]
    messages.append({"role": "user", "content": target_q})

    # Only use target's own tool definitions
    tools = convert_to_ollama_tools(entry["function"])

    result_json = generate_ollama(messages, tools, model=MODEL)
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
        }
        with file_lock:
            with open(output_path, "a") as f:
                f.write(json.dumps(output, ensure_ascii=False) + "\n")
        return True
    return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, required=True,
                        help="Path to error_notebook_subset.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: results/error_notebook_responses.jsonl)")
    args = parser.parse_args()

    # Load subset
    with open(args.subset, "r") as f:
        subset = json.load(f)

    correct_examples = subset["correct_examples"]
    error_examples = subset["error_examples"]

    print(f"Correct examples: {len(correct_examples)}")
    print(f"Error examples: {len(error_examples)}")

    # Build few-shot messages
    fewshot_messages = []

    # System message
    fewshot_messages.append({
        "role": "system",
        "content": (
            "You are a helpful assistant that calls tools accurately. "
            "Pay close attention to parameter types and values. "
            "Below are some examples of correct and incorrect tool calls for reference."
        )
    })

    # Correct examples first
    fewshot_messages.extend(build_correct_fewshot(correct_examples))

    # Then error-correction examples
    fewshot_messages.extend(build_error_correction_fewshot(error_examples))

    print(f"Total few-shot messages: {len(fewshot_messages)}")

    # Load test data
    output_path = args.output or os.path.join(RESULT_DIR, "error_notebook_responses.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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

    print(f"Processing {len(to_process)} entries with {MAX_WORKERS} workers...")
    success = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_entry, e, fewshot_messages, output_path): e
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
