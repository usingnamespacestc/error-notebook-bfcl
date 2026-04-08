"""
Classify pool predictions into correct/incorrect with error types.
Outputs a classified JSONL with error_type field for each entry.
"""
import json
import os
import sys
from typing import List, Dict
from tqdm import tqdm
from collections import Counter

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import bfcl_ast_checker
from bfcl_eval.constants.enums import Language
from eval.ast_evaluator import extract_tool_calls, map_category_to_language

POOL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pool.jsonl")
# Defaults, can be overridden by command-line args
PREDICTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "pool_predictions.jsonl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "pool_classified.jsonl")


def format_tool_calls_for_checker(tool_calls: List[Dict]) -> List[Dict]:
    """Convert extracted tool calls to the format expected by ast_checker."""
    formatted = []
    for call in tool_calls:
        if isinstance(call, dict):
            if "function" in call:
                func_data = call["function"]
                name = func_data.get("name")
                args = func_data.get("arguments", {})
            else:
                name = call.get("name")
                args = call.get("arguments", {})

            if not name and len(call) == 1:
                name = list(call.keys())[0]
                args = call[name]

            if name:
                formatted.append({name: args})
    return formatted


def classify(predictions_path=PREDICTIONS_PATH, output_path=OUTPUT_PATH, model_name="gemma4:26b"):
    # Load pool data
    pool_map = {}
    with open(POOL_PATH, "r") as f:
        for line in f:
            entry = json.loads(line)
            pool_map[entry["id"]] = entry

    # Load predictions
    with open(predictions_path, "r") as f:
        predictions = [json.loads(line) for line in f]

    print(f"Pool entries: {len(pool_map)}, Predictions: {len(predictions)}")

    results = []
    error_type_counts = Counter()
    correct_count = 0

    for pred in tqdm(predictions, desc="Classifying"):
        entry_id = pred["id"]
        if entry_id not in pool_map:
            continue

        entry = pool_map[entry_id]
        response = pred["response"]

        # Extract and format tool calls
        tool_calls = extract_tool_calls(response)
        formatted_calls = format_tool_calls_for_checker(tool_calls)

        lang = map_category_to_language(entry["category"])

        try:
            eval_result = bfcl_ast_checker.ast_checker(
                entry["function"],
                formatted_calls,
                entry["ground_truth"],
                lang,
                entry["category"],
                model_name
            )
        except Exception as e:
            eval_result = {
                "valid": False,
                "error": [f"Checker error: {str(e)}"],
                "error_type": "checker_error"
            }

        is_correct = eval_result["valid"]
        error_type = eval_result.get("error_type", "none") if not is_correct else "none"

        if is_correct:
            correct_count += 1
        else:
            error_type_counts[error_type] += 1

        results.append({
            "id": entry_id,
            "category": entry["category"],
            "correct": is_correct,
            "error_type": error_type,
            "error": eval_result.get("error", []),
            "model_output": formatted_calls,
            "response_raw": response,
        })

    # Write classified results
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print summary
    total = len(results)
    print(f"\n=== Classification Summary ===")
    print(f"Total: {total}")
    print(f"Correct: {correct_count} ({correct_count/total:.1%})")
    print(f"Incorrect: {total - correct_count} ({(total-correct_count)/total:.1%})")
    print(f"\nError type distribution:")
    for et, count in error_type_counts.most_common():
        print(f"  {et}: {count} ({count/(total-correct_count):.1%} of errors)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, default=PREDICTIONS_PATH,
                        help="Path to predictions JSONL")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH,
                        help="Path to output classified JSONL")
    parser.add_argument("--model-name", type=str, default="gemma4:26b",
                        help="Model name for ast_checker")
    args = parser.parse_args()
    classify(predictions_path=args.predictions, output_path=args.output,
             model_name=args.model_name)
