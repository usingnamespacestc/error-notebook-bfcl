import json
import os
from typing import Dict, Any
import sys

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eval.ast_evaluator import extract_tool_calls, map_category_to_language
import eval.bfcl_ast_checker as bfcl_ast_checker

def evaluate_file(response_path: str, data_path: str) -> Dict[str, Any]:
    if not os.path.exists(response_path):
        return None
        
    gt_map = {}
    with open(data_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            gt_map[entry["id"]] = entry
            
    results = []
    with open(response_path, "r") as f:
        for line in f:
            resp_entry = json.loads(line)
            id = resp_entry["id"]
            if id not in gt_map: continue
            
            entry = gt_map[id]
            tool_calls = extract_tool_calls(resp_entry["response"])
            
            formatted_calls = []
            for call in tool_calls:
                if isinstance(call, dict):
                    if "function" in call:
                        fd = call["function"]; name = fd.get("name"); args = fd.get("arguments", {})
                    else:
                        name = call.get("name"); args = call.get("arguments", {})
                    if not name and len(call) == 1:
                        name = list(call.keys())[0]; args = call[name]
                    if name: formatted_calls.append({name: args})
            
            lang = map_category_to_language(entry["category"])
            try:
                eval_result = bfcl_ast_checker.ast_checker(
                    entry["function"], formatted_calls, entry["ground_truth"], lang, entry["category"], "gemma4:26b"
                )
            except:
                eval_result = {"valid": False}
                
            results.append({"category": entry["category"], "valid": eval_result["valid"]})
            
    stats = {}
    for res in results:
        cat = res["category"]
        if cat not in stats: stats[cat] = {"correct": 0, "total": 0}
        stats[cat]["total"] += 1
        if res["valid"]: stats[cat]["correct"] += 1
        
    summary = {
        "overall": sum(s["correct"] for s in stats.values()) / len(results) if results else 0,
        "categories": {cat: s["correct"] / s["total"] for cat, s in stats.items()},
        "total": len(results)
    }
    return summary

def main():
    BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
    DATA_PATH = os.path.join(BASE_DIR, "data", "test.jsonl")
    
    conditions = {
        "Zero-shot": os.path.join(BASE_DIR, "results", "zero_shot_responses.jsonl"),
        "Random 5-shot": os.path.join(BASE_DIR, "results", "random_kshot_responses.jsonl"),
        "Selected 5-shot": os.path.join(BASE_DIR, "results", "selected_kshot_responses.jsonl")
    }
    
    print(f"{'Condition':<20} | {'Accuracy':<10} | {'Total':<6}")
    print("-" * 45)
    
    all_summaries = {}
    for name, path in conditions.items():
        summary = evaluate_file(path, DATA_PATH)
        if summary:
            print(f"{name:<20} | {summary['overall']:>9.2%} | {summary['total']:>6}")
            all_summaries[name] = summary
        else:
            print(f"{name:<20} | {'N/A':>10} | {'N/A':>6}")
            
    # Category comparison for Selected vs Zero-shot
    if "Selected 5-shot" in all_summaries and "Zero-shot" in all_summaries:
        print("\nCategory breakdown (Selected vs Zero-shot):")
        sel = all_summaries["Selected 5-shot"]["categories"]
        zero = all_summaries["Zero-shot"]["categories"]
        
        print(f"{'Category':<25} | {'Zero':>8} | {'Selected':>8} | {'Diff':>8}")
        print("-" * 60)
        for cat in sorted(set(sel.keys()) | set(zero.keys())):
            z_acc = zero.get(cat, 0); s_acc = sel.get(cat, 0)
            print(f"{cat:<25} | {z_acc:>8.1%} | {s_acc:>8.1%} | {s_acc-z_acc:>+8.1%}")

if __name__ == "__main__":
    main()
