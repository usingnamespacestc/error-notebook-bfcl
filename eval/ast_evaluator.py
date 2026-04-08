import json
import os
import re
from typing import List, Dict, Any
from tqdm import tqdm
import sys

# Add current dir to path for bfcl_ast_checker
sys.path.append(os.path.dirname(__file__))
import bfcl_ast_checker
from bfcl_eval.constants.enums import Language

RESPONSE_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "zero_shot_responses.jsonl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test.jsonl")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "zero_shot_eval.json")

def extract_tool_calls(response: str) -> List[Dict]:
    # Handle thinking part if present
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = re.sub(r"<thought>.*?</thought>", "", response, flags=re.DOTALL).strip()
    
    # Try to find a JSON list or dict in the response
    try:
        # Look for the last [...] or {...} in the response
        # Most models output tool calls at the end
        matches = list(re.finditer(r"(\[.*\]|\{.*\})", response, re.DOTALL))
        if matches:
            for match in reversed(matches):
                candidate = match.group(1)
                try:
                    data = json.loads(candidate)
                    if isinstance(data, list):
                        return data
                    if isinstance(data, dict):
                        return [data]
                except:
                    continue
    except:
        pass
    
    return []

def map_category_to_language(category: str) -> Language:
    cat_lower = category.lower()
    if "java" in cat_lower and "javascript" not in cat_lower:
        return Language.JAVA
    if "javascript" in cat_lower:
        return Language.JAVASCRIPT
    return Language.PYTHON

def evaluate():
    if not os.path.exists(RESPONSE_PATH):
        print(f"Error: Response path {RESPONSE_PATH} does not exist.")
        return

    # Load ground truth
    gt_map = {}
    with open(DATA_PATH, "r") as f:
        for line in f:
            entry = json.loads(line)
            gt_map[entry["id"]] = entry

    results = []
    
    with open(RESPONSE_PATH, "r") as f:
        lines = f.readlines()
        
    print(f"Evaluating {len(lines)} responses...")
    
    for line in tqdm(lines):
        resp_entry = json.loads(line)
        id = resp_entry["id"]
        if id not in gt_map:
            continue
            
        entry = gt_map[id]
        response = resp_entry["response"]
        
        tool_calls = extract_tool_calls(response)
        
        # Format tool_calls for bfcl_ast_checker
        formatted_calls = []
        for call in tool_calls:
            if isinstance(call, dict):
                # Ollama/OpenAI format: {"function": {"name": "...", "arguments": {...}}}
                if "function" in call:
                    func_data = call["function"]
                    name = func_data.get("name")
                    args = func_data.get("arguments", {})
                else:
                    # Standard format: {"name": "...", "arguments": {...}}
                    name = call.get("name")
                    args = call.get("arguments", {})
                
                # If name is still not present, it might be { "func_name": { "arg1": "val1" } }
                if not name and len(call) == 1:
                    name = list(call.keys())[0]
                    args = call[name]
                
                if name:
                    formatted_calls.append({name: args})
        
        lang = map_category_to_language(entry["category"])
        
        try:
            eval_result = bfcl_ast_checker.ast_checker(
                entry["function"],
                formatted_calls,
                entry["ground_truth"],
                lang,
                entry["category"],
                "gemma4:26b"
            )
        except Exception as e:
            eval_result = {"valid": False, "error": [f"Checker error: {str(e)}"]}
        
        results.append({
            "id": id,
            "category": entry["category"],
            "valid": eval_result["valid"],
            "error": eval_result.get("error", []),
            "model_output": formatted_calls,
            "ground_truth": entry["ground_truth"],
            "raw_response": response[:200] + "..." if len(response) > 200 else response
        })

    # Calculate metrics
    stats = {}
    for res in results:
        cat = res["category"]
        if cat not in stats:
            stats[cat] = {"correct": 0, "total": 0}
        stats[cat]["total"] += 1
        if res["valid"]:
            stats[cat]["correct"] += 1
            
    summary = {
        "overall_accuracy": sum(s["correct"] for s in stats.values()) / len(results) if results else 0,
        "category_stats": {cat: s["correct"] / s["total"] for cat, s in stats.items()},
        "total": len(results)
    }
    
    final_output = {
        "summary": summary,
        "results": results
    }
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation complete.")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.2%}")
    for cat, acc in summary['category_stats'].items():
        print(f"  {cat}: {acc:.2%}")

if __name__ == "__main__":
    evaluate()
