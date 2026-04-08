import json
from typing import List, Dict, Any

def convert_bfcl_to_ollama_tc(ground_truth: List[Dict]) -> List[Dict]:
    """
    Converts BFCL ground_truth format to Ollama tool_calls format.
    BFCL: [{"func_name": {"param": ["val1", "val2"]}}]
    Ollama tool_calls: [{"function": {"name": "func_name", "arguments": {"param": "val1"}}}]

    This is the format used in assistant message's tool_calls field.
    """
    tool_calls = []
    for call in ground_truth:
        for func_name, args in call.items():
            simplified_args = {}
            for arg_name, arg_val in args.items():
                if isinstance(arg_val, list) and len(arg_val) > 0:
                    # Skip empty strings (means "optional, can omit")
                    non_empty = [v for v in arg_val if v != ""]
                    if non_empty:
                        simplified_args[arg_name] = non_empty[0]
                    # If all values are "", skip this optional param entirely
                else:
                    simplified_args[arg_name] = arg_val

            tool_calls.append({
                "function": {
                    "name": func_name,
                    "arguments": simplified_args
                }
            })
    return tool_calls


def build_fewshot_messages(subset: List[Dict]) -> List[Dict]:
    """
    Build few-shot messages using proper Ollama tool_calls format.
    Each example becomes: user message + assistant message with tool_calls field.
    """
    messages = []
    for ex in subset:
        # User question
        if isinstance(ex["question"][0], list):
            q = ex["question"][0][0]["content"]
        else:
            q = ex["question"][0]["content"]
        messages.append({"role": "user", "content": q})

        # Assistant tool call (using tool_calls field, not content)
        tool_calls = convert_bfcl_to_ollama_tc(ex["ground_truth"])
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls
        })

    return messages

def merge_tools(target_tools: List[Dict], few_shot_entries: List[Dict]) -> List[Dict]:
    """
    Merges tool definitions from target and all few-shot entries to ensure
    model has access to all referred tools.
    """
    all_tools_map = {}
    
    # Process target tools
    for tool in target_tools:
        name = tool["name"]
        all_tools_map[name] = tool
        
    # Process few-shot tools
    for entry in few_shot_entries:
        for tool in entry["function"]:
            name = tool["name"]
            if name not in all_tools_map:
                all_tools_map[name] = tool
                
    return list(all_tools_map.values())
