import json
from typing import List, Dict

CATEGORIES_WITH_GOLD = [
    "BFCL_v3_simple",
    "BFCL_v3_multiple",
    "BFCL_v3_parallel",
    "BFCL_v3_parallel_multiple",
    "BFCL_v3_live_simple",
    "BFCL_v3_live_multiple",
    "BFCL_v3_live_parallel",
    "BFCL_v3_live_parallel_multiple",
    "BFCL_v3_java",
    "BFCL_v3_javascript",
    "BFCL_v3_sql",
]

def load_jsonl(path: str) -> List[Dict]:
    result = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                result.append(json.loads(line))
    return result
