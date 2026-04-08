"""
BFCL V3 数据探索脚本
分析数据格式、统计各类别数量、估算token长度
"""
import os
import sys
import json

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import load_jsonl, CATEGORIES_WITH_GOLD

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PA_DIR = os.path.join(DATA_DIR, "possible_answer")


def main():
    print("=" * 60)
    print("BFCL V3 Data Exploration")
    print("=" * 60)

    total = 0
    category_stats = {}

    for cat in CATEGORIES_WITH_GOLD:
        raw_path = os.path.join(RAW_DIR, f"{cat}.json")
        pa_path = os.path.join(PA_DIR, f"{cat}.json")

        if not os.path.exists(raw_path) or not os.path.exists(pa_path):
            print(f"  SKIP {cat}: missing files")
            continue

        raw_data = load_jsonl(raw_path)
        pa_data = load_jsonl(pa_path)

        # Check ID alignment
        raw_ids = {d["id"] for d in raw_data}
        pa_ids = {d["id"] for d in pa_data}
        aligned = raw_ids == pa_ids

        # Estimate prompt size (chars)
        prompt_lens = []
        for entry in raw_data:
            q = json.dumps(entry["question"])
            f = json.dumps(entry["function"])
            prompt_lens.append(len(q) + len(f))

        # Gold answer sizes
        gold_lens = []
        num_calls_list = []
        for entry in pa_data:
            gt = entry["ground_truth"]
            gold_lens.append(len(json.dumps(gt)))
            num_calls_list.append(len(gt))

        avg_prompt = sum(prompt_lens) / len(prompt_lens)
        avg_gold = sum(gold_lens) / len(gold_lens)
        avg_calls = sum(num_calls_list) / len(num_calls_list)

        category_stats[cat] = {
            "count": len(raw_data),
            "aligned": aligned,
            "avg_prompt_chars": int(avg_prompt),
            "avg_gold_chars": int(avg_gold),
            "avg_num_calls": round(avg_calls, 1),
        }
        total += len(raw_data)

        print(f"\n{cat}:")
        print(f"  Examples: {len(raw_data)}")
        print(f"  IDs aligned: {aligned}")
        print(f"  Avg prompt size: {int(avg_prompt)} chars")
        print(f"  Avg gold answer size: {int(avg_gold)} chars")
        print(f"  Avg function calls per example: {round(avg_calls, 1)}")

    print(f"\n{'=' * 60}")
    print(f"Total examples with gold answers: {total}")
    print(f"{'=' * 60}")

    # Estimate tokens (using ~4.78 chars/token from our Gemma4 calibration)
    CHARS_PER_TOKEN = 4.78
    print(f"\nToken estimates (@ {CHARS_PER_TOKEN} chars/token):")
    for cat, stats in category_stats.items():
        prompt_tokens = int(stats["avg_prompt_chars"] / CHARS_PER_TOKEN)
        gold_tokens = int(stats["avg_gold_chars"] / CHARS_PER_TOKEN)
        print(f"  {cat}: prompt ~{prompt_tokens} tok, gold ~{gold_tokens} tok")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    main()
