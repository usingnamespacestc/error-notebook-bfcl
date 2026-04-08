"""
BFCL V3 数据划分脚本
按类别进行stratified 70/30划分，生成pool.jsonl和test.jsonl
"""
import json
import os
import sys
import random
from collections import Counter

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import load_jsonl, CATEGORIES_WITH_GOLD as CATEGORIES

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PA_DIR = os.path.join(DATA_DIR, "possible_answer")


def main():
    random.seed(SEED)

    pool_data = []
    test_data = []

    for cat in CATEGORIES:
        raw_path = os.path.join(RAW_DIR, f"{cat}.json")
        pa_path = os.path.join(PA_DIR, f"{cat}.json")

        raw_entries = load_jsonl(raw_path)
        pa_entries = load_jsonl(pa_path)

        # Build ID -> ground_truth mapping
        pa_map = {d["id"]: d["ground_truth"] for d in pa_entries}

        # Merge raw + gold, skip entries without matching gold answer
        merged = []
        skipped = 0
        for entry in raw_entries:
            eid = entry["id"]
            if eid not in pa_map:
                skipped += 1
                continue
            merged.append({
                "id": eid,
                "category": cat.replace("BFCL_v3_", ""),
                "question": entry["question"],
                "function": entry["function"],
                "ground_truth": pa_map[eid],
            })

        if skipped:
            print(f"  {cat}: skipped {skipped} entries without matching gold answer")

        # Stratified split: 70% pool, 30% test
        random.shuffle(merged)
        split_idx = int(len(merged) * 0.7)
        pool_data.extend(merged[:split_idx])
        test_data.extend(merged[split_idx:])

        print(f"  {cat}: {len(merged)} total -> {split_idx} pool / {len(merged) - split_idx} test")

    # Shuffle final datasets
    random.shuffle(pool_data)
    random.shuffle(test_data)

    # Save
    pool_path = os.path.join(DATA_DIR, "pool.jsonl")
    test_path = os.path.join(DATA_DIR, "test.jsonl")

    with open(pool_path, "w") as f:
        for entry in pool_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(test_path, "w") as f:
        for entry in test_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(pool_data)} pool + {len(test_data)} test = {len(pool_data) + len(test_data)}")
    print(f"Saved: {pool_path}")
    print(f"Saved: {test_path}")

    # Category distribution check
    print("\nCategory distribution:")
    print(f"  {'Category':<25} {'Pool':>6} {'Test':>6} {'Ratio':>8}")
    pool_cats = Counter(d["category"] for d in pool_data)
    test_cats = Counter(d["category"] for d in test_data)
    for cat in sorted(set(pool_cats) | set(test_cats)):
        p = pool_cats.get(cat, 0)
        t = test_cats.get(cat, 0)
        ratio = p / (p + t) if (p + t) > 0 else 0
        print(f"  {cat:<25} {p:>6} {t:>6} {ratio:>7.1%}")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    main()
