"""
Select representative error examples using K-Medoids clustering.
For each error type, selects the most representative samples.
Also selects correct examples for the "positive" part of the error notebook.

Output: error_notebook_subset.json containing:
  - correct_examples: k correct examples (diverse, representative)
  - error_examples: k incorrect examples with their error types and model outputs
"""
import json
import os
import sys
import argparse
import numpy as np
from typing import List, Dict
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

POOL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pool.jsonl")
CLASSIFIED_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "pool_classified.jsonl")
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings", "pool_embeddings.npy")
IDS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings", "pool_ids.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "error_notebook_subset.json")


def kmedoids_select(embeddings: np.ndarray, k: int) -> List[int]:
    """
    Simple K-Medoids (PAM) selection.
    Returns indices of k medoid points from embeddings.
    """
    from sklearn_extra.cluster import KMedoids

    n = embeddings.shape[0]
    if n <= k:
        return list(range(n))

    km = KMedoids(n_clusters=k, metric="cosine", random_state=42, max_iter=300)
    km.fit(embeddings)
    return km.medoid_indices_.tolist()


def kmedoids_select_fallback(embeddings: np.ndarray, k: int) -> List[int]:
    """
    Fallback K-Medoids using sklearn KMeans + nearest point selection.
    Used when sklearn_extra is not available.
    """
    from sklearn.cluster import KMeans

    n = embeddings.shape[0]
    if n <= k:
        return list(range(n))

    # Normalize for cosine-like behavior
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = embeddings / norms

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(normed)

    # For each cluster center, find the nearest actual data point
    medoid_indices = []
    for center in kmeans.cluster_centers_:
        dists = np.linalg.norm(normed - center, axis=1)
        idx = np.argmin(dists)
        # Avoid duplicates
        while idx in medoid_indices:
            dists[idx] = np.inf
            idx = np.argmin(dists)
        medoid_indices.append(idx)

    return medoid_indices


def maxsum_diversity_select(embeddings: np.ndarray, k: int) -> List[int]:
    """
    Greedy MaxSum diversity selection.
    Selects k points that maximize pairwise distance (diversity).
    """
    n = embeddings.shape[0]
    if n <= k:
        return list(range(n))

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = embeddings / norms

    # Cosine similarity matrix
    sim = normed @ normed.T

    # Start with the point closest to centroid (most representative)
    centroid = normed.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    first = np.argmax(normed @ centroid)

    selected = [first]
    for _ in range(k - 1):
        # For each candidate, compute max similarity to already selected
        max_sims = sim[:, selected].max(axis=1)
        # Mask already selected
        max_sims[selected] = np.inf
        # Select the one with minimum max-similarity (most different from all selected)
        next_idx = np.argmin(max_sims)
        selected.append(next_idx)

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Number of examples per side (k correct + k error)")
    parser.add_argument("--method", choices=["kmedoids", "diversity"], default="kmedoids",
                        help="Selection method")
    parser.add_argument("--classified", type=str, default=CLASSIFIED_PATH,
                        help="Path to classified JSONL")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH,
                        help="Output path for subset JSON")
    args = parser.parse_args()
    k = args.k

    # Load pool data
    pool_map = {}
    with open(POOL_PATH, "r") as f:
        for line in f:
            entry = json.loads(line)
            pool_map[entry["id"]] = entry

    # Load classified results
    classified = []
    with open(args.classified, "r") as f:
        for line in f:
            classified.append(json.loads(line))

    # Load embeddings
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(IDS_PATH, "r") as f:
        pool_ids = json.load(f)
    id_to_emb_idx = {pid: i for i, pid in enumerate(pool_ids)}

    # Split into correct and incorrect
    correct_entries = [c for c in classified if c["correct"]]
    error_entries = [c for c in classified if not c["correct"]]

    print(f"Correct: {len(correct_entries)}, Errors: {len(error_entries)}")

    # === Select k correct examples (diverse representatives) ===
    correct_ids = [c["id"] for c in correct_entries if c["id"] in id_to_emb_idx]
    correct_emb_indices = [id_to_emb_idx[cid] for cid in correct_ids]
    correct_embeddings = embeddings[correct_emb_indices]

    select_fn = kmedoids_select if args.method == "kmedoids" else maxsum_diversity_select

    if args.method == "kmedoids":
        correct_selected_local = kmedoids_select_fallback(correct_embeddings, k)
    else:
        correct_selected_local = maxsum_diversity_select(correct_embeddings, k)

    correct_selected_ids = [correct_ids[i] for i in correct_selected_local]
    print(f"\nSelected {len(correct_selected_ids)} correct examples:")
    for cid in correct_selected_ids:
        entry = pool_map[cid]
        print(f"  {cid} ({entry['category']})")

    # === Select k error examples (proportional to error type frequency, diverse within type) ===
    # Group errors by type
    error_by_type = {}
    for e in error_entries:
        et = e["error_type"]
        if et not in error_by_type:
            error_by_type[et] = []
        error_by_type[et].append(e)

    print(f"\nError type distribution:")
    for et, entries in sorted(error_by_type.items(), key=lambda x: -len(x[1])):
        print(f"  {et}: {len(entries)}")

    # Allocate budget proportionally to error type frequency
    total_errors = len(error_entries)
    type_budgets = {}
    remaining = k
    sorted_types = sorted(error_by_type.items(), key=lambda x: -len(x[1]))

    for et, entries in sorted_types:
        # Proportional allocation, at least 1 per type (if budget allows)
        budget = max(1, round(k * len(entries) / total_errors))
        budget = min(budget, remaining, len(entries))
        if remaining <= 0:
            budget = 0
        type_budgets[et] = budget
        remaining -= budget

    # If we have leftover budget, give to largest types
    if remaining > 0:
        for et, entries in sorted_types:
            if remaining <= 0:
                break
            extra = min(remaining, len(entries) - type_budgets[et])
            if extra > 0:
                type_budgets[et] += extra
                remaining -= extra

    print(f"\nBudget allocation (total k={k}):")
    for et, budget in type_budgets.items():
        if budget > 0:
            print(f"  {et}: {budget}")

    # Select within each error type
    error_selected = []
    for et, budget in type_budgets.items():
        if budget <= 0:
            continue
        type_entries = error_by_type[et]
        type_ids = [e["id"] for e in type_entries if e["id"] in id_to_emb_idx]

        if not type_ids:
            continue

        type_emb_indices = [id_to_emb_idx[tid] for tid in type_ids]
        type_embeddings = embeddings[type_emb_indices]

        if args.method == "kmedoids":
            selected_local = kmedoids_select_fallback(type_embeddings, budget)
        else:
            selected_local = maxsum_diversity_select(type_embeddings, budget)

        for idx in selected_local:
            selected_id = type_ids[idx]
            # Find the classified entry
            entry_cls = next(e for e in type_entries if e["id"] == selected_id)
            error_selected.append({
                "id": selected_id,
                "error_type": et,
                "error": entry_cls["error"],
                "model_output": entry_cls["model_output"],
            })

    print(f"\nSelected {len(error_selected)} error examples:")
    for es in error_selected:
        entry = pool_map[es["id"]]
        print(f"  {es['id']} ({entry['category']}) — {es['error_type']}")

    # === Build output ===
    output = {
        "k": k,
        "method": args.method,
        "correct_examples": [],
        "error_examples": [],
    }

    for cid in correct_selected_ids:
        entry = pool_map[cid]
        output["correct_examples"].append(entry)

    for es in error_selected:
        entry = pool_map[es["id"]]
        output["error_examples"].append({
            **entry,
            "error_type": es["error_type"],
            "error_detail": es["error"],
            "model_wrong_output": es["model_output"],
        })

    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
