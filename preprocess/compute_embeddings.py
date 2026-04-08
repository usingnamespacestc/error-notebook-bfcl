import json
import os
import requests
import numpy as np
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "embeddings")
POOL_PATH = os.path.join(DATA_DIR, "pool.jsonl")
TEST_PATH = os.path.join(DATA_DIR, "test.jsonl")
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "all-minilm"

def get_embedding(text):
    payload = {
        "model": MODEL,
        "input": text
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["embeddings"][0]
    except Exception as e:
        print(f"Error getting embedding for text: {text[:50]}... -> {e}")
        return None

def extract_text(entry):
    try:
        if isinstance(entry["question"][0], list):
            return entry["question"][0][0]["content"]
        else:
            return entry["question"][0]["content"]
    except Exception as e:
        print(f"Error extracting text from entry {entry.get('id')}: {e}")
        return ""

def process_file(path, name):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Processing {name} ({path})...")
    with open(path, "r") as f:
        entries = [json.loads(line) for line in f]
    
    embeddings = []
    ids = []
    for entry in tqdm(entries):
        text = extract_text(entry)
        emb = get_embedding(text)
        if emb is not None:
            embeddings.append(emb)
            ids.append(entry["id"])
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    emb_array = np.array(embeddings)
    np.save(os.path.join(OUTPUT_DIR, f"{name}_embeddings.npy"), emb_array)
    with open(os.path.join(OUTPUT_DIR, f"{name}_ids.json"), "w") as f:
        json.dump(ids, f)
    
    print(f"Saved {len(ids)} embeddings to {OUTPUT_DIR}")

def main():
    process_file(POOL_PATH, "pool")
    process_file(TEST_PATH, "test")

if __name__ == "__main__":
    main()
