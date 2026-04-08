import json
import os
import sys
import random
import numpy as np
from typing import List, Dict, Set
from tqdm import tqdm
import math
import time

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scoring.teacher_forcing import TeacherForcingScorer
from scoring.mutual_predictability import MutualPredictabilityScorer
from scoring.diversity_penalty import DiversityScorer
from scoring.objective import ObjectiveScorer

class SimulatedAnnealingSearch:
    def __init__(
        self,
        pool: List[Dict],
        objective_scorer: ObjectiveScorer,
        k: int = 5,
        t0: float = 10.0,
        decay: float = 0.995,
        max_iterations: int = 500
    ):
        self.pool = pool
        self.objective_scorer = objective_scorer
        self.k = k
        self.t0 = t0
        self.decay = decay
        self.max_iterations = max_iterations
        
    def run(self, log_path: str = None) -> Dict:
        # Initial state: random sample from pool
        current_indices = random.sample(range(len(self.pool)), self.k)
        current_subset = [self.pool[i] for i in current_indices]
        
        current_score_dict = self.objective_scorer.score(current_subset)
        current_u = current_score_dict["u_s"]
        
        best_subset = list(current_subset)
        best_u = current_u
        best_score_dict = dict(current_score_dict)
        
        log = []
        temp = self.t0
        
        print(f"Starting Simulated Annealing...")
        print(f"Initial U: {current_u:.4f} (P_theta: {current_score_dict['p_theta']:.4f}, I_s: {current_score_dict['i_s']:.4f})")
        
        for i in range(self.max_iterations):
            # Neighborhood operation: replace one sample
            new_indices = list(current_indices)
            idx_to_replace = random.randrange(self.k)
            
            # Find a new sample not in current subset
            available_pool_indices = set(range(len(self.pool))) - set(current_indices)
            new_sample_idx = random.choice(list(available_pool_indices))
            
            new_indices[idx_to_replace] = new_sample_idx
            new_subset = [self.pool[idx] for idx in new_indices]
            
            new_score_dict = self.objective_scorer.score(new_subset)
            new_u = new_score_dict["u_s"]
            
            delta_u = new_u - current_u
            
            accepted = False
            if delta_u > 0:
                accepted = True
            else:
                prob = math.exp(delta_u / (temp + 1e-9))
                if random.random() < prob:
                    accepted = True
            
            if accepted:
                current_indices = new_indices
                current_subset = new_subset
                current_u = new_u
                current_score_dict = new_score_dict
                
                if current_u > best_u:
                    best_u = current_u
                    best_subset = list(current_subset)
                    best_score_dict = dict(current_score_dict)
                    print(f"Iteration {i}: New Best U: {best_u:.4f}")
            
            # Log
            log_entry = {
                "iteration": i,
                "u_s": current_u,
                "p_theta": current_score_dict["p_theta"],
                "i_s": current_score_dict["i_s"],
                "temp": temp,
                "accepted": accepted,
                "delta_u": delta_u,
                "subset_ids": [ex["id"] for ex in current_subset]
            }
            log.append(log_entry)
            
            if log_path:
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            
            # Update temp
            temp *= self.decay
            
            # Optional: print every 10 iterations
            if i % 1 == 0: # Print every iteration for better visibility during dev
                print(f"Iteration {i}: U={current_u:.4f}, Best U={best_u:.4f}, T={temp:.4f}")
                
        return {
            "best_subset": best_subset,
            "best_u": best_u,
            "best_scores": best_score_dict,
            "log": log
        }

def main():
    BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
    POOL_PATH = os.path.join(BASE_DIR, "data", "pool.jsonl")
    EMB_PATH = os.path.join(BASE_DIR, "data", "embeddings", "pool_embeddings.npy")
    IDS_PATH = os.path.join(BASE_DIR, "data", "embeddings", "pool_ids.json")
    RESULT_DIR = os.path.join(BASE_DIR, "results")
    LOG_PATH = os.path.join(RESULT_DIR, "annealing_log.jsonl")
    
    os.makedirs(RESULT_DIR, exist_ok=True)
    # Clear existing log
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    
    with open(POOL_PATH, "r") as f:
        pool = [json.loads(line) for line in f]
        
    tf_scorer = TeacherForcingScorer()
    mp_scorer = MutualPredictabilityScorer(tf_scorer)
    div_scorer = DiversityScorer(EMB_PATH, IDS_PATH)
    
    objective_scorer = ObjectiveScorer(mp_scorer, div_scorer, alpha=0.01)
    
    search = SimulatedAnnealingSearch(
        pool=pool,
        objective_scorer=objective_scorer,
        k=5,
        t0=1.0,
        decay=0.995, # Slower decay for more exploration
        max_iterations=200
    )
    
    results = search.run(log_path=LOG_PATH)
    
    # Save best subset
    with open(os.path.join(RESULT_DIR, "selected_subset.json"), "w") as f:
        json.dump(results["best_subset"], f, indent=2, ensure_ascii=False)
            
    print(f"Search complete. Best U: {results['best_u']}")

if __name__ == "__main__":
    main()
