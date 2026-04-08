import numpy as np
import json
from typing import List, Dict

class DiversityScorer:
    def __init__(self, embeddings_path: str, ids_path: str):
        self.embeddings = np.load(embeddings_path)
        with open(ids_path, "r") as f:
            self.ids = json.load(f)
        self.id_to_idx = {id_: idx for idx, id_ in enumerate(self.ids)}
        
        # Precompute similarity matrix if not too large (1818 x 1818 is small)
        norm = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized_embeddings = self.embeddings / (norm + 1e-9)
        self.sim_matrix = np.matmul(normalized_embeddings, normalized_embeddings.T)

    def score(self, subset_ids: List[str]) -> float:
        """
        Calculates diversity penalty: I(S) = Σ_i Σ_{j≠i} cosine_similarity(emb(x_i), emb(x_j))
        """
        idxs = [self.id_to_idx[id_] for id_ in subset_ids if id_ in self.id_to_idx]
        if len(idxs) < 2:
            return 0.0
            
        sub_matrix = self.sim_matrix[np.ix_(idxs, idxs)]
        # Sum all elements except diagonal
        total_sim = np.sum(sub_matrix) - np.trace(sub_matrix)
        # Note: PLAN.md says Σ_i Σ_{j≠i}, which counts each pair twice (i,j) and (j,i).
        # This is consistent with total_sim as calculated above.
        return float(total_sim)

if __name__ == "__main__":
    import json
    import os
    # Test
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    scorer = DiversityScorer(
        os.path.join(BASE_DIR, "data", "embeddings", "pool_embeddings.npy"),
        os.path.join(BASE_DIR, "data", "embeddings", "pool_ids.json")
    )
    test_ids = scorer.ids[:5]
    print(f"Diversity score for first 5: {scorer.score(test_ids)}")
