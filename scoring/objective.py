import os
import sys
from typing import List, Dict

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scoring.mutual_predictability import MutualPredictabilityScorer
from scoring.diversity_penalty import DiversityScorer

class ObjectiveScorer:
    def __init__(self, mp_scorer: MutualPredictabilityScorer, div_scorer: DiversityScorer, alpha: float = 1.0):
        self.mp_scorer = mp_scorer
        self.div_scorer = div_scorer
        self.alpha = alpha
        
    def score(self, subset: List[Dict]) -> Dict:
        """
        U(S) = α · P_θ(S) - I(S)
        """
        p_theta = self.mp_scorer.score_subset(subset)
        
        subset_ids = [ex["id"] for ex in subset]
        i_s = self.div_scorer.score(subset_ids)
        
        u_s = self.alpha * p_theta - i_s
        
        return {
            "u_s": u_s,
            "p_theta": p_theta,
            "i_s": i_s,
            "alpha": self.alpha
        }
