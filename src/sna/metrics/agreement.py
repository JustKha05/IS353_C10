"""Module tính độ đồng thuận (Agreement) giữa các Explainer: Jaccard, Cohen's Kappa, Rank Correlation."""

from typing import List, Set, Any
import numpy as np
from scipy.stats import kendalltau, spearmanr


def compute_jaccard_at_k(scores_1: List[float], scores_2: List[float], k: int) -> float:
    """Tính độ tương đồng Jaccard của top-k cạnh quan trọng giữa 2 Explainer."""
    if len(scores_1) == 0 or len(scores_2) == 0:
        return 0.0

    k = min(k, len(scores_1))
    top_1 = set(np.argsort(scores_1)[::-1][:k])
    top_2 = set(np.argsort(scores_2)[::-1][:k])

    intersection = len(top_1.intersection(top_2))
    union = len(top_1.union(top_2))
    return float(intersection / union) if union > 0 else 0.0


def compute_kendall_tau(scores_1: List[float], scores_2: List[float]) -> float:
    """Tính hệ số tương quan thứ hạng Kendall's tau-b."""
    if len(scores_1) < 2:
        return 0.0
    res = kendalltau(scores_1, scores_2)
    val = res.correlation if hasattr(res, 'correlation') else res[0]
    return float(val) if not np.isnan(val) else 0.0
