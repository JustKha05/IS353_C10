"""Nearest-first Baseline (Kha phụ trách ở T03).
Xếp hạng cạnh theo khoảng cách tới node đích (1-hop trước, 2-hop sau).
Hòa điểm được chọn ngẫu nhiên bằng seed theo quy ước BG08.
"""

import time
from typing import Dict, Any, Optional
import numpy as np
import torch
from sna.core.types import ExplainerResult, CandidateEdgesResult
from sna.explainers.base import BaseExplainer


class NearestFirstExplainer(BaseExplainer):
    def explain(
        self,
        model: torch.nn.Module,
        data: Dict[str, Any],
        node_id: int,
        target_class: int,
        candidate_edges: CandidateEdgesResult,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 0
    ) -> ExplainerResult:
        start_time = time.time()
        rng = np.random.RandomState(seed)
        scores = []

        # Cạnh nối trực tiếp với node_id (1-hop) có khoảng cách = 1 -> điểm cao hơn
        # Cạnh 2-hop có khoảng cách = 2 -> điểm thấp hơn
        for (u, v) in candidate_edges.candidate_pairs:
            tie_breaker = rng.uniform(0.0, 0.05)
            if u == node_id or v == node_id:
                # 1-hop edge
                score = 1.0 + tie_breaker
            else:
                # 2-hop edge
                score = 0.5 + tie_breaker
            scores.append(float(score))

        return ExplainerResult(
            edge_ids=list(candidate_edges.edge_ids),
            scores=scores,
            runtime_sec=time.time() - start_time,
            status="success",
            metadata={'baseline': 'nearest_first', 'seed': seed}
        )
