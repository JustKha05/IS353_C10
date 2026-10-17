"""Random Edge Baseline (Kha phụ trách ở T03).
Gán điểm số ngẫu nhiên đều cho các cạnh ứng viên để làm mốc tham chiếu ngẫu nhiên.
"""

import time
from typing import Dict, Any, Optional
import numpy as np
import torch
from sna.core.types import ExplainerResult, CandidateEdgesResult
from sna.explainers.base import BaseExplainer


class RandomEdgeExplainer(BaseExplainer):
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
        m = candidate_edges.num_candidate_edges

        # Sinh điểm ngẫu nhiên đều [0, 1) theo seed
        rng = np.random.RandomState(seed)
        scores = rng.uniform(0.0, 1.0, size=m).tolist()

        return ExplainerResult(
            edge_ids=list(candidate_edges.edge_ids),
            scores=[float(s) for s in scores],
            runtime_sec=time.time() - start_time,
            status="success",
            metadata={'baseline': 'random_edge', 'seed': seed}
        )
