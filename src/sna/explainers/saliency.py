"""Saliency Explainer (Khôi phụ trách).
Tính gradient tuyệt đối của logit mục tiêu theo cạnh (|grad|), gộp max 2 chiều (BG06).
"""

import time
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
from sna.core.types import ExplainerResult, CandidateEdgesResult
from sna.explainers.base import BaseExplainer


class SaliencyExplainer(BaseExplainer):
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
        try:
            x = data['x']
            edge_index = data['edge_index']
            device = next(model.parameters()).device
            model.eval()

            num_edges = edge_index.size(1)
            edge_weight = torch.ones(num_edges, device=device, requires_grad=True)

            x_dev = x.to(device)
            edge_index_dev = edge_index.to(device)

            h = model.c1(x_dev, edge_index_dev, edge_weight=edge_weight)
            h = F.relu(h)
            out = model.c2(h, edge_index_dev, edge_weight=edge_weight)

            target_score = out[node_id, target_class]
            target_score.backward()

            grad = edge_weight.grad
            if grad is None:
                raise ValueError("Không tính được gradient theo edge_weight.")

            abs_grad = torch.abs(grad).detach().cpu()

            # Mapping cạnh
            src = edge_index[0].cpu().numpy()
            dst = edge_index[1].cpu().numpy()
            edge_lookup = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(src, dst))}

            scores = []
            for (u, v) in candidate_edges.candidate_pairs:
                idx_uv = edge_lookup.get((u, v))
                idx_vu = edge_lookup.get((v, u))

                val_uv = abs_grad[idx_uv].item() if idx_uv is not None else 0.0
                val_vu = abs_grad[idx_vu].item() if idx_vu is not None else 0.0

                score = max(val_uv, val_vu)
                scores.append(float(score))

            return ExplainerResult(
                edge_ids=list(candidate_edges.edge_ids),
                scores=scores,
                runtime_sec=time.time() - start_time,
                status="success"
            )

        except Exception as e:
            return ExplainerResult(
                edge_ids=[],
                scores=[],
                runtime_sec=time.time() - start_time,
                status="failed",
                error_message=str(e)
            )
