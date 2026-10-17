"""GNNExplainer (Khôi phụ trách).
Tối ưu continuous edge mask bằng Cross-Entropy + Sparsity L1 + Entropy (BG06).
"""

import time
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
from sna.core.types import ExplainerResult, CandidateEdgesResult
from sna.explainers.base import BaseExplainer


class GNNExplainer(BaseExplainer):
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
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            cfg = config or {}
            epochs = cfg.get('epochs', 200)
            lr = cfg.get('lr', 0.01)
            coeff_size = cfg.get('coeff_size', 0.005)
            coeff_ent = cfg.get('coeff_ent', 0.1)

            x = data['x']
            edge_index = data['edge_index']
            device = next(model.parameters()).device
            model.eval()

            sub_edge_index = candidate_edges.subgraph_edge_index.to(device)
            edge_to_pair_idx = candidate_edges.edge_to_pair_idx.to(device)
            num_candidates = candidate_edges.num_candidate_edges

            mask_logits = torch.nn.Parameter(torch.randn(num_candidates, device=device) * 0.1 + 1.0)
            optimizer = torch.optim.Adam([mask_logits], lr=lr)

            num_total_edges = edge_index.size(1)
            base_edge_weight = torch.ones(num_total_edges, device=device)

            src_sub = sub_edge_index[0].cpu().numpy()
            dst_sub = sub_edge_index[1].cpu().numpy()
            src_all = edge_index[0].cpu().numpy()
            dst_all = edge_index[1].cpu().numpy()
            all_edge_map = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(src_all, dst_all))}

            sub_to_orig_indices = torch.tensor(
                [all_edge_map[(int(u), int(v))] for u, v in zip(src_sub, dst_sub)],
                dtype=torch.long, device=device
            )

            x_dev = x.to(device)
            edge_index_dev = edge_index.to(device)

            for epoch in range(epochs):
                optimizer.zero_grad()
                pair_probs = torch.sigmoid(mask_logits)
                sub_probs = pair_probs[edge_to_pair_idx]

                current_weight = base_edge_weight.clone()
                current_weight[sub_to_orig_indices] = sub_probs

                h = model.c1(x_dev, edge_index_dev, edge_weight=current_weight)
                h = F.relu(h)
                out = model.c2(h, edge_index_dev, edge_weight=current_weight)

                log_probs = F.log_softmax(out[node_id:node_id+1], dim=-1)
                loss_pred = -log_probs[0, target_class]
                loss_size = coeff_size * torch.mean(pair_probs)

                eps = 1e-12
                ent = -pair_probs * torch.log(pair_probs + eps) - (1 - pair_probs) * torch.log(1 - pair_probs + eps)
                loss_ent = coeff_ent * torch.mean(ent)

                loss = loss_pred + loss_size + loss_ent
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                final_scores = torch.sigmoid(mask_logits).detach().cpu().numpy().tolist()

            return ExplainerResult(
                edge_ids=list(candidate_edges.edge_ids),
                scores=[float(s) for s in final_scores],
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
