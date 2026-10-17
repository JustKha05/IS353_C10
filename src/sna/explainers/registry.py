"""Explainer Registry & Unified API (BG04)."""

from typing import Dict, Any, Optional, Tuple, List, Union
import torch
from sna.core.types import ExplainerResult, CandidateEdgesResult
from sna.data.candidate_edges import extract_candidate_edges
from .base import BaseExplainer
from .saliency import SaliencyExplainer
from .gnnexplainer import GNNExplainer
from .pgexplainer import PGExplainer, PGExplainerModel
from .baselines.random_edge import RandomEdgeExplainer
from .baselines.nearest_first import NearestFirstExplainer


EXPLAINER_REGISTRY = {
    'saliency': SaliencyExplainer,
    'gnnexplainer': GNNExplainer,
    'pgexplainer': PGExplainer,
    'pge': PGExplainer,
    'random': RandomEdgeExplainer,
    'random_edge': RandomEdgeExplainer,
    'nearest_first': NearestFirstExplainer,
}


def get_explainer(method_name: str, **kwargs) -> BaseExplainer:
    """Factory function lấy đối tượng Explainer theo tên."""
    key = method_name.lower().strip()
    if key not in EXPLAINER_REGISTRY:
        raise ValueError(f"Phương pháp không hợp lệ: '{method_name}'. Hợp lệ: {list(EXPLAINER_REGISTRY.keys())}")
    cls = EXPLAINER_REGISTRY[key]
    return cls(**kwargs)


def explain_edges(
    model: torch.nn.Module,
    data: Dict[str, Any],
    node_id: int,
    target_class: int,
    candidate_edges: Optional[Union[CandidateEdgesResult, Dict]] = None,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 0
) -> Tuple[List[int], List[float], float, str, str]:
    """Hàm giải thích cạnh chuẩn quy ước BG04:
    explain_edges(model, data, node_id, target_class, candidate_edges, config, seed)
    -> (edge_ids, scores, runtime_sec, status, error_message)
    """
    cfg = config or {}
    method_name = cfg.get('method', 'gnnexplainer')

    # Trích xuất candidate edges nếu chưa có
    if candidate_edges is None:
        cand_obj = extract_candidate_edges(data['edge_index'], node_id, num_hops=2)
    elif isinstance(candidate_edges, dict):
        cand_obj = CandidateEdgesResult(
            node_id=node_id,
            edge_ids=candidate_edges['edge_ids'],
            candidate_pairs=candidate_edges['candidate_pairs'],
            subgraph_nodes=candidate_edges['subgraph_nodes'],
            subgraph_edge_index=candidate_edges['subgraph_edge_index'],
            edge_to_pair_idx=candidate_edges['edge_to_pair_idx'],
            num_candidate_edges=candidate_edges['num_candidate_edges']
        )
    else:
        cand_obj = candidate_edges

    explainer = get_explainer(method_name)
    result = explainer.explain(
        model=model,
        data=data,
        node_id=node_id,
        target_class=target_class,
        candidate_edges=cand_obj,
        config=cfg,
        seed=seed
    )

    return result.to_tuple()
