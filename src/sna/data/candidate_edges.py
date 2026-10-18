"""Trích xuất tập cạnh ứng viên trong k-hop neighborhood (BG05, BG07)."""

from typing import Dict, List, Tuple, Optional
import torch
from torch_geometric.utils import k_hop_subgraph
from sna.core.types import CandidateEdgesResult


def extract_candidate_edges(
    edge_index: torch.Tensor,
    node_id: int,
    num_hops: int = 2,
    num_nodes: Optional[int] = None
) -> CandidateEdgesResult:
    """Trích xuất tập cạnh ứng viên 2-hop cho node_id, loại self-loops, mapping (min, max)."""
    if edge_index.numel() == 0 or edge_index.size(1) == 0:
        return CandidateEdgesResult(
            node_id=node_id,
            edge_ids=[],
            candidate_pairs=[],
            subgraph_nodes=torch.tensor([node_id], device=edge_index.device, dtype=torch.long),
            subgraph_edge_index=torch.empty((2, 0), dtype=torch.long, device=edge_index.device),
            edge_to_pair_idx=torch.empty(0, dtype=torch.long, device=edge_index.device),
            num_candidate_edges=0
        )

    if num_nodes is None:
        max_id = int(edge_index.max().item()) if edge_index.numel() > 0 else 0
        num_nodes = max(max_id + 1, node_id + 1)
    else:
        num_nodes = max(num_nodes, node_id + 1)

    subset, sub_edge_index, _, _ = k_hop_subgraph(
        node_idx=node_id,
        num_hops=num_hops,
        edge_index=edge_index,
        relabel_nodes=False,
        num_nodes=num_nodes
    )

    # Loại bỏ self-loops (u == v)
    non_self_loop = sub_edge_index[0] != sub_edge_index[1]
    sub_edge_index = sub_edge_index[:, non_self_loop]

    src = sub_edge_index[0].cpu().numpy()
    dst = sub_edge_index[1].cpu().numpy()

    # Gom cạnh vô hướng thành canonical pair (min, max) theo BG07
    pair_list: List[Tuple[int, int]] = []
    seen = set()

    for u, v in zip(src, dst):
        pair = (int(min(u, v)), int(max(u, v)))
        if pair not in seen:
            seen.add(pair)
            pair_list.append(pair)

    edge_ids = list(range(len(pair_list)))
    pair_to_id = {p: i for i, p in enumerate(pair_list)}

    edge_to_pair_idx = torch.tensor(
        [pair_to_id[(int(min(u, v)), int(max(u, v)))] for u, v in zip(src, dst)],
        dtype=torch.long,
        device=edge_index.device
    )

    return CandidateEdgesResult(
        node_id=node_id,
        edge_ids=edge_ids,
        candidate_pairs=pair_list,
        subgraph_nodes=subset,
        subgraph_edge_index=sub_edge_index,
        edge_to_pair_idx=edge_to_pair_idx,
        num_candidate_edges=len(pair_list)
    )
