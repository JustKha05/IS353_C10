"""Core domain types and dataclasses.
Tuân thủ các quy ước BG04, BG05, BG06, BG07, BG10, BG11.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch


@dataclass
class CandidateEdgesResult:
    """Tập cạnh ứng viên được trích xuất cho một target node (BG05, BG07)."""
    node_id: int
    edge_ids: List[int]
    candidate_pairs: List[Tuple[int, int]]
    subgraph_nodes: torch.Tensor
    subgraph_edge_index: torch.Tensor
    edge_to_pair_idx: torch.Tensor
    num_candidate_edges: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'edge_ids': self.edge_ids,
            'candidate_pairs': self.candidate_pairs,
            'subgraph_nodes': self.subgraph_nodes,
            'subgraph_edge_index': self.subgraph_edge_index,
            'edge_to_pair_idx': self.edge_to_pair_idx,
            'num_candidate_edges': self.num_candidate_edges
        }


@dataclass
class ExplainerResult:
    """Kết quả giải thích trả về từ hàm explain_edges (BG04)."""
    edge_ids: List[int]
    scores: List[float]
    runtime_sec: float
    status: str                         # 'success' | 'failed' | 'skipped'
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_tuple(self) -> Tuple[List[int], List[float], float, str, str]:
        """Tương thích ngược 100% với tuple BG04."""
        return self.edge_ids, self.scores, self.runtime_sec, self.status, self.error_message


@dataclass
class MetricResult:
    """Kết quả tính toán chỉ số đánh giá (Fidelity, Agreement, AUROC) (BG09, BG10)."""
    metric_name: str
    value: float
    budget_type: str                    # 'top_k' | 'ratio' | 'threshold'
    budget_val: float                   # k=1, 3, 5, 10 hoặc p=0.1, 0.2, 0.5
    method_name: str
    node_id: int
    details: Dict[str, Any] = field(default_factory=dict)
