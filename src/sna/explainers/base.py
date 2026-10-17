"""Base interface cho mọi Explainer trong project (BG04)."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
from sna.core.types import ExplainerResult, CandidateEdgesResult


class BaseExplainer(ABC):
    """Lớp cơ sở trừu tượng cho tất cả các phương pháp giải thích cạnh."""

    @abstractmethod
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
        """Hàm giải thích cạnh trả về ExplainerResult theo chuẩn BG04."""
        pass
