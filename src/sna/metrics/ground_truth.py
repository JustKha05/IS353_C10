"""Module tính độ trùng khớp với Ground Truth motif (BA-Shapes) (Exp 2 - Sanity Check)."""

from typing import List, Tuple, Dict
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def compute_ground_truth_auc(
    candidate_pairs: List[Tuple[int, int]],
    edge_scores: List[float],
    gt_edge_mask: torch.Tensor,
    edge_index: torch.Tensor
) -> Dict[str, float]:
    """Tính ROC-AUC và PR-AUC giữa điểm số explainer và nhãn cạnh ground truth motif."""
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    gt_mask_np = gt_edge_mask.cpu().numpy()

    # Tìm các cặp (min, max) thuộc motif nhà
    gt_pairs = set()
    for j, is_gt in enumerate(gt_mask_np):
        if is_gt:
            gt_pairs.add((int(min(src[j], dst[j])), int(max(src[j], dst[j]))))

    # Nhãn nhị phân cho các cạnh candidate
    y_true = np.array([1 if p in gt_pairs else 0 for p in candidate_pairs])
    y_scores = np.array(edge_scores)

    # Nếu toàn bộ nhãn là 0 hoặc toàn bộ là 1 (không đủ 2 lớp để tính AUC)
    if len(np.unique(y_true)) < 2:
        return {'roc_auc': float('nan'), 'pr_auc': float('nan'), 'gt_edges_count': int(sum(y_true))}

    roc_auc = roc_auc_score(y_true, y_scores)
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(rec, prec)

    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'gt_edges_count': int(sum(y_true))
    }
