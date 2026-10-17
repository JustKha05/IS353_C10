"""Module tính toán Fidelity (Fid-, Fid+, k*) theo quy ước BG08, BG09.
Xử lý đồng thời cả hai chiều của cạnh vô hướng theo BG07.
"""

from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F


def compute_fidelity(
    model: torch.nn.Module,
    data: Dict,
    node_id: int,
    target_class: int,
    candidate_pairs: List[Tuple[int, int]],
    edge_scores: List[float],
    top_k: int = 5
) -> Dict[str, float]:
    """Tính Fidelity- (xóa cạnh quan trọng nhất) và Fidelity+ (chỉ giữ cạnh quan trọng).

    Args:
        model: GCN model (eval).
        data: Dict dữ liệu (x, edge_index,...).
        node_id: Node cần đánh giá.
        target_class: Lớp dự đoán gốc của GCN.
        candidate_pairs: Danh sách các cặp cạnh vô hướng (u, v).
        edge_scores: Điểm số tương ứng cho mỗi cặp cạnh.
        top_k: Ngân sách số cạnh quan trọng cần chọn.

    Returns:
        dict {'fid_drop': Fid-, 'fid_inv': Fid+, 'orig_prob': float, 'unmasked_prob': float}
    """
    model.eval()
    device = next(model.parameters()).device
    x = data['x'].to(device)
    edge_index = data['edge_index'].to(device)

    # 1. Đo xác suất ban đầu của target_class trên đồ thị gốc
    with torch.no_grad():
        orig_out = model(x, edge_index)
        orig_prob = F.softmax(orig_out[node_id], dim=-1)[target_class].item()

    if len(candidate_pairs) == 0:
        return {'fid_drop': 0.0, 'fid_inv': 0.0, 'orig_prob': orig_prob}

    # 2. Xếp hạng cạnh theo điểm số (giảm dần)
    k = min(top_k, len(candidate_pairs))
    sorted_indices = np.argsort(edge_scores)[::-1]
    top_pairs = set(candidate_pairs[idx] for idx in sorted_indices[:k])

    # 3. Tính Fid- (Fidelity Deletion / Drop): Xóa cả 2 chiều của top_k cạnh
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    del_mask = []
    for u, v in zip(src, dst):
        pair = (int(min(u, v)), int(max(u, v)))
        # Giữ lại cạnh nếu KHÔNG nằm trong top_pairs (tức là xóa top_pairs)
        del_mask.append(pair not in top_pairs)

    del_edge_index = edge_index[:, torch.tensor(del_mask, dtype=torch.bool, device=device)]
    with torch.no_grad():
        del_out = model(x, del_edge_index)
        del_prob = F.softmax(del_out[node_id], dim=-1)[target_class].item()

    fid_drop = orig_prob - del_prob

    # 4. Tính Fid+ (Fidelity Preservation / Infidelity): Chỉ giữ lại top_k cạnh
    keep_mask = []
    for u, v in zip(src, dst):
        pair = (int(min(u, v)), int(max(u, v)))
        keep_mask.append(pair in top_pairs)

    # Đảm bảo không rỗng
    if sum(keep_mask) > 0:
        keep_edge_index = edge_index[:, torch.tensor(keep_mask, dtype=torch.bool, device=device)]
        with torch.no_grad():
            keep_out = model(x, keep_edge_index)
            keep_prob = F.softmax(keep_out[node_id], dim=-1)[target_class].item()
        fid_inv = orig_prob - keep_prob
    else:
        fid_inv = orig_prob

    return {
        'fid_drop': float(fid_drop),     # Fid- (càng cao càng tốt: xóa cạnh quan trọng làm giảm mạnh xác suất)
        'fid_inv': float(fid_inv),       # Fid+ (càng thấp càng tốt: chỉ giữ cạnh quan trọng thì xác suất gần như không đổi)
        'orig_prob': float(orig_prob),
        'k': k
    }
