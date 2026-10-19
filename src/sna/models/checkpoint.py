"""Quản lý nạp và lưu Checkpoint cho các mô hình GNN."""

import pathlib
from typing import Dict, Tuple, Optional
import torch
from sna.core.config import DATA_DIR
from .gcn import GCN


def load_gcn_checkpoint(
    dataset_name: str,
    root: Optional[pathlib.Path] = None
) -> Tuple[GCN, torch.Tensor, Dict]:
    """Nạp GCN model đã khóa eval() và pred từ checkpoint (BG02, BG03)."""
    root_path = root or DATA_DIR
    ckpt_path = root_path / dataset_name.lower().strip() / "gcn_ckpt.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GCN(
        in_dim=ck["in_dim"],
        hidden_dim=ck["hidden"],
        out_dim=ck["n_classes"],
        dropout=ck.get("dropout", 0.0)
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()

    # Khóa toàn bộ tham số
    for p in model.parameters():
        p.requires_grad_(False)

    return model, ck["pred"], ck.get("metrics", {})
