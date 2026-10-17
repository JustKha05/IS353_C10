"""Bộ nạp dữ liệu thống nhất (Dataset Loader) cho toàn bộ đồ án.
Hỗ trợ BA-Shapes, Cora và sẵn sàng mở rộng cho Amazon Photo, Coauthor CS, ogbn-arxiv,...
"""

import json
import pathlib
from typing import Dict, List, Optional, Tuple
import torch
from sna.core.config import DATA_DIR


def load_dataset(
    dataset_name: str,
    root: Optional[pathlib.Path] = None
) -> Tuple[Dict, torch.Tensor, List[int]]:
    """Nạp dữ liệu đồ thị, danh sách dự đoán gốc, và danh sách node validation.

    Args:
        dataset_name: 'bashapes' | 'cora' (hoặc tên dataset mở rộng sau này).
        root: Thư mục chứa dữ liệu (mặc định t01_handover).

    Returns:
        (data_dict, pred_tensor, val_nodes_list)
    """
    root_path = root or DATA_DIR
    ds_dir = root_path / dataset_name.lower().strip()

    if not ds_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {ds_dir}")

    # Nạp data.pt
    data = torch.load(ds_dir / "data.pt", weights_only=False)

    # Nạp pred từ checkpoint nếu có
    ckpt_path = ds_dir / "gcn_ckpt.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        pred = ckpt.get("pred", None)
    else:
        pred = None

    # Nạp val_nodes
    val_nodes_path = ds_dir / "val_nodes.json"
    if val_nodes_path.exists():
        with open(val_nodes_path, "r", encoding="utf-8") as f:
            val_nodes = json.load(f)["nodes"]
    else:
        # Nếu chưa có thì lấy từ val_mask
        val_mask = data.get("val_mask")
        val_nodes = [i for i, v in enumerate(val_mask) if v] if val_mask is not None else []

    return data, pred, val_nodes
