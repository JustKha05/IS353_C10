"""Pipeline chạy thực nghiệm hàng loạt cho Task T05 và T06 (BG10, BG11).
Tự động xuất edge_scores.csv và evaluation.csv.
"""

from typing import List, Dict, Any, Optional, Tuple
import pathlib
import time
import pandas as pd
import torch
from sna.data.candidate_edges import extract_candidate_edges
from sna.explainers.registry import get_explainer
from sna.metrics.fidelity import compute_fidelity
from sna.metrics.ground_truth import compute_ground_truth_auc


def run_batch_explaining(
    model: torch.nn.Module,
    data: Dict[str, Any],
    dataset_name: str,
    node_ids: List[int],
    methods: List[str] = ['saliency', 'gnnexplainer', 'pgexplainer'],
    output_dir: Optional[pathlib.Path] = None,
    seed: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chạy batch trên danh sách validation nodes, xuất bảng điểm và đánh giá mẫu.

    Returns:
        (df_edge_scores, df_evaluations)
    """
    model.eval()
    device = next(model.parameters()).device

    # Nếu có PGExplainer, khởi tạo và train trước để dùng lại cho mọi node (tiết kiệm thời gian)
    pg_instance = None
    if any(m in ('pgexplainer', 'pge') for m in methods):
        from sna.explainers.pgexplainer import PGExplainerModel
        print("Huấn luyện trước PGExplainer trên tập train_mask...")
        pg_instance = PGExplainerModel(model=model, emb_dim=model.c1.out_channels, device=device)
        pg_instance.train_on_graph(data=data, epochs=40, lr=0.005, seed=seed)
        print("PGExplainer đã sẵn sàng!")

    scores_records = []
    eval_records = []

    total_runs = len(node_ids) * len(methods)
    print(f"Bắt đầu chạy batch: {len(node_ids)} nodes x {len(methods)} methods = {total_runs} lượt...")

    # Nạp pred
    with torch.no_grad():
        all_preds = model(data['x'].to(device), data['edge_index'].to(device)).argmax(dim=-1).cpu()

    for idx, node_id in enumerate(node_ids):
        target_class = int(all_preds[node_id])
        cand = extract_candidate_edges(data['edge_index'], node_id, num_hops=2)

        for method in methods:
            run_id = f"{dataset_name}_{method}_node{node_id}_s{seed}"
            kwargs = {'trained_model': pg_instance} if method in ('pgexplainer', 'pge') else {}
            explainer = get_explainer(method, **kwargs)
            res = explainer.explain(
                model=model,
                data=data,
                node_id=node_id,
                target_class=target_class,
                candidate_edges=cand,
                seed=seed
            )

            if res.status == "success":
                # 1. Ghi nhận edge_scores (BG10)
                for eid, (u, v), s in zip(res.edge_ids, cand.candidate_pairs, res.scores):
                    scores_records.append({
                        'run_id': run_id,
                        'dataset': dataset_name,
                        'method': method,
                        'node_id': node_id,
                        'edge_id': eid,
                        'src': u,
                        'dst': v,
                        'score': s
                    })

                # 2. Tính Fidelity đánh giá sơ bộ (top-k=5)
                fid_res = compute_fidelity(
                    model=model,
                    data=data,
                    node_id=node_id,
                    target_class=target_class,
                    candidate_pairs=cand.candidate_pairs,
                    edge_scores=res.scores,
                    top_k=5
                )

                eval_records.append({
                    'run_id': run_id,
                    'dataset': dataset_name,
                    'method': method,
                    'node_id': node_id,
                    'budget_type': 'top_k',
                    'budget_val': 5,
                    'runtime_sec': res.runtime_sec,
                    'fid_drop': fid_res['fid_drop'],
                    'fid_inv': fid_res['fid_inv']
                })

    df_scores = pd.DataFrame(scores_records)
    df_evals = pd.DataFrame(eval_records)

    if output_dir is not None:
        out_path = pathlib.Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        df_scores.to_csv(out_path / "edge_scores.csv", index=False, encoding='utf-8')
        df_evals.to_csv(out_path / "evaluation.csv", index=False, encoding='utf-8')
        print(f"Đã lưu kết quả tại: {out_path}")

    return df_scores, df_evals
