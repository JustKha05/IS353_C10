"""Kiểm thử các phương pháp giải thích cạnh (Saliency, GNNExplainer, PGExplainer, Baselines).
Chạy trên cả hai dataset BA-Shapes và Cora với GPU CUDA (RTX 3050 Laptop).
"""

import sys
import pathlib
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import sna


def run_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== KIỂM THỬ EXPLAINERS TRÊN THIẾT BỊ: {device} ===")

    for ds_name in ('bashapes', 'cora'):
        print(f"\n==========================================")
        print(f"--- ĐANG KIỂM THỬ TRÊN DATASET: {ds_name.upper()} ---")
        print(f"==========================================")

        # 1. Nạp dữ liệu và mô hình
        data, pred, val_nodes = sna.load_dataset(ds_name)
        model, _, _ = sna.load_gcn_checkpoint(ds_name)
        model = model.to(device)

        # Chọn node validation đầu tiên
        test_node = val_nodes[0]
        target_class = int(pred[test_node])
        true_label = int(data['y'][test_node])

        print(f"Node kiểm thử: {test_node}")
        print(f"Nhãn thật (y): {true_label} | Dự đoán GCN (pred): {target_class}")

        # 2. Trích xuất tập cạnh ứng viên 2-hop theo BG05, BG07
        cand = sna.extract_candidate_edges(data['edge_index'], test_node, num_hops=2)
        m = cand.num_candidate_edges
        print(f"Số cạnh ứng viên (m): {m}")

        # 3. Saliency
        print("\n[1/4] Đang chạy Saliency...")
        edge_ids, scores_sal, rt_sal, st_sal, err_sal = sna.explain_edges(
            model=model,
            data=data,
            node_id=test_node,
            target_class=target_class,
            candidate_edges=cand,
            config={'method': 'saliency'},
            seed=0
        )
        assert st_sal == "success", f"Saliency lỗi: {err_sal}"
        assert len(scores_sal) == m, f"Số điểm ({len(scores_sal)}) khác số cạnh ({m})"
        print(f"  -> Saliency OK | Runtime: {rt_sal:.4f}s | Max score: {max(scores_sal):.4f}")

        # 4. GNNExplainer
        print("\n[2/4] Đang chạy GNNExplainer (50 epochs nhanh)...")
        edge_ids, scores_gnn, rt_gnn, st_gnn, err_gnn = sna.explain_edges(
            model=model,
            data=data,
            node_id=test_node,
            target_class=target_class,
            candidate_edges=cand,
            config={'method': 'gnnexplainer', 'epochs': 50, 'lr': 0.02},
            seed=0
        )
        assert st_gnn == "success", f"GNNExplainer lỗi: {err_gnn}"
        assert len(scores_gnn) == m, f"Số điểm ({len(scores_gnn)}) khác số cạnh ({m})"
        print(f"  -> GNNExplainer OK | Runtime: {rt_gnn:.4f}s | Max score: {max(scores_gnn):.4f}")

        # 5. PGExplainer
        print("\n[3/4] Đang huấn luyện & chạy PGExplainer (10 epochs nhanh)...")
        edge_ids, scores_pg, rt_pg, st_pg, err_pg = sna.explain_edges(
            model=model,
            data=data,
            node_id=test_node,
            target_class=target_class,
            candidate_edges=cand,
            config={'method': 'pgexplainer', 'epochs': 10, 'lr': 0.01},
            seed=0
        )
        assert st_pg == "success", f"PGExplainer lỗi: {err_pg}"
        assert len(scores_pg) == m, f"Số điểm ({len(scores_pg)}) khác số cạnh ({m})"
        print(f"  -> PGExplainer OK | Runtime: {rt_pg:.4f}s | Max score: {max(scores_pg):.4f}")

        # 6. Baselines (Random & Nearest-First)
        print("\n[4/4] Đang kiểm tra Baselines...")
        for b_name in ('random', 'nearest_first'):
            _, b_scores, b_rt, b_st, _ = sna.explain_edges(
                model=model,
                data=data,
                node_id=test_node,
                target_class=target_class,
                candidate_edges=cand,
                config={'method': b_name},
                seed=0
            )
            assert b_st == "success"
            print(f"  -> Baseline {b_name:15s} OK | Runtime: {b_rt:.4f}s")

    print("\n" + "=" * 50)
    print(">>> TẤT CẢ CÁC BÀI KIỂM THỬ EXPLAINERS ĐỀU THÀNH CÔNG! <<<")
    print("=" * 50)


if __name__ == '__main__':
    run_test()
