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

    # 7. Kiểm thử trường hợp biên: Node có đúng 1 cạnh ứng viên (Degree-1 / Leaf node)
    print("\n[5/5] Đang kiểm tra trường hợp biên (Node có 1 cạnh ứng viên - Cora node 3)...")
    cora_data, cora_pred, _ = sna.load_dataset('cora')
    cora_model, _, _ = sna.load_gcn_checkpoint('cora')
    cora_model = cora_model.to(device)
    cand_deg1 = sna.extract_candidate_edges(cora_data['edge_index'], 3, num_hops=2)
    assert cand_deg1.num_candidate_edges == 1
    edge_ids, scores_deg1, _, st_deg1, err_deg1 = sna.explain_edges(
        model=cora_model,
        data=cora_data,
        node_id=3,
        target_class=int(cora_pred[3]),
        candidate_edges=cand_deg1,
        config={'method': 'pgexplainer', 'epochs': 5},
        seed=0
    )
    assert st_deg1 == "success", f"Degree-1 node failed: {err_deg1}"
    assert len(scores_deg1) == 1, f"Expected 1 score, got {len(scores_deg1)}"
    print(f"  -> Degree-1 Node OK | Score: {scores_deg1[0]:.4f}")

    # 8. Kiểm thử mở rộng (Reviewer 2 & 3 Adversarial Edge Cases):
    print("\n[6/6] Đang kiểm tra các trường hợp biên bổ sung (Adversarial Robustness)...")
    import numpy as np
    from sna.explainers.pgexplainer import PGExplainerMLP, PGExplainerModel
    mlp_test = PGExplainerMLP(emb_dim=16)

    # a) Target vector broadcasting: (N, D) + (N, D) + (D,)
    u_2d = torch.randn(5, 16)
    v_2d = torch.randn(5, 16)
    t_1d = torch.randn(16)
    out_broad = mlp_test(u_2d, v_2d, t_1d)
    assert out_broad.shape == torch.Size([5]), f"Expected shape [5], got {out_broad.shape}"

    # b) Target singleton 2D broadcasting: (N, D) + (N, D) + (1, D)
    t_single = torch.randn(1, 16)
    out_single = mlp_test(u_2d, v_2d, t_single)
    assert out_single.shape == torch.Size([5]), f"Expected shape [5], got {out_single.shape}"

    # c) Reviewer 3: Asymmetric broadcasting (N, D) + (1, D) or (D,) and (D,) + (N, D)
    v_1d = torch.randn(16)
    v_single = torch.randn(1, 16)
    u_1d = torch.randn(16)
    u_single = torch.randn(1, 16)
    out_asym1 = mlp_test(u_2d, v_1d, t_1d)
    assert out_asym1.shape == torch.Size([5]), f"Expected [5], got {out_asym1.shape}"
    out_asym2 = mlp_test(u_2d, v_single, t_1d)
    assert out_asym2.shape == torch.Size([5]), f"Expected [5], got {out_asym2.shape}"
    out_asym3 = mlp_test(u_1d, v_2d, t_1d)
    assert out_asym3.shape == torch.Size([5]), f"Expected [5], got {out_asym3.shape}"
    out_asym4 = mlp_test(u_single, v_2d, t_1d)
    assert out_asym4.shape == torch.Size([5]), f"Expected [5], got {out_asym4.shape}"
    out_asym5 = mlp_test(u_single, v_1d, t_1d)
    assert out_asym5.shape == torch.Size([1]), f"Expected [1], got {out_asym5.shape}"

    # d) Reviewer 3: Pre-concatenated 1D and 2D single / batch inputs
    cat_1d = torch.randn(48)
    assert mlp_test(cat_1d).shape == torch.Size([1])
    cat_empty = torch.randn(0, 48)
    assert mlp_test(cat_empty).shape == torch.Size([0])

    # e) Auto-detection của emb_dim trong PGExplainerModel (c1 & conv1)
    auto_pge = PGExplainerModel(model=cora_model)
    assert auto_pge.emb_dim == cora_model.c1.out_channels, f"Expected {cora_model.c1.out_channels}, got {auto_pge.emb_dim}"

    class CustomConv1Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Linear(16, 32)
            self.conv1.out_channels = 32
        def forward(self, x, edge_index):
            return self.conv1(x)

    custom_model = CustomConv1Model()
    conv1_pge = PGExplainerModel(model=custom_model)
    assert conv1_pge.emb_dim == 32

    # f) Train mask dạng PyTorch tensor chỉ số nguyên (Integer index tensor)
    synth_data = {
        'x': torch.randn(10, 16),
        'edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        'train_mask': torch.tensor([0, 2], dtype=torch.long)
    }
    synth_model = sna.models.gcn.GCN(16, 16, 2)
    pge_synth = PGExplainerModel(synth_model)
    synth_time = pge_synth.train_on_graph(synth_data, epochs=2)
    assert pge_synth.is_trained
    assert synth_time > 0

    # g) Reviewer 3: Train mask dạng NumPy integer array (ngăn ngừa lỗi index corruption [1, 2] thay vì [0, 2])
    synth_data_np = {
        'x': torch.randn(10, 16),
        'edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        'train_mask': np.array([0, 2])
    }
    pge_synth_np = PGExplainerModel(synth_model)
    synth_time_np = pge_synth_np.train_on_graph(synth_data_np, epochs=2)
    assert pge_synth_np.is_trained

    # h) Reviewer 3: Đồ thị có Node cô lập vượt quá index max của edge_index (ngăn ngừa PyG k_hop_subgraph IndexError)
    edge_index_small = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    isolated_cand = sna.extract_candidate_edges(edge_index_small, node_id=5)
    assert isolated_cand.num_candidate_edges == 0
    assert isolated_cand.subgraph_nodes.tolist() == [5]
    assert isolated_cand.edge_to_pair_idx.device == edge_index_small.device

    # i) Đồ thị rỗng không có cạnh (Empty edge_index)
    empty_cand = sna.extract_candidate_edges(torch.empty((2, 0), dtype=torch.long), node_id=0)
    assert empty_cand.num_candidate_edges == 0
    print("  -> Tất cả các trường hợp biên bổ sung (Asym Broadcasting, Pre-concatenated, conv1, NumPy train_mask, Isolated node bounds, Empty graph) ĐỀU ĐẠT!")

    print("\n" + "=" * 50)
    print(">>> TẤT CẢ CÁC BÀI KIỂM THỬ EXPLAINERS ĐỀU THÀNH CÔNG! <<<")
    print("=" * 50)


if __name__ == '__main__':
    run_test()
