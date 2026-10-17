"""Kiểm thử toàn diện kiến trúc phần mềm Modular Monolith 'sna' package.
Kiểm tra data loader, models, explainers, metrics, và batch pipeline.
"""

import sys
import pathlib
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import sna


def test_modular_package():
    print(f"=== TEST MODULAR MONOLITH 'sna' PACKAGE (v{sna.__version__}) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Test Data Loader & Models
    print("\n[1/4] Kiểm tra Data Loader & Model Checkpoint...")
    data, pred, val_nodes = sna.load_dataset('bashapes')
    model, ckpt_pred, metrics = sna.load_gcn_checkpoint('bashapes')
    model = model.to(device)
    print(f"  -> Nạp BA-Shapes thành công: {data['stats']['nodes']} nodes, {len(val_nodes)} val nodes.")
    print(f"  -> Checkpoint metrics: {metrics}")

    # 2. Test Candidate Edges
    print("\n[2/4] Kiểm tra Trích xuất Candidate Edges...")
    test_node = val_nodes[0]
    target_class = int(pred[test_node])
    cand = sna.extract_candidate_edges(data['edge_index'], test_node, num_hops=2)
    print(f"  -> Node {test_node} (class {target_class}) có {cand.num_candidate_edges} cạnh ứng viên 2-hop.")

    # 3. Test Explainers & Metrics
    print("\n[3/4] Kiểm tra các Explainer và Metrics...")
    methods = ['saliency', 'gnnexplainer', 'pgexplainer', 'random', 'nearest_first']
    scores_dict = {}

    for m in methods:
        edge_ids, scores, rt, st, err = sna.explain_edges(
            model=model,
            data=data,
            node_id=test_node,
            target_class=target_class,
            candidate_edges=cand,
            config={'method': m, 'epochs': 30},
            seed=0
        )
        assert st == "success", f"Method {m} thất bại: {err}"
        scores_dict[m] = scores
        # Tính Fidelity
        fid = sna.compute_fidelity(model, data, test_node, target_class, cand.candidate_pairs, scores, top_k=5)
        print(f"  -> {m:15s} | Runtime: {rt:.4f}s | Fid- (drop): {fid['fid_drop']:.4f} | Fid+ (inv): {fid['fid_inv']:.4f}")

    # Tính Agreement Jaccard giữa GNNExplainer và PGExplainer
    jaccard = sna.compute_jaccard_at_k(scores_dict['gnnexplainer'], scores_dict['pgexplainer'], k=5)
    print(f"  -> Jaccard@5 (GNNExplainer vs PGExplainer): {jaccard:.4f}")

    # 4. Test Batch Pipeline (2 nodes)
    print("\n[4/4] Kiểm tra Pipeline Chạy Batch (Task T05)...")
    sample_nodes = val_nodes[:2]
    df_scores, df_evals = sna.run_batch_explaining(
        model=model,
        data=data,
        dataset_name='bashapes',
        node_ids=sample_nodes,
        methods=['saliency', 'pgexplainer'],
        seed=0
    )
    print(f"  -> Batch Runner OK: {len(df_scores)} edge scores, {len(df_evals)} evaluations.")

    print("\n" + "=" * 60)
    print(">>> KIẾN TRÚC MODULAR PACKAGE 'sna' HOẠT ĐỘNG HOÀN HẢO! <<<")
    print("=" * 60)


if __name__ == '__main__':
    test_modular_package()
