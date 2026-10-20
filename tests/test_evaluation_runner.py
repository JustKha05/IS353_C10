"""Exercise the real CLI pipeline using a tiny saved GCN and synthetic scores."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from argparse import Namespace
import json
import pandas as pd
import torch
from sna.models import GCN
from sna.data import extract_candidate_edges
from sna.evaluation.runner import run


def test_pipeline_and_missing_method(tmp_path):
    root = tmp_path / 'data'
    ds = root / 'bashapes'
    ds.mkdir(parents=True)
    torch.manual_seed(0)
    model = GCN(2, 3, 2).eval()
    data = {'x': torch.ones(4, 2),
            'edge_index': torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
            'train_mask': torch.tensor([False, True, False, False]),
            'val_mask': torch.tensor([True, False, False, False]),
            'test_mask': torch.tensor([False, False, True, True]),
            'gt_edge_mask': torch.tensor([True, True, False, False, False, False])}
    with torch.no_grad():
        pred = model(data['x'], data['edge_index']).argmax(-1)
    torch.save(data, ds / 'data.pt')
    torch.save({'state_dict': model.state_dict(), 'in_dim': 2, 'hidden': 3, 'n_classes': 2,
                'pred': pred}, ds / 'gcn_ckpt.pt')
    (ds / 'config.json').write_text('{}')
    (ds / 'val_nodes.json').write_text(json.dumps({'nodes': [0]}))
    pairs = extract_candidate_edges(data['edge_index'], 0, num_nodes=4).candidate_pairs
    rows = [dict(dataset='bashapes', method=m, node_id=0, target_class=int(pred[0]),
                 edge_id=i, u=u, v=v, importance_score=float(i), seed=42, run_id='synthetic_test')
            for m in ['saliency', 'gnnexplainer', 'pgexplainer']
            for i, (u, v) in enumerate(pairs)]
    source = tmp_path / 'scores.csv'
    pd.DataFrame(rows).to_csv(source, index=False)
    args = Namespace(output=tmp_path / 'report', scores=source, data_root=root,
                     datasets=['bashapes'], nodes_per_dataset=1, budgets=[1, 3],
                     seeds=[0, 1], device='cpu', audit_only=False)
    assert run(args) == 0
    result = pd.read_csv(args.output / 'metrics.csv')
    assert len(result) == (3 + 2 * 2) * 2
    assert result.status.eq('success').all()
    assert (args.output / 'bashapes_fidelity.png').exists()
    # All-constant Random scores must yield exactly 0.5 AUROC, not jittered values.
    assert result.loc[result.method == 'random', 'roc_auc'].eq(.5).all()
    missing = pd.DataFrame(rows)
    missing = missing[missing.method != 'pgexplainer']
    missing.to_csv(source, index=False)
    args.output = tmp_path / 'missing_report'
    assert run(args) == 1
    errors = pd.read_csv(args.output / 'errors.csv')
    assert errors.reason.str.contains('missing_method').any()
