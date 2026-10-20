"""Small graph regression checks; no Drive assets required."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import math
import numpy as np
import pytest
import torch
from sna.evaluation.core import (rank_edges, baseline_scores, agreement,
                                ground_truth_metrics, fidelity_selected)
from sna.evaluation.core import validate_scores


def test_seeded_ties_and_distance():
    pairs = [(0, 1), (1, 2), (2, 3), (1, 3)]
    s = baseline_scores(pairs, 0, 'nearest_first')
    assert s.tolist() == [0, -1, -2, -1]
    assert np.array_equal(rank_edges(s, 9), rank_edges(s, 9))
    assert rank_edges(s, 9)[0] == 0
    r = baseline_scores(pairs, 0, 'random')
    assert not r.any()
    assert math.isnan(agreement(r, s, 2)['kendall_tau_b'])
    assert math.isnan(agreement([], [], 0)['jaccard'])
    with pytest.raises(ValueError):
        agreement([1], [1, 2], 1)


def test_gt_constant_random_and_undefined():
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    gt = torch.tensor([True, True, False, False])
    result = ground_truth_metrics([(0, 1), (1, 2)], [0, 0], edges, gt, [0])
    assert result['roc_auc'] == .5
    assert result['average_precision'] == .5
    assert result['precision'] == result['recall'] == 1
    single = ground_truth_metrics([(0, 1)], [0], edges, gt, [0])
    assert math.isnan(single['roc_auc'])
    assert single['auc_reason']
    bad_gt = torch.tensor([True, False, False, False])
    with pytest.raises(ValueError, match='directions'):
        ground_truth_metrics([(0, 1)], [0], edges, bad_gt, [])


class EdgeCounter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.))
        self.seen = []

    def forward(self, x, edge_index):
        self.seen.append(edge_index.detach().clone())
        n = edge_index.shape[1]
        return torch.stack([torch.full((len(x),), float(n)), torch.ones(len(x))], dim=-1)


def test_fidelity_removes_both_directions_and_forwards_empty_keep():
    model = EdgeCounter()
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    data = {'x': torch.ones(3, 1), 'edge_index': edges}
    result = fidelity_selected(model, data, 0, 0, [(0, 1), (1, 2)], [0])
    assert model.seen[1].tolist() == [[1, 2], [2, 1]]
    assert model.seen[2].tolist() == [[0, 1], [1, 0]]
    assert result['fid_drop'] > 0
    result = fidelity_selected(model, data, 0, 0, [(0, 1), (1, 2)], [])
    assert model.seen[-1].shape == (2, 0)
    # Even with no selected edges the predictor has a nonzero class probability.
    assert result['keep_prob'] > 0
    with pytest.raises(ValueError):
        validate_scores([1, 0], 1)
