"""Shared undirected edge selection, baseline and metric conventions."""
from collections import deque
import numpy as np
import torch
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score, average_precision_score


def validate_scores(scores, size=None):
    a = np.asarray(scores, dtype=float)
    if a.ndim != 1 or (size is not None and len(a) != size) or not np.isfinite(a).all():
        raise ValueError('Scores must be finite, one-dimensional and match candidate edges')
    return a


def rank_edges(scores, seed=0):
    """Raw scores stay unchanged; seeded uniform permutation breaks only ties."""
    a = validate_scores(scores)
    permutation = np.random.default_rng(seed).permutation(len(a))
    return permutation[np.argsort(-a[permutation], kind='stable')]


def baseline_scores(pairs, node_id, method):
    if method == 'random':
        return np.zeros(len(pairs), dtype=float)
    if method != 'nearest_first':
        raise ValueError(method)
    adjacency = {}
    for u, v in pairs:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)
    distance = {node_id: 0}
    queue = deque([node_id])
    while queue:
        u = queue.popleft()
        for v in adjacency.get(u, []):
            if v not in distance:
                distance[v] = distance[u] + 1
                queue.append(v)
    if any(u not in distance or v not in distance for u, v in pairs):
        raise ValueError('Candidate graph contains edges disconnected from target')
    return np.array([-min(distance[u], distance[v]) for u, v in pairs], dtype=float)


def top_indices(order, k):
    if int(k) != k or k < 0:
        raise ValueError('Budget must be a non-negative integer')
    return np.asarray(order, dtype=int)[:min(int(k), len(order))]


def agreement(scores_a, scores_b, k, seed=0, order_a=None, order_b=None):
    a = validate_scores(scores_a)
    b = validate_scores(scores_b, len(a))
    oa = rank_edges(a, seed) if order_a is None else order_a
    ob = rank_edges(b, seed) if order_b is None else order_b
    sa, sb = set(top_indices(oa, k)), set(top_indices(ob, k))
    jac = len(sa & sb) / len(sa | sb) if sa | sb else float('nan')
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        tau, reason = float('nan'), 'constant_scores_or_too_few_edges'
    else:
        tau = float(kendalltau(a, b, variant='b').statistic)
        reason = '' if np.isfinite(tau) else 'undefined_rank_correlation'
    return {'jaccard': jac, 'kendall_tau_b': tau, 'tau_reason': reason,
            'jaccard_reason': '' if sa | sb else 'empty_selected_union'}


def ground_truth_metrics(pairs, scores, edge_index, gt_edge_mask, selected):
    out = {key: float('nan') for key in ('roc_auc', 'average_precision', 'precision', 'recall')}
    if gt_edge_mask is None:
        return {**out, 'gt_reason': 'no_ground_truth', 'gt_positives': None,
                'auc_reason': 'no_ground_truth', 'precision_reason': 'no_ground_truth',
                'recall_reason': 'no_ground_truth'}
    e = edge_index.detach().cpu().numpy()
    mask = torch.as_tensor(gt_edge_mask).detach().cpu().numpy().astype(bool)
    if mask.ndim != 1 or len(mask) != e.shape[1]:
        raise ValueError('gt_edge_mask must align with columns of edge_index')
    labels_by_pair = {}
    for j, (u, v) in enumerate(e.T):
        p = (min(int(u), int(v)), max(int(u), int(v)))
        if p in labels_by_pair and labels_by_pair[p] != bool(mask[j]):
            raise ValueError('Ground truth disagrees between directions of an edge')
        labels_by_pair[p] = bool(mask[j])
    labels = np.array([labels_by_pair[tuple(p)] for p in pairs], dtype=int)
    a = validate_scores(scores, len(labels))
    pos = int(labels.sum())
    chosen = np.asarray(selected, dtype=int)
    hits = int(labels[chosen].sum())
    out['precision'] = hits / len(chosen) if len(chosen) else float('nan')
    out['recall'] = hits / pos if pos else float('nan')
    both = 0 < pos < len(labels)
    if both:
        out['roc_auc'] = float(roc_auc_score(labels, a))
        out['average_precision'] = float(average_precision_score(labels, a))
    return {**out, 'gt_positives': pos, 'gt_reason': '',
            'auc_reason': '' if both else 'single_class_or_empty_candidates',
            'precision_reason': '' if len(chosen) else 'empty_selection',
            'recall_reason': '' if pos else 'no_positive_candidate'}


def fidelity_selected(model, data, node_id, target_class, pairs, selected, orig_prob=None):
    """Deletion: remove selected pairs. Keep: retain only selected original pairs.

    Features stay fixed. Self-loops originally present stay in both graphs;
    GCNConv's implicit self-loops also remain active, even for empty selection.
    """
    device = next(model.parameters()).device
    x, edges = data['x'].to(device), data['edge_index'].to(device)
    chosen = {tuple(pairs[i]) for i in selected}
    endpoints = edges.detach().cpu().t().tolist()
    selected_mask = torch.tensor([(min(u, v), max(u, v)) in chosen for u, v in endpoints],
                                 device=device, dtype=torch.bool)
    loops = edges[0] == edges[1]
    def probability(e):
        with torch.no_grad():
            return float(model(x, e)[node_id].softmax(-1)[target_class].cpu())
    model.eval()
    original = probability(edges) if orig_prob is None else orig_prob
    deleted = probability(edges[:, ~selected_mask])
    kept = probability(edges[:, selected_mask | loops])
    return {'orig_prob': original, 'delete_prob': deleted, 'keep_prob': kept,
            'fid_drop': original - deleted, 'fid_inv': original - kept}
