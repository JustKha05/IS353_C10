"""Re-evaluate T05 CSVs without retraining explainers (CPU by default)."""
import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sna.core.config import PROJECT_ROOT
from sna.data import extract_candidate_edges
from sna.models import GCN
from .core import (rank_edges, baseline_scores, top_indices, agreement,
                   ground_truth_metrics, fidelity_selected, validate_scores)

METHODS = ['saliency', 'gnnexplainer', 'pgexplainer', 'random', 'nearest_first']
BASELINES = {'random', 'nearest_first'}
REQUIRED = ['dataset', 'method', 'node_id', 'target_class', 'edge_id', 'u', 'v',
            'importance_score', 'seed', 'run_id']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def seed_for(dataset, node, method, repeat):
    text = f'{dataset}:{node}:{method}:{repeat}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'big')


def read_scores(path):
    df = pd.read_csv(path)
    missing = set(REQUIRED) - set(df)
    if missing:
        raise ValueError(f'CSV missing columns: {sorted(missing)}')
    # Legacy baseline points are deliberately regenerated using T03 conventions.
    df = df[~df.method.isin(BASELINES)].copy()
    unknown = set(df.method) - set(METHODS)
    if unknown:
        raise ValueError(f'Unsupported methods: {unknown}')
    if df.empty or df[REQUIRED].isna().any().any():
        raise ValueError('Input is empty or has missing required fields')
    for col in ['node_id', 'target_class', 'edge_id', 'u', 'v', 'seed']:
        values = pd.to_numeric(df[col], errors='raise')
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f'Invalid integer field: {col}')
        df[col] = values.astype(int)
    validate_scores(df.importance_score)
    if (df.u >= df.v).any():
        raise ValueError('Input edges must be canonical undirected pairs u < v')
    if 'model_type' in df and set(df.model_type) != {'gcn'}:
        raise ValueError('This evaluator requires model_type=gcn')
    return df


def summary(rows, groups, metrics):
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    # Baseline repeats are averaged within node first: repeats are not extra nodes.
    node_groups = groups + ['node_id']
    numeric = [m for m in metrics if m in df]
    per_node = df.groupby(node_groups, dropna=False)[numeric].mean().reset_index()
    result = []
    for key, part in per_node.groupby(groups, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        for metric in numeric:
            vals = part[metric].dropna()
            result.append(dict(zip(groups, key), metric=metric,
                               mean=vals.mean(), std=vals.std(ddof=1),
                               n_valid_nodes=len(vals), n_total_nodes=len(part)))
    return pd.DataFrame(result)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = args.scores.resolve()
    df = read_scores(source)
    errors, audit, metrics, agreements, runtime, ranks = [], [], [], [], [], []
    manifest = {'mode': 'audit_only' if args.audit_only else 'full',
                'source_scores_sha256': sha256(source), 'source_scores': source.name,
                'budgets': args.budgets, 'baseline_repeat_seeds': args.seeds,
                'datasets': args.datasets, 'nodes_per_dataset': args.nodes_per_dataset,
                'python': platform.python_version(), 'torch': torch.__version__,
                'numpy': np.__version__, 'pandas': pd.__version__,
                'device': args.device, 'assets': {},
                'checkpoint_provenance': 'CSV has no checkpoint hash. Prediction and candidate checks do not prove identical original weights.',
                'runtime_note': 'T05 CSV has no per-node explainer runtime. Not reconstructed from evaluation duration.',
                'keep_policy': 'selected edges only on full feature matrix; explicit self-loops retained; model implicit self-loops remain',
                'auc_policy': 'raw tied scores; single-class candidates => NaN; AP is average_precision_score, not trapezoidal PR-AUC',
                'std_policy': 'sample SD across nodes (ddof=1), baseline repeats averaged per node first',
                'tie_policy': 'seeded uniform permutation, stable descending raw scores, one order reused across all budgets'}
    try:
        manifest['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip()
        manifest['git_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=PROJECT_ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        manifest['git_commit'] = None
    for ds in args.datasets:
        ds_dir = args.data_root / ds
        with open(ds_dir / 'val_nodes.json', encoding='utf-8-sig') as f:
            expected = list(map(int, json.load(f)['nodes'][:args.nodes_per_dataset]))
        if len(expected) != args.nodes_per_dataset or len(set(expected)) != len(expected):
            raise ValueError(f'{ds}: not enough unique validation nodes in val_nodes.json')
        observed = set(df.loc[df.dataset == ds, 'node_id'])
        if not observed.issubset(set(expected)):
            errors.append(dict(dataset=ds, node_id=None, method='input', reason=f'extra_nodes_outside_selected_validation_list:{sorted(observed-set(expected))}'))
        data = model = pred = None
        if not args.audit_only:
            try:
                for name in ['data.pt', 'gcn_ckpt.pt', 'config.json', 'val_nodes.json']:
                    manifest['assets'][f'{ds}/{name}'] = sha256(ds_dir / name)
                data = torch.load(ds_dir / 'data.pt', map_location='cpu', weights_only=False)
                ck = torch.load(ds_dir / 'gcn_ckpt.pt', map_location='cpu', weights_only=False)
                model = GCN(ck['in_dim'], ck['hidden'], ck['n_classes'], ck.get('dropout', 0.0))
                model.load_state_dict(ck['state_dict'])
                model = model.to(args.device).eval()
                for p in model.parameters():
                    p.requires_grad_(False)
                with torch.no_grad():
                    pred = model(data['x'].to(args.device), data['edge_index'].to(args.device)).argmax(-1).cpu()
                if not torch.equal(pred, torch.as_tensor(ck['pred']).cpu()):
                    raise ValueError('Recomputed checkpoint predictions differ from saved pred')
                e = data['edge_index'].cpu().t().tolist()
                directed = {(u, v) for u, v in e}
                if any((v, u) not in directed for u, v in directed):
                    raise ValueError('Graph is not symmetric undirected')
            except Exception as exc:
                errors.append(dict(dataset=ds, node_id=None, method='assets', reason=f'{type(exc).__name__}: {exc}'))
                model = None
        for node in expected:
            node_df = df[(df.dataset == ds) & (df.node_id == node)]
            shared_error = ''
            pairs = []
            if node_df.empty:
                shared_error = 'missing_all_methods_for_expected_node'
            else:
                # Every method must map edge IDs to the same original pairs.
                ref = node_df[node_df.method == node_df.method.iloc[0]].sort_values('edge_id')
                pairs = list(zip(ref.u.tolist(), ref.v.tolist()))
                if ref.duplicated('edge_id').any() or ref.duplicated(['u', 'v']).any():
                    shared_error = 'duplicate_reference_edges'
                if not args.audit_only and model is not None:
                    try:
                        if not bool(data['val_mask'][node]) or bool(data['test_mask'][node]) or bool(data['train_mask'][node]):
                            raise ValueError('Target is not validation-only')
                        cand = extract_candidate_edges(data['edge_index'], node, num_hops=2,
                                                       num_nodes=len(data['x']))
                        if set(pairs) != set(cand.candidate_pairs):
                            raise ValueError('CSV candidates differ from graph 2-hop candidates')
                    except Exception as exc:
                        shared_error = str(exc)
            valid = {}
            for method in METHODS[:3]:
                part = node_df[node_df.method == method].sort_values('edge_id')
                reason = shared_error
                if not reason:
                    if part.empty:
                        reason = 'missing_method_for_expected_node'
                    elif part.duplicated('edge_id').any() or part.duplicated(['u', 'v']).any():
                        reason = 'duplicate_edges_or_multiple_runs'
                    elif part[['seed', 'run_id', 'target_class']].nunique().max() > 1:
                        reason = 'multiple_source_runs_or_target_classes'
                    elif list(zip(part.u, part.v)) != pairs or part.edge_id.tolist() != ref.edge_id.tolist():
                        reason = 'edge_id_pair_mapping_mismatch'
                    elif node_df.target_class.nunique() != 1:
                        reason = 'methods_disagree_on_target_class'
                    elif not args.audit_only and model is not None and int(part.target_class.iloc[0]) != int(pred[node]):
                        reason = 'target_class_differs_from_checkpoint'
                audit.append(dict(dataset=ds, node_id=node, method=method, n_edges=len(part),
                                  status='failed' if reason else 'success', reason=reason,
                                  graph_verified=not args.audit_only and model is not None and not reason))
                if reason:
                    errors.append(dict(dataset=ds, node_id=node, method=method, reason=reason))
                else:
                    valid[method] = part.importance_score.to_numpy(float)
                    runtime.append(dict(dataset=ds, node_id=node, method=method,
                                        score_runtime_sec=float('nan'), status='unavailable',
                                        reason='per_node_runtime_not_in_T05_CSV'))
            if args.audit_only:
                continue
            if model is None or shared_error:
                continue
            if not valid:
                continue
            target = int(pred[node])
            # Canonical sort for deterministic tie-breaking independent of CSV row order.
            permutation = sorted(range(len(pairs)), key=lambda i: pairs[i])
            pairs = [pairs[i] for i in permutation]
            valid = {m: a[permutation] for m, a in valid.items()}
            with torch.no_grad():
                original = float(model(data['x'].to(args.device), data['edge_index'].to(args.device))[node].softmax(-1)[target].cpu())
            orders = {}
            for method in METHODS:
                if method in BASELINES:
                    start = time.perf_counter()
                    scores = baseline_scores(pairs, node, method)
                    elapsed = time.perf_counter() - start
                    repeats = args.seeds
                elif method in valid:
                    scores, elapsed, repeats = valid[method], float('nan'), [int(node_df[node_df.method == method].seed.iloc[0])]
                else:
                    continue
                for repeat in repeats:
                    start = time.perf_counter()
                    tie_seed = seed_for(ds, node, method, repeat)
                    order = rank_edges(scores, tie_seed)
                    rank_sec = time.perf_counter() - start
                    orders[(method, repeat)] = (scores, order)
                    if method in BASELINES:
                        runtime.append(dict(dataset=ds, node_id=node, method=method, repeat_seed=repeat,
                                            score_runtime_sec=elapsed, ranking_runtime_sec=rank_sec,
                                            status='measured', reason='CPU baseline scoring; excludes fidelity'))
                    for position, i in enumerate(order):
                        ranks.append(dict(dataset=ds, node_id=node, method=method, repeat_seed=repeat,
                                          target_class=target, tie_seed=tie_seed, u=pairs[i][0], v=pairs[i][1],
                                          raw_score=float(scores[i]), rank=position + 1))
                    for budget in args.budgets:
                        start = time.perf_counter()
                        selected = top_indices(order, budget)
                        row = dict(dataset=ds, node_id=node, target_class=target, method=method,
                                   repeat_seed=repeat, tie_seed=tie_seed, budget=budget,
                                   k_eff=len(selected), n_candidates=len(pairs), status='success')
                        try:
                            row.update(fidelity_selected(model, data, node, target, pairs, selected, original))
                            gt = data.get('gt_edge_mask') if ds == 'bashapes' else None
                            row.update(ground_truth_metrics(pairs, scores, data['edge_index'], gt, selected))
                        except Exception as exc:
                            row.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
                            errors.append(dict(dataset=ds, node_id=node, method=method, budget=budget, reason=row['reason']))
                            # Do not retain partial metrics from a failed evaluation.
                            for key in ['fid_drop', 'fid_inv', 'roc_auc', 'average_precision', 'precision', 'recall']:
                                row[key] = float('nan')
                        row['evaluation_runtime_sec'] = time.perf_counter() - start
                        metrics.append(row)
            # Primary explainer pairs once; baseline pairs match the same repeat.
            for ma, mb in combinations(METHODS, 2):
                repeats = args.seeds if ma in BASELINES or mb in BASELINES else [0]
                for repeat in repeats:
                    ka = (ma, repeat) if ma in BASELINES else next((key for key in orders if key[0] == ma), None)
                    kb = (mb, repeat) if mb in BASELINES else next((key for key in orders if key[0] == mb), None)
                    if ka not in orders or kb not in orders:
                        continue
                    sa, oa = orders[ka]
                    sb, ob = orders[kb]
                    for budget in args.budgets:
                        agreements.append(dict(dataset=ds, node_id=node, method_a=ma, method_b=mb,
                                               repeat_seed=repeat, budget=budget,
                                               **agreement(sa, sb, budget, order_a=oa, order_b=ob)))
    for name, rows in [('audit', audit), ('errors', errors), ('metrics', metrics),
                       ('agreement', agreements), ('runtime', runtime), ('ranked_edges', ranks)]:
        pd.DataFrame(rows or [], columns=None if rows else ['status', 'reason']).to_csv(output / f'{name}.csv', index=False)
    metric_summary = summary(metrics, ['dataset', 'method', 'budget'],
                             ['fid_drop', 'fid_inv', 'roc_auc', 'average_precision', 'precision', 'recall'])
    agreement_summary = summary(agreements, ['dataset', 'method_a', 'method_b', 'budget'],
                                ['jaccard', 'kendall_tau_b'])
    if not metric_summary.empty:
        metric_summary['n_total_nodes'] = args.nodes_per_dataset
    if not agreement_summary.empty:
        agreement_summary['n_total_nodes'] = args.nodes_per_dataset
    metric_summary.to_csv(output / 'summary.csv', index=False)
    agreement_summary.to_csv(output / 'agreement_summary.csv', index=False)
    dataset_stats = []
    for ds in args.datasets:
        for method in METHODS[:3]:
            sub = df[(df.dataset == ds) & (df.method == method)]
            counts = sub.groupby('node_id').size()
            dataset_stats.append(dict(dataset=ds, method=method, n_nodes=len(counts),
                                      edge_rows=len(sub), mean_candidate_edges=counts.mean()))
    pd.DataFrame(dataset_stats).to_csv(output / 'dataset_stats.csv', index=False)
    runtime_summary = summary(runtime, ['dataset', 'method'], ['score_runtime_sec', 'ranking_runtime_sec'])
    runtime_summary.to_csv(output / 'runtime_summary.csv', index=False)
    manifest['error_count'] = len(errors)
    manifest['status'] = 'completed_with_errors' if errors else ('audit_passed' if args.audit_only else 'completed')
    manifest['expected_nodes'] = {ds: args.nodes_per_dataset for ds in args.datasets}
    manifest['scored_nodes'] = {ds: len({r['node_id'] for r in metrics if r['dataset'] == ds and r['status'] == 'success'}) for ds in args.datasets}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    write_report(output, manifest, metric_summary, agreement_summary)
    print(f"{manifest['status']}: {output}")
    print(f'Audit rows={len(audit)}, metric rows={len(metrics)}, errors={len(errors)}')
    return 1 if errors else 0


def write_report(output, manifest, metric_summary, agreement_summary):
    lines = ['# Đánh giá T03/T06/T09 — Đức + Kha', '',
             f"Chế độ: **{manifest['mode']}**. Trạng thái: **{manifest['status']}**.", '',
             '## Phạm vi và khả năng truy vết', '',
             '- Chỉ đánh giá danh sách validation đã chọn; không huấn luyện lại GCN/explainer.',
             '- manifest.json chứa hash đầu vào, dữ liệu/checkpoint khi có; ranked_edges.csv lưu thứ tự chọn cạnh.',
             '- CSV gốc không chứa hash checkpoint: khớp dự đoán/tập cạnh chưa chứng minh cùng trọng số lúc tạo scores.',
             '- Mỗi repeat baseline dùng một thứ tự cho tất cả ngân sách; điểm thô không thêm nhiễu.',
             '- SD trong bảng là độ lệch chuẩn mẫu giữa các node, không phải sai số chuẩn và không phải giữa các repeat.',
             '- Fidelity Drop = p(gốc) − p(xóa); Inv = p(gốc) − p(giữ). Giá trị âm được giữ nguyên.',
             '- Inv âm nghĩa là xác suất tăng khi chỉ giữ cạnh; không tự động coi âm hơn là giải thích tốt hơn.',
             '- Giữ cạnh: chỉ giữ cặp được chọn trên toàn bộ đặc trưng; self-loop của mô hình vẫn hoạt động.',
             '- AUROC chỉ tính khi có hai lớp; AP là Average Precision, không đồng nhất PR-AUC hình thang.',
             '- Không có ground truth Cora; metric không xác định ghi NaN và lý do, không thay bằng 0.',
             '- T05 không có thời gian explainer từng node: runtime.csv ghi unavailable, không suy đoán.', '']
    if manifest['mode'] == 'audit_only':
        lines += ['## Kết quả kiểm toán CSV', '',
                  'Chưa nạp .pt, chưa xác minh graph/checkpoint/val_mask và chưa tính fidelity.',
                  'Các CSV metrics/summary của chế độ này không phải kết quả đánh giá mô hình.', '']
    else:
        lines += ['## Bảng kết quả', '', '| Dataset | Method | k | Metric | Mean | SD | Node hợp lệ |',
                  '|---|---|---:|---|---:|---:|---:|']
        for r in metric_summary.to_dict('records'):
            lines.append(f"| {r['dataset']} | {r['method']} | {r['budget']} | {r['metric']} | {r['mean']:.4f} | {r['std']:.4f} | {r['n_valid_nodes']}/{r['n_total_nodes']} |")
        lines += ['', '## Nhận xét sơ bộ tự sinh', '']
        chosen_k = 5 if 5 in manifest['budgets'] else manifest['budgets'][0]
        for ds in manifest['datasets']:
            sub = metric_summary[(metric_summary.dataset == ds) & (metric_summary.budget == chosen_k)] if not metric_summary.empty else pd.DataFrame()
            for metric, label in [('fid_drop', 'Fidelity Drop'), ('roc_auc', 'motif AUROC')]:
                eligible = sub[(sub.metric == metric) & sub['mean'].notna()] if not sub.empty else sub
                if not eligible.empty:
                    r = eligible.sort_values('mean', ascending=False).iloc[0]
                    lines.append(f"- {ds}, top-{chosen_k}: {r['method']} có trung bình {label} cao nhất trong các phương pháp tính được ({r['mean']:.4f}; {r['n_valid_nodes']} node). Đây là mô tả mẫu, chưa phải kết luận vượt trội có ý nghĩa thống kê.")
        if not agreement_summary.empty:
            sub = agreement_summary[(agreement_summary.metric == 'jaccard') & (agreement_summary.budget == chosen_k) & ~agreement_summary.method_a.isin(BASELINES) & ~agreement_summary.method_b.isin(BASELINES)]
            if not sub.empty:
                lines.append(f"- Jaccard@{chosen_k} trung bình giữa các explainer nằm trong khoảng {sub['mean'].min():.4f}–{sub['mean'].max():.4f}; đồng thuận không đồng nghĩa với đúng ground truth.")
        negatives = pd.read_csv(output / 'metrics.csv')
        if 'fid_drop' in negatives:
            lines.append(f"- Có {int((negatives.fid_drop < 0).sum())} dòng Fidelity Drop âm (bao gồm các repeat/ngân sách); cần xem node cụ thể trong metrics.csv, không loại bỏ âm thầm.")
        plot_summary(metric_summary, output)
    lines += ['', '## Giới hạn và phần còn thiếu', '',
              f"- Số lỗi/đầu vào thiếu: {manifest['error_count']}; chi tiết errors.csv. Node thành công: {manifest['scored_nodes']}.",
              '- Một checkpoint mỗi dataset, một lượt explainer T05; repeat baseline không thay thế nhiều seed GNN/explainer.',
              '- Top-k có thể chiếm tỷ lệ rất khác nhau giữa node; xem k_eff và n_candidates.',
              '- Thiếu hash checkpoint và thời gian trong bàn giao T05 cần được nhóm tạo scores xác nhận/bổ sung.',
              '- Hướng tiếp: khóa giao thức, bổ sung log thời gian/provenance, tăng node và seed trước khi suy rộng.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def plot_summary(df, output):
    if df.empty:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for ds in df.dataset.unique():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric in zip(axes, ['fid_drop', 'fid_inv']):
            for method in METHODS:
                sub = df[(df.dataset == ds) & (df.method == method) & (df.metric == metric)].sort_values('budget')
                if not sub.empty:
                    ax.errorbar(sub.budget, sub['mean'], yerr=sub['std'].fillna(0), marker='o', capsize=3, label=method)
            ax.axhline(0, color='gray', linewidth=.7)
            ax.set(xlabel='Requested top-k (clipped per node)', ylabel=metric, title=f'{ds}: {metric}')
            ax.grid(alpha=.2)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f'{ds}_fidelity.png', dpi=160)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores', type=Path, default=PROJECT_ROOT / 'notebooks/01_midterm/edge_scores.csv')
    p.add_argument('--data-root', type=Path, default=PROJECT_ROOT / 'data')
    p.add_argument('--output', type=Path, default=PROJECT_ROOT / 'notebooks/05_duc_kha_evaluation/results')
    p.add_argument('--datasets', nargs='+', choices=['bashapes', 'cora'], default=['bashapes', 'cora'])
    p.add_argument('--nodes-per-dataset', type=int, default=20)
    p.add_argument('--budgets', type=int, nargs='+', default=[1, 3, 5, 10])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--audit-only', action='store_true')
    args = p.parse_args()
    if args.nodes_per_dataset < 1 or min(args.budgets) < 1 or len(set(args.seeds)) != len(args.seeds) or len(set(args.budgets)) != len(args.budgets):
        p.error('Positive node count/budgets and unique budgets/seeds required')
    if args.audit_only and args.output == PROJECT_ROOT / 'notebooks/05_duc_kha_evaluation/results':
        args.output = PROJECT_ROOT / 'notebooks/05_duc_kha_evaluation/audit_checks'
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
