"""Generate T06 handover and update only the marked README evaluation section."""
import json
from pathlib import Path
import pandas as pd

START = '<!-- DUC_KHA_RESULTS_START -->'
END = '<!-- DUC_KHA_RESULTS_END -->'


def build_handover(root, results):
    root, results = Path(root), Path(results)
    manifest = json.loads((results / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['mode'] != 'full' or manifest['status'] != 'completed':
        raise ValueError('Only completed full evaluation can update the handover')
    df = pd.read_csv(results / 'summary.csv')
    metrics = pd.read_csv(results / 'metrics.csv')
    audit = pd.read_csv(results / 'audit.csv')
    agreement_rows = pd.read_csv(results / 'agreement.csv')
    ag = pd.read_csv(results / 'agreement_summary.csv')
    ds_stats = pd.read_csv(results / 'dataset_stats.csv')
    def cell(dataset, method, metric):
        row = df[(df.dataset == dataset) & (df.method == method) & (df.budget == 5) & (df.metric == metric)]
        if len(row) != 1:
            raise ValueError(f'Missing/duplicate top-5 summary: {dataset}/{method}/{metric}')
        r = row.iloc[0]
        return f"{r['mean']:.4f} ± {r['std']:.4f}" if pd.notna(r['mean']) else 'N/A'
    tables = []
    for ds, title in [('bashapes', 'BA-Shapes'), ('cora', 'Cora')]:
        tables += [f'### {title} — top-5', '',
                   '| Phương pháp | Fidelity Drop | Fidelity Inv | Motif AUROC |',
                   '|---|---:|---:|---:|']
        for method in ['random', 'nearest_first', 'saliency', 'gnnexplainer', 'pgexplainer']:
            tables.append(f"| {method} | {cell(ds, method, 'fid_drop')} | {cell(ds, method, 'fid_inv')} | {cell(ds, method, 'roc_auc')} |")
        tables += ['']
    tables += ['### Agreement giữa ba explainer — top-5', '',
               '| Dataset | Cặp phương pháp | Jaccard@5 |', '|---|---|---:|']
    part = ag[(ag.budget == 5) & (ag.metric == 'jaccard') &
              ~ag.method_a.isin(['random', 'nearest_first']) & ~ag.method_b.isin(['random', 'nearest_first'])]
    for r in part.itertuples():
        tables.append(f'| {r.dataset} | {r.method_a} / {r.method_b} | {r.mean:.4f} ± {r.std:.4f} |')
    tables += ['', 'SD là độ lệch chuẩn mẫu giữa node (ddof=1). Baseline được trung bình qua repeat trong mỗi node trước khi tổng hợp.',
               'Fidelity: 20 node/dataset. Motif AUROC BA-Shapes: 10/20 node có hai lớp; Cora không có ground truth.',
               'Drop = p(gốc) − p(xóa). Inv = p(gốc) − p(giữ); Inv âm không tự động có nghĩa tốt hơn.', '']
    section = '\n'.join(tables)
    readme = root / 'README.md'
    text = readme.read_text(encoding='utf-8')
    if START not in text or END not in text:
        raise ValueError('README lacks evaluation markers; refusing to replace unrelated content')
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    readme.write_text(before + START + '\n' + section + END + after, encoding='utf-8')
    notes = ['# BẢNG SỐ LIỆU MIDTERM T06 — ĐỨC + KHA', '',
             '**Đề tài:** Making Graph Machine Learning Models Explainable — Nhóm 09.',
             '**Phụ trách:** Kha: baseline/agreement; Đức: fidelity/ground truth; hai bạn tổng hợp T06, rà soát T09.',
             '**Phạm vi:** không gồm slide. Nguồn điểm giải thích: T05 của Thiện + Khôi.', '',
             '## 1. Gói bàn giao', '',
             '- Notebook: [01_baseline_evaluation.ipynb](../notebooks/05_duc_kha_evaluation/01_baseline_evaluation.ipynb).',
             '- [Kết quả đầy đủ](../notebooks/05_duc_kha_evaluation/results/): metrics, summary, agreement, runtime, ranks, audit, manifest, biểu đồ và report.',
             '- 40 node validation, 3 explainer, 2 baseline; k = 1, 3, 5, 10; 5 repeat baseline.',
             f'- {len(audit)} audit rows; {len(metrics)} metric rows; {len(agreement_rows)} agreement rows trong lượt này.',
             '- Các con số trên là số dòng kết quả, không phải số mẫu độc lập.', '',
             '## 2. Khác biệt so với bảng T05', '',
             '- Random: điểm thô bằng nhau; nearest-first: khoảng cách BFS; không thêm nhiễu vào điểm AUC.',
             '- Thứ tự phá hòa theo seed được dùng lại cho tất cả ngân sách trong cùng repeat.',
             '- Thống kê trên từng node, SD mẫu; ghi số node hợp lệ và lý do NaN.',
             '- Chỉ bộ đánh giá mới dùng giao thức này; giữ nguyên module và notebook T05 gốc để bảo toàn kết quả lịch sử.',
             '- Kết quả khác T05 do giao thức đánh giá thay đổi không có nghĩa scores T05 bị thay.',
             '- Trung bình candidate/node: BA-Shapes 173,65; Cora 35,15 (mỗi explainer).', '',
             '## 3. Bảng kết quả', '', section,
             '## 4. Cách đọc và nhận xét sơ bộ', '',
             f"1. BA-Shapes top-5, nearest-first Drop = {cell('bashapes', 'nearest_first', 'fid_drop')}; Random = {cell('bashapes', 'random', 'fid_drop')}. Cần đối chiếu baseline đơn giản.",
             f"2. PGExplainer BA-Shapes AUROC = {cell('bashapes', 'pgexplainer', 'roc_auc')}; cần đọc cùng số node hợp lệ trong summary.csv, không suy rộng toàn graph.",
             f"3. Saliency BA-Shapes Drop = {cell('bashapes', 'saliency', 'fid_drop')}, AUROC = {cell('bashapes', 'saliency', 'roc_auc')}: ảnh hưởng dự đoán và khớp motif là hai tiêu chí khác nhau.",
             '4. Cora có Drop trung bình nhỏ và SD lớn; chưa đủ kết luận phương pháp nào vượt trội tổng quát.',
             f"5. Jaccard@5 giữa các explainer từ {part['mean'].min():.4f} đến {part['mean'].max():.4f}; đồng thuận không thay thế fidelity hoặc ground truth.", '',
             'Các nhận xét số cụ thể trên thuộc lượt 40 node đã kiểm tra; nếu thay cấu hình phải rà soát lại chúng.', '',
             '## 5. Giá trị bất thường và giới hạn', '',
             '- BA-Shapes: 4 node có toàn cạnh positive, 6 node không có positive ⇒ 10 node không có AUROC; giữ NaN.',
             '- Recall BA-Shapes có 14 node hợp lệ; precision có 20. Không so số node hợp lệ giữa metric như cùng mẫu.',
             '- Hai node Cora có dưới 5 candidate, nên k_eff < 5; xem metrics.csv.',
             '- Giá trị fidelity âm được giữ nguyên và giải thích, không cắt về 0.',
             '- Một checkpoint/dataset và lượt explainer T05 hiện có; 5 repeat baseline không phải 5 seed GNN.',
             '- Runtime từng node của ba explainer chưa có trong CSV T05; runtime.csv ghi unavailable.',
             '- CSV nguồn thiếu checkpoint hash lúc tạo scores. Các kiểm tra dự đoán/candidate chưa chứng minh cùng trọng số nguồn.',
             '- Phần thiếu runtime/provenance phải nhận xác nhận/log từ nhóm T05, không tạo số liệu thay thế.',
             '- Không train lại GCN/explainer và không dùng test để điều chỉnh prototype.', '',
             '## 6. Nguồn kết quả và trạng thái', '',
             'Kết quả trong results/ được nhập nguyên từ lượt chạy thành công trên máy Kha, đã kiểm tra các bảng số học và thứ hạng.',
             'Notebook có thể đọc kết quả bàn giao hoặc chạy lại khi có đúng data.pt/gcn_ckpt.pt.',
             f"SHA-256 CSV nguồn: `{manifest['source_scores_sha256']}`.",
             'manifest.json giữ nguyên cấu hình/hash của lượt chạy gốc; không sửa manifest để giả nhận là lượt chạy notebook mới.', '',
             'T03 và phần tính toán T06 có kết quả; T09 có bảng và nhận xét. Runtime explainer/provenance nguồn vẫn là giới hạn bàn giao.', '']
    doc = root / 'docs/BANG_SO_LIEU_MIDTERM_T06_DUC_KHA.md'
    doc.write_text('\n'.join(notes), encoding='utf-8')
    return doc
