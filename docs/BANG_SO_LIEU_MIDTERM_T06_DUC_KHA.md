# BẢNG SỐ LIỆU MIDTERM T06 — ĐỨC + KHA

**Đề tài:** Making Graph Machine Learning Models Explainable — Nhóm 09.
**Phụ trách:** Kha: baseline/agreement; Đức: fidelity/ground truth; hai bạn tổng hợp T06, rà soát T09.
**Phạm vi:** không gồm slide. Nguồn điểm giải thích: T05 của Thiện + Khôi.

## 1. Gói bàn giao

- Notebook: [01_baseline_evaluation.ipynb](../notebooks/05_duc_kha_evaluation/01_baseline_evaluation.ipynb).
- [Kết quả đầy đủ](../notebooks/05_duc_kha_evaluation/results/): metrics, summary, agreement, runtime, ranks, audit, manifest, biểu đồ và report.
- 40 node validation, 3 explainer, 2 baseline; k = 1, 3, 5, 10; 5 repeat baseline.
- 120 audit rows; 2080 metric rows; 6080 agreement rows trong lượt này.
- Các con số trên là số dòng kết quả, không phải số mẫu độc lập.

## 2. Khác biệt so với bảng T05

- Random: điểm thô bằng nhau; nearest-first: khoảng cách BFS; không thêm nhiễu vào điểm AUC.
- Thứ tự phá hòa theo seed được dùng lại cho tất cả ngân sách trong cùng repeat.
- Thống kê trên từng node, SD mẫu; ghi số node hợp lệ và lý do NaN.
- Chỉ bộ đánh giá mới dùng giao thức này; giữ nguyên module và notebook T05 gốc để bảo toàn kết quả lịch sử.
- Kết quả khác T05 do giao thức đánh giá thay đổi không có nghĩa scores T05 bị thay.
- Trung bình candidate/node: BA-Shapes 173,65; Cora 35,15 (mỗi explainer).

## 3. Bảng kết quả

### BA-Shapes — top-5

| Phương pháp | Fidelity Drop | Fidelity Inv | Motif AUROC |
|---|---:|---:|---:|
| random | 0.1666 ± 0.2201 | 0.3983 ± 0.3515 | 0.5000 ± 0.0000 |
| nearest_first | 0.3728 ± 0.4060 | 0.1508 ± 0.2636 | 0.6525 ± 0.0828 |
| saliency | 0.3089 ± 0.3356 | 0.0900 ± 0.2273 | 0.4206 ± 0.3425 |
| gnnexplainer | 0.3088 ± 0.3771 | 0.2111 ± 0.4259 | 0.7392 ± 0.2078 |
| pgexplainer | 0.2195 ± 0.3216 | 0.3248 ± 0.4258 | 0.8000 ± 0.4216 |

### BA-Shapes — Average Precision (điểm thô)

| Phương pháp | Average Precision | Node hợp lệ |
|---|---:|---:|
| random | 0.6474 ± 0.3598 | 10/20 |
| nearest_first | 0.6952 ± 0.3719 | 10/20 |
| saliency | 0.6608 ± 0.3359 | 10/20 |
| gnnexplainer | 0.7570 ± 0.4003 | 10/20 |
| pgexplainer | 0.8010 ± 0.4195 | 10/20 |

AP dùng toàn bộ điểm cạnh, không phụ thuộc top-k; bảng này lấy các dòng budget=5 để tránh lặp. Bộ đánh giá hiện tại chỉ tổng hợp AP trên node có hai lớp (10/20), cùng tập hợp lệ với AUROC. Đây là quy ước của bộ đánh giá; AP vẫn có thể được định nghĩa khi toàn nhãn positive. AP không phải PR-AUC hình thang. Với Random điểm hằng, AP bằng tỷ lệ cạnh positive của từng node, không mặc định bằng 0.5.

### Cora — top-5

| Phương pháp | Fidelity Drop | Fidelity Inv | Motif AUROC |
|---|---:|---:|---:|
| random | -0.0128 ± 0.0763 | 0.0224 ± 0.2180 | N/A |
| nearest_first | 0.0114 ± 0.2473 | -0.0422 ± 0.0946 | N/A |
| saliency | 0.0438 ± 0.2530 | -0.0796 ± 0.0892 | N/A |
| gnnexplainer | 0.0316 ± 0.0810 | -0.0504 ± 0.2645 | N/A |
| pgexplainer | -0.0077 ± 0.0745 | -0.0051 ± 0.2560 | N/A |

### Agreement giữa ba explainer — top-5

| Dataset | Cặp phương pháp | Jaccard@5 |
|---|---|---:|
| bashapes | gnnexplainer / pgexplainer | 0.3810 ± 0.3406 |
| bashapes | saliency / gnnexplainer | 0.4190 ± 0.3974 |
| bashapes | saliency / pgexplainer | 0.3563 ± 0.3086 |
| cora | gnnexplainer / pgexplainer | 0.3137 ± 0.3637 |
| cora | saliency / gnnexplainer | 0.2685 ± 0.3505 |
| cora | saliency / pgexplainer | 0.2923 ± 0.3691 |

### Tương quan điểm cạnh giữa ba explainer — Kendall tau-b

| Dataset | Cặp phương pháp | Kendall tau-b | Node hợp lệ |
|---|---|---:|---:|
| bashapes | gnnexplainer / pgexplainer | 0.1469 ± 0.2401 | 18/20 |
| bashapes | saliency / gnnexplainer | -0.0381 ± 0.4615 | 20/20 |
| bashapes | saliency / pgexplainer | -0.0560 ± 0.2635 | 18/20 |
| cora | gnnexplainer / pgexplainer | 0.1016 ± 0.2974 | 20/20 |
| cora | saliency / gnnexplainer | 0.0157 ± 0.4227 | 20/20 |
| cora | saliency / pgexplainer | 0.1221 ± 0.3149 | 20/20 |

Kendall dùng điểm trên toàn tập cạnh ứng viên, không chỉ top-5; bảng lấy các dòng budget=5. Các trường hợp không xác định được giữ NaN. Jaccard đo trùng tập cạnh được chọn, còn Kendall đo tương quan thứ hạng điểm.

### Runtime baseline trong lượt chạy notebook

Đơn vị: giây/node; mean ± SD giữa node sau khi trung bình repeat trong node. Số liệu được chép từ output hiển thị (6 chữ số thập phân); có thể thay đổi giữa các lần chạy/máy.

| Dataset | Baseline | Tạo điểm | Xếp hạng | Node hợp lệ |
|---|---|---:|---:|---:|
| bashapes | random | 0.000005 ± 0.000001 | 0.000173 ± 0.000026 | 20/20 |
| bashapes | nearest_first | 0.000119 ± 0.000137 | 0.000178 ± 0.000037 | 20/20 |
| cora | random | 0.000019 ± 0.000003 | 0.000340 ± 0.000068 | 20/20 |
| cora | nearest_first | 0.000114 ± 0.000057 | 0.000342 ± 0.000072 | 20/20 |

Thời gian tạo điểm/xếp hạng ba explainer trong bảng runtime hiện là N/A (0/20 node có số đo); không dùng thời gian fidelity thay thế. Bảng này không phải tổng thời gian chạy đánh giá.

SD là độ lệch chuẩn mẫu giữa node (ddof=1). Baseline được trung bình qua repeat trong mỗi node trước khi tổng hợp.
Fidelity: 20 node/dataset. Motif AUROC BA-Shapes: 10/20 node có hai lớp; Cora không có ground truth.
Drop = p(gốc) − p(xóa). Inv = p(gốc) − p(giữ); Inv âm không tự động có nghĩa tốt hơn.

## 4. Cách đọc và nhận xét sơ bộ

1. BA-Shapes top-5, nearest-first Drop = 0.3728 ± 0.4060; Random = 0.1666 ± 0.2201. Cần đối chiếu baseline đơn giản.
2. PGExplainer BA-Shapes AUROC = 0.8000 ± 0.4216; cần đọc cùng số node hợp lệ trong summary.csv, không suy rộng toàn graph.
3. Saliency BA-Shapes Drop = 0.3089 ± 0.3356, AUROC = 0.4206 ± 0.3425: ảnh hưởng dự đoán và khớp motif là hai tiêu chí khác nhau.
4. Cora có Drop trung bình nhỏ và SD lớn; chưa đủ kết luận phương pháp nào vượt trội tổng quát.
5. Jaccard@5 giữa các explainer từ 0.2685 đến 0.4190; đồng thuận không thay thế fidelity hoặc ground truth.

Các nhận xét số cụ thể trên thuộc lượt 40 node đã kiểm tra; nếu thay cấu hình phải rà soát lại chúng.

## 5. Giá trị bất thường và giới hạn

- BA-Shapes: 4 node có toàn cạnh positive, 6 node không có positive ⇒ 10 node không có AUROC; giữ NaN.
- Recall BA-Shapes có 14 node hợp lệ; precision có 20. Không so số node hợp lệ giữa metric như cùng mẫu.
- Hai node Cora 528 và 233 có dưới 5 candidate, nên k_eff < 5; xem metrics.csv.
- Có 856/2080 dòng Fidelity Drop âm, gồm nhiều phương pháp, repeat và ngân sách; đây không phải 856 node độc lập.
- Giá trị fidelity âm được giữ nguyên và giải thích, không cắt về 0.
- Một checkpoint/dataset và lượt explainer T05 hiện có; 5 repeat baseline không phải 5 seed GNN.
- Runtime từng node của ba explainer chưa có trong CSV T05; runtime.csv ghi unavailable.
- CSV nguồn thiếu checkpoint hash lúc tạo scores. Các kiểm tra dự đoán/candidate chưa chứng minh cùng trọng số nguồn.
- Phần thiếu runtime/provenance phải nhận xác nhận/log từ nhóm T05, không tạo số liệu thay thế.
- Không train lại GCN/explainer và không dùng test để điều chỉnh prototype.

## 6. Nguồn kết quả và trạng thái

Notebook đính kèm đã chạy lại predictor trên máy Kha với `RUN_EVALUATION = True`, CPU; kernel: `D:\Projects\IS353_C10-clean\.venv\Scripts\python.exe`.
Output báo `completed`, 20 node BA-Shapes và 20 node Cora, 120 audit rows, 2080 metric rows, 6080 agreement rows, 0 lỗi.
Các cell Python có execution count liên tiếp 13–23, không lưu output lỗi.

Các kiểm tra trong notebook đã PASS:

- Đủ số dòng; mọi metric row có status success; audit xác minh graph.
- Fidelity Drop/Inv khớp phép trừ xác suất; k_eff khớp ngân sách và số candidate.
- Mean, SD và số node hợp lệ của sáu metric khớp bảng tổng hợp sau khi trung bình repeat trong từng node.
- Toàn bộ Jaccard khớp tập cạnh từ ranked_edges.csv.

Các bảng fidelity, AUROC và Jaccard ban đầu khớp output mới ở độ chính xác 4 chữ số thập phân nên được giữ nguyên. Bản này bổ sung AP, Kendall và runtime baseline từ output notebook.
`UPDATE_DOCS = False` trong lượt chạy đã lưu, vì vậy cell cuối chưa tự cập nhật docs/README; bảng này được đối chiếu và cập nhật riêng.

Notebook kiểm tra hash CSV với cả LF/CRLF và không báo lỗi. Hash được ghi trong bản bàn giao trước: `102b49ffa3cc0a8e1acd187a0f2c2535fbf5c5b9d59dcaccc6e12076336152c2`. Notebook không in hash mới, nên không xác nhận đây là hash byte của CSV trong lượt mới. Cấu hình/hash thực tế của lượt chạy mới cần lấy từ `results/manifest.json` do runner xuất; không sửa thủ công manifest.

Phạm vi rà soát lần này: code và output đã lưu trong notebook, đối chiếu Markdown. Không chạy lại predictor độc lập và không có mã helper `src/sna/evaluation/` hay CSV mới đính kèm để kiểm toán thuật toán hoặc giá trị đầy đủ ngoài output hiển thị.

T03 và phần tính toán T06 có kết quả; T09 có bảng và nhận xét. Runtime explainer/provenance nguồn vẫn là giới hạn bàn giao.
