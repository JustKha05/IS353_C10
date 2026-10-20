# Đánh giá T03/T06/T09 — Đức + Kha

Chế độ: **full**. Trạng thái: **completed**.

## Phạm vi và khả năng truy vết

- Chỉ đánh giá danh sách validation đã chọn; không huấn luyện lại GCN/explainer.
- manifest.json chứa hash đầu vào, dữ liệu/checkpoint khi có; ranked_edges.csv lưu thứ tự chọn cạnh.
- CSV gốc không chứa hash checkpoint: khớp dự đoán/tập cạnh chưa chứng minh cùng trọng số lúc tạo scores.
- Mỗi repeat baseline dùng một thứ tự cho tất cả ngân sách; điểm thô không thêm nhiễu.
- SD trong bảng là độ lệch chuẩn mẫu giữa các node, không phải sai số chuẩn và không phải giữa các repeat.
- Fidelity Drop = p(gốc) − p(xóa); Inv = p(gốc) − p(giữ). Giá trị âm được giữ nguyên.
- Inv âm nghĩa là xác suất tăng khi chỉ giữ cạnh; không tự động coi âm hơn là giải thích tốt hơn.
- Giữ cạnh: chỉ giữ cặp được chọn trên toàn bộ đặc trưng; self-loop của mô hình vẫn hoạt động.
- AUROC chỉ tính khi có hai lớp; AP là Average Precision, không đồng nhất PR-AUC hình thang.
- Không có ground truth Cora; metric không xác định ghi NaN và lý do, không thay bằng 0.
- T05 không có thời gian explainer từng node: runtime.csv ghi unavailable, không suy đoán.

## Bảng kết quả

| Dataset | Method | k | Metric | Mean | SD | Node hợp lệ |
|---|---|---:|---|---:|---:|---:|
| bashapes | gnnexplainer | 1 | fid_drop | 0.0926 | 0.1903 | 20/20 |
| bashapes | gnnexplainer | 1 | fid_inv | 0.3914 | 0.4150 | 20/20 |
| bashapes | gnnexplainer | 1 | roc_auc | 0.7392 | 0.2078 | 10/20 |
| bashapes | gnnexplainer | 1 | average_precision | 0.7570 | 0.4003 | 10/20 |
| bashapes | gnnexplainer | 1 | precision | 0.6000 | 0.5026 | 20/20 |
| bashapes | gnnexplainer | 1 | recall | 0.1429 | 0.0605 | 14/20 |
| bashapes | gnnexplainer | 3 | fid_drop | 0.2191 | 0.2737 | 20/20 |
| bashapes | gnnexplainer | 3 | fid_inv | 0.2891 | 0.4448 | 20/20 |
| bashapes | gnnexplainer | 3 | roc_auc | 0.7392 | 0.2078 | 10/20 |
| bashapes | gnnexplainer | 3 | average_precision | 0.7570 | 0.4003 | 10/20 |
| bashapes | gnnexplainer | 3 | precision | 0.5833 | 0.4941 | 20/20 |
| bashapes | gnnexplainer | 3 | recall | 0.4167 | 0.1820 | 14/20 |
| bashapes | gnnexplainer | 5 | fid_drop | 0.3088 | 0.3771 | 20/20 |
| bashapes | gnnexplainer | 5 | fid_inv | 0.2111 | 0.4259 | 20/20 |
| bashapes | gnnexplainer | 5 | roc_auc | 0.7392 | 0.2078 | 10/20 |
| bashapes | gnnexplainer | 5 | average_precision | 0.7570 | 0.4003 | 10/20 |
| bashapes | gnnexplainer | 5 | precision | 0.5400 | 0.4639 | 20/20 |
| bashapes | gnnexplainer | 5 | recall | 0.6429 | 0.2913 | 14/20 |
| bashapes | gnnexplainer | 10 | fid_drop | 0.2088 | 0.3226 | 20/20 |
| bashapes | gnnexplainer | 10 | fid_inv | 0.2805 | 0.4108 | 20/20 |
| bashapes | gnnexplainer | 10 | roc_auc | 0.7392 | 0.2078 | 10/20 |
| bashapes | gnnexplainer | 10 | average_precision | 0.7570 | 0.4003 | 10/20 |
| bashapes | gnnexplainer | 10 | precision | 0.5300 | 0.4522 | 20/20 |
| bashapes | gnnexplainer | 10 | recall | 0.8571 | 0.3631 | 14/20 |
| bashapes | nearest_first | 1 | fid_drop | 0.1857 | 0.2381 | 20/20 |
| bashapes | nearest_first | 1 | fid_inv | 0.2749 | 0.2459 | 20/20 |
| bashapes | nearest_first | 1 | roc_auc | 0.6525 | 0.0828 | 10/20 |
| bashapes | nearest_first | 1 | average_precision | 0.6952 | 0.3719 | 10/20 |
| bashapes | nearest_first | 1 | precision | 0.5800 | 0.4938 | 20/20 |
| bashapes | nearest_first | 1 | recall | 0.1381 | 0.0611 | 14/20 |
| bashapes | nearest_first | 3 | fid_drop | 0.2357 | 0.3369 | 20/20 |
| bashapes | nearest_first | 3 | fid_inv | 0.1494 | 0.1850 | 20/20 |
| bashapes | nearest_first | 3 | roc_auc | 0.6525 | 0.0828 | 10/20 |
| bashapes | nearest_first | 3 | average_precision | 0.6952 | 0.3719 | 10/20 |
| bashapes | nearest_first | 3 | precision | 0.5800 | 0.4880 | 20/20 |
| bashapes | nearest_first | 3 | recall | 0.4143 | 0.1777 | 14/20 |
| bashapes | nearest_first | 5 | fid_drop | 0.3728 | 0.4060 | 20/20 |
| bashapes | nearest_first | 5 | fid_inv | 0.1508 | 0.2636 | 20/20 |
| bashapes | nearest_first | 5 | roc_auc | 0.6525 | 0.0828 | 10/20 |
| bashapes | nearest_first | 5 | average_precision | 0.6952 | 0.3719 | 10/20 |
| bashapes | nearest_first | 5 | precision | 0.5320 | 0.4528 | 20/20 |
| bashapes | nearest_first | 5 | recall | 0.6333 | 0.2801 | 14/20 |
| bashapes | nearest_first | 10 | fid_drop | 0.4065 | 0.4199 | 20/20 |
| bashapes | nearest_first | 10 | fid_inv | 0.0775 | 0.2116 | 20/20 |
| bashapes | nearest_first | 10 | roc_auc | 0.6525 | 0.0828 | 10/20 |
| bashapes | nearest_first | 10 | average_precision | 0.6952 | 0.3719 | 10/20 |
| bashapes | nearest_first | 10 | precision | 0.5230 | 0.4521 | 20/20 |
| bashapes | nearest_first | 10 | recall | 0.8405 | 0.3614 | 14/20 |
| bashapes | pgexplainer | 1 | fid_drop | 0.0740 | 0.1904 | 20/20 |
| bashapes | pgexplainer | 1 | fid_inv | 0.4721 | 0.4054 | 20/20 |
| bashapes | pgexplainer | 1 | roc_auc | 0.8000 | 0.4216 | 10/20 |
| bashapes | pgexplainer | 1 | average_precision | 0.8010 | 0.4195 | 10/20 |
| bashapes | pgexplainer | 1 | precision | 0.6000 | 0.5026 | 20/20 |
| bashapes | pgexplainer | 1 | recall | 0.1429 | 0.0605 | 14/20 |
| bashapes | pgexplainer | 3 | fid_drop | 0.1657 | 0.2638 | 20/20 |
| bashapes | pgexplainer | 3 | fid_inv | 0.4156 | 0.4089 | 20/20 |
| bashapes | pgexplainer | 3 | roc_auc | 0.8000 | 0.4216 | 10/20 |
| bashapes | pgexplainer | 3 | average_precision | 0.8010 | 0.4195 | 10/20 |
| bashapes | pgexplainer | 3 | precision | 0.6000 | 0.5026 | 20/20 |
| bashapes | pgexplainer | 3 | recall | 0.4286 | 0.1816 | 14/20 |
| bashapes | pgexplainer | 5 | fid_drop | 0.2195 | 0.3216 | 20/20 |
| bashapes | pgexplainer | 5 | fid_inv | 0.3248 | 0.4258 | 20/20 |
| bashapes | pgexplainer | 5 | roc_auc | 0.8000 | 0.4216 | 10/20 |
| bashapes | pgexplainer | 5 | average_precision | 0.8010 | 0.4195 | 10/20 |
| bashapes | pgexplainer | 5 | precision | 0.6000 | 0.5026 | 20/20 |
| bashapes | pgexplainer | 5 | recall | 0.7143 | 0.3026 | 14/20 |
| bashapes | pgexplainer | 10 | fid_drop | 0.2089 | 0.3226 | 20/20 |
| bashapes | pgexplainer | 10 | fid_inv | 0.1552 | 0.2771 | 20/20 |
| bashapes | pgexplainer | 10 | roc_auc | 0.8000 | 0.4216 | 10/20 |
| bashapes | pgexplainer | 10 | average_precision | 0.8010 | 0.4195 | 10/20 |
| bashapes | pgexplainer | 10 | precision | 0.5300 | 0.4522 | 20/20 |
| bashapes | pgexplainer | 10 | recall | 0.8571 | 0.3631 | 14/20 |
| bashapes | random | 1 | fid_drop | 0.1210 | 0.1337 | 20/20 |
| bashapes | random | 1 | fid_inv | 0.4119 | 0.3802 | 20/20 |
| bashapes | random | 1 | roc_auc | 0.5000 | 0.0000 | 10/20 |
| bashapes | random | 1 | average_precision | 0.6474 | 0.3598 | 10/20 |
| bashapes | random | 1 | precision | 0.5300 | 0.4555 | 20/20 |
| bashapes | random | 1 | recall | 0.1262 | 0.0573 | 14/20 |
| bashapes | random | 3 | fid_drop | 0.1836 | 0.2202 | 20/20 |
| bashapes | random | 3 | fid_inv | 0.3854 | 0.3616 | 20/20 |
| bashapes | random | 3 | roc_auc | 0.5000 | 0.0000 | 10/20 |
| bashapes | random | 3 | average_precision | 0.6474 | 0.3598 | 10/20 |
| bashapes | random | 3 | precision | 0.5233 | 0.4502 | 20/20 |
| bashapes | random | 3 | recall | 0.3738 | 0.1701 | 14/20 |
| bashapes | random | 5 | fid_drop | 0.1666 | 0.2201 | 20/20 |
| bashapes | random | 5 | fid_inv | 0.3983 | 0.3515 | 20/20 |
| bashapes | random | 5 | roc_auc | 0.5000 | 0.0000 | 10/20 |
| bashapes | random | 5 | average_precision | 0.6474 | 0.3598 | 10/20 |
| bashapes | random | 5 | precision | 0.5180 | 0.4495 | 20/20 |
| bashapes | random | 5 | recall | 0.6167 | 0.2867 | 14/20 |
| bashapes | random | 10 | fid_drop | 0.2191 | 0.3267 | 20/20 |
| bashapes | random | 10 | fid_inv | 0.2677 | 0.3788 | 20/20 |
| bashapes | random | 10 | roc_auc | 0.5000 | 0.0000 | 10/20 |
| bashapes | random | 10 | average_precision | 0.6474 | 0.3598 | 10/20 |
| bashapes | random | 10 | precision | 0.5210 | 0.4525 | 20/20 |
| bashapes | random | 10 | recall | 0.8357 | 0.3629 | 14/20 |
| bashapes | saliency | 1 | fid_drop | 0.0961 | 0.2377 | 20/20 |
| bashapes | saliency | 1 | fid_inv | 0.3792 | 0.3887 | 20/20 |
| bashapes | saliency | 1 | roc_auc | 0.4206 | 0.3425 | 10/20 |
| bashapes | saliency | 1 | average_precision | 0.6608 | 0.3359 | 10/20 |
| bashapes | saliency | 1 | precision | 0.4500 | 0.5104 | 20/20 |
| bashapes | saliency | 1 | recall | 0.1071 | 0.0829 | 14/20 |
| bashapes | saliency | 3 | fid_drop | 0.1896 | 0.3001 | 20/20 |
| bashapes | saliency | 3 | fid_inv | 0.3044 | 0.3733 | 20/20 |
| bashapes | saliency | 3 | roc_auc | 0.4206 | 0.3425 | 10/20 |
| bashapes | saliency | 3 | average_precision | 0.6608 | 0.3359 | 10/20 |
| bashapes | saliency | 3 | precision | 0.5000 | 0.4393 | 20/20 |
| bashapes | saliency | 3 | recall | 0.3571 | 0.1712 | 14/20 |
| bashapes | saliency | 5 | fid_drop | 0.3089 | 0.3356 | 20/20 |
| bashapes | saliency | 5 | fid_inv | 0.0900 | 0.2273 | 20/20 |
| bashapes | saliency | 5 | roc_auc | 0.4206 | 0.3425 | 10/20 |
| bashapes | saliency | 5 | average_precision | 0.6608 | 0.3359 | 10/20 |
| bashapes | saliency | 5 | precision | 0.5100 | 0.4376 | 20/20 |
| bashapes | saliency | 5 | recall | 0.6071 | 0.2743 | 14/20 |
| bashapes | saliency | 10 | fid_drop | 0.4102 | 0.4165 | 20/20 |
| bashapes | saliency | 10 | fid_inv | 0.0828 | 0.2098 | 20/20 |
| bashapes | saliency | 10 | roc_auc | 0.4206 | 0.3425 | 10/20 |
| bashapes | saliency | 10 | average_precision | 0.6608 | 0.3359 | 10/20 |
| bashapes | saliency | 10 | precision | 0.5300 | 0.4522 | 20/20 |
| bashapes | saliency | 10 | recall | 0.8571 | 0.3631 | 14/20 |
| cora | gnnexplainer | 1 | fid_drop | -0.0011 | 0.0298 | 20/20 |
| cora | gnnexplainer | 1 | fid_inv | 0.0236 | 0.2541 | 20/20 |
| cora | gnnexplainer | 1 | roc_auc | nan | nan | 0/20 |
| cora | gnnexplainer | 1 | average_precision | nan | nan | 0/20 |
| cora | gnnexplainer | 1 | precision | nan | nan | 0/20 |
| cora | gnnexplainer | 1 | recall | nan | nan | 0/20 |
| cora | gnnexplainer | 3 | fid_drop | 0.0049 | 0.0549 | 20/20 |
| cora | gnnexplainer | 3 | fid_inv | -0.0349 | 0.2740 | 20/20 |
| cora | gnnexplainer | 3 | roc_auc | nan | nan | 0/20 |
| cora | gnnexplainer | 3 | average_precision | nan | nan | 0/20 |
| cora | gnnexplainer | 3 | precision | nan | nan | 0/20 |
| cora | gnnexplainer | 3 | recall | nan | nan | 0/20 |
| cora | gnnexplainer | 5 | fid_drop | 0.0316 | 0.0810 | 20/20 |
| cora | gnnexplainer | 5 | fid_inv | -0.0504 | 0.2645 | 20/20 |
| cora | gnnexplainer | 5 | roc_auc | nan | nan | 0/20 |
| cora | gnnexplainer | 5 | average_precision | nan | nan | 0/20 |
| cora | gnnexplainer | 5 | precision | nan | nan | 0/20 |
| cora | gnnexplainer | 5 | recall | nan | nan | 0/20 |
| cora | gnnexplainer | 10 | fid_drop | 0.0108 | 0.1119 | 20/20 |
| cora | gnnexplainer | 10 | fid_inv | -0.0570 | 0.2483 | 20/20 |
| cora | gnnexplainer | 10 | roc_auc | nan | nan | 0/20 |
| cora | gnnexplainer | 10 | average_precision | nan | nan | 0/20 |
| cora | gnnexplainer | 10 | precision | nan | nan | 0/20 |
| cora | gnnexplainer | 10 | recall | nan | nan | 0/20 |
| cora | nearest_first | 1 | fid_drop | 0.0165 | 0.0649 | 20/20 |
| cora | nearest_first | 1 | fid_inv | 0.0153 | 0.1669 | 20/20 |
| cora | nearest_first | 1 | roc_auc | nan | nan | 0/20 |
| cora | nearest_first | 1 | average_precision | nan | nan | 0/20 |
| cora | nearest_first | 1 | precision | nan | nan | 0/20 |
| cora | nearest_first | 1 | recall | nan | nan | 0/20 |
| cora | nearest_first | 3 | fid_drop | 0.0332 | 0.1520 | 20/20 |
| cora | nearest_first | 3 | fid_inv | -0.0301 | 0.1302 | 20/20 |
| cora | nearest_first | 3 | roc_auc | nan | nan | 0/20 |
| cora | nearest_first | 3 | average_precision | nan | nan | 0/20 |
| cora | nearest_first | 3 | precision | nan | nan | 0/20 |
| cora | nearest_first | 3 | recall | nan | nan | 0/20 |
| cora | nearest_first | 5 | fid_drop | 0.0114 | 0.2473 | 20/20 |
| cora | nearest_first | 5 | fid_inv | -0.0422 | 0.0946 | 20/20 |
| cora | nearest_first | 5 | roc_auc | nan | nan | 0/20 |
| cora | nearest_first | 5 | average_precision | nan | nan | 0/20 |
| cora | nearest_first | 5 | precision | nan | nan | 0/20 |
| cora | nearest_first | 5 | recall | nan | nan | 0/20 |
| cora | nearest_first | 10 | fid_drop | 0.0228 | 0.2536 | 20/20 |
| cora | nearest_first | 10 | fid_inv | -0.0553 | 0.0478 | 20/20 |
| cora | nearest_first | 10 | roc_auc | nan | nan | 0/20 |
| cora | nearest_first | 10 | average_precision | nan | nan | 0/20 |
| cora | nearest_first | 10 | precision | nan | nan | 0/20 |
| cora | nearest_first | 10 | recall | nan | nan | 0/20 |
| cora | pgexplainer | 1 | fid_drop | -0.0195 | 0.0571 | 20/20 |
| cora | pgexplainer | 1 | fid_inv | 0.0228 | 0.2544 | 20/20 |
| cora | pgexplainer | 1 | roc_auc | nan | nan | 0/20 |
| cora | pgexplainer | 1 | average_precision | nan | nan | 0/20 |
| cora | pgexplainer | 1 | precision | nan | nan | 0/20 |
| cora | pgexplainer | 1 | recall | nan | nan | 0/20 |
| cora | pgexplainer | 3 | fid_drop | -0.0158 | 0.0477 | 20/20 |
| cora | pgexplainer | 3 | fid_inv | 0.0186 | 0.2475 | 20/20 |
| cora | pgexplainer | 3 | roc_auc | nan | nan | 0/20 |
| cora | pgexplainer | 3 | average_precision | nan | nan | 0/20 |
| cora | pgexplainer | 3 | precision | nan | nan | 0/20 |
| cora | pgexplainer | 3 | recall | nan | nan | 0/20 |
| cora | pgexplainer | 5 | fid_drop | -0.0077 | 0.0745 | 20/20 |
| cora | pgexplainer | 5 | fid_inv | -0.0051 | 0.2560 | 20/20 |
| cora | pgexplainer | 5 | roc_auc | nan | nan | 0/20 |
| cora | pgexplainer | 5 | average_precision | nan | nan | 0/20 |
| cora | pgexplainer | 5 | precision | nan | nan | 0/20 |
| cora | pgexplainer | 5 | recall | nan | nan | 0/20 |
| cora | pgexplainer | 10 | fid_drop | -0.0144 | 0.0878 | 20/20 |
| cora | pgexplainer | 10 | fid_inv | 0.0252 | 0.2338 | 20/20 |
| cora | pgexplainer | 10 | roc_auc | nan | nan | 0/20 |
| cora | pgexplainer | 10 | average_precision | nan | nan | 0/20 |
| cora | pgexplainer | 10 | precision | nan | nan | 0/20 |
| cora | pgexplainer | 10 | recall | nan | nan | 0/20 |
| cora | random | 1 | fid_drop | -0.0033 | 0.0266 | 20/20 |
| cora | random | 1 | fid_inv | 0.0184 | 0.2516 | 20/20 |
| cora | random | 1 | roc_auc | nan | nan | 0/20 |
| cora | random | 1 | average_precision | nan | nan | 0/20 |
| cora | random | 1 | precision | nan | nan | 0/20 |
| cora | random | 1 | recall | nan | nan | 0/20 |
| cora | random | 3 | fid_drop | -0.0061 | 0.0399 | 20/20 |
| cora | random | 3 | fid_inv | 0.0277 | 0.2225 | 20/20 |
| cora | random | 3 | roc_auc | nan | nan | 0/20 |
| cora | random | 3 | average_precision | nan | nan | 0/20 |
| cora | random | 3 | precision | nan | nan | 0/20 |
| cora | random | 3 | recall | nan | nan | 0/20 |
| cora | random | 5 | fid_drop | -0.0128 | 0.0763 | 20/20 |
| cora | random | 5 | fid_inv | 0.0224 | 0.2180 | 20/20 |
| cora | random | 5 | roc_auc | nan | nan | 0/20 |
| cora | random | 5 | average_precision | nan | nan | 0/20 |
| cora | random | 5 | precision | nan | nan | 0/20 |
| cora | random | 5 | recall | nan | nan | 0/20 |
| cora | random | 10 | fid_drop | -0.0186 | 0.0780 | 20/20 |
| cora | random | 10 | fid_inv | 0.0248 | 0.1857 | 20/20 |
| cora | random | 10 | roc_auc | nan | nan | 0/20 |
| cora | random | 10 | average_precision | nan | nan | 0/20 |
| cora | random | 10 | precision | nan | nan | 0/20 |
| cora | random | 10 | recall | nan | nan | 0/20 |
| cora | saliency | 1 | fid_drop | 0.0684 | 0.0635 | 20/20 |
| cora | saliency | 1 | fid_inv | -0.0774 | 0.1905 | 20/20 |
| cora | saliency | 1 | roc_auc | nan | nan | 0/20 |
| cora | saliency | 1 | average_precision | nan | nan | 0/20 |
| cora | saliency | 1 | precision | nan | nan | 0/20 |
| cora | saliency | 1 | recall | nan | nan | 0/20 |
| cora | saliency | 3 | fid_drop | 0.0562 | 0.1560 | 20/20 |
| cora | saliency | 3 | fid_inv | -0.1134 | 0.1000 | 20/20 |
| cora | saliency | 3 | roc_auc | nan | nan | 0/20 |
| cora | saliency | 3 | average_precision | nan | nan | 0/20 |
| cora | saliency | 3 | precision | nan | nan | 0/20 |
| cora | saliency | 3 | recall | nan | nan | 0/20 |
| cora | saliency | 5 | fid_drop | 0.0438 | 0.2530 | 20/20 |
| cora | saliency | 5 | fid_inv | -0.0796 | 0.0892 | 20/20 |
| cora | saliency | 5 | roc_auc | nan | nan | 0/20 |
| cora | saliency | 5 | average_precision | nan | nan | 0/20 |
| cora | saliency | 5 | precision | nan | nan | 0/20 |
| cora | saliency | 5 | recall | nan | nan | 0/20 |
| cora | saliency | 10 | fid_drop | 0.0313 | 0.2530 | 20/20 |
| cora | saliency | 10 | fid_inv | -0.0709 | 0.1168 | 20/20 |
| cora | saliency | 10 | roc_auc | nan | nan | 0/20 |
| cora | saliency | 10 | average_precision | nan | nan | 0/20 |
| cora | saliency | 10 | precision | nan | nan | 0/20 |
| cora | saliency | 10 | recall | nan | nan | 0/20 |

## Nhận xét sơ bộ tự sinh

- bashapes, top-5: nearest_first có trung bình Fidelity Drop cao nhất trong các phương pháp tính được (0.3728; 20 node). Đây là mô tả mẫu, chưa phải kết luận vượt trội có ý nghĩa thống kê.
- bashapes, top-5: pgexplainer có trung bình motif AUROC cao nhất trong các phương pháp tính được (0.8000; 10 node). Đây là mô tả mẫu, chưa phải kết luận vượt trội có ý nghĩa thống kê.
- cora, top-5: saliency có trung bình Fidelity Drop cao nhất trong các phương pháp tính được (0.0438; 20 node). Đây là mô tả mẫu, chưa phải kết luận vượt trội có ý nghĩa thống kê.
- Jaccard@5 trung bình giữa các explainer nằm trong khoảng 0.2685–0.4190; đồng thuận không đồng nghĩa với đúng ground truth.
- Có 856 dòng Fidelity Drop âm (bao gồm các repeat/ngân sách); cần xem node cụ thể trong metrics.csv, không loại bỏ âm thầm.

## Giới hạn và phần còn thiếu

- Số lỗi/đầu vào thiếu: 0; chi tiết errors.csv. Node thành công: {'bashapes': 20, 'cora': 20}.
- Một checkpoint mỗi dataset, một lượt explainer T05; repeat baseline không thay thế nhiều seed GNN/explainer.
- Top-k có thể chiếm tỷ lệ rất khác nhau giữa node; xem k_eff và n_candidates.
- Thiếu hash checkpoint và thời gian trong bàn giao T05 cần được nhóm tạo scores xác nhận/bổ sung.
- Hướng tiếp: khóa giao thức, bổ sung log thời gian/provenance, tăng node và seed trước khi suy rộng.
