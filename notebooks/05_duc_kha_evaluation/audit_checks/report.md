# Đánh giá T03/T06/T09 — Đức + Kha

Chế độ: **audit_only**. Trạng thái: **audit_passed**.

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

## Kết quả kiểm toán CSV

Chưa nạp .pt, chưa xác minh graph/checkpoint/val_mask và chưa tính fidelity.
Các CSV metrics/summary của chế độ này không phải kết quả đánh giá mô hình.


## Giới hạn và phần còn thiếu

- Số lỗi/đầu vào thiếu: 0; chi tiết errors.csv. Node thành công: {'bashapes': 0, 'cora': 0}.
- Một checkpoint mỗi dataset, một lượt explainer T05; repeat baseline không thay thế nhiều seed GNN/explainer.
- Top-k có thể chiếm tỷ lệ rất khác nhau giữa node; xem k_eff và n_candidates.
- Thiếu hash checkpoint và thời gian trong bàn giao T05 cần được nhóm tạo scores xác nhận/bổ sung.
- Hướng tiếp: khóa giao thức, bổ sung log thời gian/provenance, tăng node và seed trước khi suy rộng.
