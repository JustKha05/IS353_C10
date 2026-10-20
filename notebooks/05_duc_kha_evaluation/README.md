# T03/T06/T09 — Đức + Kha

- Kha: Random, nearest-first, agreement. Đức: fidelity xóa/giữ và ground-truth metrics.
- Notebook chính: `01_baseline_evaluation.ipynb`.
- Kết quả: `results/`; bảng bàn giao: `../../docs/BANG_SO_LIEU_MIDTERM_T06_DUC_KHA.md`.

## Chạy trên Windows / VS Code

1. Giải nén gói bổ sung vào thư mục clone sạch của GitHub gốc (thư mục có README.md và src).
2. Chạy `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` rồi `.\.venv\Scripts\python.exe -m pip install -e .`.
3. Mở notebook, chọn kernel **Python (IS353 Duc Kha)**. Ô đầu in đường dẫn Python để kiểm tra.
4. Mặc định `RUN_EVALUATION = False`: Run All sẽ kiểm toán CSV và trình bày các kết quả thật đã bàn giao. Không nhận đây là lượt chạy predictor mới.
5. Muốn chạy lại: có `data/<dataset>/data.pt`, `gcn_ckpt.pt`, `config.json`, `val_nodes.json` đúng bộ T05 cho cả bashapes và cora. Đặt `RUN_EVALUATION = True`, rồi Restart Kernel → Run All. Kết quả mới ghi vào `results/`.
6. Chỉ đặt `UPDATE_DOCS = True` sau khi xem kết quả mới; cell cuối sẽ cập nhật bảng docs và phần README có marker. Bảng T05 gốc vẫn giữ nguyên trong docs.

Các phần notebook: vai trò file, giao thức T03, cấu hình, audit, chạy predictor hoặc đọc kết quả cũ, đối chiếu metrics, bảng top-5, agreement, runtime/lỗi, biểu đồ, nhận xét và xuất docs.

## Bộ helper trong src/sna/evaluation

- `core.py`: chọn cạnh, baseline, fidelity, ground truth, agreement.
- `runner.py`: kiểm tra scores/validation/checkpoint, chạy nhiều k/repeat và xuất CSV/report/biểu đồ.
- `handover.py`: cập nhật bảng T06 và mục kết quả README từ summary CSV.

Không cần chạy các test của repo để dùng notebook. Tests mới chỉ phục vụ kiểm tra logic của bộ đánh giá bằng dữ liệu nhỏ, không cần Drive.

## Nguồn kết quả đang kèm theo

results/ chứa nguyên kết quả trong `duc_kha_results.zip` từ lượt chạy CPU trên máy Kha; không phải số liệu tạo giả và chưa phải lượt chạy notebook mới. manifest giữ nguyên nguồn. Đã kiểm tra 120 audit, 2.080 metric, 6.080 Jaccard; số học và tổng hợp nhất quán. Không chạy lại predictor trong môi trường chuẩn bị gói vì chưa có `.pt` thật.

Kết quả gốc T05 vẫn có giá trị theo giao thức của nó; T06 đổi baseline/phá hòa, repeat và SD nên không trộn hai bảng. Ví dụ AUROC Random mới chính xác 0,5 khi đủ hai lớp; PGExplainer top-k có thể đổi do hòa điểm; SD mẫu khác SD tổng thể.

Hai giới hạn cần nhóm T05 bổ sung: runtime explainer từng node và hash checkpoint lúc tạo scores. Giữ rõ trong báo cáo; không tự tạo số thay thế.
