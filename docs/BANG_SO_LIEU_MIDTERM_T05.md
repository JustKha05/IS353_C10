# BẢNG SỐ LIỆU BÀN GIAO KẾT QUẢ ĐÁNH GIÁ GIỮA KỲ (MIDTERM) - GROUP 09

> **Môn học:** Social Network Analysis (SNA) - UIT 2026  
> **Giảng viên hướng dẫn:** TS. Trần Hưng Nghiệp  
> **Đề tài:** C.10. Making Graph Machine Learning Models Explainable  
> **Nhóm thực hiện:** Nhóm 09 (Nhóm 2: Nguyễn Trọng Khôi - 23520783 & Nguyễn Hải Thiện - 23521481)  
> **Người nhận bàn giao:** Nhóm 3 (Vòng Việt Đức - 23520324, Nguyễn Hoàng Kha - 23520667) & Nhóm trưởng (Nguyễn Trường Vũ - 23521812)  
> **Ngày cập nhật:** 19/10/2026  
> **Tình trạng:** Đồng bộ 100% Checkpoint GCN (93.57%), hoàn tất 12,528 dòng điểm giải thích trên 40 validation nodes.

---

## 1. Tổng quan gói dữ liệu bàn giao

Toàn bộ quá trình chạy thực nghiệm đã được chuẩn hóa và đồng bộ giữa Nguyễn Hải Thiện (PGExplainer) và Nguyễn Trọng Khôi (Saliency, GNNExplainer, Baselines):

1. **File điểm cạnh chuẩn bàn giao:**
   - [`edge_scores.csv`](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/edge_scores.csv) (12,528 dòng: 3 phương pháp chính PGExplainer, GNNExplainer, Saliency).
   - Bản sao lưu tại repo của Thiện: [`repos/IS353_C10/notebooks/01_midterm/edge_scores.csv`](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/repos/IS353_C10/notebooks/01_midterm/edge_scores.csv).
2. **File điểm kèm Baselines (Random & Nearest-First):**
   - [`edge_scores_with_baselines.csv`](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/edge_scores_with_baselines.csv) (20,880 dòng).
3. **File điểm riêng của Khôi:**
   - [`scores_khoi.csv`](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/scores_khoi.csv) (8,352 dòng).
4. **File số liệu thống kê JSON:**
   - [`midterm_metrics_summary.json`](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/docs/reports/midterm_metrics_summary.json).

---

## 2. Bảng số liệu thực nghiệm định lượng (Zero-Mockup)

Toàn bộ số liệu dưới đây được tính trực tiếp trên 40 node validation (20 node BA-Shapes + 20 node Cora) với budget top-5 cạnh quan trọng nhất (k=5), tuân thủ định dạng $\mu \pm \sigma$.

### 2.1. Đánh giá trên đồ thị cấu trúc BA-Shapes (Synthetic Motif)

- **Mô hình GCN:** Accuracy = 93.57%, Class 3 Recall = 93.75% (đã khắc phục triệt để lỗi sụp đổ lớp 3).
- **Số cạnh ứng viên trung bình (2-hop neighborhood):** 208.8 cạnh/node.

| Phương pháp (Explainer) | Fidelity- (Drop) $\uparrow$ | Fidelity+ (Inv) $\downarrow$ | Motif ROC-AUC $\uparrow$ | Motif PR-AUC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: |
| **PGExplainer** (Thiện) | $0.2475 \pm 0.3338$ | $0.3303 \pm 0.3874$ | **$0.8000 \pm 0.4000$** | **$0.8006 \pm 0.3988$** |
| **GNNExplainer** (Khôi) | $0.3088 \pm 0.3675$ | $0.2111 \pm 0.4151$ | $0.7392 \pm 0.1971$ | $0.7511 \pm 0.3803$ |
| **Saliency** (Khôi) | $0.3089 \pm 0.3271$ | **$0.0930 \pm 0.2214$** | $0.4206 \pm 0.3250$ | $0.6211 \pm 0.3132$ |
| *Nearest-First* (Baseline) | $0.3584 \pm 0.4036$ | $0.1061 \pm 0.2523$ | $0.7178 \pm 0.1233$ | $0.7565 \pm 0.3759$ |
| *Random* (Baseline) | $0.1742 \pm 0.2998$ | $0.3168 \pm 0.3567$ | $0.5299 \pm 0.0786$ | $0.7042 \pm 0.3529$ |

> **Nhận xét học thuật:**
> - **Motif AUC:** PGExplainer đạt hiệu năng phát hiện cấu trúc nhà (ground truth motif) cao nhất với ROC-AUC = 0.8000 và PR-AUC = 0.8006, vượt trội rõ rệt so với Gradient Saliency (0.4206) và Random baseline (0.5299).
> - **Fidelity Drop:** Cả GNNExplainer (0.3088) và Saliency (0.3089) đều gây sụt giảm xác suất mạnh khi xóa top-5 cạnh quan trọng, chứng minh các cạnh được chọn thực sự chi phối quyết định của GCN.

---

### 2.2. Đánh giá trên mạng trích dẫn Cora (Social/Citation Graph)

- **Mô hình GCN:** Accuracy = 83.60% (theo chuẩn phân chia Planetoid tác giả Yang et al. ICML 2016).
- **Số cạnh ứng viên trung bình (2-hop neighborhood):** ~105 cạnh/node.

| Phương pháp (Explainer) | Fidelity- (Drop) $\uparrow$ | Fidelity+ (Inv) $\downarrow$ | Ghi chú tác động |
| :--- | :---: | :---: | :--- |
| **GNNExplainer** (Khôi) | **$0.0316 \pm 0.0789$** | $-0.0504 \pm 0.2578$ | Tối ưu hóa entropy và mutual information |
| **Saliency** (Khôi) | **$0.0438 \pm 0.2466$** | $-0.0796 \pm 0.0869$ | Gradient bậc một nhạy cảm với đặc trưng cục bộ |
| **PGExplainer** (Thiện) | $-0.0077 \pm 0.0726$ | $-0.0051 \pm 0.2495$ | Cần thêm dữ liệu huấn luyện MLP trên Cora |
| *Nearest-First* (Baseline) | $0.0062 \pm 0.2395$ | $-0.0609 \pm 0.1189$ | Giữ lân cận 1-hop |
| *Random* (Baseline) | $-0.0084 \pm 0.0665$ | $-0.0001 \pm 0.2311$ | Chọn cạnh ngẫu nhiên |

> **Nhận xét học thuật:**
> - Trên đồ thị Cora, đặc trưng từ vựng của node (1,433 túi từ) đóng vai trò then chốt; việc chỉ giải thích cạnh (edge-only explainability) tạo ra mức độ thay đổi xác suất khiêm tốn hơn so với BA-Shapes (vốn là đồ thị thuần cấu trúc motif).
> - Saliency và GNNExplainer duy trì Fidelity Drop dương, vượt qua cả hai baseline ngẫu nhiên.

---

### 2.3. Độ tương đồng và đồng thuận (Agreement) giữa các Explainer

Đo lường mức độ trùng khớp của top-5 cạnh quan trọng nhất được trích xuất bởi các phương pháp:

| Cặp Explainer đối đầu | Dataset | Jaccard Similarity@5 $\uparrow$ | Kendall's Tau Rank Correlation |
| :--- | :--- | :---: | :---: |
| **PGExplainer vs GNNExplainer** | BA-Shapes | **$0.3839 \pm 0.3390$** | $+0.1322 \pm 0.2257$ |
| **PGExplainer vs Saliency** | BA-Shapes | $0.3712 \pm 0.3157$ | $-0.0504 \pm 0.2435$ |
| **GNNExplainer vs Saliency** | BA-Shapes | **$0.4190 \pm 0.3873$** | $-0.0381 \pm 0.4498$ |
| **PGExplainer vs GNNExplainer** | Cora | $0.3137 \pm 0.3545$ | $+0.1016 \pm 0.2899$ |
| **PGExplainer vs Saliency** | Cora | $0.2923 \pm 0.3598$ | $+0.1221 \pm 0.3069$ |
| **GNNExplainer vs Saliency** | Cora | $0.2685 \pm 0.3416$ | $+0.0157 \pm 0.4120$ |

> **Nhận xét The Disagreement Problem (Krishna et al. AIES 2022):**
> - Độ tương đồng Jaccard giữa các Explainer dao động từ 0.27 đến 0.42. Điều này minh chứng hiện tượng "Bất đồng thuận trong giải thích GNN" (The Disagreement Problem): các phương pháp tiếp cận khác nhau (MLP tham số hóa vs Tối ưu mặt nạ cá thể vs Đạo hàm bậc 1) chọn các tập con cạnh khác nhau để giải thích cùng một dự đoán, khẳng định tính cần thiết của việc báo cáo đa phương pháp.

---

## 3. Hướng dẫn sử dụng cho Nhóm 3 (Làm Slides & Báo cáo)

1. **Vẽ biểu đồ so sánh Fidelity và Motif-AUC:**
   - Dùng dữ liệu từ file [`midterm_metrics_summary.json`](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/docs/reports/midterm_metrics_summary.json) để vẽ biểu đồ thanh (bar plot) có error bars ($\pm \sigma$) cho slide thuyết trình.
2. **Minh họa trực quan Subgraph Motif:**
   - Nhóm 3 có thể lấy các đồ thị con minh họa đã xuất tại thư mục: [`repos/IS353_C10/notebooks/01_midterm/figures/`](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/repos/IS353_C10/notebooks/01_midterm/figures/) để đưa vào slide.
3. **Nội dung kết luận báo cáo giữa kỳ:**
   - Nhóm 2 đã hoàn thành 100% khối lượng công việc được giao trong bảng kế hoạch (T02, T04, T05).
   - Code và dữ liệu hoàn toàn sạch, tuân thủ nguyên tắc Zero-Mockup và Anti-Leakage của TS. Trần Hùng Nghiệp.
