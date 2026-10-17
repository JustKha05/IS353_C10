# SNA Group 09 - Experiment Suite

Thư mục này chứa mã nguồn thực nghiệm, cấu hình và kết quả phân tích phục vụ đồ án:
**Making Graph Machine Learning Models Explainable (Topic C.10)** - GVHD: TS. Trần Hưng Nghiệp.

---

## Danh Sách 8 Thí Nghiệm Nghiên Cứu (Research Experiments)

| Thư mục | Tên Thí Nghiệm | Mục Tiêu Khoa Học | Thành Viên Phụ Trách Chính |
|---|---|---|---|
| [`exp01_model_training`](file:///experiments/exp01_model_training) | **Exp 1: Huấn luyện GNN Baselines** | Huấn luyện GCN, GAT, GraphSAGE trên Cora, BA-Shapes... Đảm bảo test accuracy & macro-F1 chuẩn mực. | Chung (Thiện & Khôi) |
| [`exp02_sanity_check`](file:///experiments/exp02_sanity_check) | **Exp 2: Sanity Checks cho Explainers** | Kiểm thử Adebayo et al. (Model parameter randomization, edge shuffle) để xác nhận explainer không giải thích ngẫu nhiên. | Khôi |
| [`exp03_fidelity_curve`](file:///experiments/exp03_fidelity_curve) | **Exp 3: Đường cong Fidelity & Sparsity** | Đo Fidelity- (Sufficiency drop) và Fidelity+ (Preservation) theo các ngưỡng sparsity $k \in \{5\%, 10\%, 20\%, ...\}$. | Thiện |
| [`exp04_agreement`](file:///experiments/exp04_agreement) | **Exp 4: Độ đồng thuận giữa các Explainers** | Đo Jaccard@k và Kendall's $\tau$ giữa từng cặp phương pháp (Saliency vs GNNExplainer vs PGExplainer). | Khôi |
| [`exp05_stability_variance`](file:///experiments/exp05_stability_variance) | **Exp 5: Độ ổn định & Phương sai** | Đo tính ổn định của giải thích qua nhiều seed ngẫu nhiên và qua các kiến trúc khác nhau (GCN vs GAT). | Thiện |
| [`exp06_controlled_variants`](file:///experiments/exp06_controlled_variants) | **Exp 6: Đánh giá Ground-Truth Motif** | Đo ROC-AUC và PR-AUC trên BA-Shapes (house motif) để so sánh với nhãn giải thích chân lý thật. | Khôi |
| [`exp07_dataset_profiling`](file:///experiments/exp07_dataset_profiling) | **Exp 7: Tương quan thuộc tính đồ thị** | Phân tích ảnh hưởng của mật độ đồ thị, homophily, độ phân nhánh node tới độ tin cậy của explainer. | Kha |
| [`exp08_disagreement_analysis`](file:///experiments/exp08_disagreement_analysis) | **Exp 8: Phân tích Ca Bất Đồng & Case Studies** | Tìm các node mà các phương pháp giải thích mâu thuẫn nhất, trực quan hóa và giải thích nguyên nhân đồ thị. | Nhóm (Báo cáo cuối kỳ) |

---

## Nguyên Tắc Thực Nghiệm
1. **Không code cứng (hardcode):** Mỗi thí nghiệm sử dụng cấu hình YAML từ `configs/` và gọi các module tái sử dụng từ `sna.*`.
2. **Batch Runner:** Chạy giải thích hàng loạt xuất dữ liệu chuẩn hóa dạng `edge_scores.csv` và `evaluation.csv`.
3. **Artifacts:** Checkpoint và ma trận trọng số đồ thị lớn được upload lên Google Drive. Kết quả số liệu tóm tắt (`summary.json`, bảng CSV) và đồ thị vẽ được commit vào Git để làm minh chứng.
