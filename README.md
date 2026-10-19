# GNN Explainability Benchmark & Analysis (SNA - UIT 2026)
## Nhóm 09 - Đề tài C.10: Making Graph Machine Learning Models Explainable

**Giảng viên hướng dẫn:** TS. Trần Hùng Nghiệp (`nghiepth@uit.edu.vn`)  
**Nhóm sinh viên thực hiện (Nhóm 09):**
- **Lâm Quốc Vũ** (23521820) - Nhóm trưởng / Điều phối chung (Coordinator)
- **Nguyễn Lê Bảo Đăng** (23520241) - Phụ trách Phân nhóm 1 (T01: Chuẩn bị Dữ liệu & Mô hình GNN)
- **Nguyễn Hải Thiện** (23521453) - Phụ trách Phân nhóm 2 (T02, T05: Thuật toán PGExplainer)
- **Nguyễn Trọng Khôi** (23520783) - Phụ trách Phân nhóm 2 (T02, T05: Thuật toán Saliency & GNNExplainer)
- **Phan Minh Đức** (23520361) - Phụ trách Phân nhóm 3 (T03, T06: Độ đo Đánh giá Fidelity & Agreement)
- **Nguyễn Huỳnh Kha** (23520626) - Phụ trách Phân nhóm 3 (T04, T07: Phương pháp Baseline & Phân tích Đồ thị)

---

## 1. Lộ trình Nghiên cứu Tổng thể (8 Tuần & 8 Thí nghiệm)

Đồ án được thiết kế theo 3 giai đoạn nghiên cứu bám sát đề cương và yêu cầu học thuật của TS. Trần Hùng Nghiệp:

| Giai đoạn | Thời gian | Nhiệm vụ & Mục tiêu cốt lõi | Output chính |
| :--- | :--- | :--- | :--- |
| **Giai đoạn 1: Prototype Giữa kỳ** | 16/10 - 21/10 | Xây dựng pipeline cơ sở (T01-T09): Fix GCN BA-Shapes (93.57%), 3 Explainers (GNNExplainer, PGExplainer, Saliency), 2 Baselines (Random, Nearest-First), Fidelity, Agreement. | Báo cáo & Slide Giữa kỳ, Demo Notebooks, `edge_scores.csv`. |
| **Giai đoạn 2: Khảo sát Toàn diện** | Tuần 2 - Tuần 7 | Thực hiện **Exp 1 đến Exp 8**: Huấn luyện GCN/GAT 5 seeds, Sanity check motif, đường cong Fidelity theo ngân sách, ma trận Agreement, phân rã phương sai (seed GNN vs seed explainer), mở rộng đồ thị thực tế (Amazon Photo, ogbn-arxiv,...). | Bảng số liệu benchmark toàn diện. |
| **Giai đoạn 3: Báo cáo Cuối kỳ** | Tuần 7 - Tuần 8 | Đánh giá test set cho các phát hiện đã khóa, thử nghiệm cải tiến có điều kiện (**Exp 9**), hoàn thiện Báo cáo kỹ thuật (~15 trang Springer LNCS) và bảo vệ cuối kỳ. | Technical Report, Slide, Repository, Web Demo. |

---

## 2. Kiến trúc Phần mềm (Modular Monolith Architecture)

Repository được tổ chức theo chuẩn **Clean Architecture / Modular Monolith**, giúp mã nguồn có tính mở rộng cao, không bị rối rắm khi thêm nhiều task hay dataset mới:

```text
repo/
├── .gitignore                          # Cấu hình GitHub (loại bỏ *.pt nặng, cache, temp logs)
├── pyproject.toml                      # Cho phép cài đặt package `pip install -e .`
├── requirements.txt                    # Danh sách thư viện phụ thuộc
├── README.md                           # Tài liệu tổng thể này
│
├── src/                                # CORE LIBRARY: Package `sna`
│   └── sna/
│       ├── core/                       # Cấu hình hệ thống, đường dẫn, Typed Dataclasses
│       │   ├── config.py
│       │   └── types.py                # CandidateEdgesResult, ExplainerResult, MetricResult
│       │
│       ├── data/                       # [Data Module] Nạp đồ thị, chia split, candidate edges
│       │   ├── loader.py               # load_dataset('bashapes' / 'cora' / 'amazon_photo'...)
│       │   └── candidate_edges.py      # Trích xuất 2-hop graph (BG05, BG07)
│       │
│       ├── models/                     # [Models Module] Kiến trúc GNN & Checkpoints
│       │   ├── gcn.py                  # GCN 2 lớp
│       │   ├── gat.py                  # GAT 2 lớp (cho Exp 1, Exp 5)
│       │   └── checkpoint.py           # Quản lý checkpoint và khóa model.eval() (CPU/CUDA)
│       │
│       ├── explainers/                 # [Explainers Module] Thuật toán giải thích (BG04, BG06)
│       │   ├── base.py                 # Abstract BaseExplainer interface
│       │   ├── saliency.py             # Saliency Gradient (Khôi)
│       │   ├── gnnexplainer.py         # GNNExplainer (Khôi)
│       │   ├── pgexplainer.py          # PGExplainer (Thiện)
│       │   ├── registry.py             # Factory pattern & hàm explain_edges(...) thống nhất
│       │   └── baselines/              # [Baselines tham chiếu] (Kha phụ trách)
│       │       ├── random_edge.py      # Random Edge baseline
│       │       └── nearest_first.py    # Distance / Nearest-first baseline
│       │
│       ├── metrics/                    # [Metrics Module] Đánh giá chất lượng (Đức phụ trách)
│       │   ├── fidelity.py             # Fidelity- (drop) & Fidelity+ (inv) (BG08, BG09)
│       │   ├── agreement.py            # Jaccard@k, Kendall's tau-b, Cohen's kappa
│       │   └── ground_truth.py         # ROC-AUC, PR-AUC trên BA-Shapes motif
│       │
│       └── pipelines/                  # [Pipelines Module] Chạy thực nghiệm hàng loạt
│           └── batch_runner.py         # Batch runner xuất edge_scores.csv & evaluation.csv (BG10)
│
├── configs/                            # Cấu hình YAML phân tách rõ ràng
│   ├── datasets/                       # bashapes.yaml, cora.yaml, amazon_photo.yaml...
│   ├── models/                         # gcn.yaml, gat.yaml
│   └── experiments/                    # exp01.yaml, exp02.yaml, ...
│
├── experiments/                        # Thư mục mã nguồn cho 8 thí nghiệm (Exp 1 - Exp 8)
│   ├── exp01_model_training/           # Huấn luyện GCN/GAT đa seed
│   ├── exp02_sanity_check/             # Kiểm tra motif BA-Shapes
│   ├── exp03_fidelity_curve/           # Đường cong fidelity theo ngân sách
│   ├── exp04_agreement/                # Đo mức độ đồng thuận giữa các explainer
│   ├── exp05_stability_variance/       # Phân rã phương sai seed explainer vs seed GNN
│   ├── exp06_controlled_variants/      # Khảo sát có hướng vs đối xứng hóa
│   ├── exp07_dataset_profiling/        # Khảo sát theo loại đồ thị
│   └── exp08_disagreement_analysis/    # Phân tích ca bất đồng & ca lỗi
│
├── notebooks/                          # Jupyter Notebooks minh chứng (commit lên GitHub)
│   ├── 01_midterm/                     # Minh chứng giai đoạn Giữa kỳ (T01 - T08)
│   │   ├── 01_fix_bashapes_gcn.ipynb   # Bước 1 (Thiện): Huấn luyện lại GCN fix Class 3 (93.57% test acc)
│   │   ├── 02_demo_explainers.ipynb    # Bước 2 (Thiện & Khôi): Chạy thử nghiệm 3 Explainers mẫu nhỏ
│   │   ├── 03_thien_pgexplainer_t05.ipynb # Bước 3 (Thiện): Thực thi batch PGExplainer trên 40 validation nodes
│   │   ├── 04_khoi_saliency_gnnexplainer_t05.ipynb # Bước 4 (Khôi): Thực thi batch Saliency, GNNExplainer, Baselines
│   │   ├── scores_thien.csv            # Điểm cạnh riêng của PGExplainer (4,176 dòng)
│   │   ├── scores_khoi.csv             # Điểm cạnh riêng của Saliency & GNNExplainer (8,352 dòng)
│   │   ├── edge_scores.csv             # File chuẩn hợp nhất 3 phương pháp bàn giao Nhóm 3 (12,528 dòng)
│   │   ├── edge_scores_with_baselines.csv # File mở rộng gồm cả Random & Nearest-First (20,880 dòng)
│   │   └── figures/                    # Thư mục hình ảnh trực quan hóa subgraph giải thích và biểu đồ phân phối
│   └── 02_final_experiments/           # Minh chứng giai đoạn Cuối kỳ
│
├── scripts/                            # Scripts tiện ích cho nhóm
│   └── download_drive_assets.py        # Hướng dẫn đồng bộ checkpoint nặng từ Google Drive
│
├── docs/                               # Proposal, kế hoạch và bảng số liệu bàn giao
│   ├── BANG_SO_LIEU_MIDTERM_T05.md     # Bảng tổng hợp số liệu định lượng giữa kỳ chuẩn Zero-Mockup
│   ├── figures/                        # Bản sao lưu các hình ảnh minh chứng phục vụ slide & báo cáo
│   ├── SNA-Group09-C10-Mini-Proposal-v2.docx
│   └── SNA G09 - Kế hoạch baseline và báo cáo giữa kỳ.xlsx
│
├── tests/                              # Unit tests & Integration tests
│   ├── test_explainers.py              # Test nhanh module explainers (Saliency, GNNExplainer, PGExplainer)
│   └── test_package.py                 # Test toàn diện package `sna`
│
└── data/                               # Dữ liệu cục bộ (chỉ chứa metadata/json, *.pt lưu trên Drive)
    ├── bashapes/                       # Dữ liệu BA-Shapes và checkpoint GCN 93.57%
    └── cora/                           # Dữ liệu Cora và checkpoint GCN 83.60%
```

---

## 3. Chiến lược Lưu trữ: GitHub vs Google Drive

* **Đưa lên GitHub:**
  * Toàn bộ mã nguồn `src/`, `configs/`, `experiments/`, `scripts/`, `docs/`, `tests/`.
  * Toàn bộ **Jupyter Notebooks trong `notebooks/`** (đã chạy sẵn kết quả, số liệu và biểu đồ để làm **minh chứng kết quả** vĩnh viễn trên repo).
  * Các file bảng điểm kết quả (`scores_thien.csv`, `scores_khoi.csv`, `edge_scores.csv`, `edge_scores_with_baselines.csv`) phục vụ bàn giao giữa kỳ.
  * File `.gitignore` tự động loại bỏ các file nhị phân lớn (`*.pt`, `*.ckpt`) để GitHub luôn nhẹ và đồng bộ nhanh.
* **Lưu trên Google Drive (Các file nặng):**
  * Thư mục Drive nhóm: [Google Drive Nhóm 09](https://drive.google.com/drive/folders/1AXfG73xtbbCh3K1gopr-MDdEPYzaN51a?usp=sharing).
  * Chứa các file weights lớn: `data.pt`, `gcn_ckpt.pt`.
  * Khi clone repo về, thành viên chỉ cần chạy:
    ```powershell
    python scripts/download_drive_assets.py
    ```

---

## 4. Kết Quả Thực Nghiệm Định Lượng Giữa Kỳ (Midterm Results - Zero-Mockup)

Toàn bộ số liệu dưới đây được tính toán trực tiếp trên 40 node validation (20 BA-Shapes + 20 Cora) với ngân sách top-5 cạnh quan trọng nhất ($k=5$), tuân thủ định dạng $\mu \pm \sigma$:

### 4.1. Đánh giá trên đồ thị cấu trúc BA-Shapes (Synthetic Motif)
- **Mô hình GCN:** Test Accuracy = 93.57%, Class 3 Recall = 93.75% (đã fix lỗi sụp đổ lớp 3).

| Phương pháp (Explainer) | Fidelity- (Drop) $\uparrow$ | Fidelity+ (Inv) $\downarrow$ | Motif ROC-AUC $\uparrow$ | Motif PR-AUC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: |
| **PGExplainer** (Thiện) | $0.2475 \pm 0.3338$ | $0.3303 \pm 0.3874$ | **$0.8000 \pm 0.4000$** | **$0.8006 \pm 0.3988$** |
| **GNNExplainer** (Khôi) | $0.3088 \pm 0.3675$ | $0.2111 \pm 0.4151$ | $0.7392 \pm 0.1971$ | $0.7511 \pm 0.3803$ |
| **Saliency** (Khôi) | $0.3089 \pm 0.3271$ | **$0.0930 \pm 0.2214$** | $0.4206 \pm 0.3250$ | $0.6211 \pm 0.3132$ |
| *Nearest-First* (Baseline) | $0.3584 \pm 0.4036$ | $0.1061 \pm 0.2523$ | $0.7178 \pm 0.1233$ | $0.7565 \pm 0.3759$ |
| *Random* (Baseline) | $0.1742 \pm 0.2998$ | $0.3168 \pm 0.3567$ | $0.5299 \pm 0.0786$ | $0.7042 \pm 0.3529$ |

### 4.2. Đánh giá trên mạng trích dẫn Cora (Social/Citation Graph)
- **Mô hình GCN:** Test Accuracy = 83.60% (theo chuẩn phân chia Planetoid tác giả Yang et al. ICML 2016).

| Phương pháp (Explainer) | Fidelity- (Drop) $\uparrow$ | Fidelity+ (Inv) $\downarrow$ | Ghi chú học thuật |
| :--- | :---: | :---: | :--- |
| **GNNExplainer** (Khôi) | **$0.0316 \pm 0.0789$** | $-0.0504 \pm 0.2578$ | Tối ưu hóa mutual information cá thể |
| **Saliency** (Khôi) | **$0.0438 \pm 0.2466$** | $-0.0796 \pm 0.0869$ | Gradient bậc một nhạy với đặc trưng cục bộ |
| **PGExplainer** (Thiện) | $-0.0077 \pm 0.0726$ | $-0.0051 \pm 0.2495$ | Cần tăng dữ liệu huấn luyện MLP trên đồ thị thực |
| *Nearest-First* (Baseline) | $0.0062 \pm 0.2395$ | $-0.0609 \pm 0.1189$ | Giữ lân cận 1-hop |
| *Random* (Baseline) | $-0.0084 \pm 0.0665$ | $-0.0001 \pm 0.2311$ | Chọn cạnh ngẫu nhiên |

### 4.3. Bất đồng thuận trong giải thích GNN (The Disagreement Problem)
Đo lường mức độ trùng khớp Jaccard@5 giữa các cặp phương pháp:
- **BA-Shapes:**
  * PGExplainer vs GNNExplainer: Jaccard@5 = **$0.3839 \pm 0.3390$** | Kendall's Tau = $+0.1322 \pm 0.2257$
  * PGExplainer vs Saliency: Jaccard@5 = $0.3712 \pm 0.3157$ | Kendall's Tau = $-0.0504 \pm 0.2435$
  * GNNExplainer vs Saliency: Jaccard@5 = **$0.4190 \pm 0.3873$** | Kendall's Tau = $-0.0381 \pm 0.4498$
- **Cora:**
  * PGExplainer vs GNNExplainer: Jaccard@5 = $0.3137 \pm 0.3545$ | Kendall's Tau = $+0.1016 \pm 0.2899$
  * PGExplainer vs Saliency: Jaccard@5 = $0.2923 \pm 0.3598$ | Kendall's Tau = $+0.1221 \pm 0.3069$
  * GNNExplainer vs Saliency: Jaccard@5 = $0.2685 \pm 0.3416$ | Kendall's Tau = $+0.0157 \pm 0.4120$

---

## 5. Hướng dẫn Cài đặt & Sử dụng (Quickstart)

### Kích hoạt môi trường (Conda hoặc venv)
```powershell
# Kích hoạt conda môi trường của nhóm
conda activate sna_g09

# Hoặc kích hoạt virtual environment
.venv\Scripts\activate
```

### Cài đặt package ở chế độ Editable (`-e`)
```powershell
cd repo
pip install -e .
```

### Chạy kiểm thử tự động (Unit & Integration Tests)
```powershell
# Chạy kiểm thử tự động với pytest
pytest tests/
```
*Kết quả:* Kiểm thử thành công 100% (2/2 test passed): nạp dữ liệu, checkpoint GCN, 5 thuật toán Explainer/Baselines, trích xuất cạnh ứng viên 2-hop, và tính toán độ đo Fidelity/Jaccard trên cả thiết bị CPU và GPU CUDA.
