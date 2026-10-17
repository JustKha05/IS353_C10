# GNN Explainability Benchmark & Analysis (SNA - UIT 2026)
## Nhóm 09 – Đề tài C.10: Making Graph Machine Learning Models Explainable

**Giảng viên hướng dẫn:** TS. Trần Hưng Nghiệp (`nghiepth@uit.edu.vn`)  
**Thành viên:** Nguyễn Hải Thiện, Nguyễn Trọng Khôi, Nguyễn Lê Bảo Đăng, Nguyễn Trường Vũ, Vòng Việt Đức, Nguyễn Hoàng Kha.

---

## 1. Lộ trình Nghiên cứu Tổng thể (8 Tuần & 8 Thí nghiệm)

Đồ án được thiết kế dài hạn theo 2 giai đoạn:

| Giai đoạn | Thời gian | Nhiệm vụ & Mục tiêu cốt lõi | Output chính |
| :--- | :--- | :--- | :--- |
| **Giai đoạn 1: Prototype Giữa kỳ** | 16/10 – 21/10 | Xây dựng pipeline cơ sở (T01–T09): Fix GCN BA-Shapes, 3 Explainers (GNNExplainer, PGExplainer, Saliency), 2 Baselines (Random, Nearest-first), Fidelity, Agreement. | Báo cáo & Slide Giữa kỳ, Demo Notebook. |
| **Giai đoạn 2: Khảo sát Toàn diện** | Tuần 2 – Tuần 7 | Thực hiện **Exp 1 đến Exp 8**: Huấn luyện GCN/GAT 5 seeds, Sanity check motif, đường cong Fidelity, ma trận Agreement, phân rã phương sai (seed GNN vs seed explainer), mở rộng đồ thị thực tế (Amazon Photo, ogbn-arxiv,...). | Bảng số liệu benchmark toàn diện. |
| **Giai đoạn 3: Báo cáo Cuối kỳ** | Tuần 7 – Tuần 8 | Đánh giá test set cho các phát hiện đã khóa, thử nghiệm cải tiến có điều kiện (**Exp 9**), hoàn thiện Báo cáo kỹ thuật (~15 trang) và bảo vệ cuối kỳ. | Technical Report, Slide, Repository. |

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
│       │   └── checkpoint.py           # Quản lý checkpoint và khóa model.eval()
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
│   ├── 01_midterm/                     # Minh chứng giai đoạn Giữa kỳ
│   │   ├── 01_fix_bashapes_gcn.ipynb   # Bước 1 của Thiện: Fix Class 3 BA-Shapes
│   │   ├── 02_demo_explainers.ipynb    # Bước 2 của Thiện & Khôi: Chạy 3 Explainers
│   │   └── sample_edge_scores.csv      # File kết quả mẫu chuẩn BG10
│   └── 02_final_experiments/           # Minh chứng giai đoạn Cuối kỳ
│
├── scripts/                            # Scripts tiện ích cho nhóm
│   └── download_drive_assets.py        # Hướng dẫn đồng bộ checkpoint nặng từ Google Drive
│
├── docs/                               # Proposal v2 & Excel theo dõi tiến độ
│   ├── SNA-Group09-C10-Mini-Proposal-v2.docx
│   └── SNA G09 – Kế hoạch baseline và báo cáo giữa kỳ.xlsx
│
├── tests/                              # Unit tests & Integration tests
│   ├── test_explainers.py              # Test nhanh module explainers
│   └── test_package.py                 # Test toàn diện package `sna`
│
└── data/                               # Dữ liệu cục bộ (chỉ chứa metadata/json, *.pt lưu trên Drive)
    ├── bashapes/
    └── cora/
```

---

## 3. Chiến lược Lưu trữ: GitHub vs Google Drive

* **Đưa lên GitHub:**
  * Toàn bộ mã nguồn `src/`, `configs/`, `experiments/`, `scripts/`, `docs/`, `tests/`.
  * Toàn bộ **Jupyter Notebooks trong `notebooks/`** (đã chạy sẵn kết quả, số liệu và biểu đồ để làm **minh chứng kết quả** vĩnh viễn trên repo).
  * File `.gitignore` tự động loại bỏ các file nhị phân lớn (`*.pt`, `*.ckpt`) để GitHub luôn nhẹ và đồng bộ nhanh.
* **Lưu trên Google Drive (Các file nặng):**
  * Thư mục Drive nhóm: [Google Drive Nhóm 09](https://drive.google.com/drive/folders/1AXfG73xtbbCh3K1gopr-MDdEPYzaN51a?usp=sharing).
  * Chứa các file weights lớn: `data.pt`, `gcn_ckpt.pt`.
  * Khi clone repo về, thành viên chỉ cần chạy:
    ```powershell
    python scripts/download_drive_assets.py
    ```

---

## 4. Hướng dẫn Cài đặt & Sử dụng (Quickstart)

### Kích hoạt môi trường Conda
```powershell
conda activate sna_g09
```

### Cài đặt package ở chế độ Editable (`-e`)
```powershell
cd repo
pip install -e .
```

### Chạy kiểm thử toàn diện
```powershell
python tests/test_package.py
```
*Kết quả:* Kiểm thử thành công 100% Data Loader, Models, 5 Explainers/Baselines, Metrics (Fidelity/Jaccard), và Batch Pipeline trên GPU CUDA.
