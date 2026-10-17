"""Script hướng dẫn tải các asset nặng (weights, data) từ Google Drive của Nhóm 09.
Tránh commit các file nhị phân lớn (.pt, .ckpt) lên GitHub.
"""

import pathlib
import sys

DRIVE_LINK = "https://drive.google.com/drive/folders/1AXfG73xtbbCh3K1gopr-MDdEPYzaN51a?usp=sharing"

ASSET_MAP = {
    "bashapes": [
        "data.pt",
        "gcn_ckpt.pt",
        "config.json",
        "val_nodes.json"
    ],
    "cora": [
        "data.pt",
        "gcn_ckpt.pt",
        "config.json",
        "val_nodes.json"
    ]
}


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    target_dir = root / "data" if (root / "data").exists() else root / "t01_handover"

    print("=" * 60)
    print("SNA Nhóm 09 - Hướng dẫn đồng bộ dữ liệu từ Google Drive")
    print("=" * 60)
    print(f"\nThư mục Google Drive của nhóm:\n  👉 {DRIVE_LINK}\n")
    print(f"Thư mục lưu trữ đích trên máy local:\n  👉 {target_dir}\n")

    missing = []
    for ds, files in ASSET_MAP.items():
        for f in files:
            p = target_dir / ds / f
            if not p.exists():
                missing.append(f"{ds}/{f}")

    if not missing:
        print("✅ Toàn bộ các file dữ liệu và checkpoint đã có sẵn trên máy của bạn!")
    else:
        print("⚠️ Các file sau đang thiếu, vui lòng tải từ Drive và đặt vào đúng thư mục:")
        for m in missing:
            print(f"  - t01_handover/{m}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
