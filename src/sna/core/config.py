"""Cấu hình và đường dẫn hệ thống toàn cục cho đồ án."""

import pathlib
from typing import Dict, Any
import json
import yaml

# Thư mục gốc dự án
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" if (PROJECT_ROOT / "data").exists() else PROJECT_ROOT / "t01_handover"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DOCS_DIR = PROJECT_ROOT / "docs"

# Google Drive folder của nhóm
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1AXfG73xtbbCh3K1gopr-MDdEPYzaN51a?usp=sharing"


def load_yaml_config(config_path: pathlib.Path) -> Dict[str, Any]:
    """Nạp file cấu hình YAML."""
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json_config(config_path: pathlib.Path) -> Dict[str, Any]:
    """Nạp file cấu hình JSON."""
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)
