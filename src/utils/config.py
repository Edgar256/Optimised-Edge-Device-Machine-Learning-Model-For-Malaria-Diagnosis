"""Configuration loading and typed access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import find_project_root, resolve_path

DEFAULT_CONFIG = "config/default.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML configuration file and return it as a dictionary."""
    path = resolve_path(config_path or DEFAULT_CONFIG)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping, got {type(config).__name__}")

    return config


def get_raw_data_path(config: dict[str, Any] | None = None) -> Path:
    """Return the absolute path to the primary raw CSV export."""
    cfg = config or load_config()
    data_cfg = cfg["data"]
    paths_cfg = cfg["paths"]
    return resolve_path(Path(paths_cfg["raw_data_dir"]) / data_cfg["raw_filename"])
