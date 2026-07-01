"""Filesystem path resolution relative to the project root."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from *start* until a directory containing main.py is found."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "main.py").is_file() and (candidate / "config").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate project root (expected main.py and config/ in the same directory)."
    )


def resolve_path(relative: str | Path, project_root: Path | None = None) -> Path:
    """Return an absolute path for *relative* under the project root."""
    root = project_root or find_project_root()
    return (root / relative).resolve()
