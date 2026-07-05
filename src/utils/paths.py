"""Filesystem path resolution relative to the project root."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward until a directory containing main.py and config/ is found."""
    anchors: list[Path] = []
    if start is not None:
        anchors.append(start.resolve())
    anchors.append(Path.cwd().resolve())
    # Fallback when the process cwd is not the repo root (e.g. some PaaS layouts).
    anchors.append(Path(__file__).resolve())

    seen: set[Path] = set()
    for anchor in anchors:
        for candidate in [anchor, *anchor.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "main.py").is_file() and (candidate / "config").is_dir():
                return candidate
    raise FileNotFoundError(
        "Could not locate project root (expected main.py and config/ in the same directory)."
    )


def resolve_path(relative: str | Path, project_root: Path | None = None) -> Path:
    """Return an absolute path for *relative* under the project root."""
    root = project_root or find_project_root()
    return (root / relative).resolve()
