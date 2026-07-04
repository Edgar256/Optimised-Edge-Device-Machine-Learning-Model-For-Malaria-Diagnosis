"""Remove generated training artifacts before a fresh model run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.config import load_config
from src.utils.paths import resolve_path

# Preserve directory placeholders tracked by git.
KEEP_NAMES = {".gitkeep"}


@dataclass
class CleanupResult:
    """Summary of deleted paths."""

    deleted: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    dry_run: bool = False
    include_processed: bool = False


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(resolve_path(".")))
    except ValueError:
        return str(path)


def _artifact_roots(config: dict[str, Any], *, include_processed: bool) -> list[Path]:
    paths_cfg = config["paths"]
    roots = [
        resolve_path(paths_cfg["models_dir"]),
        resolve_path(paths_cfg["results_dir"]),
        resolve_path(paths_cfg["figures_dir"]),
        resolve_path(paths_cfg["reports_dir"]),
    ]
    if include_processed:
        roots.append(resolve_path(paths_cfg["processed_data_dir"]))
    return roots


def _iter_deletable(root: Path) -> list[Path]:
    """Return files and directories under root that should be removed (deepest first)."""
    if not root.is_dir():
        return []

    entries: list[Path] = []
    for path in root.rglob("*"):
        if path.name in KEEP_NAMES:
            continue
        entries.append(path)

    # Delete files first, then directories (deepest paths first).
    entries.sort(key=lambda p: (p.is_dir(), -len(p.parts)))
    return entries


def clean_training_artifacts(
    config: dict[str, Any] | None = None,
    *,
    include_processed: bool = False,
    dry_run: bool = False,
) -> CleanupResult:
    """Delete past models, results, figures, and reports before retraining.

    By default clears:

    - ``models/`` (baseline and optimized joblib artifacts)
    - ``results/`` (metrics CSVs and manifests)
    - ``figures/`` (ROC / confusion-matrix plots)
    - ``reports/`` (markdown reports, explainability, Chapter 4 tables)

    Does **not** touch ``data/raw/``. Pass ``include_processed=True`` to also
    clear ``data/processed/`` (pipeline and processed CSVs).

    ``.gitkeep`` files are preserved so directory structure remains in git.
    """
    cfg = config or load_config()
    result = CleanupResult(dry_run=dry_run, include_processed=include_processed)

    for root in _artifact_roots(cfg, include_processed=include_processed):
        if not root.exists():
            result.skipped.append(_display_path(root))
            continue

        for path in _iter_deletable(root):
            display = _display_path(path)
            if dry_run:
                result.deleted.append(display)
                continue
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
                result.deleted.append(display)
            elif path.is_dir():
                # Only remove empty dirs (children already deleted).
                try:
                    path.rmdir()
                    result.deleted.append(display)
                except OSError:
                    result.skipped.append(display)

    return result
