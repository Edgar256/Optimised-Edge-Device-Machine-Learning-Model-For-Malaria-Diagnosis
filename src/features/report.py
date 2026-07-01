"""Feature engineering report generation."""

from __future__ import annotations

from typing import Any

from src.features.engineering import (
    EXCLUDED_FEATURE_DEFINITIONS,
    FEATURE_DEFINITIONS,
    FeatureEngineeringResult,
)
from src.utils.config import load_config


def render_feature_engineering_report(
    result: FeatureEngineeringResult,
    config: dict[str, Any] | None = None,
) -> str:
    """Render a markdown report documenting engineered features and medical rationale."""
    cfg = config or load_config()
    stats = result.stats
    lines = [
        "# Feature Engineering Report",
        "",
        "Triage-time clinical features derived from the cleaned processed dataset. "
        "**Post-diagnosis columns are excluded to prevent target leakage.**",
        "",
        "## 1. Overview",
        "",
        f"- **Generated (UTC):** {stats.get('generated_at_utc')}",
        f"- **Input rows:** {stats.get('input_rows')}",
        f"- **Output rows:** {stats.get('output_rows')}",
        f"- **Engineered features created:** {stats.get('engineered_feature_count')}",
        f"- **Output dataset:** `{stats.get('output_path')}`",
        "",
        "## 2. Feature catalogue",
        "",
        "| Feature | Type | Definition | Medical rationale |",
        "|---------|------|------------|-------------------|",
    ]

    for feature in FEATURE_DEFINITIONS:
        lines.append(
            f"| `{feature['name']}` | {feature['type']} | {feature['definition']} | {feature['rationale']} |"
        )

    lines.extend(
        [
            "",
            "## 3. Excluded features (leakage prevention)",
            "",
            "The following features were **not** created because they rely on post-diagnosis "
            "information or unavailable measurements:",
            "",
            "| Feature | Reason excluded |",
            "|---------|-----------------|",
        ]
    )
    for excluded in EXCLUDED_FEATURE_DEFINITIONS:
        lines.append(f"| {excluded['name']} | {excluded['reason']} |")

    post_diagnosis = cfg["data"]["post_diagnosis_columns"]
    lines.extend(
        [
            "",
            "Post-diagnosis columns blocked by configuration:",
            "",
            ", ".join(f"`{column}`" for column in post_diagnosis),
            "",
            "## 4. Feature distributions",
            "",
        ]
    )

    distributions = stats.get("distributions", {})
    for feature_name in stats.get("engineered_features", []):
        distribution = distributions.get(feature_name)
        if distribution is None:
            continue
        lines.append(f"### `{feature_name}`")
        lines.append("")
        if isinstance(distribution, dict) and "mean" in distribution:
            lines.append(
                f"- min: {distribution['min']}, max: {distribution['max']}, "
                f"mean: {distribution['mean']}, median: {distribution['median']}"
            )
        else:
            for value, count in distribution.items():
                lines.append(f"- `{value}`: {count}")
        lines.append("")

    lines.extend(
        [
            "## 5. Data quality caveats",
            "",
            "- **Chills and fatigue** are excluded from `core_symptom_count` because both exceed "
            "95% `yes` in the source data, suggesting form-default selection rather than "
            "discriminative clinical signal.",
            "- **Fever duration** may contain imputed values from the preprocessing pipeline "
            "where negative durations were corrected to missing before median imputation.",
            "- **No body-temperature or hemoglobin** measurements exist; `fever_severity` and "
            "`anemia_signs_x_pediatric` use the closest available clinical proxies.",
            "- **RDT and microscopy** are intentionally excluded; interaction features involving "
            "these tests would encode diagnostic outcomes rather than triage-time presentation.",
            "",
        ]
    )
    return "\n".join(lines)
