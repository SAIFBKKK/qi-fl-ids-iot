from __future__ import annotations

from collections import Counter
from pathlib import Path

from ciciot_feature_mapper import FeatureMappingResult
from feature_schema import FeatureSchema


def build_gap_report(
    mappings: list[FeatureMappingResult],
    schema: FeatureSchema,
    output_mode: str,
    pcap_path: str | None,
    scaler_available: bool,
    scaler_used: bool,
    projection_error: str | None = None,
) -> str:
    unsupported_counter: Counter[str] = Counter()
    approximated_counter: Counter[str] = Counter()
    for mapping in mappings:
        unsupported_counter.update(mapping.unsupported_features)
        approximated_counter.update(mapping.approximated_features)

    lines = [
        "# P16 Feature Extraction Gap Report",
        "",
        "This report documents the safe CICIoT2023-inspired feature extraction prototype.",
        "The prototype uses NFStream first because it is Python-native and easier to integrate.",
        "CICIoT2023 used pcap files as raw capture input and CSV files as extracted packet-window features; DPKT was used in that workflow, with CICFlowMeter and NFStream noted as alternative flow tools.",
        "",
        "## Safety Scope",
        "",
        "- Reads only local pcap files when provided.",
        "- Performs passive feature extraction and controlled dry-run validation.",
        "- Does not generate traffic and does not reproduce real attacks.",
        "- Does not use test holdout data for training or tuning.",
        "",
        "## Run Summary",
        "",
        f"- pcap_path: `{pcap_path or 'not provided'}`",
        f"- output_mode: `{output_mode}`",
        f"- mapped_flows: `{len(mappings)}`",
        f"- feature_order_count: `{len(schema.feature_names)}`",
        f"- selected_mask_id: `{schema.selected_mask_id}`",
        f"- selected_feature_count: `{len(schema.selected_indices)}`",
        f"- scaler_available: `{scaler_available}`",
        f"- scaler_used: `{scaler_used}`",
    ]
    if projection_error:
        lines.append(f"- projection_error: `{projection_error}`")

    lines.extend(["", "## Unsupported Features", ""])
    if unsupported_counter:
        for feature, count in sorted(unsupported_counter.items()):
            lines.append(f"- `{feature}`: unsupported in `{count}` flow(s)")
    else:
        lines.append("- None for the processed flows.")

    lines.extend(["", "## Approximated Features", ""])
    if approximated_counter:
        for feature, count in sorted(approximated_counter.items()):
            examples = [mapping.approximated_features[feature] for mapping in mappings if feature in mapping.approximated_features]
            note = examples[0] if examples else "approximated"
            lines.append(f"- `{feature}`: `{count}` flow(s). {note}")
    else:
        lines.append("- None for the processed flows.")

    lines.extend(
        [
            "",
            "## Known Gaps",
            "",
            "- Exact CICIoT2023 feature semantics are not claimed.",
            "- NFStream exposes flow statistics, but not every CICIoT-style feature is guaranteed by every pcap/source.",
            "- Scaled modes require all 28 original features to be present because the final robust scaler was fitted on the 28-feature order.",
            "- Missing features remain unsupported; they are not filled with synthetic values.",
            "",
        ]
    )
    return "\n".join(lines)


def write_gap_report(
    path: str | Path,
    mappings: list[FeatureMappingResult],
    schema: FeatureSchema,
    output_mode: str,
    pcap_path: str | None,
    scaler_available: bool,
    scaler_used: bool,
    projection_error: str | None = None,
) -> None:
    report = build_gap_report(
        mappings=mappings,
        schema=schema,
        output_mode=output_mode,
        pcap_path=pcap_path,
        scaler_available=scaler_available,
        scaler_used=scaler_used,
        projection_error=projection_error,
    )
    Path(path).write_text(report, encoding="utf-8")

