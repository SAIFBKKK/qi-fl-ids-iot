from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from ciciot_feature_mapper import FeatureMappingResult, map_flows
from feature_gap_report import write_gap_report
from feature_schema import INPUT_MODES, FeatureSchema, load_schema, schema_summary, selected_from_original
from nfstream_adapter import FlowStats, dry_run_flow_stats, extract_flows_from_pcap
from scaler_adapter import scale_original_28_rows, scaler_exists


DEFAULT_CSV_NAME = "p16_extracted_features.csv"
DEFAULT_JSON_NAME = "p16_extracted_features.json"
DEFAULT_GAP_REPORT_NAME = "p16_feature_extraction_gap_report.md"


@dataclass
class ExtractionRecord:
    flow_id: str
    input_mode: str
    feature_order: list[str]
    features: list[float | None]
    unsupported_features: list[str] = field(default_factory=list)
    approximated_features: dict[str, str] = field(default_factory=dict)
    source: str = "nfstream"

    def as_json(self) -> dict[str, object]:
        return {
            "flow_id": self.flow_id,
            "input_mode": self.input_mode,
            "feature_order": self.feature_order,
            "features": self.features,
            "unsupported_features": self.unsupported_features,
            "approximated_features": self.approximated_features,
            "source": self.source,
        }


@dataclass
class ExtractionResult:
    output_mode: str
    feature_order: list[str]
    selected_mask_id: str
    records: list[ExtractionRecord]
    gaps: list[str]
    scaler_available: bool
    scaler_used: bool
    projection_error: str | None = None

    def to_payloads(self, node_id: str = "pcap-extractor") -> list[dict[str, object]]:
        return [
            {
                "flow_id": record.flow_id,
                "node_id": node_id,
                "input_mode": record.input_mode,
                "features": record.features,
                "unsupported_features": record.unsupported_features,
                "approximated_features": record.approximated_features,
            }
            for record in self.records
        ]


class ProjectionError(ValueError):
    pass


def _unsupported_for_feature_order(mapping: FeatureMappingResult, feature_order: list[str]) -> list[str]:
    unsupported = set(mapping.unsupported_features)
    return [feature for feature in feature_order if feature in unsupported]


def project_mappings(
    mappings: list[FeatureMappingResult],
    output_mode: str,
    schema: FeatureSchema | None = None,
    scaler_path: str | Path | None = None,
) -> tuple[list[ExtractionRecord], bool]:
    feature_schema = schema or load_schema()
    if output_mode not in INPUT_MODES:
        raise ProjectionError(f"unsupported output mode: {output_mode}")
    if not mappings:
        return [], False

    original_rows = [mapping.values for mapping in mappings]
    scaler_used = False
    scaled_rows: list[list[float]] = []
    if output_mode in {"original_28_scaled", "selected_12_scaled"}:
        scaled_rows, scaler_used = scale_original_28_rows(original_rows, scaler_path)
        if not scaler_used:
            raise ProjectionError("scaled output requested but the L1 robust scaler is not available")

    records: list[ExtractionRecord] = []
    for index, mapping in enumerate(mappings):
        if output_mode == "original_28_unscaled":
            feature_order = feature_schema.feature_names
            features: list[float | None] = mapping.values
        elif output_mode == "original_28_scaled":
            feature_order = feature_schema.feature_names
            features = scaled_rows[index]
        else:
            feature_order = feature_schema.selected_feature_names
            features = selected_from_original(scaled_rows[index], feature_schema.selected_indices)

        records.append(
            ExtractionRecord(
                flow_id=mapping.flow_id,
                input_mode=output_mode,
                feature_order=feature_order,
                features=features,
                unsupported_features=_unsupported_for_feature_order(mapping, feature_order),
                approximated_features={
                    name: note for name, note in mapping.approximated_features.items() if name in set(feature_order)
                },
                source=mapping.source,
            )
        )
    return records, scaler_used


def result_from_mappings(
    mappings: list[FeatureMappingResult],
    output_mode: str,
    schema: FeatureSchema | None = None,
    scaler_path: str | Path | None = None,
    empty_message: str | None = None,
) -> ExtractionResult:
    feature_schema = schema or load_schema()
    try:
        records, scaler_used = project_mappings(mappings, output_mode, feature_schema, scaler_path)
        projection_error = None
    except ProjectionError as exc:
        records = []
        scaler_used = False
        projection_error = str(exc)
    except ValueError as exc:
        records = []
        scaler_used = False
        projection_error = str(exc)

    gaps = [
        "NFStream-first experimental prototype; exact CICIoT2023 feature equivalence is not claimed.",
        "Missing features remain unsupported and are not imputed.",
    ]
    if empty_message:
        gaps.append(empty_message)
    if projection_error:
        gaps.append(projection_error)

    return ExtractionResult(
        output_mode=output_mode,
        feature_order=records[0].feature_order if records else feature_schema.feature_names,
        selected_mask_id=feature_schema.selected_mask_id,
        records=records,
        gaps=gaps,
        scaler_available=scaler_exists(scaler_path),
        scaler_used=scaler_used,
        projection_error=projection_error,
    )


def extract_pcap_to_records(
    pcap_path: str | Path | None,
    output_mode: str = "original_28_unscaled",
    scaler_path: str | Path | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> ExtractionResult:
    schema = load_schema()
    flows: list[FlowStats]
    if dry_run:
        flows = dry_run_flow_stats()
    elif pcap_path is None:
        return ExtractionResult(
            output_mode=output_mode,
            feature_order=schema.feature_names,
            selected_mask_id=schema.selected_mask_id,
            records=[],
            gaps=["No pcap_path supplied; dry_run=False, so no flows were extracted."],
            scaler_available=scaler_exists(scaler_path),
            scaler_used=False,
        )
    else:
        flows = extract_flows_from_pcap(pcap_path, limit=limit)

    mappings = map_flows(flows, schema)
    return result_from_mappings(mappings, output_mode, schema, scaler_path)


def run_extraction(
    pcap_path: str | Path | None,
    output_mode: str,
    output_dir: str | Path,
    scaler_path: str | Path | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> ExtractionResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    schema = load_schema()
    flows = dry_run_flow_stats() if dry_run else extract_flows_from_pcap(pcap_path, limit=limit) if pcap_path else []
    mappings = map_flows(flows, schema)
    empty_message = "No pcap_path supplied; dry_run=False, so no flows were extracted." if not dry_run and not pcap_path else None
    result = result_from_mappings(mappings, output_mode, schema, scaler_path, empty_message=empty_message)
    write_csv(output_path / DEFAULT_CSV_NAME, result)
    write_json(output_path / DEFAULT_JSON_NAME, result)
    write_gap_report(
        output_path / DEFAULT_GAP_REPORT_NAME,
        mappings=mappings,
        schema=schema,
        output_mode=output_mode,
        pcap_path=str(pcap_path) if pcap_path else None,
        scaler_available=result.scaler_available,
        scaler_used=result.scaler_used,
        projection_error=result.projection_error,
    )
    return result


def write_json(path: str | Path, result: ExtractionResult) -> None:
    payload = {
        "schema_version": "p16_feature_extraction_v1",
        "output_mode": result.output_mode,
        "feature_order": result.feature_order,
        "selected_mask_id": result.selected_mask_id,
        "scaler_available": result.scaler_available,
        "scaler_used": result.scaler_used,
        "gaps": result.gaps,
        "records": [record.as_json() for record in result.records],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: str | Path, result: ExtractionResult) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["flow_id", "input_mode", "unsupported_features", "approximated_features", *result.feature_order])
        for record in result.records:
            writer.writerow(
                [
                    record.flow_id,
                    record.input_mode,
                    json.dumps(record.unsupported_features),
                    json.dumps(record.approximated_features),
                    *["" if value is None else value for value in record.features],
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NFStream-first P16 pcap-to-feature extractor prototype.")
    parser.add_argument("--pcap")
    parser.add_argument("--output-mode", choices=list(INPUT_MODES), default="original_28_unscaled")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--scaler-path")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--schema", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.schema:
        print(json.dumps(schema_summary(), indent=2))
        return 0
    result = run_extraction(
        pcap_path=args.pcap,
        output_mode=args.output_mode,
        output_dir=args.output_dir,
        scaler_path=args.scaler_path,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "records": len(result.records),
                "output_mode": result.output_mode,
                "scaler_used": result.scaler_used,
                "projection_error": result.projection_error,
                "output_dir": str(Path(args.output_dir).resolve()),
            },
            indent=2,
        )
    )
    return 0 if result.projection_error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
