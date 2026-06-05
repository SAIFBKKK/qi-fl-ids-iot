# P16 NFStream-First Feature Extraction Prototype

Safe CICIoT2023-inspired data-processing path:

`local pcap -> NFStream flow statistics -> 28-feature mapping -> optional robust scaler -> QGA mask -> IDS payload`

The prototype follows only the safe processing idea: pcap is a raw local capture
format, and CSV/JSON rows are extracted feature representations. It does not
generate traffic and does not reproduce real attacks.

## Output Modes

- `original_28_unscaled`: preserve the exact 28-feature order from
  `outputs/artifacts/features/feature_names.json`.
- `original_28_scaled`: apply the local L1 robust scaler if available.
- `selected_12_scaled`: scale the 28-feature row, then apply the final QGA mask
  `conservative_seed_42`.

Scaled modes require all 28 original features to be present. Missing values are
not imputed.

## Usage

```bash
python pcap_to_features.py --pcap sample.pcap --output-mode original_28_unscaled --output-dir ./p16_extract
python pcap_to_features.py --pcap sample.pcap --output-mode selected_12_scaled --output-dir ./p16_extract
```

Outputs:

- `p16_extracted_features.csv`
- `p16_extracted_features.json`
- `p16_feature_extraction_gap_report.md`

Dry run without a pcap:

```bash
python pcap_to_features.py --dry-run --output-mode selected_12_scaled --output-dir ./p16_extract_dry
```

## Gap Policy

The mapper never silently invents missing features. Each row carries
`unsupported_features` and `approximated_features`; the markdown gap report
summarizes exactly which features were unsupported or approximated. NFStream is
used first because it is Python-native. CICIoT2023 used DPKT in its pipeline and
mentions CICFlowMeter and NFStream as alternative tools.
