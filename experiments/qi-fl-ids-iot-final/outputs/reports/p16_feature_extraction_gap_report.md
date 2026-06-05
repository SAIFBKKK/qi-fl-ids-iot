# P16 Feature Extraction Gap Report

The P16 `live_feature_extractor` is an experimental bridge toward:

`raw/pcap packets -> feature extractor -> 28 features -> scaler -> IDS`

It is inspired by CICIoT2023/Sensors, where packet captures and processed CSV
files are published separately, and CSV features are extracted from packet
windows between hosts. The paper reports DPKT-based extraction and also mentions
CICFlowMeter and NFStream as possible alternatives.

## Current Prototype Coverage

- Passive local pcap read through DPKT.
- Per-flow packet windows.
- Alignment to the 28 final L1 feature names.
- Optional robust scaler adapter when the local scaler is present.
- JSON/CSV output compatible with the final API payload shape after scaling and
  once unsupported/null values have been resolved.

## Gaps

- The exact CICIoT2023 feature extraction semantics are not reimplemented.
- Labels are not inferred from raw pcap and must come from controlled replay
  metadata.
- Application protocol indicators are inferred from common ports only.
- Header length, duration, rate, flag counters, IAT, and packet statistics are
  local derivations and must be validated before production use.
- Missing or uncomputable features are emitted as `null` or listed in
  `unsupported_features`.

## Future Work

- Compare prototype output against a known CICIoT2023 CSV slice from the same
  controlled pcap.
- Add a reproducible scaler packaging decision for deployment.
- Decide whether the future production extractor should use DPKT directly or a
  maintained flow tool such as CICFlowMeter/NFStream.
- Keep live raw capture out of P16; add it only after an isolated-network safety
  review.

