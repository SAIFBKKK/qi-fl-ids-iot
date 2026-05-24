# P16 Experimental Feature Extraction Prototype

Conceptual path:

`local controlled pcap -> passive packet read -> flow/window grouping -> 28 feature alignment -> optional scaler -> final IDS payload`

This prototype follows the CICIoT2023 processing idea at a safe lab level: it
reads local pcap files passively and groups packets into fixed-size windows. It
does not capture live traffic and does not generate traffic.

The final API expects scaled features. Use `--scale` only when the local robust
scaler is present and compatible:

```bash
python pcap_to_features.py --pcap sample.pcap --input-mode original_28_scaled --scale --output-json flows.json
```

The extractor is experimental. Some CICIoT2023 feature semantics depend on the
original extraction scripts and labeling context, so the prototype emits
`unsupported_features` and `approximations` rather than silently claiming full
equivalence.

