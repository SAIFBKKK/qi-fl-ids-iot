# P16.11 Controlled Scenario Observation Design

This directory defines how future controlled Kali scenario observations will be named, mapped to PacketWindow(30) features, and presented as live deployment evidence.

P16.11 does not activate scenarios. It provides design artifacts only.

## Contents

- `scenario_observation_catalog.yaml`: scenario labels, expected observations, MQTT topics, dashboard evidence, and design-only status.
- `feature_expectations.yaml`: expected live-lab feature tendencies for each selected scenario.
- `observation_evidence_template.md`: future evidence template for a controlled observation.
- `safe_observation_scope.md`: safety boundary for P16.11.
- `docs/p16_11_controlled_scenario_observation_design.md`: full design narrative.
- `docs/scenario_to_feature_mapping.md`: scenario-to-feature mapping table.
- `docs/dashboard_observation_flow.md`: how `/demo` should present future evidence.
- `docs/future_activation_boundaries.md`: rules before any future activation.

## Selected Scenarios

- `icmp_flood_like`
- `tcp_syn_recon_like`
- `http_slow_like`

The selected names are labels for future controlled observation design. They are not executed in P16.11.
