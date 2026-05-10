# P8.1.5 QGA Mask Selection Decision

## Selected Mask

- Mask ID: `conservative_seed_42`
- Profile: `conservative`
- Seed: `42`
- Features: `12`
- Engineering score: `0.3468113475231663`

## Rationale

The selected mask maximizes the engineering score over validation-only true-Flower short runs.
The global test holdout was not used for mask selection.

## Filtering

- Required rounds: `5`
- Required runtime: `true_flower_runtime=true`
- Required test handling: `test_sent_to_clients=false`
- Required scenario count per mask: `>= 3`

## Warnings

- Ignored stale short runs: `1`
- Ignored incomplete masks: `0`

## Output

- `outputs/qga_feature_selection/final_selected_mask/feature_mask.json`
- `outputs/qga_feature_selection/final_selected_mask/selected_features.json`
- `outputs/qga_feature_selection/final_selected_mask/selection_decision.json`
