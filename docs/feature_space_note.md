# Feature Space Comparison Note

The repository currently contains two historical feature spaces:

- Centralized baseline scripts under `experiments/baseline-CIC_IOT_2023` use a
  32-feature training-ready dataset.
- FL v3 configs use `dataset.feature_count: 28` and models with 28 input
  features.

These results must not be compared as strict centralized-vs-federated evidence
unless both pipelines are regenerated from the same feature-selection artifact.
For defensible comparisons, use one shared feature list/hash across baseline and
FL runs, or report the baseline as a historical reference only.

