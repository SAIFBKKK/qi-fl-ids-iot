from __future__ import annotations

from feature_schema import expected_28_features


MASK_ID = "conservative_seed_42"
SELECTED_INDICES = [0, 2, 3, 4, 6, 13, 14, 19, 20, 25, 26, 27]


def selected_indices() -> list[int]:
    return list(SELECTED_INDICES)


def selected_features() -> list[str]:
    features = expected_28_features()
    return [features[index] for index in SELECTED_INDICES]


def selected_mask_id() -> str:
    return MASK_ID

