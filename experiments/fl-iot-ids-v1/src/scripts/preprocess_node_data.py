from __future__ import annotations

import argparse
from pathlib import Path

from src.common.logger import get_logger
from src.common.paths import ARTIFACTS_DIR, DATA_DIR
from src.data.preprocessor import BaselinePreprocessor


logger = get_logger("preprocess_node_data")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess one node local dataset using baseline artifacts."
    )
    parser.add_argument(
        "--node-id",
        type=str,
        required=True,
        help="Node identifier, e.g. node1, node2, node3",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Optional explicit input CSV path. Default: data/raw/<node-id>/train.csv",
    )
    parser.add_argument(
        "--output-npz",
        type=str,
        default=None,
        help="Optional explicit output NPZ path. Default: data/processed/<node-id>/train_preprocessed.npz",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=str(ARTIFACTS_DIR),
        help="Artifacts directory path",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    node_id = args.node_id

    input_csv = (
        Path(args.input_csv)
        if args.input_csv
        else DATA_DIR / "raw" / node_id / "train.csv"
    )

    output_npz = (
        Path(args.output_npz)
        if args.output_npz
        else DATA_DIR / "processed" / node_id / "train_preprocessed.npz"
    )

    artifacts_dir = Path(args.artifacts_dir)

    logger.info("Starting preprocessing for node: %s", node_id)
    logger.info("Input CSV: %s", input_csv)
    logger.info("Output NPZ: %s", output_npz)
    logger.info("Artifacts dir: %s", artifacts_dir)

    preprocessor = BaselinePreprocessor(artifacts_dir=artifacts_dir)
    preprocessor.load_artifacts()

    X, y, feature_names = preprocessor.process_csv(input_csv)
    preprocessor.save_npz(output_npz, X, y, feature_names)

    logger.info("Preprocessing completed successfully for %s", node_id)


if __name__ == "__main__":
    main()