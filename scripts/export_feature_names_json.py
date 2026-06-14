"""
Export canonical feature_names.json from Model Factory bundles.
Verifies all 4 feature_names.pkl are strictly identical before writing.
"""
import difflib
import json
import pickle
import sys
import warnings
from pathlib import Path

BASE = Path(__file__).parent.parent / "experiments/fl-iot-ids-v3/outputs/model_factory_30rounds"

SOURCES = {
    "weak": BASE / "weak" / "feature_names.pkl",
    "medium": BASE / "medium" / "feature_names.pkl",
    "powerful": BASE / "powerful" / "feature_names.pkl",
    "deployment_data": BASE / "deployment_data" / "feature_names.pkl",
}

TARGETS = [
    BASE / "feature_names.json",
    BASE / "weak" / "feature_names.json",
    BASE / "medium" / "feature_names.json",
    BASE / "powerful" / "feature_names.json",
    BASE / "deployment_data" / "feature_names.json",
]


def load_pkl(path: Path) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    loaded = {}
    for name, path in SOURCES.items():
        if not path.exists():
            print(f"[ERREUR] Fichier manquant : {path}", file=sys.stderr)
            sys.exit(1)
        loaded[name] = load_pkl(path)
        print(f"  Charge {name}: {len(loaded[name])} features")

    reference_name = "weak"
    reference = loaded[reference_name]
    all_equal = True

    for name, features in loaded.items():
        if name == reference_name:
            continue
        if tuple(features) != tuple(reference):
            all_equal = False
            diff = list(difflib.unified_diff(
                [f"{i}: {v}" for i, v in enumerate(reference)],
                [f"{i}: {v}" for i, v in enumerate(features)],
                fromfile=f"{reference_name}/feature_names.pkl",
                tofile=f"{name}/feature_names.pkl",
                lineterm="",
            ))
            print(f"\n[ERREUR] DIVERGENCE entre '{reference_name}' et '{name}':", file=sys.stderr)
            print("\n".join(diff), file=sys.stderr)

    if not all_equal:
        print("\n[ABORT] Les feature_names.pkl ne sont pas identiques.", file=sys.stderr)
        sys.exit(1)

    feature_count = len(reference)
    if feature_count != 28:
        warnings.warn(
            f"WARNING : {feature_count} features (attendu 28). Verifier avec Saif.",
            stacklevel=2,
        )
        print(f"[WARNING] Nombre de features = {feature_count} (attendu 28)")
    else:
        print(f"\n[OK] 4 feature_names.pkl identiques, {feature_count} features")

    print(f"  Features : {reference}")

    payload = json.dumps(reference, indent=2, ensure_ascii=False)
    for target in TARGETS:
        target.write_text(payload, encoding="utf-8")

    print(f"\n[OK] feature_names.json ecrit dans {len(TARGETS)} emplacements :")
    for t in TARGETS:
        print(f"  {t}")


if __name__ == "__main__":
    main()
