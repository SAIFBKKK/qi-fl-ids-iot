import os
import numpy as np
from collections import Counter

BASE_DIR = "data/processed"
SCENARIOS = ["normal_noniid", "absent_local", "rare_expert"]
NODES = ["node1", "node2", "node3"]

BENIGN_CLASS = 1

def load_node(scenario, node):
    path = os.path.join(BASE_DIR, scenario, node, "train_preprocessed.npz")
    data = np.load(path)
    return data["X"], data["y"]


def check_basic_stats(X):
    return {
        "mean": float(X.mean()),
        "std": float(X.std()),
        "nan": int(np.isnan(X).sum()),
        "inf": int(np.isinf(X).sum())
    }


def check_classes(y):
    return {
        "num_classes": len(set(y)),
        "classes": sorted(list(set(y)))
    }


def compute_imbalance(y):
    counts = Counter(y)
    return max(counts.values()) / min(counts.values())


def run_validation():
    print("\n========== DATA VALIDATION ==========\n")

    for scenario in SCENARIOS:
        print(f"\n===== Scenario: {scenario} =====")

        global_classes = set()

        for node in NODES:
            X, y = load_node(scenario, node)

            stats = check_basic_stats(X)
            cls = check_classes(y)
            imbalance = compute_imbalance(y)

            global_classes |= set(y)

            print(f"\n[{node}]")
            print(f" shape: {X.shape}")
            print(f" mean: {stats['mean']:.4f}")
            print(f" std : {stats['std']:.4f}")
            print(f" NaN : {stats['nan']} | Inf: {stats['inf']}")
            print(f" classes: {cls['num_classes']}")
            print(f" imbalance: {imbalance:.2f}")

            # 🔴 Check benign presence
            if BENIGN_CLASS not in set(y):
                print(" ❌ ERROR: Benign class missing!")
            else:
                print(" ✅ Benign present")

            # ⚠️ Check scaling
            if not (0.8 <= stats["std"] <= 1.2):
                print(" ⚠️ WARNING: std outside expected range")

        print("\n[GLOBAL CHECK]")
        print(f" total unique classes: {len(global_classes)}")

        if scenario != "rare_expert":
            if len(global_classes) != 34:
                print(" ❌ ERROR: Missing global classes")
            else:
                print(" ✅ All classes present globally")

        else:
            print(" ℹ️ rare_expert scenario uses split class design")

    print("\n========== VALIDATION DONE ==========\n")


if __name__ == "__main__":
    run_validation()
