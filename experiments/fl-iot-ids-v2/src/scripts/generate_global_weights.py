"""
Génère class_weights_34.pkl depuis la distribution globale du dataset complet.
À exécuter UNE SEULE FOIS avant tout entraînement FL.

Usage:
    python -m src.scripts.generate_global_weights
"""
from __future__ import annotations

import pickle

import torch

from src.common.paths import ARTIFACTS_DIR

NUM_CLASSES = 34

# Distribution globale — somme des 3 nodes (normal_noniid, 9.4M samples)
GLOBAL_COUNTS: dict[int, int] = {
    0:  57639  + 91726  + 635,
    1:  249246 + 5413   + 45341,
    2:  71959  + 49928  + 171063,
    3:  38385  + 189685 + 42380,
    4:  8613   + 252979 + 38408,
    5:  228994 + 17531  + 53475,
    6:  30878  + 92309  + 176813,
    7:  29303  + 103170 + 167527,
    8:  215179 + 77619  + 7202,
    9:  119031 + 77706  + 103263,
    10: 261266 + 782    + 37952,
    11: 279782 + 767    + 19451,
    12: 52601  + 81842  + 165557,
    13: 236    + 287892 + 11872,
    14: 22224  + 190688 + 87088,
    15: 2192   + 286682 + 11126,
    16: 238782 + 58526  + 2692,
    17: 32235  + 10761  + 257004,
    18: 292603 + 5952   + 1445,
    19: 219664 + 21585  + 58751,
    20: 24545  + 215884 + 59571,
    21: 25020  + 251297 + 23683,
    22: 27771  + 267261 + 4968,
    23: 4812   + 32284  + 262904,
    24: 9260   + 42277  + 248463,
    25: 256052 + 23866  + 20082,
    26: 69720  + 28833  + 201447,
    27: 179533 + 30454  + 90013,
    28: 25965  + 86581  + 554,
    29: 16199  + 257200 + 26601,
    30: 256738 + 61     + 5451,
    31: 1521   + 56992  + 4087,
    32: 66018  + 1855   + 232127,
    33: 20631  + 110679 + 18690,
}


def generate() -> None:
    assert len(GLOBAL_COUNTS) == NUM_CLASSES, \
        f"GLOBAL_COUNTS a {len(GLOBAL_COUNTS)} entrées, attendu {NUM_CLASSES}"

    counts = torch.tensor(
        [GLOBAL_COUNTS[i] for i in range(NUM_CLASSES)],
        dtype=torch.float32,
    )
    counts = counts.clamp(min=1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES

    # Sur-pondération explicite de la classe benign (class 0).
    # Node3 n'a que 635 exemples benign — le signal est noyé sans boost.
    BENIGN_BOOST = 5.0
    weights[0] = weights[0] * BENIGN_BOOST
    weights = weights / weights.sum() * NUM_CLASSES  # renormalise

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / "class_weights_34.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(weights, f)

    total_samples = sum(GLOBAL_COUNTS.values())
    print(f"Dataset global : {total_samples:,} samples | {NUM_CLASSES} classes")
    print(f"Sauvegardé     : {out_path}")
    print(f"Classe 0 (benign) weight après boost x{BENIGN_BOOST} : {weights[0]:.4f}")
    print()
    print("Top 5 classes les plus pondérées (rares) :")
    top5 = weights.argsort(descending=True)[:5]
    for idx in top5:
        i = idx.item()
        print(f"  class {i:2d} → weight={weights[i]:.4f}  (count={GLOBAL_COUNTS[i]:>8,})")
    print()
    print("Top 5 classes les moins pondérées (dominantes) :")
    bot5 = weights.argsort(descending=False)[:5]
    for idx in bot5:
        i = idx.item()
        print(f"  class {i:2d} → weight={weights[i]:.6f}  (count={GLOBAL_COUNTS[i]:>8,})")


if __name__ == "__main__":
    generate()
