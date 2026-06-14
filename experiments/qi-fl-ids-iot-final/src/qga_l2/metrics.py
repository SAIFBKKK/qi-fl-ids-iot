"""Metrics re-exports for P8-b QGA L2."""

from qga_l2.fitness_l2 import macro_metrics_from_confusion, multiclass_confusion

__all__ = ["macro_metrics_from_confusion", "multiclass_confusion"]
