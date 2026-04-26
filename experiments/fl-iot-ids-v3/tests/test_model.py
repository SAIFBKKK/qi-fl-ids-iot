from src.fl.aggregation_hooks import aggregate_evaluate_metrics, aggregate_fit_metrics


def test_fit_cost_metrics_are_summed_not_weighted_average():
    aggregated = aggregate_fit_metrics(
        [
            (
                100,
                {
                    "train_loss_last": 0.4,
                    "train_time_sec": 10.0,
                    "update_size_bytes": 1000,
                },
            ),
            (
                300,
                {
                    "train_loss_last": 0.2,
                    "train_time_sec": 20.0,
                    "update_size_bytes": 2000,
                },
            ),
        ]
    )

    assert aggregated["train_loss_last"] == 0.25
    assert aggregated["train_time_sec"] == 30.0
    assert aggregated["update_size_bytes"] == 3000.0


def test_rare_recall_is_recomputed_from_global_counts():
    aggregated = aggregate_evaluate_metrics(
        [
            (
                1000,
                {
                    "rare_class_recall": 1.0,
                    "tp_class_0": 1.0,
                    "fn_class_0": 0.0,
                    "fp_class_0": 0.0,
                },
            ),
            (
                1,
                {
                    "rare_class_recall": 0.0,
                    "tp_class_0": 0.0,
                    "fn_class_0": 9.0,
                    "fp_class_0": 0.0,
                },
            ),
        ]
    )

    assert aggregated["rare_class_recall"] == 0.1
    assert aggregated["rare_macro_f1"] == (2.0 / 11.0)
