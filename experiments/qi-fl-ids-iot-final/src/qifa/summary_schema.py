"""Expected artifacts and figure lists for QIFA."""

from __future__ import annotations


def expected_artifacts() -> list[str]:
    return [
        "runs/{run_id}/artifacts/run_summary.json",
        "runs/{run_id}/artifacts/run_manifest.json",
        "runs/{run_id}/artifacts/model_config.json",
        "runs/{run_id}/artifacts/metrics_rounds.csv",
        "runs/{run_id}/artifacts/metrics_clients.csv",
        "runs/{run_id}/artifacts/aggregation_weights.csv",
        "runs/{run_id}/artifacts/qifa_scores.csv",
        "runs/{run_id}/artifacts/qifa_probabilities.csv",
        "runs/{run_id}/artifacts/qifa_amplitudes.csv",
        "runs/{run_id}/artifacts/qifa_entropy.csv",
        "runs/{run_id}/artifacts/bandwidth_rounds.csv",
        "runs/{run_id}/artifacts/metrics_val.json",
        "runs/{run_id}/artifacts/metrics_test.json",
        "runs/{run_id}/artifacts/threshold.json",
        "runs/{run_id}/artifacts/threshold_sweep.csv",
        "runs/{run_id}/artifacts/classification_report.json",
        "runs/{run_id}/artifacts/confusion_matrix.csv",
        "runs/{run_id}/artifacts/comparison_with_p5.json",
        "runs/{run_id}/checkpoints/best_global_model.pth",
        "runs/{run_id}/checkpoints/last_global_model.pth",
        "runs/{run_id}/logs/flower_server.log",
        "runs/{run_id}/logs/flower_clients.log",
        "runs/{run_id}/logs/run_console.log",
    ]


def expected_figures() -> list[str]:
    return [
        "qifa_weights_by_round.png",
        "qifa_scores_by_client.png",
        "qifa_probabilities_by_round.png",
        "qifa_vs_fedavg_macro_f1.png",
        "qifa_vs_fedavg_fpr.png",
        "qifa_convergence_loss.png",
        "qifa_client_contribution_heatmap.png",
        "qifa_alpha_robustness_heatmap.png",
        "qifa_gamma_sensitivity.png",
        "p9_ablation_summary.png",
    ]
