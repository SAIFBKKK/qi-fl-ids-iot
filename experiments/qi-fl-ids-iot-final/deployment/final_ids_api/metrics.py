from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter


@dataclass
class ApiMetrics:
    started_at: float
    predictions_total: int = 0
    prediction_errors_total: int = 0

    @classmethod
    def create(cls) -> "ApiMetrics":
        return cls(started_at=perf_counter())

    def as_prometheus(self, ready: bool) -> str:
        uptime = perf_counter() - self.started_at
        lines = [
            "# HELP final_ids_api_ready Readiness state for the final IDS API.",
            "# TYPE final_ids_api_ready gauge",
            f"final_ids_api_ready {1 if ready else 0}",
            "# HELP final_ids_api_predictions_total Total prediction rows served.",
            "# TYPE final_ids_api_predictions_total counter",
            f"final_ids_api_predictions_total {self.predictions_total}",
            "# HELP final_ids_api_prediction_errors_total Total prediction errors.",
            "# TYPE final_ids_api_prediction_errors_total counter",
            f"final_ids_api_prediction_errors_total {self.prediction_errors_total}",
            "# HELP final_ids_api_uptime_seconds Process uptime seconds.",
            "# TYPE final_ids_api_uptime_seconds gauge",
            f"final_ids_api_uptime_seconds {uptime:.3f}",
            "",
        ]
        return "\n".join(lines)
