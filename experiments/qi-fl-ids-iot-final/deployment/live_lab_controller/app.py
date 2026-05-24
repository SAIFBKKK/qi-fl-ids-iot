from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from metrics import ControllerMetrics
from node_registry import NodeRegistration, NodeRegistry, SUPPORTED_INPUT_MODES


class RegisterNodeRequest(BaseModel):
    node_id: str = Field(..., min_length=1)
    hostname: str = Field(..., min_length=1)
    cpu_count: int = Field(..., gt=0)
    ram_gb: float = Field(..., gt=0)
    device_type: str = Field(default="unknown", min_length=1)
    mqtt_topic: str | None = None


class RegisterNodeResponse(BaseModel):
    node_id: str
    assigned_tier: str
    model_id: str
    selected_mask_id: str
    supported_input_modes: list[str]
    mqtt_publish_topic: str
    mqtt_prediction_topic: str
    mqtt_alert_topic: str


registry = NodeRegistry()
controller_metrics = ControllerMetrics()
app = FastAPI(title="QI-FL-IDS-IoT P16 Live Lab Controller", version="1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "live-lab-controller"}


@app.get("/ready")
def ready() -> dict[str, Any]:
    return {
        "ready": True,
        "service": "live-lab-controller",
        "supported_input_modes": list(SUPPORTED_INPUT_MODES),
        "registered_nodes": registry.count(),
    }


@app.post("/register-node", response_model=RegisterNodeResponse)
def register_node(payload: RegisterNodeRequest) -> dict[str, object]:
    try:
        node = registry.register(
            NodeRegistration(
                node_id=payload.node_id,
                hostname=payload.hostname,
                cpu_count=payload.cpu_count,
                ram_gb=payload.ram_gb,
                device_type=payload.device_type,
                mqtt_topic=payload.mqtt_topic,
            )
        )
        controller_metrics.mark_registration(node.assigned_tier)
        return node.assignment()
    except ValueError as exc:
        controller_metrics.mark_error(str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/nodes")
def nodes() -> dict[str, object]:
    node_list = registry.list_nodes()
    return {"count": len(node_list), "nodes": node_list}


@app.get("/assignments")
def assignments() -> dict[str, object]:
    assignment_list = registry.assignments()
    return {"count": len(assignment_list), "assignments": assignment_list}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return controller_metrics.prometheus_text(registry.count())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8020, reload=False)

