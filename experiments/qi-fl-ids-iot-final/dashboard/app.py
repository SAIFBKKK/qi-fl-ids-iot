from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from data_loader import load_evaluations, load_figures, load_registry, load_summary


DASHBOARD_DIR = Path(__file__).resolve().parent
FINAL_DIR = DASHBOARD_DIR.parent
REPO_ROOT = FINAL_DIR.parents[1]
SCRIPTS_DIR = FINAL_DIR / "src" / "scripts"

app = FastAPI(title="QI-FL-IDS-IoT Final L1 Dashboard", version="1.5")
app.mount("/static", StaticFiles(directory=DASHBOARD_DIR / "static"), name="static")
templates = Jinja2Templates(directory=DASHBOARD_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "summary": load_summary(),
            "registry": load_registry(),
            "evaluations": load_evaluations(),
        },
    )


@app.get("/api/summary")
async def api_summary() -> dict[str, Any]:
    return load_summary()


@app.get("/api/models")
async def api_models() -> dict[str, Any]:
    return load_registry()


@app.get("/api/evaluations")
async def api_evaluations() -> dict[str, Any]:
    return {"models": load_evaluations()}


@app.get("/api/figures")
async def api_figures() -> dict[str, Any]:
    return load_figures()


@app.post("/api/evaluate/{model_id}")
async def api_evaluate(model_id: str) -> dict[str, Any]:
    registry = load_registry()
    if model_id not in {model.get("model_id") for model in registry.get("models", [])}:
        raise HTTPException(status_code=404, detail="Unknown model_id")

    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    if str(DASHBOARD_DIR) not in sys.path:
        sys.path.insert(0, str(DASHBOARD_DIR))

    from evaluation.evaluator import evaluate_models, write_evaluation_outputs

    rows, warnings = evaluate_models(DASHBOARD_DIR / "model_registry.json")
    write_evaluation_outputs(rows, warnings)
    selected = [row for row in rows if row.get("model_id") == model_id]
    return {"model_id": model_id, "evaluation": selected, "warnings": warnings}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "p13-dashboard"}


@app.get("/ready")
async def ready() -> dict[str, Any]:
    summary = load_summary()
    registry = load_registry()
    return {
        "ready": bool(summary and registry.get("models")),
        "summary_loaded": bool(summary),
        "models": len(registry.get("models", [])),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8013, reload=False)
