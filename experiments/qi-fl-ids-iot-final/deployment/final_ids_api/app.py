from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from metrics import ApiMetrics
from model_loader import L1MLPNotAvailable, TorchL1Model
from preprocessor import FeatureValidationError, L1Preprocessor
from schemas import BatchPredictRequest, BatchPredictResponse, PredictRequest, PredictResponse


app = FastAPI(title="QI-FL-IDS-IoT Final IDS API", version="1.6")
model = TorchL1Model()
preprocessor = L1Preprocessor()
api_metrics = ApiMetrics.create()


def _threshold(request_threshold: float | None) -> float:
    if request_threshold is not None:
        return float(request_threshold)
    return float(model.selected_model.get("threshold", 0.4))


def _format_prediction(probability_attack: float, threshold: float) -> PredictResponse:
    prediction = int(probability_attack >= threshold)
    return PredictResponse(
        prediction=prediction,
        probability_attack=probability_attack,
        label="attack" if prediction == 1 else "normal",
        model_id=model.selected_model.get("model_id", "p8_fedavg_qga_l1"),
        selected_features_count=int(model.selected_model.get("features_count", 12)),
        threshold=threshold,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "final-ids-api"}


@app.get("/ready")
async def ready() -> dict[str, Any]:
    return {"ready": model.ready, "message": "ready" if model.ready else model.error or "model artifact unavailable"}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    return api_metrics.as_prometheus(model.ready)


@app.get("/model/info")
async def model_info() -> dict[str, Any]:
    return {
        "ready": model.ready,
        "error": model.error,
        "selected_model": model.selected_model,
        "deployment_manifest": model.manifest,
        "feature_schema": preprocessor.schema,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest) -> PredictResponse:
    try:
        X = preprocessor.transform_one(payload.features)
        probabilities = model.predict_proba(X)
        response = _format_prediction(float(probabilities[0, 1]), _threshold(payload.threshold))
        api_metrics.predictions_total += 1
        return response
    except FeatureValidationError as exc:
        api_metrics.prediction_errors_total += 1
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except L1MLPNotAvailable as exc:
        api_metrics.prediction_errors_total += 1
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    try:
        X = preprocessor.transform_batch(payload.rows)
        probabilities = model.predict_proba(X)
        threshold = _threshold(payload.threshold)
        predictions = [_format_prediction(float(value), threshold) for value in probabilities[:, 1]]
        api_metrics.predictions_total += len(predictions)
        return BatchPredictResponse(
            predictions=predictions,
            model_id=model.selected_model.get("model_id", "p8_fedavg_qga_l1"),
            count=len(predictions),
        )
    except FeatureValidationError as exc:
        api_metrics.prediction_errors_total += 1
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except L1MLPNotAvailable as exc:
        api_metrics.prediction_errors_total += 1
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8014, reload=False)
