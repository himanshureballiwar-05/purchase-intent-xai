"""FastAPI service: ``/predict``, ``/predict/batch``, ``/health``, ``/model/info``.

Run:  ``uvicorn src.api.main:app --reload``  ->  docs at http://localhost:8000/docs
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    PredictionResponse,
    SessionFeatures,
)
from src.config.settings import get_settings
from src.models.predictor import Predictor
from src.utils.logging_utils import get_logger

logger = get_logger("api")


def create_app(artifact_dir: str | Path | None = None) -> FastAPI:
    """App factory (tests pass their own ``artifact_dir``)."""
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.predictor = Predictor(artifact_dir)  # loaded ONCE at startup
        except FileNotFoundError as exc:
            logger.error("Model artifacts missing: %s", exc)
            app.state.predictor = None
        yield

    app = FastAPI(
        title=settings.section("api")["title"],
        version=str(settings.section("project")["version"]),
        description="Predicts whether a browsing session will end in a purchase, with SHAP explanations.",
        lifespan=lifespan,
    )

    def _predictor(request: Request) -> Predictor:
        pred = request.app.state.predictor
        if pred is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
        return pred

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        logger.info(
            "%s %s -> %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - start) * 1000,
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request) -> HealthResponse:
        pred = request.app.state.predictor
        if pred is None:
            return HealthResponse(status="degraded", model_loaded=False)
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_name=pred.metadata["model_name"],
            model_version=pred.metadata["trained_at"],
        )

    @app.get("/model/info", tags=["system"])
    def model_info(request: Request) -> dict:
        meta = _predictor(request).metadata
        keys = [
            "model_name",
            "strategy",
            "tuned",
            "threshold",
            "use_page_values",
            "trained_at",
            "test_metrics_at_threshold",
            "top_features",
            "break_even_probability",
        ]
        return {k: meta[k] for k in keys}

    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
    def predict(
        session: SessionFeatures, request: Request, explain: bool = True
    ) -> PredictionResponse:
        pred = _predictor(request)
        k = int(settings.section("api")["top_k_contributors"])
        result = pred.predict([session.model_dump()], explain=explain, top_k=k)[0]
        return PredictionResponse(**result)

    @app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
    def predict_batch(
        batch: BatchRequest, request: Request, explain: bool = False
    ) -> BatchResponse:
        pred = _predictor(request)
        results = pred.predict([s.model_dump() for s in batch.sessions], explain=explain)
        return BatchResponse(
            count=len(results), predictions=[PredictionResponse(**r) for r in results]
        )

    return app


app = create_app()
