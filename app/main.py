from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.chat import router as chat_router
from app.api.ingest import router as ingest_router
from app.api.dependencies import build_container
from app.api.middleware import request_id_middleware
from app.services.ingest_service import configure_ingest_service, ingest_all_documents


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(item) for item in value]
    return value


def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM Workflow Service",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.container = build_container()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.middleware("http")(request_id_middleware)
    app.include_router(chat_router)
    app.include_router(ingest_router)

    @app.on_event("startup")
    async def startup_ingestion() -> None:
        container = app.state.container
        configure_ingest_service(container.retrieval_service)
        ingest_all_documents()

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", None) or "unknown-request"
        sanitized_errors = _sanitize_for_json(exc.errors())
        return JSONResponse(
            status_code=422,
            content={
                "detail": sanitized_errors,
                "request_id": request_id,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None) or "unknown-request"
        return JSONResponse(
            status_code=500,
            content={
                "detail": "internal_server_error",
                "request_id": request_id,
            },
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
