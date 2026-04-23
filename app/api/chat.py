from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from app.api.dependencies import get_chat_service
from app.core.context import RequestContext
from app.models.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService


router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat_endpoint(
    payload: ChatRequest,
    request: Request,
    response: Response,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        raise HTTPException(
            status_code=415,
            detail="Unsupported content type. Use application/json for POST /chat.",
        )

    request_id = getattr(request.state, "request_id", None) or "unknown-request"
    result = chat_service.handle_chat(
        payload,
        RequestContext(
            request_id=request_id,
            trace_id=request_id,
            metadata={"path": request.url.path, "method": request.method},
        ),
    )
    response.headers["X-Trace-ID"] = result.trace_id
    response.headers["X-Request-ID"] = result.request_id
    return result
