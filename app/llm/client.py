from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import logging
import os

from langchain_google_genai import ChatGoogleGenerativeAI


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LLMClientConfig:
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.2
    timeout_seconds: int = 20
    fallback_models: tuple[str, ...] = (
        "gemini-1.5-flash",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-2.5-flash",
    )


_cached_client: ChatGoogleGenerativeAI | None = None
_cached_config: LLMClientConfig | None = None
_disabled_until: datetime | None = None


def llm_available() -> bool:
    if _disabled_until is None:
        return True
    return datetime.utcnow() >= _disabled_until


def mark_llm_failure(cooldown_seconds: int = 120) -> None:
    global _disabled_until
    _disabled_until = datetime.utcnow() + timedelta(seconds=cooldown_seconds)


def mark_llm_success() -> None:
    global _disabled_until
    _disabled_until = None


def _resolve_google_api_key() -> str:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    # Prefer GEMINI_API_KEY when both are present.
    api_key = gemini_api_key or google_api_key
    if not api_key:
        raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY.")

    # langchain-google-genai prefers GOOGLE_API_KEY when both are set.
    # Normalize to the chosen key to avoid accidentally using a stale key.
    if gemini_api_key:
        if google_api_key and google_api_key != gemini_api_key:
            logger.warning("Both GOOGLE_API_KEY and GEMINI_API_KEY are set with different values; forcing GOOGLE_API_KEY to GEMINI_API_KEY for consistency")
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
    return api_key


def build_model_candidates(primary_model: str, fallback_models: tuple[str, ...] | None = None) -> list[str]:
    models: list[str] = [primary_model]
    for model in fallback_models or ():
        if model and model not in models:
            models.append(model)
    return models


def classify_llm_error(exc: Exception) -> tuple[str, bool]:
    message = str(exc)
    lowered = message.lower()
    is_quota_error = "429" in message or "resource_exhausted" in lowered or "quota exceeded" in lowered
    if is_quota_error:
        return "quota_exceeded", True
    if "not_found" in lowered or "not found" in lowered:
        return "model_not_found", False
    if "timeout" in lowered or "deadline" in lowered:
        return "timeout", False
    if "missing gemini api key" in lowered:
        return "missing_api_key", False
    return "unknown", False


def get_llm_debug_key_info() -> dict[str, object]:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    selected_key = gemini_api_key or google_api_key
    key_source = "GEMINI_API_KEY" if gemini_api_key else ("GOOGLE_API_KEY" if google_api_key else "missing")
    key_fingerprint = None
    if selected_key:
        key_fingerprint = hashlib.sha256(selected_key.encode("utf-8")).hexdigest()[:10]
    return {
        "selected_key_source": key_source,
        "has_gemini_api_key": bool(gemini_api_key),
        "has_google_api_key": bool(google_api_key),
        "keys_match": (gemini_api_key == google_api_key) if gemini_api_key and google_api_key else None,
        "selected_key_fingerprint": key_fingerprint,
    }


def create_llm_client(config: LLMClientConfig | None = None) -> ChatGoogleGenerativeAI:
    cfg = config or LLMClientConfig()
    timeout_seconds = max(cfg.timeout_seconds, 10)
    return ChatGoogleGenerativeAI(
        model=cfg.model_name,
        temperature=cfg.temperature,
        timeout=timeout_seconds,
        max_retries=0,
        api_key=_resolve_google_api_key()
    )


def get_llm_client(config: LLMClientConfig | None = None) -> ChatGoogleGenerativeAI:
    global _cached_client
    global _cached_config

    cfg = config or LLMClientConfig()
    if _cached_client is None or _cached_config != cfg:
        _cached_client = create_llm_client(cfg)
        _cached_config = cfg
    return _cached_client
