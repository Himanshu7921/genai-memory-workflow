from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("app.orchestrator")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


LOGGER = _build_logger()


@dataclass(slots=True)
class TraceLogger:
    trace_id: str
    user_id: str
    session_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def log_node_start(self, node_name: str, attempt: int, extra: dict[str, Any] | None = None) -> None:
        LOGGER.info(
            "node_start trace_id=%s user_id=%s session_id=%s node=%s attempt=%s %s",
            self.trace_id,
            self.user_id,
            self.session_id,
            node_name,
            attempt,
            extra or {},
        )

    def log_node_end(self, node_name: str, success: bool, duration_ms: int, extra: dict[str, Any] | None = None) -> None:
        LOGGER.info(
            "node_end trace_id=%s user_id=%s session_id=%s node=%s success=%s duration_ms=%s %s",
            self.trace_id,
            self.user_id,
            self.session_id,
            node_name,
            success,
            duration_ms,
            extra or {},
        )

    def log_warning(self, message: str, extra: dict[str, Any] | None = None) -> None:
        LOGGER.warning(
            "trace_id=%s user_id=%s session_id=%s warning=%s %s",
            self.trace_id,
            self.user_id,
            self.session_id,
            message,
            extra or {},
        )
