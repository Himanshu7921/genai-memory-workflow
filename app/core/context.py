from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RequestContext:
    request_id: str
    trace_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
