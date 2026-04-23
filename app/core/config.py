from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    session_recent_turns: int = 8
    session_summary_trigger_turns: int = 4
    session_context_char_budget: int = 2400
    session_summary_max_chars: int = 1600
    session_write_throttle_seconds: int = 30
    user_write_throttle_seconds: int = 60
    fact_decay_days: int = 30
    pinned_fact_decay_days: int = 180
    prune_low_priority_below: float = 0.25


DEFAULT_MEMORY_CONFIG = MemoryConfig()
