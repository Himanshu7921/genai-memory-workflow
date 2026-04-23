from __future__ import annotations

from dataclasses import dataclass
from time import sleep
from typing import Callable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_delay_seconds: float = 0.25
    backoff_factor: float = 2.0
    max_delay_seconds: float = 2.0


class RetryExhaustedError(RuntimeError):
    pass


def run_with_retry(
    operation: Callable[[], T],
    *,
    policy: RetryPolicy,
    should_retry: Callable[[BaseException], bool] | None = None,
) -> tuple[T, int]:
    attempt = 0
    delay = policy.initial_delay_seconds
    last_error: BaseException | None = None
    should_retry = should_retry or (lambda error: True)

    while attempt < policy.max_attempts:
        try:
            return operation(), attempt
        except Exception as exc:
            if not should_retry(exc):
                raise
            last_error = exc
            attempt += 1
            if attempt >= policy.max_attempts:
                break
            sleep(min(delay, policy.max_delay_seconds))
            delay *= policy.backoff_factor

    raise RetryExhaustedError(str(last_error) if last_error else "retry exhausted")
