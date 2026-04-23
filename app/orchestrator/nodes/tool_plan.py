from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.client import (
    LLMClientConfig,
    build_model_candidates,
    classify_llm_error,
    get_llm_client,
    get_llm_debug_key_info,
    mark_llm_failure,
    mark_llm_success,
)
from app.models.orchestration import OrchestrationState, ToolPlanResult
from app.orchestrator.nodes.base import NodeResult


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolPlanningNode:
    name: str = "tool_planning"
    llm_temperature: float = 0.0
    llm_model: str = "gemini-flash-lite-latest"
    max_tool_calls: int = 3
    llm_timeout_seconds: int = 20
    llm_fallback_models: tuple[str, ...] = (
        "gemini-1.5-flash",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-2.5-flash",
    )

    def run(self, state: OrchestrationState) -> NodeResult:
        tool_calls = self._plan_with_llm(state)
        if tool_calls is None:
            tool_calls = self._heuristic_plan(state)
            reason = "heuristic_plan_fallback"
            state.metadata["tool_plan_mode"] = "heuristic_fallback"
        else:
            reason = "llm_plan"
            state.metadata["tool_plan_mode"] = "llm"

        state.tool_plan = ToolPlanResult(tool_calls=tool_calls[: self.max_tool_calls], reason=reason)
        return NodeResult(state=state)

    def _plan_with_llm(self, state: OrchestrationState) -> list[dict] | None:
        has_retrieval_chunks = bool(state.retrieval_context and state.retrieval_context.chunks)
        system_prompt = (
            "You are a tool planner for an orchestrator backend. "
            "Return only raw JSON with schema: {\"tool_calls\": [{\"tool_name\": str, \"arguments\": object}]}. "
            "Allowed tools: calculator, slow_tool, flaky_tool, retrieval_tool. "
            "If no tools are needed, return {\"tool_calls\": []}. "
            "Never return more than 3 tool calls."
        )
        user_prompt = (
            f"message: {state.turn.message}\n"
            f"intent: {state.intent.intent if state.intent else 'unknown'}\n"
            f"document_ids: {state.turn.document_ids}\n"
            f"has_retrieval_chunks: {has_retrieval_chunks}\n"
            "Plan tool calls now as JSON only."
        )

        state.metadata["tool_plan_llm_request"] = {
            "primary_model": self.llm_model,
            "candidate_models": build_model_candidates(self.llm_model, self.llm_fallback_models),
            "temperature": self.llm_temperature,
            "timeout_seconds": max(self.llm_timeout_seconds, 10),
            "system_chars": len(system_prompt),
            "user_chars": len(user_prompt),
            **get_llm_debug_key_info(),
        }

        attempt_errors: list[dict[str, object]] = []
        for model_name in build_model_candidates(self.llm_model, self.llm_fallback_models):
            try:
                llm = get_llm_client(
                    LLMClientConfig(
                        model_name=model_name,
                        temperature=self.llm_temperature,
                        timeout_seconds=self.llm_timeout_seconds,
                        fallback_models=self.llm_fallback_models,
                    )
                )
            except Exception as exc:
                mark_llm_failure()
                error_category, is_quota_error = classify_llm_error(exc)
                attempt_errors.append(
                    {
                        "stage": "client_init",
                        "model": model_name,
                        "error_type": type(exc).__name__,
                        "error_category": error_category,
                        "quota_exceeded": is_quota_error,
                        "error": str(exc),
                    }
                )
                logger.warning(
                    "tool_planning_llm_client_init_failed",
                    extra={"model": model_name, "error": str(exc), "error_type": type(exc).__name__},
                )
                continue

            try:
                response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            except Exception as exc:
                mark_llm_failure()
                error_category, is_quota_error = classify_llm_error(exc)
                attempt_errors.append(
                    {
                        "stage": "invoke",
                        "model": model_name,
                        "error_type": type(exc).__name__,
                        "error_category": error_category,
                        "quota_exceeded": is_quota_error,
                        "error": str(exc),
                    }
                )
                logger.warning(
                    "tool_planning_llm_invoke_failed",
                    extra={"model": model_name, "error": str(exc), "error_type": type(exc).__name__},
                )
                continue
            mark_llm_success()

            payload = self._parse_llm_tool_plan(getattr(response, "content", ""))
            if payload is None:
                attempt_errors.append(
                    {
                        "stage": "parse",
                        "model": model_name,
                        "error_type": "InvalidToolPlanJson",
                        "error_category": "invalid_json",
                        "quota_exceeded": False,
                        "error": "Tool planning response could not be parsed as valid JSON",
                    }
                )
                continue

            state.metadata["tool_plan_llm_model"] = model_name
            state.metadata["tool_plan_llm_attempts"] = len(attempt_errors) + 1
            return payload

        if attempt_errors:
            last = attempt_errors[-1]
            state.metadata["tool_plan_llm_error"] = f"{last.get('error_type')}: {last.get('error')}"
            state.metadata["tool_plan_llm_error_type"] = str(last.get("error_type", "unknown"))
            state.metadata["tool_plan_llm_error_category"] = str(last.get("error_category", "unknown"))
            state.metadata["tool_plan_llm_quota_exceeded"] = bool(last.get("quota_exceeded", False))
            state.metadata["tool_plan_llm_attempt_errors"] = attempt_errors
            state.metadata["tool_plan_llm_attempts"] = len(attempt_errors)

        return None

    def _parse_llm_tool_plan(self, raw: object) -> list[dict] | None:
        text = raw if isinstance(raw, str) else str(raw)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start : end + 1]

        try:
            payload = json.loads(candidate)
        except Exception:
            return None

        calls_raw = payload.get("tool_calls")
        if not isinstance(calls_raw, list):
            return None

        allowed = {"calculator", "slow_tool", "flaky_tool", "retrieval_tool"}
        cleaned: list[dict] = []
        for item in calls_raw:
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("tool_name", "")).strip()
            if tool_name not in allowed:
                continue
            arguments = item.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            cleaned.append({"tool_name": tool_name, "arguments": arguments})
            if len(cleaned) >= self.max_tool_calls:
                break
        return cleaned

    def _heuristic_plan(self, state: OrchestrationState) -> list[dict]:
        message = state.turn.message.lower().strip()
        tool_calls: list[dict] = []

        if state.intent and state.intent.intent == "calculation":
            expression = self._extract_math_expression(state.turn.message)
            tool_calls.append({"tool_name": "calculator", "arguments": {"expression": expression}})

        if any(token in message for token in ["slow", "delay", "wait"]):
            tool_calls.append({"tool_name": "slow_tool", "arguments": {"message": state.turn.message}})

        if any(token in message for token in ["flaky", "retry", "unstable"]):
            tool_calls.append({"tool_name": "flaky_tool", "arguments": {"payload": {"message": state.turn.message}}})

        if (
            state.intent
            and state.intent.intent == "grounded_qa"
            and state.retrieval_context
            and state.retrieval_context.chunks
            and not self._is_personal_memory_question(message)
        ):
            tool_calls.append(
                {
                    "tool_name": "retrieval_tool",
                    "arguments": {
                        "query": state.turn.message,
                        "document_ids": state.turn.document_ids,
                        "top_k": 5,
                    },
                }
            )
        return tool_calls

    def _extract_math_expression(self, message: str) -> str:
        # Keep only arithmetic tokens for calculator safety/compatibility.
        filtered = "".join(ch for ch in message if ch in "0123456789+-*/().% ")
        collapsed = re.sub(r"\s+", " ", filtered).strip()
        return collapsed or message

    def _is_personal_memory_question(self, message: str) -> bool:
        normalized = re.sub(r"\s+", " ", message.strip().lower())
        patterns = [
            r"\bwhat\s+is\s+my\s+name\b",
            r"\bwho\s+am\s+i\b",
            r"\bhow\s+old\s+(?:am\s+i|i\s+am)\b",
            r"\bwhat(?:'s|\s+is)\s+my\s+age\b",
            r"\brisk\s+(?:tolerance|tolerence|tollerance|profile)\b",
            r"\b(?:tolerance|tolerence|tollerance)\s+risk\b",
        ]
        return any(re.search(pattern, normalized) for pattern in patterns)
