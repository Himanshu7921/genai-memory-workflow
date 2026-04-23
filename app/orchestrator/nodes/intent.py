from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from time import perf_counter

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
from app.models.orchestration import IntentResult, ModelSelection, OrchestrationState
from app.orchestrator.nodes.base import NodeResult


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IntentClassificationNode:
    name: str = "intent_classification"
    llm_temperature: float = 0.0
    llm_model: str = "gemini-flash-lite-latest"
    llm_timeout_seconds: int = 20
    llm_fallback_models: tuple[str, ...] = (
        "gemini-1.5-flash",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-2.5-flash",
    )

    def run(self, state: OrchestrationState) -> NodeResult:
        start = perf_counter()
        result = self._classify_with_llm(state)
        if result is None:
            intent, labels = self._heuristic_fallback(state.turn.message)
            state.intent = IntentResult(
                intent=intent,
                confidence=0.78,
                labels=labels,
                model_selection=ModelSelection(model_name="classifier-heuristic", fallback_used=True, reason="llm_unavailable_fallback"),
            )
            state.metadata["intent_classification_mode"] = "heuristic_fallback"
            state.metadata["intent_classified_at"] = start
            return NodeResult(state=state)

        intent, confidence, labels = result
        if intent == "grounded_qa" and not state.turn.document_ids and not self._is_explicit_document_query(state.turn.message):
            intent = "general_qa"
            labels = [label for label in labels if label != "retrieval_needed"]
            if "answer_only" not in labels:
                labels.append("answer_only")
            state.metadata["intent_rewrite_reason"] = "grounded_qa_without_documents"

        state.intent = IntentResult(
            intent=intent,
            confidence=confidence,
            labels=labels,
            model_selection=ModelSelection(model_name=self.llm_model, fallback_used=False, reason="llm_classifier"),
        )
        state.metadata["intent_classification_mode"] = "llm"
        state.metadata["intent_classified_at"] = start
        return NodeResult(state=state)

    def _classify_with_llm(self, state: OrchestrationState) -> tuple[str, float, list[str]] | None:
        message = state.turn.message
        system_prompt = (
            "You are an intent classifier for an orchestration backend. "
            "Return only JSON with keys: intent, confidence, labels. "
            "Allowed intent values: calculation, grounded_qa, memory_write, general_qa. "
            "Use grounded_qa only when the user explicitly asks about supplied documents, files, policies, or citations. "
            "For open-ended knowledge or advice without required document evidence, use general_qa. "
            "Confidence must be a float between 0 and 1. "
            "labels must be a list of short tags."
        )
        user_prompt = (
            "Classify this user message into one allowed intent.\n"
            f"message: {message}\n"
            "Respond with raw JSON only."
        )

        state.metadata["intent_llm_request"] = {
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
                    "intent_classification_llm_client_init_failed",
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
                    "intent_classification_llm_invoke_failed",
                    extra={"model": model_name, "error": str(exc), "error_type": type(exc).__name__},
                )
                continue
            mark_llm_success()

            parsed = self._parse_classifier_json(getattr(response, "content", ""))
            if parsed is None:
                attempt_errors.append(
                    {
                        "stage": "parse",
                        "model": model_name,
                        "error_type": "InvalidClassifierJson",
                        "error_category": "invalid_json",
                        "quota_exceeded": False,
                        "error": "Classifier response could not be parsed as valid JSON",
                    }
                )
                continue

            intent, confidence, labels = parsed
            if intent not in {"calculation", "grounded_qa", "memory_write", "general_qa"}:
                attempt_errors.append(
                    {
                        "stage": "parse",
                        "model": model_name,
                        "error_type": "InvalidIntent",
                        "error_category": "invalid_intent",
                        "quota_exceeded": False,
                        "error": f"Unexpected intent value: {intent}",
                    }
                )
                continue

            state.metadata["intent_llm_model"] = model_name
            state.metadata["intent_llm_attempts"] = len(attempt_errors) + 1
            return intent, confidence, labels

        if attempt_errors:
            last = attempt_errors[-1]
            state.metadata["intent_llm_error"] = f"{last.get('error_type')}: {last.get('error')}"
            state.metadata["intent_llm_error_type"] = str(last.get("error_type", "unknown"))
            state.metadata["intent_llm_error_category"] = str(last.get("error_category", "unknown"))
            state.metadata["intent_llm_quota_exceeded"] = bool(last.get("quota_exceeded", False))
            state.metadata["intent_llm_attempt_errors"] = attempt_errors
            state.metadata["intent_llm_attempts"] = len(attempt_errors)

        return None

    def _parse_classifier_json(self, raw: object) -> tuple[str, float, list[str]] | None:
        text = raw if isinstance(raw, str) else str(raw)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = match.group(0) if match else text
        try:
            payload = json.loads(candidate)
        except Exception:
            return None

        intent = str(payload.get("intent", "")).strip().lower()
        confidence_raw = payload.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except Exception:
            confidence = 0.0
        confidence = min(max(confidence, 0.0), 1.0)
        labels_raw = payload.get("labels")
        if isinstance(labels_raw, list):
            labels = [str(item).strip().lower() for item in labels_raw if str(item).strip()]
        else:
            labels = []
        return intent, confidence, labels

    def _heuristic_fallback(self, message: str) -> tuple[str, list[str]]:
        text = message.lower().strip()
        if any(token in text for token in ["calculate", "sum", "add", "+", "minus", "multiply"]):
            intent = "calculation"
            labels = ["tool_needed", "math"]
        elif any(token in text for token in ["document", "file", "policy", "warranty", "search", "find"]):
            intent = "grounded_qa"
            labels = ["retrieval_needed"]
        elif any(token in text for token in ["remember", "save", "my name", "i prefer", "i work", "i am"]):
            intent = "memory_write"
            labels = ["memory_write"]
        else:
            intent = "general_qa"
            labels = ["answer_only"]
        return intent, labels

    def _is_explicit_document_query(self, message: str) -> bool:
        text = message.lower().strip()
        return any(token in text for token in ["document", "docs", "file", "policy", "manual", "warranty", "citation", "source"])
