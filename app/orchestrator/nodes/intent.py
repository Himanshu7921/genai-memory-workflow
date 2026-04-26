from __future__ import annotations

from dataclasses import dataclass, field
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
    llm_available,
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
    llm_timeout_seconds: float = 2.0
    llm_trigger_threshold: float = 0.9
    llm_fallback_models: tuple[str, ...] = ()
    cache_ttl_seconds: float = 120.0
    use_llm_for_ambiguous: bool = True
    _intent_cache: dict[str, tuple[float, tuple[str, float, list[str], str]]] = field(default_factory=dict)

    def run(self, state: OrchestrationState) -> NodeResult:
        start = perf_counter()
        message = state.turn.message
        cache_key = self._normalize(message)
        cached = self._get_cached(cache_key)

        if cached is not None:
            intent, confidence, labels, reason = cached
            state.metadata["intent_classification_mode"] = "cache"
            state.metadata["intent_classification_reason"] = reason
            model_selection = ModelSelection(model_name="classifier-cache", fallback_used=False, reason="cache_hit")
        else:
            rule_intent, rule_confidence, rule_labels, rule_reason, ambiguous = self._classify_with_rules(message)
            state.metadata["intent_rule_ambiguous"] = ambiguous
            state.metadata["intent_rule_reason"] = rule_reason

            should_use_llm = self.use_llm_for_ambiguous and rule_confidence < self.llm_trigger_threshold
            state.metadata["intent_rule_confidence"] = rule_confidence

            if not should_use_llm:
                intent, confidence, labels = rule_intent, rule_confidence, rule_labels
                state.metadata["intent_classification_mode"] = "rules"
                state.metadata["intent_classification_reason"] = rule_reason
                state.metadata["intent_used_llm"] = False
                logger.info(
                    "intent_used_llm",
                    extra={"used_llm": False, "reason": rule_reason, "confidence": rule_confidence},
                )
                model_selection = ModelSelection(model_name="classifier-rules", fallback_used=False, reason="deterministic_signals")
            else:
                llm_result = self._classify_with_llm(state)
                if llm_result is not None:
                    intent, confidence, labels = llm_result
                    state.metadata["intent_classification_mode"] = "llm_disambiguation"
                    state.metadata["intent_classification_reason"] = "low_rule_confidence"
                    state.metadata["intent_used_llm"] = True
                    logger.info(
                        "intent_used_llm",
                        extra={"used_llm": True, "reason": "low_rule_confidence", "confidence": rule_confidence},
                    )
                    model_selection = ModelSelection(model_name=self.llm_model, fallback_used=False, reason="llm_disambiguation")
                else:
                    intent, confidence, labels = rule_intent, max(0.55, rule_confidence - 0.1), rule_labels
                    state.metadata["intent_classification_mode"] = "rules_fallback"
                    state.metadata["intent_classification_reason"] = f"llm_unavailable:{rule_reason}"
                    state.metadata["intent_used_llm"] = False
                    logger.info(
                        "intent_used_llm",
                        extra={"used_llm": False, "reason": "llm_unavailable", "confidence": rule_confidence},
                    )
                    model_selection = ModelSelection(model_name="classifier-rules", fallback_used=True, reason="llm_unavailable")

            self._set_cached(cache_key, (intent, confidence, labels, state.metadata["intent_classification_reason"]))

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
            model_selection=model_selection,
        )
        state.metadata["intent_classified_at"] = start
        state.metadata["intent_classification_duration_ms"] = int((perf_counter() - start) * 1000)
        return NodeResult(state=state)

    def _get_cached(self, key: str) -> tuple[str, float, list[str], str] | None:
        cached = self._intent_cache.get(key)
        if cached is None:
            return None
        written_at, value = cached
        if perf_counter() - written_at > self.cache_ttl_seconds:
            self._intent_cache.pop(key, None)
            return None
        return value

    def _set_cached(self, key: str, value: tuple[str, float, list[str], str]) -> None:
        self._intent_cache[key] = (perf_counter(), value)

    def _normalize(self, message: str) -> str:
        return re.sub(r"\s+", " ", message.strip().lower())

    def _classify_with_rules(self, message: str) -> tuple[str, float, list[str], str, bool]:
        text = self._normalize(message)
        is_question = "?" in text or text.startswith(("what", "who", "how", "when", "where", "why", "can", "could", "do", "does", "is "))

        scores = {
            "calculation": 0.0,
            "grounded_qa": 0.0,
            "memory_write": 0.0,
            "general_qa": 0.35,
        }
        reasons: dict[str, list[str]] = {k: [] for k in scores}

        if re.search(r"\b\d+\s*[\+\-\*/%]\s*\d+\b", text) or any(
            token in text for token in ["calculate", "sum", "add", "subtract", "multiply", "divide", "equation"]
        ):
            scores["calculation"] += 3.0
            reasons["calculation"].append("math_pattern")

        if any(
            token in text
            for token in ["document", "docs", "file", "policy", "manual", "warranty", "citation", "source", "according to"]
        ):
            scores["grounded_qa"] += 3.0
            reasons["grounded_qa"].append("document_reference")

        if any(token in text for token in ["remember that", "save this", "note that", "store this"]):
            scores["memory_write"] += 3.2
            reasons["memory_write"].append("explicit_memory_write")

        if not is_question and re.search(
            r"\b(my name is|i am\s+\d+|i work at|i prefer|my favorite|my favourite|my risk tolerance is)\b",
            text,
        ):
            scores["memory_write"] += 2.4
            reasons["memory_write"].append("first_person_fact_statement")

        if any(token in text for token in ["what is my", "who am i", "how old am i"]):
            scores["general_qa"] += 1.2
            reasons["general_qa"].append("memory_query_not_write")

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_intent, top_score = ordered[0]
        second_score = ordered[1][1]
        ambiguous = (top_score - second_score) < 0.8

        if top_intent == "calculation":
            labels = ["tool_needed", "math"]
        elif top_intent == "grounded_qa":
            labels = ["retrieval_needed"]
        elif top_intent == "memory_write":
            labels = ["memory_write"]
        else:
            labels = ["answer_only"]

        confidence = min(0.89, 0.55 + (top_score * 0.1) - (0.12 if ambiguous else 0.0))
        confidence = max(0.5, confidence)
        reason = ",".join(reasons[top_intent]) if reasons[top_intent] else "default_general"
        return top_intent, confidence, labels, reason, ambiguous

    def _classify_with_llm(self, state: OrchestrationState) -> tuple[str, float, list[str]] | None:
        if not llm_available():
            return None

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
            "timeout_seconds": self.llm_timeout_seconds,
            "system_chars": len(system_prompt),
            "user_chars": len(user_prompt),
            "prompt_tokens_estimate": int((len(system_prompt) + len(user_prompt)) / 4),
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
                invoke_start = perf_counter()
                response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
                state.metadata["intent_llm_invoke_ms"] = int((perf_counter() - invoke_start) * 1000)
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
        intent, _confidence, labels, _reason, _ambiguous = self._classify_with_rules(message)
        return intent, labels

    def _is_explicit_document_query(self, message: str) -> bool:
        text = message.lower().strip()
        return any(token in text for token in ["document", "docs", "file", "policy", "manual", "warranty", "citation", "source"])
