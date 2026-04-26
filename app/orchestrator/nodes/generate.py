from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
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
from app.llm.prompts import build_response_prompt_bundle
from app.models.memory import FactStatus
from app.models.orchestration import GeneratedResponse, ModelSelection, OrchestrationState
from app.orchestrator.context_builder import ContextBuilder
from app.orchestrator.nodes.base import NodeResult
from app.retrieval.embeddings import EmbeddingProvider, HashEmbeddingProvider, cosine_similarity
from app.retrieval.embeddings_hf import HFEmbeddingProvider


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_memory_relevance_embedder() -> EmbeddingProvider:
    try:
        return HFEmbeddingProvider()
    except Exception:
        logger.warning("memory_relevance_embedding_provider_fallback", exc_info=True)
        return HashEmbeddingProvider()



def _chunk_memory_context(memory_context: str) -> list[str]:
    chunks: list[str] = []
    for raw_line in re.split(r"[\r\n]+", memory_context.strip()):
        line = raw_line.strip().lstrip("- ").strip()
        if not line:
            continue
        for piece in re.split(r"(?<=[.;!?])\s+", line):
            candidate = piece.strip()
            if len(candidate) >= 6:
                chunks.append(candidate)
    return chunks


def memory_relevance_score(
    query: str,
    memory_context: str,
    *,
    threshold: float = 0.58,
    embedding_provider: EmbeddingProvider | None = None,
) -> tuple[bool, float, str | None]:
    normalized_query = query.strip()
    if not normalized_query:
        return False, 0.0, None

    chunks = _chunk_memory_context(memory_context)
    if not chunks:
        return False, 0.0, None

    embedder = embedding_provider or _get_memory_relevance_embedder()
    query_embedding = embedder.embed_text(normalized_query)
    if not query_embedding:
        return False, 0.0, None

    max_similarity = 0.0
    best_chunk: str | None = None
    chunk_embeddings = embedder.embed_texts(chunks)
    for chunk, chunk_embedding in zip(chunks, chunk_embeddings):
        similarity = cosine_similarity(query_embedding, chunk_embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            best_chunk = chunk

    return max_similarity >= threshold, max_similarity, best_chunk


def is_memory_relevant(query: str, memory_context: str, threshold: float = 0.58) -> bool:
    relevant, _, _ = memory_relevance_score(query, memory_context, threshold=threshold)
    return relevant


@dataclass(slots=True)
class ResponseGenerationNode:
    name: str = "response_generation"
    llm_temperature: float = 0.2
    llm_model: str = "gemini-flash-lite-latest"
    llm_timeout_seconds: float = 4.0
    llm_invoke_retries: int = 1
    llm_fallback_models: tuple[str, ...] = ()
    memory_relevance_threshold: float = 0.58
    context_builder: ContextBuilder = field(default_factory=ContextBuilder)

    def run(self, state: OrchestrationState) -> NodeResult:
        sources: list[dict] = []
        used_llm = False
        memory_context_text = self._build_memory_context_text(state)
        memory_relevant, memory_similarity, matched_chunk = memory_relevance_score(
            state.turn.message,
            memory_context_text,
            threshold=self.memory_relevance_threshold,
        )
        if state.intent and state.intent.intent == "memory_write":
            memory_relevant = False

        state.metadata["response_memory_relevant"] = memory_relevant
        state.metadata["response_memory_max_similarity"] = round(memory_similarity, 4)
        state.metadata["response_memory_similarity_threshold"] = self.memory_relevance_threshold
        state.metadata["response_memory_matched_chunk"] = matched_chunk or ""
        state.metadata["response_memory_context_chars"] = len(memory_context_text)
        logger.info(
            "memory_relevance_decision",
            extra={
                "query": state.turn.message,
                "max_similarity": round(memory_similarity, 4),
                "threshold": self.memory_relevance_threshold,
                "memory_used": memory_relevant,
            },
        )

        # Collect all context sources for LLM
        if state.tool_results:
            for result in state.tool_results.results:
                if result.success:
                    sources.append({"type": "tool", "name": result.tool_name, "latency_ms": result.latency_ms})
                else:
                    sources.append({"type": "tool", "name": result.tool_name, "failed": True, "error": result.error})

        if state.retrieval_context and state.retrieval_context.chunks:
            for chunk in state.retrieval_context.chunks:
                sources.append(
                    {
                        "type": "document",
                        "document_id": chunk.document_id,
                        "chunk_id": chunk.chunk_id,
                        "score": chunk.score,
                    }
                )

        if memory_relevant and state.memory_snapshot and state.memory_snapshot.summary and state.memory_snapshot.summary.summary:
            sources.append({"type": "session_memory", "session_id": state.turn.session_id})

        # Check for memory-answerable questions and add to context
        memory_answer: str | None = None
        memory_source: str | None = None
        if memory_relevant:
            memory_answer, memory_source = self._answer_from_memory(state)
        if memory_answer is not None:
            source_type = memory_source or "session_memory"
            if not any(item.get("type") == source_type for item in sources):
                sources.append({"type": source_type, "user_id": state.turn.user_id})
            state.metadata["response_memory_match"] = memory_answer
            state.metadata["response_memory_source"] = memory_source

        system_instruction_preview = build_response_prompt_bundle(
            state,
            memory_answer=None,
            include_memory_context=memory_relevant,
        ).system_instruction
        model_tier = "primary"
        if "fallback" in self.llm_model.lower() or "8" in self.llm_model.lower():
            model_tier = "fallback"
        context_result = self.context_builder.build(
            state=state,
            system_prompt=system_instruction_preview,
            include_memory_context=memory_relevant,
            model_tier=model_tier,
        )
        scoped_state = context_result.scoped_state
        state.metadata["context_budget_audit"] = context_result.audit
        state.metadata["context_eviction_events"] = list(context_result.audit.get("eviction_events", []))
        state.metadata["context_budget_model_tier"] = model_tier

        # ALWAYS route through LLM. Fallback is only allowed when LLM call fails.
        llm_answer = self._generate_with_llm(scoped_state, memory_answer=memory_answer, include_memory_context=memory_relevant)
        
        if llm_answer:
            final_answer = llm_answer
            used_llm = True
        else:
            # Only fallback if LLM completely fails
            final_answer = self._heuristic_response_fallback(state, memory_answer=memory_answer)
            used_llm = False

        state.metadata["response_used_llm"] = used_llm
        state.metadata["llm_status"] = "success" if used_llm else "failed"
        state.metadata["response_generation_mode"] = "llm" if used_llm else "local"
        if not used_llm and "response_llm_error" in state.metadata:
            state.metadata["response_fallback_reason"] = "llm_error"
            state.metadata["fallback_reason"] = str(state.metadata.get("response_llm_error"))
        elif not used_llm:
            state.metadata["fallback_reason"] = "llm_no_content"
        else:
            state.metadata["fallback_reason"] = ""

        state.response = GeneratedResponse(
            final_answer=final_answer,
            sources=sources,
            model_selection=ModelSelection(
                model_name=self.llm_model if used_llm else "response-heuristic",
                fallback_used=not used_llm,
                reason="llm_generator" if used_llm else "local_generator",
            ),
            incomplete=False,
        )
        return NodeResult(state=state)

    def _generate_with_llm(self, state: OrchestrationState, memory_answer: str | None = None, include_memory_context: bool = True) -> str | None:
        prompt = build_response_prompt_bundle(state, memory_answer=memory_answer, include_memory_context=include_memory_context)
        logger.info(
            "LLM_CALL_STARTED",
            extra={
                "node": self.name,
                "model": self.llm_model,
                "timeout_seconds": self.llm_timeout_seconds,
                "system_chars": len(prompt.system_instruction),
                "user_chars": len(prompt.user_prompt),
            },
        )
        state.metadata["response_llm_request"] = {
            "primary_model": self.llm_model,
            "candidate_models": build_model_candidates(self.llm_model, self.llm_fallback_models),
            "temperature": self.llm_temperature,
            "timeout_seconds": self.llm_timeout_seconds,
            "system_chars": len(prompt.system_instruction),
            "user_chars": len(prompt.user_prompt),
            "prompt_tokens_estimate": int((len(prompt.system_instruction) + len(prompt.user_prompt)) / 4),
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
                    "response_generation_llm_client_init_failed",
                    extra={"model": model_name, "error": str(exc), "error_type": type(exc).__name__},
                )
                continue

            response = None
            for invoke_attempt in range(self.llm_invoke_retries + 1):
                try:
                    invoke_start = perf_counter()
                    response = llm.invoke(
                        [
                            SystemMessage(content=prompt.system_instruction),
                            HumanMessage(content=prompt.user_prompt),
                        ]
                    )
                    state.metadata["response_llm_invoke_ms"] = int((perf_counter() - invoke_start) * 1000)
                    break
                except Exception as exc:
                    mark_llm_failure()
                    error_category, is_quota_error = classify_llm_error(exc)
                    attempt_errors.append(
                        {
                            "stage": "invoke",
                            "model": model_name,
                            "invoke_attempt": invoke_attempt,
                            "error_type": type(exc).__name__,
                            "error_category": error_category,
                            "quota_exceeded": is_quota_error,
                            "error": str(exc),
                        }
                    )
                    logger.warning(
                        "response_generation_llm_invoke_failed",
                        extra={
                            "model": model_name,
                            "invoke_attempt": invoke_attempt,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    )
            if response is None:
                continue

            content = getattr(response, "content", "")
            answer = self._coerce_text(content)
            if not answer:
                attempt_errors.append(
                    {
                        "stage": "invoke",
                        "model": model_name,
                        "error_type": "EmptyResponse",
                        "error_category": "empty_response",
                        "quota_exceeded": False,
                        "error": "Model returned empty content",
                    }
                )
                continue

            if state.intent is not None and state.intent.intent != "grounded_qa":
                answer = re.sub(r"^No relevant information found\.?\s*", "", answer, flags=re.IGNORECASE).strip()
                if not answer:
                    continue

            mark_llm_success()
            logger.info(
                "LLM_CALL_SUCCESS",
                extra={
                    "node": self.name,
                    "model": model_name,
                    "invoke_ms": state.metadata.get("response_llm_invoke_ms"),
                    "invoke_retries": self.llm_invoke_retries,
                },
            )
            state.metadata["response_llm_model"] = model_name
            state.metadata["response_llm_attempts"] = len(attempt_errors) + 1
            return answer

        if attempt_errors:
            last = attempt_errors[-1]
            logger.warning(
                "LLM_CALL_FAILED",
                extra={
                    "node": self.name,
                    "model": last.get("model"),
                    "error_type": last.get("error_type"),
                    "error_category": last.get("error_category"),
                    "error": last.get("error"),
                },
            )
            state.metadata["response_llm_error"] = f"{last.get('error_type')}: {last.get('error')}"
            state.metadata["response_llm_error_type"] = str(last.get("error_type", "unknown"))
            state.metadata["response_llm_error_category"] = str(last.get("error_category", "unknown"))
            state.metadata["response_llm_quota_exceeded"] = bool(last.get("quota_exceeded", False))
            state.metadata["response_llm_attempt_errors"] = attempt_errors
            state.metadata["response_llm_attempts"] = len(attempt_errors)

        return None

    def _build_memory_context_text(self, state: OrchestrationState) -> str:
        if state.memory_snapshot is None:
            return ""

        parts: list[str] = []
        if state.memory_snapshot.summary and state.memory_snapshot.summary.summary.strip():
            parts.append(state.memory_snapshot.summary.summary.strip())
        if state.memory_snapshot.summary:
            for fact in state.memory_snapshot.summary.protected_facts:
                if fact.value.strip():
                    parts.append(f"{fact.canonical_key} {fact.value.strip()}")
        for fact in state.memory_snapshot.user_facts:
            if fact.status == FactStatus.ACTIVE and fact.value.strip():
                parts.append(f"{fact.canonical_key} {fact.value.strip()}")
        for fact in state.memory_snapshot.pinned_facts:
            if fact.value.strip():
                parts.append(f"{fact.canonical_key} {fact.value.strip()}")
        return "\n".join(parts)

    def _heuristic_response_fallback(self, state: OrchestrationState, memory_answer: str | None = None) -> str:
        # If we have a direct memory answer and LLM failed, return it
        if memory_answer:
            return memory_answer
        
        message = state.turn.message.strip().lower()
        if "investment" not in message and "strategy" not in message:
            return state.turn.message

        risk = None
        age = None
        if state.memory_snapshot:
            for fact in state.memory_snapshot.user_facts:
                if fact.status != FactStatus.ACTIVE:
                    continue
                if fact.canonical_key == "risk_tolerance" and fact.value.strip() and risk is None:
                    risk = fact.value.strip().lower()
                if fact.canonical_key == "age" and fact.value.strip() and age is None:
                    age = fact.value.strip()

        age_clause = f" at age {age}" if age else ""

        if risk == "low":
            return (
                f"Given your low risk tolerance{age_clause}, consider a conservative strategy: "
                "focus on capital preservation with high-quality bonds, diversified index exposure, and regular rebalancing. "
                "Keep emergency cash and avoid concentrated high-volatility positions."
            )
        if risk in {"medium", "moderate"}:
            return "A balanced strategy may fit: mix diversified equity index funds with bonds, and rebalance periodically based on your goals and horizon."
        if risk in {"high", "aggressive"}:
            return "An aggressive strategy may fit: emphasize diversified equity growth assets with disciplined risk controls and periodic portfolio review."

        if age:
            return (
                f"Given your age ({age}), a long-term diversified strategy can be a strong starting point: "
                "use broad index funds as a core, invest regularly, and keep adequate emergency savings. "
                "To personalize further, share your risk tolerance and time horizon."
            )

        return "A useful starting point is a diversified long-term plan using broad index funds and regular investing. Share your risk tolerance and time horizon so I can personalize it."

    def _coerce_text(self, content: object) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                elif isinstance(item, dict):
                    possible = item.get("text")
                    if isinstance(possible, str) and possible.strip():
                        parts.append(possible.strip())
                else:
                    raw = str(item).strip()
                    if raw:
                        parts.append(raw)
            return "\n".join(parts).strip()
        return str(content).strip()

    def _build_retrieval_refusal(self, state: OrchestrationState) -> str | None:
        if state.intent is None or state.intent.intent != "grounded_qa":
            return None
        if state.retrieval_context is None:
            return "No relevant information found in the available documents."
        if state.retrieval_context.empty or not state.retrieval_context.chunks:
            return "No relevant information found in the available documents."
        best_score = max((chunk.score for chunk in state.retrieval_context.chunks), default=0.0)
        if best_score < 0.2:
            return "No relevant information found in the available documents."
        return None

    def _format_tool_output(self, tool_name: str, output: object) -> str:
        if tool_name == "retrieval_tool" and isinstance(output, dict):
            chunks = output.get("chunks") or []
            if not chunks:
                return "I did not find relevant document context."
            top = chunks[0]
            content = str(top.get("content", "")).strip()
            return content or "I found relevant context, but no readable content was returned."
        if isinstance(output, dict) and "result" in output:
            return str(output["result"])
        return str(output)

    def _answer_from_memory(self, state: OrchestrationState) -> tuple[str | None, str | None]:
        message = state.turn.message.strip().lower()
        if state.memory_snapshot is None:
            return None, None

        if "what is my name" in message or "who am i" in message:
            answer = self._answer_name_from_memory(state)
            if answer is not None:
                return answer, "user_memory"

        if self._is_risk_tolerance_question(message):
            answer = self._answer_risk_tolerance_from_memory(state)
            if answer is not None:
                return answer, "user_memory"

        if self._is_age_question(message):
            answer = self._answer_age_from_memory(state)
            if answer is not None:
                return answer, "user_memory"

        answer = self._answer_from_session_turns(state, message)
        if answer is not None:
            return answer, "session_memory"

        return None, None

    def _answer_from_session_turns(self, state: OrchestrationState, message: str) -> str | None:
        if state.memory_snapshot is None:
            return None

        user_turns = [turn.content.strip() for turn in state.memory_snapshot.session_turns if turn.role == "user" and turn.content.strip()]
        if not user_turns:
            return None

        if "what did i just say" in message or "what did i tell you" in message:
            return user_turns[-1]

        attr_match = re.search(r"\bwhat\s+is\s+my\s+([a-z][a-z\s]{0,30})\??$", message)
        if attr_match:
            attr = re.sub(r"\s+", " ", attr_match.group(1)).strip()
            pattern = re.compile(rf"\bmy\s+{re.escape(attr)}\s+is\s+(.+)", re.IGNORECASE)
            for previous in reversed(user_turns):
                m = pattern.search(previous)
                if m:
                    return m.group(1).strip(" .,!?:;\"'")

        return None

    def _answer_name_from_memory(self, state: OrchestrationState) -> str | None:
        if state.memory_snapshot is None:
            return None

        # Prefer canonical name fact if present.
        for fact in state.memory_snapshot.user_facts:
            if fact.status != FactStatus.ACTIVE:
                continue
            if fact.canonical_key == "name" and fact.value.strip():
                return fact.value.strip()

        # Backward-compatible fallback for earlier stored facts like
        for fact in state.memory_snapshot.user_facts:
            if fact.status != FactStatus.ACTIVE:
                continue
            if fact.canonical_key in {"self_identifier", "name"}:
                extracted = self._extract_name(fact.value)
                if extracted:
                    return extracted

        return None

    def _answer_risk_tolerance_from_memory(self, state: OrchestrationState) -> str | None:
        if state.memory_snapshot is None:
            return None

        for fact in state.memory_snapshot.user_facts:
            if fact.status != FactStatus.ACTIVE:
                continue
            if fact.canonical_key == "risk_tolerance" and fact.value.strip():
                return fact.value.strip()

        for fact in state.memory_snapshot.user_facts:
            if fact.status != FactStatus.ACTIVE:
                continue
            if fact.canonical_key in {"preference", "self_identifier"}:
                extracted = self._extract_risk_tolerance(fact.value)
                if extracted:
                    return extracted

        return None

    def _extract_name(self, text: str) -> str | None:
        match = re.search(
            r"\bmy\s+name\s+is\s+([A-Za-z][A-Za-z\s'.-]{0,80}?)(?=\s+(?:and\b|but\b|my\b)|[,.!?;]|$)",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        return match.group(1).strip(" .,!?:;\"'")

    def _is_risk_tolerance_question(self, message: str) -> bool:
        normalized = re.sub(r"\s+", " ", message.strip().lower())
        return any(
            phrase in normalized
            for phrase in [
                "risk tolerance",
                "risk tolerence",
                "risk tollerance",
                "risk profile",
                "tolerance risk",
                "tolerence risk",
            ]
        )

    def _extract_risk_tolerance(self, text: str) -> str | None:
        level_pattern = r"(low|medium|high|conservative|moderate|aggressive)"
        risk_term_pattern = r"(?:tolerance|tolerence|tollerance|profile)"
        normalized = re.sub(r"\s+", " ", text.strip().lower())

        patterns = [
            rf"\brisk\s+{risk_term_pattern}\s*(?:is|=|:)?\s*{level_pattern}\b",
            rf"\b{risk_term_pattern}\s+risk\s*(?:is|=|:)?\s*{level_pattern}\b",
            rf"\b(?:i\s+am|i'm|im)\s+{level_pattern}\s+on\s+risk\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if match:
                return match.group(1).lower()

        return None

    def _is_age_question(self, message: str) -> bool:
        normalized = re.sub(r"\s+", " ", message.strip().lower())
        return any(
            phrase in normalized
            for phrase in [
                "how old am i",
                "how old i am",
                "what is my age",
                "what's my age",
            ]
        )

    def _answer_age_from_memory(self, state: OrchestrationState) -> str | None:
        if state.memory_snapshot is None:
            return None

        for fact in state.memory_snapshot.user_facts:
            if fact.status != FactStatus.ACTIVE:
                continue
            if fact.canonical_key == "age" and fact.value.strip():
                return fact.value.strip()

        for fact in state.memory_snapshot.user_facts:
            if fact.status != FactStatus.ACTIVE:
                continue
            if fact.canonical_key in {"self_identifier", "age"}:
                extracted = self._extract_age(fact.value)
                if extracted is not None:
                    return str(extracted)

        return None

    def _extract_age(self, text: str) -> int | None:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        patterns = [
            r"\bi\s+am\s+(\d{1,3})\s*(?:years?\s+old|yrs?\s+old|yo)?\b",
            r"\bmy\s+age\s+(?:is|=|:)\s*(\d{1,3})\b",
            r"\bage\s+(?:is|=|:)\s*(\d{1,3})\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if not match:
                continue
            age = int(match.group(1))
            if 0 < age < 125:
                return age

        return None
