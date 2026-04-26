from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from app.models.memory import MemorySnapshot, SessionSummary, SessionTurn
from app.models.orchestration import OrchestrationState
from app.models.retrieval import RetrievalContext
from app.models.tools import ToolExecutionSummary


MODEL_BUDGETS: dict[str, int] = {
    "primary": 128000,
    "fallback": 8000,
}


DEFAULT_ALLOCATION: dict[str, int] = {
    "system_prompt": 1200,
    "l3_user_facts": 2500,
    "l2_summary": 2000,
    "recent_turns": 6000,
    "retrieved_docs": 14000,
    "tools": 1000,
    "current_turn": 500,
    "response_headroom": 8000,
}


def estimate_tokens(text: str) -> int:
    normalized = text or ""
    if not normalized.strip():
        return 0
    try:
        import tiktoken  # type: ignore

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(normalized))
    except Exception:
        return max(1, int(len(normalized) / 4))


@dataclass(slots=True)
class ContextBuildResult:
    scoped_state: OrchestrationState
    audit: dict[str, Any]


class ContextBuilder:
    def __init__(
        self,
        *,
        model_budgets: dict[str, int] | None = None,
        default_allocation: dict[str, int] | None = None,
    ) -> None:
        self.model_budgets = dict(model_budgets or MODEL_BUDGETS)
        self.default_allocation = dict(default_allocation or DEFAULT_ALLOCATION)

    def build(
        self,
        *,
        state: OrchestrationState,
        system_prompt: str,
        include_memory_context: bool,
        model_tier: str,
    ) -> ContextBuildResult:
        budget_total = int(self.model_budgets.get(model_tier, self.model_budgets["fallback"]))
        allocation = dict(self.default_allocation)

        sections = {
            "system_prompt": system_prompt,
            "current_turn": state.turn.message,
            "l2_summary": "",
            "l3_user_facts": {
                "protected": [],
                "user": [],
            },
            "recent_turns": [],
            "retrieved_docs": [],
            "tools": [],
        }

        protected_fact_rows: list[tuple[Any, str]] = []
        user_fact_rows: list[tuple[Any, str]] = []
        summary_text = ""
        session_turns: list[SessionTurn] = []
        if include_memory_context and state.memory_snapshot is not None:
            if state.memory_snapshot.summary is not None:
                summary_text = state.memory_snapshot.summary.summary.strip()
                protected_fact_rows = [
                    (fact, f"{fact.canonical_key}: {fact.value.strip()}")
                    for fact in state.memory_snapshot.summary.protected_facts
                    if fact.value.strip()
                ]
            user_fact_rows = [
                (fact, f"{fact.canonical_key}: {fact.value.strip()}")
                for fact in state.memory_snapshot.user_facts
                if getattr(fact, "status", None) and fact.status.value == "active" and fact.value.strip()
            ]
            session_turns = list(state.memory_snapshot.session_turns)

        if include_memory_context:
            sections["l2_summary"] = self._truncate_to_budget(summary_text, allocation["l2_summary"])
            selected_protected, selected_user = self._select_facts_with_budget(
                protected_fact_rows=protected_fact_rows,
                user_fact_rows=user_fact_rows,
                token_budget=allocation["l3_user_facts"],
            )
            sections["l3_user_facts"] = {
                "protected": selected_protected,
                "user": selected_user,
            }
            sections["recent_turns"] = self._select_turns_with_budget(
                session_turns,
                allocation["recent_turns"],
            )

        if state.retrieval_context is not None and state.retrieval_context.chunks:
            ranked_chunks = sorted(state.retrieval_context.chunks, key=lambda chunk: chunk.score, reverse=True)
            sections["retrieved_docs"] = self._select_docs_with_budget(ranked_chunks, allocation["retrieved_docs"])

        if state.tool_results is not None and state.tool_results.results:
            sections["tools"] = self._select_tools_with_budget(state.tool_results.results, allocation["tools"])

        events: list[str] = []
        context_budget = max(0, budget_total - allocation["response_headroom"])

        while self._context_tokens(sections) > context_budget:
            before_tokens = self._context_tokens(sections)

            if sections["l2_summary"]:
                compressed = self._compress_summary(str(sections["l2_summary"]))
                if compressed != sections["l2_summary"]:
                    sections["l2_summary"] = compressed
                    events.append("Compressed L2 summary")

            if self._context_tokens(sections) > context_budget and sections["recent_turns"]:
                turns = sections["recent_turns"]
                if isinstance(turns, list) and turns:
                    turns.pop(0)
                    events.append("Dropped 1 oldest turn")

            if self._context_tokens(sections) > context_budget and sections["retrieved_docs"]:
                docs = sections["retrieved_docs"]
                if isinstance(docs, list) and docs:
                    before_count = len(docs)
                    docs.pop()
                    events.append(f"Reduced RAG chunks from {before_count} -> {len(docs)}")

            if self._context_tokens(sections) > context_budget and sections["l3_user_facts"]:
                facts = sections["l3_user_facts"]
                if isinstance(facts, dict):
                    protected = facts.get("protected", [])
                    user = facts.get("user", [])
                    if isinstance(user, list) and user:
                        user.pop()
                        events.append("Trimmed 1 L3 user fact")
                    elif isinstance(protected, list) and protected:
                        protected.pop()
                        events.append("Trimmed 1 protected fact")

            if self._context_tokens(sections) > context_budget and sections["tools"]:
                tool_rows = sections["tools"]
                if isinstance(tool_rows, list) and tool_rows:
                    tool_rows.pop(0)
                    events.append("Evicted 1 tool output")

            if self._context_tokens(sections) >= before_tokens:
                events.append("Budget pressure remains with non-evictable content")
                break

        allocation_tokens = {
            "system_prompt": estimate_tokens(str(sections["system_prompt"])),
            "l3_user_facts": self._l3_tokens(sections["l3_user_facts"]),
            "l2_summary": estimate_tokens(str(sections["l2_summary"])),
            "retrieved_docs": self._list_tokens(sections["retrieved_docs"]),
            "recent_turns": self._turn_tokens(sections["recent_turns"]),
            "tools": self._list_tokens(sections["tools"]),
            "current_user_turn": estimate_tokens(str(sections["current_turn"])),
            "response_headroom": allocation["response_headroom"],
        }
        used = sum(allocation_tokens.values())
        unused = max(0, budget_total - used)
        allocation_tokens["unused"] = unused

        scoped_state = self._build_scoped_state(state=state, sections=sections)
        audit = {
            "model": f"{model_tier}-{budget_total}",
            "budget_total": budget_total,
            "allocation": {
                key: {
                    "tokens": value,
                    "pct": round((value / budget_total) * 100.0, 2) if budget_total > 0 else 0.0,
                }
                for key, value in allocation_tokens.items()
            },
            "eviction_events": events,
        }
        return ContextBuildResult(scoped_state=scoped_state, audit=audit)

    def _build_scoped_state(self, *, state: OrchestrationState, sections: dict[str, Any]) -> OrchestrationState:
        scoped_snapshot = state.memory_snapshot
        if state.memory_snapshot is not None:
            l3_facts = sections["l3_user_facts"] if isinstance(sections["l3_user_facts"], dict) else {}
            selected_protected = l3_facts.get("protected", []) if isinstance(l3_facts, dict) else []
            selected_user = l3_facts.get("user", []) if isinstance(l3_facts, dict) else []
            summary = state.memory_snapshot.summary
            if summary is not None:
                scoped_summary = SessionSummary(
                    session_id=summary.session_id,
                    user_id=summary.user_id,
                    summary=str(sections["l2_summary"]),
                    pinned_facts=list(summary.pinned_facts),
                    protected_facts=list(selected_protected),
                    turn_count=summary.turn_count,
                    updated_at=summary.updated_at,
                )
            else:
                scoped_summary = None

            scoped_snapshot = MemorySnapshot(
                session_turns=list(sections["recent_turns"]),
                summary=scoped_summary,
                user_facts=list(selected_user),
                pinned_facts=list(state.memory_snapshot.pinned_facts),
                corpus_chunks=list(state.memory_snapshot.corpus_chunks),
                budget_audit=state.memory_snapshot.budget_audit,
            )

        scoped_retrieval = state.retrieval_context
        if state.retrieval_context is not None:
            scoped_retrieval = RetrievalContext(
                query=state.retrieval_context.query,
                chunks=list(sections["retrieved_docs"]),
                used_keyword_fallback=state.retrieval_context.used_keyword_fallback,
                empty=not bool(sections["retrieved_docs"]),
            )

        scoped_tools = state.tool_results
        if state.tool_results is not None:
            scoped_tools = ToolExecutionSummary(
                requested=len(sections["tools"]),
                succeeded=sum(1 for item in sections["tools"] if item.success),
                failed=sum(1 for item in sections["tools"] if not item.success),
                results=list(sections["tools"]),
            )

        return OrchestrationState(
            turn=state.turn,
            trace=state.trace,
            memory_snapshot=scoped_snapshot,
            retrieval_context=scoped_retrieval,
            retrieval_audit=state.retrieval_audit,
            intent=state.intent,
            tool_plan=state.tool_plan,
            tool_results=scoped_tools,
            response=state.response,
            memory_budget_audit=state.memory_budget_audit,
            errors=list(state.errors),
            metadata=dict(state.metadata),
        )

    def _select_facts_with_budget(self, *, protected_fact_rows: list[tuple[Any, str]], user_fact_rows: list[tuple[Any, str]], token_budget: int) -> tuple[list[Any], list[Any]]:
        selected_protected: list[Any] = []
        selected_user: list[Any] = []
        spent = 0
        for fact, line in protected_fact_rows:
            line_tokens = estimate_tokens(line)
            if spent + line_tokens > token_budget:
                break
            selected_protected.append(fact)
            spent += line_tokens

        for fact, line in user_fact_rows:
            line_tokens = estimate_tokens(line)
            if spent + line_tokens > token_budget:
                break
            selected_user.append(fact)
            spent += line_tokens

        return selected_protected, selected_user

    def _select_turns_with_budget(self, turns: list[SessionTurn], token_budget: int) -> list[SessionTurn]:
        if token_budget <= 0:
            return []
        selected_reversed: list[SessionTurn] = []
        spent = 0
        for turn in reversed(turns):
            turn_text = f"{turn.role}: {turn.content}"
            turn_tokens = estimate_tokens(turn_text)
            if spent + turn_tokens > token_budget:
                continue
            selected_reversed.append(turn)
            spent += turn_tokens
        return list(reversed(selected_reversed))

    def _select_docs_with_budget(self, chunks: list[Any], token_budget: int) -> list[Any]:
        selected: list[Any] = []
        spent = 0
        for chunk in chunks:
            row = chunk.content.strip()
            row_tokens = estimate_tokens(row)
            if spent + row_tokens > token_budget:
                continue
            selected.append(chunk)
            spent += row_tokens
        return selected

    def _select_tools_with_budget(self, results: list[Any], token_budget: int) -> list[Any]:
        selected: list[Any] = []
        spent = 0
        for result in results:
            row = self._tool_result_text(result)
            row_tokens = estimate_tokens(row)
            if spent + row_tokens > token_budget:
                continue
            selected.append(result)
            spent += row_tokens
        return selected

    def _truncate_to_budget(self, text: str, token_budget: int) -> str:
        if token_budget <= 0:
            return ""
        normalized = text.strip()
        if estimate_tokens(normalized) <= token_budget:
            return normalized
        low = 0
        high = len(normalized)
        best = ""
        while low <= high:
            mid = (low + high) // 2
            candidate = normalized[:mid].rstrip()
            if estimate_tokens(candidate) <= token_budget:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1
        return best

    def _compress_summary(self, summary: str) -> str:
        normalized = re.sub(r"\s+", " ", summary.strip())
        if len(normalized) <= 80:
            return normalized
        midpoint = max(40, len(normalized) // 2)
        compressed = normalized[:midpoint].rstrip()
        return compressed

    def _context_tokens(self, sections: dict[str, Any]) -> int:
        return (
            estimate_tokens(str(sections["system_prompt"]))
            + self._l3_tokens(sections["l3_user_facts"])
            + estimate_tokens(str(sections["l2_summary"]))
            + self._list_tokens(sections["retrieved_docs"])
            + self._turn_tokens(sections["recent_turns"])
            + self._list_tokens(sections["tools"])
            + estimate_tokens(str(sections["current_turn"]))
        )

    def _list_tokens(self, value: Any) -> int:
        if not isinstance(value, list):
            return 0
        total = 0
        for item in value:
            if hasattr(item, "tool_name"):
                total += estimate_tokens(self._tool_result_text(item))
                continue
            if hasattr(item, "content"):
                total += estimate_tokens(str(item.content))
            else:
                total += estimate_tokens(str(item))
        return total

    def _l3_tokens(self, value: Any) -> int:
        if not isinstance(value, dict):
            return 0
        protected = value.get("protected", [])
        user = value.get("user", [])
        return self._list_tokens(protected) + self._list_tokens(user)

    def _turn_tokens(self, turns: Any) -> int:
        if not isinstance(turns, list):
            return 0
        return sum(estimate_tokens(f"{turn.role}: {turn.content}") for turn in turns)

    def _tool_result_text(self, result: Any) -> str:
        if not getattr(result, "success", False):
            return f"{getattr(result, 'tool_name', 'tool')}: failed {getattr(result, 'error', '')}"
        output = getattr(result, "output", None)
        if isinstance(output, (dict, list)):
            return f"{getattr(result, 'tool_name', 'tool')}: {json.dumps(output, default=str)[:700]}"
        return f"{getattr(result, 'tool_name', 'tool')}: {str(output)[:700]}"