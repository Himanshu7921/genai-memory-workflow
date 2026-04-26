from __future__ import annotations

from dataclasses import dataclass

from app.models.memory import FactStatus
from app.models.orchestration import OrchestrationState


@dataclass(frozen=True, slots=True)
class PromptBundle:
    system_instruction: str
    user_prompt: str


def build_response_prompt_bundle(state: OrchestrationState, memory_answer: str | None = None, include_memory_context: bool = True) -> PromptBundle:
    summary = ""
    if state.memory_snapshot and state.memory_snapshot.summary:
        summary = state.memory_snapshot.summary.summary.strip()[:500]

    protected_lines: list[str] = []
    user_facts_lines: list[str] = []
    semantic_preference_lines: list[str] = []
    pinned_lines: list[str] = []
    if state.memory_snapshot:
        protected_facts = state.memory_snapshot.summary.protected_facts if state.memory_snapshot.summary else []
        for fact in protected_facts[:40]:
            if fact.value.strip():
                protected_lines.append(f"- {fact.canonical_key}: {fact.value.strip()[:140]}")
        for fact in state.memory_snapshot.user_facts[:8]:
            if fact.status == FactStatus.ACTIVE and fact.value.strip():
                user_facts_lines.append(f"- {fact.canonical_key}: {fact.value.strip()[:140]}")
        for fact in state.memory_snapshot.pinned_facts[:5]:
            if fact.value.strip():
                pinned_lines.append(f"- {fact.canonical_key}: {fact.value.strip()[:140]} ({fact.reason})")
    semantic_user_facts = state.turn.retrieved_memory.get("semantic_user_facts", []) if state.turn.retrieved_memory else []
    for fact in semantic_user_facts[:5]:
        value = str(fact.get("value", "")).strip()
        if not value:
            continue
        if value.lower().startswith("i "):
            formatted = value[0].lower() + value[1:]
        else:
            formatted = value
        semantic_preference_lines.append(f"- {formatted[:140]}")

    document_lines: list[str] = []
    if state.retrieval_context:
        for chunk in state.retrieval_context.chunks[:3]:
            snippet = chunk.content.strip().replace("\n", " ")
            document_lines.append(
                f"- doc={chunk.document_id} chunk={chunk.chunk_id} score={chunk.score:.3f}: {snippet[:220]}"
            )

    tool_lines: list[str] = []
    if state.tool_results:
        for result in state.tool_results.results[:2]:
            if result.success:
                tool_lines.append(f"- {result.tool_name}: {_format_tool_result(result.output)}")
            else:
                tool_lines.append(f"- {result.tool_name}: failed ({result.error})")

    # Strict system prompt: be concise, direct, no fluff, no over-personalization
    system_instruction = (
        "You are a production assistant providing concise, direct answers. "
        "Rules:\n"
        "1. Answer directly in plain language with minimal verbosity.\n"
        "2. Use provided context (memory, docs, tools) only when relevant.\n"
        "3. Do NOT ask follow-up questions unless explicitly required by the user.\n"
        "4. Do NOT add filler text or assistant-style preamble.\n"
        "5. Avoid over-personalization: use neutral phrasing like 'Based on your profile' instead of using user names or assumptions.\n"
        "6. If context is insufficient, state that clearly without asking follow-up questions.\n"
        "7. Do not output raw JSON, Python dicts, internal metadata, or debugging artifacts.\n"
        "8. Keep responses short and actionable.\n"
        "9. Use memory context only when it directly answers the current query; otherwise ignore it.\n"
        "10. For identity-related questions (name, age, who am I), prioritize User Facts (L3) when present."
    )

    memory_context = ""
    if memory_answer:
        memory_context = f"\nDirect Match Found (from memory): {memory_answer}\nFormat this naturally in your response. Use it to answer the query directly."

    user_prompt = (
        "## User Query\n"
        f"{state.turn.message}\n\n"
        "## Memory Context (Protected Facts + L2 + L3)\n"
        f"Protected Facts (L2):\n{chr(10).join(protected_lines) if include_memory_context and protected_lines else ('- none' if include_memory_context else '- omitted due to irrelevance gate')}\n"
        f"Session Summary (L2):\n{summary if include_memory_context and summary else ('none' if include_memory_context else 'omitted due to irrelevance gate')}\n"
        f"User Facts (L3):\n{chr(10).join(user_facts_lines) if include_memory_context and user_facts_lines else ('- none' if include_memory_context else '- omitted due to irrelevance gate')}\n"
        f"[User Preferences]\n{chr(10).join(semantic_preference_lines) if include_memory_context and semantic_preference_lines else ('- none' if include_memory_context else '- omitted due to irrelevance gate')}\n"
        f"Pinned Facts:\n{chr(10).join(pinned_lines) if include_memory_context and pinned_lines else ('- none' if include_memory_context else '- omitted due to irrelevance gate')}\n\n"
        "## Retrieved Documents\n"
        f"{chr(10).join(document_lines) if document_lines else '- none'}\n\n"
        "## Tool Results\n"
        f"{chr(10).join(tool_lines) if tool_lines else '- none'}\n"
        f"{memory_context}\n\n"
        "## Response Instructions\n"
        "- Answer directly and concisely.\n"
        "- Use available context before asking for more information.\n"
        "- Do not repeat questions or ask for missing details.\n"
        "- Keep your answer short and actionable."
    )

    return PromptBundle(system_instruction=system_instruction, user_prompt=user_prompt)


def _format_tool_result(output: object) -> str:
    if isinstance(output, dict) and "result" in output:
        return str(output["result"])
    if isinstance(output, dict):
        chunks = output.get("chunks")
        if isinstance(chunks, list):
            if not chunks:
                return "no retrieved chunks"
            top = chunks[0]
            if isinstance(top, dict):
                content = str(top.get("content", "")).strip()
                return content or "retrieved chunk with empty content"
    return str(output)
