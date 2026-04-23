from __future__ import annotations

from dataclasses import dataclass

from app.models.memory import FactStatus
from app.models.orchestration import OrchestrationState


@dataclass(frozen=True, slots=True)
class PromptBundle:
    system_instruction: str
    user_prompt: str


def build_response_prompt_bundle(state: OrchestrationState, memory_answer: str | None = None) -> PromptBundle:
    summary = ""
    if state.memory_snapshot and state.memory_snapshot.summary:
        summary = state.memory_snapshot.summary.summary.strip()

    user_facts_lines: list[str] = []
    pinned_lines: list[str] = []
    if state.memory_snapshot:
        for fact in state.memory_snapshot.user_facts:
            if fact.status == FactStatus.ACTIVE and fact.value.strip():
                user_facts_lines.append(f"- {fact.canonical_key}: {fact.value.strip()}")
        for fact in state.memory_snapshot.pinned_facts:
            if fact.value.strip():
                pinned_lines.append(f"- {fact.canonical_key}: {fact.value.strip()} ({fact.reason})")

    document_lines: list[str] = []
    if state.retrieval_context:
        for chunk in state.retrieval_context.chunks:
            snippet = chunk.content.strip().replace("\n", " ")
            document_lines.append(
                f"- doc={chunk.document_id} chunk={chunk.chunk_id} score={chunk.score:.3f}: {snippet[:500]}"
            )

    tool_lines: list[str] = []
    if state.tool_results:
        for result in state.tool_results.results:
            if result.success:
                tool_lines.append(f"- {result.tool_name}: {_format_tool_result(result.output)}")
            else:
                tool_lines.append(f"- {result.tool_name}: failed ({result.error})")

    # Strict system prompt: be concise, direct, no fluff, no over-personalization
    system_instruction = (
        "You are a production assistant providing concise, direct answers. "
        "Rules:\n"
        "1. Answer the user query directly and concisely. Avoid unnecessary explanations or preamble.\n"
        "2. Use provided context (memory facts, documents, tool results) when relevant.\n"
        "3. Do NOT ask follow-up questions unless explicitly required by the user.\n"
        "4. Do NOT add assistant-style fluff like 'Thank you for asking' or 'I'd be happy to help'.\n"
        "5. Avoid over-personalization: use neutral phrasing like 'Based on your profile' instead of using user names or assumptions.\n"
        "6. If context is insufficient, state that clearly without asking follow-up questions.\n"
        "7. Do not output raw JSON, Python dicts, internal metadata, or debugging artifacts.\n"
        "8. Keep responses short and actionable."
    )

    memory_context = ""
    if memory_answer:
        memory_context = f"\nDirect Match Found (from memory): {memory_answer}\nFormat this naturally in your response. Use it to answer the query directly."

    user_prompt = (
        "## User Query\n"
        f"{state.turn.message}\n\n"
        "## Memory Context (L2 + L3)\n"
        f"Session Summary (L2):\n{summary or 'none'}\n"
        f"User Facts (L3):\n{chr(10).join(user_facts_lines) if user_facts_lines else '- none'}\n"
        f"Pinned Facts:\n{chr(10).join(pinned_lines) if pinned_lines else '- none'}\n\n"
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
