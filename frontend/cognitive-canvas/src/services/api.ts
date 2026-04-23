import axios from "axios";
import type { ChatResponseMeta } from "@/lib/types";

export interface ChatRequest {
  user_id: string;
  session_id: string;
  message: string;
  document_ids?: string[];
}

export interface ChatApiOptions {
  baseUrl: string;
  timeoutMs?: number;
}

function normalizeResponse(data: Record<string, unknown>): ChatResponseMeta {
  const traceEvents = Array.isArray(data.trace_events)
    ? (data.trace_events as Array<Record<string, unknown>>)
    : [];

  const trace = traceEvents.map((event) => ({
    step: String(event.node_name ?? event.event_type ?? "step"),
    label: String(event.event_type ?? event.node_name ?? "step"),
    latency_ms:
      typeof event.duration_ms === "number" ? event.duration_ms : undefined,
    detail:
      typeof event.success === "boolean"
        ? event.success
          ? "success"
          : "failed"
        : undefined,
  }));

  const budgetAudit =
    data.budget_audit && typeof data.budget_audit === "object"
      ? (data.budget_audit as Record<string, unknown>)
      : undefined;

  const allocation = budgetAudit
    ? {
        session_recent_turns_used: Number(
          budgetAudit.session_recent_turns_used ?? 0,
        ),
        session_summary_chars: Number(budgetAudit.session_summary_chars ?? 0),
        user_facts_used: Number(budgetAudit.user_facts_used ?? 0),
        pinned_facts_used: Number(budgetAudit.pinned_facts_used ?? 0),
        corpus_items_used: Number(budgetAudit.corpus_items_used ?? 0),
      }
    : undefined;

  const evictions = Array.isArray(budgetAudit?.evicted_items)
    ? (budgetAudit?.evicted_items as string[])
    : undefined;

  const final_answer: string =
    (typeof data.final_answer === "string" && data.final_answer) ||
    (typeof data.answer === "string" && data.answer) ||
    (typeof data.response === "string" && data.response) ||
    (typeof data.message === "string" && data.message) ||
    "";

  return {
    ...data,
    final_answer,
    trace_id:
      typeof data.trace_id === "string"
        ? data.trace_id
        : typeof data.request_id === "string"
          ? data.request_id
          : undefined,
    trace,
    budget_audit: budgetAudit
      ? {
          ...budgetAudit,
          allocation,
          evictions,
        }
      : undefined,
    raw: data,
  };
}

/**
 * Sends a chat message to the backend POST /chat endpoint.
 * Returns the raw JSON response — caller is responsible for shape handling.
 */
export async function postChat(
  payload: ChatRequest,
  opts: ChatApiOptions,
): Promise<ChatResponseMeta & { final_answer: string }> {
  const url = `${opts.baseUrl.replace(/\/$/, "")}/chat`;
  const res = await axios.post(url, payload, {
    timeout: opts.timeoutMs ?? 60_000,
    headers: { "Content-Type": "application/json" },
  });
  const data =
    res.data && typeof res.data === "object"
      ? (res.data as Record<string, unknown>)
      : {};
  return normalizeResponse(data) as ChatResponseMeta & { final_answer: string };
}
