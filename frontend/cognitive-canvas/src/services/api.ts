import axios from "axios";
import type { ChatResponseMeta, RetrievedChunk, ToolCall } from "@/lib/types";

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
  const traceRoot =
    data.trace && typeof data.trace === "object"
      ? (data.trace as Record<string, unknown>)
      : undefined;

  const traceEvents = Array.isArray(data.trace_events)
    ? (data.trace_events as Array<Record<string, unknown>>)
    : Array.isArray(traceRoot?.events)
      ? (traceRoot.events as Array<Record<string, unknown>>)
      : [];

  const trace = traceEvents.map((event) => ({
    step: String(event.node_name ?? event.event_type ?? "step"),
    label: String(event.node_name ?? event.event_type ?? "step"),
    latency_ms: typeof event.duration_ms === "number" ? event.duration_ms : undefined,
    detail: typeof event.success === "boolean" ? (event.success ? "success" : "failed") : undefined,
  }));

  const budgetAudit =
    data.budget_audit && typeof data.budget_audit === "object"
      ? (data.budget_audit as Record<string, unknown>)
      : undefined;

  const allocation = (() => {
    if (!budgetAudit) {
      return undefined;
    }

    const rawAllocation = budgetAudit.allocation;
    if (rawAllocation && typeof rawAllocation === "object") {
      const normalized: Record<string, number> = {};
      for (const [key, value] of Object.entries(rawAllocation as Record<string, unknown>)) {
        if (typeof value === "number") {
          normalized[key] = value;
          continue;
        }
        if (value && typeof value === "object") {
          const tokens = (value as Record<string, unknown>).tokens;
          if (typeof tokens === "number") {
            normalized[key] = tokens;
          }
        }
      }
      if (Object.keys(normalized).length > 0) {
        return normalized;
      }
    }

    return {
      session_recent_turns_used: Number(budgetAudit.session_recent_turns_used ?? 0),
      session_summary_chars: Number(budgetAudit.session_summary_chars ?? 0),
      user_facts_used: Number(budgetAudit.user_facts_used ?? 0),
      pinned_facts_used: Number(budgetAudit.pinned_facts_used ?? 0),
      corpus_items_used: Number(budgetAudit.corpus_items_used ?? 0),
    };
  })();

  const evictions =
    Array.isArray(budgetAudit?.evictions)
      ? (budgetAudit?.evictions as string[])
      : Array.isArray(budgetAudit?.eviction_events)
        ? (budgetAudit?.eviction_events as string[])
        : Array.isArray(budgetAudit?.evicted_items)
          ? (budgetAudit?.evicted_items as string[])
          : undefined;

  const final_answer: string =
    (typeof data.final_answer === "string" && data.final_answer) ||
    (typeof data.answer === "string" && data.answer) ||
    (typeof data.response === "string" && data.response) ||
    (typeof data.message === "string" && data.message) ||
    "";

  const rawSources = Array.isArray(data.sources)
    ? (data.sources as Array<Record<string, unknown>>)
    : [];

  const metadata =
    data.metadata && typeof data.metadata === "object"
      ? (data.metadata as Record<string, unknown>)
      : undefined;

  const extractText = (source: Record<string, unknown>) => {
    const metadataObject =
      source.metadata && typeof source.metadata === "object"
        ? (source.metadata as Record<string, unknown>)
        : undefined;
    const candidates = [
      source.content,
      source.text,
      source.chunk_content,
      source.snippet,
      metadataObject?.content,
      metadataObject?.text,
      metadataObject?.chunk_content,
      metadataObject?.snippet,
      metadataObject?.excerpt,
    ];
    for (const candidate of candidates) {
      if (typeof candidate === "string" && candidate.trim()) {
        return candidate.trim();
      }
    }
    return "";
  };

  const normalizedDocuments: RetrievedChunk[] = rawSources
    .filter((source) => String(source.type ?? "") === "document")
    .map((source) => ({
      ...source,
      type: String(source.type ?? "document"),
      document_id: typeof source.document_id === "string" ? source.document_id : undefined,
      chunk_id: typeof source.chunk_id === "string" ? source.chunk_id : undefined,
      content: extractText(source),
      metadata:
        source.metadata && typeof source.metadata === "object"
          ? (source.metadata as Record<string, unknown>)
          : undefined,
    }));

  const normalizedTools: ToolCall[] = rawSources
    .filter((source) => String(source.type ?? "") === "tool")
    .map((source) => {
      const metadataObject =
        source.metadata && typeof source.metadata === "object"
          ? (source.metadata as Record<string, unknown>)
          : undefined;
      const input =
        source.input ?? metadataObject?.input ?? metadataObject?.arguments ?? metadataObject?.payload;
      const output =
        source.output ?? source.result ?? metadataObject?.output ?? metadataObject?.result ?? metadataObject;

      return {
        ...source,
        type: String(source.type ?? "tool"),
        name: String(source.name ?? "tool"),
        latency_ms: typeof source.latency_ms === "number" ? source.latency_ms : undefined,
        status:
          typeof source.status === "string"
            ? source.status
            : source.failed
              ? "failed"
              : "success",
        input,
        output,
        result: source.result ?? metadataObject?.result,
        document_id: typeof source.document_id === "string" ? source.document_id : undefined,
        chunk_id: typeof source.chunk_id === "string" ? source.chunk_id : undefined,
        metadata: metadataObject,
      };
    });

  const writeBackSummary =
    (metadata && typeof metadata.write_back_summary === "string" && metadata.write_back_summary) ||
    (typeof data.write_back_summary === "string" && data.write_back_summary) ||
    "";

  const writeBackFactsRaw =
    (metadata && metadata.write_back_facts) || (data.write_back_facts as unknown);

  const normalizedFacts = Array.isArray(writeBackFactsRaw)
    ? writeBackFactsRaw
        .map((fact) => {
          if (typeof fact === "string") {
            return { value: fact.trim() };
          }
          if (fact && typeof fact === "object") {
            const item = fact as Record<string, unknown>;
            const value =
              (typeof item.value === "string" && item.value.trim()) ||
              (typeof item.fact === "string" && item.fact.trim()) ||
              (typeof item.text === "string" && item.text.trim()) ||
              (typeof item.summary === "string" && item.summary.trim()) ||
              JSON.stringify(item);
            return {
              key: typeof item.key === "string" ? item.key : undefined,
              value,
              confidence: typeof item.confidence === "number" ? item.confidence : undefined,
              pinned: typeof item.pinned === "boolean" ? item.pinned : undefined,
            };
          }
          return null;
        })
        .filter((fact): fact is { key?: string; value: string; confidence?: number; pinned?: boolean } =>
          Boolean(fact && fact.value),
        )
    : [];

  const memory =
    writeBackSummary || normalizedFacts.length > 0
      ? {
          l2: writeBackSummary
            ? {
                summary: writeBackSummary,
              }
            : undefined,
          l3: normalizedFacts.length > 0 ? { facts: normalizedFacts } : undefined,
        }
      : undefined;

  return {
    ...data,
    final_answer,
    sources: rawSources as ChatResponseMeta["sources"],
    retrieved_chunks: normalizedDocuments,
    tool_calls: normalizedTools,
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
    trace_events: traceEvents,
    tools: normalizedTools,
    memory: memory ?? data.memory,
    metadata: metadata ?? undefined,
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
    res.data && typeof res.data === "object" ? (res.data as Record<string, unknown>) : {};
  return normalizeResponse(data) as ChatResponseMeta & { final_answer: string };
}
