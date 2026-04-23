export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  createdAt: number;
  meta?: ChatResponseMeta;
  pending?: boolean;
  error?: string;
}

export interface RetrievedChunk {
  id?: string;
  document_id?: string;
  title?: string;
  content?: string;
  text?: string;
  score?: number;
  source?: string;
  [k: string]: unknown;
}

export interface ToolCall {
  name: string;
  input?: unknown;
  output?: unknown;
  result?: unknown;
  latency_ms?: number;
  status?: string;
  [k: string]: unknown;
}

export interface BudgetAudit {
  total_tokens?: number;
  used_tokens?: number;
  remaining_tokens?: number;
  allocation?: Record<string, number>;
  evictions?: Array<{ layer?: string; reason?: string; tokens?: number } | string>;
  [k: string]: unknown;
}

export interface MemoryL2 {
  summary?: string;
  recent_topics?: string[];
  [k: string]: unknown;
}

export interface MemoryL3 {
  facts?: Array<{ key?: string; value?: string; confidence?: number; pinned?: boolean } | string>;
  pinned?: string[];
  contradictions_resolved?: Array<{ from?: string; to?: string; at?: number }>;
  [k: string]: unknown;
}

export interface TraceStep {
  step: string;
  label?: string;
  latency_ms?: number;
  detail?: string;
  [k: string]: unknown;
}

export interface ChatResponseMeta {
  final_answer?: string;
  sources?: RetrievedChunk[];
  retrieved_chunks?: RetrievedChunk[];
  used_llm?: boolean;
  trace_id?: string;
  trace?: TraceStep[];
  trace_events?: Array<{
    event_type?: string;
    node_name?: string;
    duration_ms?: number;
    success?: boolean;
    metadata?: Record<string, unknown>;
    [k: string]: unknown;
  }>;
  budget_audit?: BudgetAudit;
  tools?: ToolCall[];
  tool_calls?: ToolCall[];
  memory?: {
    l2?: MemoryL2;
    l3?: MemoryL3;
    l4?: RetrievedChunk[];
  };
  raw?: unknown;
  [k: string]: unknown;
}

export interface ChatSession {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
  documentIds: string[];
}
