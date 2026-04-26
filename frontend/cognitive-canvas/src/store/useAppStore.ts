import { create } from "zustand";
import { persist } from "zustand/middleware";
import type {
  ChatMessage,
  ChatSession,
  ChatResponseMeta,
  WorkflowObservabilityState,
} from "@/lib/types";

export const DEFAULT_BACKEND_BASE_URL =
  (import.meta.env.VITE_BACKEND_BASE_URL?.trim().replace(/\/$/, "") ?? "") ||
  "http://127.0.0.1:8000";
const DEFAULT_USER_ID = "user_local";

const uid = () =>
  globalThis.crypto?.randomUUID?.() ??
  Math.random().toString(36).slice(2) + Date.now().toString(36);

const makeUserId = () => "user_" + Math.random().toString(36).slice(2, 8);

interface Settings {
  baseUrl: string;
  userId: string;
  debugMode: boolean;
}

interface AppState {
  settings: Settings;
  sessions: ChatSession[];
  activeSessionId: string | null;
  observability: WorkflowObservabilityState;
  setSettings: (s: Partial<Settings>) => void;
  createSession: (title?: string) => string;
  deleteSession: (id: string) => void;
  renameSession: (id: string, title: string) => void;
  setActiveSession: (id: string) => void;
  appendMessage: (sessionId: string, msg: ChatMessage) => void;
  updateMessage: (sessionId: string, id: string, patch: Partial<ChatMessage>) => void;
  setSessionDocs: (sessionId: string, ids: string[]) => void;
  clearObservability: () => void;
  setObservabilityFromMeta: (meta: ChatResponseMeta) => void;
  getActiveSession: () => ChatSession | undefined;
  getLatestMeta: () => ChatResponseMeta | undefined;
}

const EMPTY_OBSERVABILITY: WorkflowObservabilityState = {
  traceEvents: [],
  ragSources: [],
  toolEvents: [],
  budgetAudit: undefined,
  sessionSummary: undefined,
  userFacts: [],
};

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      settings: {
        baseUrl: DEFAULT_BACKEND_BASE_URL,
        userId: DEFAULT_USER_ID,
        debugMode: false,
      },
      sessions: [],
      activeSessionId: null,
      observability: EMPTY_OBSERVABILITY,

      setSettings: (s) => set((st) => ({ settings: { ...st.settings, ...s } })),

      createSession: (title) => {
        const id = uid();
        const session: ChatSession = {
          id,
          title: title ?? "New chat",
          createdAt: Date.now(),
          updatedAt: Date.now(),
          messages: [],
          documentIds: [],
        };
        set((st) => ({
          sessions: [session, ...st.sessions],
          activeSessionId: id,
        }));
        return id;
      },

      deleteSession: (id) =>
        set((st) => {
          const sessions = st.sessions.filter((s) => s.id !== id);
          const activeSessionId =
            st.activeSessionId === id ? (sessions[0]?.id ?? null) : st.activeSessionId;
          return { sessions, activeSessionId };
        }),

      renameSession: (id, title) =>
        set((st) => ({
          sessions: st.sessions.map((s) =>
            s.id === id ? { ...s, title, updatedAt: Date.now() } : s,
          ),
        })),

      setActiveSession: (id) => set({ activeSessionId: id }),

      appendMessage: (sessionId, msg) =>
        set((st) => ({
          sessions: st.sessions.map((s) =>
            s.id === sessionId
              ? {
                  ...s,
                  messages: [...s.messages, msg],
                  updatedAt: Date.now(),
                  title:
                    s.messages.length === 0 && msg.role === "user"
                      ? msg.content.slice(0, 48)
                      : s.title,
                }
              : s,
          ),
        })),

      updateMessage: (sessionId, id, patch) =>
        set((st) => ({
          sessions: st.sessions.map((s) =>
            s.id === sessionId
              ? {
                  ...s,
                  messages: s.messages.map((m) => (m.id === id ? { ...m, ...patch } : m)),
                  updatedAt: Date.now(),
                }
              : s,
          ),
        })),

      setSessionDocs: (sessionId, ids) =>
        set((st) => ({
          sessions: st.sessions.map((s) => (s.id === sessionId ? { ...s, documentIds: ids } : s)),
        })),

      clearObservability: () => set({ observability: EMPTY_OBSERVABILITY }),

      setObservabilityFromMeta: (meta) => {
        const traceEvents = meta.trace ?? [];
        const ragSources = meta.retrieved_chunks ?? meta.sources ?? [];
        const toolEvents = meta.tool_calls ?? meta.tools ?? [];
        const sessionSummary =
          (meta.metadata && typeof meta.metadata.write_back_summary === "string"
            ? meta.metadata.write_back_summary
            : undefined) ??
          (typeof meta.write_back_summary === "string" ? meta.write_back_summary : undefined) ??
          meta.memory?.l2?.summary;

        const rawFacts =
          (meta.metadata && meta.metadata.write_back_facts) ??
          (meta.write_back_facts as unknown) ??
          meta.memory?.l3?.facts ??
          [];

        const userFacts = Array.isArray(rawFacts)
          ? rawFacts
              .map((fact) => {
                if (typeof fact === "string") {
                  return { value: fact.trim() };
                }
                if (!fact || typeof fact !== "object") {
                  return null;
                }
                const item = fact as Record<string, unknown>;
                const value =
                  (typeof item.value === "string" && item.value.trim()) ||
                  (typeof item.fact === "string" && item.fact.trim()) ||
                  (typeof item.text === "string" && item.text.trim()) ||
                  (typeof item.summary === "string" && item.summary.trim()) ||
                  "";
                if (!value) return null;
                return {
                  key: typeof item.key === "string" ? item.key : undefined,
                  value,
                  confidence: typeof item.confidence === "number" ? item.confidence : undefined,
                  pinned: typeof item.pinned === "boolean" ? item.pinned : undefined,
                };
              })
              .filter(
                (fact): fact is { key?: string; value: string; confidence?: number; pinned?: boolean } =>
                  Boolean(fact),
              )
          : [];

        set({
          observability: {
            traceEvents,
            ragSources,
            toolEvents,
            budgetAudit: meta.budget_audit,
            sessionSummary,
            userFacts,
          },
        });
      },

      getActiveSession: () => {
        const { sessions, activeSessionId } = get();
        return sessions.find((s) => s.id === activeSessionId);
      },

      getLatestMeta: () => {
        const session = get().getActiveSession();
        if (!session) return undefined;
        for (let i = session.messages.length - 1; i >= 0; i--) {
          const m = session.messages[i];
          if (m.role === "assistant" && m.meta) return m.meta;
        }
        return undefined;
      },
    }),
    {
      name: "genai-dashboard-store",
      version: 2,
      merge: (persistedState, currentState) => {
        const persisted = persistedState as Partial<AppState> | undefined;
        const current = currentState as AppState;

        if (!persisted) {
          return current;
        }

        const persistedObservability = persisted.observability ?? {};
        const currentObservability = current.observability;

        return {
          ...current,
          ...persisted,
          settings: {
            ...current.settings,
            ...persisted.settings,
          },
          observability: {
            ...currentObservability,
            ...persistedObservability,
            traceEvents: persistedObservability.traceEvents ?? currentObservability.traceEvents,
            ragSources: persistedObservability.ragSources ?? currentObservability.ragSources,
            toolEvents: persistedObservability.toolEvents ?? currentObservability.toolEvents,
            budgetAudit: persistedObservability.budgetAudit ?? currentObservability.budgetAudit,
            sessionSummary:
              persistedObservability.sessionSummary ?? currentObservability.sessionSummary,
            userFacts: persistedObservability.userFacts ?? currentObservability.userFacts,
          },
        };
      },
      migrate: (persistedState: unknown, version: number) => {
        if (!persistedState || typeof persistedState !== "object") {
          return persistedState as AppState;
        }

        const state = persistedState as AppState;
        if (version < 2) {
          const currentBaseUrl = state.settings?.baseUrl?.trim?.() ?? "";
          const currentUserId = state.settings?.userId?.trim?.() ?? "";
          return {
            ...state,
            settings: {
              ...state.settings,
              baseUrl: currentBaseUrl || DEFAULT_BACKEND_BASE_URL,
              userId: currentUserId || makeUserId(),
            },
          };
        }

        if (!state.settings?.userId || state.settings.userId === DEFAULT_USER_ID) {
          return {
            ...state,
            settings: {
              ...state.settings,
              userId: makeUserId(),
            },
          };
        }

        return state;
      },
    },
  ),
);
