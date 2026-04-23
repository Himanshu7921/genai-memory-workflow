import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { ChatMessage, ChatSession, ChatResponseMeta } from "@/lib/types";

const DEFAULT_BACKEND_BASE_URL = "http://127.0.0.1:8000";
const DEFAULT_USER_ID = "user_local";

const uid = () =>
  (globalThis.crypto?.randomUUID?.() ??
    Math.random().toString(36).slice(2) + Date.now().toString(36));

const makeUserId = () =>
  "user_" + Math.random().toString(36).slice(2, 8);

interface Settings {
  baseUrl: string;
  userId: string;
  debugMode: boolean;
}

interface AppState {
  settings: Settings;
  sessions: ChatSession[];
  activeSessionId: string | null;
  setSettings: (s: Partial<Settings>) => void;
  createSession: (title?: string) => string;
  deleteSession: (id: string) => void;
  renameSession: (id: string, title: string) => void;
  setActiveSession: (id: string) => void;
  appendMessage: (sessionId: string, msg: ChatMessage) => void;
  updateMessage: (sessionId: string, id: string, patch: Partial<ChatMessage>) => void;
  setSessionDocs: (sessionId: string, ids: string[]) => void;
  getActiveSession: () => ChatSession | undefined;
  getLatestMeta: () => ChatResponseMeta | undefined;
}

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

      setSettings: (s) =>
        set((st) => ({ settings: { ...st.settings, ...s } })),

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
            st.activeSessionId === id
              ? (sessions[0]?.id ?? null)
              : st.activeSessionId;
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
                  messages: s.messages.map((m) =>
                    m.id === id ? { ...m, ...patch } : m,
                  ),
                  updatedAt: Date.now(),
                }
              : s,
          ),
        })),

      setSessionDocs: (sessionId, ids) =>
        set((st) => ({
          sessions: st.sessions.map((s) =>
            s.id === sessionId ? { ...s, documentIds: ids } : s,
          ),
        })),

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
