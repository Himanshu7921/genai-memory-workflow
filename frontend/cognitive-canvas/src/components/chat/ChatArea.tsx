import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Send, Paperclip, Bot, User, Loader2, AlertCircle } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { postChat } from "@/services/api";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import type { ChatMessage } from "@/lib/types";

const uid = () =>
  globalThis.crypto?.randomUUID?.() ??
  Math.random().toString(36).slice(2) + Date.now().toString(36);

function formatTime(ts: number) {
  return new Date(ts).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function ChatArea() {
  const session = useAppStore((s) => s.getActiveSession());
  const settings = useAppStore((s) => s.settings);
  const appendMessage = useAppStore((s) => s.appendMessage);
  const updateMessage = useAppStore((s) => s.updateMessage);
  const createSession = useAppStore((s) => s.createSession);

  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [session?.messages.length, session?.id]);

  async function handleSend() {
    const text = input.trim();
    if (!text || sending) return;

    let sid = session?.id;
    if (!sid) sid = createSession();

    const userMsg: ChatMessage = {
      id: uid(),
      role: "user",
      content: text,
      createdAt: Date.now(),
    };
    appendMessage(sid, userMsg);
    setInput("");

    const assistantId = uid();
    appendMessage(sid, {
      id: assistantId,
      role: "assistant",
      content: "",
      createdAt: Date.now(),
      pending: true,
    });

    if (!settings.baseUrl) {
      updateMessage(sid, assistantId, {
        pending: false,
        error:
          "Backend URL not configured. Open Settings (top-right) and paste your /chat endpoint base URL.",
      });
      return;
    }

    setSending(true);
    try {
      const docs =
        useAppStore
          .getState()
          .sessions.find((s) => s.id === sid)
          ?.documentIds ?? [];
      const resp = await postChat(
        {
          user_id: settings.userId,
          session_id: sid,
          message: text,
          document_ids: docs,
        },
        { baseUrl: settings.baseUrl },
      );

      // Simulate streaming by progressively revealing the answer
      const full = resp.final_answer ?? "";
      const chunkSize = Math.max(2, Math.ceil(full.length / 60));
      let i = 0;
      await new Promise<void>((resolve) => {
        const tick = () => {
          i = Math.min(full.length, i + chunkSize);
          updateMessage(sid!, assistantId, {
            content: full.slice(0, i),
            pending: i < full.length,
            meta: i >= full.length ? resp : undefined,
          });
          if (i >= full.length) {
            resolve();
            return;
          }
          setTimeout(tick, 20);
        };
        tick();
      });
    } catch (err) {
      const msg =
        (
          err as {
            response?: { data?: { detail?: string; message?: string } };
            message?: string;
          }
        )?.response?.data?.detail ??
        (
          err as {
            response?: { data?: { detail?: string; message?: string } };
            message?: string;
          }
        )?.response?.data?.message ??
        (err as Error)?.message ??
        "Request failed";
      updateMessage(sid, assistantId, {
        pending: false,
        error: msg,
      });
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="flex h-full flex-col bg-background">
      <div
        ref={scrollRef}
        className="scroll-thin flex-1 overflow-y-auto px-4 py-6 sm:px-8"
      >
        <div className="mx-auto max-w-3xl space-y-6">
          {(!session || session.messages.length === 0) && <EmptyState />}
          {session?.messages.map((m) => (
            <MessageBubble key={m.id} m={m} />
          ))}
        </div>
      </div>

      <div className="border-t border-border bg-background/80 px-4 py-4 backdrop-blur sm:px-8">
        <div className="mx-auto max-w-3xl">
          <div className="flex items-end gap-2 rounded-2xl border border-border bg-card p-2 shadow-sm focus-within:ring-2 focus-within:ring-ring">
            <Button
              size="icon"
              variant="ghost"
              className="h-9 w-9 shrink-0 text-muted-foreground"
              title="Attach document_ids (coming soon)"
            >
              <Paperclip className="h-4 w-4" />
            </Button>
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Ask anything — memory, retrieval, and tools will activate automatically…"
              className="min-h-[44px] flex-1 resize-none border-0 bg-transparent px-1 text-sm shadow-none focus-visible:ring-0"
              rows={1}
            />
            <Button
              onClick={handleSend}
              disabled={sending || !input.trim()}
              size="icon"
              className="h-9 w-9 shrink-0 rounded-xl"
            >
              {sending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
          <p className="mt-2 text-center text-[10px] text-muted-foreground">
            user_id <span className="font-mono">{settings.userId}</span>
            {session && (
              <>
                {" · "}session_id{" "}
                <span className="font-mono">{session.id.slice(0, 8)}</span>
              </>
            )}
          </p>
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ m }: { m: ChatMessage }) {
  const isUser = m.role === "user";
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={cn("flex gap-3", isUser ? "flex-row-reverse" : "flex-row")}
    >
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-lg",
          isUser
            ? "bg-primary text-primary-foreground"
            : "gradient-primary text-primary-foreground shadow-lg shadow-primary/20",
        )}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>
      <div
        className={cn(
          "min-w-0 max-w-[85%] rounded-2xl px-4 py-3 text-sm",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-card text-card-foreground border border-border",
        )}
      >
        {m.error ? (
          <div className="flex items-start gap-2 text-destructive">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{m.error}</span>
          </div>
        ) : m.pending && !m.content ? (
          <TypingDots />
        ) : (
          <div className="whitespace-pre-wrap leading-relaxed">
            {m.content}
            {m.pending && (
              <span className="ml-0.5 inline-block h-3 w-1.5 -translate-y-0.5 animate-pulse bg-current align-middle" />
            )}
          </div>
        )}
        <div
          className={cn(
            "mt-1.5 text-[10px]",
            isUser
              ? "text-primary-foreground/70"
              : "text-muted-foreground",
          )}
        >
          {formatTime(m.createdAt)}
          {m.meta?.used_llm && (
            <span className="ml-2">· {m.meta.used_llm}</span>
          )}
          {m.meta?.trace_id && (
            <span className="ml-2 font-mono">
              · {String(m.meta.trace_id).slice(0, 8)}
            </span>
          )}
        </div>
      </div>
    </motion.div>
  );
}

function TypingDots() {
  return (
    <div className="flex h-4 items-center gap-1">
      <span className="typing-dot h-1.5 w-1.5 rounded-full bg-current" />
      <span className="typing-dot h-1.5 w-1.5 rounded-full bg-current" />
      <span className="typing-dot h-1.5 w-1.5 rounded-full bg-current" />
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex h-[60vh] flex-col items-center justify-center text-center">
      <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl gradient-primary shadow-xl shadow-primary/30">
        <Bot className="h-7 w-7 text-primary-foreground" />
      </div>
      <h2 className="text-2xl font-semibold tracking-tight">
        Start a new conversation
      </h2>
      <p className="mt-2 max-w-md text-sm text-muted-foreground">
        Memora remembers what you discuss across sessions. Watch the right panel
        to see memory layers, retrieved documents, tool calls, and token budget
        in real time.
      </p>
      <div className="mt-6 grid grid-cols-2 gap-2 text-left sm:grid-cols-4">
        {[
          { k: "L2", label: "Session" },
          { k: "L3", label: "User facts" },
          { k: "L4", label: "Documents" },
          { k: "Tools", label: "Execution" },
        ].map((x) => (
          <div
            key={x.k}
            className="rounded-xl border border-border bg-card p-3"
          >
            <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              {x.k}
            </p>
            <p className="text-sm font-medium">{x.label}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
