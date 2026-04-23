import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Plus,
  MessageSquare,
  Trash2,
  Pencil,
  Check,
  X,
  Sparkles,
} from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface Props {
  onClose?: () => void;
}

export function SessionSidebar({ onClose }: Props) {
  const sessions = useAppStore((s) => s.sessions);
  const activeId = useAppStore((s) => s.activeSessionId);
  const createSession = useAppStore((s) => s.createSession);
  const setActive = useAppStore((s) => s.setActiveSession);
  const remove = useAppStore((s) => s.deleteSession);
  const rename = useAppStore((s) => s.renameSession);

  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  return (
    <aside className="flex h-full w-full flex-col border-r border-border bg-sidebar text-sidebar-foreground">
      <div className="flex items-center gap-2 border-b border-sidebar-border px-4 py-4">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg gradient-primary shadow-lg shadow-primary/20">
          <Sparkles className="h-4 w-4 text-primary-foreground" />
        </div>
        <div className="flex-1">
          <p className="text-sm font-semibold tracking-tight">Memora</p>
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
            GenAI Workflow
          </p>
        </div>
      </div>

      <div className="px-3 pt-3">
        <Button
          onClick={() => {
            createSession();
            onClose?.();
          }}
          className="w-full justify-start gap-2 rounded-xl"
          variant="default"
        >
          <Plus className="h-4 w-4" />
          New session
        </Button>
      </div>

      <div className="mt-3 px-4 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        Sessions · L2 memory
      </div>

      <nav className="scroll-thin mt-1 flex-1 space-y-0.5 overflow-y-auto px-2 pb-4">
        <AnimatePresence initial={false}>
          {sessions.length === 0 && (
            <p className="px-3 py-8 text-center text-xs text-muted-foreground">
              No sessions yet.
              <br />
              Start a new conversation.
            </p>
          )}
          {sessions.map((s) => {
            const last = s.messages[s.messages.length - 1];
            const preview = last?.content?.slice(0, 60) ?? "Empty session";
            const isActive = s.id === activeId;
            const isEditing = editingId === s.id;

            return (
              <motion.div
                key={s.id}
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.15 }}
                className={cn(
                  "group relative cursor-pointer rounded-lg px-3 py-2 transition-colors",
                  isActive
                    ? "bg-sidebar-accent"
                    : "hover:bg-sidebar-accent/60",
                )}
                onClick={() => {
                  if (!isEditing) {
                    setActive(s.id);
                    onClose?.();
                  }
                }}
              >
                {isEditing ? (
                  <div className="flex items-center gap-1">
                    <input
                      autoFocus
                      value={draft}
                      onChange={(e) => setDraft(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          rename(s.id, draft.trim() || s.title);
                          setEditingId(null);
                        }
                        if (e.key === "Escape") setEditingId(null);
                      }}
                      className="w-full rounded-md border border-input bg-background px-2 py-1 text-sm outline-none focus:ring-2 focus:ring-ring"
                    />
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        rename(s.id, draft.trim() || s.title);
                        setEditingId(null);
                      }}
                      className="rounded p-1 hover:bg-accent"
                    >
                      <Check className="h-3.5 w-3.5" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setEditingId(null);
                      }}
                      className="rounded p-1 hover:bg-accent"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                ) : (
                  <div className="flex items-start gap-2">
                    <MessageSquare
                      className={cn(
                        "mt-0.5 h-3.5 w-3.5 shrink-0",
                        isActive ? "text-primary" : "text-muted-foreground",
                      )}
                    />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium">{s.title}</p>
                      <p className="truncate text-[11px] text-muted-foreground">
                        {preview}
                      </p>
                    </div>
                    <div className="flex shrink-0 items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setEditingId(s.id);
                          setDraft(s.title);
                        }}
                        className="rounded p-1 hover:bg-accent"
                      >
                        <Pencil className="h-3 w-3" />
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          remove(s.id);
                        }}
                        className="rounded p-1 text-destructive hover:bg-destructive/10"
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    </div>
                  </div>
                )}
              </motion.div>
            );
          })}
        </AnimatePresence>
      </nav>
    </aside>
  );
}
