import { useState } from "react";
import {
  Brain,
  FileText,
  Wrench,
  Gauge,
  Activity,
  Code,
  ChevronRight,
} from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { cn } from "@/lib/utils";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type {
  BudgetAudit,
  MemoryL2,
  MemoryL3,
  RetrievedChunk,
  ToolCall,
  TraceStep,
} from "@/lib/types";

export function InspectorPanel() {
  const meta = useAppStore((s) => s.getLatestMeta());
  const debug = useAppStore((s) => s.settings.debugMode);

  const l2: MemoryL2 | undefined = meta?.memory?.l2;
  const l3: MemoryL3 | undefined = meta?.memory?.l3;
  const docs: RetrievedChunk[] =
    meta?.retrieved_chunks ??
    meta?.sources ??
    meta?.memory?.l4 ??
    [];
  const tools: ToolCall[] = meta?.tool_calls ?? meta?.tools ?? [];
  const budget: BudgetAudit | undefined = meta?.budget_audit;
  const trace: TraceStep[] = meta?.trace ?? [];

  return (
    <aside className="flex h-full w-full flex-col border-l border-border bg-card/40">
      <div className="border-b border-border px-4 py-4">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          Inspector
        </p>
        <p className="text-sm font-semibold">Workflow Observability</p>
      </div>

      <Tabs defaultValue="memory" className="flex flex-1 flex-col overflow-hidden">
        <TabsList className="mx-3 mt-3 grid grid-cols-5 bg-muted/60 p-1">
          <TabTrigger value="memory" icon={<Brain className="h-3.5 w-3.5" />} label="Mem" />
          <TabTrigger value="rag" icon={<FileText className="h-3.5 w-3.5" />} label="RAG" />
          <TabTrigger value="tools" icon={<Wrench className="h-3.5 w-3.5" />} label="Tools" />
          <TabTrigger value="budget" icon={<Gauge className="h-3.5 w-3.5" />} label="Budget" />
          <TabTrigger value="trace" icon={<Activity className="h-3.5 w-3.5" />} label="Trace" />
        </TabsList>

        <div className="scroll-thin flex-1 overflow-y-auto px-4 py-4">
          <TabsContent value="memory" className="m-0 space-y-4">
            <MemoryView l2={l2} l3={l3} />
          </TabsContent>
          <TabsContent value="rag" className="m-0 space-y-3">
            <RagView docs={docs} />
          </TabsContent>
          <TabsContent value="tools" className="m-0 space-y-3">
            <ToolsView tools={tools} />
          </TabsContent>
          <TabsContent value="budget" className="m-0 space-y-3">
            <BudgetView budget={budget} />
          </TabsContent>
          <TabsContent value="trace" className="m-0 space-y-3">
            <TraceView trace={trace} />
          </TabsContent>
        </div>

        {debug && (
          <div className="border-t border-border bg-background/60 px-4 py-3">
            <p className="mb-2 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              <Code className="h-3 w-3" /> Raw response
            </p>
            <pre className="scroll-thin max-h-48 overflow-auto rounded-lg bg-muted/60 p-2 font-mono text-[10px] leading-relaxed">
              {meta ? JSON.stringify(meta.raw ?? meta, null, 2) : "—"}
            </pre>
          </div>
        )}
      </Tabs>
    </aside>
  );
}

function TabTrigger({
  value,
  icon,
  label,
}: {
  value: string;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <TabsTrigger
      value={value}
      className="flex flex-col gap-0.5 px-1 py-1.5 text-[10px] data-[state=active]:bg-card data-[state=active]:shadow-sm"
    >
      {icon}
      {label}
    </TabsTrigger>
  );
}

function SectionTitle({
  color,
  label,
  hint,
}: {
  color: string;
  label: string;
  hint?: string;
}) {
  return (
    <div className="mb-2 flex items-center gap-2">
      <span
        className="h-2 w-2 rounded-full"
        style={{ backgroundColor: `var(--color-${color})` }}
      />
      <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
      {hint && (
        <p className="ml-auto text-[10px] text-muted-foreground">{hint}</p>
      )}
    </div>
  );
}

function EmptyHint({ children }: { children: React.ReactNode }) {
  return (
    <p className="rounded-lg border border-dashed border-border px-3 py-4 text-center text-[11px] text-muted-foreground">
      {children}
    </p>
  );
}

function MemoryView({ l2, l3 }: { l2?: MemoryL2; l3?: MemoryL3 }) {
  const facts = l3?.facts ?? [];
  const pinned = l3?.pinned ?? [];
  const contradictions = l3?.contradictions_resolved ?? [];

  return (
    <>
      <div>
        <SectionTitle color="l2" label="L2 · Session memory" />
        {l2?.summary ? (
          <div className="rounded-lg border border-border bg-card p-3 text-xs leading-relaxed">
            {l2.summary}
          </div>
        ) : (
          <EmptyHint>Session summary will appear after first reply.</EmptyHint>
        )}
        {l2?.recent_topics && l2.recent_topics.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {l2.recent_topics.map((t, i) => (
              <span
                key={i}
                className="rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground"
              >
                {t}
              </span>
            ))}
          </div>
        )}
      </div>

      <div>
        <SectionTitle
          color="l3"
          label="L3 · User facts"
          hint={facts.length ? `${facts.length} facts` : undefined}
        />
        {facts.length === 0 ? (
          <EmptyHint>No persistent user facts yet.</EmptyHint>
        ) : (
          <ul className="space-y-1.5">
            {facts.map((f, i) => {
              const isStr = typeof f === "string";
              const key = isStr ? `fact_${i}` : f.key ?? `fact_${i}`;
              const value = isStr ? f : f.value ?? JSON.stringify(f);
              const conf = !isStr ? f.confidence : undefined;
              const isPinned = !isStr && f.pinned;
              return (
                <li
                  key={i}
                  className="flex items-start gap-2 rounded-lg border border-border bg-card px-3 py-2 text-xs"
                >
                  <div className="min-w-0 flex-1">
                    <p className="font-mono text-[10px] text-muted-foreground">
                      {key}
                    </p>
                    <p className="truncate">{String(value)}</p>
                  </div>
                  {conf !== undefined && (
                    <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                      {(conf * 100).toFixed(0)}%
                    </span>
                  )}
                  {isPinned && (
                    <span className="shrink-0 rounded bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium text-primary">
                      pinned
                    </span>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {pinned.length > 0 && (
        <div>
          <SectionTitle color="l3" label="Pinned" />
          <ul className="space-y-1">
            {pinned.map((p, i) => (
              <li
                key={i}
                className="rounded-md bg-primary/5 px-2 py-1 text-xs text-primary"
              >
                {p}
              </li>
            ))}
          </ul>
        </div>
      )}

      {contradictions.length > 0 && (
        <div>
          <SectionTitle color="l3" label="Contradictions resolved" />
          <ul className="space-y-1.5">
            {contradictions.map((c, i) => (
              <li
                key={i}
                className="rounded-lg border border-border bg-card p-2 text-[11px]"
              >
                <span className="line-through text-muted-foreground">
                  {c.from}
                </span>
                <ChevronRight className="mx-1 inline h-3 w-3" />
                <span className="font-medium">{c.to}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </>
  );
}

function RagView({ docs }: { docs: RetrievedChunk[] }) {
  return (
    <>
      <SectionTitle
        color="l4"
        label="L4 · Retrieved chunks"
        hint={docs.length ? `${docs.length} hits` : undefined}
      />
      {docs.length === 0 ? (
        <EmptyHint>No documents retrieved for the latest message.</EmptyHint>
      ) : (
        <ul className="space-y-2">
          {docs.map((d, i) => {
            const text = d.content ?? d.text ?? "";
            const score = d.score;
            return (
              <li
                key={i}
                className="rounded-lg border border-border bg-card p-3"
              >
                <div className="mb-1.5 flex items-center justify-between gap-2">
                  <p className="truncate text-xs font-medium">
                    {d.title ?? d.source ?? d.document_id ?? `chunk_${i + 1}`}
                  </p>
                  {score !== undefined && (
                    <span
                      className="shrink-0 rounded px-1.5 py-0.5 font-mono text-[10px]"
                      style={{
                        backgroundColor: `color-mix(in oklab, var(--color-l4) 18%, transparent)`,
                        color: `var(--color-l4)`,
                      }}
                    >
                      {Number(score).toFixed(3)}
                    </span>
                  )}
                </div>
                <p className="line-clamp-3 text-[11px] leading-relaxed text-muted-foreground">
                  {text || "—"}
                </p>
                {score !== undefined && (
                  <div className="mt-2 h-1 overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.max(0, Math.min(1, Number(score))) * 100}%`,
                        backgroundColor: `var(--color-l4)`,
                      }}
                    />
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </>
  );
}

function ToolsView({ tools }: { tools: ToolCall[] }) {
  return (
    <>
      <SectionTitle
        color="tool"
        label="Tool execution"
        hint={tools.length ? `${tools.length} call${tools.length > 1 ? "s" : ""}` : undefined}
      />
      {tools.length === 0 ? (
        <EmptyHint>No tools were invoked.</EmptyHint>
      ) : (
        <ul className="space-y-2">
          {tools.map((t, i) => (
            <li
              key={i}
              className="rounded-lg border border-border bg-card p-3 text-xs"
            >
              <div className="mb-1 flex items-center justify-between gap-2">
                <p className="font-mono text-xs font-medium">{t.name}</p>
                <div className="flex items-center gap-1.5">
                  {t.status && (
                    <span
                      className={cn(
                        "rounded px-1.5 py-0.5 text-[10px]",
                        t.status === "success" || t.status === "ok"
                          ? "bg-emerald-500/15 text-emerald-400"
                          : "bg-destructive/15 text-destructive",
                      )}
                    >
                      {t.status}
                    </span>
                  )}
                  {t.latency_ms !== undefined && (
                    <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                      {t.latency_ms}ms
                    </span>
                  )}
                </div>
              </div>
              <pre className="scroll-thin max-h-32 overflow-auto rounded bg-muted/60 p-2 font-mono text-[10px]">
                {JSON.stringify(t.output ?? t.result ?? t.input ?? {}, null, 2)}
              </pre>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

function BudgetView({ budget }: { budget?: BudgetAudit }) {
  if (!budget) {
    return (
      <>
        <SectionTitle color="chart-1" label="Token budget" />
        <EmptyHint>No budget audit on the latest response.</EmptyHint>
      </>
    );
  }

  const allocation = budget.allocation ?? {};
  const total =
    budget.total_tokens ??
    Object.values(allocation).reduce((a, b) => a + (Number(b) || 0), 0);
  const used = budget.used_tokens;
  const remaining = budget.remaining_tokens;

  const palette = ["chart-1", "chart-2", "chart-3", "chart-4", "chart-5"];
  const entries = Object.entries(allocation);

  return (
    <>
      <SectionTitle color="chart-1" label="Token budget" hint={total ? `${total} tok` : undefined} />

      {entries.length > 0 && (
        <>
          <div className="flex h-2 overflow-hidden rounded-full bg-muted">
            {entries.map(([k, v], i) => {
              const pct = total ? (Number(v) / total) * 100 : 0;
              return (
                <div
                  key={k}
                  style={{
                    width: `${pct}%`,
                    backgroundColor: `var(--color-${palette[i % palette.length]})`,
                  }}
                  title={`${k}: ${v}`}
                />
              );
            })}
          </div>
          <ul className="mt-3 space-y-1.5">
            {entries.map(([k, v], i) => (
              <li
                key={k}
                className="flex items-center gap-2 text-xs"
              >
                <span
                  className="h-2 w-2 shrink-0 rounded-full"
                  style={{
                    backgroundColor: `var(--color-${palette[i % palette.length]})`,
                  }}
                />
                <span className="capitalize text-muted-foreground">
                  {k.replace(/_/g, " ")}
                </span>
                <span className="ml-auto font-mono">{v}</span>
              </li>
            ))}
          </ul>
        </>
      )}

      {(used !== undefined || remaining !== undefined) && (
        <div className="mt-3 grid grid-cols-2 gap-2">
          {used !== undefined && (
            <Stat label="Used" value={`${used}`} />
          )}
          {remaining !== undefined && (
            <Stat label="Remaining" value={`${remaining}`} />
          )}
        </div>
      )}

      {budget.evictions && budget.evictions.length > 0 && (
        <div className="mt-4">
          <SectionTitle color="destructive" label="Evictions" />
          <ul className="space-y-1">
            {budget.evictions.map((e, i) => (
              <li
                key={i}
                className="rounded-md bg-destructive/10 px-2 py-1.5 text-[11px] text-destructive"
              >
                {typeof e === "string"
                  ? e
                  : `${e.layer ?? "?"} · ${e.reason ?? "trimmed"}${
                      e.tokens ? ` · -${e.tokens} tok` : ""
                    }`}
              </li>
            ))}
          </ul>
        </div>
      )}
    </>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-card p-2">
      <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
      <p className="font-mono text-sm">{value}</p>
    </div>
  );
}

function TraceView({ trace }: { trace: TraceStep[] }) {
  return (
    <>
      <SectionTitle color="primary" label="Workflow trace" />
      {trace.length === 0 ? (
        <EmptyHint>No trace steps reported.</EmptyHint>
      ) : (
        <ol className="relative space-y-3 border-l border-border pl-4">
          {trace.map((s, i) => (
            <li key={i} className="relative">
              <span className="absolute -left-[21px] top-1 h-2.5 w-2.5 rounded-full bg-primary ring-4 ring-background" />
              <div className="rounded-lg border border-border bg-card p-2.5">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-xs font-medium capitalize">
                    {s.label ?? s.step}
                  </p>
                  {s.latency_ms !== undefined && (
                    <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                      {s.latency_ms}ms
                    </span>
                  )}
                </div>
                {s.detail && (
                  <p className="mt-1 text-[11px] text-muted-foreground">
                    {s.detail}
                  </p>
                )}
              </div>
            </li>
          ))}
        </ol>
      )}
    </>
  );
}
