import { useEffect, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { Menu, PanelRight, Moon, Sun, Github } from "lucide-react";
import { SessionSidebar } from "@/components/sidebar/SessionSidebar";
import { ChatArea } from "@/components/chat/ChatArea";
import { InspectorPanel } from "@/components/panels/InspectorPanel";
import { SettingsDialog } from "@/components/SettingsDialog";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent } from "@/components/ui/sheet";
import { useTheme } from "@/components/theme-provider";
import { useAppStore } from "@/store/useAppStore";

export const Route = createFileRoute("/")({
  component: Dashboard,
});

function Dashboard() {
  const { theme, toggle } = useTheme();
  const sessions = useAppStore((s) => s.sessions);
  const activeId = useAppStore((s) => s.activeSessionId);
  const createSession = useAppStore((s) => s.createSession);
  const setActive = useAppStore((s) => s.setActiveSession);

  const [mobileSidebar, setMobileSidebar] = useState(false);
  const [mobileInspector, setMobileInspector] = useState(false);

  // Ensure there's always at least one active session selected if any exist
  useEffect(() => {
    if (sessions.length > 0 && !sessions.find((s) => s.id === activeId)) {
      setActive(sessions[0].id);
    }
  }, [sessions, activeId, setActive]);

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden bg-background text-foreground">
      {/* Top bar */}
      <header className="flex h-12 shrink-0 items-center gap-2 border-b border-border bg-background/80 px-3 backdrop-blur">
        <Button
          variant="ghost"
          size="icon"
          className="h-9 w-9 lg:hidden"
          onClick={() => setMobileSidebar(true)}
        >
          <Menu className="h-4 w-4" />
        </Button>

        <div className="flex items-center gap-2">
          <span className="hidden text-sm font-semibold tracking-tight sm:block">
            <span className="text-gradient">Memora</span>
          </span>
          <span className="hidden rounded-full border border-border bg-card px-2 py-0.5 text-[10px] font-medium text-muted-foreground sm:block">
            beta
          </span>
        </div>

        <div className="ml-auto flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-9 w-9"
            onClick={toggle}
            title="Toggle theme"
          >
            {theme === "dark" ? (
              <Sun className="h-4 w-4" />
            ) : (
              <Moon className="h-4 w-4" />
            )}
          </Button>
          <SettingsDialog />
          <Button
            variant="ghost"
            size="icon"
            className="hidden h-9 w-9 sm:flex"
            asChild
          >
            <a
              href="https://github.com"
              target="_blank"
              rel="noreferrer"
              aria-label="GitHub"
            >
              <Github className="h-4 w-4" />
            </a>
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-9 w-9 xl:hidden"
            onClick={() => setMobileInspector(true)}
            title="Inspector"
          >
            <PanelRight className="h-4 w-4" />
          </Button>
        </div>
      </header>

      {/* 3-column workspace */}
      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[280px_1fr] xl:grid-cols-[280px_1fr_360px]">
        <div className="hidden min-h-0 lg:block">
          <SessionSidebar />
        </div>

        <main className="min-h-0">
          <ChatArea />
        </main>

        <div className="hidden min-h-0 xl:block">
          <InspectorPanel />
        </div>
      </div>

      {/* Mobile sidebar */}
      <Sheet open={mobileSidebar} onOpenChange={setMobileSidebar}>
        <SheetContent side="left" className="w-[280px] p-0">
          <SessionSidebar onClose={() => setMobileSidebar(false)} />
        </SheetContent>
      </Sheet>

      {/* Mobile inspector */}
      <Sheet open={mobileInspector} onOpenChange={setMobileInspector}>
        <SheetContent side="right" className="w-[340px] p-0">
          <InspectorPanel />
        </SheetContent>
      </Sheet>

      {/* Auto-create first session on first visit */}
      <FirstRun
        sessionsCount={sessions.length}
        onCreate={() => createSession()}
      />
    </div>
  );
}

function FirstRun({
  sessionsCount,
  onCreate,
}: {
  sessionsCount: number;
  onCreate: () => void;
}) {
  useEffect(() => {
    if (sessionsCount === 0) onCreate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return null;
}
