import { useState } from "react";
import { Settings as SettingsIcon } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { DEFAULT_BACKEND_BASE_URL, useAppStore } from "@/store/useAppStore";

export function SettingsDialog() {
  const settings = useAppStore((s) => s.settings);
  const setSettings = useAppStore((s) => s.setSettings);

  const [open, setOpen] = useState(false);
  const [baseUrl, setBaseUrl] = useState(settings.baseUrl);
  const [userId, setUserId] = useState(settings.userId);
  const [debug, setDebug] = useState(settings.debugMode);

  function save() {
    setSettings({
      baseUrl: baseUrl.trim().replace(/\/$/, "") || DEFAULT_BACKEND_BASE_URL,
      userId: userId.trim() || settings.userId,
      debugMode: debug,
    });
    setOpen(false);
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        setOpen(o);
        if (o) {
          setBaseUrl(settings.baseUrl || DEFAULT_BACKEND_BASE_URL);
          setUserId(settings.userId);
          setDebug(settings.debugMode);
        }
      }}
    >
      <DialogTrigger asChild>
        <Button size="icon" variant="ghost" className="h-9 w-9">
          <SettingsIcon className="h-4 w-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Connection settings</DialogTitle>
          <DialogDescription>
            Configure where the frontend should send POST /chat requests.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-2">
          <div className="space-y-1.5">
            <Label htmlFor="baseUrl">Backend base URL</Label>
            <Input
              id="baseUrl"
              placeholder="https://api.example.com"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
            />
            <p className="text-[11px] text-muted-foreground">
              The frontend will POST to <code>{"{baseUrl}/chat"}</code>.
            </p>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="userId">user_id</Label>
            <Input id="userId" value={userId} onChange={(e) => setUserId(e.target.value)} />
            <p className="text-[11px] text-muted-foreground">
              Same user_id across sessions reuses L3 user memory.
            </p>
          </div>
          <div className="flex items-center justify-between rounded-lg border border-border p-3">
            <div>
              <Label htmlFor="debug" className="text-sm">
                Debug mode
              </Label>
              <p className="text-[11px] text-muted-foreground">
                Show raw JSON response under the inspector.
              </p>
            </div>
            <Switch id="debug" checked={debug} onCheckedChange={setDebug} />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button onClick={save}>Save</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
