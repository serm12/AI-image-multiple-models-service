import { createContext, useContext, useEffect, useState } from "react";
import { toast } from "sonner";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { AlertCircle, Copy, Loader2 } from "lucide-react";
export const Locale = createContext("zh-CN");
export function useT() {
  const locale = useContext(Locale);
  return (zh: string, en: string) => (locale === "en" ? en : zh);
}
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { credentials: "same-origin", ...init });
  if (
    response.redirected &&
    new URL(response.url).pathname === "/admin/login"
  ) {
    window.location.assign(
      "/admin/login?next=" +
        encodeURIComponent(location.pathname + location.search),
    );
    throw new Error("登录已过期 / Session expired");
  }
  if (!response.ok) {
    let message = `${response.status} · 请求失败 / Request failed`;
    try {
      const data = await response.json();
      if (typeof data.detail === "string") message = data.detail;
      else if (Array.isArray(data.detail))
        message = data.detail
          .map(
            (item: { loc: string[]; msg: string }) =>
              `${item.loc.slice(1).join(".")}: ${item.msg}`,
          )
          .join("; ");
    } catch {
      /* use status */
    }
    throw new Error(message);
  }
  return response.json();
}
export function post<T>(path: string, value: unknown) {
  return api<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value),
  });
}
export function useData<T>(path: string) {
  const [data, setData] = useState<T>();
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    api<T>(path, { signal: controller.signal })
      .then(setData)
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [path, revision]);
  return {
    data,
    setData,
    error,
    loading,
    reload: () => setRevision((n) => n + 1),
  };
}
export function Pick({
  value,
  onChange,
  options,
  label,
  disabled = false,
}: {
  value: string;
  onChange: (v: string) => void;
  options: [string, string][];
  label: string;
  disabled?: boolean;
}) {
  return (
    <Select
      value={value || "__all"}
      onValueChange={(v) => onChange(v === "__all" ? "" : v)}
      disabled={disabled}
    >
      <SelectTrigger className="w-full bg-white" aria-label={label}>
        <SelectValue placeholder={label} />
      </SelectTrigger>
      <SelectContent>
        {options.map(([key, name]) => (
          <SelectItem key={key || "__all"} value={key || "__all"}>
            {name}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
export function ErrorBox({
  message,
  retry,
}: {
  message: string;
  retry?: () => void;
}) {
  const t = useT();
  return (
    <div
      role="alert"
      className="my-4 flex flex-wrap items-center gap-3 rounded-md border border-destructive/25 bg-red-50 p-4 text-sm text-destructive"
    >
      <AlertCircle className="size-4" />
      {message}
      {retry && (
        <Button variant="outline" size="sm" onClick={retry}>
          {t("重试", "Retry")}
        </Button>
      )}
    </div>
  );
}
export function Loading() {
  const t = useT();
  return (
    <div
      role="status"
      className="flex items-center gap-2 py-12 text-sm text-muted-foreground"
    >
      <Loader2 className="size-4 animate-spin" />
      {t("正在加载…", "Loading…")}
    </div>
  );
}
export function CopyButton({ value }: { value: string }) {
  const t = useT();
  return (
    <Button
      variant="ghost"
      size="icon"
      className="size-7 shrink-0"
      aria-label={t("复制", "Copy")}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value);
          toast.success(t("已复制", "Copied"));
        } catch {
          toast.error(
            t(
              "复制失败，请手动选择文本",
              "Copy failed. Select the text manually.",
            ),
          );
        }
      }}
    >
      <Copy className="size-3.5" />
    </Button>
  );
}
export function safeUrl(value: unknown) {
  if (typeof value !== "string") return undefined;
  try {
    const url = new URL(value, location.origin);
    return ["https:", "http:"].includes(url.protocol) ? url.href : undefined;
  } catch {
    return undefined;
  }
}
