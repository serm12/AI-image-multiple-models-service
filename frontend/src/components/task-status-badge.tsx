import { Badge } from "@/components/ui/badge";
import { useT } from "@/lib/admin";

const success = "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-300";
const failure = "border-red-200 bg-red-50 text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300";
const running = "border-blue-200 bg-blue-50 text-blue-700 dark:border-blue-800 dark:bg-blue-950 dark:text-blue-300";
const pending = "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-300";

export function TaskStatusBadge({ status }: { status: string }) {
  const t = useT();
  const states: Record<string, [string, string]> = {
    succeeded: [t("成功", "Succeeded"), success],
    success: [t("成功", "Succeeded"), success],
    completed: [t("成功", "Succeeded"), success],
    failed: [t("失败", "Failed"), failure],
    error: [t("失败", "Failed"), failure],
    running: [t("处理中", "Running"), running],
    processing: [t("处理中", "Processing"), running],
    pending: [t("等待中", "Pending"), pending],
    queued: [t("排队中", "Queued"), pending],
    cancelled: [t("已取消", "Cancelled"), "bg-muted text-muted-foreground"],
    canceled: [t("已取消", "Cancelled"), "bg-muted text-muted-foreground"],
  };
  const [label, color] = states[status] ?? [status, "bg-muted text-muted-foreground"];
  return <Badge variant="outline" className={color} title={status}>{label}</Badge>;
}
