import { useEffect, useRef, useState, type FormEvent } from "react";
import { toast } from "sonner";
import {
  Images,
  Star,
  RefreshCw,
  Search,
  ImageOff,
  ExternalLink,
  ChevronLeft,
  ChevronRight,
  SlidersHorizontal,
  ShoppingCart,
  MousePointerClick,
  CreditCard,
  CircleCheck,
  Timer,
  Trash2,
  Loader2,
  RotateCcw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Table,
  TableHeader,
  TableRow,
  TableHead,
  TableBody,
  TableCell,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import {
  useT,
  api,
  useData,
  post,
  Pick,
  ErrorBox,
  Loading,
  CopyButton,
  safeUrl,
} from "@/lib/admin";
import { taskQuery } from "@/lib/task-query";
import { TaskStatusBadge } from "@/components/task-status-badge";
import { SourceInfo } from "@/components/source-info";
type Image = {
  url: string;
  thumbnail: string;
  filename: string;
  reference: boolean;
};
type Task = {
  task_id: string;
  status: string;
  starred: boolean;
  trashed_at: string;
  purge_after: string;
  cleanup_started: boolean;
  images: Image[];
  display_times: Record<string, string>;
  api_provider: string;
  client_ip: string;
  client_country: string;
  customer_email: string;
  customer_id: string;
  storefront_visitor_id: string;
  customer_logged_in: boolean;
  source_product_title: string;
  source_page_title: string;
  source_page_url: string;
  request_url: string;
  traffic_source: string;
  traffic_referrer: string;
  utm_source: string;
  utm_medium: string;
  utm_campaign: string;
  generation_duration_seconds: number | null;
  prompt: string;
  edit_instructions: string;
  description: string;
  source_task_id: string;
  subscription_status: string;
  subscription_reason: string;
  customer_sync: { state?: string; detail?: string; customer_id?: string };
  shopify_orders: { url: string; name: string; financial_status: string }[];
  ip_task_sequence: number;
  ip_task_total: number;
  storefront_event_times: Record<string, string>;
  added_to_cart: boolean;
  checkout_intent: boolean;
  checkout_started: boolean;
  checkout_completed: boolean;
};
type Result = {
  statistics?: Record<string, number>;
  tasks: Task[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  has_previous: boolean;
  has_next: boolean;
};
const stages = [
  "added_to_cart",
  "checkout_intent",
  "checkout_started",
  "checkout_completed",
] as const;
const stageIcons = [ShoppingCart, MousePointerClick, CreditCard, CircleCheck] as const;
const stageColors = [
  "border-blue-200 bg-blue-50 text-blue-700 dark:border-blue-800 dark:bg-blue-950 dark:text-blue-300",
  "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-300",
  "border-violet-200 bg-violet-50 text-violet-700 dark:border-violet-800 dark:bg-violet-950 dark:text-violet-300",
  "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
] as const;
export function Tasks({ timezones }: { timezones: string[] }) {
  const t = useT();
  const [query, setQuery] = useState(() => taskQuery(location.search));
  const result = useData<Result>("/admin/api/tasks" + query);
  const queryRef = useRef(query);
  queryRef.current = query;
  const [detail, setDetail] = useState<Task>();
  const [gallery, setGallery] = useState<{ images: Image[]; index: number }>();
  const [filters, setFilters] = useState(false);
  const [busy, setBusy] = useState<string[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [deleteTargets, setDeleteTargets] = useState<string[]>([]);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState("");
  const [selectionAction, setSelectionAction] = useState<"trash" | "restore">("trash");
  const inRecycleBin = new URLSearchParams(query).get("recycle_bin") === "true";
  const deleteLock = useRef(false);
  const cancelDelete = useRef<HTMLButtonElement>(null);
  const visibleIds = result.data?.tasks.filter((task) => !task.cleanup_started).map((task) => task.task_id) || [];
  const selectedIds = selected.filter((id) => visibleIds.includes(id));
  const allSelected = visibleIds.length > 0 && selectedIds.length === visibleIds.length;
  useEffect(() => { setSelected([]); }, [query]);
  useEffect(() => {
    setSelected((current) => current.filter((id) => result.data?.tasks.some((task) => task.task_id === id)));
  }, [result.data]);
  function selectTask(id: string, checked: boolean) {
    setSelected((current) => checked ? [...new Set([...current, id])] : current.filter((item) => item !== id));
  }
  async function deleteSelected() {
    if (deleteLock.current || !deleteTargets.length) return;
    deleteLock.current = true;
    setDeleting(true);
    setDeleteError("");
    const startedQuery = query;
    try {
      const saved = await post<{ deleted: string[]; failed: { task_id: string; reason: string }[] }>(
        "/admin/api/tasks/" + (selectionAction === "restore" ? "restore" : "delete"), { task_ids: deleteTargets, confirmed: true },
      );
      const removed = new Set(saved.deleted);
      setSelected((current) => current.filter((id) => !removed.has(id)));
      if (queryRef.current === startedQuery) result.setData((current) => current ? {
        ...current, tasks: current.tasks.filter((task) => !removed.has(task.task_id)),
        total: Math.max(0, current.total - saved.deleted.length),
      } : current);
      if (detail && removed.has(detail.task_id)) setDetail(undefined);
      if (saved.deleted.length) toast.success(selectionAction === "restore"
        ? t(`已恢复 ${saved.deleted.length} 条任务`, `Restored ${saved.deleted.length} tasks`)
        : t(`已将 ${saved.deleted.length} 条任务移入回收站`, `Moved ${saved.deleted.length} tasks to the recycle bin`));
      if (saved.failed.length) {
        setDeleteTargets(saved.failed.map((item) => item.task_id));
        const reasons: Record<string, string> = {
          task_running: t("任务仍在处理中", "Task is still processing"),
          task_not_found: t("任务不存在或已清理", "Task no longer exists"),
          cleanup_in_progress: t("已开始永久清理，无法恢复", "Permanent cleanup has started; cannot restore"),
          storage_error: t("存储暂不可用，请重试", "Storage unavailable; please retry"),
        };
        setDeleteError(saved.failed.map((item) => `${item.task_id}: ${reasons[item.reason] || t("操作失败，请重试", "Operation failed; please retry")}`).join("\n"));
      } else {
        setDeleteTargets([]);
      }
      try {
        const refreshed = await api<Result>("/admin/api/tasks" + startedQuery);
        if (queryRef.current === startedQuery) result.setData(refreshed);
      } catch {
        toast.error(t("操作已保存，列表更新失败，请点击刷新。", "Changes saved. Could not refresh the list; please refresh."));
      }
    } catch (error) {
      setDeleteError((error as Error).message);
    } finally {
      deleteLock.current = false;
      setDeleting(false);
    }
  }
  const labels = [
    t("加入购物车", "Added to cart"),
    t("结账意向", "Checkout intent"),
    t("开始结账", "Checkout started"),
    t("完成结账", "Checkout completed"),
  ];
  useEffect(() => {
    const pop = () => setQuery(taskQuery(location.search));
    addEventListener("popstate", pop);
    return () => removeEventListener("popstate", pop);
  }, []);
  function update(params: URLSearchParams) {
    const next = taskQuery(params.toString());
    history.pushState(null, "", "/admin/tasks" + next);
    setQuery(next);
  }
  function change(key: string, value: string) {
    const params = new URLSearchParams(query);
    params.set("page", "1");
    if (value) params.set(key, value);
    else params.delete(key);
    update(params);
  }
  function search(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = new FormData(e.currentTarget);
    const params = new URLSearchParams();
    form.forEach((value, key) => {
      if (String(value)) params.append(key, String(value));
    });
    params.set("page", "1");
    params.set(
      "page_size",
      new URLSearchParams(query).get("page_size") || "25",
    );
    update(params);
  }
  async function star(task: Task) {
    const original = task.starred;
    const startedQuery = query;
    const patch = (starred: boolean) => result.setData((current) => current ? {
      ...current,
      tasks: current.tasks.map((item) => item.task_id === task.task_id ? { ...item, starred } : item),
    } : current);
    setBusy((v) => [...v, task.task_id]);
    patch(!original);
    try {
      const saved = await post<{ starred: boolean }>(
        "/admin/api/tasks/" + encodeURIComponent(task.task_id) + "/star",
        { starred: !original },
      );
      patch(saved.starred);
      if (new URLSearchParams(startedQuery).get("starred") === "true") {
        // Refresh filtered pagination in the background without unmounting the table.
        try {
          const refreshed = await api<Result>("/admin/api/tasks" + startedQuery);
          if (queryRef.current === startedQuery) result.setData(refreshed);
        } catch {
          toast.error(t("收藏已保存，列表更新失败，请点击刷新。", "Saved. Could not update the list; please refresh."));
        }
      }
    } catch (e) {
      patch(original);
      toast.error((e as Error).message);
    } finally {
      setBusy((v) => v.filter((id) => id !== task.task_id));
    }
  }
  const params = new URLSearchParams(query);
  const active = [...params.entries()].filter(
    ([key, value]) => !["page", "page_size", "recycle_bin"].includes(key) && value,
  );
  function funnel(task: Task) {
    return (
      <div className="flex flex-wrap gap-1">
        {stages.map((stage, i) => {
          const Icon = stageIcons[i];
          return (
          <Badge
            key={stage}
            variant="outline"
            className={
              task[stage]
                ? stageColors[i]
                : "border-border/60 text-muted-foreground"
            }
            title={task.storefront_event_times?.[stage] || ""}
          >
            <Icon aria-hidden="true" />
            {labels[i]}
          </Badge>
          );
        })}
      </div>
    );
  }
  function customer(task: Task) {
    const id = (
      task.customer_sync?.customer_id ||
      task.customer_id ||
      ""
    ).replace("gid://shopify/Customer/", "");
    return /^\d+$/.test(id)
      ? `https://admin.shopify.com/store/paintingify/customers/${id}`
      : undefined;
  }
  const subscription: Record<string, string> = {
    accepted: t("订阅已接受", "Subscription accepted"),
    verification_required: t("订阅需要验证", "Verification required"),
    failed: t("订阅失败", "Subscription failed"),
    unconfirmed: t("订阅未确认", "Subscription unconfirmed"),
    not_attempted: t("未尝试订阅", "Subscription not attempted"),
    queued: t("等待同步", "Sync queued"),
    retry: t("正在重试同步", "Retrying sync"),
    subscribed: t("已订阅", "Subscribed"),
    preserved: t("保留原订阅状态", "Subscription preserved"),
  };
  return (
    <>
      <div className="mb-5 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4" aria-label={t("任务统计", "Task statistics")} aria-busy={result.loading}>
        {[
          { key: "total", label: t("任务总数", "Total tasks"), Icon: Images },
          { key: "added_to_cart", label: t("已加购任务", "Added to cart"), Icon: ShoppingCart },
          { key: "checkout_intent", label: t("结账意向任务", "Checkout intent"), Icon: MousePointerClick },
          { key: "checkout_completed", label: t("已完成结账任务", "Checkout completed"), Icon: CircleCheck },
        ].map(({ key, label, Icon }) => {
          const stats = result.loading || result.error ? undefined : result.data?.statistics;
          const value = stats?.[key];
          const percentage = stats && stats.total > 0 && value != null ? value / stats.total * 100 : 0;
          const selected = params.getAll("conversion_status").includes(key);
          return <div key={key} className="relative rounded-2xl border bg-card px-5 py-5 transition-colors sm:px-6">
            {key !== "total" && <button
              type="button"
              aria-label={label}
              aria-pressed={selected}
              disabled={deleting}
              className="absolute inset-0 z-10 cursor-pointer rounded-2xl outline-none hover:bg-muted/20 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed"
              onClick={() => {
                const next = new URLSearchParams(query);
                if (selected) next.delete("conversion_status", key);
                else next.append("conversion_status", key);
                next.set("page", "1");
                update(next);
              }}
            />}
            <div className="flex items-center justify-between gap-3 text-sm text-muted-foreground">
              <span>{label}</span><Icon className="size-4" aria-hidden="true" />
            </div>
            <div className="mt-5 flex items-end justify-between gap-3">
              <span className="text-3xl font-semibold tracking-tight tabular-nums">{value == null ? "—" : value.toLocaleString()}</span>
              <div className="text-right text-xs text-muted-foreground">
                {key !== "total" && value != null && <div className="mb-1 font-medium tabular-nums text-foreground">{percentage.toFixed(1)}%</div>}
                {key === "total" ? t("当前筛选 · 全部分页", "All matching tasks") : t("占匹配任务比例", "Share of matching tasks")}
              </div>
            </div>
          </div>;
        })}
      </div>
      <section aria-label={t("任务列表", "Task list")} className="min-w-0 overflow-hidden rounded-2xl border bg-card">
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-4 sm:px-5">
        <div className="flex items-center gap-3">
          <Button
            variant={filters ? "default" : "outline"}
            className={filters ? "bg-neutral-900 text-white hover:bg-neutral-800" : undefined}
            onClick={() => setFilters(!filters)}
            aria-expanded={filters}
            aria-controls="task-filters"
          >
            <SlidersHorizontal className="size-4" />
            {t("筛选", "Filters")}
            {active.length > 0 && (
              <Badge variant="secondary">{active.length}</Badge>
            )}
          </Button>
        <span className="text-sm text-muted-foreground">
          {result.data
            ? `${result.data.total.toLocaleString()} ${t("条任务", "tasks")}`
            : t("正在查询", "Loading tasks")}
        </span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {selectedIds.length > 0 && !result.loading && (
            <>
              <span className="mr-1 text-sm text-muted-foreground">{t(`已选 ${selectedIds.length} 条`, `${selectedIds.length} selected`)}</span>
              <Button variant="ghost" disabled={deleting} onClick={() => setSelected([])}>{t("取消选择", "Clear selection")}</Button>
              <Button variant={inRecycleBin ? "default" : "destructive"} disabled={deleting} onClick={() => { setSelectionAction(inRecycleBin ? "restore" : "trash"); setDeleteError(""); setDeleteTargets([...selectedIds]); }}>
                {inRecycleBin ? <RotateCcw className="size-4" /> : <Trash2 className="size-4" />}{inRecycleBin ? t("恢复所选", "Restore selected") : t("删除所选", "Delete selected")}
              </Button>
            </>
          )}
        <Button variant={inRecycleBin ? "secondary" : "outline"} disabled={deleting} onClick={() => change("recycle_bin", inRecycleBin ? "" : "true")}>
          {inRecycleBin ? <ChevronLeft className="size-4" /> : <Trash2 className="size-4" />}
          {inRecycleBin ? t("返回任务", "Back to tasks") : t("回收站", "Recycle bin")}
        </Button>
        <Button
          variant="outline"
          onClick={result.reload}
          disabled={result.loading}
        >
          <RefreshCw
            className={"size-4 " + (result.loading ? "animate-spin" : "")}
          />
          {t("刷新", "Refresh")}
        </Button>

          <Button
            variant={params.get("starred") === "true" ? "default" : "outline"}
            onClick={() =>
              change("starred", params.get("starred") === "true" ? "" : "true")
            }
          >
            <Star className="size-4" />
            {t("收藏", "Starred")}
          </Button>
          <div className="w-36">
            <TimeRange value={params.get("time_range") || ""} onChange={(value) => change("time_range", value)} />
          </div>

        </div>
      </div>
      {inRecycleBin && <div className="mx-4 mb-4 rounded-lg bg-muted/60 px-4 py-3 text-sm sm:mx-5">
        <span className="font-medium">{t("回收站", "Recycle bin")}</span>
        <span className="ml-2 text-muted-foreground">{t("保留 30 天，期间可恢复；到期后自动永久清理任务数据和图片，相关下载将失效。", "Kept for 30 days and restorable during this period. After expiry, task data and images are permanently removed and related downloads become unavailable.")}</span>
      </div>}
      <form
        id="task-filters"
        hidden={!filters}
        key={query}
        onSubmit={search}
        className="space-y-5 px-4 pb-5 pt-1 sm:px-5"
      >
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
          <div className="field">
            <Label htmlFor="task-search">{t("任务 ID", "Task ID")}</Label>
            <Input
              id="task-search"
              name="task_id"
              placeholder={t("搜索任务…", "Search tasks…")}
              defaultValue={params.get("task_id") || ""}
            />
          </div>
          <div className="field">
            <Label htmlFor="user-search">
              {t("客户 / 邮箱", "Customer / email")}
            </Label>
            <Input
              id="user-search"
              name="user"
              placeholder={t(
                "邮箱、客户或访客 ID",
                "Email, customer or visitor ID",
              )}
              defaultValue={params.get("user") || ""}
            />
          </div>
          <div className="field">
            <Label>{t("时间范围", "Time range")}</Label>
            <TimeRange value={params.get("time_range") || ""} />
          </div>
          <div className="field">
            <Label htmlFor="ip-search">IP</Label>
            <Input
              id="ip-search"
              name="client_ip"
              defaultValue={params.get("client_ip") || ""}
            />
          </div>
          <div className="field">
            <Label htmlFor="provider-search">
              {t("模型供应商", "Provider")}
            </Label>
            <Input
              id="provider-search"
              name="provider"
              defaultValue={params.get("provider") || ""}
            />
          </div>
        </div>
        <fieldset className="space-y-3">
          <legend className="text-sm font-medium">{t("转化状态", "Conversion status")}</legend>
          <div className="flex flex-wrap gap-x-6 gap-y-3">
            {stages.map((stage, i) => (
              <div key={stage} className="flex items-center gap-2">
                <Checkbox
                  id={stage}
                  name="conversion_status"
                  value={stage}
                  defaultChecked={params
                    .getAll("conversion_status")
                    .includes(stage)}
                />
                <Label htmlFor={stage}>{labels[i]}</Label>
              </div>
            ))}
          </div>
        </fieldset>
          <div className="flex items-center justify-end gap-2">
            <Button type="submit">
              <Search className="size-4" />
              {t("查询", "Search")}
            </Button>
            {active.length > 0 && (
              <Button
                type="button"
                variant="ghost"
                onClick={() => update(new URLSearchParams(inRecycleBin ? "recycle_bin=true" : ""))}
              >
                {t("清除", "Clear")}
              </Button>
            )}
          </div>

        {inRecycleBin && <input type="hidden" name="recycle_bin" value="true" />}
        {params.get("starred") === "true" && (
          <input type="hidden" name="starred" value="true" />
        )}
      </form>
      {active.length > 0 && (
        <div className="flex flex-wrap gap-2 px-4 pb-4 sm:px-5">
          {active.map(([key, value]) => (
            <Badge variant="secondary" key={key + value}>
              {(
                {
                  task_id: t("任务", "Task"),
                  user: t("客户", "Customer"),
                  provider: t("供应商", "Provider"),
                  client_ip: "IP",
                  time_range: t("时间", "Time"),
                  starred: t("收藏", "Starred"),
                  conversion_status: t("转化", "Conversion"),
                } as Record<string, string>
              )[key] || key}
              :{" "}
              {key === "time_range" ? ({ today: t("今天", "Today"), yesterday: t("昨天", "Yesterday"), this_week: t("本周", "This week"), this_month: t("本月", "This month"), last_7_days: t("最近 7 天", "Last 7 days"), last_30_days: t("最近 30 天", "Last 30 days") } as Record<string, string>)[value] || value : stages.includes(value as (typeof stages)[number])
                ? labels[stages.indexOf(value as (typeof stages)[number])]
                : value}
              <button
                type="button"
                className="ml-1 px-1"
                aria-label={t("移除筛选", "Remove filter")}
                onClick={() => {
                  const next = new URLSearchParams(query);
                  next.delete(key, value);
                  next.set("page", "1");
                  update(next);
                }}
              >
                ×
              </button>
            </Badge>
          ))}
        </div>
      )}
      {result.error && (
        <ErrorBox message={result.error} retry={result.reload} />
      )}
      {result.loading ? (
        <Loading />
      ) : (
        result.data && (
          <>
            <div className="task-table-shell overflow-hidden border-t">
              <Table className="min-w-[1360px] table-fixed [&_td]:py-3 [&_td]:px-3">
                <colgroup>
                  {[3, 3, 17, 10, 26, 8, 19, 14].map((width, i) => <col key={i} style={{ width: `${width}%` }} />)}
                </colgroup>
                <TableHeader className="sticky top-0 z-10 bg-muted shadow-[0_1px_0_var(--border)] [&_th]:px-3 [&_th]:text-xs [&_th]:text-muted-foreground">
                  <TableRow>
                    <TableHead className="w-10 text-center">
                      <Checkbox aria-label={t("全选当前页", "Select current page")}
                        disabled={!visibleIds.length || deleting}
                        checked={allSelected ? true : selectedIds.length > 0 ? "indeterminate" : false}
                        onCheckedChange={(checked) => setSelected(checked === true ? visibleIds : [])} />
                    </TableHead>
                    <TableHead className="w-10">
                      <span className="sr-only">{t("收藏", "Star")}</span>
                    </TableHead>
                    {[
                      t("图片 / 任务", "Images / task"),
                      t("状态 / 模型", "Status / provider"),
                      t("客户 / 转化", "Customer / conversion"),
                      t("关联订单", "Linked orders"),
                      t("来源", "Source"),
                      t("时间 / 耗时", "Time / duration"),
                    ].map((label) => (
                      <TableHead key={label}>{label}</TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {result.data.tasks.map((task) => (
                    <TableRow key={task.task_id} data-state={selectedIds.includes(task.task_id) ? "selected" : undefined} className="hover:bg-muted/40">
                      <TableCell className="align-top text-center">
                        <Checkbox className="mt-2" aria-label={t("选择任务", "Select task") + " " + task.task_id}
                          checked={selectedIds.includes(task.task_id)} disabled={deleting || task.cleanup_started}
                          onCheckedChange={(checked) => selectTask(task.task_id, checked === true)} />
                      </TableCell>
                      <TableCell className="align-top">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="size-8"
                          disabled={busy.includes(task.task_id)}
                          aria-label={
                            task.starred
                              ? t("取消收藏", "Unstar")
                              : t("收藏任务", "Star task")
                          }
                          aria-pressed={task.starred}
                          onClick={() => star(task)}
                        >
                          <Star
                            className={
                              "size-4 " +
                              (task.starred
                                ? "fill-amber-400 text-amber-500"
                                : "text-muted-foreground")
                            }
                          />
                        </Button>
                      </TableCell>
                      <TableCell className="align-top">
                        {task.purge_after && <Badge variant="secondary" className="mb-2" title={new Date(task.purge_after).toLocaleString()}>
                          <Timer className="size-3" />{task.cleanup_started ? t("正在清理 / 等待重试", "Cleanup pending") : t(`清理日期：${new Date(task.purge_after).toLocaleDateString()}`, `Cleanup: ${new Date(task.purge_after).toLocaleDateString()}`)}
                        </Badge>}
                        <div className="flex gap-1.5 overflow-x-auto pb-2">
                          {task.images.length ? (
                            task.images.map((image, index) => (
                              <button
                                key={image.url + index}
                                className="relative shrink-0 rounded-md border bg-muted focus-visible:ring-2"
                                onClick={() =>
                                  setGallery({ images: task.images, index })
                                }
                                aria-label={
                                  t("预览图片", "Preview image") +
                                  " " +
                                  image.filename
                                }
                              >
                                <img
                                  className="size-16 rounded-md object-cover"
                                  src={image.thumbnail}
                                  loading="lazy"
                                  alt={image.filename}
                                />
                                {image.reference && (
                                  <span className="absolute inset-x-0 bottom-0 rounded-b-md bg-primary/85 text-[10px] text-white">
                                    {t("参考图", "Reference")}
                                  </span>
                                )}
                              </button>
                            ))
                          ) : (
                            <div className="grid h-16 w-24 place-items-center rounded-md bg-muted text-muted-foreground">
                              <ImageOff className="size-5" />
                            </div>
                          )}
                        </div>
                        <div className="flex items-center gap-1">
                          <button
                            className="min-w-0 truncate text-left text-xs text-muted-foreground hover:text-foreground hover:underline"
                            onClick={() => setDetail(task)}
                            title={task.task_id}
                          >
                            {task.task_id}
                          </button>
                          <CopyButton value={task.task_id} />
                        </div>
                        {task.edit_instructions && (
                          <p className="table-detail max-w-60 truncate">
                            {task.edit_instructions}
                          </p>
                        )}
                      </TableCell>
                      <TableCell className="align-top">
                        <TaskStatusBadge status={task.status} />
                        <p className="mt-2 break-words whitespace-normal text-xs leading-relaxed text-muted-foreground">
                          {task.api_provider || "—"}
                        </p>
                      </TableCell>
                      <TableCell className="align-top whitespace-normal">
                        <div className="flex min-w-0 items-center gap-1 whitespace-nowrap text-[13px]">
                          {customer(task) ? (
                            <a
                              className="min-w-0 max-w-[52%] truncate font-medium text-foreground hover:underline"
                              title={task.customer_email || task.customer_id}
                              target="_blank"
                              rel="noreferrer"
                              href={customer(task)}
                            >
                              {task.customer_email || task.customer_id}
                            </a>
                          ) : (
                            <span className="min-w-0 max-w-[52%] truncate font-medium" title={task.customer_email || task.customer_id || task.storefront_visitor_id}>
                              {task.customer_email ||
                                task.customer_id ||
                                task.storefront_visitor_id ||
                                t("匿名访客", "Anonymous visitor")}
                            </span>
                          )}
                          {task.customer_email && (
                            <CopyButton value={task.customer_email} />
                          )}
                          <span className="ml-1 text-xs text-muted-foreground">
                          {task.customer_logged_in
                            ? t("已登录客户", "Signed-in customer")
                            : t("访客", "Visitor")}{" "}
                          ·{" "}
                          {subscription[
                            task.customer_sync?.state ||
                              task.subscription_status
                          ] || t("订阅未记录", "Subscription not recorded")}
                          </span>
                        </div>
                        <div className="mt-2.5">{funnel(task)}</div>
                        <div className="mt-2.5 flex items-center gap-2 whitespace-nowrap">
                            <div className="flex shrink-0 items-center gap-1.5 text-xs">
                              <CountryFlag country={task.client_country} />
                              <span title={task.client_ip || undefined}>
                                {task.client_ip?.includes(":") && task.client_ip.length > 20
                                  ? `${task.client_ip.slice(0, 9)}…${task.client_ip.slice(-9)}`
                                  : task.client_ip || "—"}
                              </span>
                              {task.client_ip && <CopyButton value={task.client_ip} />}
                            </div>
                            <div className="flex shrink-0 items-center gap-1">
                              {task.ip_task_sequence > 0 && (
                                <Badge variant="outline" className="border-blue-200 bg-blue-50 text-blue-700 dark:border-blue-800 dark:bg-blue-950 dark:text-blue-300" title={t("该 IP 的第几条生成任务", "Generation task number for this IP")}>
                                  #{task.ip_task_sequence}
                                </Badge>
                              )}
                              {task.ip_task_total > 0 && (
                                <Badge variant="outline" className="border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-300" title={t("该 IP 的生成任务总数", "Total generation tasks for this IP")}>
                                  {t(`共 ${task.ip_task_total} 次`, `${task.ip_task_total} total`)}
                                </Badge>
                              )}
                            </div>
                        </div>
                      </TableCell>
                      <TableCell className="align-top">
                        {task.shopify_orders.length ? (
                          <div className="space-y-3">
                            {task.shopify_orders.map((order) => (
                              <div key={order.url} className="space-y-1.5">
                                <a className="inline-flex items-center gap-1 text-sm font-semibold text-primary hover:underline" href={safeUrl(order.url)} target="_blank" rel="noreferrer">
                                  {order.name}<ExternalLink className="size-3" aria-hidden="true" />
                                </a>
                                <div>
                                  <Badge variant="outline" className={order.financial_status === "PAID" ? stageColors[3] : "text-muted-foreground"}>
                                    {({
                                      PAID: t("已付款", "Paid"),
                                      PARTIALLY_PAID: t("部分付款", "Partially paid"),
                                      PENDING: t("待付款", "Pending"),
                                      AUTHORIZED: t("已授权", "Authorized"),
                                      PARTIALLY_REFUNDED: t("部分退款", "Partially refunded"),
                                      REFUNDED: t("已退款", "Refunded"),
                                      VOIDED: t("已作废", "Voided"),
                                    } as Record<string, string>)[order.financial_status] || order.financial_status || t("未知", "Unknown")}
                                  </Badge>
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : <span className="text-xs text-muted-foreground">—</span>}
                      </TableCell>
                      <TableCell className="align-top whitespace-normal">
                        <SourceInfo task={task} />
                      </TableCell>
                      <TableCell className="align-top">
                        {Object.entries(task.display_times).map(
                          ([zone, time]) => (
                            <div key={zone} className="mb-1.5 grid grid-cols-[3.5rem_1fr] items-baseline gap-2 text-xs" title={`${zone}: ${time || "—"}`}>
                              <span className="truncate text-muted-foreground">
                                {({ "Asia/Shanghai": t("上海", "Shanghai"), "America/New_York": t("纽约", "New York"), "America/Los_Angeles": t("洛杉矶", "LA") } as Record<string, string>)[zone] || zone.split("/").pop()?.replaceAll("_", " ")}
                              </span>
                              <time className="tabular-nums">
                                {time ? time.replace(/^\d{4}-/, "") : "—"}
                              </time>
                            </div>
                          ),
                        )}
                        <Badge variant="secondary" className="mt-1 gap-1" title={t("生成耗时", "Generation duration")}>
                          <Timer aria-hidden="true" />
                          {task.generation_duration_seconds != null
                            ? `${Number(task.generation_duration_seconds).toFixed(1)} s`
                            : "—"}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              {!result.data.tasks.length && (
                <div className="px-5 py-16 text-center">
                  <ImageOff className="mx-auto mb-4 size-8 text-muted-foreground" />
                  <p className="font-medium">
                    {inRecycleBin && active.length === 0 ? t("回收站为空", "Recycle bin is empty") : t("没有匹配的任务", "No matching tasks")}
                  </p>
                  <p className="page-description">
                    {inRecycleBin ? t("移入回收站的任务会在这里显示，保留 30 天。", "Recycled tasks appear here and are kept for 30 days.") : t(
                      "调整筛选条件，或等待新的图片生成任务。",
                      "Adjust your filters or wait for a new image task.",
                    )}
                  </p>
                </div>
              )}
            </div>
            <div className="flex flex-wrap items-center justify-between gap-3 border-t px-4 py-4 sm:px-5">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                {t("每页", "Rows per page")}
                <div className="w-24">
                  <Pick
                    label={t("每页条数", "Rows per page")}
                    value={String(result.data.page_size)}
                    onChange={(v) => change("page_size", v)}
                    options={["10", "25", "50", "100"].map((v) => [v, v])}
                  />
                </div>
              </div>
              <div className="flex items-center gap-3 text-sm">
                <form
                  className="flex items-center gap-2"
                  onSubmit={(e) => {
                    e.preventDefault();
                    const target = new FormData(e.currentTarget).get("page");
                    params.set("page", String(target));
                    update(params);
                  }}
                >
                  <Label htmlFor="jump-page" className="sr-only">
                    {t("跳至页码", "Go to page")}
                  </Label>
                  <Input
                    key={result.data.page}
                    id="jump-page"
                    name="page"
                    type="number"
                    min={1}
                    max={result.data.total_pages}
                    defaultValue={result.data.page}
                    className="w-20"
                    required
                  />
                  <span>/ {result.data.total_pages}</span>
                  <Button size="sm" variant="ghost">
                    {t("跳转", "Go")}
                  </Button>
                </form>
                <Button
                  variant="outline"
                  size="icon"
                  aria-label={t("上一页", "Previous page")}
                  disabled={!result.data.has_previous}
                  onClick={() => {
                    params.set("page", String(result.data!.page - 1));
                    update(params);
                  }}
                >
                  <ChevronLeft />
                </Button>
                <Button
                  variant="outline"
                  size="icon"
                  aria-label={t("下一页", "Next page")}
                  disabled={!result.data.has_next}
                  onClick={() => {
                    params.set("page", String(result.data!.page + 1));
                    update(params);
                  }}
                >
                  <ChevronRight />
                </Button>
              </div>
            </div>
          </>
        )
      )}
      </section>
      <WorkspaceStatus timezones={timezones} />
      <Sheet
        open={!!detail}
        onOpenChange={(open) => {
          if (!open) setDetail(undefined);
        }}
      >
        <SheetContent className="w-full overflow-y-auto sm:max-w-xl">
          <SheetHeader>
            <SheetTitle>{t("任务详情", "Task details")}</SheetTitle>
            <SheetDescription className="break-all">
              {detail?.task_id}
            </SheetDescription>
          </SheetHeader>
          {detail && (
            <div className="space-y-6 px-5 pb-8">
              {funnel(detail)}
              {[
                ["description", t("描述", "Description")],
                [
                  "edit_instructions",
                  t("客户修改要求", "Customer edit instructions"),
                ],
                ["prompt", t("生成提示词", "Generation prompt")],
                ["source_task_id", t("来源任务", "Source task")],
              ].map(([key, label]) => {
                const value = String(detail[key as keyof Task] || "");
                return (
                  value && (
                    <section key={key}>
                      <div className="mb-2 flex items-center justify-between">
                        <h3 className="text-sm font-semibold">{label}</h3>
                        <CopyButton value={value} />
                      </div>
                      <p className="whitespace-pre-wrap break-words rounded-md bg-muted p-4 text-sm leading-relaxed">
                        {value}
                      </p>
                    </section>
                  )
                );
              })}
              <h3 className="text-sm font-semibold">
                {t("完整任务记录", "Full task record")}
              </h3>
              <pre className="overflow-auto rounded-md bg-muted p-4 text-xs">
                {JSON.stringify(detail, null, 2)}
              </pre>
            </div>
          )}
        </SheetContent>
      </Sheet>
      <Dialog open={deleteTargets.length > 0} onOpenChange={(open) => { if (!open && !deleting) { setDeleteTargets([]); setDeleteError(""); } }}>
        <DialogContent showCloseButton={!deleting} onOpenAutoFocus={(event) => { event.preventDefault(); cancelDelete.current?.focus(); }}
          onEscapeKeyDown={(event) => { if (deleting) event.preventDefault(); }}
          onPointerDownOutside={(event) => { if (deleting) event.preventDefault(); }}>
          <DialogHeader>
            <DialogTitle>{selectionAction === "restore" ? t(`恢复 ${deleteTargets.length} 条任务？`, `Restore ${deleteTargets.length} tasks?`) : t(`将 ${deleteTargets.length} 条任务移入回收站？`, `Move ${deleteTargets.length} tasks to the recycle bin?`)}</DialogTitle>
            <DialogDescription>{selectionAction === "restore"
              ? t("所选任务将回到任务列表，保留图片、收藏和关联信息。", "Selected tasks return to the task list with their images, stars and links preserved.")
              : t("保留 30 天，期间可在回收站恢复。到期后自动永久删除任务数据和图片，相关图片链接及订单原图下载可能失效。不会删除 Shopify 订单。", "Kept for 30 days and restorable from the recycle bin. After expiry, task data and images are permanently deleted; image links and order artwork downloads may stop working. Shopify orders are not deleted.")}</DialogDescription>
          </DialogHeader>
          <ul className="max-h-36 space-y-1 overflow-auto rounded-md bg-muted p-3 text-xs text-muted-foreground">
            {deleteTargets.map((id) => <li key={id} className="break-all">{id}</li>)}
          </ul>
          {deleteError && <p role="alert" className="whitespace-pre-wrap break-words text-sm text-destructive">{deleteError}</p>}
          <DialogFooter>
            <Button ref={cancelDelete} variant="outline" disabled={deleting} onClick={() => setDeleteTargets([])}>{t("取消", "Cancel")}</Button>
            <Button variant={selectionAction === "restore" ? "default" : "destructive"} disabled={deleting} onClick={deleteSelected}>
              {deleting ? <Loader2 className="size-4 animate-spin" /> : <Trash2 className="size-4" />}
              {deleting ? t("正在处理…", "Processing…") : selectionAction === "restore" ? t("确认恢复", "Confirm restore") : t("移入回收站", "Move to recycle bin")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <Dialog
        open={!!gallery}
        onOpenChange={(open) => {
          if (!open) setGallery(undefined);
        }}
      >
        <DialogContent className="max-h-[95svh] overflow-auto sm:max-w-5xl">
          <DialogHeader>
            <DialogTitle>{t("图片预览", "Image preview")}</DialogTitle>
            <DialogDescription>
              {gallery &&
                `${gallery.index + 1} / ${gallery.images.length} · ${gallery.images[gallery.index].filename}`}
            </DialogDescription>
          </DialogHeader>
          {gallery && (
            <>
              <img
                className="max-h-[68svh] w-full rounded-md bg-muted object-contain"
                src={safeUrl(gallery.images[gallery.index].url)}
                alt={gallery.images[gallery.index].filename}
              />
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    disabled={gallery.index === 0}
                    onClick={() =>
                      setGallery({ ...gallery, index: gallery.index - 1 })
                    }
                  >
                    <ChevronLeft />
                    {t("上一张", "Previous")}
                  </Button>
                  <Button
                    variant="outline"
                    disabled={gallery.index === gallery.images.length - 1}
                    onClick={() =>
                      setGallery({ ...gallery, index: gallery.index + 1 })
                    }
                  >
                    {t("下一张", "Next")}
                    <ChevronRight />
                  </Button>
                </div>
                <Button asChild>
                  <a
                    href={safeUrl(gallery.images[gallery.index].url)}
                    target="_blank"
                    rel="noreferrer"
                  >
                    <ExternalLink className="size-4" />
                    {t("打开原图 / 下载", "Open original / download")}
                  </a>
                </Button>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
function TimeRange({ value, onChange }: { value: string; onChange?: (value: string) => void }) {
  const t = useT();
  const [selected, setSelected] = useState(value);
  useEffect(() => setSelected(value), [value]);
  return (
    <>
      <input type="hidden" name="time_range" value={selected} />
      <Pick
        label={t("时间范围", "Time range")}
        value={selected}
        onChange={(next) => { setSelected(next); onChange?.(next); }}
        options={[
          ["", t("全部时间", "All time")],
          ["today", t("今天", "Today")],
          ["yesterday", t("昨天", "Yesterday")],
          ["this_week", t("本周", "This week")],
          ["last_7_days", t("最近 7 天", "Last 7 days")],
          ["this_month", t("本月", "This month")],
          ["last_30_days", t("最近 30 天", "Last 30 days")],
        ]}
      />
    </>
  );
}

function WorkspaceStatus({ timezones }: { timezones: string[] }) {
  const t = useT();
  const health = useData<{ status: string }>("/health");
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const tick = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(tick);
  }, []);
  useEffect(() => {
    const poll = setInterval(health.reload, 60000);
    return () => clearInterval(poll);
  }, []);
  return (
    <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-muted-foreground">
      <span className="flex items-center gap-2">
        <span
          className={
            "size-1.5 rounded-full " +
            (health.error
              ? "bg-amber-500"
              : health.data
                ? "bg-emerald-500"
                : "bg-slate-400")
          }
        />
        {health.error
          ? t("服务状态暂不可用", "Service status unavailable")
          : health.data
            ? t("服务在线", "Service online")
            : t("检查服务状态…", "Checking service…")}
      </span>
      {timezones.map((zone) => (
        <span key={zone}>
          {zone}{" "}
          <time className="ml-1 tabular-nums">
            {new Intl.DateTimeFormat("en-GB", {
              timeZone: zone,
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
            }).format(now)}
          </time>
        </span>
      ))}
    </div>
  );
}

function CountryFlag({ country }: { country: string }) {
  const code = (country || "").trim().toUpperCase();
  if (!/^[A-Z]{2}$/.test(code) || ["XX", "T1"].includes(code)) return null;
  return <img key={code} src={`https://flagcdn.com/w40/${code.toLowerCase()}.png`} width={20} height={15} alt={code} title={code} loading="lazy" decoding="async" referrerPolicy="no-referrer" className="h-[15px] w-5 shrink-0 rounded-[2px] object-cover" onError={(event) => { event.currentTarget.hidden = true; }} />;
}
