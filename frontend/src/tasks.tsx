import { useEffect, useState, type FormEvent } from "react";
import { toast } from "sonner";
import {
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
  const [detail, setDetail] = useState<Task>();
  const [gallery, setGallery] = useState<{ images: Image[]; index: number }>();
  const [filters, setFilters] = useState(false);
  const [busy, setBusy] = useState<string[]>([]);
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
    setBusy((v) => [...v, task.task_id]);
    try {
      await post(
        "/admin/api/tasks/" + encodeURIComponent(task.task_id) + "/star",
        { starred: !task.starred },
      );
      result.reload();
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setBusy((v) => v.filter((id) => id !== task.task_id));
    }
  }
  const params = new URLSearchParams(query);
  const active = [...params.entries()].filter(
    ([key, value]) => !["page", "page_size"].includes(key) && value,
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
                : "text-muted-foreground/65"
            }
            title={task.storefront_event_times?.[stage] || ""}
          >
            <Icon aria-hidden="true" />
            {labels[i]}
            {task[stage] ? " ✓" : ""}
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
  const paid = (task: Task) =>
    /cpc|ppc|paid|ads/i.test(task.utm_medium + " " + task.traffic_source);
  return (
    <>
      <div className="mb-7 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="page-heading">{t("任务记录", "Image tasks")}</h1>
          <p className="page-description">
            {t(
              "查看生成结果、客户来源与订单转化。",
              "Review generated images, customer sources and order conversions.",
            )}
          </p>
        </div>
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
      </div>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <span className="text-sm text-muted-foreground">
          {result.data
            ? `${result.data.total.toLocaleString()} ${t("条任务", "tasks")}`
            : t("正在查询", "Loading tasks")}
        </span>
        <div className="flex gap-2">
          <Button
            variant={params.get("starred") === "true" ? "default" : "outline"}
            onClick={() =>
              change("starred", params.get("starred") === "true" ? "" : "true")
            }
          >
            <Star className="size-4" />
            {t("收藏", "Starred")}
          </Button>
          <Button
            variant="outline"
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
        </div>
      </div>
      <form
        id="task-filters"
        hidden={!filters}
        key={query}
        onSubmit={search}
        className="mb-5 rounded-lg border bg-white p-4"
      >
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
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
          <div className="flex items-end gap-2">
            <Button type="submit">
              <Search className="size-4" />
              {t("查询", "Search")}
            </Button>
            {active.length > 0 && (
              <Button
                type="button"
                variant="ghost"
                onClick={() => update(new URLSearchParams())}
              >
                {t("清除", "Clear")}
              </Button>
            )}
          </div>
        </div>
        <div
          className="mt-4 grid gap-4 border-t pt-4 sm:grid-cols-2"
        >
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
          <div className="flex flex-wrap gap-4 sm:col-span-2">
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
        </div>
        {params.get("starred") === "true" && (
          <input type="hidden" name="starred" value="true" />
        )}
      </form>
      {active.length > 0 && (
        <div className="mb-4 flex flex-wrap gap-2">
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
              {stages.includes(value as (typeof stages)[number])
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
            <div className="overflow-hidden rounded-lg border bg-white">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-10">
                      <span className="sr-only">{t("收藏", "Star")}</span>
                    </TableHead>
                    {[
                      t("图片 / 任务", "Images / task"),
                      t("状态 / 模型", "Status / provider"),
                      t("客户 / 转化", "Customer / conversion"),
                      t("来源", "Source"),
                      t("时间 / 耗时", "Time / duration"),
                    ].map((label) => (
                      <TableHead key={label}>{label}</TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {result.data.tasks.map((task) => (
                    <TableRow key={task.task_id}>
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
                      <TableCell className="min-w-64 max-w-80 align-top">
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
                            className="max-w-64 truncate text-left text-xs font-medium text-primary hover:underline"
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
                        <Button
                          variant="ghost"
                          size="sm"
                          className="-ml-3 h-7 text-xs text-muted-foreground"
                          onClick={() => setDetail(task)}
                        >
                          {t("查看详情与提示词", "Details & prompt")}
                        </Button>
                      </TableCell>
                      <TableCell className="align-top">
                        <TaskStatusBadge status={task.status} />
                        <p className="mt-2 max-w-36 break-words whitespace-normal text-xs">
                          {task.api_provider || "—"}
                        </p>
                      </TableCell>
                      <TableCell className="min-w-64 max-w-80 align-top whitespace-normal">
                        <div className="flex items-center gap-1 text-xs">
                          {customer(task) ? (
                            <a
                              className="break-all text-primary hover:underline"
                              target="_blank"
                              rel="noreferrer"
                              href={customer(task)}
                            >
                              {task.customer_email || task.customer_id}
                            </a>
                          ) : (
                            <span className="break-all">
                              {task.customer_email ||
                                task.customer_id ||
                                task.storefront_visitor_id ||
                                t("匿名访客", "Anonymous visitor")}
                            </span>
                          )}
                          {task.customer_email && (
                            <CopyButton value={task.customer_email} />
                          )}
                        </div>
                        <p className="table-detail">
                          {task.customer_logged_in
                            ? t("已登录客户", "Signed-in customer")
                            : t("访客", "Visitor")}{" "}
                          ·{" "}
                          {subscription[
                            task.customer_sync?.state ||
                              task.subscription_status
                          ] || t("订阅未记录", "Subscription not recorded")}
                        </p>
                        <div className="mt-3">{funnel(task)}</div>
                        {task.shopify_orders.map((order) => (
                          <a
                            key={order.url}
                            className="mt-2 block text-xs text-primary hover:underline"
                            href={safeUrl(order.url)}
                            target="_blank"
                            rel="noreferrer"
                          >
                            {order.name} · {order.financial_status} ↗
                          </a>
                        ))}
                        <div className="mt-3 flex flex-wrap items-start gap-4">
                          <div className="min-w-0">
                            <div className="flex items-center gap-1 text-xs">
                              <span className="break-all">{task.client_ip || "—"}</span>
                              {task.client_ip && <CopyButton value={task.client_ip} />}
                            </div>
                            <div className="mt-1 flex flex-wrap gap-1">
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
                          <CountryBadge country={task.client_country} />
                        </div>
                      </TableCell>
                      <TableCell className="max-w-52 align-top whitespace-normal">
                        <a
                          className="line-clamp-2 text-xs text-primary hover:underline"
                          href={safeUrl(task.source_page_url)}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {task.source_product_title ||
                            task.source_page_title ||
                            task.source_page_url ||
                            "—"}
                        </a>
                        <p className="mt-2 text-xs">
                          {task.traffic_source || t("直接访问", "Direct")}
                          {paid(task) && (
                            <Badge className="ml-1" variant="secondary">
                              {t("付费", "Paid")}
                            </Badge>
                          )}
                        </p>
                        <p className="table-detail break-all">
                          {[task.utm_source, task.utm_medium, task.utm_campaign]
                            .filter(Boolean)
                            .join(" · ") || task.traffic_referrer}
                        </p>
                      </TableCell>
                      <TableCell className="min-w-40 align-top">
                        {Object.entries(task.display_times).map(
                          ([zone, time]) => (
                            <div key={zone} className="mb-2">
                              <p className="text-[10px] text-muted-foreground">
                                {zone}
                              </p>
                              <p className="text-xs tabular-nums">
                                {time || "—"}
                              </p>
                            </div>
                          ),
                        )}
                        <p className="table-detail">
                          {task.generation_duration_seconds != null
                            ? `${Number(task.generation_duration_seconds).toFixed(1)} s`
                            : "—"}
                        </p>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              {!result.data.tasks.length && (
                <div className="px-5 py-16 text-center">
                  <ImageOff className="mx-auto mb-4 size-8 text-muted-foreground" />
                  <p className="font-medium">
                    {t("没有匹配的任务", "No matching tasks")}
                  </p>
                  <p className="page-description">
                    {t(
                      "调整筛选条件，或等待新的图片生成任务。",
                      "Adjust your filters or wait for a new image task.",
                    )}
                  </p>
                </div>
              )}
            </div>
            <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
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
function TimeRange({ value }: { value: string }) {
  const t = useT();
  const [selected, setSelected] = useState(value);
  return (
    <>
      <input type="hidden" name="time_range" value={selected} />
      <Pick
        label={t("时间范围", "Time range")}
        value={selected}
        onChange={setSelected}
        options={[
          ["", t("全部时间", "All time")],
          ["today", t("今天", "Today")],
          ["yesterday", t("昨天", "Yesterday")],
          ["last_7_days", t("最近 7 天", "Last 7 days")],
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

function CountryBadge({ country }: { country: string }) {
  const code = (country || "").trim().toUpperCase();
  const flag = /^[A-Z]{2}$/.test(code) && !["XX", "T1"].includes(code);
  return (
    <Badge variant="secondary" className="gap-1.5" title={code || undefined}>
      {flag && <img key={code} src={`https://flagcdn.com/w40/${code.toLowerCase()}.png`} width={20} height={15} alt="" aria-hidden="true" loading="lazy" decoding="async" referrerPolicy="no-referrer" className="h-[15px] w-5 shrink-0 rounded-[2px] object-cover" onError={(event) => { event.currentTarget.hidden = true; }} />}
      {code || "—"}
    </Badge>
  );
}
