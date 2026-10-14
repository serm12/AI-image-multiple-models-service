import { useEffect, useState, type ReactNode } from "react";
import { toast } from "sonner";
import {
  Save,
  Eye,
  Upload,
  Plus,
  X,
  CheckCircle2,
  KeyRound,
  Globe2,
  Clock3,
  Stamp,
  Hash,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { useT, useData, post, Pick, ErrorBox, Loading } from "@/lib/admin";
export function Settings() {
  const t = useT();
  const items = [
    ["watermark", t("水印与品牌", "Watermark & branding"), Stamp],
    ["time", t("时区", "Timezones"), Clock3],
    ["keys", t("模型密钥", "Provider keys"), KeyRound],
    ["algorithm", t("文件命名", "File naming"), Hash],
    ["language", t("语言", "Language"), Globe2],
  ] as const;
  const initial = location.hash.slice(1);
  return (
    <>
      <h1 className="page-heading">{t("系统设置", "Settings")}</h1>
      <p className="page-description mb-7">
        {t(
          "管理生成图片的显示效果与后台偏好。",
          "Manage image presentation and workspace preferences.",
        )}
      </p>
      <Tabs
        defaultValue={
          items.some(([key]) => key === initial) ? initial : "watermark"
        }
        onValueChange={(value) => history.replaceState(null, "", "#" + value)}
      >
        <TabsList className="mb-6 h-auto! max-w-full flex-wrap justify-start gap-1">
          {items.map(([key, label, Icon]) => (
            <TabsTrigger className="h-auto flex-none gap-2 px-4 py-1.5 data-active:bg-card data-active:text-foreground data-active:shadow-xs" key={key} value={key}>
              <Icon className="size-4" />
              {label}
            </TabsTrigger>
          ))}
        </TabsList>
        <TabsContent value="watermark">
          <Watermark />
        </TabsContent>
        <TabsContent value="time">
          <TimeSettings />
        </TabsContent>
        <TabsContent value="keys">
          <ProviderKeys />
        </TabsContent>
        <TabsContent value="algorithm">
          <Algorithm />
        </TabsContent>
        <TabsContent value="language">
          <Language />
        </TabsContent>
      </Tabs>
    </>
  );
}
function SaveButton({ busy }: { busy: boolean }) {
  const t = useT();
  return (
    <Button type="submit" disabled={busy}>
      <Save className="size-4" />
      {busy ? t("正在保存…", "Saving…") : t("保存设置", "Save settings")}
    </Button>
  );
}
function Field({
  label,
  children,
  id,
}: {
  label: string;
  children: ReactNode;
  id?: string;
}) {
  return (
    <div className="field">
      <Label htmlFor={id}>{label}</Label>
      {children}
    </div>
  );
}
function useSave() {
  const [busy, setBusy] = useState(false);
  const t = useT();
  async function save(work: () => Promise<unknown>) {
    setBusy(true);
    try {
      await work();
      window.dispatchEvent(new Event("admin:settings-saved"));
      toast.success(t("设置已保存", "Settings saved"));
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return { busy, save };
}
type WatermarkValue = {
  layout: string;
  opacity: number;
  size: number;
  angle: number;
  gap_x: number;
  gap_y: number;
  corner_text: string;
  corner_position: string;
  corner_enabled: boolean;
  brand_enabled: boolean;
  brand_scale: number;
  brand_opacity: number;
};
function Watermark() {
  const t = useT();
  const result = useData<WatermarkValue>("/admin/watermark/settings");
  const [value, setValue] = useState<WatermarkValue>();
  const [files, setFiles] = useState<{ watermark?: File; brand?: File }>({});
  const [preview, setPreview] = useState("");
  const [previewBusy, setPreviewBusy] = useState(false);
  const saving = useSave();
  useEffect(() => {
    if (result.data) setValue(result.data);
  }, [result.data]);
  useEffect(
    () => () => {
      if (preview) URL.revokeObjectURL(preview);
    },
    [preview],
  );
  function patch<K extends keyof WatermarkValue>(key: K, v: WatermarkValue[K]) {
    setValue((current) => (current ? { ...current, [key]: v } : current));
  }
  function payload() {
    const form = new FormData();
    form.set("settings", JSON.stringify(value));
    for (const [key, file] of Object.entries(files))
      if (file) form.set(key, file);
    return form;
  }
  async function request(path: string) {
    const response = await fetch("/admin/watermark/" + path, {
      method: "POST",
      body: payload(),
    });
    if (response.redirected) {
      location.assign("/admin/login");
      throw new Error(t("请重新登录", "Please sign in again"));
    }
    if (!response.ok) {
      let message = t(
        "操作失败，请检查设置后重试",
        "Request failed. Check your settings and retry.",
      );
      try {
        const data = await response.json();
        if (typeof data.detail === "string") message = data.detail;
      } catch {
        /* status fallback */
      }
      throw new Error(message);
    }
    return response;
  }
  async function renderPreview() {
    setPreviewBusy(true);
    try {
      const response = await request("preview");
      setPreview(URL.createObjectURL(await response.blob()));
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setPreviewBusy(false);
    }
  }
  if (result.error)
    return <ErrorBox message={result.error} retry={result.reload} />;
  if (!value) return <Loading />;
  const numbers: [keyof WatermarkValue, string, number, number, number][] = [
    ["opacity", t("水印透明度", "Watermark opacity"), 0, 1, 0.01],
    ["size", t("水印宽度（像素）", "Watermark width (px)"), 40, 1000, 1],
    ["angle", t("旋转角度", "Rotation angle"), -180, 180, 1],
    ["gap_x", t("水平间距", "Horizontal spacing"), 10, 1000, 1],
    ["gap_y", t("垂直间距", "Vertical spacing"), 10, 1000, 1],
  ];
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        void saving.save(async () => {
          await request("settings");
        });
      }}
      className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(300px,.7fr)]"
    >
      <div className="space-y-5">
        <section className="panel space-y-5">
          <h2 className="font-semibold">{t("水印布局", "Watermark layout")}</h2>
          <Field label={t("排列方式", "Layout")}>
            <Pick
              label={t("排列方式", "Layout")}
              value={value.layout}
              onChange={(v) => patch("layout", v)}
              options={[
                ["uniform_diagonal", t("斜向平铺", "Diagonal tiles")],
                ["grid", t("网格平铺", "Grid")],
                ["random", t("随机排列", "Random")],
                ["center", t("居中", "Centered")],
              ]}
            />
          </Field>
          <div className="grid gap-4 sm:grid-cols-2">
            {numbers.map(([key, label, min, max, step]) => (
              <Field key={key} label={label} id={key}>
                <Input
                  id={key}
                  type="number"
                  required
                  min={min}
                  max={max}
                  step={step}
                  value={String(value[key])}
                  onChange={(e) =>
                    patch(
                      key,
                      e.target.value === ""
                        ? ("" as unknown as number)
                        : Number(e.target.value),
                    )
                  }
                />
              </Field>
            ))}
          </div>
          <AssetUpload
            kind="watermark"
            file={files.watermark}
            onChange={(file) => setFiles({ ...files, watermark: file })}
          />
        </section>
        <section className="panel space-y-5">
          <div className="flex items-center justify-between">
            <Label htmlFor="corner_enabled">
              {t("角标文字", "Corner label")}
            </Label>
            <Switch
              id="corner_enabled"
              checked={value.corner_enabled}
              onCheckedChange={(v) => patch("corner_enabled", v)}
            />
          </div>
          <Field label={t("角标内容", "Label text")} id="corner_text">
            <Input
              id="corner_text"
              maxLength={200}
              value={value.corner_text}
              onChange={(e) => patch("corner_text", e.target.value)}
            />
          </Field>
          <Field label={t("角标位置", "Position")}>
            <Pick
              label={t("角标位置", "Position")}
              value={value.corner_position}
              onChange={(v) => patch("corner_position", v)}
              options={[
                ["top_left", t("左上角", "Top left")],
                ["top_right", t("右上角", "Top right")],
              ]}
            />
          </Field>
        </section>
        <section className="panel space-y-5">
          <div className="flex items-center justify-between">
            <Label htmlFor="brand_enabled">{t("品牌标识", "Brand logo")}</Label>
            <Switch
              id="brand_enabled"
              checked={value.brand_enabled}
              onCheckedChange={(v) => patch("brand_enabled", v)}
            />
          </div>
          <AssetUpload
            kind="brand"
            file={files.brand}
            onChange={(file) => setFiles({ ...files, brand: file })}
          />
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label={t("品牌标识比例", "Brand scale")} id="brand_scale">
              <Input
                id="brand_scale"
                type="number"
                required
                min={0.05}
                max={0.6}
                step={0.01}
                value={value.brand_scale}
                onChange={(e) => patch("brand_scale", Number(e.target.value))}
              />
            </Field>
            <Field
              label={t("品牌标识透明度", "Brand opacity")}
              id="brand_opacity"
            >
              <Input
                id="brand_opacity"
                type="number"
                required
                min={0}
                max={1}
                step={0.01}
                value={value.brand_opacity}
                onChange={(e) => patch("brand_opacity", Number(e.target.value))}
              />
            </Field>
          </div>
        </section>
        <SaveButton busy={saving.busy} />
      </div>
      <section className="panel xl:sticky xl:top-6">
        <div className="mb-5 flex items-center justify-between">
          <h2 className="font-semibold">{t("效果预览", "Preview")}</h2>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={previewBusy}
            onClick={(e) => {
              if (e.currentTarget.form?.reportValidity()) void renderPreview();
            }}
          >
            <Eye className="size-4" />
            {previewBusy
              ? t("生成中…", "Rendering…")
              : t("更新预览", "Update preview")}
          </Button>
        </div>
        {preview ? (
          <img
            src={preview}
            alt={t("水印效果预览", "Watermark preview")}
            className="max-h-[65vh] w-full rounded-md object-contain"
          />
        ) : (
          <div className="grid aspect-[9/11] place-content-center rounded-md bg-muted p-8 text-center">
            <Stamp className="mx-auto mb-4 size-8 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">
              {t(
                "调整参数后点击“更新预览”查看效果。",
                "Adjust your settings, then select Update preview.",
              )}
            </p>
          </div>
        )}
        <p className="mt-4 text-xs leading-5 text-muted-foreground">
          {t(
            "预览不会保存设置。保存后，新生成的图片使用此配置。",
            "Preview does not save changes. Saved settings apply to newly generated images.",
          )}
        </p>
      </section>
    </form>
  );
}
function AssetUpload({
  kind,
  file,
  onChange,
}: {
  kind: "watermark" | "brand";
  file?: File;
  onChange: (file: File) => void;
}) {
  const t = useT();
  const [url, setUrl] = useState("/admin/watermark/assets/" + kind);
  useEffect(() => {
    if (!file) return;
    const object = URL.createObjectURL(file);
    setUrl(object);
    return () => URL.revokeObjectURL(object);
  }, [file]);
  return (
    <div className="flex flex-wrap items-center gap-4 rounded-md border border-dashed p-4">
      <img
        src={url}
        className="size-16 rounded bg-muted object-contain"
        alt={
          kind === "brand"
            ? t("品牌标识", "Brand logo")
            : t("水印素材", "Watermark asset")
        }
        onError={(e) => {
          e.currentTarget.style.visibility = "hidden";
        }}
        onLoad={(e) => {
          e.currentTarget.style.visibility = "visible";
        }}
      />
      <div className="min-w-0 flex-1">
        <Label htmlFor={kind} className="mb-2">
          <Upload className="size-4" />
          {kind === "brand"
            ? t("上传品牌标识", "Upload brand logo")
            : t("上传水印", "Upload watermark")}
        </Label>
        <Input
          id={kind}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          onChange={(e) => {
            const next = e.target.files?.[0];
            if (!next) return;
            if (next.size > 5 * 1024 * 1024) {
              toast.error(
                t("图片不能超过 5 MB", "Image must be 5 MB or smaller"),
              );
              e.target.value = "";
              return;
            }
            onChange(next);
          }}
        />
        <p className="mt-2 text-xs text-muted-foreground">
          PNG / JPEG / WebP · ≤ 5 MB
        </p>
      </div>
    </div>
  );
}
type TimeValue = {
  effective_timezone: string;
  shop_timezone: string;
  comparison_timezones: string[];
  timezones: string[];
  sync_failed: boolean;
};
function TimeSettings() {
  const t = useT();
  const result = useData<TimeValue>("/admin/time-settings");
  const [zone, setZone] = useState("");
  const [comparisons, setComparisons] = useState<string[]>([]);
  const saving = useSave();
  useEffect(() => {
    if (result.data) {
      setZone(result.data.effective_timezone);
      setComparisons(result.data.comparison_timezones);
    }
  }, [result.data]);
  if (result.error)
    return <ErrorBox message={result.error} retry={result.reload} />;
  if (!result.data) return <Loading />;
  const options = result.data.timezones.map(
    (v) =>
      [
        v,
        v +
          (v === result.data!.shop_timezone
            ? t("（店铺默认）", " (Shop default)")
            : ""),
      ] as [string, string],
  );
  return (
    <form
      className="panel max-w-2xl space-y-6"
      onSubmit={(e) => {
        e.preventDefault();
        void saving.save(async () => {
          const saved = await post<TimeValue>("/admin/time-settings", {
            timezone: zone,
            comparison_timezones: comparisons,
          });
          result.setData(saved);
        });
      }}
    >
      <h2 className="font-semibold">{t("时间显示", "Time display")}</h2>
      <p className="text-sm text-muted-foreground">
        {t("店铺时区", "Shop timezone")}: {result.data.shop_timezone || "UTC"}
      </p>
      {result.data.sync_failed && (
        <ErrorBox
          message={t(
            "店铺时区同步失败，当前显示已保存的时区。",
            "Shop timezone sync failed. Showing the saved timezone.",
          )}
          retry={result.reload}
        />
      )}
      <Field label={t("主要时区", "Primary timezone")}>
        <Pick
          label={t("主要时区", "Primary timezone")}
          value={zone}
          onChange={setZone}
          options={options}
        />
      </Field>
      <div className="space-y-3">
        <Label>
          {t("对照时区（最多 5 个）", "Comparison timezones (up to 5)")}
        </Label>
        {comparisons.map((value, index) => (
          <div key={index} className="flex gap-2">
            <Pick
              label={t("对照时区", "Comparison timezone")}
              value={value}
              onChange={(v) =>
                setComparisons((current) =>
                  current.map((item, i) => (i === index ? v : item)),
                )
              }
              options={options}
            />
            <Button
              type="button"
              size="icon"
              variant="outline"
              aria-label={t("移除时区", "Remove timezone")}
              onClick={() =>
                setComparisons((current) =>
                  current.filter((_, i) => i !== index),
                )
              }
            >
              <X className="size-4" />
            </Button>
          </div>
        ))}
        <Button
          type="button"
          variant="outline"
          disabled={comparisons.length >= 5}
          onClick={() =>
            setComparisons([
              ...comparisons,
              result.data!.timezones.find(
                (item) => item !== zone && !comparisons.includes(item),
              ) || "UTC",
            ])
          }
        >
          <Plus className="size-4" />
          {t("添加时区", "Add timezone")}
        </Button>
      </div>
      <SaveButton busy={saving.busy} />
    </form>
  );
}
function ProviderKeys() {
  const t = useT();
  const result = useData<Record<string, { configured: boolean }>>(
    "/admin/provider-credentials",
  );
  const [values, setValues] = useState<Record<string, string>>({});
  const saving = useSave();
  if (result.error)
    return <ErrorBox message={result.error} retry={result.reload} />;
  if (!result.data) return <Loading />;
  return (
    <form
      className="panel max-w-2xl space-y-6"
      onSubmit={(e) => {
        e.preventDefault();
        void saving.save(async () => {
          result.setData(await post("/admin/provider-credentials", values));
          setValues({});
        });
      }}
    >
      <h2 className="font-semibold">
        {t("模型供应商密钥", "Provider credentials")}
      </h2>
      <p className="text-sm text-muted-foreground">
        {t(
          "留空保留现有密钥。已保存的密钥不会回显。",
          "Leave blank to keep existing keys. Saved keys are never displayed.",
        )}
      </p>
      {Object.entries(result.data).map(([name, status]) => (
        <div className="field" key={name}>
          <div className="flex items-center justify-between">
            <Label htmlFor={name}>
              {name === "FAL_API_KEY" ? "fal.ai" : "AIAPIROUTE"}
            </Label>
            <Badge variant="secondary">
              {status.configured ? (
                <>
                  <CheckCircle2 className="size-3" />
                  {t("已配置", "Configured")}
                </>
              ) : (
                t("未配置", "Not configured")
              )}
            </Badge>
          </div>
          <Input
            id={name}
            type="password"
            autoComplete="new-password"
            maxLength={4096}
            value={values[name] || ""}
            placeholder={t("输入新密钥", "Enter a new key")}
            onChange={(e) => setValues({ ...values, [name]: e.target.value })}
          />
        </div>
      ))}
      <SaveButton busy={saving.busy} />
    </form>
  );
}
function Algorithm() {
  const t = useT();
  const result = useData<{
    factor: number;
    year: number;
    algorithm_value: number;
  }>("/admin/algorithm-settings");
  const [factor, setFactor] = useState("");
  const saving = useSave();
  useEffect(() => {
    if (result.data) setFactor(String(result.data.factor));
  }, [result.data]);
  if (result.error)
    return <ErrorBox message={result.error} retry={result.reload} />;
  if (!result.data) return <Loading />;
  return (
    <form
      className="panel max-w-2xl space-y-6"
      onSubmit={(e) => {
        e.preventDefault();
        void saving.save(async () => {
          result.setData(
            await post("/admin/algorithm-settings", { factor: Number(factor) }),
          );
        });
      }}
    >
      <h2 className="font-semibold">
        {t("原图文件命名", "Original image file naming")}
      </h2>
      <Field label={t("算法因子", "Algorithm factor")} id="factor">
        <Input
          id="factor"
          type="number"
          required
          min={1}
          max={999999999}
          step={1}
          value={factor}
          onChange={(e) => setFactor(e.target.value)}
        />
      </Field>
      <div className="rounded-md bg-muted p-4 text-sm tabular-nums">
        {result.data.year} × {factor || "—"} ={" "}
        {factor ? result.data.year * Number(factor) : "—"}
      </div>
      <SaveButton busy={saving.busy} />
    </form>
  );
}
function Language() {
  const t = useT();
  const result = useData<{ locale: string; languages: Record<string, string> }>(
    "/admin/language-settings",
  );
  const [locale, setLocale] = useState("");
  const saving = useSave();
  useEffect(() => {
    if (result.data) setLocale(result.data.locale);
  }, [result.data]);
  if (result.error)
    return <ErrorBox message={result.error} retry={result.reload} />;
  if (!result.data) return <Loading />;
  return (
    <form
      className="panel max-w-2xl space-y-6"
      onSubmit={(e) => {
        e.preventDefault();
        void saving.save(async () => {
          await post("/admin/language-settings", { locale });
          location.reload();
        });
      }}
    >
      <h2 className="font-semibold">{t("界面语言", "Interface language")}</h2>
      <p className="text-sm text-muted-foreground">
        {t(
          "更改后台显示语言，任务内容保持原文。",
          "Change the admin language. Task content keeps its original language.",
        )}
      </p>
      <Field label={t("语言", "Language")}>
        <Pick
          label={t("语言", "Language")}
          value={locale}
          onChange={setLocale}
          options={Object.entries(result.data.languages)}
        />
      </Field>
      <SaveButton busy={saving.busy} />
    </form>
  );
}
