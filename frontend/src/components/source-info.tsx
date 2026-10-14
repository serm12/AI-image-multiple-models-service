import { useState } from "react";
import { Globe } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { useT } from "@/lib/admin";

type Source = {
  source_product_title?: string; source_page_title?: string; source_page_url?: string;
  request_url?: string; traffic_source?: string; traffic_referrer?: string;
  utm_source?: string; utm_medium?: string; utm_campaign?: string;
};
function webUrl(value?: string): URL | undefined {
  try {
    const url = new URL(value || "");
    if (["https:", "http:"].includes(url.protocol) && !url.username && !url.password) return url;
  } catch { /* Missing or invalid URL. */ }
}
function SiteIcon({ host }: { host: string }) {
  const [failed, setFailed] = useState(false);
  return failed ? <Globe className="size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" /> :
    <img src={`https://www.google.com/s2/favicons?domain=${encodeURIComponent(host)}&sz=32`} width={14} height={14} alt="" aria-hidden="true" loading="lazy" referrerPolicy="no-referrer" className="size-3.5 shrink-0 object-contain" onError={() => setFailed(true)} />;
}
const channels: [RegExp, string, string][] = [
  [/^(meta|facebook|fb|facebook_ads|meta_ads)$|(^|\.)facebook\.com$/, "Meta / Facebook", "facebook.com"],
  [/^(instagram|ig)$|(^|\.)instagram\.com$/, "Instagram", "instagram.com"],
  [/^(google|google_ads|adwords)$|(^|\.)google\.[a-z.]+$/, "Google", "google.com"],
  [/^youtube$|(^|\.)youtube\.com$/, "YouTube", "youtube.com"],
  [/^(bing|microsoft)$|(^|\.)bing\.com$/, "Bing", "bing.com"],
  [/^tiktok$|(^|\.)tiktok\.com$/, "TikTok", "tiktok.com"],
  [/^pinterest$|(^|\.)pinterest\.com$/, "Pinterest", "pinterest.com"],
  [/^duckduckgo$|(^|\.)duckduckgo\.com$/, "DuckDuckGo", "duckduckgo.com"],
];
export function SourceInfo({ task }: { task: Source }) {
  const t = useT();
  const page = webUrl(task.source_page_url);
  const request = webUrl(task.request_url);
  const source = task.traffic_source || "";
  const sourceReferrer = /^referrer:/i.test(source) ? source.replace(/^referrer:/i, "") : "";
  const referrer = webUrl(task.traffic_referrer) || (sourceReferrer ? webUrl(sourceReferrer.includes("://") ? sourceReferrer : `https://${sourceReferrer}`) : undefined);
  const candidates = [task.utm_source, source.replace(/^(utm|referrer):/i, ""), referrer?.hostname];
  const channel = candidates.flatMap(value => value ? channels.filter(([pattern]) => pattern.test(value.toLowerCase())) : [])[0];
  const paid = /(^|[\s_:-])(paid|cpc|ppc|cpv|cpm|display|ads|adwords)([\s_:-]|$)/i.test(`${task.utm_medium || ""} ${source}`);
  const title = task.source_product_title || task.source_page_title || page?.hostname || "—";
  function link(url: URL, label: string, icon = false) {
    return <a href={url.href} target="_blank" rel="noreferrer" title={url.href} className="inline-flex min-w-0 max-w-full items-center gap-1.5 text-primary hover:underline">
      {icon && <SiteIcon key={url.hostname} host={url.hostname} />}<span className="truncate">{label}</span>
    </a>;
  }
  return <div className="space-y-1 text-xs">
    <div className="font-medium">{page ? link(page, title) : title}</div>
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="text-muted-foreground">{t("流量", "Traffic")}</span>
      {(channel || referrer) && <SiteIcon key={channel?.[2] || referrer?.hostname} host={channel?.[2] || referrer!.hostname} />}
      <span title={source}>{channel?.[1] || source || t("直接访问", "Direct")}</span>
      {paid && <Badge variant="secondary">{t("付费", "Paid")}</Badge>}
    </div>
    {referrer && <div className="flex min-w-0 items-center gap-1.5"><span className="shrink-0 text-muted-foreground">{t("引荐", "Referrer")}</span>{link(referrer, referrer.host + (referrer.pathname === "/" ? "" : referrer.pathname), true)}</div>}
    {[ ["source", task.utm_source], ["medium", task.utm_medium], ["campaign", task.utm_campaign] ].some(([, value]) => value) &&
      <p className="break-words text-muted-foreground">{[["source", task.utm_source], ["medium", task.utm_medium], ["campaign", task.utm_campaign]].filter(([, value]) => value).map(([key, value]) => `${key}=${value}`).join(" · ")}</p>}
    <div className="flex min-w-0 items-center gap-1.5"><span className="shrink-0 text-muted-foreground">{t("来源", "Page")}</span>{page ? link(page, page.host + page.pathname, true) : "—"}</div>
    <div className="flex min-w-0 items-center gap-1.5"><span className="shrink-0 text-muted-foreground">{t("请求", "Request")}</span>{request ? link(request, request.pathname) : "—"}</div>
  </div>;
}
