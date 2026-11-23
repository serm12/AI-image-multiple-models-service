import { useState } from "react";
import { createPortal } from "react-dom";
import { Palette, Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";
import { useT, useData, Loading, ErrorBox, CopyButton } from "@/lib/admin";
export function Styles({ searchTarget }: { searchTarget: HTMLDivElement | null }) {
  const t = useT();
  const result =
    useData<{ id: string; description: string; prompt: string }[]>(
      "/admin/api/styles",
    );
  const [query, setQuery] = useState("");
  const styles = result.data?.filter((item) =>
    [item.id, item.description, item.prompt].some((value) =>
      value.toLowerCase().includes(query.toLowerCase()),
    ),
  );
  return (
    <>
      {searchTarget && createPortal(
        <div className="flex min-w-0 items-center gap-2">
          <Search className="hidden size-4 shrink-0 text-muted-foreground sm:block" />
          <Input
            className="min-w-0"
            aria-label={t("搜索风格", "Search styles")}
            placeholder={t("搜索风格或提示词…", "Search styles or prompts…")}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <Badge variant="secondary">{styles?.length || 0}</Badge>
        </div>,
        searchTarget,
      )}
      {result.error && (
        <ErrorBox message={result.error} retry={result.reload} />
      )}{" "}
      {result.loading ? (
        <Loading />
      ) : (
        <div className="grid gap-5 xl:grid-cols-2">
          {styles?.map((style) => (
            <Card key={style.id} className="shadow-none">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <Palette className="size-5 text-primary" />
                  <CopyButton value={style.prompt} />
                </div>
                <CardTitle>{style.description || style.id}</CardTitle>
                <CardDescription>{style.id}</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="max-h-64 overflow-auto whitespace-pre-wrap text-sm leading-7 text-muted-foreground">
                  {style.prompt || t("未配置提示词", "No prompt configured")}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
      {!result.loading && styles?.length === 0 && (
        <p className="py-12 text-muted-foreground">
          {t(
            "没有匹配的风格，请尝试其他关键词。",
            "No matching styles. Try another search.",
          )}
        </p>
      )}
    </>
  );
}
