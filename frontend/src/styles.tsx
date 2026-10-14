import { useState } from "react";
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
export function Styles() {
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
      <div className="mb-7">
        <h1 className="page-heading">{t("风格目录", "Style library")}</h1>
        <p className="page-description">
          {t(
            "查看服务端风格及其生成提示词。",
            "Browse server-side styles and their generation prompts.",
          )}
        </p>
      </div>
      <div className="mb-6 flex max-w-xl items-center gap-3">
        <Search className="size-4 text-muted-foreground" />
        <Input
          aria-label={t("搜索风格", "Search styles")}
          placeholder={t("搜索风格或提示词…", "Search styles or prompts…")}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <Badge variant="secondary">{styles?.length || 0}</Badge>
      </div>
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
