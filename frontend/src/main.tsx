import { useState, useEffect, type FormEvent } from "react";
import { createRoot } from "react-dom/client";
import { Toaster } from "sonner";
import {
  Images,
  ListFilter,
  Palette,
  Settings2,
  LogOut,
  Menu,
  ArrowRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Sheet,
  SheetContent,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Locale, useT, useData, api, ErrorBox, Loading } from "@/lib/admin";
import { Tasks } from "./tasks";
import { Settings } from "./settings";
import { Styles } from "./styles";
import "./index.css";
type Session = {
  username: string;
  shopify_embedded: boolean;
  locale: string;
  version: string;
  release_date: string;
  timezones: string[];
};
function Login() {
  const t = useT();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const data = new FormData(e.currentTarget);
      const next =
        new URLSearchParams(location.search).get("next") || "/admin/tasks";
      data.set("next", next);
      const response = await fetch("/admin/login", {
        method: "POST",
        body: data,
      });
      if (response.status === 401)
        throw new Error(
          t("用户名或密码不正确", "Incorrect username or password"),
        );
      if (!response.ok)
        throw new Error(t("登录失败，请重试", "Sign in failed. Try again."));
      location.assign(response.redirected ? response.url : "/admin/tasks");
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }
  return (
    <div className="grid min-h-svh place-items-center px-5">
      <div className="w-full max-w-sm">
        <div className="mb-8 flex items-center gap-3 text-xl font-semibold">
          <div className="rounded-xl bg-foreground p-3 text-background">
            <Images />
          </div>
          Paintingify
        </div>
        <h1 className="page-heading">
          {t("登录管理后台", "Sign in to your workspace")}
        </h1>
        <p className="page-description">
          {t(
            "管理图片任务、风格与生成设置。",
            "Manage image tasks, styles and generation settings.",
          )}
        </p>
        <form className="mt-7 space-y-5" onSubmit={submit}>
          <div className="field">
            <Label htmlFor="username">{t("用户名", "Username")}</Label>
            <Input
              id="username"
              name="username"
              autoComplete="username"
              required
              autoFocus
            />
          </div>
          <div className="field">
            <Label htmlFor="password">{t("密码", "Password")}</Label>
            <Input
              id="password"
              name="password"
              type="password"
              autoComplete="current-password"
              required
            />
          </div>
          <div className="flex items-center gap-2">
            <Checkbox id="remember" name="remember" value="on" />
            <Label htmlFor="remember">{t("记住登录状态", "Remember me")}</Label>
          </div>
          {error && <ErrorBox message={error} />}
          <Button className="w-full" disabled={busy}>
            {busy ? t("正在登录…", "Signing in…") : t("登录", "Sign in")}
            <ArrowRight className="size-4" />
          </Button>
        </form>
        <p className="mt-8 text-xs text-muted-foreground">
          {t("图片生成管理工作台", "Image generation workspace")}
        </p>
      </div>
    </div>
  );
}
function Workspace() {
  const session = useData<Session>("/admin/api/session");
  const [path, setPath] = useState(location.pathname);
  const [menu, setMenu] = useState(false);
  useEffect(() => {
    const refresh = () => {
      void api<Session>("/admin/api/session")
        .then(session.setData)
        .catch(() => {});
    };
    addEventListener("admin:settings-saved", refresh);
    return () => removeEventListener("admin:settings-saved", refresh);
  }, []);
  useEffect(() => {
    const pop = () => setPath(location.pathname);
    addEventListener("popstate", pop);
    return () => removeEventListener("popstate", pop);
  }, []);
  if (session.error)
    return (
      <main className="p-8">
        <ErrorBox message={session.error} retry={session.reload} />
      </main>
    );
  if (!session.data)
    return (
      <main className="p-8">
        <Loading />
      </main>
    );
  const navigate = (next: string) => {
    history.pushState(null, "", next);
    setPath(next);
    setMenu(false);
  };
  return (
    <Locale.Provider value={session.data.locale}>
      <Shell
        path={path}
        navigate={navigate}
        session={session.data}
        menu={menu}
        setMenu={setMenu}
      />
    </Locale.Provider>
  );
}
function Shell({
  path,
  navigate,
  session,
  menu,
  setMenu,
}: {
  path: string;
  navigate: (p: string) => void;
  session: Session;
  menu: boolean;
  setMenu: (v: boolean) => void;
}) {
  const t = useT();
  const embedded = session.shopify_embedded && window.self !== window.top;
  const nav = [
    {
      path: "/admin/tasks",
      name: t("任务记录", "Image tasks"),
      icon: ListFilter,
    },
    {
      path: "/admin/styles",
      name: t("风格目录", "Style library"),
      icon: Palette,
    },
    { path: "/admin/config", name: t("系统设置", "Settings"), icon: Settings2 },
  ];
  const active = nav.find((item) => item.path === path) || nav[0];
  const navigation = (
    <>
      <div className="flex items-center gap-2.5 px-3 py-7 text-lg font-semibold">
        <Images className="size-6 text-foreground" />
        Paintingify
      </div>
      <div className="mb-4 px-3 text-xs text-muted-foreground">
        {t("图片生成工作台", "Image generation workspace")}
      </div>
      <nav className="space-y-1">
        {nav.map((item) => (
          <a
            key={item.path}
            href={item.path}
            aria-current={active.path === item.path ? "page" : undefined}
            onClick={(e) => {
              if (e.metaKey || e.ctrlKey || e.shiftKey) return;
              e.preventDefault();
              navigate(item.path);
            }}
            className={
              "flex items-center gap-3 rounded-md px-3 py-2.5 text-sm transition-colors " +
              (active.path === item.path
                ? "bg-sidebar-accent text-sidebar-accent-foreground"
                : "text-sidebar-foreground hover:bg-sidebar-accent")
            }
          >
            <item.icon className="size-4" />
            {item.name}
          </a>
        ))}
      </nav>
      <div className="mt-auto border-t px-3 pt-5 pb-4">
        <p className="truncate text-sm font-medium">{session.username}</p>
        <p className="mt-1 text-xs text-muted-foreground">
          v{session.version} · {session.release_date}
        </p>
        <form action="/admin/logout" method="post">
          <Button
            variant="ghost"
            size="sm"
            className="mt-3 -ml-3 text-muted-foreground"
          >
            <LogOut className="size-4" />
            {t("退出登录", "Sign out")}
          </Button>
        </form>
      </div>
    </>
  );
  return (
    <div
      className={
        embedded
          ? "min-h-svh bg-background"
          : "min-h-svh bg-sidebar lg:flow-root"
      }
    >
      {!embedded && (
        <aside className="fixed inset-y-0 left-0 hidden w-56 flex-col bg-sidebar px-4 lg:flex">
          {navigation}
        </aside>
      )}
      <div
        className={
          embedded
            ? "min-h-svh bg-background"
            : "min-h-svh bg-background lg:ml-56 lg:mr-2 lg:my-2 lg:min-h-[calc(100svh-1rem)] lg:rounded-xl lg:border lg:shadow-sm"
        }
      >
        {!embedded && (
        <header className="flex h-12 items-center justify-between gap-4 border-b px-4 lg:px-6">
          <div className="flex items-center gap-3">
            {!embedded && (
              <Sheet open={menu} onOpenChange={setMenu}>
                <SheetTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="lg:hidden"
                    aria-label={t("打开导航", "Open navigation")}
                  >
                    <Menu />
                  </Button>
                </SheetTrigger>
                <SheetContent side="left" className="w-64 px-4">
                  <SheetTitle className="sr-only">
                    {t("导航", "Navigation")}
                  </SheetTitle>
                  {navigation}
                </SheetContent>
              </Sheet>
            )}
            <span className="text-sm">{active.name}</span>
          </div>
          <span className="text-xs text-muted-foreground">
            {t("管理后台", "Admin workspace")}
          </span>
        </header>
        )}
        <main className="mx-auto max-w-[1800px] p-4 lg:p-6">
          {active.path === "/admin/config" ? (
            <Settings />
          ) : active.path === "/admin/styles" ? (
            <Styles />
          ) : (
            <Tasks timezones={session.timezones} />
          )}
        </main>
      </div>
    </div>
  );
}
createRoot(document.getElementById("root")!).render(
  <TooltipProvider>
    <Locale.Provider value={document.documentElement.lang}>
      <Toaster richColors position="top-right" />
      {location.pathname === "/admin/login" ? <Login /> : <Workspace />}
    </Locale.Provider>
  </TooltipProvider>,
);
