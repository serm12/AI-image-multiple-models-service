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
  PanelLeftClose,
  PanelLeftOpen,
  ArrowRight,
  ChevronRight,
  ShoppingBag,
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
import { Orders } from "./orders";
import "./index.css";
import { BreadcrumbContext, WorkspaceActions } from "./workspace-header";
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
    if (next === "/admin/orders") window.dispatchEvent(new Event("admin:orders-back"));
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
  const [collapsed, setCollapsed] = useState(false);
  const [breadcrumb, setBreadcrumb] = useState("");
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
    { path: "/admin/orders", name: t("订单管理", "Orders"), icon: ShoppingBag },
    { path: "/admin/config", name: t("系统设置", "Settings"), icon: Settings2 },
  ];
  const active = nav.find((item) => item.path === path) || nav[0];
  const navigation = (compact = false) => (
    <>
      <div className="flex items-center gap-2.5 px-3 py-7 text-lg font-semibold" title={compact ? "Paintingify" : undefined}>
        <Images className="size-6 shrink-0 text-foreground" />
        {!compact && "Paintingify"}
      </div>
      <div className={compact ? "sr-only" : "mb-4 px-3 text-xs text-muted-foreground"}>
        {t("图片生成工作台", "Image generation workspace")}
      </div>
      <nav className="space-y-1">
        {nav.filter(item=>item.path!=="/admin/config").map((item) => (
          <a
            key={item.path}
            href={item.path}
            title={compact ? item.name : undefined}
            aria-label={compact ? item.name : undefined}
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
            <item.icon className="size-4 shrink-0" />
            {!compact && item.name}
          </a>
        ))}
      </nav>
      <div className="mt-auto space-y-4 pt-6 pb-4">
        <a href="/admin/config" title={compact ? t('系统设置','Settings') : undefined} aria-label={t('系统设置','Settings')} aria-current={path==='/admin/config'?'page':undefined} onClick={e=>{if(e.metaKey||e.ctrlKey||e.shiftKey)return;e.preventDefault();navigate('/admin/config');}} className={'flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm hover:bg-neutral-100 '+(path==='/admin/config'?'bg-neutral-100 font-medium':'')}><Settings2 className="size-4 shrink-0"/>{!compact&&t('系统设置','Settings')}</a>
        <div className="border-t px-3 pt-4">
        <p className={compact ? "sr-only" : "mt-1 text-xs text-muted-foreground"}>
          v{session.version} · {session.release_date}
        </p>
        <form action="/admin/logout" method="post">
          <Button
            variant="ghost"
            size="sm"
            className="mt-3 -ml-3 text-muted-foreground"
            aria-label={t("退出登录", "Sign out")}
            title={compact ? t("退出登录", "Sign out") : undefined}
          >
            <LogOut className="size-4" />
            {!compact && t("退出登录", "Sign out")}
          </Button>
        </form>
        </div>
      </div>
    </>
  );
  return (
    <BreadcrumbContext.Provider value={setBreadcrumb}>
    <div
      className={
        embedded
          ? "min-h-svh bg-background"
          : "min-h-svh bg-sidebar lg:flow-root"
      }
    >
      {!embedded && (
        <aside id="desktop-navigation" className={"fixed inset-y-0 left-0 hidden flex-col bg-sidebar lg:flex " + (collapsed ? "w-16 px-2" : "w-56 px-4")}>
          {navigation(collapsed)}
        </aside>
      )}
      <div
        className={
          embedded
            ? "min-h-svh bg-background"
            : "min-h-svh bg-background lg:border-l " + (collapsed ? "lg:ml-16" : "lg:ml-56")
        }
      >
        <header className="sticky top-0 z-30 flex min-h-16 flex-wrap items-center justify-between gap-x-3 gap-y-2 border-b bg-white px-3 py-3 sm:px-6">
          <div className="flex min-w-0 items-center gap-2 sm:gap-3">
            {!embedded && (
              <Button
                variant="ghost"
                size="icon"
                className="hidden lg:inline-flex"
                onClick={() => setCollapsed((value) => !value)}
                aria-controls="desktop-navigation"
                aria-expanded={!collapsed}
                aria-label={collapsed ? t("展开导航", "Expand navigation") : t("收起导航", "Collapse navigation")}
                title={collapsed ? t("展开导航", "Expand navigation") : t("收起导航", "Collapse navigation")}
              >
                {collapsed ? <PanelLeftOpen /> : <PanelLeftClose />}
              </Button>
            )}
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
                  {navigation()}
                </SheetContent>
              </Sheet>
            )}
            <div id="order-detail-back" />
            <nav aria-label={t('面包屑导航','Breadcrumb')} className="min-w-0"><ol className="flex min-w-0 items-center gap-1.5 text-xs sm:gap-2 sm:text-sm"><li className="hidden sm:block"><a href="/admin/tasks" className="text-neutral-500 hover:text-neutral-900" onClick={e=>{if(e.ctrlKey||e.metaKey)return;e.preventDefault();navigate('/admin/tasks');}}>{t('工作台','Home')}</a></li><li aria-hidden="true" className="hidden sm:block"><ChevronRight className="size-3.5 text-neutral-400"/></li><li className="shrink-0">{breadcrumb ? <button className="text-neutral-500 hover:text-neutral-900" onClick={()=>{if(path==='/admin/orders')window.dispatchEvent(new Event('admin:orders-back'));else if(path==='/admin/config')location.hash='printful';else navigate(active.path);}}>{active.name}</button> : <span aria-current="page" className="font-medium">{active.name}</span>}</li>{breadcrumb&&<><li aria-hidden="true"><ChevronRight className="size-3.5 text-neutral-400"/></li><li aria-current="page" className="max-w-32 truncate font-medium">{breadcrumb}</li></>}</ol></nav>
          </div>
          <WorkspaceActions username={session.username} navigate={navigate}/>
        </header>
        <main className="mx-auto max-w-[1800px] p-4 lg:p-6">
          {active.path === "/admin/orders" ? <Orders /> : active.path === "/admin/config" ? (
            <Settings />
          ) : active.path === "/admin/styles" ? (
            <Styles />
          ) : (
            <Tasks timezones={session.timezones} />
          )}
        </main>
      </div>
    </div>
    </BreadcrumbContext.Provider>
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
