import os
import tempfile
from html import escape
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote, urlencode, urlparse, urlsplit
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from PIL import Image, ImageOps

from app.core.config import ArtStyleEnum, STYLE_PROMPTS
from app.core.helpers import get_style_description
from app.core.version import APP_RELEASE_DATE, APP_VERSION
from app.services.security import (
    ADMIN_SESSION_COOKIE,
    ADMIN_SESSION_MAX_AGE,
    create_admin_session,
    get_admin_session_user,
    require_admin_login,
    validate_admin_credentials,
)
from app.services.task_files import resolve_task_file_path
from app.services.task_query_service import list_task_summaries
from app.utils.time_utils import CHINA_TIMEZONE, CHINA_TIMEZONE_NAME


router = APIRouter()
US_EASTERN_TIMEZONE = ZoneInfo("America/New_York")
ADMIN_CONFIG_PREFIXES = (
    "ADMIN_", "AIAPIROUTE_", "WATERMARK_", "IMAGE_", "MAX_", "CORS_",
    "FAL_", "BFL_", "GEMINI_", "GOOGLE_", "FIREWORKS_", "OPENROUTER_",
    "REPLICATE_", "R2_",
)
ADMIN_CONFIG_NAMES = {
    "PORT", "DEBUG", "ALGORITHM_FACTOR", "AI_IMAGE_MEMORY_LIMIT",
    "AI_IMAGE_MEMORY_RESERVATION",
}
SENSITIVE_CONFIG_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def _admin_next(value: str | None) -> str:
    """Allow redirects only to this application's admin task pages."""
    candidate = str(value or "").strip()
    parsed = urlsplit(candidate)
    if not parsed.scheme and not parsed.netloc and parsed.path.startswith("/admin/tasks"):
        return candidate
    return "/admin/tasks"


def _login_page(next_url: str, error: bool = False) -> HTMLResponse:
    message = (
        '<p class="error" role="alert">用户名或密码不正确，请重试。</p>'
        if error else ""
    )
    return HTMLResponse(f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow"><title>登录 · AI 图片管理</title>
<style>*{{box-sizing:border-box}}body{{min-height:100vh;margin:0;display:grid;place-items:center;padding:24px;background:linear-gradient(135deg,#eef4ff,#f8fafc);color:#172033;font:15px/1.5 system-ui,-apple-system,sans-serif}}main{{width:min(100%,420px);padding:36px;border:1px solid #dbe4f0;border-radius:18px;background:#fff;box-shadow:0 18px 48px #2737521f}}h1{{margin:0 0 8px;font-size:25px}}.subtitle{{margin:0 0 27px;color:#64748b}}label{{display:block;margin:16px 0 6px;font-weight:650}}input[type=text],input[type=password]{{width:100%;height:44px;padding:0 12px;border:1px solid #cbd5e1;border-radius:9px;font:inherit}}input:focus{{outline:3px solid #b9d6ff;border-color:#3979c7}}.remember{{display:flex;align-items:center;gap:8px;margin:18px 0 22px;color:#42546d}}button{{width:100%;height:45px;border:0;border-radius:9px;background:#1769d2;color:#fff;font:650 15px inherit;cursor:pointer}}button:hover{{background:#1259b3}}.error{{margin:0 0 16px;padding:9px 11px;border-radius:8px;background:#fff1f0;color:#b42318}}.notice{{margin:22px 0 0;color:#718096;font-size:12px;text-align:center}}</style>
</head><body><main><h1>AI 图片管理</h1><p class="subtitle">请登录以查看生成记录</p>{message}
<form method="post" action="/admin/login"><input type="hidden" name="next" value="{_text(next_url)}">
<label for="username">用户名</label><input id="username" name="username" type="text" autocomplete="username" required autofocus>
<label for="password">密码</label><input id="password" name="password" type="password" autocomplete="current-password" required>
<label class="remember"><input name="remember" type="checkbox" value="true" checked> 记住我（30 天）</label>
<button type="submit">登录</button></form><p class="notice">请勿在公共设备上勾选“记住我”。</p></main></body></html>''', status_code=401 if error else 200)


@router.get("/admin/login", response_class=HTMLResponse)
def admin_login_form(request: Request, next: str = "/admin/tasks"):
    target = _admin_next(next)
    if get_admin_session_user(request.cookies.get(ADMIN_SESSION_COOKIE)):
        return RedirectResponse(target, status_code=303)
    return _login_page(target)


@router.post("/admin/login", response_class=HTMLResponse)
def admin_login(
    username: str = Form(...),
    password: str = Form(...),
    remember: str | None = Form(None),
    next: str = Form("/admin/tasks"),
):
    target = _admin_next(next)
    if not validate_admin_credentials(username, password):
        return _login_page(target, error=True)
    response = RedirectResponse(target, status_code=303)
    max_age = ADMIN_SESSION_MAX_AGE if remember else None
    response.set_cookie(
        ADMIN_SESSION_COOKIE,
        create_admin_session(username, max_age or ADMIN_SESSION_MAX_AGE),
        max_age=max_age,
        httponly=True,
        secure=True,
        samesite="strict",
        path="/admin",
    )
    return response


@router.post("/admin/logout")
def admin_logout():
    response = RedirectResponse("/admin/login", status_code=303)
    response.delete_cookie(ADMIN_SESSION_COOKIE, path="/admin")
    return response


def _text(value) -> str:
    return escape(str(value or ""))


def _config_value(name: str, value: str) -> str:
    if any(marker in name.upper() for marker in SENSITIVE_CONFIG_MARKERS):
        return "已设置（已隐藏）" if value else "未设置"
    return value or "未设置"


@router.get("/admin/styles", response_class=HTMLResponse)
def admin_styles(_username: str = Depends(require_admin_login)):
    """Show the complete server-side style catalog and its injected prompts."""
    seen_values = set()
    style_entries = []
    for style in ArtStyleEnum:
        if style.value in seen_values:
            continue
        seen_values.add(style.value)
        style_entries.append((
            style.value,
            get_style_description(style.value),
            STYLE_PROMPTS.get(style, ""),
        ))

    rows = "".join(
        f"<tr><th><code>{_text(value)}</code></th><td>{_text(description)}</td>"
        f"<td><pre>{_text(prompt or '无额外风格提示词')}</pre></td></tr>"
        for value, description, prompt in style_entries
    )
    return HTMLResponse(f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow">
<title>风格预设 · AI 图片管理</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#f4f7fb;color:#172033;font:14px/1.5 system-ui,-apple-system,sans-serif}}main{{max-width:1500px;margin:auto;padding:28px}}header{{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:18px}}h1{{margin:0;font-size:25px}}p{{color:#64748b}}a{{color:#1769d2;text-decoration:none}}.actions{{display:flex;gap:8px;flex-wrap:wrap}}.button{{padding:7px 11px;border:1px solid #d8e1ec;border-radius:8px;background:#fff}}.panel{{overflow:auto;border:1px solid #e3e8f0;border-radius:14px;background:#fff;box-shadow:0 8px 30px #18243b0d}}table{{width:100%;border-collapse:collapse;min-width:1000px}}th,td{{padding:12px 14px;border-bottom:1px solid #edf0f5;text-align:left;vertical-align:top}}th{{width:220px;background:#f8fafc;color:#40536d}}td:nth-child(2){{width:260px;color:#52647d}}code{{font:12px ui-monospace,SFMono-Regular,monospace}}pre{{margin:0;white-space:pre-wrap;word-break:break-word;font:12px/1.6 ui-monospace,SFMono-Regular,monospace;color:#304761}}.notice{{padding:10px 12px;border-radius:8px;background:#eef6ff;color:#245b9c}}</style></head><body><main><header><div><h1>风格预设</h1><p>v{_text(APP_VERSION)} · {_text(APP_RELEASE_DATE)} · 所有预设及其后端实际追加的提示词。</p></div><nav class="actions"><a class="button" href="/admin/tasks">任务记录</a><a class="button" href="/admin/config">当前配置</a></nav></header><p class="notice">“无额外风格提示词”表示该预设不会在产品提示词前自动追加内容。</p><div class="panel"><table><thead><tr><th>风格标识</th><th>说明</th><th>实际风格提示词</th></tr></thead><tbody>{rows}</tbody></table></div></main></body></html>''')


@router.get("/admin/config", response_class=HTMLResponse)
def admin_config(_username: str = Depends(require_admin_login)):
    entries = [
        (name, _config_value(name, value))
        for name, value in os.environ.items()
        if name in ADMIN_CONFIG_NAMES or name.startswith(ADMIN_CONFIG_PREFIXES)
    ]
    rows = "".join(
        f"<tr><th>{_text(name)}</th><td>{_text(value)}</td></tr>"
        for name, value in sorted(entries)
    ) or '<tr><td colspan="2">没有可显示的配置</td></tr>'
    return HTMLResponse(f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow">
<title>当前配置 · AI 图片管理</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#f4f7fb;color:#172033;font:14px/1.5 system-ui,-apple-system,sans-serif}}main{{max-width:1050px;margin:auto;padding:28px}}header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px}}h1{{margin:0;font-size:25px}}p{{color:#64748b}}a{{color:#1769d2;text-decoration:none}}.actions{{display:flex;gap:8px}}.back{{padding:7px 11px;border:1px solid #d8e1ec;border-radius:8px;background:#fff}}.panel{{overflow:auto;border:1px solid #e3e8f0;border-radius:14px;background:#fff;box-shadow:0 8px 30px #18243b0d}}table{{width:100%;border-collapse:collapse}}th,td{{padding:12px 14px;border-bottom:1px solid #edf0f5;text-align:left}}th{{width:42%;background:#f8fafc;color:#40536d;font-family:ui-monospace,SFMono-Regular,monospace}}td{{word-break:break-all}}.notice{{padding:10px 12px;border-radius:8px;background:#fff8e8;color:#8a5a00}}</style></head><body><main><header><div><h1>当前环境配置</h1><p>显示容器当前已生效的配置；密钥、Token、密码和 Secret 已隐藏。</p></div><nav class="actions"><a class="back" href="/admin/styles">风格预设</a><a class="back" href="/admin/tasks">返回任务记录</a></nav></header><p class="notice">修改服务器 .env 后需要重启容器才会反映在此页面。</p><div class="panel"><table><tbody>{rows}</tbody></table></div></main></body></html>''')


def _display_time(value, source_timezone="UTC", target_timezone=CHINA_TIMEZONE) -> str:
    raw = str(value or "")
    try:
        try:
            parsed = datetime.strptime(raw, "%Y%m%d_%H%M%S")
        except ValueError:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        source_zone = (
            CHINA_TIMEZONE
            if source_timezone == CHINA_TIMEZONE_NAME
            else timezone.utc
        )
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=source_zone)
        return (
            parsed.astimezone(target_timezone)
            .strftime("%Y-%m-%d %H:%M:%S")
        )
    except ValueError:
        return raw


def _compact_display_time(value, source_timezone="UTC", target_timezone=CHINA_TIMEZONE) -> str:
    displayed = _display_time(value, source_timezone, target_timezone)
    return displayed[5:] if len(displayed) == 19 and displayed[4] == "-" else displayed


def _task_row(task: dict) -> str:
    task_id = str(task["task_id"])
    gallery_name = _text(f"task-{task_id}")
    images = ""
    for reference in task.get("source_reference_files", []):
        reference_task_id = str(reference.get("task_id") or "")
        filename = str(reference.get("filename") or "")
        url = str(reference.get("url") or "")
        if not reference_task_id or not filename or not url:
            continue
        thumbnail_url = (
            f"/admin/tasks/{quote(reference_task_id, safe='')}/thumbnail/"
            f"{quote(filename, safe='')}"
        )
        images += (
            f'<a class="image-item image-item--reference glightbox" href="{_text(url)}" '
            f'data-gallery="{gallery_name}" data-title="编辑参考图" '
            f'title="来源任务：{_text(reference_task_id)}">'
            '<span class="image-item__badge">参考</span>'
            f'<img src="{_text(thumbnail_url)}" loading="lazy" decoding="async" '
            'alt="编辑参考图"></a>'
        )
    for url in task.get("output_files", []):
        filename = unquote(os.path.basename(urlparse(str(url)).path))
        thumbnail_url = (
            f"/admin/tasks/{quote(task_id, safe='')}/thumbnail/"
            f"{quote(filename, safe='')}"
        )
        images += (
            f'<a class="image-item glightbox" href="{_text(url)}" '
            f'data-gallery="{gallery_name}">'
            f'<img src="{_text(thumbnail_url)}" loading="lazy" decoding="async" '
            'alt="Generated image"></a>'
        )
    image_strip = (
        '<div class="image-strip" data-image-strip>'
        '<button class="image-scroll image-scroll-prev" type="button" '
        'aria-label="向左滚动图片" title="上一张">‹</button>'
        '<div class="images" tabindex="0" aria-label="生成图片列表">'
        f'{images}</div>'
        '<button class="image-scroll image-scroll-next" type="button" '
        'aria-label="向右滚动图片" title="下一张">›</button>'
        '</div>'
        if images
        else '<span class="muted">—</span>'
    )
    request_url = _text(task.get("request_url"))
    if request_url:
        request_path = urlparse(str(task.get("request_url"))).path or "/"
        request_link = (
            f'<a href="{request_url}" title="{request_url}" target="_blank" '
            f'rel="noopener">{_text(request_path)}</a>'
        )
    else:
        request_link = '<span class="muted">旧任务</span>'
    raw_source_page_url = str(task.get("source_page_url") or "")
    source_page_url = _text(raw_source_page_url)
    source_title = _text(
        task.get("source_product_title") or task.get("source_page_title")
    )
    if source_page_url:
        parsed_source = urlparse(raw_source_page_url)
        source_path = f"{parsed_source.netloc}{parsed_source.path or '/'}"
        source_handle = _text(task.get("source_product_handle"))
        source_detail = source_path
        if source_handle and source_handle not in source_path:
            source_detail = f"{source_detail} · {source_handle}"
        source_title_view = (
            f'<a class="source-title" href="{source_page_url}" title="{source_title}" '
            f'target="_blank" rel="noopener noreferrer">{source_title}</a>'
            if source_title
            else '<span class="muted">—</span>'
        )
        source_link = (
            f'<a class="source-link" href="{source_page_url}" title="{source_page_url}" '
            f'target="_blank" rel="noopener noreferrer">{_text(source_detail)}</a>'
        )
    else:
        source_title_view = source_title or '<span class="muted">—</span>'
        source_link = '<span class="muted">—</span>'
    traffic_source = _text(task.get("traffic_source")) or '<span class="muted">旧任务 / 未采集</span>'
    traffic_referrer = _text(task.get("traffic_referrer"))
    utm_parts = [
        ("utm_source", task.get("utm_source")),
        ("utm_medium", task.get("utm_medium")),
        ("utm_campaign", task.get("utm_campaign")),
    ]
    utm_summary = " · ".join(
        f"{key.replace('utm_', '')}={_text(value)}"
        for key, value in utm_parts
        if value
    )
    # The landing path is useful for data analysis but is not a traffic source.
    # Do not render it under a direct visit, where it would only show a noisy "/".
    traffic_detail = _text(utm_summary or traffic_referrer)
    traffic_attribution_line = (
        f'<div class="source-cell__detail source-cell__attribution" title="{traffic_detail}">{traffic_detail}</div>'
        if traffic_detail
        else ""
    )
    client_ip = _text(task.get("client_ip"))
    ip_task_sequence = task.get("ip_task_sequence")
    ip_task_total = task.get("ip_task_total")
    if client_ip and ip_task_sequence:
        client_ip = (
            f'<span class="client-ip__address">{client_ip}</span> '
            f'<span class="ip-task-sequence" '
            f'title="该 IP 发起的第 {_text(ip_task_sequence)} 个任务">'
            f'#{_text(ip_task_sequence)}</span> '
            f'<span class="ip-task-total" '
            f'title="该 IP 累计发起 {_text(ip_task_total)} 个任务">'
            f'共 {_text(ip_task_total)} 次</span>'
        )
    elif client_ip:
        client_ip = f'<span class="client-ip__address">{client_ip}</span>'
    elif not client_ip:
        client_ip = '<span class="muted">—</span>'
    country = _text(task.get("client_country")) or '<span class="muted">—</span>'
    task_id = _text(task_id)
    prompt = _text(task.get("prompt"))
    edit_instructions = _text(task.get("edit_instructions"))
    prompt_content = ""
    if edit_instructions:
        prompt_content += (
            '<section class="prompt-card__section prompt-card__section--edit">'
            '<strong>客户修改要求</strong>'
            f'<div>{edit_instructions}</div></section>'
        )
    if prompt:
        prompt_content += (
            '<section class="prompt-card__section">'
            f'<strong>{"完整提示词" if edit_instructions else "提示词"}</strong>'
            f'<div>{prompt}</div></section>'
        )
    prompt_view = (
        '<details class="prompt-details"><summary>查看提示词</summary>'
        f'<div class="prompt-card">{prompt_content}</div></details>'
        if prompt_content
        else '<span class="muted">—</span>'
    )
    duration_value = task.get("generation_duration_seconds")
    try:
        duration = f"{float(duration_value):.1f} 秒"
    except (TypeError, ValueError):
        duration = '<span class="muted">—</span>'
    customer_email = _text(task.get("customer_email"))
    customer_id = _text(task.get("customer_id"))
    visitor_id = _text(task.get("storefront_visitor_id"))
    if customer_email:
        customer_title = f"Customer ID: {customer_id}" if customer_id else "已登录客户"
        identity = (
            '<span class="identity identity--customer">已登录</span>'
            f'<span class="customer-email" title="{customer_title}">{customer_email}</span>'
        )
    elif customer_id:
        identity = (
            '<span class="identity identity--customer">已登录</span>'
            f'<code title="尚未采集邮箱 · Customer ID: {customer_id}">{customer_id}</code>'
        )
    elif visitor_id:
        identity = (
            '<span class="identity">访客</span>'
            f'<code title="Visitor ID: {visitor_id}">{visitor_id[:10]}…</code>'
        )
    else:
        identity = '<span class="muted">—</span>'
    event_times = task.get("storefront_event_times") or {}
    funnel_states = (
        ("added-to-cart", "added_to_cart", "加购", task.get("added_to_cart")),
        ("checkout-intent", "checkout_intent", "结账意图", task.get("checkout_intent")),
        ("checkout-started", "checkout_started", "开始结账", task.get("checkout_started")),
        ("checkout-completed", "checkout_completed", "已付款", task.get("checkout_completed")),
    )
    funnel_parts = []
    for stage_class, event_type, label, active in funnel_states:
        state_label = "已触发" if active else "未触发"
        event_time = _text(event_times.get(event_type))
        title = f"{label}：{state_label}"
        if event_time:
            title = f"{title} · {event_time}"
        funnel_parts.append(
            f'<span class="funnel-step funnel-step--{stage_class}'
            f'{" is-active" if active else ""}" title="{title}">'
            f'<span aria-hidden="true">{"✓" if active else "○"}</span> {label}</span>'
        )
    funnel = "".join(funnel_parts)
    beijing_time = _display_time(task["created_at"], task.get("time_zone"))
    eastern_time = _display_time(
        task["created_at"], task.get("time_zone"), US_EASTERN_TIMEZONE
    )
    time_cell = (
        '<div class="task-time">'
        f'<time title="北京时间：{_text(beijing_time)}"><span>北京</span>'
        f'{_text(_compact_display_time(task["created_at"], task.get("time_zone")))}</time>'
        f'<time title="美国东部：{_text(eastern_time)}"><span>美东</span>'
        f'{_text(_compact_display_time(task["created_at"], task.get("time_zone"), US_EASTERN_TIMEZONE))}</time>'
        '</div>'
    )
    return (
        "<tr>"
        f'<td><code title="{task_id}">{task_id[:17]}…</code></td>'
        f'<td><span class="status">{_text(task["status"])}</span></td>'
        f'<td>{time_cell}</td>'
        f"<td>{duration}</td>"
        f'<td class="provider">{_text(task.get("api_provider"))}</td>'
        f'<td class="client-ip">{client_ip}</td>'
        f'<td><span class="country">{country}</span></td>'
        f'<td class="identity-cell">{identity}<div class="funnel">{funnel}</div></td>'
        '<td class="source-cell">'
        f'<div class="source-cell__title">{source_title_view}</div>'
        f'<div class="source-cell__detail"><span>流量</span><strong title="{traffic_detail}">{traffic_source}</strong></div>'
        f'{traffic_attribution_line}'
        f'<div class="source-cell__detail"><span>来源</span>{source_link}</div>'
        f'<div class="source-cell__detail"><span>请求</span>{request_link}</div>'
        '</td>'
        f'<td class="prompt">{prompt_view}</td>'
        f'<td class="images-cell">{image_strip}</td>'
        "</tr>"
    )


def _pagination(
    data: dict,
    client_ip_query: str = "",
    user_query: str = "",
    task_id_query: str = "",
    provider_query: str = "",
) -> str:
    page = int(data["page"])
    page_size = int(data["page_size"])
    total_pages = int(data["total_pages"])

    def page_link(target: int, label: str, class_name: str = "") -> str:
        classes = f' class="{class_name}"' if class_name else ""
        query = urlencode(
            {
                "page": target,
                "page_size": page_size,
                **({"client_ip": client_ip_query} if client_ip_query else {}),
                **({"user": user_query} if user_query else {}),
                **({"task_id": task_id_query} if task_id_query else {}),
                **({"provider": provider_query} if provider_query else {}),
            }
        )
        return (
            f'<a{classes} href="/admin/tasks?{_text(query)}">'
            f'{label}</a>'
        )

    page_numbers = {1, total_pages}
    page_numbers.update(range(max(1, page - 2), min(total_pages, page + 2) + 1))
    number_links = []
    previous_number = 0
    for number in sorted(page_numbers):
        if previous_number and number - previous_number > 1:
            number_links.append('<span class="pagination__ellipsis">…</span>')
        if number == page:
            number_links.append(
                f'<span class="pagination__page is-current" aria-current="page">{number}</span>'
            )
        else:
            number_links.append(page_link(number, str(number), "pagination__page"))
        previous_number = number

    previous = (
        page_link(page - 1, "‹ 上一页", "pagination__direction")
        if data["has_previous"]
        else '<span class="pagination__direction is-disabled">‹ 上一页</span>'
    )
    following = (
        page_link(page + 1, "下一页 ›", "pagination__direction")
        if data["has_next"]
        else '<span class="pagination__direction is-disabled">下一页 ›</span>'
    )
    size_options = "".join(
        f'<option value="{size}"{" selected" if size == page_size else ""}>{size} 条/页</option>'
        for size in (25, 50, 100)
    )
    return (
        '<nav class="pagination" aria-label="任务分页">'
        f'<div class="pagination__links">{previous}{"".join(number_links)}{following}</div>'
        '<label class="pagination__size">每页显示 '
        f'<select id="page-size-select">{size_options}</select></label>'
        f'<span class="pagination__summary">第 {page} / {total_pages} 页</span>'
        '</nav>'
    )


@router.get("/admin/tasks/{task_id}/thumbnail/{filename}")
def admin_task_thumbnail(
    task_id: str,
    filename: str,
    _username: str = Depends(require_admin_login),
):
    """Return a small cached preview instead of transferring the original image."""
    source_path = resolve_task_file_path(task_id, filename)
    if not source_path or not os.path.isfile(source_path):
        return HTMLResponse("图片不存在", status_code=404)

    source = Path(source_path)
    cache_dir = source.parent / ".admin-thumbnails"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"{source.name}.160.jpg"

    if not cache_path.exists() or cache_path.stat().st_mtime < source.stat().st_mtime:
        temp_file = tempfile.NamedTemporaryFile(
            prefix="thumbnail-", suffix=".jpg", dir=cache_dir, delete=False
        )
        temp_path = Path(temp_file.name)
        temp_file.close()
        try:
            with Image.open(source) as opened:
                transposed = ImageOps.exif_transpose(opened)
                image = transposed.convert("RGB")
                try:
                    image.thumbnail((160, 160), Image.Resampling.LANCZOS)
                    image.save(temp_path, "JPEG", quality=72, optimize=True)
                finally:
                    image.close()
                    if transposed is not opened:
                        transposed.close()
            os.replace(temp_path, cache_path)
        finally:
            temp_path.unlink(missing_ok=True)

    return FileResponse(
        cache_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=86400"},
    )


@router.get("/admin/tasks", response_class=HTMLResponse)
def admin_tasks(
    page: int = 1,
    page_size: int = 25,
    client_ip: str = "",
    user: str = "",
    task_id: str = "",
    provider: str = "",
    _username: str = Depends(require_admin_login),
):
    client_ip_query = client_ip.strip()
    user_query = user.strip()
    task_id_query = task_id.strip()
    provider_query = provider.strip()
    data = list_task_summaries(
        page=page,
        page_size=page_size,
        client_ip_query=client_ip_query,
        user_query=user_query,
        task_id_query=task_id_query,
        provider_query=provider_query,
    )
    rows = "".join(_task_row(task) for task in data["tasks"])
    body = rows or '<tr><td colspan="13" class="empty">暂无任务记录</td></tr>'
    pagination = _pagination(
        data,
        client_ip_query,
        user_query,
        task_id_query,
        provider_query,
    )
    return HTMLResponse(
        f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow"><title>AI 图片生成记录</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/glightbox@3.3.1/dist/css/glightbox.min.css">
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f7fb;color:#172033;font:14px/1.5 system-ui,-apple-system,sans-serif}}
main{{max-width:1800px;margin:auto;padding:28px}}header{{display:flex;justify-content:space-between;align-items:end;margin-bottom:18px}}.header-actions{{display:flex;align-items:center;gap:8px;margin-top:9px}}.logout-form{{margin:0}}.logout-button,.config-link{{border:1px solid #d8e1ec;border-radius:8px;background:#fff;color:#53657e;padding:6px 10px;cursor:pointer;text-decoration:none}}
h1{{margin:0;font-size:25px}}.header-meta{{display:flex;align-items:center;gap:10px;margin-top:2px}}.count,.muted{{color:#718096}}.version{{color:#4f6380;font-size:12px;padding:2px 7px;border:1px solid #dce4ee;border-radius:999px;background:#fff}}.current-times{{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}}.current-time{{display:flex;align-items:baseline;gap:7px;padding:5px 9px;border:1px solid #dce4ee;border-radius:8px;background:#fff;color:#24344d;font-variant-numeric:tabular-nums}}.current-time__label{{color:#718096;font-size:12px}}.current-time time{{font-weight:600;white-space:nowrap}}.panel{{background:#fff;border:1px solid #e3e8f0;border-radius:14px;overflow:auto;box-shadow:0 8px 30px #18243b0d}}
.service-status{{display:inline-flex;align-items:center;gap:8px;padding:7px 11px;border:1px solid #dce7df;border-radius:999px;background:#f5fbf6;color:#28733a;font-size:13px;font-weight:600}}.service-dot{{width:8px;height:8px;border-radius:50%;background:#22a447;box-shadow:0 0 0 3px #22a44720}}.service-status.checking{{color:#718096;background:#f8fafc;border-color:#e3e8f0}}.service-status.checking .service-dot{{background:#94a3b8;box-shadow:none}}.service-status.error{{color:#b42318;background:#fff6f5;border-color:#f4d6d2}}.service-status.error .service-dot{{background:#e23b2e;box-shadow:0 0 0 3px #e23b2e20}}
table{{width:100%;border-collapse:collapse;min-width:1280px;table-layout:fixed}}th,td{{padding:8px 7px;border-bottom:1px solid #edf0f5;text-align:left;vertical-align:middle}}th{{position:sticky;top:0;z-index:2;background:#f8fafc;font-size:12px;color:#64748b;white-space:nowrap}}.column-heading{{display:inline-flex;align-items:center;gap:4px;position:relative}}.column-search-toggle{{display:grid;place-items:center;width:20px;height:20px;padding:0;border:0;border-radius:5px;background:transparent;color:#64748b;cursor:pointer}}.column-search-toggle:hover,.column-search-toggle[aria-expanded=true]{{background:#e7effa;color:#1769d2}}.column-search-toggle svg{{width:14px;height:14px}}.column-search{{display:none;position:absolute;top:calc(100% + 7px);left:0;z-index:8;width:210px;padding:8px;border:1px solid #d8e1ec;border-radius:9px;background:#fff;box-shadow:0 10px 24px #17203324}}.column-search.is-open{{display:flex;gap:6px}}.column-search input{{min-width:0;width:100%;height:30px;padding:0 8px;border:1px solid #cbd5e1;border-radius:6px;color:#24344d;font:12px system-ui,-apple-system,sans-serif}}.column-search input:focus{{outline:2px solid #b9d6ff;border-color:#3979c7}}.column-search button{{height:30px;padding:0 9px;border:0;border-radius:6px;background:#1769d2;color:#fff;font:600 12px system-ui,-apple-system,sans-serif;cursor:pointer}}tbody tr{{height:76px}}tbody tr:hover{{background:#fafcff}}
th:nth-child(1){{width:135px}}th:nth-child(2){{width:72px}}th:nth-child(3){{width:138px}}th:nth-child(4){{width:65px}}th:nth-child(5){{width:125px}}th:nth-child(6){{width:130px}}th:nth-child(7){{width:62px}}th:nth-child(8){{width:220px}}th:nth-child(9){{width:340px}}th:nth-child(10){{width:85px}}th:nth-child(11){{width:auto}}
code{{font-size:11px;white-space:nowrap}}.nowrap{{white-space:nowrap}}.task-time{{display:grid;gap:2px;font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}}.task-time time{{display:flex;align-items:center;gap:5px}}.task-time time+time{{color:#718096}}.task-time span{{display:inline-block;width:27px;color:#8a98aa;font-size:10px}}.provider{{overflow-wrap:anywhere}}.status{{display:inline-flex;align-items:center;padding:1px 6px;border-radius:999px;background:#eaf7ed;color:#247436;font-size:11px;font-weight:600;line-height:1.35}}.client-ip{{white-space:normal;line-height:1.3}}.client-ip__address{{overflow-wrap:anywhere;word-break:break-all}}.ip-task-sequence,.ip-task-total{{display:inline-flex;align-items:center;justify-content:center;margin-top:3px;padding:1px 5px;border-radius:999px;font-size:10px;font-weight:700;line-height:1.4;white-space:nowrap}}.ip-task-sequence{{min-width:26px;margin-left:3px;background:#e8f1ff;color:#245b9c}}.ip-task-total{{background:#eef7ed;color:#28733a}}.country{{display:inline-flex;min-width:32px;justify-content:center;padding:2px 6px;border-radius:6px;background:#eef3fa;color:#3f5675;font-weight:600}}.identity-cell{{display:flex;flex-direction:column;align-items:flex-start;gap:3px}}.identity-cell .funnel{{margin-top:2px}}.identity{{padding:1px 5px;border-radius:999px;background:#eef3fa;color:#3f5675;font-size:10px;font-weight:700}}.identity--customer{{background:#eaf7ed;color:#247436}}.funnel{{display:flex;flex-wrap:wrap;gap:3px}}.funnel-step{{padding:2px 5px;border-radius:6px;background:#f1f3f6;color:#98a2b3;font-size:10px;white-space:nowrap}}.funnel-step.is-active{{background:#eaf7ed;color:#247436;font-weight:700}}.source-cell{{overflow:hidden}}.source-cell__title,.source-cell__detail{{display:flex;min-width:0;align-items:center;gap:6px}}.source-cell__title a,.source-cell__detail a{{display:block;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.source-cell__detail{{font-size:11px;color:#718096}}.source-cell__detail>span{{flex:0 0 24px;color:#8a98aa}}.source-title{{font-weight:650;color:#174f9b}}a{{color:#1769d2}}th:nth-child(11),td.images-cell{{position:sticky;right:0;background:#fff;box-shadow:-8px 0 16px #1720330a}}th:nth-child(11){{z-index:4;background:#f8fafc}}tbody tr:hover td.images-cell{{background:#fafcff}}.image-strip{{display:grid;grid-template-columns:23px minmax(52px,1fr) 23px;align-items:center;gap:3px;min-width:0}}.images{{display:flex;gap:4px;min-width:0;overflow-x:auto;overscroll-behavior-inline:contain;scroll-behavior:smooth;scroll-snap-type:x proximity;scrollbar-width:thin;scrollbar-color:#a8b5c7 #edf2f7;padding:1px 1px 4px}}.images::-webkit-scrollbar{{height:5px}}.images::-webkit-scrollbar-track{{background:#edf2f7;border-radius:999px}}.images::-webkit-scrollbar-thumb{{background:#a8b5c7;border-radius:999px}}.image-item{{position:relative;flex:0 0 auto;scroll-snap-align:start}}.image-item__badge{{position:absolute;left:2px;bottom:2px;z-index:1;padding:1px 3px;border-radius:4px;background:#1769d2;color:#fff;font-size:9px;font-weight:700;line-height:1.25}}.image-item--reference img{{border-color:#1769d2;box-shadow:0 0 0 1px #1769d2}}.images img{{display:block;width:54px;height:54px;object-fit:cover;border-radius:6px;border:1px solid #dbe2ea;transition:.15s}}.images img:hover{{transform:scale(1.04)}}.image-scroll{{display:grid;place-items:center;width:23px;height:34px;padding:0;border:1px solid #dbe2ea;border-radius:7px;background:#f8fafc;color:#36516f;font:700 18px/1 system-ui;cursor:pointer}}.image-scroll:hover:not(:disabled){{background:#eaf2fb;border-color:#b9cae0}}.image-scroll:disabled{{opacity:.28;cursor:default}}.pagination{{display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:14px;margin:18px 0 2px;color:#53657e}}.pagination__links{{display:flex;align-items:center;gap:6px}}.pagination__page,.pagination__direction{{display:inline-flex;align-items:center;justify-content:center;min-width:34px;height:34px;padding:0 10px;border:1px solid #d8e1ec;border-radius:8px;background:#fff;color:#245b9c;text-decoration:none}}.pagination__page.is-current{{border-color:#366fac;background:#366fac;color:#fff}}.pagination__direction.is-disabled{{color:#a2adbb;background:#f3f6f9}}.pagination__ellipsis{{padding:0 2px}}.pagination__size{{display:flex;align-items:center;gap:6px}}.pagination__size select{{height:34px;padding:0 28px 0 9px;border:1px solid #d8e1ec;border-radius:8px;background:#fff;color:#33455e}}.pagination__summary{{font-size:12px}}.gcounter{{position:fixed;left:50%;bottom:12px;z-index:100001;transform:translateX(-50%);padding:5px 11px;border-radius:999px;background:#101827d9;color:#fff;font-size:13px;font-variant-numeric:tabular-nums;pointer-events:none}}.empty{{padding:50px;text-align:center;color:#718096}}
.customer-email{{max-width:125px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px;color:#24344d}}.funnel-step{{border:1px solid #e4e7ec;background:#f8fafc}}.funnel-step.is-active{{font-weight:700}}.funnel-step--added-to-cart.is-active{{border-color:#bfd7f6;background:#eaf3ff;color:#245b9c}}.funnel-step--checkout-intent.is-active{{border-color:#f2d394;background:#fff7e6;color:#9a6700}}.funnel-step--checkout-started.is-active{{border-color:#d7c4f5;background:#f5efff;color:#6941c6}}.funnel-step--checkout-completed.is-active{{border-color:#bfe3c7;background:#eaf7ed;color:#247436}}
.prompt-details{{position:relative}}.prompt-details summary{{cursor:pointer;color:#1769d2;white-space:nowrap;list-style:none}}.prompt-details summary::-webkit-details-marker{{display:none}}.prompt-details summary:after{{content:" ›"}}.prompt-details[open] summary:after{{content:" ×"}}.prompt-card{{position:absolute;right:0;top:30px;z-index:10;width:min(460px,70vw);max-height:360px;overflow:auto;padding:15px;border:1px solid #dbe2ea;border-radius:10px;background:#fff;box-shadow:0 14px 40px #1720332b;white-space:pre-wrap;line-height:1.65}}.prompt-card__section{{display:grid;gap:6px}}.prompt-card__section+ .prompt-card__section{{margin-top:13px;padding-top:13px;border-top:1px solid #e3e8f0}}.prompt-card__section strong{{font-size:12px;color:#53657e}}.prompt-card__section--edit{{padding:10px;border:1px solid #f1d59e;border-radius:8px;background:#fff8e8;color:#7a4d00}}.prompt-card__section--edit strong{{color:#9a6700}}
@media(max-width:700px){{main{{padding:16px}}h1{{font-size:21px}}header{{align-items:center}}.current-times{{flex-direction:column;align-items:flex-start}}.panel{{border-radius:10px}}}}
</style></head><body><main><header><div><h1>AI 图片生成记录</h1><div class="header-meta"><span class="count">共 {_text(data['total'])} 条任务</span><span class="version">v{_text(APP_VERSION)} · {_text(APP_RELEASE_DATE)}</span></div><div class="current-times" aria-label="当前时间"><span class="current-time"><span class="current-time__label">北京时间</span><time id="current-beijing-time">--</time></span><span class="current-time"><span class="current-time__label">美国东部</span><time id="current-us-eastern-time">--</time></span></div></div><div><div id="service-status" class="service-status checking"><span class="service-dot"></span><span class="service-text">状态检测中</span></div><div class="header-actions"><a class="config-link" href="/admin/styles">风格预设</a><a class="config-link" href="/admin/config">当前配置</a><form class="logout-form" method="post" action="/admin/logout"><button class="logout-button" type="submit">退出登录</button></form></div></div></header>
<div class="panel"><table><thead><tr><th><span class="column-heading">任务 ID <button class="column-search-toggle" type="button" aria-label="筛选任务 ID" aria-expanded="false" aria-controls="task-id-search"><svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="6"></circle><path d="m16 16 4 4"></path></svg></button><form id="task-id-search" class="column-search" action="/admin/tasks" method="get"><input type="hidden" name="page" value="1"><input type="hidden" name="page_size" value="{page_size}"><input type="hidden" name="client_ip" value="{_text(client_ip_query)}"><input type="hidden" name="user" value="{_text(user_query)}"><input type="hidden" name="provider" value="{_text(provider_query)}"><input name="task_id" type="search" value="{_text(task_id_query)}" placeholder="输入任务 ID 筛选" aria-label="任务 ID"><button type="submit">筛选</button></form></span></th><th>状态</th><th>时间</th><th>总耗时</th><th><span class="column-heading">Provider <button class="column-search-toggle" type="button" aria-label="筛选 Provider" aria-expanded="false" aria-controls="provider-search"><svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="6"></circle><path d="m16 16 4 4"></path></svg></button><form id="provider-search" class="column-search" action="/admin/tasks" method="get"><input type="hidden" name="page" value="1"><input type="hidden" name="page_size" value="{page_size}"><input type="hidden" name="client_ip" value="{_text(client_ip_query)}"><input type="hidden" name="user" value="{_text(user_query)}"><input type="hidden" name="task_id" value="{_text(task_id_query)}"><input name="provider" type="search" value="{_text(provider_query)}" placeholder="输入 Provider 筛选" aria-label="Provider"><button type="submit">筛选</button></form></span></th><th><span class="column-heading">访客 IP <button class="column-search-toggle" type="button" aria-label="筛选访客 IP" aria-expanded="false" aria-controls="ip-search"><svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="6"></circle><path d="m16 16 4 4"></path></svg></button><form id="ip-search" class="column-search" action="/admin/tasks" method="get"><input type="hidden" name="page" value="1"><input type="hidden" name="page_size" value="{page_size}"><input type="hidden" name="user" value="{_text(user_query)}"><input type="hidden" name="task_id" value="{_text(task_id_query)}"><input type="hidden" name="provider" value="{_text(provider_query)}"><input name="client_ip" type="search" value="{_text(client_ip_query)}" placeholder="输入 IP 筛选" aria-label="访客 IP"><button type="submit">筛选</button></form></span></th><th>国家/地区</th><th><span class="column-heading">用户 / 转化状态 <button class="column-search-toggle" type="button" aria-label="筛选用户" aria-expanded="false" aria-controls="user-search"><svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="6"></circle><path d="m16 16 4 4"></path></svg></button><form id="user-search" class="column-search" action="/admin/tasks" method="get"><input type="hidden" name="page" value="1"><input type="hidden" name="page_size" value="{page_size}"><input type="hidden" name="client_ip" value="{_text(client_ip_query)}"><input type="hidden" name="task_id" value="{_text(task_id_query)}"><input type="hidden" name="provider" value="{_text(provider_query)}"><input name="user" type="search" value="{_text(user_query)}" placeholder="邮箱、客户或访客 ID" aria-label="用户"><button type="submit">筛选</button></form></span></th><th>页面信息</th><th>提示词</th><th>图片</th></tr></thead><tbody>{body}</tbody></table></div>
{pagination}
</main><script src="https://cdn.jsdelivr.net/npm/glightbox@3.3.1/dist/js/glightbox.min.js"></script><script>
const statusEl=document.getElementById('service-status');
const statusText=statusEl.querySelector('.service-text');
const clocks=[
  [document.getElementById('current-beijing-time'),'Asia/Shanghai'],
  [document.getElementById('current-us-eastern-time'),'America/New_York']
];
const clockFormatters=new Map(clocks.map(([,zone])=>[zone,new Intl.DateTimeFormat('sv-SE',{{timeZone:zone,year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}})]));
function updateCurrentTimes(){{
  const now=new Date();
  clocks.forEach(([element,zone])=>{{
    element.textContent=clockFormatters.get(zone).format(now);
    element.dateTime=now.toISOString();
  }});
}}
updateCurrentTimes();
setInterval(updateCurrentTimes,1000);
document.getElementById('page-size-select')?.addEventListener('change',event=>{{
  const params=new URLSearchParams(window.location.search);
  params.set('page','1');
  params.set('page_size',event.target.value);
  window.location.href=`/admin/tasks?${{params.toString()}}`;
}});
document.querySelectorAll('.column-search-toggle').forEach(toggle=>{{
  const form=document.getElementById(toggle.getAttribute('aria-controls'));
  toggle.addEventListener('click',()=>{{
    const isOpen=form.classList.toggle('is-open');
    toggle.setAttribute('aria-expanded',String(isOpen));
    if(isOpen)form.querySelector('input[type=search]').focus();
  }});
}});
document.addEventListener('click',event=>{{
  if(event.target.closest('.column-heading'))return;
  document.querySelectorAll('.column-search.is-open').forEach(form=>{{
    form.classList.remove('is-open');
    document.querySelector(`[aria-controls="${{form.id}}"]`)?.setAttribute('aria-expanded','false');
  }});
}});
fetch('/health',{{cache:'no-store'}}).then(response=>{{if(!response.ok)throw new Error();return response.json()}}).then(data=>{{
  statusEl.className='service-status';
  statusText.textContent=data.status==='ok'?'服务正常':'服务异常';
  if(data.status!=='ok')statusEl.classList.add('error');
}}).catch(()=>{{statusEl.className='service-status error';statusText.textContent='服务异常'}});
document.querySelectorAll('[data-image-strip]').forEach(strip=>{{
  const viewport=strip.querySelector('.images');
  const previous=strip.querySelector('.image-scroll-prev');
  const next=strip.querySelector('.image-scroll-next');
  const update=()=>{{
    const maxScroll=Math.max(0,viewport.scrollWidth-viewport.clientWidth);
    previous.disabled=viewport.scrollLeft<=1;
    next.disabled=viewport.scrollLeft>=maxScroll-1;
  }};
  const move=direction=>viewport.scrollBy({{left:direction*Math.max(75,viewport.clientWidth*.8),behavior:'smooth'}});
  previous.addEventListener('click',()=>move(-1));
  next.addEventListener('click',()=>move(1));
  viewport.addEventListener('scroll',update,{{passive:true}});
  viewport.addEventListener('keydown',event=>{{
    if(event.key==='ArrowLeft'){{event.preventDefault();move(-1)}}
    if(event.key==='ArrowRight'){{event.preventDefault();move(1)}}
    if(event.key==='Home'){{event.preventDefault();viewport.scrollTo({{left:0,behavior:'smooth'}})}}
    if(event.key==='End'){{event.preventDefault();viewport.scrollTo({{left:viewport.scrollWidth,behavior:'smooth'}})}}
  }});
  new ResizeObserver(update).observe(viewport);
  update();
}});
const lightbox=GLightbox({{
  selector:'.glightbox',
  touchNavigation:true,
  loop:true,
  zoomable:true,
  draggable:true,
  openEffect:'fade',
  closeEffect:'fade'
}});
function updateLightboxCounter(index){{
  if(!lightbox.modal)return;
  let counter=lightbox.modal.querySelector('.gcounter');
  if(!counter){{
    counter=document.createElement('div');
    counter.className='gcounter';
    counter.setAttribute('aria-live','polite');
    lightbox.modal.appendChild(counter);
  }}
  counter.textContent=`${{index+1}} / ${{lightbox.elements.length}}`;
}}
lightbox.on('open',()=>updateLightboxCounter(lightbox.index));
lightbox.on('slide_changed',({{current}})=>updateLightboxCounter(current.slideIndex));
</script></body></html>"""
    )
