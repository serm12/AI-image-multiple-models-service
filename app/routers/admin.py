from app.services.i18n import render as _i18n, Message as _i18n_message, t as _t
import os
import tempfile
from html import escape
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote, urlencode, urlparse, urlsplit
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Form, Query, Request
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
from app.services.task_query_service import list_task_summaries, set_task_starred
from app.utils.time_utils import CHINA_TIMEZONE, CHINA_TIMEZONE_NAME


from app.routers.watermark_settings import router as watermark_settings_router

router = APIRouter()
router.include_router(watermark_settings_router)
from app.routers.time_settings import router as time_settings_router
from app.services.time_settings import effective_timezone, display_timezones
router.include_router(time_settings_router)
from app.routers.algorithm_settings import router as algorithm_settings_router
router.include_router(algorithm_settings_router)
from app.routers.provider_credentials import router as provider_credentials_router
router.include_router(provider_credentials_router)
US_EASTERN_TIMEZONE = ZoneInfo("America/New_York")
US_PACIFIC_TIMEZONE = ZoneInfo("America/Los_Angeles")
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
CONVERSION_FILTER_OPTIONS = (
    ("added_to_cart", _i18n_message('template_f0d14c5499db')),
    ("checkout_intent", _i18n_message('template_b3992285e8b2')),
    ("checkout_started", _i18n_message('template_f460e795b575')),
    ("checkout_completed", _i18n_message('template_a461133d405c')),
)
TIME_RANGE_OPTIONS = (
    ("", _i18n_message('template_1a1111c82dfe')),
    ("today", _i18n_message('template_e079844b9087')),
    ("yesterday", _i18n_message('template_5e0a77c951cd')),
    ("last_7_days", _i18n_message('template_57763cded20c')),
    ("last_30_days", _i18n_message('template_00634cb4b53a')),
)
TRAFFIC_SOURCE_ICONS = (
    ("google", "Google", "https://cdn.simpleicons.org/google/4285F4"),
    ("bing", "Bing", "https://www.bing.com/sa/simg/favicon-trans-bg-blue-mg.ico"),
    ("duckduckgo", "DuckDuckGo", "https://duckduckgo.com/favicon.ico"),
    ("facebook", "Facebook", "https://cdn.simpleicons.org/facebook/1877F2"),
    ("instagram", "Instagram", "https://cdn.simpleicons.org/instagram/E4405F"),
    ("tiktok", "TikTok", "https://cdn.simpleicons.org/tiktok/000000"),
    ("youtube", "YouTube", "https://cdn.simpleicons.org/youtube/FF0000"),
    ("pinterest", "Pinterest", "https://cdn.simpleicons.org/pinterest/BD081C"),
    ("twitter", "X", "https://cdn.simpleicons.org/x/000000"),
    ("x.com", "X", "https://cdn.simpleicons.org/x/000000"),
    ("linkedin", "LinkedIn", "https://cdn.simpleicons.org/linkedin/0A66C2"),
)


def _admin_next(value: str | None) -> str:
    """Allow redirects only to this application's admin task pages."""
    candidate = str(value or "").strip()
    parsed = urlsplit(candidate)
    if not parsed.scheme and not parsed.netloc and parsed.path.startswith("/admin/tasks"):
        return candidate
    return "/admin/tasks"


def _login_page(next_url: str, error: bool = False) -> HTMLResponse:
    message = (
        _i18n('template_6b1313f1949c')
        if error else ""
    )
    return HTMLResponse(f'{_i18n('template_dbf75f2ff239')}{message}\n<form method="post" action="/admin/login"><input type="hidden" name="next" value="{_text(next_url)}{_i18n('template_dd51954c7f8a')}', status_code=401 if error else 200)


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
        return _i18n('template_4b7366d5c95d') if value else _i18n('template_20310590eff4')
    return value or _i18n('template_20310590eff4')


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
        f"<td><pre>{_text(prompt or _i18n('template_cc29bbde8853'))}</pre></td></tr>"
        for value, description, prompt in style_entries
    )
    return HTMLResponse(f'{_i18n('template_6179db4eb1df')}{_text(APP_VERSION)} · {_text(APP_RELEASE_DATE)}{_i18n('template_b58e913342a8')}{rows}</tbody></table></div></main></body></html>')


@router.get("/admin/config", response_class=HTMLResponse)
def admin_config(_username: str = Depends(require_admin_login)):
    from app.routers.watermark_settings import settings_page
    return settings_page()


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


def _traffic_source_icon(*values: object) -> str:
    """Return a compact, trusted logo for recognized search and social sources."""
    source_text = " ".join(str(value or "").casefold() for value in values)
    for keyword, label, icon_url in TRAFFIC_SOURCE_ICONS:
        if keyword in source_text:
            return (
                f'<img class="traffic-icon" src="{icon_url}" alt="{label}" '
                f'title="{label}" width="14" height="14" loading="lazy">'
            )
    return ""


def _paid_traffic_badge(*values: object) -> str:
    """Label common paid-acquisition UTM and attribution values."""
    source_text = " ".join(str(value or "").casefold() for value in values)
    paid_markers = (
        "paid", "cpc", "ppc", "cpv", "cpm", "display", "advert",
        "adwords", "fbclid", "gclid", "msclkid",
    )
    if any(marker in source_text for marker in paid_markers):
        return _i18n('template_a844489ea6c1')
    return ""


def _task_row(task: dict, return_url: str) -> str:
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
            f'<a class="image-item image-item--reference glightbox" href="{_text(url)}" data-gallery="{gallery_name}{_i18n('template_d05675f67f08')}{_text(reference_task_id)}{_i18n('template_f240fdb5ba46')}{_text(thumbnail_url)}{_i18n('template_c7b86ba0a051')}'
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
        f'{_i18n('template_b3dfc2391711')}{images}{_i18n('template_a01bdbec5e13')}'
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
        request_link = _i18n('template_a79200734f5a')
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
    raw_traffic_source = task.get("traffic_source")
    raw_traffic_referrer = task.get("traffic_referrer")
    traffic_source = _text(raw_traffic_source) or _i18n('template_f53635f77d94')
    traffic_referrer = _text(raw_traffic_referrer)
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
    traffic_icon = _traffic_source_icon(
        raw_traffic_source, raw_traffic_referrer, task.get("utm_source")
    )
    paid_traffic_badge = _paid_traffic_badge(
        raw_traffic_source, raw_traffic_referrer, task.get("utm_medium")
    )
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
            f'<span class="client-ip__address">{client_ip}{_i18n('template_4462a1a7815f')}{_text(ip_task_sequence)}{_i18n('template_5cda91f67114')}{_text(ip_task_sequence)}{_i18n('template_98bf3b84b1f8')}{_text(ip_task_total)}{_i18n('template_3583da37e1d3')}{_text(ip_task_total)}{_i18n('template_9a4e91264c64')}'
        )
    elif client_ip:
        client_ip = f'<span class="client-ip__address">{client_ip}</span>'
    elif not client_ip:
        client_ip = '<span class="muted">—</span>'
    country_code = str(task.get("client_country") or "").strip().upper()
    country = _text(country_code) or '<span class="muted">—</span>'
    if (
        len(country_code) == 2
        and country_code.isascii()
        and country_code.isalpha()
        and country_code not in {"XX", "T1"}
    ):
        country = (
            f'<img class="country-flag" src="https://flagcdn.com/w40/{country_code.lower()}.png" '
            'width="20" height="15" alt="" aria-hidden="true" loading="lazy" '
            'decoding="async" referrerpolicy="no-referrer" onerror="this.hidden=true">'
            f'<span>{country}</span>'
        )
    raw_task_id = task_id
    task_id = _text(task_id)
    star_title = _i18n('template_52fd8704a992') if task.get("starred") else _i18n('template_b785ec7a76ae')
    star_label = _i18n('template_1dc1f9a399ca') if task.get("starred") else _i18n('template_4293266621b8')
    star_class = " is-starred" if task.get("starred") else ""
    star_control = (
        f'<form class="star-form" method="post" action="/admin/tasks/{quote(raw_task_id, safe="")}/star">'
        f'<input type="hidden" name="next" value="{_text(return_url)}">'
        f'<input type="hidden" name="starred" value="{"true" if task.get("starred") else "false"}">'
        f'<button class="star-button{star_class}" type="submit" title="{star_title}" '
        f'aria-label="{star_title}" aria-pressed="{"true" if task.get("starred") else "false"}">'
        '<span aria-hidden="true">★</span></button></form>'
    )
    prompt = _text(task.get("prompt"))
    edit_instructions = _text(task.get("edit_instructions"))
    prompt_content = ""
    if edit_instructions:
        prompt_content += (
            f'{_i18n('template_44b6c70e2d53')}{edit_instructions}</div></section>'
        )
    if prompt:
        prompt_content += (
            '<section class="prompt-card__section">'
            f'<strong>{_i18n('template_34edb121b015') if edit_instructions else _i18n('template_98b17503c159')}</strong>'
            f'<div>{prompt}</div></section>'
        )
    prompt_view = (
        f'{_i18n('template_2a3f984d7954')}{prompt_content}</div></details>'
        if prompt_content
        else '<span class="muted">—</span>'
    )
    duration_value = task.get("generation_duration_seconds")
    try:
        duration = f'{float(duration_value):.1f}{_i18n('template_bc1c87c71f42')}'
    except (TypeError, ValueError):
        duration = '<span class="muted">—</span>'
    customer_email = _text(task.get("customer_email"))
    customer_id = _text(task.get("customer_id"))
    visitor_id = _text(task.get("storefront_visitor_id"))
    linked_customer_id = str((task.get('customer_sync') or {}).get('customer_id') or task.get('customer_id') or '')
    linked_customer_id = linked_customer_id.removeprefix('gid://shopify/Customer/')
    customer_url = (f'https://admin.shopify.com/store/paintingify/customers/{linked_customer_id}'
                    if linked_customer_id.isascii() and linked_customer_id.isdigit() else '')
    if customer_email:
        customer_title = f"Customer ID: {customer_id}" if customer_id else _i18n('template_c82b767c7740')
        identity_label = _i18n('template_bb788be6b37c') if task.get("customer_logged_in") and customer_id else _i18n('template_2fe600893ce2')
        identity = (
            f'<span class="identity identity--customer">{identity_label}</span>'
            + (f'<a class="customer-email" href="{customer_url}{_i18n('template_f2cac1919c64')}{customer_email} 🔗</a>'
               if customer_url else f'<span class="customer-email" title="{customer_title}">{customer_email}</span>')
        )
    elif customer_id:
        identity = (
            _i18n('template_a83acd03042b')
            + (f'<a href="{customer_url}{_i18n('template_f2cac1919c64')}{customer_id} 🔗</a>'
               if customer_url else f'{_i18n('template_cef6d5dab960')}{customer_id}">{customer_id}</code>')
        )
    elif visitor_id:
        identity = (
            f'{_i18n('template_96e626479cd7')}{visitor_id}">{visitor_id[:10]}…</code>'
        )
    else:
        identity = '<span class="muted">—</span>'
    subscription_labels = {
        "accepted": _i18n('template_7a0f86c45e41'),
        "verification_required": _i18n('template_2a089fbc2c45'),
        "failed": _i18n('template_e0764f5a6749'),
        "unconfirmed": _i18n('template_eb74eba1754c'),
        "not_attempted": _i18n('template_1e52f26c934c'),
    }
    subscription_label = subscription_labels.get(task.get("subscription_status"), _i18n('template_53250ef070d6'))
    subscription_detail = _i18n('template_edc8a35f4861') + str(task.get("subscription_reason") or "")
    synced = task.get('customer_sync') or {}
    if synced:
        subscription_label = {'queued': _i18n('template_bfe688a5a764'), 'retry': _i18n('template_67d439b7a8c6'),
            'failed': _i18n('template_1d69f4d2229c'), 'subscribed': _i18n('template_1b08d243675e'),
            'preserved': _i18n('template_289ed03acf95')}.get(synced.get('state'), subscription_label)
        subscription_detail = synced.get('detail') or _i18n('template_31c6f244b45f')
    identity += f'<span class="muted" title="{_text(subscription_detail)}">{_text(subscription_label)}</span>'
    event_times = task.get("storefront_event_times") or {}
    funnel_states = (
        ("added-to-cart", "added_to_cart", _i18n('template_f0d14c5499db'), task.get("added_to_cart")),
        ("checkout-intent", "checkout_intent", _i18n('template_b3992285e8b2'), task.get("checkout_intent")),
        ("checkout-started", "checkout_started", _i18n('template_f460e795b575'), task.get("checkout_started")),
        ("checkout-completed", "checkout_completed", _i18n('template_a461133d405c'), task.get("checkout_completed")),
    )
    funnel_parts = []
    for stage_class, event_type, label, active in funnel_states:
        state_label = _i18n('template_fb1b45b8d31f') if active else _i18n('template_ffd199c8f31c')
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
    if task.get('checkout_completed') and not task.get('shopify_orders'):
        identity += _i18n('template_d153eecf0624')
    for order in task.get('shopify_orders', []):
        status = {'PAID': _i18n('template_a461133d405c'), 'PARTIALLY_REFUNDED': _i18n('template_6f52bfb56ce2'), 'REFUNDED': _i18n('template_4fbf5c598aac'),
                  'PENDING': _i18n('template_bb6cb7d2c01b'), 'AUTHORIZED': _i18n('template_d5498f3548b5'), 'VOIDED': _i18n('template_456ebc9b3dad'),
                  'PARTIALLY_PAID': _i18n('template_4ca78ac48c44')}.get(order.get('financial_status'), _i18n('template_5ea44804ef9c'))
        identity += (f'<a class="customer-email" href="{_text(order['url'])}{_i18n('template_769c5fdec4c1')}{_text(order['name'])} · {_text(status)} ↗</a>')
    time_cell = '<div class="task-time">' + ''.join(
        f'<time title="{_text(zone)}: {_text(_display_time(task["created_at"], task.get("time_zone"), ZoneInfo(zone)))}"><span>{_text(zone.rsplit("/", 1)[-1].replace("_", " "))}</span>{_text(_compact_display_time(task["created_at"], task.get("time_zone"), ZoneInfo(zone)))}</time>'
        for zone in display_timezones()
    ) + '</div>'
    return (
        f'<tr><td><div class="star-cell">{star_control}<code title="{task_id}">{task_id[:17]}…</code></div></td><td><span class="status">{_text(task['status'])}</span></td><td>{time_cell}</td><td>{duration}</td><td class="provider">{_text(task.get('api_provider'))}</td><td class="client-ip">{client_ip}</td><td><span class="country">{country}</span></td><td class="identity-cell">{identity}<div class="funnel">{funnel}</div></td><td class="source-cell"><div class="source-cell__title">{source_title_view}{_i18n('template_5fe7cab5d12f')}{paid_traffic_badge}{traffic_icon}<strong title="{traffic_detail}">{traffic_source}</strong></div>{traffic_attribution_line}{_i18n('template_6fd6c4689a64')}{source_link}{_i18n('template_0b76794fe50f')}{request_link}</div></td><td class="prompt">{prompt_view}</td><td class="images-cell">{image_strip}</td></tr>'
    )


def _pagination(
    data: dict,
    client_ip_query: str = "",
    user_query: str = "",
    task_id_query: str = "",
    provider_query: str = "",
    conversion_statuses: tuple[str, ...] = (),
    time_range: str = "",
    starred_only: bool = False,
) -> str:
    page = int(data["page"])
    page_size = int(data["page_size"])
    total_pages = int(data["total_pages"])

    def page_link(target: int, label: str, class_name: str = "") -> str:
        classes = f' class="{class_name}"' if class_name else ""
        query_items = [("page", target), ("page_size", page_size)]
        if client_ip_query:
            query_items.append(("client_ip", client_ip_query))
        if user_query:
            query_items.append(("user", user_query))
        if task_id_query:
            query_items.append(("task_id", task_id_query))
        if provider_query:
            query_items.append(("provider", provider_query))
        if time_range:
            query_items.append(("time_range", time_range))
        if starred_only:
            query_items.append(("starred", "true"))
        query_items.extend(("conversion_status", status) for status in conversion_statuses)
        query = urlencode(query_items)
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
        page_link(page - 1, _i18n('template_7d4b87fe917c'), "pagination__direction")
        if data["has_previous"]
        else _i18n('template_b40a647c8028')
    )
    following = (
        page_link(page + 1, _i18n('template_dd7541324981'), "pagination__direction")
        if data["has_next"]
        else _i18n('template_68cd079c4e2b')
    )
    size_options = "".join(
        f'<option value="{size}"{(' selected' if size == page_size else '')}>{size}{_i18n('template_c1354a5fc066')}'
        for size in (25, 50, 100)
    )
    jump_items = [
        ("page_size", page_size),
        ("client_ip", client_ip_query),
        ("user", user_query),
        ("task_id", task_id_query),
        ("provider", provider_query),
        ("time_range", time_range),
        *(("conversion_status", status) for status in conversion_statuses),
    ]
    if starred_only:
        jump_items.append(("starred", "true"))
    jump_fields = "".join(
        f'<input type="hidden" name="{_text(name)}" value="{_text(value)}">'
        for name, value in jump_items
        if value != ""
    )
    page_jump = (
        f'<form class="pagination__jump" action="/admin/tasks" method="get">{jump_fields}{_i18n('template_e9e29c7c2b06')}{total_pages}" value="{page}{_i18n('template_bb4292aeb2ff')}'
    )
    return (
        f'{_i18n('template_6497414084d7')}{previous}{''.join(number_links)}{following}{_i18n('template_da22c9bc88f9')}{size_options}</select></label>{page_jump}{_i18n('template_ce462b242464')}{page} / {total_pages}{_i18n('template_f4f0088a6048')}'
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
        return HTMLResponse(_i18n('template_d138c43e600e'), status_code=404)

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


@router.post("/admin/tasks/{task_id}/star")
def admin_task_star(
    task_id: str,
    starred: str = Form("false"),
    next: str = Form("/admin/tasks"),
    _username: str = Depends(require_admin_login),
):
    """Toggle the admin-only star for a task and return to the current list."""
    if set_task_starred(task_id, starred.strip().lower() != "true") is None:
        return HTMLResponse(_i18n('template_9c60c5b1f3d6'), status_code=404)
    return RedirectResponse(_admin_next(next), status_code=303)


@router.get("/admin/tasks", response_class=HTMLResponse)
def admin_tasks(
    page: int = 1,
    page_size: int = 25,
    client_ip: str = "",
    user: str = "",
    task_id: str = "",
    provider: str = "",
    conversion_status: list[str] = Query(default=[]),
    time_range: str = "",
    starred: bool = False,
    _username: str = Depends(require_admin_login),
):
    client_ip_query = client_ip.strip()
    user_query = user.strip()
    task_id_query = task_id.strip()
    provider_query = provider.strip()
    time_range_query = time_range.strip()
    if time_range_query not in {value for value, _label in TIME_RANGE_OPTIONS}:
        time_range_query = ""
    conversion_statuses = tuple(
        value
        for value, _label in CONVERSION_FILTER_OPTIONS
        if value in conversion_status
    )
    data = list_task_summaries(
        page=page,
        page_size=page_size,
        client_ip_query=client_ip_query,
        user_query=user_query,
        task_id_query=task_id_query,
        provider_query=provider_query,
        conversion_statuses=list(conversion_statuses),
        time_range=time_range_query,
        starred_only=starred,
    )
    query_items = [("page", str(page)), ("page_size", str(page_size))]
    query_items.extend((name, value) for name, value in (
        ("client_ip", client_ip_query), ("user", user_query),
        ("task_id", task_id_query), ("provider", provider_query),
        ("time_range", time_range_query),
    ) if value)
    query_items.extend(("conversion_status", status) for status in conversion_statuses)
    if starred:
        query_items.append(("starred", "true"))
    return_url = "/admin/tasks?" + urlencode(query_items)
    starred_filter_url = "/admin/tasks?" + urlencode(
        [("page", "1"), ("page_size", str(page_size))]
        + ([] if starred else [("starred", "true")])
    )
    from app.services.shopify_orders import read_order_cache
    order_links = read_order_cache().get('tasks', {})
    for task in data['tasks']:
        task['shopify_orders'] = order_links.get(task['task_id'], [])
    rows = "".join(_task_row(task, return_url) for task in data["tasks"])
    body = rows or _i18n('template_0d3a2439e735')
    pagination = _pagination(
        data,
        client_ip_query,
        user_query,
        task_id_query,
        provider_query,
        conversion_statuses,
        time_range_query,
        starred,
    )
    conversion_filter_options = "".join(
        '<label style="display:inline-flex;align-items:center;gap:3px;padding:3px 6px;'
        'border:1px solid #d8e1ec;border-radius:6px;color:#40536d;font-size:12px;cursor:pointer">'
        f'<input type="checkbox" name="conversion_status" value="{value}"'
        f'{" checked" if value in conversion_statuses else ""}> {_text(label)}'
        '</label>'
        for value, label in CONVERSION_FILTER_OPTIONS
    )
    conversion_filter_form = (
        f'<form id="conversion-search" class="column-search" action="/admin/tasks" method="get" style="width:300px;flex-wrap:wrap;align-items:center"><input type="hidden" name="page" value="1"><input type="hidden" name="page_size" value="{page_size}"><input type="hidden" name="client_ip" value="{_text(client_ip_query)}"><input type="hidden" name="user" value="{_text(user_query)}"><input type="hidden" name="task_id" value="{_text(task_id_query)}"><input type="hidden" name="provider" value="{_text(provider_query)}"><input type="hidden" name="starred" value="{('true' if starred else 'false')}"><input type="hidden" name="time_range" value="{_text(time_range_query)}{_i18n('template_5f28ded94aac')}{conversion_filter_options}{_i18n('template_a3afab951e91')}'
    )
    time_range_options = "".join(
        f'<option value="{value}"{" selected" if value == time_range_query else ""}>'
        f'{_text(label)}</option>'
        for value, label in TIME_RANGE_OPTIONS
    )
    time_range_filter = (
        '<form class="time-range-filter" action="/admin/tasks" method="get" '
        'style="display:inline-flex;align-items:center;gap:6px;margin:0 0 12px;'
        'height:36px;padding:6px 8px;border:1px solid #d8e1ec;border-radius:8px;background:#fff;color:#40536d;font-size:12px">'
        '<input type="hidden" name="page" value="1">'
        f'<input type="hidden" name="page_size" value="{page_size}">'
        f'<input type="hidden" name="client_ip" value="{_text(client_ip_query)}">'
        f'<input type="hidden" name="user" value="{_text(user_query)}">'
        f'<input type="hidden" name="task_id" value="{_text(task_id_query)}">'
        f'<input type="hidden" name="provider" value="{_text(provider_query)}">'
        + ('<input type="hidden" name="starred" value="true">' if starred else '')
        + "".join(
            f'<input type="hidden" name="conversion_status" value="{status}">'
            for status in conversion_statuses
        )
        + f'{_i18n('template_f3e8af90d327')}{time_range_options}</select></label></form>'
    )
    time_range_label = dict(TIME_RANGE_OPTIONS).get(time_range_query, _i18n('template_1a1111c82dfe'))

    def clear_filter_url(filter_name: str, filter_value: str = "") -> str:
        query_items = [("page", "1"), ("page_size", str(page_size))]
        for name, value in (
            ("client_ip", client_ip_query),
            ("user", user_query),
            ("task_id", task_id_query),
            ("provider", provider_query),
            ("time_range", time_range_query),
        ):
            if value and name != filter_name:
                query_items.append((name, value))
        if starred and filter_name != "starred":
            query_items.append(("starred", "true"))
        for status in conversion_statuses:
            if not (
                filter_name == "conversion_status" and status == filter_value
            ):
                query_items.append(("conversion_status", status))
        return "/admin/tasks?" + urlencode(query_items)

    active_filters = []
    if time_range_query:
        active_filters.append(
            (_i18n('template_dc22a0c11491'), time_range_label, clear_filter_url("time_range"))
        )
    for status, label in CONVERSION_FILTER_OPTIONS:
        if status in conversion_statuses:
            active_filters.append(
                (_i18n('template_6478e5e6d208'), label, clear_filter_url("conversion_status", status))
            )
    if user_query:
        active_filters.append((_i18n('template_063ec4bc36b6'), user_query, clear_filter_url("user")))
    if client_ip_query:
        active_filters.append(("IP", client_ip_query, clear_filter_url("client_ip")))
    if task_id_query:
        active_filters.append((_i18n('template_3e3eb34acea6'), task_id_query, clear_filter_url("task_id")))
    if provider_query:
        active_filters.append(("Provider", provider_query, clear_filter_url("provider")))
    if starred:
        active_filters.append((_i18n('template_4293266621b8'), _i18n('template_1dc1f9a399ca'), clear_filter_url("starred")))
    active_filter_chips = (
        _i18n('template_9e9b7d0c1ab0')
        + "".join(
            f'<span style="display:inline-flex;align-items:center;gap:5px;max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:3px 7px;border:1px solid #cfe0f6;border-radius:999px;background:#eef5ff;color:#245b9c"><span style="overflow:hidden;text-overflow:ellipsis">{_text(label)}：{_text(value)}</span><a href="{_text(clear_url)}{_i18n('template_433903e3ed9b')}{_text(label)}{_i18n('template_804d1da7c015')}{_text(value)}{_i18n('template_b5fce338b695')}'
            for label, value, clear_url in active_filters
        )
        + '</div>'
        if active_filters
        else ""
    )
    clock_markup = ''.join(
        f'<span class="current-time"><span class="current-time__label">{_text(zone)}</span><time data-clock-zone="{_text(zone)}">--</time></span>'
        for zone in display_timezones()
    )
    return HTMLResponse(
        f"""{_i18n('template_8e717af0e0f2')}{_text(data['total'])}{_i18n('template_7e99a8cb2afc')}{_text(APP_VERSION)} · {_text(APP_RELEASE_DATE)}{_i18n('template_eb644375d4cc')}{clock_markup}{_i18n('template_31b2e91d9ddd')}{_text(starred_filter_url)}">{(_t('ui_6ff9473563') if starred else '★ ' + _t('ui_bff836d78d'))}</a>{time_range_filter}{active_filter_chips}</div>\n<template id="conversion-search-template">{conversion_filter_form}{_i18n('template_1635e1a58008')}{page_size}"><input type="hidden" name="client_ip" value="{_text(client_ip_query)}"><input type="hidden" name="user" value="{_text(user_query)}"><input type="hidden" name="provider" value="{_text(provider_query)}"><input name="task_id" type="search" value="{_text(task_id_query)}{_i18n('template_846c1a8e355b')}{page_size}"><input type="hidden" name="client_ip" value="{_text(client_ip_query)}"><input type="hidden" name="user" value="{_text(user_query)}"><input type="hidden" name="task_id" value="{_text(task_id_query)}"><input name="provider" type="search" value="{_text(provider_query)}{_i18n('template_9cf32afef570')}{page_size}"><input type="hidden" name="user" value="{_text(user_query)}"><input type="hidden" name="task_id" value="{_text(task_id_query)}"><input type="hidden" name="provider" value="{_text(provider_query)}"><input name="client_ip" type="search" value="{_text(client_ip_query)}{_i18n('template_856c47de30d6')}{page_size}"><input type="hidden" name="client_ip" value="{_text(client_ip_query)}"><input type="hidden" name="task_id" value="{_text(task_id_query)}"><input type="hidden" name="provider" value="{_text(provider_query)}"><input name="user" type="search" value="{_text(user_query)}{_i18n('template_12caa5d405f0')}{body}</tbody></table></div>\n{pagination}\n</main><script src="https://cdn.jsdelivr.net/npm/glightbox@3.3.1/dist/js/glightbox.min.js"></script><script>\nconst statusEl=document.getElementById('service-status');\nconst statusText=statusEl.querySelector('.service-text');\nconst clocks=[...document.querySelectorAll('[data-clock-zone]')].map(element=>[element,element.dataset.clockZone]);\n{_i18n('template_e4e7470d5e25')}"""
    )
