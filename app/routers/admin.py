import os
import tempfile
from html import escape
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, ImageOps

from app.core.version import APP_RELEASE_DATE, APP_VERSION
from app.services.security import require_admin_login
from app.services.task_files import resolve_task_file_path
from app.services.task_query_service import list_task_summaries
from app.utils.time_utils import CHINA_TIMEZONE, CHINA_TIMEZONE_NAME


router = APIRouter()
US_EASTERN_TIMEZONE = ZoneInfo("America/New_York")


def _text(value) -> str:
    return escape(str(value or ""))


def _display_time(value, source_timezone="UTC", target_timezone=CHINA_TIMEZONE) -> str:
    raw = str(value or "")
    try:
        parsed = datetime.strptime(raw, "%Y%m%d_%H%M%S")
        source_zone = (
            CHINA_TIMEZONE
            if source_timezone == CHINA_TIMEZONE_NAME
            else timezone.utc
        )
        return (
            parsed.replace(tzinfo=source_zone)
            .astimezone(target_timezone)
            .strftime("%Y-%m-%d %H:%M:%S")
        )
    except ValueError:
        return raw


def _task_row(task: dict) -> str:
    task_id = str(task["task_id"])
    gallery_name = _text(f"task-{task_id}")
    images = ""
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
    if source_page_url:
        parsed_source = urlparse(raw_source_page_url)
        source_path = f"{parsed_source.netloc}{parsed_source.path or '/'}"
        source_title = _text(
            task.get("source_product_title") or task.get("source_page_title")
        )
        source_handle = _text(task.get("source_product_handle"))
        source_title_html = (
            f'<span class="source-title">{source_title}</span>' if source_title else ""
        )
        source_detail = source_path
        if source_handle and source_handle not in source_path:
            source_detail = f"{source_detail} · {source_handle}"
        source_link = (
            f'<a class="source-link" href="{source_page_url}" title="{source_page_url}" '
            f'target="_blank" rel="noopener noreferrer">{source_title_html}'
            f'<span class="source-path">{_text(source_detail)}</span></a>'
        )
    else:
        source_link = '<span class="muted">—</span>'
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
    prompt_view = (
        '<details class="prompt-details"><summary>查看提示词</summary>'
        f'<div class="prompt-card">{prompt}</div></details>'
        if prompt
        else '<span class="muted">—</span>'
    )
    duration_value = task.get("generation_duration_seconds")
    try:
        duration = f"{float(duration_value):.1f} 秒"
    except (TypeError, ValueError):
        duration = '<span class="muted">—</span>'
    return (
        "<tr>"
        f'<td><code title="{task_id}">{task_id[:17]}…</code></td>'
        f'<td><span class="status">{_text(task["status"])}</span></td>'
        f'<td class="nowrap">{_text(_display_time(task["created_at"], task.get("time_zone")))}</td>'
        f'<td class="nowrap">{_text(_display_time(task["created_at"], task.get("time_zone"), US_EASTERN_TIMEZONE))}</td>'
        f"<td>{duration}</td>"
        f'<td class="provider">{_text(task.get("api_provider"))}</td>'
        f'<td class="client-ip">{client_ip}</td>'
        f'<td><span class="country">{country}</span></td>'
        f'<td class="url">{source_link}</td>'
        f'<td class="url">{request_link}</td>'
        f'<td class="prompt">{prompt_view}</td>'
        f'<td class="images-cell">{image_strip}</td>'
        "</tr>"
    )


def _pagination(data: dict) -> str:
    page = int(data["page"])
    page_size = int(data["page_size"])
    total_pages = int(data["total_pages"])

    def page_link(target: int, label: str, class_name: str = "") -> str:
        classes = f' class="{class_name}"' if class_name else ""
        return (
            f'<a{classes} href="/admin/tasks?page={target}&amp;page_size={page_size}">'
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
    _username: str = Depends(require_admin_login),
):
    data = list_task_summaries(page=page, page_size=page_size)
    rows = "".join(_task_row(task) for task in data["tasks"])
    body = rows or '<tr><td colspan="12" class="empty">暂无任务记录</td></tr>'
    pagination = _pagination(data)
    return HTMLResponse(
        f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow"><title>AI 图片生成记录</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/glightbox@3.3.1/dist/css/glightbox.min.css">
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f7fb;color:#172033;font:14px/1.5 system-ui,-apple-system,sans-serif}}
main{{max-width:1800px;margin:auto;padding:28px}}header{{display:flex;justify-content:space-between;align-items:end;margin-bottom:18px}}
h1{{margin:0;font-size:25px}}.header-meta{{display:flex;align-items:center;gap:10px;margin-top:2px}}.count,.muted{{color:#718096}}.version{{color:#4f6380;font-size:12px;padding:2px 7px;border:1px solid #dce4ee;border-radius:999px;background:#fff}}.current-times{{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}}.current-time{{display:flex;align-items:baseline;gap:7px;padding:5px 9px;border:1px solid #dce4ee;border-radius:8px;background:#fff;color:#24344d;font-variant-numeric:tabular-nums}}.current-time__label{{color:#718096;font-size:12px}}.current-time time{{font-weight:600;white-space:nowrap}}.panel{{background:#fff;border:1px solid #e3e8f0;border-radius:14px;overflow:auto;box-shadow:0 8px 30px #18243b0d}}
.service-status{{display:inline-flex;align-items:center;gap:8px;padding:7px 11px;border:1px solid #dce7df;border-radius:999px;background:#f5fbf6;color:#28733a;font-size:13px;font-weight:600}}.service-dot{{width:8px;height:8px;border-radius:50%;background:#22a447;box-shadow:0 0 0 3px #22a44720}}.service-status.checking{{color:#718096;background:#f8fafc;border-color:#e3e8f0}}.service-status.checking .service-dot{{background:#94a3b8;box-shadow:none}}.service-status.error{{color:#b42318;background:#fff6f5;border-color:#f4d6d2}}.service-status.error .service-dot{{background:#e23b2e;box-shadow:0 0 0 3px #e23b2e20}}
table{{width:100%;border-collapse:collapse;min-width:1835px;table-layout:fixed}}th,td{{padding:13px 12px;border-bottom:1px solid #edf0f5;text-align:left;vertical-align:middle}}th{{position:sticky;top:0;z-index:2;background:#f8fafc;font-size:12px;color:#64748b;white-space:nowrap}}tbody tr{{height:94px}}tbody tr:hover{{background:#fafcff}}
th:nth-child(1){{width:180px}}th:nth-child(2){{width:105px}}th:nth-child(3),th:nth-child(4){{width:165px}}th:nth-child(5){{width:90px}}th:nth-child(6){{width:170px}}th:nth-child(7){{width:155px}}th:nth-child(8){{width:90px}}th:nth-child(9){{width:260px}}th:nth-child(10){{width:145px}}th:nth-child(11){{width:115px}}th:nth-child(12){{width:300px}}
code{{font-size:12px;white-space:nowrap}}.nowrap{{white-space:nowrap}}.provider{{overflow-wrap:anywhere}}.status{{display:inline-block;padding:4px 9px;border-radius:999px;background:#eaf7ed;color:#247436;font-weight:600}}.client-ip{{white-space:normal;line-height:1.35}}.client-ip__address{{overflow-wrap:anywhere;word-break:break-all}}.ip-task-sequence,.ip-task-total{{display:inline-flex;align-items:center;justify-content:center;margin-top:3px;padding:2px 6px;border-radius:999px;font-size:11px;font-weight:700;line-height:1.4;white-space:nowrap}}.ip-task-sequence{{min-width:28px;margin-left:4px;background:#e8f1ff;color:#245b9c}}.ip-task-total{{background:#eef7ed;color:#28733a}}.country{{display:inline-flex;min-width:34px;justify-content:center;padding:3px 7px;border-radius:6px;background:#eef3fa;color:#3f5675;font-weight:600}}.url a{{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.source-link .source-title,.source-link .source-path{{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.source-link .source-title{{font-weight:650;color:#174f9b}}.source-link .source-path{{font-size:11px;color:#718096}}a{{color:#1769d2}}th:nth-child(12),td.images-cell{{position:sticky;right:0;background:#fff;box-shadow:-8px 0 16px #1720330a}}th:nth-child(12){{z-index:4;background:#f8fafc}}tbody tr:hover td.images-cell{{background:#fafcff}}.image-strip{{display:grid;grid-template-columns:28px minmax(0,1fr) 28px;align-items:center;gap:5px;min-width:0}}.images{{display:flex;gap:7px;min-width:0;overflow-x:auto;overscroll-behavior-inline:contain;scroll-behavior:smooth;scroll-snap-type:x proximity;scrollbar-width:thin;scrollbar-color:#a8b5c7 #edf2f7;padding:2px 2px 7px}}.images::-webkit-scrollbar{{height:7px}}.images::-webkit-scrollbar-track{{background:#edf2f7;border-radius:999px}}.images::-webkit-scrollbar-thumb{{background:#a8b5c7;border-radius:999px}}.image-item{{flex:0 0 auto;scroll-snap-align:start}}.images img{{display:block;width:68px;height:68px;object-fit:cover;border-radius:8px;border:1px solid #dbe2ea;transition:.15s}}.images img:hover{{transform:scale(1.04)}}.image-scroll{{display:grid;place-items:center;width:28px;height:42px;padding:0;border:1px solid #dbe2ea;border-radius:8px;background:#f8fafc;color:#36516f;font:700 22px/1 system-ui;cursor:pointer}}.image-scroll:hover:not(:disabled){{background:#eaf2fb;border-color:#b9cae0}}.image-scroll:disabled{{opacity:.28;cursor:default}}.pagination{{display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:14px;margin:18px 0 2px;color:#53657e}}.pagination__links{{display:flex;align-items:center;gap:6px}}.pagination__page,.pagination__direction{{display:inline-flex;align-items:center;justify-content:center;min-width:34px;height:34px;padding:0 10px;border:1px solid #d8e1ec;border-radius:8px;background:#fff;color:#245b9c;text-decoration:none}}.pagination__page.is-current{{border-color:#366fac;background:#366fac;color:#fff}}.pagination__direction.is-disabled{{color:#a2adbb;background:#f3f6f9}}.pagination__ellipsis{{padding:0 2px}}.pagination__size{{display:flex;align-items:center;gap:6px}}.pagination__size select{{height:34px;padding:0 28px 0 9px;border:1px solid #d8e1ec;border-radius:8px;background:#fff;color:#33455e}}.pagination__summary{{font-size:12px}}.gcounter{{position:fixed;left:50%;bottom:12px;z-index:100001;transform:translateX(-50%);padding:5px 11px;border-radius:999px;background:#101827d9;color:#fff;font-size:13px;font-variant-numeric:tabular-nums;pointer-events:none}}.empty{{padding:50px;text-align:center;color:#718096}}
.prompt-details{{position:relative}}.prompt-details summary{{cursor:pointer;color:#1769d2;white-space:nowrap;list-style:none}}.prompt-details summary::-webkit-details-marker{{display:none}}.prompt-details summary:after{{content:" ›"}}.prompt-details[open] summary:after{{content:" ×"}}.prompt-card{{position:absolute;right:0;top:30px;z-index:10;width:min(460px,70vw);max-height:320px;overflow:auto;padding:15px;border:1px solid #dbe2ea;border-radius:10px;background:#fff;box-shadow:0 14px 40px #1720332b;white-space:pre-wrap;line-height:1.65}}
@media(max-width:700px){{main{{padding:16px}}h1{{font-size:21px}}header{{align-items:center}}.current-times{{flex-direction:column;align-items:flex-start}}.panel{{border-radius:10px}}}}
</style></head><body><main><header><div><h1>AI 图片生成记录</h1><div class="header-meta"><span class="count">共 {_text(data['total'])} 条任务</span><span class="version">v{_text(APP_VERSION)} · {_text(APP_RELEASE_DATE)}</span></div><div class="current-times" aria-label="当前时间"><span class="current-time"><span class="current-time__label">北京时间</span><time id="current-beijing-time">--</time></span><span class="current-time"><span class="current-time__label">美国东部</span><time id="current-us-eastern-time">--</time></span></div></div><div id="service-status" class="service-status checking"><span class="service-dot"></span><span class="service-text">状态检测中</span></div></header>
<div class="panel"><table><thead><tr><th>任务 ID</th><th>状态</th><th>时间（北京时间）</th><th>时间（美国东部）</th><th>总耗时</th><th>Provider</th><th>访客 IP</th><th>国家/地区</th><th>来源页面</th><th>请求 URL</th><th>提示词</th><th>图片</th></tr></thead><tbody>{body}</tbody></table></div>
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
  window.location.href=`/admin/tasks?page=1&page_size=${{event.target.value}}`;
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
