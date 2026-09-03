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
    images = ""
    for url in task.get("output_files", []):
        filename = unquote(os.path.basename(urlparse(str(url)).path))
        thumbnail_url = (
            f"/admin/tasks/{quote(task_id, safe='')}/thumbnail/"
            f"{quote(filename, safe='')}"
        )
        images += (
            f'<a href="{_text(url)}" target="_blank" rel="noopener">'
            f'<img src="{_text(thumbnail_url)}" loading="lazy" decoding="async" '
            'alt="Generated image"></a>'
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
    client_ip = _text(task.get("client_ip")) or '<span class="muted">—</span>'
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
        f'<td class="nowrap">{client_ip}</td>'
        f'<td><span class="country">{country}</span></td>'
        f'<td class="url">{request_link}</td>'
        f'<td class="prompt">{prompt_view}</td>'
        f'<td><div class="images">{images}</div></td>'
        "</tr>"
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
def admin_tasks(_username: str = Depends(require_admin_login)):
    data = list_task_summaries()
    rows = "".join(_task_row(task) for task in data["tasks"])
    body = rows or '<tr><td colspan="11" class="empty">暂无任务记录</td></tr>'
    return HTMLResponse(
        f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow"><title>AI 图片生成记录</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f7fb;color:#172033;font:14px/1.5 system-ui,-apple-system,sans-serif}}
main{{max-width:1800px;margin:auto;padding:28px}}header{{display:flex;justify-content:space-between;align-items:end;margin-bottom:18px}}
h1{{margin:0;font-size:25px}}.header-meta{{display:flex;align-items:center;gap:10px;margin-top:2px}}.count,.muted{{color:#718096}}.version{{color:#4f6380;font-size:12px;padding:2px 7px;border:1px solid #dce4ee;border-radius:999px;background:#fff}}.panel{{background:#fff;border:1px solid #e3e8f0;border-radius:14px;overflow:auto;box-shadow:0 8px 30px #18243b0d}}
.service-status{{display:inline-flex;align-items:center;gap:8px;padding:7px 11px;border:1px solid #dce7df;border-radius:999px;background:#f5fbf6;color:#28733a;font-size:13px;font-weight:600}}.service-dot{{width:8px;height:8px;border-radius:50%;background:#22a447;box-shadow:0 0 0 3px #22a44720}}.service-status.checking{{color:#718096;background:#f8fafc;border-color:#e3e8f0}}.service-status.checking .service-dot{{background:#94a3b8;box-shadow:none}}.service-status.error{{color:#b42318;background:#fff6f5;border-color:#f4d6d2}}.service-status.error .service-dot{{background:#e23b2e;box-shadow:0 0 0 3px #e23b2e20}}
table{{width:100%;border-collapse:collapse;min-width:1545px;table-layout:fixed}}th,td{{padding:13px 12px;border-bottom:1px solid #edf0f5;text-align:left;vertical-align:middle}}th{{position:sticky;top:0;z-index:2;background:#f8fafc;font-size:12px;color:#64748b;white-space:nowrap}}tbody tr{{height:94px}}tbody tr:hover{{background:#fafcff}}
th:nth-child(1){{width:180px}}th:nth-child(2){{width:105px}}th:nth-child(3),th:nth-child(4){{width:165px}}th:nth-child(5){{width:90px}}th:nth-child(6){{width:170px}}th:nth-child(7){{width:125px}}th:nth-child(8){{width:90px}}th:nth-child(9){{width:145px}}th:nth-child(10){{width:115px}}th:nth-child(11){{width:265px}}
code{{font-size:12px;white-space:nowrap}}.nowrap{{white-space:nowrap}}.provider{{overflow-wrap:anywhere}}.status{{display:inline-block;padding:4px 9px;border-radius:999px;background:#eaf7ed;color:#247436;font-weight:600}}.country{{display:inline-flex;min-width:34px;justify-content:center;padding:3px 7px;border-radius:6px;background:#eef3fa;color:#3f5675;font-weight:600}}.url a{{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}a{{color:#1769d2}}.images{{display:flex;gap:7px;overflow-x:auto;padding:2px}}.images img{{display:block;width:68px;height:68px;object-fit:cover;border-radius:8px;border:1px solid #dbe2ea;transition:.15s}}.images img:hover{{transform:scale(1.04)}}.empty{{padding:50px;text-align:center;color:#718096}}
.prompt-details{{position:relative}}.prompt-details summary{{cursor:pointer;color:#1769d2;white-space:nowrap;list-style:none}}.prompt-details summary::-webkit-details-marker{{display:none}}.prompt-details summary:after{{content:" ›"}}.prompt-details[open] summary:after{{content:" ×"}}.prompt-card{{position:absolute;right:0;top:30px;z-index:10;width:min(460px,70vw);max-height:320px;overflow:auto;padding:15px;border:1px solid #dbe2ea;border-radius:10px;background:#fff;box-shadow:0 14px 40px #1720332b;white-space:pre-wrap;line-height:1.65}}
@media(max-width:700px){{main{{padding:16px}}h1{{font-size:21px}}header{{align-items:center}}.panel{{border-radius:10px}}}}
</style></head><body><main><header><div><h1>AI 图片生成记录</h1><div class="header-meta"><span class="count">共 {_text(data['total'])} 条任务</span><span class="version">v{_text(APP_VERSION)} · {_text(APP_RELEASE_DATE)}</span></div></div><div id="service-status" class="service-status checking"><span class="service-dot"></span><span class="service-text">状态检测中</span></div></header>
<div class="panel"><table><thead><tr><th>任务 ID</th><th>状态</th><th>时间（北京时间）</th><th>时间（美国东部）</th><th>总耗时</th><th>Provider</th><th>访客 IP</th><th>国家/地区</th><th>请求 URL</th><th>提示词</th><th>图片</th></tr></thead><tbody>{body}</tbody></table></div>
</main><script>
const statusEl=document.getElementById('service-status');
const statusText=statusEl.querySelector('.service-text');
fetch('/health',{{cache:'no-store'}}).then(response=>{{if(!response.ok)throw new Error();return response.json()}}).then(data=>{{
  statusEl.className='service-status';
  statusText.textContent=data.status==='ok'?'服务正常':'服务异常';
  if(data.status!=='ok')statusEl.classList.add('error');
}}).catch(()=>{{statusEl.className='service-status error';statusText.textContent='服务异常'}});
</script></body></html>"""
    )
