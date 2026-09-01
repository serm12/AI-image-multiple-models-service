from html import escape
from datetime import datetime
from urllib.parse import urlparse

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from app.core.version import APP_RELEASE_DATE, APP_VERSION
from app.services.security import require_admin_login
from app.services.task_query_service import list_task_summaries


router = APIRouter()


def _text(value) -> str:
    return escape(str(value or ""))


def _display_time(value) -> str:
    raw = str(value or "")
    try:
        return datetime.strptime(raw, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return raw


def _task_row(task: dict) -> str:
    images = "".join(
        f'<a href="{_text(url)}" target="_blank" rel="noopener">'
        f'<img src="{_text(url)}" loading="lazy" alt="Generated image"></a>'
        for url in task.get("output_files", [])
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
    task_id = _text(task["task_id"])
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
        f'<td class="nowrap">{_text(_display_time(task["created_at"]))}</td>'
        f"<td>{duration}</td>"
        f'<td class="provider">{_text(task.get("api_provider"))}</td>'
        f'<td class="nowrap">{client_ip}</td>'
        f'<td><span class="country">{country}</span></td>'
        f'<td class="url">{request_link}</td>'
        f'<td class="prompt">{prompt_view}</td>'
        f'<td><div class="images">{images}</div></td>'
        "</tr>"
    )


@router.get("/admin/tasks", response_class=HTMLResponse)
def admin_tasks(_username: str = Depends(require_admin_login)):
    data = list_task_summaries()
    rows = "".join(_task_row(task) for task in data["tasks"])
    body = rows or '<tr><td colspan="10" class="empty">暂无任务记录</td></tr>'
    return HTMLResponse(
        f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow"><title>AI 图片生成记录</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f7fb;color:#172033;font:14px/1.5 system-ui,-apple-system,sans-serif}}
main{{max-width:1800px;margin:auto;padding:28px}}header{{display:flex;justify-content:space-between;align-items:end;margin-bottom:18px}}
h1{{margin:0;font-size:25px}}.header-meta{{display:flex;align-items:center;gap:10px;margin-top:2px}}.count,.muted{{color:#718096}}.version{{color:#4f6380;font-size:12px;padding:2px 7px;border:1px solid #dce4ee;border-radius:999px;background:#fff}}.panel{{background:#fff;border:1px solid #e3e8f0;border-radius:14px;overflow:auto;box-shadow:0 8px 30px #18243b0d}}
.service-status{{display:inline-flex;align-items:center;gap:8px;padding:7px 11px;border:1px solid #dce7df;border-radius:999px;background:#f5fbf6;color:#28733a;font-size:13px;font-weight:600}}.service-dot{{width:8px;height:8px;border-radius:50%;background:#22a447;box-shadow:0 0 0 3px #22a44720}}.service-status.checking{{color:#718096;background:#f8fafc;border-color:#e3e8f0}}.service-status.checking .service-dot{{background:#94a3b8;box-shadow:none}}.service-status.error{{color:#b42318;background:#fff6f5;border-color:#f4d6d2}}.service-status.error .service-dot{{background:#e23b2e;box-shadow:0 0 0 3px #e23b2e20}}
table{{width:100%;border-collapse:collapse;min-width:1380px;table-layout:fixed}}th,td{{padding:13px 12px;border-bottom:1px solid #edf0f5;text-align:left;vertical-align:middle}}th{{position:sticky;top:0;z-index:2;background:#f8fafc;font-size:12px;color:#64748b;white-space:nowrap}}tbody tr{{height:94px}}tbody tr:hover{{background:#fafcff}}
th:nth-child(1){{width:180px}}th:nth-child(2){{width:105px}}th:nth-child(3){{width:165px}}th:nth-child(4){{width:90px}}th:nth-child(5){{width:170px}}th:nth-child(6){{width:125px}}th:nth-child(7){{width:90px}}th:nth-child(8){{width:145px}}th:nth-child(9){{width:115px}}th:nth-child(10){{width:265px}}
code{{font-size:12px;white-space:nowrap}}.nowrap{{white-space:nowrap}}.provider{{overflow-wrap:anywhere}}.status{{display:inline-block;padding:4px 9px;border-radius:999px;background:#eaf7ed;color:#247436;font-weight:600}}.country{{display:inline-flex;min-width:34px;justify-content:center;padding:3px 7px;border-radius:6px;background:#eef3fa;color:#3f5675;font-weight:600}}.url a{{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}a{{color:#1769d2}}.images{{display:flex;gap:7px;overflow-x:auto;padding:2px}}.images img{{display:block;width:68px;height:68px;object-fit:cover;border-radius:8px;border:1px solid #dbe2ea;transition:.15s}}.images img:hover{{transform:scale(1.04)}}.empty{{padding:50px;text-align:center;color:#718096}}
.prompt-details{{position:relative}}.prompt-details summary{{cursor:pointer;color:#1769d2;white-space:nowrap;list-style:none}}.prompt-details summary::-webkit-details-marker{{display:none}}.prompt-details summary:after{{content:" ›"}}.prompt-details[open] summary:after{{content:" ×"}}.prompt-card{{position:absolute;right:0;top:30px;z-index:10;width:min(460px,70vw);max-height:320px;overflow:auto;padding:15px;border:1px solid #dbe2ea;border-radius:10px;background:#fff;box-shadow:0 14px 40px #1720332b;white-space:pre-wrap;line-height:1.65}}
@media(max-width:700px){{main{{padding:16px}}h1{{font-size:21px}}header{{align-items:center}}.panel{{border-radius:10px}}}}
</style></head><body><main><header><div><h1>AI 图片生成记录</h1><div class="header-meta"><span class="count">共 {_text(data['total'])} 条任务</span><span class="version">v{_text(APP_VERSION)} · {_text(APP_RELEASE_DATE)}</span></div></div><div id="service-status" class="service-status checking"><span class="service-dot"></span><span class="service-text">状态检测中</span></div></header>
<div class="panel"><table><thead><tr><th>任务 ID</th><th>状态</th><th>时间</th><th>总耗时</th><th>Provider</th><th>访客 IP</th><th>国家/地区</th><th>请求 URL</th><th>提示词</th><th>图片</th></tr></thead><tbody>{body}</tbody></table></div>
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
