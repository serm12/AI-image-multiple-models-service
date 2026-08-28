from html import escape

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from app.services.security import require_admin_login
from app.services.task_query_service import list_task_summaries


router = APIRouter()


def _text(value) -> str:
    return escape(str(value or ""))


def _task_row(task: dict) -> str:
    images = "".join(
        f'<a href="{_text(url)}" target="_blank" rel="noopener">'
        f'<img src="{_text(url)}" loading="lazy" alt="Generated image"></a>'
        for url in task.get("output_files", [])
    )
    request_url = _text(task.get("request_url"))
    request_link = (
        f'<a href="{request_url}" target="_blank" rel="noopener">{request_url}</a>'
        if request_url
        else '<span class="muted">旧任务未记录</span>'
    )
    client_ip = _text(task.get("client_ip")) or '<span class="muted">—</span>'
    country = _text(task.get("client_country")) or '<span class="muted">—</span>'
    duration_value = task.get("generation_duration_seconds")
    try:
        duration = f"{float(duration_value):.1f} 秒"
    except (TypeError, ValueError):
        duration = '<span class="muted">—</span>'
    return (
        "<tr>"
        f'<td><code>{_text(task["task_id"])}</code></td>'
        f'<td><span class="status">{_text(task["status"])}</span></td>'
        f'<td>{_text(task["created_at"])}</td>'
        f"<td>{duration}</td>"
        f'<td>{_text(task.get("api_provider"))}</td>'
        f"<td>{client_ip}</td>"
        f"<td>{country}</td>"
        f'<td class="url">{request_link}</td>'
        f'<td class="prompt">{_text(task.get("prompt"))}</td>'
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
*{{box-sizing:border-box}}body{{margin:0;background:#f5f7fb;color:#172033;font:14px/1.5 system-ui,-apple-system,sans-serif}}
main{{max-width:1600px;margin:auto;padding:28px}}header{{display:flex;justify-content:space-between;align-items:end;margin-bottom:18px}}
h1{{margin:0;font-size:25px}}.count,.muted{{color:#718096}}.panel{{background:#fff;border:1px solid #e3e8f0;border-radius:14px;overflow:auto;box-shadow:0 8px 30px #18243b0d}}
table{{width:100%;border-collapse:collapse;min-width:1250px}}th,td{{padding:12px 14px;border-bottom:1px solid #edf0f5;text-align:left;vertical-align:top}}th{{position:sticky;top:0;background:#f8fafc;font-size:12px;color:#64748b;text-transform:uppercase}}
code{{font-size:12px}}.status{{display:inline-block;padding:3px 8px;border-radius:999px;background:#edf7ed;color:#287a35}}.url{{max-width:260px;word-break:break-all}}.prompt{{max-width:320px;white-space:pre-wrap}}a{{color:#1769d2}}.images{{display:flex;gap:6px;max-width:360px;overflow:auto}}.images img{{display:block;width:72px;height:72px;object-fit:cover;border-radius:7px;border:1px solid #dbe2ea}}.empty{{padding:50px;text-align:center;color:#718096}}
@media(max-width:700px){{main{{padding:16px}}h1{{font-size:21px}}}}
</style></head><body><main><header><div><h1>AI 图片生成记录</h1><div class="count">共 {_text(data['total'])} 条任务</div></div><a href="/health" target="_blank">服务状态</a></header>
<div class="panel"><table><thead><tr><th>任务 ID</th><th>状态</th><th>时间</th><th>总耗时</th><th>Provider</th><th>访客 IP</th><th>国家/地区</th><th>请求 URL</th><th>提示词</th><th>图片</th></tr></thead><tbody>{body}</tbody></table></div>
</main></body></html>"""
    )
