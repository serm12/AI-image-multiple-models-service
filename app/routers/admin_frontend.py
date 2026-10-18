"""React admin entry points and authenticated view models and task actions."""
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from zoneinfo import ZoneInfo
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from app.services.security import require_admin_login, get_admin_session_user, ADMIN_SESSION_COOKIE
from app.services.i18n import selected_locale
from app.services.time_settings import display_timezones
from app.services.task_query_service import list_task_summaries, set_task_starred
from app.core.version import APP_VERSION, APP_RELEASE_DATE

DIST = Path(__file__).resolve().parents[1] / 'static' / 'admin'
router = APIRouter()

def shell(status_code=200):
    index = DIST / 'index.html'
    if not index.is_file():
        raise HTTPException(503, 'Admin frontend is not built. Run npm --prefix frontend ci and npm --prefix frontend run build.')
    content = index.read_text('utf-8').replace('lang="zh-CN"', f'lang="{selected_locale()}"')
    return HTMLResponse(content, status_code=status_code, headers={'Cache-Control': 'no-store'})

@router.get('/admin/login', include_in_schema=False)
def login_page(request: Request, next: str = '/admin/tasks'):
    from app.routers.admin import _admin_next
    if get_admin_session_user(request.cookies.get(ADMIN_SESSION_COOKIE)) or getattr(request.state, 'shopify_admin', False):
        return RedirectResponse(_admin_next(next), status_code=303)
    return shell()

@router.get('/admin', include_in_schema=False)
@router.get('/admin/', include_in_schema=False)
@router.get('/admin/tasks', include_in_schema=False)
@router.get('/admin/styles', include_in_schema=False)
@router.get('/admin/orders', include_in_schema=False)
@router.get('/admin/config', include_in_schema=False)
def page(_username: str = Depends(require_admin_login)):
    return shell()

@router.get('/admin/api/session')
def session(request: Request, _username: str = Depends(require_admin_login)):
    return JSONResponse({'username': _username, 'shopify_embedded': bool(getattr(request.state, 'shopify_admin', False)), 'locale': selected_locale(), 'version': APP_VERSION,
        'release_date': APP_RELEASE_DATE, 'timezones': display_timezones()}, headers={'Cache-Control': 'no-store'})

@router.get('/admin/api/tasks')
def tasks(page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=100),
          client_ip: str = '', user: str = '', task_id: str = '', provider: str = '',
          conversion_status: list[str] = Query(default=[]), time_range: str = '', starred: bool = False, recycle_bin: bool = False,
          _username: str = Depends(require_admin_login)):
    from app.services.shopify_orders import read_order_cache
    from app.routers.admin import _display_time
    data = list_task_summaries(page=page, page_size=page_size, client_ip_query=client_ip.strip(),
        user_query=user.strip(), task_id_query=task_id.strip(), provider_query=provider.strip(),
        conversion_statuses=conversion_status, time_range=time_range, starred_only=starred, recycle_bin=recycle_bin)
    orders = read_order_cache().get('tasks', {})
    for task in data['tasks']:
        identifier = task['task_id']
        task['shopify_orders'] = orders.get(identifier, [])
        task['display_times'] = {zone: _display_time(task['created_at'], task.get('time_zone'), ZoneInfo(zone)) for zone in display_timezones()}
        task['images'] = []
        for ref in task.get('source_reference_files', []):
            task['images'].append({'url': ref['url'], 'reference': True, 'filename': ref['filename'],
                'thumbnail': '/admin/tasks/' + quote(ref['task_id'], safe='') + '/thumbnail/' + quote(ref['filename'], safe='')})
        for url in task.get('output_files', []):
            name = unquote(Path(urlparse(str(url)).path).name)
            task['images'].append({'url': url, 'reference': False, 'filename': name,
                'thumbnail': '/admin/tasks/' + quote(identifier, safe='') + '/thumbnail/' + quote(name, safe='')})
    return JSONResponse(data, headers={'Cache-Control': 'no-store'})

class StarInput(BaseModel):
    starred: bool

@router.post('/admin/api/tasks/{task_id}/star')
def star(task_id: str, value: StarInput, request: Request, _username: str = Depends(require_admin_login)):
    if request.headers.get('origin') != str(request.base_url).rstrip('/'):
        raise HTTPException(403, 'Invalid request origin')
    result = set_task_starred(task_id, value.starred)
    if result is None:
        raise HTTPException(404, 'Task not found')
    return {'starred': result}

@router.get('/admin/api/styles')
def styles(_username: str = Depends(require_admin_login)):
    from app.core.config import ArtStyleEnum, STYLE_PROMPTS
    from app.core.helpers import get_style_description
    return JSONResponse([{'id': style.value, 'description': get_style_description(style.value),
        'prompt': STYLE_PROMPTS.get(style, '')} for style in ArtStyleEnum], headers={'Cache-Control': 'no-store'})


class TaskSelection(BaseModel):
    task_ids: list[Annotated[str, Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]{0,159}$')]] = Field(min_length=1, max_length=100)
    confirmed: Literal[True]


@router.post('/admin/api/tasks/delete')
def recycle_tasks(value: TaskSelection, request: Request, _username: str = Depends(require_admin_login)):
    if request.headers.get('origin') != str(request.base_url).rstrip('/'):
        raise HTTPException(403, 'Invalid request origin')
    from app.services.task_recycle_bin import change_tasks
    return change_tasks(value.task_ids)


@router.post('/admin/api/tasks/restore')
def restore_tasks(value: TaskSelection, request: Request, _username: str = Depends(require_admin_login)):
    if request.headers.get('origin') != str(request.base_url).rstrip('/'):
        raise HTTPException(403, 'Invalid request origin')
    from app.services.task_recycle_bin import change_tasks
    return change_tasks(value.task_ids, restore=True)
