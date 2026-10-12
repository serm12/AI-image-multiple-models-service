from app.services.i18n import render as _i18n, Message as _i18n_message
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Literal
from zoneinfo import available_timezones
from app.services.security import require_admin_login
from app.services import time_settings as settings

router = APIRouter(prefix='/admin/time-settings', dependencies=[Depends(require_admin_login)])


class TimeSettingsInput(BaseModel):
    mode: Literal['shop', 'custom'] = 'shop'
    custom_timezone: str = 'UTC'


@router.get('')
async def read_settings():
    error = False
    try:
        await settings.sync_shop_timezone(force=True)
    except Exception:
        error = True
    return JSONResponse({**settings.settings_response(), 'sync_failed': error}, headers={'Cache-Control': 'no-store'})


@router.post('')
def save_settings(value: TimeSettingsInput, request: Request):
    if request.headers.get('origin') != str(request.base_url).rstrip('/'):
        raise HTTPException(403, 'Invalid request origin')
    if value.custom_timezone not in available_timezones():
        raise HTTPException(422, _i18n('template_d220d6bfa02b'))
    settings.update(value.model_dump())
    return JSONResponse(settings.settings_response(), headers={'Cache-Control': 'no-store'})
