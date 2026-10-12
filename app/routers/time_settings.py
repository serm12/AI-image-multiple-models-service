from app.services.i18n import render as _i18n, Message as _i18n_message
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal
from zoneinfo import available_timezones
from app.services.security import require_admin_login
from app.services import time_settings as settings

router = APIRouter(prefix='/admin/time-settings', dependencies=[Depends(require_admin_login)])


class TimeSettingsInput(BaseModel):
    timezone: str | None = None
    comparison_timezones: list[str] | None = Field(default=None, max_length=5)
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
    values = value.model_dump(exclude={'timezone', 'comparison_timezones'})
    if value.timezone is not None:
        if value.timezone not in available_timezones():
            raise HTTPException(422, _i18n('template_d220d6bfa02b'))
        values.update(mode='shop' if value.timezone == settings.read().get('shop_timezone') else 'custom', custom_timezone=value.timezone)
    if value.comparison_timezones is not None:
        if any(zone not in available_timezones() for zone in value.comparison_timezones):
            raise HTTPException(422, _i18n('template_d220d6bfa02b'))
        values['comparison_timezones'] = list(dict.fromkeys(value.comparison_timezones))
    settings.update(values)
    return JSONResponse(settings.settings_response(), headers={'Cache-Control': 'no-store'})
