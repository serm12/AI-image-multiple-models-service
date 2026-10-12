from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from app.services.security import require_admin_login
from app.services import algorithm_settings as settings
from app.services.i18n import t

router = APIRouter(prefix='/admin/algorithm-settings', dependencies=[Depends(require_admin_login)])


class AlgorithmInput(BaseModel):
    factor: int = Field(strict=True, ge=1, le=999999999)


@router.get('')
def read_settings():
    return JSONResponse(settings.response(), headers={'Cache-Control': 'no-store'})


@router.post('')
def save_settings(value: AlgorithmInput, request: Request):
    if request.headers.get('origin') != str(request.base_url).rstrip('/'):
        raise HTTPException(403, t('request.invalid_origin'))
    settings.save(value.factor)
    return JSONResponse(settings.response(), headers={'Cache-Control': 'no-store'})
