from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from app.services.security import require_admin_login
from app.services import provider_credentials as credentials
from app.services.i18n import t

router = APIRouter(prefix='/admin/provider-credentials', dependencies=[Depends(require_admin_login)])


@router.get('')
def read_settings():
    try:
        return JSONResponse(credentials.status(), headers={'Cache-Control': 'no-store'})
    except Exception:
        raise HTTPException(503, t('keys_error')) from None


@router.post('')
async def save_settings(request: Request):
    if request.headers.get('origin') != str(request.base_url).rstrip('/'):
        raise HTTPException(403, t('request.invalid_origin'))
    # Do not return validation details containing submitted credentials.
    try:
        values = await request.json()
        from starlette.concurrency import run_in_threadpool
        await run_in_threadpool(credentials.save, values)
        return JSONResponse(credentials.status(), headers={'Cache-Control': 'no-store'})
    except ValueError:
        raise HTTPException(400, t('keys_error')) from None
    except Exception:
        raise HTTPException(503, t('keys_error')) from None
