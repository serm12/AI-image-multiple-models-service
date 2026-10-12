from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from app.services.security import require_admin_login
from app.services.i18n import SUPPORTED, current_locale, selected_locale, save_locale, t

router = APIRouter(prefix='/admin/language-settings', dependencies=[Depends(require_admin_login)])


class LanguageInput(BaseModel):
    locale: str


class LanguageMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        locale = selected_locale()
        token = current_locale.set(locale)
        try:
            response = await call_next(request)
            if request.url.path.startswith('/admin/'):
                response.headers['Content-Language'] = locale
                response.headers['Cache-Control'] = 'no-store'
            return response
        finally:
            current_locale.reset(token)


@router.get('')
def read_settings():
    return {'locale': selected_locale(), 'languages': SUPPORTED}


@router.post('')
def update_settings(value: LanguageInput, request: Request):
    if request.headers.get('origin') != str(request.base_url).rstrip('/'):
        raise HTTPException(403, t('request.invalid_origin'))
    if value.locale not in SUPPORTED:
        raise HTTPException(422, t('language.invalid'))
    save_locale(value.locale)
    return JSONResponse({'locale': value.locale})
