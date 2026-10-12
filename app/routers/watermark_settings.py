from app.services.i18n import render as _i18n, Message as _i18n_message
import io
import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from starlette.concurrency import run_in_threadpool
from pydantic import ValidationError
from PIL import Image, ImageDraw
from app.services.security import require_admin_login
from app.services import watermark_settings as settings

router = APIRouter(prefix='/admin/watermark', dependencies=[Depends(require_admin_login)])


def settings_page():
    from app.services.i18n import render_page
    page = Path(__file__).with_name('watermark_settings.html').read_text('utf-8')
    return HTMLResponse(render_page(page), headers={'Cache-Control': 'no-store'})


@router.get('/settings')
def read_settings(response: Response):
    response.headers['Cache-Control'] = 'no-store'
    value = settings.load_settings().model_dump(exclude={'watermark_asset', 'brand_asset'})
    return value


@router.get('/assets/{kind}')
def read_asset(kind: str):
    if kind not in {'watermark', 'brand'}:
        raise HTTPException(404)
    path = settings.asset_path(settings.load_settings(), kind)
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, headers={'Cache-Control': 'no-store'})


async def submitted_settings(request):
    # Cookie-authenticated writes must originate from this same application.
    if request.headers.get('origin') != str(request.base_url).rstrip('/'):
        raise HTTPException(403, 'Invalid request origin')
    form = await request.form(max_files=2, max_fields=24, max_part_size=6 * 1024 * 1024)
    try:
        values = json.loads(str(form.get('settings', '{}')))
        if not isinstance(values, dict) or {'watermark_asset', 'brand_asset'} & values.keys():
            raise ValueError('Invalid fields')
        current = settings.load_settings()
        result = settings.WatermarkSettings.model_validate({**values,
            'watermark_asset': current.watermark_asset, 'brand_asset': current.brand_asset})
        for field in ('watermark', 'brand'):
            upload = form.get(field)
            if upload is not None and getattr(upload, 'filename', ''):
                content = await upload.read(5 * 1024 * 1024 + 1)
                name = await run_in_threadpool(settings.store_asset, content)
                setattr(result, field + '_asset', name)
        return result
    except (ValueError, ValidationError) as error:
        raise HTTPException(400, _i18n('template_79cff0faec60') + str(error)[:350]) from None
    finally:
        await form.close()


@router.post('/settings')
async def update_settings(request: Request):
    value = await submitted_settings(request)
    await run_in_threadpool(settings.save_settings, value)
    return {'message': _i18n('template_e1383ec1b292')}


def render_preview(value):
    from app.utils.watermark_utils import add_logo_watermark
    with tempfile.TemporaryDirectory() as folder:
        source, output = Path(folder) / 'sample.png', Path(folder) / 'preview.png'
        image = Image.new('RGB', (900, 1100), '#e8e0d5')
        draw = ImageDraw.Draw(image)
        for y in range(1100):
            draw.line((0, y, 900, y), fill=(int(220-y/30), int(222-y/23), int(218-y/17)))
        draw.ellipse((160, 120, 740, 700), fill='#c5ab8e')
        draw.ellipse((260, 200, 640, 690), fill='#edcfaf')
        draw.rounded_rectangle((150, 700, 750, 1190), radius=150, fill='#4d686c')
        image.save(source)
        add_logo_watermark(str(source), str(output), resize_scale=0.6, config=settings.runtime_config(value))
        return output.read_bytes()


@router.post('/preview')
async def preview(request: Request):
    value = await submitted_settings(request)
    try:
        content = await run_in_threadpool(render_preview, value)
        return Response(content, media_type='image/png', headers={'Cache-Control': 'no-store'})
    finally:
        # Preview uploads are temporary; never replace the saved assets.
        saved = settings.load_settings()
        for field in ('watermark_asset', 'brand_asset'):
            name = getattr(value, field)
            if name and name != getattr(saved, field):
                (settings.assets_dir() / name).unlink(missing_ok=True)
