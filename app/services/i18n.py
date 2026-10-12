"""Server-rendered UI translations. Only extracted static copy is translated."""
import json
import os
import re
import uuid
from contextvars import ContextVar
from functools import lru_cache
from html import escape
from pathlib import Path
from app.core.config import DirectoryConfig

SUPPORTED = {'zh-CN': '简体中文', 'en': 'English'}
current_locale = ContextVar('ui_locale', default=None)
LOCALES = Path(__file__).resolve().parents[1] / 'locales'


def preference_path():
    return Path(DirectoryConfig.TASKS_DIR) / '.language-settings.json'


def selected_locale():
    try:
        value = json.loads(preference_path().read_text('utf-8')).get('locale')
        return value if value in SUPPORTED else 'zh-CN'
    except (ValueError, OSError):
        return 'zh-CN'


def save_locale(locale):
    if locale not in SUPPORTED:
        raise ValueError('Unsupported locale')
    path = preference_path(); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps({'locale': locale}), 'utf-8')
    os.replace(temporary, path)


@lru_cache(maxsize=4)
def catalog(locale):
    return json.loads((LOCALES / (locale + '.json')).read_text('utf-8'))


def t(key, **values):
    locale = current_locale.get() or selected_locale()
    message = catalog(locale).get(key, catalog('zh-CN').get(key, key))
    return message.format_map(values) if values else message


@lru_cache(maxsize=1)
def templates():
    return json.loads((LOCALES / 'ui-templates.json').read_text('utf-8'))


def render(key):
    parts = []
    for part in templates()[key]:
        if isinstance(part, str):
            parts.append(part)
        else:
            text = t(part['key'])
            if part['context'] == 'html':
                text = escape(text, quote=True)
            elif part['context'] == 'js':
                text = json.dumps(text, ensure_ascii=True)[1:-1].replace("'", '\\u0027').replace('<', '\\u003c')
            parts.append(text)
    return ''.join(parts).replace('lang="zh-CN"', 'lang="' + (current_locale.get() or selected_locale()) + '"')


def render_page(source):
    def replace(match):
        value = t(match[1])
        if match[2] == 'js':
            return json.dumps(value, ensure_ascii=True)[1:-1].replace("'", '\\u0027').replace('<', '\\u003c')
        return escape(value, quote=True)
    return re.sub(r'\{\{i18n:([a-z0-9_]+):(html|js)\}\}', replace, source).replace(
        'lang="zh-CN"', 'lang="' + (current_locale.get() or selected_locale()) + '"')


class Message:
    """Delay module-level option labels until a request selects its locale."""
    def __init__(self, key):
        self.key = key

    def __str__(self):
        return render(self.key)

    def __format__(self, spec):
        return format(str(self), spec)
