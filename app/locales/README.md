# Admin UI translations

`zh-CN.json` and `en.json` contain the Chinese and English interface text.
Settings → Language settings selects a shared admin language, defaulting to
Simplified Chinese. The choice is saved atomically in
`tasks/.language-settings.json` on the persistent volume.

`ui-templates.json` stores extracted static fragments from the existing Python
HTML templates. Its references specify HTML, JavaScript-string or plain-text
escaping. Business data is interpolated separately and is never translated.
The standalone settings template uses `{{i18n:key:html}}` and
`{{i18n:key:js}}` placeholders. Do not insert translations into raw JavaScript.

For new Python messages, add matching semantic keys to both catalogs and use
`app.services.i18n.t(key, **values)`. Escape translated strings at the output
boundary, just like other text. Module-level labels must resolve at request
time; `Message` provides lazy resolution for the existing extracted fragments.
Keep keys in both catalogs identical; Chinese is the fallback for missing
translations. Run `python -m unittest tests.test_i18n` after updates.

Coverage: backend admin login, navigation, task history, filters, style-preset
interface, language/time/watermark settings and their UI notifications.
Product names, prompts, customer data, watermark text and API diagnostic
details retain their original values. Shopify order extension translations
are managed separately from this backend interface.
