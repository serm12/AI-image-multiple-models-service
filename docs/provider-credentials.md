# Editable image generation credentials

Admin settings supports AIAPIROUTE_API_KEY and FAL_API_KEY. GET returns only
configured booleans; POST never returns input or validation details containing
keys. Empty fields preserve existing credentials. Existing environment variables
remain the fallback until a replacement is saved.

Overrides are Fernet-encrypted in tasks/.provider-credentials.sqlite3. Encryption
uses a purpose-separated key derived from SHOPIFY_API_SECRET (or
SHOPIFY_CLIENT_SECRET, then ADMIN_SESSION_SECRET). Preserve the chosen secret
with backups; rotating it requires re-entering the provider keys. The hidden
file cannot be served through taskfile routes. Admin authentication and same-origin
checks protect updates. No public API returns these settings.

APIConfig resolves overrides when used, and the unified client refreshes cached
provider instances when credentials change. FAL uses an explicit SyncClient per
provider instance, avoiding global environment credentials and stale SDK caches.
Requests already holding an instance keep that instance's credentials. Tests use
fake credentials and mocked providers, with no billable generation calls.

Image filename algorithm settings are separate: tasks/.algorithm-settings.json
stores the numeric factor, defaulting to ALGORITHM_FACTOR from the environment.
It changes new filenames only; no existing files or links are rewritten.
