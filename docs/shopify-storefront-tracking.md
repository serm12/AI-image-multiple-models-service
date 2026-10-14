# Paintingify conversion tracking

The app now includes `extensions/paintingify-conversions`, a Shopify Web Pixel
Extension running in the strict sandbox. It subscribes only to `checkout_started`
and `checkout_completed`; theme tracking still owns `added_to_cart`,
`checkout_intent`, and newsletter submission. Order webhooks remain the source of
linked orders and payment status. A completed browser checkout is not proof of payment.

## Deployment and activation

1. Deploy the backend (including cross-pixel deduplication).
2. Run `shopify app deploy --allow-updates` to release the pixel and the added
   `write_pixels,read_customer_events` scopes on the existing app.
3. Open Paintingify AI Backend in the installed shop and accept Shopify's new
   permissions if prompted. A verified app session reconciles the app's pixel
   via `webPixelCreate`/`webPixelUpdate`. It retries at most once every four minutes;
   activation errors never prevent the task dashboard from opening.
4. In Settings > Customer events, check the app pixel is connected. Test an AI
   product checkout with analytics consent. Confirm the matching task shows the
   checkout stage. A real payment is unnecessary to test `checkout_started`.
5. Disconnect only the old Paintingify custom pixel after confirming the app pixel.
   Leave unrelated Meta/Google pixels alone. Keeping the old pixel temporarily is
   supported: event ID and per-task checkout/stage deduplication prevent duplicates.

The extension settings contain only the installed `shopDomain`. The collection
endpoint is fixed to `https://image-api.shopupup.com/storefront-events`; no API key,
Admin token, or client secret is embedded. Requests use `text/plain`, omit cookies,
and authenticate each task with its existing `_AI Tracking Token`. The server
persists only the tracking-token hash, not the raw tracking credential.

## Event behavior and privacy

- Task association uses `_AI Task ID` and `_AI Tracking Token` line properties.
  Duplicate checkout lines for a task generate a single delivery.
- Missing/redacted line properties skip that item; missing checkout email is
  supported. Only fields exposed by Shopify are sent. Names, addresses, phone
  numbers, payment details, and whole checkout objects are never collected.
- Analytics consent is required; marketing/preferences are not requested and data
  sale is disabled. Shopify controls execution and consent replay. No DOM scraping,
  cookies, storage identifiers, or production console logging are added.
- Network errors, 429 and 5xx retry up to three attempts with the same event ID;
  permanent 4xx do not retry. At most three tasks send concurrently. Replay within
  a pixel instance is also deduplicated. Delivery remains best-effort on page close
  or blocking; server order webhooks remain independent of the browser pixel.
- Existing theme collection is unchanged by this migration; its behavior is not
  replaced by the extension's consent declaration.

## Rollback and diagnostics

The old custom pixel source remains in `docs/shopify-custom-pixel.js`. Reconnect
that pixel if needed and deactivate the app pixel in Shopify's Customer events
settings. Existing task events and order links are not deleted.

The backend logs `Shopify app pixel: active` or `authorization_required` on verified
sessions. It logs only exception types on activation failure, never credentials.
The existing `/storefront-events` logs indicate recorded/deduplicated/rejected
requests. Never log the tracking token or checkout payload while troubleshooting.

References:
- https://shopify.dev/docs/apps/build/marketing/build-web-pixels
- https://shopify.dev/docs/api/web-pixels-api/standard-events/checkout_started
- https://shopify.dev/docs/api/admin-graphql/2026-10/mutations/webPixelCreate
