# Preview customer signup

The existing theme explicitly tells customers that entering an email saves the preview and subscribes to artwork updates. The backend now processes its authenticated `newsletter_result` event immediately through a durable SQLite queue. An event must match the task's recorded email and dedicated newsletter source; a plain email on any other event is not signup consent.

The worker looks up the exact email, creates a customer if needed, and records SINGLE_OPT_IN email marketing consent using the server-recorded submission time. Existing subscribed customers are reused. UNSUBSCRIBED, PENDING, INVALID and REDACTED states are preserved. No email verification or customer login is fabricated. Shopify API errors retry with bounded exponential backoff; the task table reports the backend result separately from the front-end submission.

On the first verified admin session with `write_customers`, Shopify's official SDK exchanges an offline token. It is encrypted with a key derived from the app secret and stored in the private `.shopify-customer-sync.sqlite3` file in the persistent tasks volume. Tokens never reach the theme. Uninstall deletes the token. Secret rotation requires a fresh authorization. Recorded historical newsletter attempts are imported once, with the same validation and opt-out protection.

Authorization: release the app configuration with `write_customers`, approve the scope update in the store if Shopify requests it, and open the app. New tasks do not depend on an administrator keeping the browser open. Queued work survives process restarts. Transient failures retry up to 12 attempts; final failures remain visible for diagnosis.

References: https://shopify.dev/docs/api/admin-graphql/2026-10/mutations/customerCreate and https://shopify.dev/docs/api/admin-graphql/2026-10/mutations/customerEmailMarketingConsentUpdate
