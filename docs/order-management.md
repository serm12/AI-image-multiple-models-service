# Order management

`/admin/orders` lists live Shopify orders, previews purchased original artwork,
and submits selected physical, unfulfilled line items to Printful drafts.
The backend requires its existing Shopify offline authorization and `read_orders`
plus access to protected shipping-address data. Shopify's normal order-history
scope limits still apply; this page does not grant access to older orders.

In **Settings → Production services**, save a Printful Private Token and optionally
a Store ID (required for account-level tokens). Credentials are encrypted with
the existing provider credential store and never sent back to the browser.
Environment defaults: `PRINTFUL_TOKEN`, `PRINTFUL_STORE_ID`.
Printful also supports public OAuth apps; this implementation uses Private Tokens.
Use **Test saved connection** after saving; grant store/catalog read and order
read/write permissions in Printful.

Select items and enter their actual Printful catalog Variant IDs. IDs are never
inferred from a title. The current connector is Printful; other production services
need their own API adapters and credentials before they can be enabled.

- AuraSR 4× is optional and uses the existing FAL key (billable image processing).
- White bleed is optional; supported Canvas variants are 5, 6, 7 and 19316, using
  `printful-mcp/image-processing.js` templates. Other variants cannot enable bleed.
- The artwork is contained without stretching/cropping, centered on a white canvas.
- Disabling both options sends the purchased original image directly.
- Enabling either option requires FAL for hosting the final print file.
- Quantities and shipping addresses are fetched again before submission.
- The Shopify numeric order ID is used as Printful's `external_id`. An existing
  draft is updated; any other existing status prevents submission.
- Updating a draft replaces all its items with the selected items shown on screen.
- The code never confirms production or charges the Printful order. Review files,
  costs and shipping in Printful before manually confirming production.

Jobs and manifests are saved to `TASKS_DIR/.production.sqlite3`. Processing runs
in a server background task; the page polls its persistent status. If the server
restarts mid-job, the job is not automatically resubmitted. After an hour, **Check
Printful status** reconciles the remote order and unlocks an interrupted job for
retry. A draft may exist even if image/file polling subsequently fails.
Back up the production database and provider credential database with task data.

Printful file processing is polled briefly, and pending/failed statuses are shown.
Use **Check Printful status** to fetch the latest costs and file status.
No live orders or paid image jobs are created by the automated tests.

Validation: `python -m unittest tests.test_order_management tests.test_provider_credentials tests.test_shopify_orders tests.test_admin_frontend`
and `npm --prefix frontend run build`.

References: https://developers.printful.com/docs/ and
https://shopify.dev/docs/api/admin-graphql/unstable/queries/order

Order and line types use purchased DIGITAL/CANVAS formats, then Shopify requiresShipping as fallback. Lists fetch all line pages before classification. Current quantities exclude refunded/removed items; fulfillment alone never changes the type. Mixed orders only allow physical lines to be sent. Explicit digital variants are blocked even with shipping enabled; physical variants with shipping disabled require review.


## Existing Printful orders

Opening the order list or detail loads all Printful order pages in the configured store, cached for at most 60 seconds. Manual status sync bypasses the cache. This is read-through synchronization, not a webhook or a background Shopify fulfillment update.

Matching accepts the Shopify numeric ID, exact display number, MCP `shopify-{number}-...` references, and unambiguous `Shopify #{number}` item names. All matching replacement/reprint/canceled records remain visible with Printful status and shipment tracking. A fresh complete scan runs before any image processing or draft submission; unavailable Printful blocks submission. Multiple matches or any non-draft match require review and cannot be overwritten by this flow. A single existing draft retains its external ID; new drafts use the MCP-style reference.
