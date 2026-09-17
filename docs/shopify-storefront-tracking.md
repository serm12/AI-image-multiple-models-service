# Shopify storefront tracking

The theme records generated-image identity, the logged-in customer's email, successful cart additions, and checkout intent automatically. Guest tasks continue to use the anonymous storefront visitor ID until Shopify supplies an email during checkout.

`added_to_cart` and `checkout_intent` are theme events. `checkout_started` and `checkout_completed` are Shopify checkout events and will remain inactive until the custom pixel is connected.

To record confirmed Shopify checkout events, open **Shopify admin → Settings → Customer events**, add a custom pixel, paste the contents of `docs/shopify-custom-pixel.js`, connect it, and verify that the pixel is allowed by the store's customer privacy settings. The pixel requires Checkout Extensibility for line-item properties and customer-email access for `checkout.email`.

The pixel reads the private `_AI Task ID` and `_AI Tracking Token` line-item properties and, when Shopify makes it available, the checkout email. It does not send customer name, address, phone, or payment information. The API stores only a hash of the cart token and never persists the raw tracking token.
