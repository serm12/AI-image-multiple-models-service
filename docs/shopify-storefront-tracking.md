# Shopify storefront tracking

The theme records generated-image identity, successful cart additions, and checkout intent automatically.

To record confirmed Shopify checkout events, open **Shopify admin → Settings → Customer events**, add a custom pixel, paste the contents of `docs/shopify-custom-pixel.js`, connect it, and verify that the pixel is allowed by the store's customer privacy settings.

The pixel reads only the private `_AI Task ID` and `_AI Tracking Token` line-item properties. It does not send customer email, name, address, or payment information. The API stores only a hash of the cart token and never persists the raw tracking token.
