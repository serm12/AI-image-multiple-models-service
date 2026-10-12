# Watermark settings

`/admin/config` now displays editable watermark controls instead of environment
variables. The Shopify Settings navigation continues to use that URL.

- Source watermark artwork and bottom-right brand logo accept PNG/JPEG/WebP uploads
  up to 5 MB and 16 million pixels. Uploads are decoded and re-encoded as PNG with
  generated names; filenames and paths from the browser are never trusted.
- Layouts: uniform diagonal, regular grid, randomized tiles, centered watermark.
- Opacity, tiled size, angle, spacing, corner text/position and brand logo controls.
- Preview runs the same renderer on a built-in illustration without saving settings.
- Saving takes effect for future renders without a restart. Existing images and
  watermark-free originals are not rewritten.

Settings live in `tasks/.watermark-settings.json` and immutable uploaded assets in
`tasks/.watermark-assets/`, both on the existing persistent tasks volume. Atomic
replacement avoids partially written JSON. Each render reads an isolated settings
snapshot. Before the first save, existing environment/default rendering is preserved.
These endpoints require admin authentication; writes additionally require same-origin
requests. Asset and configuration responses are not cached.
