# Cloudflare R2 image delivery

Generated watermarked previews are returned from the local `/taskfile/` route first.
The API then uploads those previews and their generated originals to Cloudflare R2
in a background task. After the
local URL has been returned at least once and the upload has completed, later
`/task-status/{task_id}` responses use the CDN URL.

Set these variables in the production `.env` file:

```env
CLOUDFLARE_R2_ENABLED=true
CLOUDFLARE_R2_ACCOUNT_ID=
CLOUDFLARE_R2_ACCESS_KEY_ID=
CLOUDFLARE_R2_SECRET_ACCESS_KEY=
CLOUDFLARE_R2_BUCKET=
CLOUDFLARE_R2_PUBLIC_BASE_URL=https://images.example.com
CLOUDFLARE_R2_PREFIX=ai-image-tasks
```

`CLOUDFLARE_R2_PUBLIC_BASE_URL` should be the public custom domain attached to the
bucket. If any required value is missing, or an upload fails, local delivery remains
active and image generation is not failed.

Watermarked previews and generated files named `output_cropped_original_*` are
uploaded by this flow. Customer source uploads remain local and are not exposed
through the public R2 domain. Storefront task-status responses continue to return
only watermarked previews; generated originals are available for order and email links.
