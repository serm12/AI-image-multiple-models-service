# 订单详情无水印原图模块

扩展：`extensions/paintingify-order-images`，由 Shopify CLI 的 Admin Block 模板生成。
目标：`admin.order-details.block.render`，运行时 API 2026-07（CLI 官方模板版本）。

商家在订单详情页的应用区块中添加并固定「Paintingify 无水印原图」后，可以查看
当前订单每个 AI 商品的无水印原图、任务 ID 以及图片打开/下载入口。
模块包含加载、空订单、缺失原图和错误重试状态。

后端：`GET /shopify/orders/{numeric_order_id}/images`。
官方 SDK 验证 Admin UI ID token，并限制 Paintingify 店铺。
随后交换当前员工的 online token，直接查询当前订单（包含全部商品分页），
不依赖浏览器提交的任务 ID，也不使用后台缓存绕过员工的订单访问权限。
根据 `_AI Task ID` 或旧订单的购买成品 URL 找到 `output_cropped_original_*`。
参考照片、水印预览都不会当作无水印成品返回。优先使用已有 CDN 映射，缺失时使用原图文件地址。
接口返回 no-store，不持久化员工 access token。

这是订单页应用模块，不会替换商品图片、写入订单备注或向买家发送图片。
本次只发布订单扩展；主题扩展草稿暂存 `tmp/shopify-staging`，没有激活或发布。

验证：
```
python -m unittest discover -s tests -p "test_shopify*.py"
node_modules/.bin/tsc --project extensions/paintingify-order-images/tsconfig.json
shopify app build
```
# Customer order downloads

`extensions/paintingify-customer-originals` adds **Your original artwork** to
`customer-account.order-status.block.render`. Add **Paintingify Original Downloads**
to the live customer-account Order status page in the checkout/accounts editor.
The English copy matches the storefront. It displays item titles, original previews,
full-size links, and attachment downloads. Missing originals show a pending message.

The separate customer endpoint verifies Shopify's customer-account session token
using the official Python SDK and requires a signed-in `gid://shopify/Customer/...`
subject. It compares the subject with order ownership captured from signed Shopify
order webhooks; it never trusts a customer ID, image URL, or task ID from the browser.
Existing webhook subscriptions are upgraded to include customer IDs. Historical
orders from the accessible last 59 days are backfilled on the next verified admin
app visit. Orders outside that window require Shopify's read_all_orders permission
and a separate backfill. No names, email addresses, or addresses are stored for this.

Downloads use a separate 15-minute signed link, recheck current order ownership and
task association, and stream the local original with Content-Disposition: attachment.
Refresh images to renew an expired link. Existing task/CDN preview URLs retain their
existing access model; this change does not make that storage private.
Anonymous/pre-authenticated order viewers must sign in before downloading.
The orders/create, orders/paid, and orders/updated webhooks remain primary;
admin-session reconciliation remains a fallback only.

