# Shopify 应用 SDK

应用：Paintingify AI Backend（ShopUpUp 组织）。
分发方式：自定义分发（私有应用），已限定 Paintingify 单店安装。
控制台：https://dev.shopify.com/dashboard/129014438/apps/434353405953

官方仓库：https://github.com/Shopify/shopify-app-python
pip 包名：`shopifyapp`，Python 导入名：`shopify_app`。
项目已将 `shopifyapp==1.0.2` 加入 `requirements.txt`。

## 环境变量

凭据保存在后端根目录 `.env`，该文件已被 `.gitignore` 排除。
生产环境已单独配置这些变量（服务器 `.env` 和运行容器）。

- `SHOPIFY_APP_ID`：Dev Dashboard 的数字应用 ID，不用于 OAuth。
- `SHOPIFY_CLIENT_ID`：Client ID，也叫 API key。
- `SHOPIFY_CLIENT_SECRET`：Client Secret，也叫 API secret。
- `SHOPIFY_API_VERSION`：Admin GraphQL API 版本，目前配置为 `2026-10`。
- `SHOPIFY_SCOPES`：请求的权限，必须与 `shopify.app.toml` 的 `access_scopes.scopes` 同步。
- `SHOPIFY_SHOP`：安装目标店铺的 myshopify.com 子域名，当前为 `ai-generate-image`（Paintingify）。

不要把 Client Secret 放到 Shopify 主题、浏览器代码或 Git 中。

## 使用

```python
from app.services.shopify_client import get_shopify_app

shopify = get_shopify_app()
```

初始化是惰性的，不会影响现有图片服务启动，也不会自动发出网络请求。
SDK 提供请求验证、Webhook 验签、token exchange 和 Admin GraphQL 调用。
FastAPI 的异步处理函数使用 SDK 的 `_async` 网络方法。

当前已部署凭据配置和 SDK，发布权限版本并安装到 Paintingify。
尚未实现 OAuth 路由、token 持久化或取得店铺 access token。Client Secret 本身不是店铺 access token。

## 应用配置和权限

根目录 `shopify.app.toml` 关联现有应用的 Client ID，使用非嵌入式模式及
生产首页 `https://image-api.shopupup.com/shopify`。OAuth 回调尚未实现，因此 `redirect_urls` 为空。

当前 `.env`、`env.example` 和 TOML 同步声明：

- `read_orders`：读取最近 60 天订单。
- `read_customers`：读取店铺客户（不是后台员工账号）。
- `read_files`、`write_files`：读取、上传和管理 Shopify 文件。

超过 60 天的历史订单需要另行申请 `read_all_orders`；客户姓名、邮箱等字段
还受 Shopify protected customer data 访问规则约束。
当前没有订单或客户修改业务，因此不声明 `write_orders`、`write_customers`。

`.env` 是后端配置，修改它不会自动授予 Shopify 权限，SDK 初始化也不会申请权限。
TOML 中的权限需要通过 `shopify app deploy` 发布（或在 Dev Dashboard 发布对应版本），
然后安装应用并完成店铺授权，才会体现在 access token 中。
权限版本 `paintingify-python-20261012` 已发布，且已通过 Paintingify 安装授权。

参考：https://shopify.dev/docs/apps/build/cli-for-apps/app-configuration
和 https://shopify.dev/docs/api/usage/access-scopes 。

应用与店铺在同一 Shopify 组织且应用已安装时，可使用
`exchange_using_client_credentials_async(shop=...)`。其他组织的店铺需要相应的
OAuth/token exchange 授权流程；不能仅凭 App ID/Secret 访问店铺。

后续拿到有效 token 后，可调用
`shopify.admin_graphql_request_async(query, shop=shop, access_token=token, api_version=...)`。
调用方应检查 SDK 结果的 `ok`，并按官方 README 处理错误和 token 生命周期。

## 本地验证环境

Windows 验证环境位于 `.venv`（已忽略）：

```powershell
.\.venv\Scripts\python.exe -c "from app.services.shopify_client import get_shopify_app; get_shopify_app(); print('Shopify SDK ready')"
```

该环境只安装了 SDK、FastAPI、Uvicorn 及验证所需依赖；运行整个后端前需安装完整的 `requirements.txt`。

## Python 应用基础文件

Shopify 当前没有官方 Python App 模板。本项目依据官方 `shopify-app-python`
SDK 接入现有 FastAPI 服务，不是从官方 Python 模板生成。

| 文件 | 作用 |
| --- | --- |
| `shopify.app.toml` | 应用身份、权限和 API 版本 |
| `shopify.web.toml` | 将 Shopify CLI 开发进程连接到 Python 服务 |
| `package.json` | Shopify CLI 命令入口，使用已安装的全局 CLI |
| `scripts/shopify_dev.py` | 启动现有 `app.main:app`，读取 CLI 分配的端口 |
| `app/services/shopify_client.py` | SDK 初始化，支持 CLI 注入的 `SHOPIFY_API_KEY/API_SECRET` |
| `app/services/shopify_http.py` | 原始请求和 SDK 响应到 FastAPI 的适配 |
| `app/routers/shopify.py` | Shopify 会话验证、任务页跳转及 Webhook 验签入口 |
| `app/services/shopify_admin.py` | 15 分钟嵌入式登录会话及 App Bridge 接入 |
| `tests/test_shopify_integration.py` | Webhook 签名、篡改、重复请求等验证 |

路由已经挂载到现有 `app.main:app`。Webhook 目前只接受 `app/uninstalled`，
没有持久化店铺 access token；嵌入登录 cookie 最长有效 15 分钟，卸载后不再刷新。
后续引入服务端会话存储时必须补上即时撤销。
未实现的事件返回 404，未配置自动订阅，也没有伪造订单同步或隐私删除成功响应。

本地启动（先安装完整 Python 依赖）：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
npm run dev:python
```

首页为 `http://127.0.0.1:8001/shopify`（如外部环境配置 PORT 则使用该端口）。
Shopify CLI 开发入口为 `npm run dev`，需要先激活 Python 虚拟环境。
CLI 会使用登录会话、选择开发店铺并建立开发连接。
当前 TOML 使用生产应用首页，且关闭自动 URL 改写；
通过 tunnel 测试应用首页时，需将开发配置的 application_url 指向实际 tunnel 的 `/shopify`。
生产应用 URL 为 `https://image-api.shopupup.com/shopify`。

```powershell
npm run config:validate
python -m unittest discover -s tests -p test_shopify_integration.py -v
```

这是一套接入基础文件，尚不包含店铺安装 OAuth 回调、token 持久化和订单/客户业务页面。
现有自定义分发应用已安装到 Paintingify；实际调用店铺 API 前仍需完成适用的 token 获取流程。

## 2026-10-12 部署记录

嵌入模式：`embedded = true`。入口 `/shopify` 使用官方 SDK 验证 ID token，
检查 audience、签名、过期时间、issuer、destination 和 Paintingify 店铺白名单，
通过后进入 `/admin/tasks`。缺少/过期 token 时使用 SDK 的 App Bridge 恢复页面。
登录 cookie 使用 Secure、HttpOnly、SameSite=None 和 Partitioned，15 分钟有效，
页面每 5 分钟用新 Shopify ID token 刷新；后台写操作校验 Origin。
外部直接访问后台继续使用原有密码登录。iframe 支持依赖浏览器允许分区 cookie。

- 后端代码提交：`636e5e2cf6eb9d0fe4302975eef6c39f290ca8a3`。
- GitHub Actions 镜像构建成功，生产 `ai-image-api` 容器运行该提交。
- `/health` 与 `/shopify` 均返回 200，Docker healthy，Nginx 配置校验通过。
- 官方 SDK `shopifyapp==1.0.2` 在生产环境初始化成功。
- 已安装店铺：Paintingify，永久域名 `ai-generate-image.myshopify.com`。
- 回退容器保留；本地其他未提交改动没有纳入本次发布。
- 服务器已有每 10 分钟自动更新任务，脚本位于
  `/www/server/panel/scripts/custom/ai-image-auto-update.sh`。
  手动部署须与它共用 `/var/lock/ai-image-api-auto-update.lock`，
  避免同时重建容器。自动更新从 `/www/wwwroot/ai-image-api/.env` 加载凭据。
