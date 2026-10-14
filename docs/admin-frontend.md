# React / shadcn 管理后台

管理界面位于 `frontend/`，使用 React、TypeScript、Vite、Tailwind CSS 和官方 shadcn/ui 组件。Python 仍负责鉴权、图片生成、任务查询和设置持久化。

## 本地运行

需要 Node.js 22（或兼容 Vite 7 的 Node.js 版本）和项目原有 Python 环境。

```powershell
npm --prefix frontend ci
npm run admin:build
python -m app.run
```

访问后端地址的 `/admin/login`，使用原有管理员账号登录。入口继续使用 `/admin/tasks`、`/admin/styles` 和 `/admin/config`，原有书签与 Shopify iframe 路径保持兼容。HTTP 本地开发请使用 `localhost`，登录 cookie 保持 `Secure`；线上使用 HTTPS。

构建产物输出到 `app/static/admin/`，不提交 Git。未构建时管理页面返回明确的 503 提示，图片生成 API 不受影响。

## 前端开发

先启动 Python 后端，再运行：

```powershell
# 按后端实际端口设置；默认 8001
$env:ADMIN_BACKEND_URL = 'http://127.0.0.1:8001'
npm run admin:dev
```

打开 Vite 输出地址下的 `/admin/assets/`。开发代理把登录、数据、缩略图、设置和原图请求转发给 Python，并由 Vite 提供页面。生产构建直接由 FastAPI 托管，无需独立 Node.js 运行服务。

## 页面和接口

- 登录：沿用签名 session cookie、记住登录、退出登录。
- 任务：服务端分页和筛选、收藏、图片预览、参考图、提示词、完整详情、复制、客户和订单链接、转化状态、多时区时间。
- 风格目录：风格与提示词搜索、复制。
- 设置：水印预览和上传、品牌标识、时区、供应商密钥、文件命名算法、中英文。
- 新增 `/admin/api/session`、`/admin/api/tasks`、`/admin/api/styles` 和显式状态写入 `/admin/api/tasks/{task_id}/star`。全部复用原管理员鉴权，收藏写入校验 Origin。
- 设置复用现有受保护接口。原有服务端 HTML 辅助函数暂时保留供兼容测试使用；生产页面路由优先使用 React 入口。

## 构建和部署

Dockerfile 已增加独立 Node.js 构建阶段；最终 Python 镜像只接收静态产物。使用现有 Docker 构建流程即可。非 Docker 部署需要先执行 `npm --prefix frontend ci` 和 `npm run admin:build`，再重启后端。

本次迁移没有执行线上部署或提交仓库已有改动。

## 验证

```powershell
npm run admin:build
.venv/Scripts/python.exe -m unittest tests.test_admin_frontend tests.test_admin_tasks tests.test_task_list_index tests.test_watermark_settings tests.test_algorithm_settings tests.test_time_settings tests.test_i18n tests.test_provider_credentials tests.test_shopify_integration -q
```

浏览器验证使用临时任务目录和测试管理员账号，不写入真实任务或配置。已覆盖登录、收藏、筛选、分页、图片弹窗、详情抽屉、水印预览和保存、算法保存、中英文切换，以及 390px 移动布局。
