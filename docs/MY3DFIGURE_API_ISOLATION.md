# My3dFigure 专用接口与公共接口隔离

## 接口

| 用途 | My3dFigure 前端专用接口 | 公共接口继续保留 |
|---|---|---|
| 图片检测 | `POST /api/my3dfigure/check-photo` | `POST /api/check-photo`、`POST /check-photo/` |
| 图片生成 | `POST /api/my3dfigure/generate-async/` | `POST /generate-async/` |
| 任务轮询 | `GET /api/my3dfigure/task-status/{task_id}` | `GET /task-status/{task_id}` |
| 图片访问 | `GET /api/my3dfigure/taskfile/{task_id}/{filename}` | `GET /taskfile/{task_id}/{filename}` |

前端所有当前调用已经迁移。生成请求返回的 `status_url`、检测返回的 `user_photo_url`、任务返回的图片地址也使用专用前缀。原公共端点不重定向到专用端点。

## 实现范围

`app/projects/my3dfigure/` 保存独立路由、配置、YuNet 检测、精确错误反馈、提示词、生成客户端和异步任务管理器。任务文件存储在 `tasks/my3dfigure/`，复用已有 tasks 持久化挂载。公共代码不导入本项目业务实现；`app/main.py` 仅注册新路由，专用路由自身负责清理过期任务记录。

My3dFigure 检测不包含 Haar 实现，也不调用公共检测。公共人脸检测恢复 Git 原有实现；其历史规则不是本次重新设计的规则。

本次先以本地下载基准 `69385a8` 恢复公共路由、检测、预处理、生成客户端、水印图片和公共预处理测试，再执行 `git pull --ff-only origin master`，更新到 `f5b3869`（2.1.12）。公共代码保留 GitHub 最新 provider、限流和水印更新，本项目行为保持原样。没有用本地历史实现覆盖上游更新。

## 设置与以后修改

本项目配置来自 `app/projects/my3dfigure/core/config.py`。每个环境设置优先读取 `MY3DFIGURE_` 加原设置名，例如：

```
MY3DFIGURE_AIAPIROUTE_API_KEY=...
MY3DFIGURE_AIAPIROUTE_BASE_URL=...
MY3DFIGURE_AIAPIROUTE_IMAGE_RESOLUTION=1K
MY3DFIGURE_AIAPIROUTE_TIMEOUT_SECONDS=300
MY3DFIGURE_MAX_CONCURRENT_TASKS=2
MY3DFIGURE_FACE_DETECTION_YUNET_MODEL=...
```

没有专用值时兼容读取当前共享环境设置。设置本项目参数应新增专用前缀值，避免改变公共参数。配置加载不写进程环境变量，Replicate/Fal 客户端使用各自实例传入密钥。

My3dFigure 使用的本地模型位于 `models/`，Dockerfile 包含模型复制步骤。参考图编辑请求把浏览器已核验的上传原始字节直接以 `multipart/form-data` 提交给专用 `aiapiroute_gpt-image-2` 客户端，不再进行参考图比例预处理，也没有 `MY3DFIGURE_AIAPIROUTE_GPT_IMAGE2_REFERENCE_RATIO` 设置。后续本项目规则、提示词、检测和生成修改，均在专用目录实施。维护约束见该目录 `AGENTS.md`。

## 验证与边界

历史隔离测试曾覆盖公共/专用检测分流、旧错误响应格式、独立任务及客户端、原公共请求不启用 My3dFigure 提示词，以及上传→模拟生成→轮询→下载完整链路。真实图片检测曾使用浏览器导出的单人侧脸、街头单人和双人照片做回归；生成验证阻断外部 HTTP，未发起收费生成请求。相关未跟踪测试已于 2026-09-18 按用户要求清理，后续如需自动验证须重新建立测试。

迁移前原提示词已经超出当时部分旧测试的 7000/4400 字符上限：默认普通提示词 7267 字符、近照提示词 4728 字符。已与迁移前 ZIP 比对，提示词内容完全一致；此次没有为通过长度断言删减生成约束。相关的 `tests/test_direct_character_prompt_service.py` 已于 2026-09-18 按用户要求清理，因此该历史长度断言不再可执行。

这次隔离的是接口和项目业务实现，仍共用服务器进程、资源、第三方账户（未配置专用密钥时）、Python 依赖、HTTP 连接池和部分通用工具。它不是租户身份认证，也不是独立部署。以后若要求一个项目的重启、依赖升级、资源耗尽也不能影响另一个项目，需要进一步分成独立容器/服务。

## 备份

迁移前源码、测试、模型、水印和 Git 工作区/暂存区差异保存在仓库上一级的 `my3dfigure-before-isolation-20260908_102620.zip`。Git 中保留名为 `my3dfigure-isolation-before-upstream-sync-20260908` 的 stash 快照。已恢复工作区，未提交、未推送、未部署。
