# My3dFigure 专用后台维护规则

- 这里是 My3dFigure 的独立业务实现；路由前缀固定为 `/api/my3dfigure`。
- 修改本项目检测、生成、提示词、预处理、水印、任务行为时，在本目录内修改。不要修改公共 `app/routers/api.py`、`app/services/face_detection.py`、`app/clients/`、`app/utils/reference_image_utils.py` 来实现本项目需求。
- 人脸检测由专用 `services/face_detection.py` 选择一条独立管线：YuNet、Google 官方 BlazeFace Full Range 或 SCRFD-10G-KPS；一次请求只运行选中的模型，不混用结果或互作后备。修改任一模型不得改变另外两条管线。YuNet 的既有策略及结果必须保持不变。不得添加 Haar、CascadeClassifier 或用公共人脸检测器作后备。
- SCRFD-10G-KPS 按用户2026-09-30要求移除 MobileNetV2：宠物误候选仅用 SCRFD 自身的局部确认、关键点比例和源像素证据排除，不加载其他独立分类器。不能删除其他管线仍使用的共享模型文件。检测调整须保护两组原图及实际前端缩放版的数量、状态、错误码、完整提示和同脸框。
- 本目录有独立配置、生成客户端、任务管理器和图片处理。不得把它们重新导入公共业务模块，不得修改公共模块全局变量或进程环境变量。
- 专用设置使用 `MY3DFIGURE_` 前缀；不要为本项目需求修改共享 `.env` 中无前缀设置。未配置专用值时，兼容读取现有共享配置。
- `/api/my3dfigure/task-status/` 与 `/api/my3dfigure/taskfile/` 只处理 `tasks/my3dfigure/` 内本项目任务。响应中的任务和文件 URL 必须保留专用前缀。
- 当前前端以 `gpt-image-2_aiapiroute` 发起 `my3d_character` 请求；专用客户端只在该调用失败时按顺序尝试 `gpt-image-2_fal`。公共代码的 provider 名称变化不能静默改变本项目行为。
- 拉取公共上游更新时，不要整包覆盖这里；按需求单独评审、迁移更新。
- 当前未保留 My3dFigure 专用自动回归测试；不要引用或执行已清理的 `tests.test_my3d_api_isolation`、`tests.test_face_detection`、`tests.test_face_browser_uploads`。后续如需恢复自动测试，应先重新建立并验证对应测试文件。
- 本项目仍与公共接口共享服务器、Python 依赖、HTTP 连接池及部分通用工具。修改这些基础设施要评估所有项目；代码隔离不等于进程或资源隔离。
