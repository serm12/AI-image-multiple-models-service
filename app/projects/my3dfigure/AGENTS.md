# My3dFigure 专用后台维护规则

- 这里是 My3dFigure 的独立业务实现；路由前缀固定为 `/api/my3dfigure`。
- 修改本项目检测、生成、提示词、预处理、水印、任务行为时，在本目录内修改。不要修改公共 `app/routers/api.py`、`app/services/face_detection.py`、`app/clients/`、`app/utils/reference_image_utils.py` 来实现本项目需求。
- 人脸检测由专用 `services/face_detection/__init__.py` 选择一条独立管线：YuNet、Google 官方 BlazeFace Full Range 、SCRFD-10G-KPS 或 SCRFD-2.5G-KPS；一次请求只运行选中的模型，不混用结果或互作后备。修改任一模型不得改变其他管线。YuNet 的既有策略及结果必须保持不变。不得添加 Haar、CascadeClassifier 或用公共人脸检测器作后备。
- SCRFD-10G-KPS 按用户2026-09-30要求移除 MobileNetV2：宠物误候选仅用 SCRFD 自身的局部确认、关键点比例和源像素证据排除，不加载其他独立分类器。不能删除其他管线仍使用的共享模型文件。检测调整须保护两组原图及实际前端缩放版的数量、状态、错误码、完整提示和同脸框。
- SCRFD-2.5G-KPS 的12个处理模块完整复制接入时的 SCRFD-10G-KPS，但两者目录、权重、模型实例、请求缓存及处理实现完全独立。不得互相导入或抽取共享可修改的检测/质量/复核/定位逻辑；调整任一模型不得顺带改另一模型。使用专用入口的单行注释切换，一次请求只运行所选管线。
- 本目录有独立配置、生成客户端、任务管理器和图片处理。不得把它们重新导入公共业务模块，不得修改公共模块全局变量或进程环境变量。
- 专用设置使用 `MY3DFIGURE_` 前缀；不要为本项目需求修改共享 `.env` 中无前缀设置。未配置专用值时，兼容读取现有共享配置。
- `/api/my3dfigure/task-status/` 与 `/api/my3dfigure/taskfile/` 只处理 `tasks/my3dfigure/` 内本项目任务。响应中的任务和文件 URL 必须保留专用前缀。
- 2026-10-10 用户要求关闭付费 Fal 生图并保留代码。专用配置默认 `APIConfig.FAL_GPT_IMAGE2_ENABLED = False`；保留该默认行，只切换下一行 `# FAL_GPT_IMAGE2_ENABLED = True` 的注释即可开启/关闭，当前保持注释关闭。开启时恢复原有 Fal 兜底；关闭时前端继续以 `gpt-image-2_aiapiroute` 发起 `my3d_character` 请求，失败直接结束，不再尝试 Fal。显式 Fal 请求和专用 Fal 客户端调用必须在上传/提交前拒绝；Fal 实现、参数与凭据保留。公共代码的 provider 名称变化不能静默改变本项目行为。
- 拉取公共上游更新时，不要整包覆盖这里；按需求单独评审、迁移更新。
- 当前未保留 My3dFigure 专用自动回归测试；不要引用或执行已清理的 `tests.test_my3d_api_isolation`、`tests.test_face_detection`、`tests.test_face_browser_uploads`。后续如需恢复自动测试，应先重新建立并验证对应测试文件。
- 本项目仍与公共接口共享服务器、Python 依赖、HTTP 连接池及部分通用工具。修改这些基础设施要评估所有项目；代码隔离不等于进程或资源隔离。

- 2026-10-12 用户授权统一生图前旋转策略：SCRFD-2.5G与10G模型文件夹各自保留独立 source_orientation.py 输出可靠脸角证据，不选择最终整图角度、不互相导入。像素 rotate() 与选角/脸框/PNG流程统一放 services/generation/checked_source_rotation.py；通用较大/较小选角及小角度阈值仅在专用 services/generation/checked_source_rotation.py 的 SOURCE_ROTATION_SELECTION、SOURCE_ROTATION_MIN_DEGREES 和 select_source_rotation 维护。当前 maximum_absolute。此共享只限质量验证完成后的生图旋转决策，不共享检测/质量/定位/关键点复核，不跨模型导入或兜底；新增支持模型应复用同一策略入口。

- 2026-10-12 提示词维护强制检查：每次修改提示词都须核对实际发送的完整prompt字符数，包含身份锚点/坐标归属、动态参数/随机接触分支及末尾审核，不只测构建器正文。当前中转接口上限32000字符，预算目标不超过30000以留余量；单人/双人所有可达组合及实际路由拼装须覆盖，超限继续去重精简，不截断尾部规则或宣称等待可修复。记录最大长度/余量及验证证据；静态或mock检查不等于真实成图验收。

- 2026-10-12 最新用户要求角度分析恢复为两个模型目录各一份，覆盖此前合并授权。不得共用或互相导入角度实现；只共用 checked_source_rotation.py 中选角策略、3°阈值、rotate()及后验证保存流程，不混用检测器或角度结果。

- 2026-10-13 宠物检测双管线：services/pet_detection/__init__.py 通过注释一行、打开另一行选择原 YOLOX＋RTMPose（yolox_rtmpose/）或 Pets-Face-Recognition v1.3 / Keypoint R-CNN（keypoint_rcnn/）。按用户最新要求当前启用旧 YOLOX＋RTMPose，新 Keypoint R-CNN 保持注释关闭，效果测试通过后再启用。原 pet_detection.py、pet_face_visibility.py 完整保留；两套不互相导入检测/质量实现，不互作失败后备。新方案仅复用不可变本地 CLIP 模型资产辅助分类/可见性，不从旧检测器获取框或关键点。新模型取消 box_detections_per_img=1 的最终截断，但保留正常候选提议预算/NMS；逐只质量检查后统计可用宠物，再应用单宠物业务目标。模型错误不得伪报0只。专用依赖、版本、权重来源与限制见新管线 README.md。

- 2026-10-13 人脸检测目录统一：四套模型及各自处理/角度模块位于 services/face_detection/{yunet,blazeface_full,scrfd_10g,scrfd_2_5g}/。选择入口为同目录 __init__.py，公开 services.face_detection 导入保持，默认SCRFD-2.5G；仍不得跨模型共享检测实现。生成流程已按最新要求移入 services/generation/。

- 2026-10-13 宠物目录与人脸一致：services/pet_detection/ 根目录只保留 __init__.py 选择入口，两套实现分别放 yolox_rtmpose/ 和 keypoint_rcnn/。旧 pet_detection.py、pet_face_visibility.py 已移到 yolox_rtmpose/，默认使用旧 YOLOX＋RTMPose，新 Keypoint R-CNN 暂不启用；注释一行/打开另一行切换。迁移只调整模型路径层级，禁止混用两套检测结果。

- 2026-10-13 生图服务统一：services/generation/ 管理提示词、旋转、身份参考图、错误处理、异步任务、任务文件/存储与放大服务。services/face_detection/ 和 pet_detection/ 保持独立；模型自身角度分析仍留模型目录。移动仅更新导入，禁止改变业务行为。
