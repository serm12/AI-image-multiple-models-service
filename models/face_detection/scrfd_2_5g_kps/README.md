# SCRFD-2.5G-KPS 模型

- 官方模型包：InsightFace `buffalo_m`，检测权重 `det_2.5g.onnx`，包含五个人脸关键点。
- 来源：https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip
- 官方模型列表：https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md
- 检测权重 SHA-256：`041f73f47371333d1d17a6fee6c8ab4e6aecabefe398ff32cca4e2d5eaee0af9`。
- 官方模型包 SHA-256：`d98264bd8f2dc75cbc2ddce2a14e636e02bb857b3051c234b737bf3b614edca9`。
- 运行库：与现有 10G 管线相同的 InsightFace 2.0、CPU ONNX Runtime。

独立处理管线位于 `app/projects/my3dfigure/services/scrfd_2_5g/`，完整复制
2026-10-01 已验证的 SCRFD-10G-KPS 处理逻辑与阈值，仅替换模型身份、路径及权重校验。
两个目录不互相导入，不共用可修改的质量、复核、定位或缓存实现；以后可分别调整。
不同权重的实际检测结果可能不同，复制相同处理逻辑不代表与 10G 数量和框完全一致。

在 `app/projects/my3dfigure/services/face_detection.py` 中注释当前模型的导入行，
取消 `.scrfd_2_5g` 导入行的注释即可切换；一次请求只运行一条管线。
首次接入无需增加依赖或修改容器安装配置。

InsightFace 库代码使用 MIT 许可证。公开预训练权重仅许可非商业研究用途；
商业使用需另向 InsightFace 获取模型许可。本次仅本地接入。
