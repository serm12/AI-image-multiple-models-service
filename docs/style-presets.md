# 通用风格预设

所有模型共用同一套 `art_style` 标识和风格提示词。`GET /styles` 返回可选风格，`/admin/styles` 展示实际追加的提示词。

- 使用 `realistic`、`oil_painting`、`watercolor`、`acrylic` 等无模型前缀的名称。
- 相同风格不再按 Flux、Gemini、Seedream 分开。31 个预设合并为 24 个，并补充 1 个通用像素画预设 `pixel_art`（共 25 个），`watercolor_1` 并入 `watercolor`，其余具有不同效果的预设保留。
- `realistic` 不追加风格词；未选择或无法识别的风格也保持原始提示词。
- 旧名称（例如 `flux_realistic`、`gemini_oil_painting`、`seedream-4_fal_realistic`）仍可提交，会转换成通用名称保存。历史任务的旧风格名称仍可解析描述，无需改写任务文件。
- 风格词由 API 入口组合一次。模型客户端接收完整提示词，不额外追加风格词，也不默认切换成油画。

新增风格时，在 `app/core/config.py` 的 `ArtStyleEnum` 和 `STYLE_PRESETS` 中添加通用标识、描述和提示词即可，不需要按模型添加分支。旧名称兼容关系放在 `STYLE_ALIASES`。

验证：`.venv/Scripts/python.exe -m unittest discover -s tests -p test_style_catalog.py`，使用模拟请求，不调用付费生成服务。
