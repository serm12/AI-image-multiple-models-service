# 通用风格预设

所有模型共用同一套 `art_style` 标识和风格提示词。`GET /styles` 返回可选风格，`/admin/styles` 展示实际追加的提示词。

- 使用 `realistic`、`oil_painting`、`watercolor`、`acrylic` 等无模型前缀的名称。
- 相同风格不再按 Flux、Gemini、Seedream 分开。31 个预设合并为 24 个，并补充 1 个通用像素画预设 `pixel_art`（共 25 个），`watercolor_1` 并入 `watercolor`，其余具有不同效果的预设保留。
- `realistic` 不追加风格词；未选择或无法识别的风格也保持原始提示词。
- 旧名称（例如 `flux_realistic`、`gemini_oil_painting`、`seedream-4_fal_realistic`）仍可提交，会转换成通用名称保存。历史任务的旧风格名称仍可解析描述，无需改写任务文件。
- 风格词由 API 入口组合一次。模型客户端接收完整提示词，不额外追加风格词，也不默认切换成油画。

新增风格时，在 `app/core/config.py` 的 `ArtStyleEnum` 和 `STYLE_PRESETS` 中添加通用标识、描述和提示词即可，不需要按模型添加分支。旧名称兼容关系放在 `STYLE_ALIASES`。

验证：`.venv/Scripts/python.exe -m unittest discover -s tests -p test_style_catalog.py`，使用模拟请求，不调用付费生成服务。

## v2.1.40 提示词调整

使用简短连贯描述，限定绘画材质和表现技法；具体人物、服饰、姿势和场景由商品提示词决定。保留参考人物辨识度及原有肤色，厚涂在面部控制纹理强度，卡通和娃娃仍保留各自的造型比例。抽象风格在要求可辨识主体时保留轮廓与关键特征。写实预设仍不追加内容。

v2.1.39 完整提示词及旧名称映射保存在 `docs/style-history/v2.1.39.json`。本次只调整提示词和说明，不更改预设标识。代码验证不代表已验证各模型的实际出图效果。
