# tests/ - 测试脚本

| 脚本 | 说明 |
|------|------|
| `quick_test.py` | 快速验证（~30s），检查系统是否正常 |
| `e2e_test.py` | 端到端测试，验证从提交到完成的全流程 |
| `load_test.py` | 压力测试，支持并发/持续时间/混合模式 |

## 使用方法

```bash
# 快速验证
python tests/quick_test.py

# 端到端测试
python tests/e2e_test.py

# 压力测试
python tests/load_test.py --concurrency 5 --duration 120 --mode mixed
```

详细说明见 [docs/LOAD_TEST_GUIDE.md](../docs/LOAD_TEST_GUIDE.md)
