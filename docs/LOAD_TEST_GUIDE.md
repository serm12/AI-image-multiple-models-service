# 压力测试指南

## 📋 脚本说明

### 1️⃣ **tests/quick_test.py** - 快速验证脚本（推荐先用）
- ⏱️ 耗时：~30 秒
- 🎯 目的：快速检查系统是否正常、验证 5-10 个并发是否可行
- 💡 特点：无需参数，开箱即用

### 2️⃣ **tests/load_test.py** - 完整压力测试脚本
- ⏱️ 耗时：可配置（默认 60s）
- 🎯 目的：完整的性能测试、获取详细的性能指标
- 💡 特点：支持多种测试模式、详细的报告生成

---

## 🚀 使用方法

### **快速验证（第一步）**

```bash
# 先启动服务
wsl -e bash -lc "cd /mnt/c/Users/yt640/Documents/Python/AI-image-multiple-models-service && source .venv-linux/bin/activate && python -m app.run"

# 新开一个终端，运行快速测试
python tests/quick_test.py
```

**预期输出：**
```
✅ 所有测试通过！
✓ 基本连接正常
✓ 5 个并发可以支持
✓ 10 个并发可以支持
✓ 生成请求可以并发处理
✓ 混合并发正常工作

系统状态: 🟢 良好
```

---

### **完整压力测试（进阶）**

#### **模式 1：混合负载测试（推荐）**
```bash
# 测试 5 个并发，持续 120 秒
python tests/load_test.py --concurrency 5 --duration 120 --mode mixed

# 测试 10 个并发，持续 180 秒
python tests/load_test.py --concurrency 10 --duration 180 --mode mixed
```

**测试内容：**
- 30% 查询风格（/styles/）
- 30% 生成图片（/generate-async/）
- 40% 查询任务状态（/task-status/）

#### **模式 2：生成负载测试**
```bash
# 5 个并发，每个提交 1 个生成任务，等待完成
python tests/load_test.py --concurrency 5 --mode generate

# 10 个并发
python tests/load_test.py --concurrency 10 --mode generate
```

**测试内容：**
- 每个 worker 提交一个生成任务
- 轮询等待任务完成（最多 200s）
- 统计完成数量

#### **其他参数**
```bash
# 自定义 API 地址（如果不是 localhost:8001）
python tests/load_test.py --url http://192.168.1.100:8001 --concurrency 10

# 可用参数：
# --url              API 基础 URL (默认: http://localhost:8001)
# --concurrency      并发数 (默认: 5)
# --duration         测试持续时间（秒）(默认: 60，仅 mixed 模式)
# --mode             测试模式 (mixed / generate, 默认: mixed)
```

---

## 📊 理解报告

### **混合负载测试报告示例**

```
📌 查询请求 (共 240 个):
   - 平均: 0.082s
   - 中位数: 0.075s
   - 最小: 0.045s
   - 最大: 0.150s
   - 标准差: 0.025s

📌 生成请求 (共 48 个):
   - 平均: 2.340s
   - 中位数: 2.180s
   - 最小: 1.950s
   - 最大: 3.200s
   - 标准差: 0.450s

📌 状态检查 (共 312 个):
   - 平均: 0.051s
   - 中位数: 0.048s
   - 最小: 0.035s
   - 最大: 0.120s
   - 标准差: 0.018s

📈 总体统计:
   - 总请求数: 600
   - 成功数: 598
   - 失败数: 2
   - 成功率: 99.67%
```

**关键指标解释：**
- ✅ **成功率 > 95%**：正常
- ⚠️ **成功率 80-95%**：有瓶颈，需要优化
- ❌ **成功率 < 80%**：系统过载，需要降低并发或增加资源

---

## 🧪 测试场景

### **场景 1：验证 5 个并发不卡（推荐先做）**
```bash
# 第一步：快速验证
python tests/quick_test.py

# 第二步：混合负载测试
python tests/load_test.py --concurrency 5 --duration 120 --mode mixed
```

**预期结果：**
- ✅ 快速验证全过通
- ✅ 混合负载成功率 > 99%
- ✅ 查询平均响应 < 0.2s
- ✅ 生成请求能正常提交

---

### **场景 2：测试 10 个并发的极限**
```bash
# 生成负载测试（最能暴露问题）
python tests/load_test.py --concurrency 10 --mode generate

# 混合负载测试
python tests/load_test.py --concurrency 10 --duration 180 --mode mixed
```

**预期结果：**
- ✅ 10 个生成任务能全部完成（可能较慢）
- ✅ 混合负载成功率 > 95%
- ⚠️ 生成请求可能排队（正常，因为 MAX_CONCURRENT_TASKS=5）

---

### **场景 3：测试系统稳定性（长时间运行）**
```bash
# 运行 30 分钟的混合负载测试
python tests/load_test.py --concurrency 5 --duration 1800 --mode mixed
```

**观察指标：**
- 是否有内存泄漏（内存持续增长）
- 是否有连接泄漏（连接数增长）
- 响应时间是否变差（长尾延迟）

---

## 🔍 故障排查

### **问题 1：连接拒绝**
```
❌ 连接失败: Cannot connect to host localhost:8001
```
**解决方案：**
- 确保服务已启动：`python -m app.run`
- 检查端口是否正确：`.env` 中 `PORT=8001`
- 如果用 WSL，确保 WSL 网络配置正确

### **问题 2：成功率低（< 80%）**
```
❌ 成功率: 65.32%
```
**解决方案：**
- 降低并发数：改用 5 个而不是 10 个
- 增加超时时间：API 响应可能较慢
- 检查 API Key 是否有效（看服务日志）

### **问题 3：生成请求超时**
```
⏱️ 生成请求超时或排队时间过长
```
**解决方案：**
- 正常现象，说明 MAX_CONCURRENT_TASKS=5 在控制并发
- 如果觉得太慢，可在 `.env` 中增加 `MAX_CONCURRENT_TASKS=10`

### **问题 4：内存持续增长**
```
⚠️ 测试 1 小时后内存从 400MB → 1GB
```
**解决方案：**
- 检查是否有内存泄漏（已在代码中优化）
- 查看日志是否有大量错误
- 重启服务

---

## 📝 建议的测试流程

### **第一天：基础验证**
```bash
1. python tests/quick_test.py          # 快速验证（5 分钟）
2. python tests/load_test.py --concurrency 5 --duration 120 --mode mixed   # 混合负载（2 分钟）
```

### **第二天：性能测试**
```bash
1. python tests/load_test.py --concurrency 10 --duration 180 --mode mixed   # 混合负载（3 分钟）
2. python tests/load_test.py --concurrency 10 --mode generate               # 生成负载（取决于任务时间）
```

### **第三天：稳定性测试**
```bash
1. python tests/load_test.py --concurrency 5 --duration 1800 --mode mixed   # 30 分钟长测
```

---

## 💡 调优建议

### **如果 5-10 并发完全没问题**
```bash
# 可以尝试增加 MAX_CONCURRENT_TASKS
# 在 .env 中改为：
MAX_CONCURRENT_TASKS=20

# 再运行测试：
python tests/load_test.py --concurrency 20 --duration 120 --mode mixed
```

### **如果 5 个并发就开始出现问题**
```bash
# 检查系统资源
# 1. CPU 是否跑满？
# 2. 内存是否不足？
# 3. 网络是否慢？

# 可以降低并发或优化代码
```

---

## 📚 更多信息

- 完整的并发分析：见 README.md 的"系统并发容量分析"部分
- 性能优化指南：见前期优化记录
- API 文档：见 `/api-docs`

---

**准备好了吗？让我们开始测试吧！** 🚀

```bash
python tests/quick_test.py
```
