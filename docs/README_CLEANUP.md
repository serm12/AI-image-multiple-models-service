# Tasks目录清理工具 (Render部署版本)

这个工具用于定期清理tasks目录中的旧文件夹，以节省磁盘空间。

## 清理规则

1. **无watermark.md和original.md的文件夹**：如果文件夹创建时间超过7天，则删除整个文件夹
2. **有watermark.md但无original.md的文件夹**：如果文件夹创建时间超过28天，则删除整个文件夹
3. **有original.md的文件夹**：如果文件夹创建时间超过365天，则删除整个文件夹

## Render部署配置

### 1. 配置文件 (render.yaml)

```yaml
services:
  # 主Web服务
  - type: web
    name: ai-image-api
    env: python
    plan: free
    buildCommand: "pip install -r requirements.txt"
    startCommand: "python -m app.run"

  # 定时清理任务
  - type: cron
    name: cleanup-tasks
    plan: free
    # 每天凌晨2点执行（可以根据需要调整时间）
    schedule: "0 2 * * *"
    buildCommand: "pip install -r requirements.txt"
    startCommand: "python scripts/cleanup_scheduler.py"
```

### 2. 部署步骤

1. 将 `render.yaml` 文件添加到您的项目根目录
2. 推送到GitHub仓库
3. 在Render控制台中连接您的GitHub仓库
4. Render会自动检测到两个服务并部署它们

## 本地使用方法

### 试运行模式（推荐首次使用）
```bash
python scripts/cleanup_scheduler.py --dry-run
```

### 实际执行清理
```bash
python scripts/cleanup_scheduler.py
```

### 指定tasks目录路径
```bash
python scripts/cleanup_scheduler.py --tasks-dir /path/to/your/tasks
```

### 试运行模式并指定目录路径
```bash
python scripts/cleanup_scheduler.py --tasks-dir /path/to/your/tasks --dry-run
```

## 日志文件

清理操作的日志保存在 `logs/cleanup.log` 文件中。

## 注意事项

1. 请确保您的Render账户有足够的权限来部署cron jobs
2. 免费计划的cron jobs有执行频率限制
3. 建议先在本地测试清理规则，确保符合您的预期
4. 定期检查日志文件以监控清理任务的执行情况
