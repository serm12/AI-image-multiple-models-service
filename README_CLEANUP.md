# Tasks目录清理工具

这个工具用于定期清理tasks目录中的旧文件和文件夹，以节省磁盘空间。

## 清理规则

1. **无watermark.md和original.md的文件夹**：如果文件夹创建时间超过7天，则删除整个文件夹
2. **有watermark.md但无original.md的文件夹**：删除文件夹中超过28天的文件
3. **有original.md的文件夹**：删除文件夹中超过365天的文件

## 使用方法

### 试运行模式（推荐首次使用）
```bash
python cleanup_scheduler.py --dry-run
```

### 实际执行清理
```bash
python cleanup_scheduler.py
```

### 指定tasks目录路径
```bash
python cleanup_scheduler.py --tasks-dir /path/to/your/tasks
```

## 设置定时任务

### Windows任务计划程序
1. 打开"任务计划程序"
2. 创建基本任务
3. 设置触发器为每天执行
4. 操作设置为启动程序：`python`
5. 添加参数：`cleanup_scheduler.py`

### Linux/macOS Cron任务
添加到crontab中：
```bash
# 每天凌晨2点执行清理
0 2 * * * cd /path/to/your/project && python cleanup_scheduler.py
```

## 日志文件

清理操作的日志保存在 `logs/cleanup.log` 文件中。