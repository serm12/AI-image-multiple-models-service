# scripts/ - 运维脚本

| 脚本 | 说明 |
|------|------|
| `cleanup_scheduler.py` | 定期清理 tasks 目录中的旧文件 |
| `start_wsl.bat` | Windows WSL 一键启动（bat） |
| `start_wsl.sh` | WSL 内一键启动（bash） |

## 使用方法

```bash
# 清理旧任务（试运行）
python scripts/cleanup_scheduler.py --dry-run

# 清理旧任务（实际执行）
python scripts/cleanup_scheduler.py

# WSL 启动（Windows 下双击或命令行运行）
scripts\start_wsl.bat
```
