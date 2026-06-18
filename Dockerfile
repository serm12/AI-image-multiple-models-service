# ========== 构建阶段 ==========
FROM python:3.12-slim AS builder

# 安装构建依赖（opencv等需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# 先复制依赖文件，利用 Docker 缓存层
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ========== 运行阶段 ==========
FROM python:3.12-slim

# 安装运行时依赖
# libgl1, libglib2.0-0: opencv-headless 运行需要
# libmagic1: python-magic 运行需要（保留以防未来使用）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 复制已安装的 Python 包
COPY --from=builder /root/.local /root/.local

# 确保用户包在 PATH 中
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 复制项目文件
COPY app/ ./app/
COPY assets/ ./assets/
COPY scripts/ ./scripts/

# 创建数据和日志目录
RUN mkdir -p /app/tasks /app/logs

# 确保 tasks 目录有 .gitkeep
RUN touch /app/tasks/.gitkeep

EXPOSE 8001

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.getenv(\"PORT\", \"8001\")}/health')" || exit 1

# 启动服务
CMD ["python", "-m", "app.run"]
