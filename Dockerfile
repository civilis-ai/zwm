# ---- 多阶段构建 ----
# Stage 1: 基础依赖
FROM python:3.12-slim AS base

# 防止 .pyc 文件写入和 stdout/stderr 缓冲
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 安装系统依赖 (torch 需要 libgomp)
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Stage 2: 依赖层 — 利用 Docker 缓存
FROM base AS deps
COPY pyproject.toml .
# 安装完整依赖 (torch + api)
RUN pip install --no-cache-dir -e ".[api]"

# Stage 3: 源码层
FROM deps AS source
COPY src/ src/
RUN pip install --no-cache-dir -e ".[api]"

# Stage 4: 生产镜像
FROM source AS production

# 创建非 root 用户
RUN useradd --create-home --shell /bin/bash zwm
USER zwm

# 数据卷挂载点
VOLUME ["/data"]
ENV ZWM_DB_PATH=/data/zwm.db \
    ZWM_LOG_DIR=/data/runs

# 默认暴露 FastAPI 端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 默认启动 API 服务
CMD ["python", "-m", "uvicorn", "zwm.api.app:app", "--host", "0.0.0.0", "--port", "8000"]