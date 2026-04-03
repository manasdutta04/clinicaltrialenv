FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl && rm -rf /var/lib/apt/lists/*

# Install runtime dependencies
RUN pip install --no-cache-dir \
    "openenv-core==0.2.1" \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.29.0" \
    "numpy>=1.26.0" \
    "scipy>=1.12.0" \
    "pydantic>=2.6.0" \
    "websockets>=12.0" \
    "requests>=2.31.0"

COPY . .

EXPOSE 7860
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true
ENV PORT=7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
