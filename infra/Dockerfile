# ═══════════════════════════════════════════════════════════════════════════
# Meta-Agent — Multi-stage Docker build
# ═══════════════════════════════════════════════════════════════════════════

# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir . && \
    pip install --no-cache-dir ".[dev]"


# Stage 2: Production image
FROM python:3.12-slim AS production

# Security: non-root user
RUN groupadd --system meta && \
    useradd --system --gid meta --create-home meta

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY pyproject.toml ./

# Set environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8080

# Switch to non-root user
USER meta

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/health')" || exit 1

CMD ["uvicorn", "meta_agent.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
