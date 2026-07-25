# Stage 1: Dependency builder stage
FROM python:3.11-slim AS builder
WORKDIR /app

# Install system dependencies needed for compiling C extensions (XGBoost, scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Hardened production runner stage
FROM python:3.11-slim AS runner
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -u 1001 appuser

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source code with non-root ownership
COPY --chown=appuser:appuser . .

# Ensure data & cache directories exist with non-root permissions
RUN mkdir -p data data/model_cache && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Expose server port
EXPOSE 8000

# Production Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# Production Entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
