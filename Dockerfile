# =============================================================================
# emAI — multi-stage Docker build (Easypanel-ready)
# =============================================================================
# Two runtime targets:
#   dashboard: streamlit run src/ui/app.py
#   worker:    python -m src.main --all-users
# =============================================================================

# ---- Stage 1: Builder -------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for psycopg2 build
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

COPY . .
RUN pip install --no-cache-dir -e .

# ---- Stage 2: Runtime -------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime deps only (libpq for psycopg2)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Streamlit config: disable telemetry, set port
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

# Default: dashboard mode
CMD ["streamlit", "run", "src/ui/app.py", "--server.address=0.0.0.0"]
