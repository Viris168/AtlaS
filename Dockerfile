# Dockerfile — Milestone 11: container for the service.
#
#   docker build -t atlas .
#   docker run --rm -p 8000:8000 --env-file .env -e AGENT_API_TOKEN=change-me \
#       -v atlas-data:/app/data -v atlas-logs:/app/logs -v "$PWD/workspace":/app/workspace atlas
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Listen on every interface inside the container; publish the port to choose who can reach it.
    AGENT_HOST=0.0.0.0 \
    AGENT_PORT=8000 \
    # Embedding and rerank models download here on first use; mount it to keep them between runs.
    HF_HOME=/app/data/hf \
    UV_CACHE_DIR=/app/data/uv

WORKDIR /app

# run_shell needs the usual tools; curl is also the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first, so sentence-transformers doesn't pull the multi-GB CUDA build.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt uv  # uv provides uvx, which starts the MCP servers in mcp_servers.json

COPY . .

# Not root: the agent runs shell commands, so keep what they can touch small.
RUN useradd --create-home atlas \
    && mkdir -p data logs workspace knowledge \
    && chown -R atlas:atlas data logs workspace knowledge
USER atlas

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fsS -H "Authorization: Bearer ${AGENT_API_TOKEN}" http://localhost:8000/health || exit 1
CMD ["python", "server.py"]
