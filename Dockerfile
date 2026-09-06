FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    API_HOST=0.0.0.0 \
    API_PORT=8000

WORKDIR /app

COPY configs ./configs
COPY scripts ./scripts
COPY src ./src
COPY LICENSE README.md ./

RUN python scripts/generate_synthetic_corpus.py \
    && python scripts/ingest_knowledge_base.py \
    && python scripts/build_retrieval_index.py \
    && python scripts/build_bm25_index.py \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/artifacts \
    && chown appuser:appuser /app/artifacts

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["python", "scripts/serve_api.py"]
