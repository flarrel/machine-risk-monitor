FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MACHINE_RISK_ROOT=/app

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY artifacts/model ./artifacts/model
COPY static ./static
COPY templates ./templates

RUN pip install --no-cache-dir .

RUN useradd --create-home appuser && mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "machine_risk.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
