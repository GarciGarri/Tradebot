FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# tests ship in the image on purpose: the nightly job runs them as a promotion gate
COPY pyproject.toml ./
COPY tradebot ./tradebot
COPY tests ./tests
COPY config ./config
RUN pip install -e ".[dev]" && useradd --create-home --uid 10001 bot \
    && mkdir -p /app/state /app/reports && chown -R bot:bot /app
USER bot

HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 CMD ["tradebot", "healthcheck"]
CMD ["tradebot", "run"]
