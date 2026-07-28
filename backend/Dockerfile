FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml alembic.ini ./
COPY app ./app
COPY alembic ./alembic
COPY entrypoint.sh ./

RUN pip install --upgrade pip && pip install -e . && chmod +x entrypoint.sh

EXPOSE 8000

CMD ["./entrypoint.sh"]
