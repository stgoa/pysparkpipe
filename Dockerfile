# syntax=docker/dockerfile:1
FROM python:3.10-bookworm AS base

RUN apt-get update \
    && apt-get -y install --no-install-recommends openjdk-17-jdk \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 1000 --create-home --shell /bin/bash appuser

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /pysparkpipe
RUN chown appuser:appuser /pysparkpipe
ENV PATH="/pysparkpipe/.venv/bin:$PATH"
ENV UV_CACHE_DIR=/home/appuser/.cache/uv
USER appuser

COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/home/appuser/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --no-dev --no-install-project

COPY --chown=appuser:appuser pysparkpipe ./pysparkpipe
RUN --mount=type=cache,target=/home/appuser/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --no-dev

# Test image
FROM base AS tester
COPY --chown=appuser:appuser tests ./tests
RUN --mount=type=cache,target=/home/appuser/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --group dev
RUN pytest -s -vvv

# Publish image
FROM base AS publisher
ARG PYPI_TOKEN
RUN --mount=type=cache,target=/home/appuser/.cache/uv,uid=1000,gid=1000 uv build
RUN --mount=type=cache,target=/home/appuser/.cache/uv,uid=1000,gid=1000 \
    uv publish --token ${PYPI_TOKEN}
