FROM python:3.10-bookworm AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /pysparkpipe
COPY pyproject.toml uv.lock ./
ADD /pysparkpipe ./pysparkpipe
ENV PATH="/pysparkpipe/.venv/bin:$PATH"
RUN apt-get update
RUN apt-get -y install openjdk-17-jdk
RUN uv sync --frozen --no-dev
# Test image
FROM base as tester
COPY tests ./tests
RUN uv sync --frozen --group dev
RUN uv run pytest -s -vvv
# Publish image
FROM base AS publisher
ARG PYPI_TOKEN
RUN uv build
RUN uv publish --token ${PYPI_TOKEN}
