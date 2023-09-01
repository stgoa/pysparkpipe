FROM python:3.9 AS base
WORKDIR /pysparkpipe
COPY pyproject.toml ./
ADD /pysparkpipe ./pysparkpipe
ENV PATH="/root/.local/bin:$PATH"
RUN python -m pip install --upgrade pip
RUN apt-get update
RUN apt-get -y install openjdk-17-jdk
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-interaction --no-ansi
# Test image
FROM base as tester
COPY tests ./tests
RUN pip install pytest
RUN pytest -s -vvv
# Publish image
FROM base AS publisher
ARG PYPI_TOKEN
RUN poetry build
RUN poetry config pypi-token.pypi ${PYPI_TOKEN}
RUN poetry publish
