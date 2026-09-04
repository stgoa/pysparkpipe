FROM python:3.10-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN apt-get update
RUN apt-get -y install openjdk-17-jdk