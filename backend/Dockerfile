FROM python:3.11-slim AS base

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

COPY . .

RUN mkdir -p data/verdicts

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
