FROM ghcr.io/astral-sh/uv:0.4.30-python3.11-bookworm-slim
WORKDIR /app
ENV UV_SYSTEM_PYTHON=1
ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY src/app/ .

CMD  ["uv", "run", "granian", "--interface","asgi", "main.py"]
