FROM python:3.13-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu && \
    uv sync --frozen --no-dev

COPY download.py main.py inference.py app.py entrypoint.sh ./

# persist dataset/ and model.pt across container restarts
VOLUME /app/dataset

EXPOSE 7860

ENTRYPOINT ["./entrypoint.sh"]
