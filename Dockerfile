# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd \
      --system \
      --gid 10001 \
      mosaic \
    && useradd \
      --system \
      --uid 10001 \
      --gid mosaic \
      --create-home \
      mosaic

COPY requirements-api.txt ./requirements-api.txt

RUN python3 -m pip install \
      --disable-pip-version-check \
      --no-cache-dir \
      -r requirements-api.txt

COPY mosaic ./mosaic
COPY data ./data
COPY generated ./generated

RUN chown -R mosaic:mosaic /app

USER mosaic

EXPOSE 8000

HEALTHCHECK \
  --interval=10s \
  --timeout=3s \
  --start-period=10s \
  --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["python3", "-m", "uvicorn", "mosaic.api:app", "--host", "0.0.0.0", "--port", "8000"]
