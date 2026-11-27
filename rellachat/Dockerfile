# syntax=docker/dockerfile:1
#
# Rellachat voice service image.
#
# Build from the repository root, because the image installs the Pipecat framework
# from `src/` and `pyproject.toml` next to our code:
#
#   docker build -f rellachat/Dockerfile -t rellachat-voice .
#
# Configuration comes only from environment variables (`docker run --env-file`,
# or `env_file:` in rellachat/docker-compose.yml). No .env file is copied into the image.

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------
# Builder: install Pipecat with the extras we use into a self-contained venv.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

# The package version comes from setuptools_scm, which reads git tags. The build
# context has no .git, so pin the version explicitly (keep it in line with the fork's tag).
ARG PIPECAT_VERSION=1.17.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${PIPECAT_VERSION} \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN uv venv /opt/venv --python /usr/local/bin/python3

WORKDIR /build
COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY src ./src

# Every dependency ships manylinux wheels for x86_64 and aarch64 (aiortc, av, onnxruntime,
# opencv-python-headless, pyrnnoise...), except pure-Python docopt, so no compiler is needed.
# A regular (non-editable) install, so the runtime stage does not need `src/`.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
        ".[anthropic,cartesia,webrtc,runner,silero,rnnoise]"

# ---------------------------------------------------------------------------
# Runtime: the venv plus our service code, run as a non-root user.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

# libgomp1: OpenMP runtime some native wheels (onnxruntime, opencv) may load. It is small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin rellachat

ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=7860

COPY --from=builder /opt/venv /opt/venv

# Code stays root-owned (read-only for the service user). server.py adds /app to sys.path,
# so `rellachat.*` imports resolve.
WORKDIR /app
COPY rellachat ./rellachat

USER rellachat

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '7860'), timeout=4)"

CMD ["python", "rellachat/server.py"]
