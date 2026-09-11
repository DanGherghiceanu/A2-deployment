# syntax=docker/dockerfile:1

# --- Base image -------------------------------------------------------------
# "slim" over the full image: same Debian, without the compilers and docs.
# Pinned to a minor version so a rebuild months from now gets the same Python.
FROM python:3.12-slim

# Python behaviour inside a container:
#   PYTHONDONTWRITEBYTECODE  no .pyc files in a layer that gets thrown away
#   PYTHONUNBUFFERED         print/log lines reach stdout immediately, so Cloud
#                            Run captures them instead of losing a buffer on crash
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# libgomp1 is OpenMP, which torch links against at runtime. Without it the
# import fails with "libgomp.so.1: cannot open shared object file".
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Dependencies -----------------------------------------------------------
# Copied and installed before the application code so Docker can cache this
# layer. Editing app/main.py then rebuilds in seconds instead of re-downloading
# PyTorch every time.
COPY requirements-torch.txt requirements.txt ./

# CPU-only wheels: ~200 MB instead of the ~2 GB CUDA build, which is dead
# weight on a Cloud Run instance with no GPU attached.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        -r requirements-torch.txt \
    && pip install --no-cache-dir -r requirements.txt

# --- Application ------------------------------------------------------------
COPY app/ ./app/

# The trained model is baked into the image rather than downloaded at startup.
# The image is then a single self-contained artifact: same bytes, same weights,
# same predictions, on any machine that pulls it.
COPY models/resnet18_finetuned.pt ./models/resnet18_finetuned.pt

# Run as a non-root user. If the process is ever compromised it has no rights
# to write anywhere that matters.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# --- Runtime ----------------------------------------------------------------
ENV MODEL_PATH=/app/models/resnet18_finetuned.pt \
    DECISION_THRESHOLD=0.5 \
    TORCH_NUM_THREADS=1 \
    LOG_LEVEL=INFO \
    PORT=8080

# Documents the listening port. Publishing it is still up to `docker run -p`.
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,os,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/health', timeout=4).status==200 else 1)"

# Shell form so $PORT expands. Cloud Run injects PORT and the container must
# listen on it; the default of 8080 covers a plain `docker run`.
# One worker: each holds its own copy of the model in memory, and Cloud Run
# scales by adding containers rather than processes.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
