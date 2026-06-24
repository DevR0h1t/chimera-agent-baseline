# =============================================================================
# Grand Challenge container
#
# Based on vLLM (includes CUDA + PyTorch + vLLM for in-process inference).
# Runs as non-root user, no network access at runtime.
#
# Input:   /input          (read-only, mounted by platform)
# Output:  /output         (writable, agent writes results here)
# Model:   /opt/ml/model   (read-only, weights uploaded separately)
# App:     /opt/app        (read-only, application code + resources)
# =============================================================================
# Base image pinned by digest for reproducible builds. This is
# vllm/vllm-openai:latest as of 2026-06 (ships vLLM 0.23.0 + CUDA + PyTorch +
# transformers 4.x). Keep this digest in sync with the Makefile BASE_IMAGE and
# regenerate requirements.lock (`make lock`) whenever you bump it.
FROM --platform=linux/amd64 vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f

ENV PYTHONUNBUFFERED=1

# Non-root user (required by Grand Challenge)
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

# Install the pinned dependency closure resolved on top of this base image.
# The lockfile includes the transformers 5.x upgrade Gemma 4 requires, so the
# build is fully deterministic — no unpinned `pip install --upgrade`.
COPY --chown=user:user requirements.lock pyproject.toml README.md /opt/app/
RUN pip install --no-cache-dir -r requirements.lock

# Copy application code and install the package itself (deps already satisfied
# by the lockfile above, so --no-deps).
COPY --chown=user:user src/ /opt/app/src/
COPY --chown=user:user inference.py /opt/app/
RUN pip install --no-cache-dir --no-deps .

# Copy configs (single source of truth — same file Hydra reads locally),
# Jinja prompt templates, and runtime resources (RAG DB + embedding model).
COPY --chown=user:user configs/ /opt/app/configs/
COPY --chown=user:user templates/ /opt/app/templates/
COPY --chown=user:user resources/ /opt/app/resources/

USER user

ENTRYPOINT ["python3", "inference.py"]
