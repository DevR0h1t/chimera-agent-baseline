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
FROM --platform=linux/amd64 vllm/vllm-openai:latest

ENV PYTHONUNBUFFERED=1

# Non-root user (required by Grand Challenge)
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

# Gemma 4 requires transformers >=5.x; the vLLM base image ships 4.x.
# Remove this line once a vLLM release bundles transformers 5.x.
RUN pip install --no-cache-dir --upgrade transformers

# Install project dependencies. Uses a stub src/ so hatchling can resolve
# the package, then reinstalls with the actual source.
COPY --chown=user:user pyproject.toml README.md /opt/app/
RUN mkdir -p src/chimera_agent_baseline && \
    touch src/chimera_agent_baseline/__init__.py && \
    pip install --no-cache-dir --no-color . && \
    rm -rf src/

# Copy application code and install the package (no-deps, source only)
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
