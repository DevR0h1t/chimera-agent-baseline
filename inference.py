"""Grand Challenge entrypoint.

Reads inputs from ``/input`` (the same ``task<N>/agent_input/<case>/``
hierarchy used locally), runs every task present, and writes structured
predictions to ``/output/task<N>/<case>/prediction.json``. Thin wrapper around
:func:`chimera_agent_baseline.run.run_agent` that loads the same
``configs/config.yaml`` Hydra reads locally and overrides the path fields to
the GC container's mount points.

For local development, use ``make run`` instead.
"""

import asyncio
import logging
from pathlib import Path

from omegaconf import OmegaConf

from chimera_agent_baseline.rag import start_embedding_service
from chimera_agent_baseline.run import run_agent
from chimera_agent_baseline.utils import setup_logging

log = logging.getLogger(__name__)

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
CONFIG_PATH = Path("/opt/app/configs/config.yaml")
RESOURCE_PATH = Path("/opt/app/resources")
MODEL_PATH = Path("/opt/ml/model")


def load_config():
    """Load the canonical config and override paths for the GC container."""
    cfg = OmegaConf.load(CONFIG_PATH)
    OmegaConf.update(cfg, "paths.data_root", str(INPUT_PATH))
    OmegaConf.update(cfg, "paths.output_dir", str(OUTPUT_PATH))
    OmegaConf.update(cfg, "paths.resource_dir", str(RESOURCE_PATH))
    OmegaConf.update(cfg, "paths.model_dir", str(MODEL_PATH))
    return cfg


def run() -> int:
    cfg = load_config()
    setup_logging(cfg.logging.level)

    log.info("Starting agent inference (model=%s, tasks=%s)", cfg.model.model_id, list(cfg.agent.tasks))

    embed_svc = start_embedding_service(cfg.paths.resource_dir)
    try:
        asyncio.run(run_agent(cfg))
    finally:
        if embed_svc:
            embed_svc.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
