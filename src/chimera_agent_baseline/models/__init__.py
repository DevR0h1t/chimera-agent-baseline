"""Model loading via vLLM offline inference.

Loads a chat model in-process using :class:`vllm.LLM`. No HTTP server
needed. The model is loaded from ``paths.model_dir`` (local weights)
or downloaded from HuggingFace Hub via ``model.model_id``.
"""

import logging
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def load_model(cfg: DictConfig) -> BaseChatModel:
    """Load a vLLM chat model from config."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError("vLLM is required. Install with: pip install vllm") from exc

    from chimera_agent_baseline.models.vllm_offline import ChatVLLM

    model_path = _resolve_model_path(cfg)
    log.info("Loading vLLM model from %s", model_path)

    llm = LLM(model=model_path, dtype="auto", max_model_len=cfg.generation.get("max_model_len", 32768))

    params = SamplingParams(
        temperature=cfg.generation.temperature,
        max_tokens=cfg.generation.max_new_tokens,
        top_p=cfg.generation.top_p,
    )

    tool_parser = cfg.model.get("tool_parser", "gemma4")
    return ChatVLLM(llm=llm, sampling_params=params, tool_parser=tool_parser)


def _resolve_model_path(cfg: DictConfig) -> str:
    """Return local model dir if non-empty, else the HF hub model ID."""
    model_dir = Path(cfg.paths.model_dir)
    if model_dir.exists() and any(f for f in model_dir.iterdir() if not f.name.startswith(".")):
        return str(model_dir)
    return cfg.model.model_id
