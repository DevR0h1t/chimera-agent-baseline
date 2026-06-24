"""Agent entry-point.

Hydra-driven runner that starts the MCP tool server as a subprocess
(stdio transport) and runs the LangGraph ReAct + form-fill graph on
each case in the input directory, one at a time.

Usage::

    make run                                            # task 1, defaults
    make run RUN_ARGS="agent.tool_registry=task2 \\
        paths.input_dir=data/task2/agent_input"          # task 2
    make run RUN_ARGS="+experiment=qwen_local"          # swap to Qwen
    make run RUN_ARGS="agent.limit=5"                   # first 5 cases
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import hydra
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from omegaconf import DictConfig
from pydantic import ValidationError

from chimera_agent_baseline.agent.graph import create_graph
from chimera_agent_baseline.agent.prompts import build_system_prompt
from chimera_agent_baseline.case_loader import load_cases
from chimera_agent_baseline.models import load_model
from chimera_agent_baseline.output.schema import TASK_OUTPUT_MODELS
from chimera_agent_baseline.rag import start_embedding_service
from chimera_agent_baseline.utils import setup_logging

load_dotenv()
log = logging.getLogger(__name__)


_REGISTRY_TO_TASK_INT = {"task1": 1, "task2": 2, "task3": 3}


def _filter_queries(queries: list[dict], cfg: DictConfig) -> list[dict]:
    """Apply optional ``cfg.agent.pids`` / ``cfg.agent.limit`` subset filters."""
    pids = cfg.agent.get("pids")
    if pids:
        wanted = set(pids)
        out = [q for q in queries if q["case_id"] in wanted]
        missing = wanted - {q["case_id"] for q in out}
        if missing:
            log.warning("Requested pids not found in input dir: %s", sorted(missing))
        log.info("Filtered to %d hand-picked cases: %s", len(out), [q["case_id"] for q in out])
        return out
    limit = cfg.agent.get("limit")
    if limit:
        out = queries[: int(limit)]
        log.info("Limiting to first %d cases", len(out))
        return out
    return queries


async def run_agent(cfg: DictConfig) -> None:
    """Load model, connect MCP tools, run the agent on each case."""

    tool_registry = cfg.agent.tool_registry
    if tool_registry not in _REGISTRY_TO_TASK_INT:
        raise ValueError(
            f"Unknown agent.tool_registry={tool_registry!r}; expected one of {sorted(_REGISTRY_TO_TASK_INT)}"
        )
    task_int = _REGISTRY_TO_TASK_INT[tool_registry]

    queries = _filter_queries(load_cases(cfg.paths.input_dir, task=task_int), cfg)

    mcp_args = [
        "-m",
        "chimera_agent_baseline.mcp_server",
        "--data-dir",
        str(cfg.paths.input_dir),
        "--resource-dir",
        str(cfg.paths.resource_dir),
        "--tool-registry",
        tool_registry,
    ]
    # Optional image-embedding predictor tool (off by default).
    predictor = cfg.agent.get("predictor")
    if predictor and predictor.get("enabled"):
        mcp_args += ["--enable-predictor"]

    log.info("Starting MCP server (data_dir=%s, tool_registry=%s)", cfg.paths.input_dir, tool_registry)
    client = MultiServerMCPClient(
        {
            "chimera": {
                "command": sys.executable,
                "args": mcp_args,
                "transport": "stdio",
            },
        }
    )
    tools = await client.get_tools()
    log.info("Loaded %d tools from MCP server", len(tools))

    model = load_model(cfg)
    system_prompt = build_system_prompt()
    graph = create_graph(
        tools,
        model,
        system_prompt,
        step_timeout=cfg.agent.step_timeout,
        form_fill_max_retries=cfg.agent.form_fill.max_retries,
    )

    # Output mirrors the agent-input hierarchy: <output_dir>/task<N>/<case_id>/prediction.json
    task_dir = Path(cfg.paths.output_dir) / f"task{task_int}"

    n_done = 0
    for query in queries:
        case_id = query["case_id"]
        log.info("Processing case %s (task: %s)", case_id, query.get("task", "unknown"))

        # The graph's ReAct loop runs the agent and tools until a final
        # assistant message arrives, then the terminal ``form_fill`` node
        # prompts the SAME model with a per-case Pydantic schema and
        # validates with PydanticOutputParser. No external API.
        initial_state: dict[str, Any] = {
            "messages": [HumanMessage(content=query["context"])],
            "case_id": case_id,
            "task": task_int,
        }
        result = await graph.ainvoke(initial_state, {"recursion_limit": cfg.agent.max_iterations})

        # Fail-early schema check: the structured part of the prediction
        # must validate against Task1Output / Task2Output before we move
        # on. Form-fill already retries internally; if we still got here
        # with an invalid payload, something is wrong and partial outputs
        # would just mask it.
        structured = result.get("structured_response") or {}
        warnings = result.get("form_fill_warnings", [])
        try:
            TASK_OUTPUT_MODELS[task_int].model_validate(structured)
        except ValidationError as exc:
            log.error(
                "Output schema validation failed for case %s. form_fill_warnings=%s",
                case_id,
                warnings,
            )
            raise RuntimeError(
                f"Case {case_id}: structured output does not validate against "
                f"{TASK_OUTPUT_MODELS[task_int].__name__}. "
                f"form_fill_warnings={warnings}\n{exc}"
            ) from exc

        # Single output file per patient, in a per-case folder mirroring the
        # agent input. The file is exactly the validated structured record.
        case_dir = task_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        per_case_path = case_dir / "prediction.json"
        per_case_path.write_text(json.dumps(structured, indent=2))
        n_done += 1
        log.info("Case %s done -> %s", case_id, per_case_path)

    log.info("Wrote %d predictions under %s", n_done, task_dir)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.logging.level)

    embed_svc = start_embedding_service(cfg.paths.resource_dir)
    try:
        asyncio.run(run_agent(cfg))
    finally:
        if embed_svc:
            embed_svc.stop()


if __name__ == "__main__":
    main()
