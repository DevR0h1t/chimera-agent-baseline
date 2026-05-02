"""Agent entry-point.

Hydra-driven runner that starts the MCP tool server as a subprocess
(stdio transport) and runs the LangGraph ReAct + form-fill graph on
each case in the input directory, one at a time.

Usage::

    make run                                            # task 1, defaults
    make run RUN_ARGS="agent.tool_registry=task2 \\
        paths.input_dir=outputs/agent_input/task2"      # task 2
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
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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


_REGISTRY_TO_TASK_INT = {"task1": 1, "task2": 2}


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


def _action_log_from_messages(messages: list) -> list[dict[str, Any]]:
    """Reconstruct a per-case action log from the LangGraph message history."""
    entries: list[dict[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                pending[tc["id"]] = {"tool": tc["name"], "args": tc.get("args", {})}
        elif isinstance(m, ToolMessage):
            entry = pending.pop(getattr(m, "tool_call_id", ""), None) or {"tool": m.name, "args": {}}
            content = m.content if isinstance(m.content, str) else str(m.content)
            try:
                entry["result"] = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                entry["result"] = content
            entries.append(entry)
    return entries


def _final_assistant_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
            return m.content if isinstance(m.content, str) else json.dumps(m.content)
    return ""


def _thinking_trace(messages: list) -> list[dict[str, str]]:
    """Capture every assistant turn's ``reasoning_content`` if present.

    Models that route chain-of-thought through a separate channel (Qwen3+
    via llama.cpp, OpenAI o1) put it on
    ``additional_kwargs.reasoning_content``. Empty for models that
    don't expose one.
    """
    out: list[dict[str, str]] = []
    for m in messages:
        if isinstance(m, AIMessage):
            rc = (getattr(m, "additional_kwargs", {}) or {}).get("reasoning_content")
            if rc:
                out.append({"content": rc})
    return out


async def run_agent(cfg: DictConfig) -> None:
    """Load model, connect MCP tools, run the agent on each case."""

    tool_registry = cfg.agent.tool_registry
    if tool_registry not in _REGISTRY_TO_TASK_INT:
        raise ValueError(
            f"Unknown agent.tool_registry={tool_registry!r}; expected one of {sorted(_REGISTRY_TO_TASK_INT)}"
        )
    task_int = _REGISTRY_TO_TASK_INT[tool_registry]

    queries = _filter_queries(load_cases(cfg.paths.input_dir, task=task_int), cfg)

    log.info("Starting MCP server (data_dir=%s, tool_registry=%s)", cfg.paths.input_dir, tool_registry)
    client = MultiServerMCPClient(
        {
            "chimera": {
                "command": sys.executable,
                "args": [
                    "-m",
                    "chimera_agent_baseline.mcp_server",
                    "--data-dir",
                    str(cfg.paths.input_dir),
                    "--resource-dir",
                    str(cfg.paths.resource_dir),
                    "--tool-registry",
                    tool_registry,
                ],
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

    output_dir = Path(cfg.paths.output_dir)
    per_case_dir = output_dir / "predictions" / f"task{task_int}"
    per_case_dir.mkdir(parents=True, exist_ok=True)

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

        action_log = _action_log_from_messages(result["messages"])
        prediction = {
            **structured,
            "reasoning_trace": _final_assistant_text(result["messages"]),
            "thinking_trace": _thinking_trace(result["messages"]),
            "action_log": action_log,
            "form_fill_warnings": warnings,
        }

        # Persist incrementally so partial progress survives interruption.
        per_case_path = per_case_dir / f"{case_id}.json"
        per_case_path.write_text(json.dumps(prediction, indent=2))
        n_done += 1
        log.info("Case %s done (%d actions) -> %s", case_id, len(action_log), per_case_path)

    log.info("Wrote %d predictions to %s", n_done, per_case_dir)


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
