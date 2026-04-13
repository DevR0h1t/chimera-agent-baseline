"""Grand Challenge entrypoint.

Reads inputs from /input, runs the agent on each case, writes structured
predictions to /output.  This script is the ENTRYPOINT of the Grand
Challenge container.  For local development, use ``make run`` instead.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from omegaconf import OmegaConf

from chimera_agent_baseline.agent.graph import create_graph
from chimera_agent_baseline.agent.prompts import build_system_prompt
from chimera_agent_baseline.models import load_model
from chimera_agent_baseline.rag import start_embedding_service
from chimera_agent_baseline.schemas import format_case_prompt, load_queries, parse_prediction
from chimera_agent_baseline.skills import format_skills_summary, load_skills
from chimera_agent_baseline.utils import setup_logging

log = logging.getLogger(__name__)

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("/opt/app/resources")
MODEL_PATH = Path("/opt/ml/model")


def load_config() -> OmegaConf:
    """Load config from baked-in resources with container path overrides."""
    cfg = OmegaConf.load(RESOURCE_PATH / "config.yaml")
    OmegaConf.update(cfg, "paths.input_dir", str(INPUT_PATH))
    OmegaConf.update(cfg, "paths.output_dir", str(OUTPUT_PATH))
    OmegaConf.update(cfg, "paths.resource_dir", str(RESOURCE_PATH))
    OmegaConf.update(cfg, "paths.model_dir", str(MODEL_PATH))
    return cfg


async def _get_action_log(tools: list) -> list[dict]:
    """Retrieve and clear the action log from the MCP server."""
    for tool in tools:
        if tool.name == "get_action_log":
            result = await tool.ainvoke({})
            # MCP adapter may return a string or a list of content blocks
            if isinstance(result, str):
                return json.loads(result)
            if isinstance(result, list):
                text = result[0]["text"] if result and isinstance(result[0], dict) else str(result[0])
                return json.loads(text)
            return json.loads(str(result))
    return []


async def run_agent(cfg: OmegaConf) -> list[dict]:
    """Load model, connect MCP tools, run the agent on each case."""

    queries = load_queries(cfg.paths.input_dir)
    model = load_model(cfg)

    # Resolve skills directory
    skills_dir = Path(__file__).parent / "skills"
    if not skills_dir.exists():
        skills_dir = RESOURCE_PATH / "skills"
    skills = load_skills(skills_dir)
    system_prompt = build_system_prompt(format_skills_summary(skills))

    log.info("Starting MCP server (data_dir=%s)", cfg.paths.input_dir)
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
                    "--skills-dir",
                    str(skills_dir),
                ],
                "transport": "stdio",
            },
        }
    )
    tools = await client.get_tools()
    log.info("Loaded %d tools from MCP", len(tools))

    # Exclude get_action_log from tools the agent sees
    agent_tools = [t for t in tools if t.name != "get_action_log"]
    graph = create_graph(agent_tools, model, system_prompt, step_timeout=cfg.agent.step_timeout)

    predictions = []
    for query in queries:
        case_id = query["case_id"]
        task = query.get("task", "unknown")
        log.info("Processing case %s (task: %s)", case_id, task)

        prompt = format_case_prompt(query)
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=prompt)]},
            {"recursion_limit": cfg.agent.max_iterations},
        )

        final_text = ""
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.content and msg.type == "ai":
                final_text = msg.content
                break

        prediction = parse_prediction(final_text, case_id, task)

        # Retrieve action log from MCP server (framework-agnostic)
        prediction["action_log"] = await _get_action_log(tools)

        predictions.append(prediction)
        log.info("Case %s done (%d actions)", case_id, len(prediction["action_log"]))

    return predictions


def run() -> int:
    cfg = load_config()
    setup_logging(cfg.logging.level)

    log.info("Starting agent inference")
    log.info("Model: %s", cfg.model.model_id)

    # Start embedding service (long-lived, used by MCP search_guidelines)
    embed_svc = start_embedding_service(cfg.paths.resource_dir)

    try:
        predictions = asyncio.run(run_agent(cfg))

        output_file = OUTPUT_PATH / "predictions.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(predictions, indent=2))
        log.info("Wrote %d predictions to %s", len(predictions), output_file)
    finally:
        if embed_svc:
            embed_svc.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
