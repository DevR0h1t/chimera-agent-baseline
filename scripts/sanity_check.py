"""Quick sanity check: load model, connect to MCP, run one case query.

Usage:
    python scripts/sanity_check.py
    python scripts/sanity_check.py --task task2
    python scripts/sanity_check.py --task task1 --case rumc-001
"""

import argparse
import asyncio
import json
import logging
import sys

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from omegaconf import OmegaConf

from chimera_agent_baseline.agent.graph import create_graph
from chimera_agent_baseline.models import load_model
from chimera_agent_baseline.schemas import format_case_prompt, load_queries, parse_prediction
from chimera_agent_baseline.utils import setup_logging

log = logging.getLogger(__name__)


async def run(task: str, case_id: str | None) -> None:
    data_dir = f"test/input/{task}"

    cfg = OmegaConf.load("configs/config.yaml")
    OmegaConf.update(cfg, "paths.input_dir", data_dir)
    setup_logging(cfg.logging.level)

    # Pick one query
    queries = load_queries(data_dir)
    if case_id:
        query = next((q for q in queries if q["case_id"] == case_id), None)
        if not query:
            log.error("Case %s not found in %s", case_id, data_dir)
            return
    else:
        query = queries[0]

    log.info("Loading model: %s", cfg.model.model_id)
    model = load_model(cfg)
    log.info("Model loaded")

    log.info("Starting MCP server (data_dir=%s)", data_dir)
    client = MultiServerMCPClient(
        {
            "chimera": {
                "command": sys.executable,
                "args": [
                    "-m", "chimera_agent_baseline.mcp_server",
                    "--data-dir", data_dir,
                    "--resource-dir", "resources",
                    "--skills-dir", "skills",
                ],
                "transport": "stdio",
            },
        }
    )
    tools = await client.get_tools()
    log.info("Loaded %d tools: %s", len(tools), [t.name for t in tools])

    agent_tools = [t for t in tools if t.name != "get_action_log"]
    graph = create_graph(agent_tools, model, "You are a clinical decision-support agent. Use tools to gather evidence.")

    prompt = format_case_prompt(query)
    log.info("Running agent for case %s", query["case_id"])
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=prompt)]},
        {"recursion_limit": cfg.agent.max_iterations},
    )

    # Print conversation
    print("\n" + "=" * 60)
    print(f"CASE: {query['case_id']} ({query.get('task', '?')})")
    print("=" * 60)
    for msg in result["messages"]:
        role = msg.type if hasattr(msg, "type") else "unknown"
        content = msg.content if hasattr(msg, "content") else str(msg)
        tool_calls = getattr(msg, "tool_calls", None)

        print(f"\n[{role}]")
        if content:
            print(content[:1000])
        if tool_calls:
            for tc in tool_calls:
                print(f"  -> tool_call: {tc['name']}({json.dumps(tc['args'])})")

    # Parse and show prediction
    final_text = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.content and msg.type == "ai":
            final_text = msg.content
            break

    prediction = parse_prediction(final_text, query["case_id"], query.get("task", "unknown"))

    # Retrieve action log from MCP server
    for t in tools:
        if t.name == "get_action_log":
            action_log = json.loads(await t.ainvoke({}))
            prediction["action_log"] = action_log
            break

    print("\n" + "=" * 60)
    print("PARSED PREDICTION:")
    print(json.dumps(prediction, indent=2))
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="task1", help="Task subdirectory (task1, task2, task3)")
    parser.add_argument("--case", default=None, help="Specific case_id (default: first case)")
    args = parser.parse_args()
    asyncio.run(run(args.task, args.case))


if __name__ == "__main__":
    main()
