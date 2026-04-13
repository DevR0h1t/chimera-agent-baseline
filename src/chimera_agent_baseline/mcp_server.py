"""MCP server exposing precomputed clinical tools.

Runs as a stdio-based MCP server.  All tool calls are automatically
logged to an action log, retrievable via the ``get_action_log`` tool.
This provides a framework-agnostic record of what tools the agent
called and what data it received — regardless of whether the agent
uses LangGraph, AutoGen, or a custom loop.

Usage::

    python -m chimera_agent_baseline.mcp_server --data-dir test/input/task1 --resource-dir resources
"""

import argparse
import json
import logging
import tempfile
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from chimera_agent_baseline.tools.base import CaseDataStore, ToolSpec
from chimera_agent_baseline.tools.definitions import TOOL_REGISTRY
from chimera_agent_baseline.utils import setup_logging

log = logging.getLogger(__name__)

# Shared file-based action log. Uses a temp file so multiple MCP server
# subprocess invocations (one per tool call via langchain-mcp-adapters)
# all append to the same log.
_ACTION_LOG_PATH = Path(tempfile.gettempdir()) / "chimera_action_log.jsonl"


def create_server(
    data_dir: str,
    resource_dir: str | None = None,
    skills_dir: str | None = None,
    tools: list[ToolSpec] | None = None,
    name: str = "Chimera Tools",
) -> FastMCP:
    """Create an MCP server with precomputed clinical tools."""
    mcp = FastMCP(name)
    store = CaseDataStore(data_dir)
    tools = tools if tools is not None else TOOL_REGISTRY

    # -- Action log (file-based, persists across subprocess restarts) ----------

    def _logged(fn):
        """Decorator that records tool calls and results to the action log."""

        @wraps(fn)
        def wrapper(*args, **kwargs):
            call_args = _extract_call_args(fn, args, kwargs)
            result_str = fn(*args, **kwargs)
            entry = {
                "tool": fn.__name__,
                "args": call_args,
                "result": _try_parse(result_str),
                "timestamp": datetime.now(UTC).isoformat(),
            }
            with open(_ACTION_LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
            return result_str

        return wrapper

    # -- Action log retrieval / management tools -------------------------------

    @mcp.tool()
    def get_action_log() -> str:
        """Retrieve the action log of all tool calls made so far.

        Returns the complete log and clears it for the next case.
        Each entry has: tool, args, result, timestamp.
        """
        entries = []
        if _ACTION_LOG_PATH.exists():
            for line in _ACTION_LOG_PATH.read_text().strip().split("\n"):
                if line:
                    entries.append(json.loads(line))
            _ACTION_LOG_PATH.unlink()
        return json.dumps(entries)

    # -- Utility tool: list available case IDs ---------------------------------

    @mcp.tool()
    @_logged
    def list_cases() -> str:
        """List all available case IDs in the current dataset."""
        return json.dumps(store.list_case_ids())

    # -- Precomputed data tools ------------------------------------------------

    for tool_spec in tools:
        _register_precomputed_tool(mcp, store, tool_spec, _logged)

    # -- Knowledge retrieval (RAG) ---------------------------------------------

    guidelines_search = _load_guidelines_search(resource_dir)

    @mcp.tool()
    @_logged
    def search_guidelines(query: str) -> str:
        """Search clinical guidelines and protocols relevant to the query.

        Uses semantic similarity search over a knowledge base of clinical
        guidelines (e.g. EAU guidelines, NCCN protocols).
        Returns the most relevant guideline passages.
        """
        if guidelines_search is None:
            return json.dumps(
                {
                    "query": query,
                    "results": [],
                    "note": "Guidelines DB not available. Run: make process-guidelines",
                }
            )
        results = guidelines_search.query(query)
        return json.dumps({"query": query, "results": results})

    # -- Agent Skills (progressive disclosure) ---------------------------------

    loaded_skills = _load_skills(skills_dir)

    @mcp.tool()
    @_logged
    def load_skill(name: str) -> str:
        """Load the full instructions for a named skill.

        Call this before using a skill to get its detailed instructions,
        examples, and guidelines. Available skills are listed in the
        system prompt.
        """
        if name not in loaded_skills:
            available = list(loaded_skills.keys()) if loaded_skills else []
            return json.dumps({"error": f"Skill '{name}' not found. Available: {available}"})
        return loaded_skills[name]["body"]

    return mcp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_call_args(fn, args: tuple, kwargs: dict) -> dict:
    """Extract a clean dict of the arguments passed to a tool function."""
    import inspect

    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    result = {}
    for i, val in enumerate(args):
        if i < len(params):
            result[params[i]] = val
    result.update(kwargs)
    return result


def _try_parse(text: str):
    """Try to parse a JSON string, return as-is on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return text


def _load_skills(skills_dir: str | None) -> dict[str, dict]:
    if skills_dir is None:
        return {}
    try:
        from chimera_agent_baseline.skills import load_skills

        return load_skills(skills_dir)
    except Exception:
        log.warning("Failed to load skills from %s", skills_dir, exc_info=True)
        return {}


def _load_guidelines_search(resource_dir: str | None):
    if resource_dir is None:
        return None
    try:
        from chimera_agent_baseline.rag import GuidelinesSearch

        return GuidelinesSearch(resource_dir)
    except FileNotFoundError:
        log.info("Guidelines DB not found in %s — search_guidelines will return empty results", resource_dir)
        return None
    except Exception:
        log.warning("Failed to load guidelines search", exc_info=True)
        return None


def _register_precomputed_tool(
    mcp: FastMCP,
    store: CaseDataStore,
    spec: ToolSpec,
    logged,
) -> None:
    """Register a single precomputed-data tool on the MCP server."""
    field_mapping = spec.field_mapping

    def tool_fn(case_id: str) -> str:
        try:
            result = store.extract(case_id, field_mapping)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})

        if not result or set(result.keys()) == {"case_id"}:
            return json.dumps({"case_id": case_id, "note": "No data available for this tool and case."})

        return json.dumps(result)

    tool_fn.__name__ = spec.name
    tool_fn.__doc__ = spec.description
    tool_fn.__annotations__ = {"case_id": str, "return": str}

    mcp.tool()(logged(tool_fn))


# ---------------------------------------------------------------------------
# CLI entry-point (stdio transport)
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Chimera MCP tool server")
    parser.add_argument("--data-dir", required=True, help="Path to directory containing clinical-data.json")
    parser.add_argument("--resource-dir", default=None, help="Path to resources directory (guidelines_db/, etc.)")
    parser.add_argument("--skills-dir", default=None, help="Path to skills directory")
    parser.add_argument("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    setup_logging(args.log_level)
    server = create_server(args.data_dir, resource_dir=args.resource_dir, skills_dir=args.skills_dir)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
