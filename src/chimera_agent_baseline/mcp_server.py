"""MCP server exposing the per-task clinical tool registry.

Runs as a stdio-based MCP server, framework-agnostic (any MCP client
works — the runner uses :mod:`langchain_mcp_adapters`).

Usage::

    python -m chimera_agent_baseline.mcp_server \\
        --data-dir outputs/agent_input/task1 --resource-dir resources \\
        --tool-registry task1

.. note::

    The action-log layer — every tool call is recorded with ``tool``,
    ``args``, ``result``, and ``timestamp`` — is part of the challenge
    contract and powers faithfulness evaluation. Add new tools, edit
    existing ones, swap the registry; just keep the action log
    intact. Submissions whose action log has been disabled or
    tampered with will be rejected.
"""

import argparse
import json
import logging

from mcp.server.fastmcp import FastMCP

from chimera_agent_baseline.tools.base import CaseDataStore, ToolSpec
from chimera_agent_baseline.tools.definitions import TASK1_TOOLS, TASK2_TOOLS
from chimera_agent_baseline.utils import setup_logging

_REGISTRIES: dict[str, list[ToolSpec]] = {
    "task1": TASK1_TOOLS,
    "task2": TASK2_TOOLS,
}

log = logging.getLogger(__name__)


def create_server(
    data_dir: str,
    resource_dir: str | None = None,
    tools: list[ToolSpec] | None = None,
    name: str = "Chimera Tools",
) -> FastMCP:
    """Create an MCP server with precomputed clinical tools."""
    mcp = FastMCP(name)
    store = CaseDataStore(data_dir)
    tools = tools if tools is not None else TASK1_TOOLS

    # -- Precomputed data tools ------------------------------------------------

    for tool_spec in tools:
        _register_precomputed_tool(mcp, store, tool_spec)

    # -- Knowledge retrieval (RAG) ---------------------------------------------

    guidelines_search = _load_guidelines_search(resource_dir)

    @mcp.tool()
    def search_guidelines(query: str) -> str:
        """Search clinical guidelines and protocols relevant to the query.

        Uses semantic similarity search over a knowledge base of clinical
        guidelines (e.g. EAU prostate-cancer guidelines, NCCN protocols).
        Returns the most relevant guideline passages. Free-text query;
        phrase as a clinical question.
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

    return mcp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _register_precomputed_tool(mcp: FastMCP, store: CaseDataStore, spec: ToolSpec) -> None:
    """Register a single precomputed-data tool on the MCP server."""
    fields = spec.fields

    def tool_fn(case_id: str) -> str:
        try:
            result = store.extract(case_id, fields)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})

        if set(result.keys()) == {"case_id"}:
            return json.dumps({"case_id": case_id, "note": "No data available for this tool and case."})

        return json.dumps(result)

    tool_fn.__name__ = spec.name
    tool_fn.__doc__ = spec.description
    tool_fn.__annotations__ = {"case_id": str, "return": str}

    mcp.tool()(tool_fn)


# ---------------------------------------------------------------------------
# CLI entry-point (stdio transport)
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Chimera MCP tool server")
    parser.add_argument("--data-dir", required=True, help="Path to per-case agent input directory")
    parser.add_argument("--resource-dir", default=None, help="Path to resources directory (guidelines_db/, etc.)")
    parser.add_argument(
        "--tool-registry",
        choices=sorted(_REGISTRIES),
        default="task1",
        help="Which ToolSpec registry to expose. 'task1' = biopsy-decision "
        "tools (PSA trend, labs, MRI report, pathology report, previous "
        "notes, family history); 'task2' = treatment-decision tools "
        "(PSA trend + labs dropped, pathology returns richer per-core data).",
    )
    parser.add_argument("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    setup_logging(args.log_level)
    server = create_server(args.data_dir, resource_dir=args.resource_dir, tools=_REGISTRIES[args.tool_registry])
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
