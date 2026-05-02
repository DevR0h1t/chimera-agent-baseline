# Architecture

## Graph

```
HumanMessage(prompt)
        │
        ▼
  ┌──────────┐  tool_calls    ┌──────────┐
  │  agent   │───────────────▶│  tools   │  (MCP, stdio)
  │  (LLM)   │◀───────────────│          │
  └────┬─────┘   results      └──────────┘
       │ no tool_calls
       ▼
  ┌──────────────┐
  │ form_fill    │  same model + per-case Pydantic schema
  │ (LLM)        │  validated via PydanticOutputParser, retried up to 3×
  └──────┬───────┘
         ▼
       END  →  structured_response + reasoning_trace + action_log
```

The same model is used for both ReAct turns and the terminal form-fill;
the form-fill node prompts it with a concrete JSON skeleton (not a JSON
schema dump — small models echo schemas back). This is provider-neutral:
any LangChain `BaseChatModel` works.

**Source**: `src/chimera_agent_baseline/agent/graph.py`,
`src/chimera_agent_baseline/run.py`.

## MCP tool server

All tools are exposed via the [Model Context Protocol](https://modelcontextprotocol.io).
The server runs as a stdio subprocess and is framework-agnostic.

**Source**: `src/chimera_agent_baseline/mcp_server.py`.

Two registries, selected at startup via `--tool-registry task1|task2`:

| Tool | Task 1 | Task 2 | Returns |
|---|---|---|---|
| `get_psa_trend` | ✓ | — | Prior PSA values as a time series |
| `get_lab_results` | ✓ | — | Full lab panel |
| `get_mri_report` | ✓ | ✓ | mpMRI prose + PI-RADS / PSAD / volume / csPCa |
| `get_pathology_report` | ✓ | ✓ (richer) | Biopsy report + Gleason / ISUP / GP4 |
| `get_previous_notes` | ✓ | ✓ | Prior GP / urology notes |
| `get_family_history` | ✓ | ✓ | First-degree PCa history |
| `search_guidelines` | ✓ | ✓ | Semantic search over EAU corpus |
| `get_action_log` | runner-only | runner-only | The append-only action log |

For Task 2, lab panel and PSA trend are surfaced in the prompt context
up front (the urologist arrives at the MDT with them), so the matching
tools are dropped from the registry.

### Adding a custom tool

Define a `ToolSpec` in `src/chimera_agent_baseline/tools/definitions.py`
and append it to `TASK1_TOOLS` (or `TASK2_TOOLS`):

```python
MY_TOOL = ToolSpec(
    name="get_my_data",
    description="Retrieve my custom data for a patient case.",
    fields=("my_field", "another_field"),
)
```

`fields` lists the top-level keys from each case's `tools.json` that
this tool should return. Missing keys are silently omitted (so a
biopsy-naïve case calling `get_pathology_report` returns just
`{case_id}` plus a "no data" note). For tools that don't follow the
precomputed-data pattern (e.g. an API call), register them directly in
`mcp_server.py` with the `@mcp.tool()` decorator (see
`search_guidelines`).

## Action log

The MCP server logs every tool call with `tool`, `args`, `result`,
`timestamp`. The runner calls `get_action_log` per case to retrieve and
clear it. This is used for faithfulness evaluation — verifying the
agent's reasoning trace references evidence it actually fetched.

## RAG (search_guidelines)

Clinical guidelines are chunked, embedded with
`google/embeddinggemma-300m`, and persisted to ChromaDB at
`resources/guidelines_db/`. The embedding model runs in a separate
process on CPU (sentence-transformers via UDS), so it does not compete
with the agent LLM for GPU memory.

To rebuild with different guidelines:

```bash
python scripts/process_guidelines.py --pdf path/to/your/guidelines.pdf
```

## Output

Each case writes
`test/output/predictions/task<N>/<pid>.json` with:

| Field | Source | Purpose |
|---|---|---|
| Decision fields (`biopsy_recommendation`, `cspca_probability_self`, `treatment_recommendation`, …) | form-fill node | Scored numerically |
| `confidence`, `decision_summary`, `variable_ratings` | form-fill node | Reasoning capture |
| `reasoning_trace` | last assistant message | Qualitative review |
| `thinking_trace[]` | `additional_kwargs.reasoning_content` | Captured for models that emit chain-of-thought separately (Qwen3+, OpenAI o1) |
| `action_log[]` | MCP server | Faithfulness verification |
| `form_fill_warnings[]` | form-fill node | Validation retries / post-hoc downgrades |
