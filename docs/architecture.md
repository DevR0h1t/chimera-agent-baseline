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
  │ (LLM)        │  validated via PydanticOutputParser; retries on
  │              │  validation error (agent.form_fill.max_retries)
  └──────┬───────┘
         ▼
       END  →  structured_response (the validated per-case record)
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

Two registries, selected via the Hydra config key
`agent.tool_registry=task1|task2`:

Both tasks expose the same six tools — the masked "Extended EHR view"
documents the urologist could reveal — plus guideline search:

| Tool | Task 1 | Task 2 | Returns |
|---|---|---|---|
| `get_psa_trend` | ✓ | ✓ | Prior PSA values as a time series |
| `get_lab_results` | ✓ | ✓ | Full lab panel |
| `get_mri_report` | ✓ | ✓ | mpMRI report prose |
| `get_pathology_report` | ✓ | ✓ | Biopsy report prose (task 1: "no data" when no prior biopsy) |
| `get_previous_notes` | ✓ | ✓ | Prior GP / urology notes |
| `get_family_history` | ✓ | ✓ | First-degree PCa history |
| `search_guidelines` | ✓ | ✓ | Semantic search over EAU corpus |

The structured headline values (PI-RADS, PSA density, volume, csPCa,
Gleason / ISUP, …) are in `prompt.json` up front, so the tools serve only
the free-text documents behind them.

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

`fields` lists the top-level keys from each case's `clinical.json` that
this tool should return. Missing keys are silently omitted (so a
biopsy-naïve case calling `get_pathology_report` returns just
`{case_id}` plus a "no data" note). For tools that don't follow the
precomputed-data pattern (e.g. an API call), register them directly in
`mcp_server.py` with the `@mcp.tool()` decorator (see
`search_guidelines`).

### What you can change

You're free to **add** tools, change a tool's `description`, or
expand its `fields`. Don't rename or remove the existing baseline
tools — `output/schema.py` (locked) maps each rateable
reasoning-variable to the tool that backs it (e.g. `pirads →
get_mri_report`), and renaming a tool silently makes those variables
un-rateable.

## Tool use is not audited

The participant container is a black box, so tool calls cannot be
enforced or recorded by the harness. Only the final structured output is
evaluated — there is no action log. The MCP tools exist to let the agent
retrieve the masked EHR documents; using them well is in the
participant's interest, not a graded artifact.

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

Each case writes a single file,
`test/output/task<N>/<case_id>/prediction.json`, which is exactly the
validated structured record (no extra trace/log fields):

| Field | Tasks | Purpose |
|---|---|---|
| `case_id`, `task` | 1, 2, 3 | Identifiers |
| `biopsy_decision` (bool) | 1 | The decision to be evaluated |
| `action` (1 of 4) | 2 | The decision to be evaluated |
| `months_to_recurrence` (float) | 3 | The numeric prognosis |
| `confidence` (clear/borderline/uncertain) | 1, 2 | Decision confidence |
| `variable_weights` (not_used/noted/important/decisive per variable) | 1, 2 | Reasoning capture |
| `reasoning` (≥40 chars) | 1, 2, 3 | Free-text reasoning |

`form_fill_warnings` (validation retries) are logged, not written to the
output file.
