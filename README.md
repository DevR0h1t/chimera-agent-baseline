<p align="center">
  <img src="docs/images/thumbnail.png" alt="CHIMERA Agent" width="250">
</p>

# Chimera Agent Baseline

Baseline agent for the
[CHIMERA-Agent challenge](https://chimera-agent.grand-challenge.org/chimera-agent/).
A LangGraph ReAct loop calls clinical tools served via MCP, retrieves
guidelines via RAG, and emits a structured per-case decision through a
terminal form-fill node.

## Quick start

```bash
uv venv && source .venv/bin/activate
cp .env.example .env                    # add HF_TOKEN
make install
make test
```

Download the LLM and embedding model (requires accepted licenses on
[Gemma 4](https://huggingface.co/google/gemma-4-E2B-it) and
[embeddinggemma](https://huggingface.co/google/embeddinggemma-300m)):

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-E2B-it', local_dir='model/')"
make process-guidelines    # builds resources/guidelines_db (~1.2 GB)
```

Get the per-case agent inputs (`prompt.json` + `tools.json` per
patient, organised by task):

```bash
# TODO: replace with the actual download once published.
# Expected to extract into outputs/agent_input/task1/ and
# outputs/agent_input/task2/, each with one PT-XXXX subdirectory per case.
```

Run the agent (NVIDIA GPU with ≥16 GB VRAM):

```bash
make run                                                     # task 1
make run RUN_ARGS="agent.tool_registry=task2 \
    paths.input_dir=outputs/agent_input/task2"               # task 2
make run RUN_ARGS="agent.limit=5"                            # 5 cases
```

Per-case predictions land in `test/output/predictions/task<N>/<pid>.json`.

## Layout

| Local path | Container path | Contents |
|---|---|---|
| `outputs/agent_input/task{1,2}/<pid>/{prompt,tools}.json` | `/input` | Per-case agent inputs |
| `test/output/` | `/output` | Predictions written by the agent |
| `model/` | `/opt/ml/model` | LLM weights (gitignored) |
| `resources/` | `/opt/app/resources` | Config, guidelines DB, embedding model |

The participant ships `prompt.json`; `tools.json` is read only by the MCP
server and reaches the agent through tool calls.

To test in the GC Docker container:

```bash
make gc-build
make gc-test                                                  # task 1
make gc-test INPUT=outputs/agent_input/task2                  # task 2
```

## Per-case I/O

Each case is a directory under `/input` (read-only) containing two files
the harness already knows how to read. You write one JSON per case to
`/output/predictions/task<N>/<case_id>.json`, mirroring the input tree.

```
/input/<case_id>/prompt.json   # patient context rendered into the agent prompt
/input/<case_id>/tools.json    # served by the MCP server through tool calls
/output/predictions/task<N>/<case_id>.json   # per-case prediction (Task1Output / Task2Output)
```

### Inputs

`prompt.json` — patient context the agent always sees (no tool call
needed). Same shape both tasks; task-2 additionally surfaces `labs` and
`psa_trend` here, since the urologist arrives at the MDT with them:

| Key | Type | Tasks | Notes |
|---|---|---|---|
| `case_id` | str | 1, 2 | matches the directory name |
| `task` | str | 1, 2 | `mri_diagnostic` (1) or `risk_stratification` (2) |
| `query` | str | 1, 2 | one-sentence ask |
| `encounter` | object | 1, 2 | `department`, `date`, `referrer`, `type` |
| `current_psa` | object | 1, 2 | `value` (float, ng/mL), `date` |
| `patient` | object | 1, 2 | `age`, `vitals`, `allergies`, `medication`, `pmhx`, `social`, `lifestyle` |
| `note_sections` | list | 1, 2 | `[{s: "Chief complaint", t: "..."}, ...]` |
| `recent_other` | list[str] | 1, 2 | recent unrelated appointments |
| `admin` | str | 1, 2 | free-text admin note |
| `labs` | list | 2 only | `[{name, value, unit, ref, date}, ...]` |
| `psa_trend` | list | 2 only | `[{date, value}, ...]` |

`tools.json` — payloads the MCP server returns when the agent calls a
tool. The agent only sees these fields if it actually invokes the
matching tool (see [docs/architecture.md](docs/architecture.md) for the
tool→field mapping).

| Key | Type | Tasks | Served by |
|---|---|---|---|
| `case_id` | str | 1, 2 | echoed by every tool |
| `imaging_report` | str | 1, 2 | `get_mri_report` |
| `pirads`, `prostate_volume`, `psa_density`, `cspca_pred` | str / float | 1, 2 | `get_mri_report` |
| `pathology_report` | str | 1, 2 | `get_pathology_report` |
| `biopsies`, `prior_biopsy` | list / str | 1, 2 | `get_pathology_report` |
| `bx_gl_prim`, `bx_gl_sec`, `bx_gl_tert`, `bx_isup`, `bx_isup_pred`, `cores_positive`, `cores_total`, `max_core_pct`, `lvi`, `pni`, `growth_pattern`, `high_risk_patterns`, `tumor_location`, `ct` | mixed | 2 only | `get_pathology_report` (richer task-2 panel) |
| `previous_notes` | list | 1, 2 | `get_previous_notes` |
| `family_history` | str | 1, 2 | `get_family_history` |
| `psa_trend` | list | 1 only | `get_psa_trend` |
| `labs` | list | 1 only | `get_lab_results` |

The split between `prompt.json` and `tools.json` reflects what a
clinician would have at a glance versus what they'd actively pull up.
The split is *per task*, not per field: labs and PSA history live in
`tools.json` for Task 1 (the agent has to call `get_lab_results` /
`get_psa_trend` to see them) but in `prompt.json` for Task 2 (already
in hand at the MDT). For any single case, each field appears in
exactly one of the two files.

To **add** a clinical tool, register a new `ToolSpec` in
`src/chimera_agent_baseline/tools/definitions.py` and append it to
`TASK1_TOOLS` / `TASK2_TOOLS` — the new tool's `fields` map to one or
more `tools.json` keys.

Do **not** rename or remove the existing tools. The reasoning-variable
schema in `output/schema.py` (locked) maps each rateable variable to
the tool that backs it (e.g. `pirads → get_mri_report`); renaming the
tool silently makes those variables un-rateable. Editing a tool's
description or expanding its `fields` is fine.

### Outputs

The structured submission contract is the Pydantic models in
`src/chimera_agent_baseline/output/schema.py` (`Task1Output`,
`Task2Output`). Outputs that don't validate are rejected.

```jsonc
// /output/predictions/task1/PT-XXXX.json
{
  "case_id": "PT-XXXX",
  "task": "mri_diagnostic",
  "biopsy_recommendation": true,           // bool
  "repeat_test": null,                     // str | null
  "confidence": "Borderline",              // Clear | Borderline | Uncertain
  "decision_summary": "...",               // ≥40 chars, names 2-4 driving factors
  "variable_ratings": {                    // one per TASK1_VARIABLES key
    "psa": {"rating": "Important", "reasoning": "..."},
    "pirads": {"rating": "Decisive", "reasoning": "..."}
    // ...
  },
  "reasoning_trace": "...",                // last assistant message
  "thinking_trace": [...],                 // per-turn reasoning_content (Qwen3, o1, ...)
  "action_log": [...],                     // every tool call (faithfulness eval)
  "form_fill_warnings": []
}
```

Task-2 replaces `biopsy_recommendation` / `repeat_test` with a single
`treatment_recommendation` (`active_surveillance` |
`radical_prostatectomy` | `radiotherapy` | `focal_therapy` |
`hormonal_therapy` | `watchful_waiting`) and uses the task-2 variable
set.

### Docker invocation

The Grand Challenge container reads `/input` and writes `/output`:

```bash
docker run --rm --gpus all \
    -v $PWD/outputs/agent_input/task1:/input:ro \
    -v $PWD/test/output:/output \
    chimera-agent-baseline
```

`make gc-test` is exactly this with the housekeeping (cleans
`test/output/`, sets perms, tags the image). The `:ro` on `/input`
matches the GC platform — your container must not write to it.

## Documentation

| Document | What it covers |
|---|---|
| [User Manual](docs/user-manual.md) | Setup, commands, troubleshooting |
| [Architecture](docs/architecture.md) | Components, ReAct + form-fill graph |
| [Model Configuration](docs/models.md) | Swapping models, providers, experiment overlays |
| [Challenge Tasks](docs/chimera.md) | Task definitions, inputs, outputs |

## What to change

| Goal | Where |
|---|---|
| Change the system prompt | `src/chimera_agent_baseline/agent/prompts.py` |
| Edit the case prompt template | `templates/prompts/agent_prompt.j2` |
| Add a clinical tool | `src/chimera_agent_baseline/tools/definitions.py` (`TASK1_TOOLS` / `TASK2_TOOLS`) |
| Swap the LLM (vLLM) | `configs/config.yaml` → `model.model_id`, `model.tool_parser` |
| Swap the LLM (OpenAI-compatible) | `configs/experiment/qwen_local.yaml` (`+experiment=qwen_local`) |
| Change the agent loop | `src/chimera_agent_baseline/agent/graph.py` |
| Tune the form-fill prompt / retry | `src/chimera_agent_baseline/agent/form_fill.py` |
| Rebuild the RAG corpus | `scripts/process_guidelines.py` |

## What NOT to change

Two pieces are part of the challenge contract — submissions that
violate either are rejected. Everything else (system prompt, tools,
models, agent graph, form-fill node, configs, even the entry-point if
you want) is fair game.

| File | Why it's locked |
|---|---|
| `src/chimera_agent_baseline/output/schema.py` | The Pydantic submission shape. The final JSON must validate against `Task1Output` / `Task2Output`. |
| `src/chimera_agent_baseline/mcp_server.py` (action-log layer) | Every tool call is logged with `tool`, `args`, `result`, `timestamp` — used for faithfulness evaluation. Add tools, swap registries, but keep the log intact. |
