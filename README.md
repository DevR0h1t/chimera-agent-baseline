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

Get the per-case agent inputs (`prompt.json` + `clinical.json` per
patient, organised by task):

```bash
# TODO: replace with the actual download once published.
# Expected to extract into data/task1/agent_input/ and
# data/task2/agent_input/, each with one case subdirectory per patient.
```

Run the agent (NVIDIA GPU with ≥16 GB VRAM):

```bash
make run                                                     # task 1
make run RUN_ARGS="agent.tool_registry=task2 \
    paths.input_dir=data/task2/agent_input"                  # task 2
make run RUN_ARGS="agent.limit=5"                            # 5 cases
```

Per-case predictions land in `test/output/task<N>/<case_id>/prediction.json`.

## Layout

| Local path | Container path | Contents |
|---|---|---|
| `data/task{1,2}/agent_input/<case>/{prompt,clinical}.json` | `/input` | Per-case agent inputs |
| `test/output/` | `/output` | Predictions written by the agent |
| `model/` | `/opt/ml/model` | LLM weights (gitignored) |
| `resources/` | `/opt/app/resources` | Config, guidelines DB, embedding model |

The participant ships `prompt.json`; `clinical.json` is read only by the
MCP server and reaches the agent through tool calls.

To test in the GC Docker container:

```bash
make gc-build
make gc-test                                                  # task 1
make gc-test INPUT=data/task2/agent_input                     # task 2
```

## Per-case I/O

Each case is a directory under `/input` (read-only) containing two files
the harness already knows how to read. You write one JSON per case to
`/output/task<N>/<case_id>/prediction.json`, mirroring the input tree.

```
/input/<case_id>/prompt.json    # patient context rendered into the agent prompt
/input/<case_id>/clinical.json  # served by the MCP server through tool calls
/output/task<N>/<case_id>/prediction.json   # the prediction (Task1Output / Task2Output / Task3Output)
```

The two files mirror the urologist forms: `prompt.json` is the read-only
"Clinical data" panel the urologist saw up front, and `clinical.json`
holds the form's masked "Extended EHR view" documents — revealed only on
request, i.e. via MCP tool calls.

### Inputs

`prompt.json` — the flat structured record the agent always sees (no tool
call needed): identifiers, encounter, headline values, and the structured
clinical panel. Same shape both tasks; task 2 adds the biopsy fields.

| Key | Type | Tasks | Notes |
|---|---|---|---|
| `case_id` | str | 1, 2 | matches the directory name |
| `task` | int | 1, 2 | `1` (biopsy decision) or `2` (treatment decision) |
| `psa`, `age`, `months` | num | 1, 2 | headline PSA, age, months since last PSA |
| `pirads`, `psad`, `psav`, `psap`, `vol`, `cspca` | mixed | 1, 2 | PI-RADS, PSA density / velocity / prior, prostate volume, csPCa prob |
| `dre`, `bx` | str | 1, 2 | DRE findings, prior-biopsy bucket |
| `medhx`, `meds`, `notes`, `pmhx`, `allergies` | mixed | 1, 2 | medical history, medication, summary notes |
| `vitals` | object | 1, 2 | `weight`, `height`, `bmi`, `bp`, `hr`, `smoking`, … |
| `enc_dept`, `enc_date`, `enc_ref`, `enc_type` | str | 1, 2 | encounter |
| `note_sections` | list | 1, 2 | `[{s: "Chief complaint", t: "..."}, ...]` |
| `occupation`, `marital`, `living`, `next_of_kin`, `alcohol`, `exercise`, `ipss`, `recent_other`, `admin` | mixed | 1, 2 | social / lifestyle / admin |
| `ct`, `bx_isup`, `bx_gl_prim`, `bx_gl_sec`, `bx_gl_tert` | mixed | 2 | clinical stage + biopsy Gleason / ISUP |

There is no `query` field — the task question is fixed per task in the
prompt template (`templates/prompts/agent_prompt.j2`).

`clinical.json` — the documents the MCP server returns when the agent
calls a tool. The agent only sees these if it actually invokes the
matching tool (see [docs/architecture.md](docs/architecture.md)).

| Key | Type | Tasks | Served by |
|---|---|---|---|
| `case_id` | str | 1, 2 | echoed by every tool |
| `radiology_report` | str | 1, 2 | `get_mri_report` |
| `pathology_report` | str | 2 | `get_pathology_report` (task 1: no prior pathology → "no data") |
| `previous_notes` | list | 1, 2 | `get_previous_notes` |
| `laboratory_results` | list | 1, 2 | `get_lab_results` |
| `psa_trend` | list | 1, 2 | `get_psa_trend` |
| `family_history` | str | 1, 2 | `get_family_history` |

Both tasks expose the same six tools. To **add** a clinical tool,
register a new `ToolSpec` in
`src/chimera_agent_baseline/tools/definitions.py` and append it to
`TASK1_TOOLS` / `TASK2_TOOLS` — the new tool's `fields` map to one or
more `clinical.json` keys.

### Outputs

The structured submission contract is the Pydantic models in
`src/chimera_agent_baseline/output/schema.py` (`Task1Output`,
`Task2Output`, `Task3Output`). Each is a single JSON file per case;
outputs that don't validate are rejected. Enum tokens are lowercase, to
match the urologist forms.

```jsonc
// /output/task1/PT-XXXX/prediction.json
{
  "case_id": "PT-XXXX",
  "task": 1,
  "biopsy_decision": true,                 // bool
  "confidence": "borderline",              // clear | borderline | uncertain
  "variable_weights": {                    // one per TASK1_VARIABLES key
    "psa": "important",                    // not_used | noted | important | decisive
    "pirads": "decisive"
    // ...
  },
  "reasoning": "..."                       // ≥40 chars, names 2-4 driving factors
}
```

Task 2 replaces `biopsy_decision` with a single `action` — one of
`active_surveillance` | `continued_surveillance` | `watchful_waiting` |
`active_treatment` — and uses the task-2 variable set. Task 3 is a
numeric prognosis: `months_to_recurrence` (float) + `reasoning` only (no
weights / confidence).

Task 3 (recurrence prognosis) uses a much-simplified `prompt.json` —
`case_id`, `task`, `age`, `psa`, and `dre` (the free-text physical
examination) — and a 5-tool `clinical.json`: `radiology_report`,
`pathology_report` (biopsy), `surgical_pathology_report`,
`previous_notes`, `family_history`. The loader, tools, prompt template,
and output contract all handle task 3; only the task-3 case data is not
yet shipped.

### Docker invocation

The Grand Challenge container reads `/input` and writes `/output`:

```bash
docker run --rm --gpus all \
    -v $PWD/data/task1/agent_input:/input:ro \
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

One piece is part of the challenge contract — submissions that violate it
are rejected. Everything else (system prompt, tools, models, agent graph,
form-fill node, configs, even the entry-point if you want) is fair game.
The participant container is a black box, so tool use is not enforced or
audited — only the final structured output is evaluated.

| File | Why it's locked |
|---|---|
| `src/chimera_agent_baseline/output/schema.py` | The Pydantic submission shape. The final JSON must validate against `Task1Output` / `Task2Output` / `Task3Output`. |
