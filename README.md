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

Run the agent (NVIDIA GPU with ≥16 GB VRAM):

```bash
make run                                                     # task 1
make run RUN_ARGS="agent.tool_registry=task2 \
    paths.input_dir=outputs/agent_input/task2"               # task 2
make run RUN_ARGS="agent.limit=5"                            # 5 cases
```

Per-case predictions land in `test/output/predictions/task<N>/<pid>.json`;
the aggregate is `test/output/predictions.json`.

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
make gc-test GC_INPUT_DIR=outputs/agent_input/task2           # task 2
```

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

The structured submission contract lives in
`src/chimera_agent_baseline/output/schema.py`. Submissions whose final
output does not validate against `Task1Output` / `Task2Output` are
rejected. Swap models, tools, prompts, and orchestration freely — keep
the shape.
