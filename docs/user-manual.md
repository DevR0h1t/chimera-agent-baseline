# User Manual

## Requirements

| Requirement | Why |
|---|---|
| Python 3.12+ | Language runtime |
| [uv](https://docs.astral.sh/uv/) | Package manager |
| NVIDIA GPU ≥ 16 GB VRAM | vLLM in-process inference (Gemma 4 ≈ 10 GB) |
| CUDA 12.x + drivers | Required by vLLM |
| Docker + NVIDIA Container Toolkit | For `make gc-test` |
| ~15 GB disk | Model weights + guidelines DB + embedding model |
| HuggingFace token | For gated models (Gemma 4, embeddinggemma) |
| `make` | Build automation |

## Setup

```bash
git clone <repo-url> && cd chimera-agent-challenge
uv venv && source .venv/bin/activate
make install
cp .env.example .env                    # add HF_TOKEN
```

Accept licenses on
[Gemma 4](https://huggingface.co/google/gemma-4-E2B-it) and
[embeddinggemma](https://huggingface.co/google/embeddinggemma-300m),
then download the LLM and rebuild the guidelines DB:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-E2B-it', local_dir='model/')"
make process-guidelines
make test
```

## Common commands

```bash
make run                                                    # task 1
make run RUN_ARGS="agent.tool_registry=task2 \
    paths.input_dir=outputs/agent_input/task2"              # task 2
make run RUN_ARGS="agent.limit=5"                           # smoke 5 cases
make run RUN_ARGS="+experiment=qwen_local"                  # swap model

python scripts/run_all_tasks.py                             # task 1 then 2

make lint
make format

make gc-build
make gc-test                                                # docker, task 1
make gc-test GC_INPUT_DIR=outputs/agent_input/task2         # docker, task 2
make gc-save                                                # export tarballs
```

First `make run` takes ~30 s for vLLM to compile CUDA graphs (cached
after).

## Project layout

```
src/chimera_agent_baseline/
├── agent/
│   ├── graph.py              ReAct + form-fill wiring (~85 LoC)
│   ├── form_fill.py          Terminal node: prompt + parse against output schema
│   └── prompts.py            System prompt (modify to change agent behaviour)
├── output/
│   └── schema.py             Pydantic submission contract (Task1Output / Task2Output)
├── models/
│   ├── __init__.py           Provider switch (vllm / openai)
│   ├── vllm_offline.py       vLLM in-process wrapper
│   └── openai_compat.py      ChatOpenAI subclass that surfaces reasoning_content
├── tools/
│   ├── base.py               ToolSpec + CaseDataStore
│   └── definitions.py        TASK1_TOOLS / TASK2_TOOLS (add custom tools here)
├── case_loader.py            Reads <pid>/prompt.json into agent queries
├── mcp_server.py             MCP server + action log
├── rag.py                    Guidelines search (ChromaDB + embeddings)
├── run.py                    Local Hydra entry-point
└── utils.py                  Logging

configs/config.yaml           Defaults (paths, model, generation, agent)
configs/experiment/           Overlays via +experiment=<name>
templates/prompts/            Jinja templates for the agent prompt
resources/                    Guidelines DB + embedding model + GC config copy
outputs/agent_input/task{1,2}/<pid>/{prompt,tools}.json   Per-case inputs
inference.py                  Grand Challenge container entrypoint
```

## What to modify

| Goal | File | How |
|---|---|---|
| Change the system prompt | `agent/prompts.py` | Edit `SYSTEM_PROMPT` |
| Edit the per-case prompt | `templates/prompts/agent_prompt.j2` | Jinja2 |
| Add a clinical tool | `tools/definitions.py` | Append `ToolSpec` to `TASK1_TOOLS` / `TASK2_TOOLS` |
| Add a non-data tool | `mcp_server.py` | New `@mcp.tool()` function |
| Swap the LLM (vLLM) | `configs/config.yaml` | `model.model_id` + `model.tool_parser` |
| Swap the LLM (OpenAI-compatible) | `configs/experiment/*.yaml` | `+experiment=<name>` |
| Change the agent loop | `agent/graph.py` | Edit graph nodes / router |
| Tune the form-fill prompt / retry | `agent/form_fill.py` | Skeleton + retry logic — but keep the schema |
| Rebuild the RAG corpus | `scripts/process_guidelines.py` | Pass your PDF |

## What NOT to modify

- `output/schema.py` — submissions whose final output does not validate
  against `Task1Output` / `Task2Output` are rejected. Add tools, change
  prompts, swap orchestration, but keep this shape.
- `mcp_server.py` action-log layer — the log is part of the evaluation.
- `inference.py` — the GC container entry-point's external interface.

## Switching models

See [docs/models.md](models.md). One-liner via experiment overlay:

```bash
make run RUN_ARGS="+experiment=qwen_local"
```

## Container workflow

```bash
make gc-build
make gc-test                                                # local sanity check
make gc-save                                                # export tarballs
```

See the
[Grand Challenge documentation](https://grand-challenge.org/documentation/)
for container conventions.

## Troubleshooting

- **CUDA OOM**: lower `generation.max_model_len` (default 32768) — try 8192.
- **Guidelines DB missing**: `make process-guidelines` (needs `HF_TOKEN`).
- **HF 403 Forbidden**: accept the model license on HuggingFace.
- **vLLM compile takes long on first run**: ~30 s, cached in `~/.cache/vllm/`.
- **`make gc-test` permission error on `/output`**: `chmod 777 test/output`.
