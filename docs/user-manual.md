# User Manual

Reference companion to the [README](../README.md). The README walks you
through quick start, I/O, and Docker; this page covers requirements,
the full project layout, and troubleshooting.

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

## Useful commands beyond the README

```bash
make run-all                                                # task 1 then 2 in one go
make run RUN_ARGS="+experiment=qwen_local"                  # swap LLM via overlay
make lint
make format
make gc-save                                                # export image + model tarballs
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
data/task{1,2}/agent_input/<case>/{prompt,clinical}.json  Per-case inputs
inference.py                  Grand Challenge container entrypoint
```

## What NOT to modify

See the README's "What NOT to change" table — one locked file:
`output/schema.py` (the submission schema). Tool use is not audited (the
participant container is a black box; only the final structured output is
evaluated), so everything else — including `inference.py`, the tools, and
the agent graph — is fair game.

## Troubleshooting

- **CUDA OOM**: lower `generation.max_model_len` (default 32768) — try 8192.
- **Guidelines DB missing**: `make process-guidelines` (needs `HF_TOKEN`).
- **HF 403 Forbidden**: accept the model license on HuggingFace.
- **vLLM compile takes long on first run**: ~30 s, cached in `~/.cache/vllm/`.
- **`make gc-test` permission error on `/output`**: `chmod 777 test/output`.
- **Form-fill validation fails repeatedly**: check
  `form_fill_warnings[]` in the prediction record. Common causes are
  weak instruction-following on small models — try
  `+experiment=qwen_local` or raise `agent.form_fill.max_retries`.
