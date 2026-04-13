# User Manual

## System requirements

| Requirement | Why |
|-------------|-----|
| Python 3.12+ | Language runtime |
| [uv](https://docs.astral.sh/uv/) | Package manager (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh \| sh`) |
| NVIDIA GPU ≥16GB VRAM | vLLM in-process inference (Gemma 4 uses ~10GB) |
| CUDA 12.x + drivers | Required by vLLM |
| Docker + NVIDIA Container Toolkit | For `make gc-test` (container testing) |
| ~15GB disk space | Model weights (~10GB) + guidelines DB (~25MB) + embedding model (~1.2GB) |
| HuggingFace account | Access to gated models (Gemma 4, embeddinggemma) |
| `make` | Build automation (standard on Linux/macOS) |

## Setup

### 1. Create environment

```bash
git clone <repo-url>
cd chimera-agent-challenge
uv venv
source .venv/bin/activate
make install
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your HuggingFace token (get one at
https://huggingface.co/settings/tokens). Required for downloading
the Gemma 4 model and the embedding model.

### 3. Accept model licenses

Visit these HuggingFace pages and accept the license:
- https://huggingface.co/google/gemma-4-E2B-it
- https://huggingface.co/google/embeddinggemma-300m

### 4. Download model weights

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-E2B-it', local_dir='model/')"
```

This downloads ~10GB to `model/`. Skip this step if you want vLLM to
download on first run (requires internet).

### 5. Build the guidelines database

The guidelines database (`resources/guidelines_db/`) is included in the
repository and works out of the box. To rebuild it (e.g. after adding
your own guideline PDFs), run:

```bash
make process-guidelines
```

This downloads the embedding model (~1.2GB, saved to
`resources/embedding_model/`), processes the PDF, and rebuilds the
database. Takes ~40 seconds on GPU.

### 6. Run tests

```bash
make test
```

### 7. Run the agent

```bash
make run
```

First run takes ~30 seconds for vLLM to compile CUDA graphs (cached
afterwards). Processes all cases in `test/input/task1/` by default.

## Common commands

```bash
# Run on a specific task
make run RUN_ARGS="paths.input_dir=test/input/task2"

# Quick test on one case
python scripts/sanity_check.py --task task1 --case rumc-001

# Lint and format
make lint
make format

# Build and test the Grand Challenge container
make gc-build
make gc-test                                    # requires GPU + Docker
make gc-test GC_INPUT_DIR=test/input/task2      # different task

# Export container for upload
make gc-save
```

## Project layout

```
src/chimera_agent_baseline/
├── agent/
│   ├── graph.py              ReAct agent loop (modify for custom orchestration)
│   └── prompts.py            System prompt (modify to change agent behaviour)
├── models/
│   ├── __init__.py           Model loading (vLLM or OpenAI API)
│   └── vllm_offline.py       vLLM in-process wrapper
├── tools/
│   ├── base.py               ToolSpec + CaseDataStore (infrastructure)
│   └── definitions.py        Tool definitions (add custom tools here)
├── mcp_server.py             MCP server + action logging
├── rag.py                    Guidelines search (ChromaDB + embeddings)
├── schemas.py                I/O schemas + prompt templates
├── skills.py                 Agent Skills loader
└── utils.py                  Logging setup

configs/config.yaml           All parameters (model, generation, paths, agent)
skills/                       Agent Skills (SKILL.md files)
resources/                    Guidelines DB + embedding model (generated)
model/                        LLM weights (downloaded, gitignored)
test/input/task{1,2,3}/       Test data (queries.json + clinical-data.json)
inference.py                  Grand Challenge container entrypoint
Dockerfile                    Container definition
```

## What to modify

| Goal | File(s) | How |
|------|---------|-----|
| Change the system prompt | `agent/prompts.py` | Edit `SYSTEM_PROMPT` |
| Add a clinical tool | `tools/definitions.py` | Add a `ToolSpec` to `TOOL_REGISTRY` |
| Add a non-data tool | `mcp_server.py` | Add `@mcp.tool()` function |
| Swap the LLM | `configs/config.yaml` | Change `model.model_id` + `model.tool_parser` |
| Add a skill | `skills/<name>/SKILL.md` | Create directory + SKILL.md |
| Change the agent loop | `agent/graph.py` | Replace LangGraph with your own orchestration |
| Add knowledge to RAG | `scripts/process_guidelines.py` | Change PDF path, rerun |
| Use a different framework | Replace `agent/`, `run.py` | Keep MCP server + schemas |

## What NOT to modify

- **`schemas.py`** — output format is part of the challenge specification
- **`mcp_server.py` action logging** — the action log must remain intact for evaluation
- **`inference.py`** — Grand Challenge entrypoint (structure must stay compatible)

## Switching models

See [docs/models.md](models.md) for detailed instructions. Quick version:

```yaml
# configs/config.yaml
model:
  model_id: meta-llama/Llama-3.1-8B-Instruct
  tool_parser: llama
```

Download weights, and you're done. No code changes needed.

## Container workflow

The Grand Challenge container runs everything in a single process:
vLLM loads the model, MCP server provides tools, agent runs the
ReAct loop. No network access.

```bash
# Build
make gc-build

# Test locally (mirrors GC runtime: --gpus all, --network none)
make gc-test

# Export for upload
make gc-save
# Uploads: chimera-agent-baseline.tar.gz (Algorithm > Containers)
#          model.tar.gz (Algorithm > Models)
```

See the [Grand Challenge documentation](https://grand-challenge.org/documentation/)
for container conventions.

## Troubleshooting

**"CUDA out of memory"**: Reduce `generation.max_model_len` in config
(default: 32768). Try 8192 for 16GB GPUs.

**"Guidelines DB not found"**: Run `make process-guidelines`. Requires
HF_TOKEN in `.env` for the embedding model download.

**Model downloads fail (403 Forbidden)**: Accept the model license on
HuggingFace and ensure HF_TOKEN is set in `.env`.

**vLLM compilation takes long**: First run compiles CUDA graphs (~30s).
This is cached in `~/.cache/vllm/` for subsequent runs.

**Container permission error on `/output`**: Run
`chmod 777 test/output` before `make gc-test`.
