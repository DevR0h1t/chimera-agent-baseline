<p align="center">
  <img src="docs/images/thumbnail.png" alt="CHIMERA Agent" width="250">
</p>

# Chimera Agent Baseline

Baseline agent for the [CHIMERA-Agent challenge](https://chimera-agent.grand-challenge.org/chimera-agent/).
Uses a ReAct reasoning loop with clinical tools served via MCP and
guideline retrieval via RAG.

## Quick start

```bash
uv venv && source .venv/bin/activate
cp .env.example .env                    # add your HF_TOKEN
make install
make test
```

Download model weights and embedding model (requires accepted licenses on
[Gemma 4](https://huggingface.co/google/gemma-4-E2B-it) and
[embeddinggemma](https://huggingface.co/google/embeddinggemma-300m)):

```bash
# LLM (~10GB)
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-E2B-it', local_dir='model/')"

# Embedding model for guidelines search (~1.2GB, saved to resources/)
make process-guidelines
```

Run the agent (requires NVIDIA GPU with ≥16GB VRAM):

```bash
make run
```

Output is written to `test/output/predictions.json`.

## Local testing layout

The repo mirrors the Grand Challenge filesystem so the same code works
locally and in the container:

| Local path | Container path | Contents |
|------------|---------------|----------|
| `test/input/task{1,2,3}/` | `/input` | Queries + clinical data (read-only) |
| `test/output/` | `/output` | Agent predictions (written by the agent) |
| `model/` | `/opt/ml/model` | LLM weights (gitignored) |
| `resources/` | `/opt/app/resources` | Config, guidelines DB, embedding model |

To test a specific task:

```bash
make run RUN_ARGS="paths.input_dir=test/input/task2"
```

To test in a Docker container (mirrors Grand Challenge runtime):

```bash
make gc-build
make gc-test                                    # task 1 (default)
make gc-test GC_INPUT_DIR=test/input/task3      # task 3
```

On Grand Challenge, `/input` and `/output` are mounted by the platform.
Model weights are uploaded separately and mounted at `/opt/ml/model`.
No changes to the MCP server or agent code are needed.

## Documentation

| Document | What it covers |
|----------|---------------|
| [User Manual](docs/user-manual.md) | Requirements, setup, commands, troubleshooting |
| [Architecture](docs/architecture.md) | Components, how to add tools / skills / knowledge |
| [Model Configuration](docs/models.md) | Swapping models, tool parsers, experiment overlays |
| [Challenge Tasks](docs/chimera.md) | Task definitions, inputs, outputs, metrics |

## What to change

| Goal | Where |
|------|-------|
| Change the system prompt | `src/chimera_agent_baseline/agent/prompts.py` |
| Add a clinical tool | `src/chimera_agent_baseline/tools/definitions.py` |
| Add an Agent Skill | `skills/<name>/SKILL.md` |
| Swap the LLM | `configs/config.yaml` → `model.model_id` + `model.tool_parser` |
| Change the agent loop | `src/chimera_agent_baseline/agent/graph.py` |
| Add knowledge to RAG | `scripts/process_guidelines.py` → rebuild with your PDF |
