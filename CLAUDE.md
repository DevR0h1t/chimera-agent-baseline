# Chimera Agent Baseline

Baseline agent for the CHIMERA-Agent challenge, built with LangGraph.

## Stack

- **LangGraph** for agent graph construction and execution
- **LangChain** for LLM abstractions and tool integration
- **Hydra** + **OmegaConf** for config management (composition, CLI overrides)
- **python-dotenv** for environment variable loading (API keys, HF tokens, etc.)
- **pytest** for testing
- **ruff** for formatting and linting

## Development principles

### Config-driven development

**All agent parameters must live in YAML config files, never hardcoded in Python.**

```
configs/
├── config.yaml               <- base config with all defaults (Hydra entrypoint)
└── experiment/               <- experiment overrides (only what differs)
```

How it works:
- `config.yaml` defines every parameter with sensible defaults
- Experiment configs override what changes: `+experiment=my_exp`
- CLI overrides for quick iteration: `model.hf_id=meta-llama/...` (bare key=value)

When adding a new parameter:
1. Add it to the relevant section in `configs/config.yaml`
2. Access it in code via `cfg.section.param` (dot notation)
3. Never use magic numbers or hardcoded values in Python files

### Code quality

- **Never use `print()` for logging.** Use Python's `logging` module.
- **Type hints** on all public function signatures.
- **Format with `ruff format`**, lint with `ruff check`. Run `make lint` before committing.

## Project structure

```
src/chimera_agent_baseline/     Python package — distributed via pip install
configs/
  config.yaml                   Base config (Hydra entrypoint) — all defaults
  experiment/                   Experiment override configs (+experiment=name)
templates/prompts/              Jinja prompt templates the agent renders at runtime
tests/                          Tests
scripts/                        Helper scripts (process_guidelines.py, etc.)
Dockerfile                      Grand Challenge container build
inference.py                    GC container entrypoint
```

## Workflows

### Local development
```bash
source .venv/bin/activate
make install
make test
make run
```

### Grand Challenge container
```bash
make gc-build                            # build the GC Docker image
make gc-run INPUT=outputs/agent_input/task1
make gc-save                             # export image + model tarballs
```

## Environment variables

Use `.env` (loaded by python-dotenv) for secrets and per-system config:

```bash
HF_TOKEN=...
```

## Git conventions

- **Commit messages**: single line, imperative mood (e.g., "Add tool selection logic")
- **No Co-Authored-By** in commit messages
- **Tagging**: tag releases and milestones with semver (e.g., `git tag v0.1.0`)
- Run `make lint` before committing
