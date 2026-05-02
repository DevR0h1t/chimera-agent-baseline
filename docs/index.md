# Documentation

## Getting started

Start here: **[User Manual](user-manual.md)** — requirements, setup, first run.

## Reference

- **[Architecture](architecture.md)** — how the agent works, how to add
  tools and knowledge
- **[Model Configuration](models.md)** — swapping models, tool call parsers,
  experiment overlays
- **[Challenge Tasks](chimera.md)** — task definitions, inputs, outputs,
  evaluation metrics

## Guidelines

The baseline includes a searchable database of the
[EAU Prostate Cancer Guidelines (2026)](https://uroweb.org/guidelines/prostate-cancer),
accessible via the `search_guidelines` MCP tool. The database is pre-built
and included in the repository (`resources/guidelines_db/`).

To rebuild it (e.g. after adding additional guideline PDFs):

```bash
make process-guidelines
```

This requires a HuggingFace token (`HF_TOKEN` in `.env`) for the
embedding model download.
