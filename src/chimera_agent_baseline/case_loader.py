"""Runtime case loader.

Reads the per-patient ``<pid>/prompt.json`` shipped at
``outputs/agent_input/task<N>/`` and renders the baseline agent's
prompt narrative through a Jinja template. Returns one query dict per
case in the form ``{case_id, task, query, context}``.

The companion ``<pid>/tools.json`` is read separately by the MCP server
(:mod:`chimera_agent_baseline.tools.base`); it never reaches this
module.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

TASK1_NAME = "mri_diagnostic"
TASK2_NAME = "risk_stratification"


def render_baseline_prompt(
    payload: dict[str, Any],
    templates_dir: Path | str = "templates/prompts",
    template_name: str = "agent_prompt.j2",
) -> str:
    """Render the agent prompt from a ``prompt.json`` payload.

    Participants can swap ``template_name`` for their own Jinja file.
    The template receives the full payload in scope, plus task-2-only
    keys (``labs`` / ``psa_trend``) defaulting to empty lists, so a
    single template file can handle both tasks.
    """
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(str(templates_dir)), keep_trailing_newline=True)
    ctx = {"labs": [], "psa_trend": [], **payload}
    return env.get_template(template_name).render(**ctx)


def load_cases(
    cases_dir: Path | str,
    *,
    task: int = 1,
    templates_dir: Path | str = "templates/prompts",
    template_name: str = "agent_prompt.j2",
) -> list[dict[str, Any]]:
    """Read per-case ``prompt.json`` files into agent queries.

    Each entry has shape ``{case_id, task, context}``. ``context`` is
    the rendered prompt narrative — the question text is interpolated
    into it by the Jinja template.
    """
    cases_dir = Path(cases_dir)
    if not cases_dir.is_dir():
        raise FileNotFoundError(f"cases_dir {cases_dir} is not a directory")

    out: list[dict[str, Any]] = []
    for sub in sorted(p for p in cases_dir.iterdir() if p.is_dir() and p.name.startswith("PT-")):
        prompt_path = sub / "prompt.json"
        if not prompt_path.exists():
            log.warning("Skipping %s: no prompt.json", sub)
            continue
        try:
            payload = json.loads(prompt_path.read_text())
        except json.JSONDecodeError as e:
            log.warning("Skipping %s: invalid JSON (%s)", prompt_path, e)
            continue
        out.append(
            {
                "case_id": payload["case_id"],
                "task": payload.get("task") or (TASK2_NAME if task == 2 else TASK1_NAME),
                "context": render_baseline_prompt(payload, templates_dir, template_name),
            }
        )
    return out
