"""Input/output schemas for the CHIMERA-Agent challenge.

Defines the expected query format (input) and prediction format (output)
for each task.  Uses Jinja2 templates for prompt formatting.

The model outputs only the numeric/boolean prediction fields as JSON.
The reasoning trace is compiled separately from the system-level
execution history (see ``trace.py``).
"""

import json
import logging
import re
from pathlib import Path

from jinja2 import Template

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schemas — one per task
#
# These define the prediction fields the model must output as JSON.
# The model produces both predictions AND a reasoning trace.
# The system separately records an action log (tool calls + observations)
# via trace.py for faithfulness verification.
# ---------------------------------------------------------------------------

TASK_OUTPUT_SCHEMAS: dict[str, dict] = {
    "mri_diagnostic": {
        "cspca_probability": "float (0-1)",
        "biopsy_recommendation": "bool",
    },
    "risk_stratification": {
        "active_surveillance_eligibility": "bool",
    },
    "bcr_prediction": {
        "bcr_risk_1yr": "float (0-1)",
        "bcr_risk_2yr": "float (0-1)",
        "bcr_risk_5yr": "float (0-1)",
    },
}

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

CASE_PROMPT_TEMPLATE = Template(
    """\
Case ID: {{ case_id }}
Context: {{ context }}
Question: {{ query }}

First, call the tools with case_id="{{ case_id }}" to gather evidence. \
Then write your reasoning trace explaining your clinical assessment, \
referencing the specific values you retrieved. Finally, provide your \
answer as a JSON block:

```json
{
{% for field, type in output_fields.items() %}\
    "{{ field }}": {{ type }}{{ "," if not loop.last }}
{% endfor %}\
}
```"""
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_queries(input_dir: str | Path) -> list[dict]:
    """Load per-case queries from ``queries.json`` in the input directory."""
    queries_file = Path(input_dir) / "queries.json"
    if not queries_file.exists():
        raise FileNotFoundError(
            f"No queries.json found in {input_dir}. Each task directory must contain a queries.json file."
        )
    queries = json.loads(queries_file.read_text())
    log.info("Loaded %d queries from %s", len(queries), queries_file)
    return queries


def format_case_prompt(query: dict) -> str:
    """Render a case query into an agent prompt using Jinja2."""
    task = query.get("task", "unknown")
    output_fields = TASK_OUTPUT_SCHEMAS.get(task, {})

    return CASE_PROMPT_TEMPLATE.render(
        case_id=query["case_id"],
        context=query["context"],
        query=query["query"],
        output_fields=output_fields,
    )


def parse_prediction(text: str, case_id: str, task: str) -> dict:
    """Extract prediction values and reasoning trace from the agent's response.

    The model's final response contains:
    1. A reasoning trace (free text before the JSON block)
    2. A JSON block with the prediction fields

    Both are extracted and returned together.
    """
    reasoning_trace = ""
    prediction = {"case_id": case_id}

    match = re.search(r"```json\s*(\{[^`]*\})\s*```", text, re.DOTALL)
    if match:
        # Everything before the JSON block is the reasoning trace
        reasoning_trace = text[: match.start()].strip()

        try:
            prediction.update(json.loads(match.group(1)))
        except json.JSONDecodeError:
            log.warning("Found JSON block for %s but failed to parse it", case_id)
    else:
        log.warning("Could not parse structured output for %s", case_id)
        reasoning_trace = text
        schema = TASK_OUTPUT_SCHEMAS.get(task, {})
        for key in schema:
            prediction[key] = None

    prediction["reasoning_trace"] = reasoning_trace
    return prediction
