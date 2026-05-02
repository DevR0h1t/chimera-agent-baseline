"""Terminal form-fill node — prompt + parse against the output schema.

The ReAct loop ends when the model stops issuing tool calls. The router
then routes to this node, which prompts the same model to emit a JSON
object matching the per-case Pydantic shape (built dynamically from
:mod:`chimera_agent_baseline.output.schema`) and validates with
:class:`langchain_core.output_parsers.PydanticOutputParser`.

We deliberately avoid ``model.with_structured_output`` — it relies on
function-calling support that varies wildly across providers (Gemma 4's
offline tool-call parser, for instance, sometimes does not emit a tool
call when there is only one forced schema-tool). Prompt-and-parse is
provider-neutral: any LangChain ``BaseChatModel`` works.

This module is fair game to edit — change the skeleton, swap the
parser, tune the retry strategy, replace the whole node. The only
constraint is that the final JSON must validate against
:class:`Task1Output` / :class:`Task2Output`; submissions whose final
output does not validate are rejected.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser

from chimera_agent_baseline.output.schema import (
    build_dynamic_model,
    eligible_variables,
    normalise_to_full_shape,
)

log = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are filling out a structured reasoning form for the prostate-cancer "
    "case you just analysed. The user message contains your decision "
    "transcript and the action log of the tools you called. Output a SINGLE "
    "JSON object matching the supplied schema EXACTLY — no extra keys, no "
    "markdown fences, no commentary before or after the JSON."
)

def make_form_fill_node(model: BaseChatModel, max_retries: int = 3):
    """Return a LangGraph node closure that captures the unbound *model*.

    *max_retries* is the number of validation attempts before the node
    falls back to a stub structured response (the run-level validator
    in :mod:`chimera_agent_baseline.run` will then raise on the stub).
    """

    def form_fill(state: dict[str, Any]) -> dict[str, Any]:
        messages = state["messages"]
        task = int(state.get("task", 1))
        case_id = state.get("case_id", "unknown")

        called = _called_tools_from_messages(messages)
        transcript = _final_assistant_text(messages)
        elig = eligible_variables(task, called)

        Dynamic = build_dynamic_model(task, called)
        parser = PydanticOutputParser(pydantic_object=Dynamic)
        skeleton = _build_skeleton_instructions(task, elig)

        log.info(
            "form_fill: case=%s task=%d called_tools=%s eligible_vars=%s",
            case_id,
            task,
            sorted(called) or [],
            elig,
        )

        base_user = _user_prompt(case_id, task, transcript, sorted(called), elig) + "\n\n" + skeleton

        # Provider-agnostic retry loop. We keep the conversation flat — a
        # single system + user pair, regenerated on retry — because small
        # models can be derailed by long error-laden histories. The retry
        # message names the missing/invalid fields explicitly.
        warnings: list[str] = []
        structured_response: dict[str, Any] | None = None
        retry_hint: str | None = None

        for attempt in range(1, max_retries + 1):
            user_content = base_user if not retry_hint else f"{base_user}\n\n{retry_hint}"
            try:
                response = model.invoke(
                    [
                        SystemMessage(content=_SYSTEM_PROMPT),
                        HumanMessage(content=user_content),
                    ]
                )
                raw = response.content if isinstance(response.content, str) else json.dumps(response.content)
                obj = parser.parse(_extract_json_object(raw))
                structured_response = obj.model_dump(mode="json")
                break
            except Exception as exc:  # noqa: BLE001 — must not crash the graph
                warnings.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                log.warning("form_fill parse failed (attempt %d) for %s: %s", attempt, case_id, exc)
                if attempt == max_retries:
                    break
                retry_hint = (
                    "Your previous attempt did not validate. The validation "
                    f"error was:\n{exc}\n\n"
                    "Emit the JSON object exactly matching the shape above. "
                    "Every required key must appear. Output ONLY the JSON."
                )

        if structured_response is None:
            log.error("form_fill: all %d attempts failed for %s — writing partial", max_retries, case_id)
            structured_response = {
                "case_id": case_id,
                "task": "risk_stratification" if task == 2 else "mri_diagnostic",
                "confidence": "Uncertain",
                "decision_summary": "(Form-fill structured call failed validation; see form_fill_warnings.)",
                "variable_ratings": {},
            }
            warnings.append("invalid_schema=True")

        # Pad omitted variables to "Not used" so downstream eval sees the
        # full static shape.
        full = normalise_to_full_shape(task, structured_response)
        return {"structured_response": full, "form_fill_warnings": warnings}

    return form_fill


# ---------------------------------------------------------------------------
# Prompt builders + helpers
# ---------------------------------------------------------------------------


def _user_prompt(
    case_id: str,
    task: int,
    transcript: str,
    called_tools: list[str],
    eligible: list[str],
) -> str:
    tools_line = ", ".join(called_tools) if called_tools else "(no tools called)"
    return (
        f"Case ID: {case_id}\n"
        f"Task: {task}\n\n"
        "Your reasoning transcript (final assistant message from the ReAct "
        'loop):\n"""\n'
        f"{transcript}\n"
        '"""\n\n'
        f"Tools you called during the ReAct loop: {tools_line}\n\n"
        "Eligible variables for rating (you may ONLY rate these — every "
        "other variable was either out of scope for this task or behind a "
        "tool you did not call):\n"
        f"  {', '.join(eligible) if eligible else '(none)'}\n\n"
        "Now fill out the form. Use 'Not used' / 'Noted' / 'Important' / "
        "'Decisive' for ratings; concise per-variable reasoning; an overall "
        "confidence; and a focused decision_summary naming the 2-4 factors "
        "that most influenced your recommendation."
    )


def _build_skeleton_instructions(task: int, eligible: list[str]) -> str:
    """Concrete JSON-skeleton format instructions.

    PydanticOutputParser's ``get_format_instructions()`` dumps the full
    JSON schema (with ``$defs``, ``$ref``, ``additionalProperties``,
    etc.). Small models tested in the wild (Gemma 4 E2B) sometimes echo
    that schema back verbatim instead of producing an instance. A
    concrete shape with placeholder values is far more robust and is
    still unambiguous about which keys are required.
    """
    rating_enum = '"Not used" | "Noted" | "Important" | "Decisive"'
    confidence_enum = '"Clear" | "Borderline" | "Uncertain"'

    if task == 1:
        task_const = '"mri_diagnostic"'
        decision_lines = [
            '  "biopsy_recommendation": <true | false>,',
            '  "repeat_test": <string with extra test you would request, or null>,',
        ]
    else:
        task_const = '"risk_stratification"'
        treatment_enum = (
            '"active_surveillance" | "watchful_waiting" | "radical_prostatectomy" '
            '| "radiotherapy" | "focal_therapy" | "hormonal_therapy"'
        )
        decision_lines = [f'  "treatment_recommendation": <{treatment_enum}>,']

    rating_lines = [
        f'    "{var}": {{"rating": <{rating_enum}>, "reasoning": "<one or two sentences>"}},' for var in eligible
    ]
    if rating_lines:
        rating_lines[-1] = rating_lines[-1].rstrip(",")

    skeleton = "\n".join(
        [
            "{",
            '  "case_id": "<the case id>",',
            f'  "task": {task_const},',
            f'  "confidence": <{confidence_enum}>,',
            '  "decision_summary": "<at least 40 chars; name the 2-4 factors that drove your call>",',
            *decision_lines,
            '  "variable_ratings": {',
            *rating_lines,
            "  }",
            "}",
        ]
    )

    return (
        "Output a SINGLE JSON object matching exactly this shape (replace "
        "every <...> with a real value):\n\n"
        f"{skeleton}\n\n"
        f"`variable_ratings` MUST include all and only these keys: {', '.join(eligible)}.\n"
        "Use the rating value 'Not used' (with empty reasoning) for any variable "
        "that did not influence your decision. Do NOT emit any other keys at the "
        "top level. Do NOT wrap in markdown fences. Do NOT echo the schema."
    )


def _extract_json_object(text: str) -> str:
    """Best-effort extraction of a JSON object from raw model output."""
    if not text:
        return text
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _called_tools_from_messages(messages: list) -> set[str]:
    return {m.name for m in messages if isinstance(m, ToolMessage) and m.name}


def _final_assistant_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
            return m.content if isinstance(m.content, str) else json.dumps(m.content)
    return ""
