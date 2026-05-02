"""Structured output contract for the agent.

This module defines the **submission schema** every agent must produce
to be eligible for evaluation. Outputs that do not validate against
:class:`Task1Output` / :class:`Task2Output` are rejected. Participants
may swap models, tools, prompts, and orchestration freely, but this
shape is fixed.

After the ReAct loop produces a final assistant message, the terminal
form-fill node issues a separate prompt-and-parse call that populates
the per-task Pydantic shape: a decision, an overall confidence, a
focused decision summary, and a per-variable rating + reasoning.

Per-task variable→tool mapping drives a *dynamic* schema at runtime:
only variables that are present in the prompt context OR backed by a
tool the agent actually called appear as required fields for that case.
The validated payload is then normalised back to the full static shape
with ``Not used`` / empty reasoning for omitted variables, so downstream
eval sees a uniform record.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------


class Rating(StrEnum):
    NOT_USED = "Not used"
    NOTED = "Noted"
    IMPORTANT = "Important"
    DECISIVE = "Decisive"


class Confidence(StrEnum):
    CLEAR = "Clear"
    BORDERLINE = "Borderline"
    UNCERTAIN = "Uncertain"


class TreatmentChoice(StrEnum):
    """Treatment options for Task-2 risk-stratification."""

    ACTIVE_SURVEILLANCE = "active_surveillance"
    WATCHFUL_WAITING = "watchful_waiting"
    RADICAL_PROSTATECTOMY = "radical_prostatectomy"
    RADIOTHERAPY = "radiotherapy"
    FOCAL_THERAPY = "focal_therapy"
    HORMONAL_THERAPY = "hormonal_therapy"


class VariableReasoning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rating: Rating = Field(description="How decisive this value was for the decision.")
    reasoning: str = Field(
        default="",
        description=(
            "One- or two-sentence explanation of how this value influenced "
            "the decision. May be empty when rating is 'Not used'."
        ),
    )


# ---------------------------------------------------------------------------
# Per-task variable -> tool mapping. ``None`` means the variable is in
# the prompt context and so always rateable, regardless of which tools
# were called. A string means "this tool must have been called for the variable
# to be rateable".
# ---------------------------------------------------------------------------


TASK1_VARIABLES: dict[str, str | None] = {
    "psa": None,
    "age": None,
    "dre": None,
    "comorbidity": None,
    "prior_biopsy": None,
    "pirads": "get_mri_report",
    "psa_density": "get_mri_report",
    "prostate_volume": "get_mri_report",
    "cspca": "get_mri_report",
    "family_history": "get_family_history",
}


TASK2_VARIABLES: dict[str, str | None] = {
    "psa": None,
    "age": None,
    "dre": None,
    "comorbidity": None,
    "ct": None,
    "pirads": "get_mri_report",
    "psa_density": "get_mri_report",
    "prostate_volume": "get_mri_report",
    "cspca": "get_mri_report",
    "family_history": "get_family_history",
    "bx_isup": "get_pathology_report",
    "bx_isup_pred": "get_pathology_report",
    "bx_gl_prim": "get_pathology_report",
    "bx_gl_sec": "get_pathology_report",
}


VARIABLES_BY_TASK: dict[int, dict[str, str | None]] = {
    1: TASK1_VARIABLES,
    2: TASK2_VARIABLES,
}


# ---------------------------------------------------------------------------
# Static "full-shape" output models — used to normalise the dynamic
# response back into a consistent on-disk record.
# ---------------------------------------------------------------------------


class _BaseOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    confidence: Confidence
    decision_summary: str = Field(
        min_length=40,
        description=("Overall reasoning for the recommendation, naming the 2-4 factors that most influenced it."),
    )


class Task1Output(_BaseOutput):
    task: Literal["mri_diagnostic"] = "mri_diagnostic"
    biopsy_recommendation: bool = Field(description="True = recommend biopsy, False = defer / no biopsy.")
    repeat_test: str | None = Field(
        default=None,
        description=(
            "Free-text describing any additional test you would request before "
            "deciding (e.g. repeat PSA in 3 months, PSMA-PET). Use null if no "
            "additional test is needed."
        ),
    )
    variable_ratings: dict[str, VariableReasoning] = Field(
        default_factory=dict,
        description="One entry per Task-1 reasoning variable (see TASK1_VARIABLES).",
    )


class Task2Output(_BaseOutput):
    task: Literal["risk_stratification"] = "risk_stratification"
    treatment_recommendation: TreatmentChoice
    variable_ratings: dict[str, VariableReasoning] = Field(
        default_factory=dict,
        description="One entry per Task-2 reasoning variable (see TASK2_VARIABLES).",
    )


TASK_OUTPUT_MODELS: dict[int, type[_BaseOutput]] = {
    1: Task1Output,
    2: Task2Output,
}


# ---------------------------------------------------------------------------
# Dynamic schema builder
# ---------------------------------------------------------------------------


def eligible_variables(task: int, called_tools: set[str]) -> list[str]:
    """Variables eligible for rating in the structured output.

    A variable is eligible if its required tool is None (in the prompt) or
    in ``called_tools`` (the agent actually invoked it during the ReAct
    loop). The action log surfaces tool names like ``get_mri_report``.
    """
    mapping = VARIABLES_BY_TASK[task]
    return [v for v, req in mapping.items() if req is None or req in called_tools]


def build_dynamic_model(task: int, called_tools: set[str]) -> type[BaseModel]:
    """Construct a per-case pydantic model with only the eligible ratings.

    Used as the ``json_schema`` constraint on the structured-output LLM
    call. After validation, ``normalise_to_full_shape`` re-pads omitted
    variables back to the static task model.
    """
    eligible = eligible_variables(task, called_tools)
    base = TASK_OUTPUT_MODELS[task]

    # Per-variable fields modelled as optional VariableReasoning entries.
    var_fields: dict[str, Any] = {
        name: (VariableReasoning, Field(description=f"Rating + reasoning for '{name}'.")) for name in eligible
    }
    DynamicVariableRatings = create_model(  # noqa: N806 — dynamic class name
        f"DynamicTask{task}VariableRatings",
        __config__=ConfigDict(extra="forbid"),
        **var_fields,
    )

    # Replace the loose ``dict[str, VariableReasoning]`` with the strict
    # per-variable model, so the JSON schema sent to the LLM enumerates
    # exactly the eligible keys.
    field_overrides: dict[str, Any] = {
        "variable_ratings": (
            DynamicVariableRatings,
            Field(description="Per-variable rating + reasoning."),
        ),
    }

    Dynamic = create_model(  # noqa: N806
        f"Dynamic{base.__name__}",
        __base__=base,
        **field_overrides,
    )
    return Dynamic


def normalise_to_full_shape(
    task: int,
    raw_payload: dict[str, Any],
) -> dict[str, Any]:
    """Pad any variable not present in ``raw_payload['variable_ratings']``.

    Missing variables get ``{"rating": "Not used", "reasoning": ""}`` so
    the final on-disk record always has the same field set per task.
    """
    full_payload = dict(raw_payload)
    ratings = dict(full_payload.get("variable_ratings") or {})
    for var in VARIABLES_BY_TASK[task]:
        if var not in ratings:
            ratings[var] = {"rating": Rating.NOT_USED.value, "reasoning": ""}
    full_payload["variable_ratings"] = ratings
    return full_payload
