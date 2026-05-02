"""Tool definitions for the CHIMERA-Agent baseline.

Two registries — one per task — are exposed by the MCP server. Each
:class:`~chimera_agent_baseline.tools.base.ToolSpec` lists the fields
from the per-case ``tools.json`` it returns.

To add a custom tool, append a new ``ToolSpec`` to the appropriate
registry below.
"""

from chimera_agent_baseline.tools.base import ToolSpec

# ---------------------------------------------------------------------------
# Per-patient encounter tools.
#
# The agent receives the always-visible patient context (encounter, vitals,
# PMHx, DRE prose, csPCa pill, age, headline PSA / PI-RADS) up front in the
# rendered user prompt; the tools below cover the masked / on-demand
# sections.
# ---------------------------------------------------------------------------

PSA_TREND = ToolSpec(
    name="get_psa_trend",
    description=(
        "Retrieve the patient's prior PSA measurements as a time series "
        "(date + value). Reveals PSA velocity / trajectory beyond the "
        "single headline value already in the prompt."
    ),
    fields=("psa_trend",),
)

LAB_RESULTS = ToolSpec(
    name="get_lab_results",
    description=(
        "Retrieve the full laboratory panel for the patient (PSA, free PSA, "
        "haematology, renal function, testosterone, alkaline phosphatase, "
        "LDH, etc.) as a list of {name, val, date, flag}."
    ),
    fields=("labs",),
)

MRI_REPORT = ToolSpec(
    name="get_mri_report",
    description=(
        "Retrieve the radiologist's full mpMRI report as free text, plus the "
        "structured imaging values it contains (PI-RADS, prostate volume, "
        "PSA density, AI csPCa probability)."
    ),
    fields=("imaging_report", "pirads", "prostate_volume", "psa_density", "cspca_pred"),
)

PATHOLOGY_REPORT = ToolSpec(
    name="get_pathology_report",
    description=(
        "Retrieve the pathologist's full biopsy report as free text, plus "
        "structured per-timepoint biopsy data (Gleason patterns, ISUP grade "
        "group, cribriform / intraductal / %GP4 when reported). Returns an "
        "empty report if the patient has not had a biopsy."
    ),
    fields=("pathology_report", "biopsies", "prior_biopsy"),
)

PREVIOUS_NOTES = ToolSpec(
    name="get_previous_notes",
    description=(
        "Retrieve previous GP / urology consultation notes for this patient, as a list of {date, author, text}."
    ),
    fields=("previous_notes",),
)

FAMILY_HISTORY = ToolSpec(
    name="get_family_history",
    description=(
        "Ask the patient about their first-degree family history of "
        "prostate cancer (father / brothers). Returns 'Yes', 'No', or "
        "'Unknown'. This is anamnesis elicited during the consultation "
        "rather than information present in the EHR up front, so it must "
        "be actively requested."
    ),
    fields=("family_history",),
)


# Task 1 — biopsy decision. Full per-encounter tool set.
TASK1_TOOLS: list[ToolSpec] = [
    PSA_TREND,
    LAB_RESULTS,
    MRI_REPORT,
    PATHOLOGY_REPORT,
    PREVIOUS_NOTES,
    FAMILY_HISTORY,
]


# Task 2 — treatment decision. The urologist arrives at the MDT with the
# lab panel and PSA trend already in hand (in the prompt context up
# front), so those tools drop out. Pathology returns a richer per-core
# detail set since treatment choice depends on it.
PATHOLOGY_REPORT_TREATMENT = ToolSpec(
    name="get_pathology_report",
    description=(
        "Retrieve the full pathology report for the patient's prostate "
        "biopsy / biopsies. Returns the prose report plus structured "
        "per-timepoint detail: Gleason patterns, ISUP grade group "
        "(reported and AI-predicted), cores positive / total, max core %, "
        "dominant growth pattern, presence of high-risk patterns "
        "(cribriform / IDC-P), perineural invasion (PNI), lymphovascular "
        "invasion (LVI), and tumour location."
    ),
    fields=(
        "pathology_report",
        "biopsies",
        "bx_isup",
        "bx_gl_prim",
        "bx_gl_sec",
        "bx_gl_tert",
        "bx_isup_pred",
        "ct",
        "cores_positive",
        "cores_total",
        "max_core_pct",
        "growth_pattern",
        "high_risk_patterns",
        "pni",
        "lvi",
        "tumor_location",
        "prior_biopsy",
    ),
)


TASK2_TOOLS: list[ToolSpec] = [
    MRI_REPORT,
    PATHOLOGY_REPORT_TREATMENT,
    PREVIOUS_NOTES,
    FAMILY_HISTORY,
]
