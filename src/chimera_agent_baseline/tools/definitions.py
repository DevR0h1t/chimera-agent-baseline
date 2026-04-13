"""Tool definitions for the CHIMERA-Agent challenge.

Each :class:`~chimera_agent_baseline.tools.base.ToolSpec` declares a
normalized field mapping that bridges the different naming conventions
across data sources (RUMC / Karolinska).  Source fields are tried in
order — the first match wins.

To add a custom tool, create a new ``ToolSpec`` and append it to
:data:`TOOL_REGISTRY`.
"""

from chimera_agent_baseline.tools.base import ToolSpec

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

CLINICAL_INFO = ToolSpec(
    name="get_clinical_info",
    description=(
        "Retrieve patient clinical information including demographics, PSA "
        "levels, and medical history. Simulates a query to an electronic "
        "health record (EHR) system."
    ),
    field_mapping={
        "case_id": ["case_id"],
        "source": ["source"],
        "age": ["age", "age_at_prostatectomy", "MR_age"],
        "birthdate": ["birthdate", "BIRTH"],
        "psa": ["pre_operative_PSA", "X_RESULT"],
        "medical_history": ["Anamnes"],
        "previous_treatment": ["yT"],
        "previous_biopsy": ["prev_biopsy"],
        "referral_date": ["REFERRAL_DATE"],
    },
)

GLEASON_GRADES = ToolSpec(
    name="get_gleason_grades",
    description=(
        "Retrieve Gleason grading results from computational pathology "
        "analysis. Returns primary, secondary, and tertiary Gleason "
        "patterns plus the ISUP grade group. Simulates an AI pathology "
        "model that grades biopsy or prostatectomy tissue."
    ),
    field_mapping={
        "case_id": ["case_id"],
        "primary_gleason": ["primary_gleason", "GLEASON1"],
        "secondary_gleason": ["secondary_gleason", "GLEASON2"],
        "tertiary_gleason": ["tertiary_gleason"],
        "isup_grade": ["ISUP"],
        "diagnosis_description": ["DIAG_DESCRIPTION"],
    },
)

MRI_FINDINGS = ToolSpec(
    name="get_mri_findings",
    description=(
        "Retrieve MRI analysis results including PI-RADS score, lesion "
        "detection, and prostate volume assessment. Simulates an AI "
        "radiology model analysing prostate MRI."
    ),
    field_mapping={
        "case_id": ["case_id"],
        "pirads": ["PIRADS", "Pirads_High"],
        "lesion_detected": ["Lesion"],
        "prior_lesion": ["Lesion_anam"],
        "prostate_volume": ["prostata_vol"],
        "imaging_date": ["pre-op_image_date", "provdat"],
    },
)

PATHOLOGY_STAGING = ToolSpec(
    name="get_pathology_staging",
    description=(
        "Retrieve pathological staging information (TNM classification). Simulates a pathology staging assessment."
    ),
    field_mapping={
        "case_id": ["case_id"],
        "pt_stage": ["pT_stage", "tkategori_beskrivning"],
        "n_stage": ["nkategori_beskrivning"],
        "m_stage": ["mkategori_beskrivning"],
    },
)

SURGICAL_PATHOLOGY = ToolSpec(
    name="get_surgical_pathology",
    description=(
        "Retrieve surgical pathology findings from prostatectomy specimen "
        "analysis: margin status, capsular penetration, seminal vesicle "
        "invasion, lymphovascular invasion, and lymph node involvement."
    ),
    field_mapping={
        "case_id": ["case_id"],
        "positive_lymph_nodes": ["positive_lymph_nodes"],
        "capsular_penetration": ["capsular_penetration"],
        "positive_surgical_margins": ["positive_surgical_margins"],
        "invasion_seminal_vesicles": ["invasion_seminal_vesicles"],
        "lymphovascular_invasion": ["lymphovascular_invasion"],
    },
)

FOLLOW_UP = ToolSpec(
    name="get_follow_up",
    description=(
        "Retrieve patient follow-up data including biochemical recurrence "
        "(BCR) status, PSA at recurrence, and follow-up timeline. "
        "Available for post-operative cases."
    ),
    field_mapping={
        "case_id": ["case_id"],
        "bcr": ["BCR"],
        "bcr_psa": ["BCR_PSA"],
        "bcr_date": ["BCR_date"],
        "last_follow_up_date": ["last_follow_up_date"],
        "time_to_event_months": ["time_to_follow-up/BCR"],
        "surgery_date": ["rarp_date"],
        "report_date": ["report_date"],
    },
)

# ---------------------------------------------------------------------------
# Registry — add custom tools here
# ---------------------------------------------------------------------------

TOOL_REGISTRY: list[ToolSpec] = [
    CLINICAL_INFO,
    GLEASON_GRADES,
    MRI_FINDINGS,
    PATHOLOGY_STAGING,
    SURGICAL_PATHOLOGY,
    FOLLOW_UP,
]
