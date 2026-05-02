# CHIMERA-agent Challenge

## Links

- [Challenge website](https://chimera-agent.grand-challenge.org/chimera-agent/)

## Overview

Participants design agent-level decision policies that leverage pre-built, modality-specific foundation models for prostate cancer clinical decision-making. Submissions must include structured reasoning traces explaining how evidence was interpreted and integrated. Input data is provided as precomputed features (`.h5`) and clinical variables (`.csv`) — raw images are not used at inference time.

## Tasks

### Task 1: MRI-Only Diagnostic Decision

Decide whether biopsy is warranted from multiparametric MRI features and basic clinical variables (age, PSA, PSA density, prostate volume), and explain which evidence drove the decision.

- **Input:** Precomputed MRI features (`.h5`) + clinical variables (`.csv`)
- **Output:** JSON containing:
  1. **Biopsy recommendation** — binary yes/no
  2. **Repeat test** — free-text describing any additional test you would request first, or null
  3. **Variable ratings** — per-variable Decisive / Important / Noted / Not used + reasoning
  4. **Decision summary + confidence**
- **Ground truth:** Histopathology-confirmed csPCa (ISUP Grade Group >= 2); negatives confirmed by longitudinal PSA follow-up
- **Provided tool:** MRI prostate zone segmentation
- **Note:** an earlier draft of the spec asked for a continuous csPCa probability evaluated via AUROC. That has been dropped — the agent's decision is binary. The MRI tool's `cspca_pred` (a model-derived probability) is still surfaced as evidence the agent can weigh and rate.

### Task 2: MRI + Biopsy Risk Stratification

After biopsy, determine whether a patient qualifies for active surveillance or requires radical prostatectomy by synthesizing MRI findings, biopsy pathology, and PSA-related clinical data (guided by EAU/NCCN standards). The agent must explicitly flag contradictory factors against active surveillance candidacy.

- **Input:** Precomputed MRI features (`.h5`) + precomputed biopsy WSI features (`.h5`) + clinical variables (`.csv`: PSA, age, Gleason Grade Group, biopsy burden)
- **Output:** JSON (~5 KB) containing:
  1. **Active surveillance eligibility** — binary yes/no (evaluated via AUROC)
  2. **Reasoning trace** — structured explanation documenting evidence sources and explicitly flagging contradictory factors against active surveillance
- **Metric:** AUROC
- **Provided tools:** Automated Gleason grading (biopsy WSI), MRI-based prostate cancer detection

### Task 3: Prostatectomy Pathology Prediction

Predict biochemical recurrence (BCR) risk at 1, 2, and 5 years post-prostatectomy by integrating prostatectomy WSI, biopsy findings, preoperative MRI, and longitudinal PSA kinetics. Cases intentionally include missing modalities to test robustness.

- **Input:** Precomputed features (`.h5`) for prostatectomy WSI, biopsy WSI, and MRI + clinical variables (`.csv`: age, Gleason Grade Group, surgical margins, extracapsular extension, longitudinal PSA)
- **Output:** JSON (~5 KB) containing:
  1. **BCR risk at 1 year** — continuous probability
  2. **BCR risk at 2 years** — continuous probability
  3. **BCR risk at 5 years** — continuous probability (evaluated via C-index)
  4. **Reasoning trace** — structured prognostic explanation identifying dominant recurrence factors and describing how missing or conflicting data was handled
- **Ground truth:** BCR defined as confirmed PSA rise >= 0.2 ng/mL post-prostatectomy; non-recurrent patients censored
- **Metric:** C-index
- **Provided tools:** Standardized preprocessing, automated Gleason grading, MRI cancer detection

## Dataset Splits

| Split | Task 1 | Task 2 | Task 3 | Source |
|---|---|---|---|---|
| Train | 75 | 75 | 75 | Radboudumc |
| Validation | 75 | 75 | 75 | Radboudumc (up to 5 submissions) |
| Test | 250 | 300 | 250 | Incl. 100 external (Karolinska), 1 submission |
