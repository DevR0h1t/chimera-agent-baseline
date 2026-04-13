# CHIMERA-agent Challenge

## Links

- [Challenge website](https://chimera-agent.grand-challenge.org/chimera-agent/)

## Overview

Participants design agent-level decision policies that leverage pre-built, modality-specific foundation models for prostate cancer clinical decision-making. Submissions must include structured reasoning traces explaining how evidence was interpreted and integrated. Input data is provided as precomputed features (`.h5`) and clinical variables (`.csv`) — raw images are not used at inference time.

## Tasks

### Task 1: MRI-Only Diagnostic Decision

Estimate the probability of clinically significant prostate cancer (csPCa) and recommend whether biopsy is warranted, using only multiparametric MRI features and basic clinical variables (age, PSA, PSA density, prostate volume).

- **Input:** Precomputed MRI features (`.h5`) + clinical variables (`.csv`)
- **Output:** JSON (~5 KB) containing:
  1. **csPCa probability** — continuous score (evaluated via AUROC)
  2. **Biopsy recommendation** — binary yes/no
  3. **Reasoning trace** — structured explanation referencing PI-RADS scores, lesion characteristics, zonal anatomy, and capsular contact
- **Ground truth:** Histopathology-confirmed csPCa (ISUP Grade Group >= 2); negatives confirmed by longitudinal PSA follow-up
- **Metric:** AUROC
- **Provided tool:** MRI prostate zone segmentation

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
