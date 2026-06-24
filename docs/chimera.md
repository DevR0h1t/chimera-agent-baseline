# CHIMERA-Agent Challenge

## Links

- [Challenge website](https://chimera-agent.grand-challenge.org/chimera-agent/)

## Overview

Participants build an agent that makes prostate-cancer clinical decisions
from structured clinical data and masked EHR documents, and explains its
reasoning. Each case ships as JSON, not raw images:

- `prompt.json` — the structured clinical panel the agent always sees.
- `clinical.json` — the masked "Extended EHR view" documents, revealed
  only when the agent calls the matching tool.
- `features.json` — optional frozen foundation-model image embeddings
  (MRI / biopsy / prostatectomy), for participants who want to build a
  predictor tool on top.

See [README → Per-case I/O](../README.md#per-case-io) for the field-level
layout and [architecture.md](architecture.md) for the tool registry. The
exact submission shape for every task is the Pydantic contract in
`src/chimera_agent_baseline/output/schema.py`.

## Tasks

### Task 1 — MRI-only biopsy decision

Decide whether to biopsy from MRI findings and basic clinical variables
(age, PSA, PSA density, prostate volume), and explain what drove the call.

- **Output** (`Task1Output`): `biopsy_decision` (bool), `confidence`
  (`clear` / `borderline` / `uncertain`), `variable_weights` (per
  variable: `not_used` / `noted` / `important` / `decisive`), and
  `reasoning` (≥ 40 chars).
- **Tools**: MRI report, biopsy pathology (returns "no data" when there is
  no prior biopsy), previous notes, lab results, PSA trend, family
  history, guideline search.
- **Ground truth**: histopathology-confirmed csPCa (ISUP Grade Group ≥ 2);
  negatives confirmed by longitudinal PSA follow-up.

### Task 2 — Treatment decision after biopsy

Choose the management strategy by synthesising MRI findings, biopsy
pathology, and PSA-related data (guided by EAU/NCCN standards), flagging
factors that argue against a more conservative path.

- **Output** (`Task2Output`): a single `action` — one of
  `active_surveillance`, `continued_surveillance`, `watchful_waiting`, or
  `active_treatment` — plus `confidence`, `variable_weights`, and
  `reasoning`.
- **Tools**: the same set as Task 1, now backed by a real biopsy
  pathology report.

### Task 3 — Recurrence prognosis after prostatectomy

Predict time to biochemical recurrence (BCR) from prostatectomy
pathology, biopsy findings, preoperative MRI, and longitudinal PSA. Cases
intentionally include missing modalities to test robustness.

- **Output** (`Task3Output`): `months_to_recurrence` (float) and
  `reasoning`. No weights or confidence.
- **Tools**: MRI report, biopsy pathology, surgical (prostatectomy)
  pathology, previous notes, family history, guideline search. (PSA trend
  and the lab panel are dropped for this task.)
- **Ground truth**: BCR defined as a confirmed PSA rise ≥ 0.2 ng/mL
  post-prostatectomy; non-recurrent patients are censored.

## Dataset splits

| Split | Task 1 | Task 2 | Task 3 | Source |
|---|---|---|---|---|
| Train | 75 | 75 | 75 | Radboudumc |
| Validation | 75 | 75 | 75 | Radboudumc (up to 5 submissions) |
| Test | 250 | 300 | 250 | Incl. 100 external (Karolinska), 1 submission |
