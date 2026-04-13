---
name: guideline-search
description: >
  Search clinical guidelines for evidence-based recommendations on prostate
  cancer management. Use when you need treatment protocols, staging criteria,
  risk stratification rules, or diagnostic thresholds from the EAU guidelines.
metadata:
  author: chimera
  version: "1.0"
---

## When to use

Activate this skill when the clinical question involves:
- Treatment decisions (active surveillance vs. surgery vs. radiotherapy)
- Risk stratification criteria (low / intermediate / high risk groups)
- Diagnostic thresholds (PSA cutoffs, PI-RADS interpretation, biopsy indication)
- Staging classification (TNM, ISUP grade groups, Gleason scoring)
- Follow-up protocols (PSA monitoring, biochemical recurrence definition)

## How to search

Call `search_guidelines(query)` with a **focused clinical query**.

Good queries are specific and use clinical terminology:

| Scenario | Good query | Bad query |
|----------|-----------|-----------|
| Treatment | "active surveillance eligibility ISUP grade 1 low-risk" | "treatment options" |
| Diagnosis | "PI-RADS 4 biopsy indication PSA density threshold" | "should we biopsy" |
| Staging | "pT3a positive margins adjuvant radiotherapy indication" | "staging" |
| Recurrence | "biochemical recurrence definition PSA after prostatectomy" | "recurrence" |

Tips:
- Include the specific clinical variables you're reasoning about
- Use standard abbreviations (ISUP, PI-RADS, PSA, GG, BCR)
- If the first search is too broad, narrow it with additional terms

## Interpreting results

Each result contains:
- **text**: passage from the EAU Prostate Cancer Guidelines (March 2026)
- **page**: source page number
- **section**: guideline section header
- **score**: relevance (higher = more relevant, range 0-1)

When citing guideline recommendations in your reasoning:
1. Reference the page number and section
2. Distinguish between "Strong" and "Weak" recommendations
3. Note the level of evidence when mentioned
