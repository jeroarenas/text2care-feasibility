# TEXT2CARE feasibility check: patient-level extraction from a synthetic lung cancer case

This repository contains a small feasibility check run in November 2026 while preparing the research proposal
TEXT2CARE (*From clinical text to care: privacy-preserving local AI for structuring clinical reports, patient
stratification, and decision support*), a collaboration between Universidad Carlos III de Madrid (UC3M) and the
Thoracic Surgery Department of Hospital Universitario Virgen del Rocío (HUVR, Seville).

**The case is fully synthetic. It contains no data from real patients.**

## What was done

A single synthetic patient with lung cancer is described in 18 clinical documents in Spanish (history, laboratory,
CT, PET-CT, pathology and molecular tests, EBUS staging, tumor boards, surgery, progress notes, discharge, follow-up,
recurrence and second progression), covering about three years. The case comes with the expected value of
75 variables at two levels: 48 stated in or derivable from a single document (level 1) and 27 that require
integrating several documents over time (level 2).

Each model received all 18 documents at once, as one patient history, together with the list of variables and
their requested coding, and had to return, for each variable, a value, the documents used, a short supporting quote
and a confidence from 0 to 100, as JSON constrained by a schema (Ollama structured outputs). Each model was run
six times per level: once at temperature 0 and five times at temperature 0.7 (the cloud model does not accept a
temperature setting, so all its runs use the default sampling).

## Results (first run of each model)

| Model | Run | Level 1 (48) | Level 2 (27) | Time per patient | Values flagged by run disagreement (errors among them) |
|---|---|---|---|---|---|
| gpt-oss-120b | Local | 46 | 26 | 6.6 min | 7 (2 of 2) |
| MedGemma-27B (Q4_K_M) | Local | 43 | 23 | 7.3 min | 17 (8 of 8) |
| Qwen3-32B (no reasoning) | Local | 44 | 22 | 15.6 min | 46 (7 of 7) |
| Claude Opus 5.5 | Cloud (API) | 48 | 26 | < 1 min | 5 (1 of 1) |

Local models were served with Ollama on a single NVIDIA DGX Spark. A value counts as correct when it matches the
expected value or an equivalent allowed by the requested coding (e.g., "RP" for partial response, "T2b" for "cT2b");
the full list of accepted equivalences is in `scripts/eval_pilot.py`. With exact string matching, no model scores
above 69%. Two discrepancies of the local models come from rules that the schema did not state (the coding of KRAS
and the event that dates a recurrence). The confidence declared by the local models is uninformative, whereas
disagreement across runs flags all their errors.

**Caveats.** This is one synthetic patient, so the results show feasibility, not accuracy. The case was prepared
with the help of ChatGPT, and gpt-oss belongs to the same model family, which may favor it.

## Contents

- `data/original/`: the synthetic case as prepared at HUVR (18 documents, guide, expected output for levels 1 and 2,
  and a summary for the tumor board).
- `data/case.txt`, `data/docs/`: plain-text version of the documents. `data/schema_N1.json`, `data/schema_N2.json`:
  variables requested from the models. `data/gold.json`: expected values (never sent to the models).
- `scripts/build_data.py`: rebuilds `data/` from `data/original/`. `scripts/run_pilot.py`: runs a model through
  Ollama, an OpenAI-compatible API or the Anthropic API. `scripts/eval_pilot.py`: evaluation.
- `results/`: raw outputs of every run. `results_eval.json`: per-field verdicts produced by `scripts/eval_pilot.py`.

## Reproducing

```
python3 scripts/run_pilot.py --model gpt-oss:120b --host http://localhost:11434
python3 scripts/eval_pilot.py
```

Only the Python standard library is needed (plus `openpyxl` for `build_data.py`).

## Credits and license

Synthetic case: F. García Gómez, Thoracic Surgery Department, HUVR. Pilot and code: J. Arenas García, UC3M.
Code is released under the MIT license (`LICENSE`). The synthetic case and the results are released under the
Creative Commons Attribution 4.0 International license (CC BY 4.0).
