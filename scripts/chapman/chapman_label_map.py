"""SNOMED-CT → ECGFounder-head label map for Chapman-Shaoxing / Ningbo.

These PhysioNet datasets (a.k.a. the PhysioNet/CinC 2021 12-lead corpora) label each
record with one or more **SNOMED-CT** codes in the WFDB header line:  `# Dx: 426783006,...`.
This maps the relevant codes to the ECGFounder 150-class head indices, focused on the
detection-scope heads — especially the data-limited SVT(93) / VT(98) / Pause(142).

HONEST COVERAGE NOTE (verify at ingest with --report-unmapped):
  • SVT (93): WELL covered — Chapman/Ningbo carry SVT, atrial tachy, AVNRT, AVRT, PSVT.
  • Normal(2)/Brady(4)/AFib(5)/SinusTachy(6): well covered (bonus — already OK on PTB-XL).
  • VT (98): SPARSE — these are 10-s resting ECGs; sustained VT is rare. Expect few.
  • Pause (142): essentially ABSENT — no clean sinus-pause/asystole SNOMED in this corpus.
    (Pause still needs CinC-2015 asystole / holter; Chapman does NOT solve it.)

CODE CONFIDENCE: the SNOMED codes below are the standard PhysioNet-2021 Dx_mapping
values. A few (marked ⚠) should be confirmed against the dataset's shipped
`ConditionNames_SNOMED-CT.csv`; ingest_chapman.py logs any code it can't map so the
table can be extended from real data.
"""
from __future__ import annotations
from typing import Iterable

import numpy as np

# scope-head display names (for logging)
HEAD_NAMES = {2: "NORMAL ECG", 4: "SINUS BRADYCARDIA", 5: "ATRIAL FIBRILLATION",
              6: "SINUS TACHYCARDIA", 93: "SUPRAVENTRICULAR TACHYCARDIA",
              98: "VENTRICULAR TACHYCARDIA", 142: "WITH SINUS PAUSE"}

# SNOMED-CT code (str) → founder head index. Many-to-one is intentional (an SVT
# family of codes all route to head 93). Only scope-relevant codes are mapped.
SNOMED_TO_HEAD: dict[str, int] = {
    # ── normal / rate (well covered) ──
    "426783006": 2,    # sinus rhythm                → NORMAL ECG (rhythm-normal proxy)
    "426177001": 4,    # sinus bradycardia
    "164889003": 5,    # atrial fibrillation
    "427084000": 6,    # sinus tachycardia
    # ── SVT family → head 93 (the primary win) ──
    "426761007": 93,   # supraventricular tachycardia
    "713422000": 93,   # atrial tachycardia
    "233896004": 93,   # AV nodal reentrant tachycardia (AVNRT)
    "233897008": 93,   # AV reentrant tachycardia (AVRT)
    "67198005":  93,   # paroxysmal supraventricular tachycardia
    "164890007": 93,   # atrial flutter            ⚠ (organized SV tachyarrhythmia; map optional)
    # ── VT → head 98 (sparse in 10-s resting ECG) ──
    "164895002": 98,   # ventricular tachycardia   ⚠ verify code
    # ── Pause / SA dysfunction → head 142 (likely absent; here if present) ──
    "65778007":  142,  # sinoatrial block          ⚠ closest available; not true sinus pause
}

# Codes you may want to EXCLUDE from the AFib-flutter conflation, etc., can be tuned
# here. Atrial flutter (164890007) is debatable for head 93 — flip to drop if undesired.
OPTIONAL_CODES = {"164890007"}            # remove from SNOMED_TO_HEAD if you don't want AFL→SVT

SCOPE_HEADS = sorted(set(SNOMED_TO_HEAD.values()))


def labels_from_dx(dx_field: str, n_classes: int = 150):
    """Parse a header `Dx:` field → (150-dim multi-hot vector, mapped_heads, unmapped_codes).

    `dx_field` is the comma-separated SNOMED list (with or without the 'Dx:' prefix).
    Unmapped codes are returned so the caller can audit coverage.
    """
    vec = np.zeros(n_classes, dtype=np.float32)
    txt = dx_field.split("Dx:")[-1] if "Dx:" in dx_field else dx_field
    codes = [c.strip() for c in txt.replace(";", ",").split(",") if c.strip()]
    mapped, unmapped = [], []
    for c in codes:
        h = SNOMED_TO_HEAD.get(c)
        if h is None:
            unmapped.append(c)
        else:
            vec[h] = 1.0
            mapped.append(h)
    return vec, sorted(set(mapped)), unmapped


def _self_test():
    print("Chapman/Ningbo SNOMED → founder-head map (scope heads):")
    for h in SCOPE_HEADS:
        codes = [c for c, hh in SNOMED_TO_HEAD.items() if hh == h]
        print(f"  head {h:>3} {HEAD_NAMES.get(h,''):<28} ← {len(codes)} code(s): {codes}")
    # synthetic example: an SVT + sinus-tachy record with one unknown code
    vec, mapped, unmapped = labels_from_dx("Dx: 426761007,427084000,999999999")
    print("\nexample 'Dx: 426761007,427084000,999999999':")
    print(f"  mapped heads = {mapped}  (expect [6, 93])")
    print(f"  unmapped     = {unmapped}  (expect ['999999999'])")
    print(f"  vec sum      = {int(vec.sum())}")
    assert mapped == [6, 93] and unmapped == ["999999999"] and vec[93] == 1 and vec[6] == 1
    print("\nself-test OK ✓")


if __name__ == "__main__":
    _self_test()
