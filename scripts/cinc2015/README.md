# CinC-2015 ingestion (VT + Pause/asystole) — prototype

Open-access (no DUA) ICU source for the two heads Chapman can't supply:
**VT (98)** and **Pause/asystole (142)**. PhysioNet/CinC 2015 — "Reducing False
Arrhythmia Alarms in the ICU".

## Files
- `cinc2015_label_map.py` — alarm-type → founder head; **true/false verdict → positive vs hard-negative**.
- `ingest_cinc2015.py` — read ICU WFDB → pick ECG lead → last-10s-before-alarm → 250→500 Hz →
  single-lead npz + manifest, record-stratified split. `--dry-run` validates with **no data**.

## Why it's structured differently from Chapman
| aspect | CinC-2015 | (vs Chapman/PTB-XL) |
|---|---|---|
| leads | ~2 ECG (II, V) + ABP/PLETH | 12-lead |
| **angles** | **single-lead only — NO Lead I → no 4-angle derivation** | 4 angles |
| record | ~5 min @ 250 Hz, alarm at the end | 10 s @ 500 Hz |
| window | **last 10 s before the alarm** | whole record |
| label source | **alarm type + true/false verdict** | SNOMED multi-hot |

## The label logic (the valuable part)
Each record is a triggered alarm with a **type** and a **true/false** verdict:
- **TRUE** Asystole → positive for head 142; **TRUE** VT → positive for head 98.
- **FALSE** alarm → all-zero label = a **HARD negative** (looked like the arrhythmia but wasn't) —
  exactly the kind of negative that teaches the head to *not* over-fire.
- Ventricular Flutter/Fib → out of scope → skipped.

## Alarm → head
| alarm | head |
|---|---|
| Asystole | 142 (Pause proxy) |
| Ventricular Tachycardia | 98 |
| Bradycardia (extreme) | 4 (bonus) |
| Tachycardia (extreme) | 6 ⚠ (not necessarily *sinus*) |
| Ventricular Flutter/Fib | — (skip) |

## Get the data (open)
```
wget -r -np -nH --cut-dirs=3 https://physionet.org/files/challenge-2015/1.0.0/training/ -P data/cinc2015
```

## Run
```bash
python scripts/cinc2015/ingest_cinc2015.py --dry-run                     # validate, no data
python scripts/cinc2015/ingest_cinc2015.py --data-dir data/cinc2015 \
       --out data/cinc2015_fuzzy [--truth-csv truth.csv] --report-unmapped
```
Outputs: `cinc2015_leadII.npz` (`ecg (n,1,5000)`, `ecg_ids`, `labels (n,150)`) +
`cinc2015_manifest.csv` (`record_id, alarm_type, true_alarm, head, fold, split, ecg_lead`).

## Wire into the probe
Join `cinc2015_leadII.npz` as the **60°/Lead-II stream only** (no 4-angle aug — single lead).
Pairs with `chapman_fuzzy` (SVT) + PTB-XL fuzzy; keep splits disjoint per source via each
manifest's `split` column.

## Caveats
- **Verify the true/false verdict location** in the actual `.hea` headers — `parse_alarm` scans
  comments and falls back to `--truth-csv`. If the release stores verdicts only in a separate
  answers file, pass it via `--truth-csv` (`record,true_alarm`).
- "Tachycardia"→6 and the asystole→Pause mapping are clinical approximations — confirm before use.
- Single-lead only; **Pause is still scarce** even here (asystole ≠ pure sinus pause) — but it's
  the best open label available for head 142.
- **Prototype** — not yet run on real data (no CinC-2015 download present).
