# Chapman-Shaoxing / Ningbo ingestion (SVT top-up) — prototype

Open-access (no DUA) 12-lead source to top up the data-limited scope heads, primarily
**SVT (93)**. Drop-in counterpart to the PTB-XL fuzzy dataset: same 4-angle derivation,
same npz layout, record-stratified split mirroring the global 1-8/9/10 fold rule.

## Files
- `chapman_label_map.py` — SNOMED-CT → founder-head map (scope heads). `labels_from_dx()`.
- `ingest_chapman.py` — read 12-lead WFDB → derive {45,60,75,90}° → per-angle npz + split
  manifest. `--dry-run` validates with **no data**.

## Honest coverage (the key caveat)
| scope head | Chapman/Ningbo | use this corpus? |
|---|---|---|
| **SVT (93)** | well covered (SVT, atrial tachy, AVNRT, AVRT, PSVT) | **✅ yes — the win** |
| Normal(2)/Brady(4)/AFib(5)/SinusTachy(6) | well covered | bonus (already OK on PTB-XL) |
| **VT (98)** | sparse (10-s resting ECG; sustained VT rare) | ⚠ few — supplement with CinC-2015 |
| **Pause (142)** | ~absent (no clean sinus-pause/asystole SNOMED) | ❌ no — needs CinC-2015 asystole |

So Chapman **solves SVT**, partially helps VT, and does **not** address Pause.

## Get the data (open access)
PhysioNet — no credentials/DUA:
- Chapman-Shaoxing: `https://physionet.org/content/ecg-arrhythmia/` (a.k.a. "A 12-lead ECG database")
- Ningbo: included in the same release / PhysioNet-2021 training set.
```
wget -r -np -nH --cut-dirs=3 https://physionet.org/files/ecg-arrhythmia/1.0.0/ -P data/chapman_ningbo
```
Expected `--data-dir` layout: WFDB records (`*.hea` + `*.mat`/`*.dat`) recursively; each
`.hea` has a `# Dx: <snomed,...>` line and 12 leads.

## Run
```bash
python scripts/chapman/ingest_chapman.py --dry-run                       # validate, no data
python scripts/chapman/ingest_chapman.py --data-dir data/chapman_ningbo \
       --out data/chapman_fuzzy --n-per-head 900 --report-unmapped
```
Outputs to `--out`: `train_{45,60,75,90}deg.npz` (`ecg (n,1,5000)`, `ecg_ids`, `labels (n,150)`)
+ `chapman_split_manifest.csv` (`record_id, fold, split, labels, scope_heads`).

## Wire into the probe
Concatenate the Chapman angle npz with the PTB-XL fuzzy npz in `MultiAngleDataset`, and
combine the split manifests (Chapman's `split` column + `ptbxl_splits`) so train/val/test
stay disjoint per source. SVT(93) positives then jump from PTB-XL's 5 test / 32 train into
the hundreds → its fold-10 test ROC finally becomes trustworthy.

## Caveats
- **Verify ⚠ codes** (VT 164895002, SA-block 65778007, AFL→SVT 164890007) against the
  dataset's `ConditionNames_SNOMED-CT.csv`; `--report-unmapped` logs every unmapped code so
  the map can be extended from real data.
- Record-level split assumes ~1 record/patient (true for Chapman/Ningbo). If a patient↔records
  table exists, fold on `patient_id` instead.
- `.mat` records need scipy/wfdb's mat support; resample to 500 Hz is minimal (no filtering) —
  the model's `ECGPreprocessor` still applies its pipeline downstream.
- This is a **prototype** — not yet run on real data (no Chapman download present).
