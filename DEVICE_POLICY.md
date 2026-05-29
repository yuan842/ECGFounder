# Project Device Policy

**Effective**: 2026-05-26
**Default device order**: **MPS → CUDA → CPU** (Apple M-series prioritized)

---

## Single source of truth

All PyTorch device selection in this project now flows through one helper:

```python
from device_utils import resolve_device
device = resolve_device(args.device)   # respects CLI, env var, then default
# or, for scripts without a CLI flag:
device = resolve_device()
```

There are no longer any hard-coded `torch.device('mps' if ... else ...)` chains
in the application scripts.  If you need to change the project-wide default,
edit **`device_utils.py`** — nothing else.

---

## Resolution order

`resolve_device(cli_value, policy)` resolves in this priority:

1. **Explicit `cli_value`** (e.g. `--device cpu`).  Always honored, falls back
   with a warning if the requested device is unavailable.
2. **Environment variable** `ECGFOUNDER_DEVICE` (one of `mps`, `cuda`, `cuda:0`,
   `cpu`, `auto`).
3. **Project default policy** (`auto`), which applies the fallback chain
   **MPS → CUDA → CPU**.

---

## Refactored scripts

The following scripts now call `resolve_device()`:

| Script |
|---|
| `eval_ecg_tprex.py` |
| `finetune_afib.py` |
| `compare_models_on_fp_dataset.py` |
| `afib_deploy.py` |
| `compare_all_models.py`, `compare_afib_models.py`, `compare_mitdb_models.py`, `compare_150class_ptbxl.py` |
| `eval_ambulatory_model.py`, `eval_doctor_removed.py`, `eval_comprehensive_tprex.py` |
| `eval_compare_models_tprex.py`, `eval_fuzzylead2.py`, `eval_models_tprex_fixed.py` |
| `mitdb_eval.py`, `mitdb_eval_singlelead.py`, `mitdb_eval_standard.py` |
| `ptbxl_eval.py`, `ptbxl_eval_lead_ii.py`, `ptbxl_eval_subset.py` |
| `finetune_afib_fuzzy.py` |

(21 files in total.)

---

## Usage

### Defaults — nothing to do
On an Apple Silicon machine, `python3 <script>.py` automatically uses MPS.

### Overriding for one run
```bash
python3 compare_models_on_fp_dataset.py --device cpu      # force CPU
python3 compare_models_on_fp_dataset.py --device cuda:0   # force CUDA GPU 0
```

### Overriding for a shell session
```bash
export ECGFOUNDER_DEVICE=cpu
python3 finetune_afib.py     # will run on CPU
```

### Sanity-check
```bash
python3 -m device_utils       # prints the resolved device
# → [ECGFounder] device = mps   (policy: MPS → CUDA → CPU)
```

---

## Why MPS first?

- The active-development hardware for this project is Apple M-series.
- Net1D (~25 M parameters, 5,000-sample input) is ~**5–10× faster on MPS** than CPU.
- The previous default — implicit MPS-first chain duplicated in every script —
  was correct in intent but fragile in practice (one bad copy and a script
  silently fell back to CPU).  The centralized helper eliminates that risk.

---

## Adding new scripts

```python
# At the top of any new training/eval script:
from device_utils import resolve_device

def main():
    device = resolve_device()        # MPS-first by project policy
    ...
```

If the script has an argparse CLI:

```python
parser.add_argument('--device', default=None,
                    help='Override device (cpu/cuda/mps). Default: project policy.')
args = parser.parse_args()
device = resolve_device(args.device)
```

---

## Known MPS caveats

- A few ops (e.g. `torch.bincount`, certain `index_put_` patterns) are not yet
  implemented on MPS as of PyTorch 2.x and will raise `NotImplementedError`.
  In that case, fall back to CPU for that operation only — do *not* override
  the policy globally.
- MPS performs best with float32; mixed precision is not as mature as on CUDA.
- Reproducibility: set `torch.manual_seed` *and* `torch.mps.manual_seed`.
