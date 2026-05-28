# Preprocessing Consolidation Plan

**Date**: 2026-05-27
**Goal**: Replace 3 partially-overlapping preprocessing implementations with a single `preprocessing.py` module that handles fzark, ECG-FP, PTB-XL, and MIT-BIH uniformly.

---

## 1. Current State Audit

### 1.1 Three implementations of the same 6-step pipeline

| Implementation | Used by | Input shape | Normalization | Filter stage |
|---|---|---|---|---|
| `robust_preprocess()` inline in eval scripts | `eval_ecg_tprex.py`, `eval_doctor_removed.py`, `compare_tp_fzark_with_full_suppression.py`, `compare_all_classes_with_full_suppression.py` | 1-D `(N,)` | winsorized z-score | yes |
| `UniversalECGAdapter` class | (none after cleanup; available as utility) | 2-D `(C, N)` | plain z-score | yes |
| `RobustECGAdapter` class | `mitdb_eval.py` | 2-D `(C, N)` | winsorized z-score | yes |
| `PTBXL_SingleLead_Dataset` (inline in `ptbxl_eval.py`) | `ptbxl_eval.py` | 2-D `(12, N)` slice → 1-D | plain z-score | **none** |

### 1.2 Step-by-step comparison

| Step | JSON pipeline | UniversalECGAdapter | RobustECGAdapter | PTB-XL eval | Inconsistency? |
|---|---|---|---|---|---|
| 1. Load | JSON frames → `÷ MAGNIFICATION` (1000) → mV | WFDB → mV (caller-converted) | WFDB → mV (caller-converted) | WFDB `rdsamp` (already mV) | format-specific (OK) |
| 2. Lead select | n/a (1-D in) | `align_leads()` by name | `align_leads()` by name | manual slice `data[1:2, :]` | **inline slice in PTB-XL** |
| 3. Notch | 50 Hz fixed @ Q=30 | 50/60 Hz param @ Q=30 | 50/60 Hz param @ Q=30 | **none** | **PTB-XL skips** |
| 4. Bandpass | 0.67–40 Hz, butter(4) | 0.67–40 Hz, butter(4) | 0.67–40 Hz, butter(4) | **none** | **PTB-XL skips** |
| 5. Baseline | medfilt `0.4s | 1` kernel | medfilt `0.4s + 1` kernel | medfilt `0.4s + 1` kernel | **none** | **PTB-XL skips**; kernel formula differs trivially |
| 6. Resample | linear interp1d | linear interp1d | linear interp1d | linear interp1d | OK |
| 7. Crop/pad | center, symmetric | center, symmetric | center, symmetric | n/a (resample-to-fit) | **PTB-XL strategy differs** |
| 8. Normalize | winsorized z-score `[1.5, 98.5]` | plain z-score | winsorized z-score `[1.5, 98.5]` | plain z-score | mixed |

### 1.3 Problems with the current state

1. **Copy-paste drift risk**: The same `robust_preprocess()` lives in 4 scripts. A bug fix in one won't propagate.
2. **PTB-XL preprocessing is structurally different**: no filtering at all, different resample strategy. This is the largest inconsistency.
3. **No single source of truth** for constants (`FS_OUT=500`, `TARGET_LEN=5000`, `POWERLINE_HZ`, etc.) — each script re-declares them.
4. **No format-aware loaders**: each script wires its own JSON / WFDB loader.
5. **No unit tests** — silent regressions are easy.
6. **MAGNIFICATION=1000** only applies to fzark/FP; other datasets are already in physical units.

---

## 2. Design — `preprocessing.py`

### 2.1 Single class, configurable behaviour

```python
class ECGPreprocessor:
    """Unified preprocessing for the 1-lead ECGFounder.

    Pipeline:
        load → select_lead → notch → bandpass → baseline → resample
              → crop_pad → normalize → torch.float32 (1, 5000)

    All steps are toggleable via the constructor; defaults match the
    fzark / FP / MIT-BIH winsorized recipe (the model's training pipeline).
    """

    def __init__(
        self,
        target_fs: int = 500,
        target_len: int = 5000,
        target_lead: str = "II",
        powerline_hz: int = 50,              # 50 EU, 60 US
        apply_notch: bool = True,
        apply_bandpass: bool = True,
        bandpass_hz: tuple[float, float] = (0.67, 40.0),
        apply_baseline: bool = True,
        baseline_window_s: float = 0.4,
        normalize: str = "winsorize",        # "winsorize" | "zscore" | "none"
        win_limits: tuple[float, float] = (1.5, 98.5),
    ):
        ...

    # Format-agnostic core ---------------------------------------------------
    def process(
        self,
        signal: np.ndarray,                  # shape (C, N) or (N,)
        fs_in: int,
        source_leads: list[str] | None = None,  # required if C > 1
    ) -> torch.Tensor:                       # shape (1, target_len)
        ...

    # Format-specific loaders (thin wrappers around `process`) -------------
    def from_fzark_json(self, json_path: str) -> torch.Tensor: ...
    def from_wfdb(self, record_path: str)   -> torch.Tensor: ...
```

### 2.2 Factory methods (the "what's right for this dataset" knowledge)

```python
@classmethod
def for_fzark(cls)   -> "ECGPreprocessor":
    """fzark / ECG-FP / Vivalink Holter recordings @ 128 Hz, EU 50 Hz grid."""
    return cls(powerline_hz=50, normalize="winsorize")

@classmethod
def for_ptbxl(cls)   -> "ECGPreprocessor":
    """PTB-XL @ 500 Hz, German 50 Hz grid, already lightly pre-cleaned.

    Apply full filtering anyway for consistency with the training pipeline.
    """
    return cls(powerline_hz=50, normalize="winsorize")

@classmethod
def for_mitdb(cls)   -> "ECGPreprocessor":
    """MIT-BIH @ 360 Hz, US 60 Hz grid, MLII as Lead II."""
    return cls(powerline_hz=60, target_lead="II", normalize="winsorize")
```

### 2.3 Format-specific loaders

The loaders return a `(channels, samples)` numpy array + sampling rate + lead names, then delegate to `process()`:

```python
def from_fzark_json(self, json_path: str) -> torch.Tensor:
    with open(json_path) as f:
        records = json.load(f)
    raw = np.concatenate([np.array(r['data']['ecg'], dtype=np.float32)
                          for r in records])
    raw = raw / 1000.0                       # MAGNIFICATION → mV
    return self.process(raw[np.newaxis, :], fs_in=128, source_leads=["II"])

def from_wfdb(self, record_path: str) -> torch.Tensor:
    sig, meta = wfdb.rdsamp(record_path)     # (N, C) float, physical units
    sig = sig.T                              # → (C, N)
    return self.process(sig, fs_in=meta['fs'], source_leads=meta['sig_name'])
```

### 2.4 Why one class, not a function

- **State**: derived quantities (kernel size, filter coefficients) can be computed once per `fs_in` value and cached.
- **Configurability**: 12 parameters → kwargs are clearer than a sprawling function signature.
- **Factories**: `ECGPreprocessor.for_mitdb()` is more self-documenting than `preprocess(signal, powerline_hz=60, ...)`.

---

## 3. Configuration Matrix

What each dataset's factory resolves to:

| Setting | `for_fzark()` | `for_ptbxl()` | `for_mitdb()` |
|---|---|---|---|
| native fs | 128 Hz | 500 Hz | 360 Hz |
| target fs | 500 Hz | 500 Hz | 500 Hz |
| target len | 5000 | 5000 | 5000 |
| target lead | II (only one) | II (sliced from 12) | II (= MLII) |
| powerline | 50 Hz | 50 Hz | 60 Hz |
| notch | ✅ | ✅ | ✅ |
| bandpass | 0.67–40 Hz | 0.67–40 Hz | 0.67–40 Hz |
| baseline (median) | ✅ 0.4s | ✅ 0.4s | ✅ 0.4s |
| crop/pad | center | center | center |
| normalize | winsorize z-score | winsorize z-score | winsorize z-score |

**Key change**: PTB-XL gets full filtering. Currently it skips notch/bandpass/baseline entirely, which is silently different from how every other dataset is processed. If the model was trained on filtered signals, the PTB-XL inference is operating off-manifold.

---

## 4. API Comparison

### 4.1 Before (today)

```python
# eval_ecg_tprex.py
FS_IN, FS_OUT = 128, 500
TARGET_LEN, MAGNIFICATION = 5000, 1000
POWERLINE_HZ, WIN_LIMITS = 50, (1.5, 98.5)

def load_json_ecg(json_path):  # 5 lines
    ...
def robust_preprocess(signal_1d, fs_in=FS_IN):  # ~40 lines
    ...

ecg_mv = load_json_ecg(json_path)
tensor = robust_preprocess(ecg_mv)
```

### 4.2 After

```python
# eval_ecg_tprex.py
from preprocessing import ECGPreprocessor
prep = ECGPreprocessor.for_fzark()
tensor = prep.from_fzark_json(json_path)
```

### 4.3 MIT-BIH before/after

Before (`mitdb_eval.py` uses `RobustECGAdapter`):
```python
adapter = RobustECGAdapter(target_leads=['II'], target_length=5000, target_fs=500)
sig, meta = wfdb.rdsamp(record_path)
tensor = adapter.standardize(sig.T, source_leads=meta['sig_name'],
                             fs_in=meta['fs'], powerline_hz=60)
```
After:
```python
prep = ECGPreprocessor.for_mitdb()
tensor = prep.from_wfdb(record_path)
```

### 4.4 PTB-XL before/after

Before (`ptbxl_eval.py` does inline slice + resample only):
```python
data = wfdb.rdsamp(self.ecg_path + hash_file_name)[0].T   # (12, N)
data = data[1:2, :]                                       # Lead II
data = self.z_score_normalization(data)
signal = self.resample_unequal(data, 500, 5000)           # stretches to 5000 samples
```
After:
```python
prep = ECGPreprocessor.for_ptbxl()
tensor = prep.from_wfdb(self.ecg_path + hash_file_name)
```
Side benefit: PTB-XL inputs are now consistent with the training-time preprocessing.

---

## 5. Files Affected

| File | Change |
|---|---|
| **New**: `preprocessing.py` | Add the consolidated module |
| **New**: `tests/test_preprocessing.py` | Round-trip tests, shape/dtype/range checks |
| `eval_ecg_tprex.py` | Replace `load_json_ecg` + `robust_preprocess` with `ECGPreprocessor.for_fzark().from_fzark_json()` |
| `eval_doctor_removed.py` | Same |
| `compare_tp_fzark_with_full_suppression.py` | Same |
| `compare_all_classes_with_full_suppression.py` | Same |
| `mitdb_eval.py` | Replace `UniversalECGAdapter` / `RobustECGAdapter` usage with `ECGPreprocessor.for_mitdb().from_wfdb()` |
| `ptbxl_eval.py` | Replace inline `PTBXL_SingleLead_Dataset` preprocessing with `ECGPreprocessor.for_ptbxl().from_wfdb()` |
| **Delete**: `standardize_external_dataset.py` | Superseded by `preprocessing.py` |
| **Delete**: `standardize_external_dataset_robust.py` | Superseded |
| `standardize_mitdb.py` | Update import |
| `split_mitdb.py` | Update import |

**Net delta**: -2 files, ~200 LOC removed, ~250 LOC added (with tests).

---

## 6. Testing Strategy

`tests/test_preprocessing.py`:

```python
def test_output_shape_dtype():
    # Same shape regardless of input fs
    for fs_in in [128, 360, 500, 1000]:
        sig = np.random.randn(1, fs_in * 10).astype(np.float32)
        t = ECGPreprocessor().process(sig, fs_in=fs_in)
        assert t.shape == (1, 5000)
        assert t.dtype == torch.float32

def test_zero_input_passes_cleanly():
    # No NaNs from filter ringing or divide-by-zero in normalize
    sig = np.zeros((1, 1280))
    t = ECGPreprocessor().process(sig, fs_in=128)
    assert not torch.isnan(t).any()

def test_winsorize_removes_extreme_outliers():
    # Inject a 1000-sigma spike; output should not be dominated by it
    sig = np.random.randn(1, 1280)
    sig[0, 100] = 1000.0
    t = ECGPreprocessor(normalize="winsorize").process(sig, fs_in=128)
    assert torch.abs(t).max() < 10

def test_factory_settings():
    # for_mitdb uses 60 Hz powerline
    assert ECGPreprocessor.for_mitdb().powerline_hz == 60
    assert ECGPreprocessor.for_ptbxl().powerline_hz == 50

def test_byte_identical_to_legacy():
    # Compare output of new preprocessor against the old robust_preprocess
    # on a fixed seed. Tolerance ~1e-6 (filtfilt edge effects).
    ...

def test_lead_alignment_picks_correct_channel():
    # Build a (3, N) signal where channel 1 is a unique marker (sine wave),
    # the others are noise. Selecting lead 'II' must return the marker.
    ...
```

---

## 7. Migration Steps

Sequenced so each step leaves the repo in a working state:

1. **Add** `preprocessing.py` + `tests/test_preprocessing.py`. Don't touch any existing scripts yet.
2. **Run tests** — verify the new module is functionally equivalent to the legacy `robust_preprocess` on a fixed seed (within 1e-6 of legacy output).
3. **Migrate one script** (`eval_ecg_tprex.py`) → verify metrics on a held-out fzark sample match the pre-migration baseline within tolerance.
4. **Migrate remaining JSON scripts** (`eval_doctor_removed.py`, both `compare_*_with_full_suppression.py`).
5. **Migrate `mitdb_eval.py`** → verify MIT-BIH metrics match.
6. **Migrate `ptbxl_eval.py`** → note: metrics WILL change because PTB-XL now goes through filtering. Record the new baseline.
7. **Delete** `standardize_external_dataset.py` and `standardize_external_dataset_robust.py`.
8. **Update** `standardize_mitdb.py` and `split_mitdb.py` to use the new module.

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| PTB-XL metrics shift because filtering is now applied | Expected. Record the new baseline; document the change. Filtering brings PTB-XL onto the same preprocessing manifold as the training data. |
| Numerical mismatch between `filtfilt` edge handling in old vs new code | Add a `test_byte_identical_to_legacy` test with a known seed; tolerate <1e-6 diff. |
| Cached filter coefficients break under threading | Use `functools.lru_cache` keyed on `(fs_in,)` only; values are read-only numpy arrays — safe. |
| Caller passes wrong `source_leads` order | Validate that `len(source_leads) == signal.shape[0]` in `process()`; raise `ValueError` with a clear message. |
| Backward compat: callers still importing `UniversalECGAdapter` | Keep a 5-line shim that emits `DeprecationWarning` and forwards to `ECGPreprocessor`. Remove the shim in a follow-up. |

---

## 9. Constants Reference

These move into `preprocessing.py` as module-level constants (single source of truth):

```python
# preprocessing.py
TARGET_FS        = 500       # Hz — ECGFounder native sampling rate
TARGET_LEN       = 5000      # samples = 10 s @ 500 Hz
DEFAULT_BANDPASS = (0.67, 40.0)
NOTCH_Q          = 30.0
BASELINE_WINDOW_S = 0.4
WIN_LIMITS       = (1.5, 98.5)
FZARK_MAGNIFICATION = 1000   # int → mV scale for Vivalink JSON
```

Scripts no longer redeclare these; they import what they need.

---

## 10. Open Questions for Decision

1. **Should `for_ptbxl()` skip filtering** to preserve the historical PTB-XL eval behavior?
   - **Recommended**: No — apply filtering for consistency. Document the metric shift.
2. **Should we keep `UniversalECGAdapter` as a deprecation shim?**
   - **Recommended**: Yes for one release, then delete.
3. **Should preprocessing be GPU-aware** (run filters on torch instead of numpy)?
   - **Recommended**: Not in this iteration. Inference dominates wall time; preprocessing is CPU-cheap.
4. **Should `from_wfdb()` auto-detect powerline** from record metadata?
   - **Recommended**: No — keep it explicit via the factory. WFDB doesn't always carry country info.

---

## 11. Summary

- **One module** (`preprocessing.py`) replaces 3 implementations.
- **One pipeline** runs identically across fzark, ECG-FP, PTB-XL, and MIT-BIH.
- **Factory methods** encode the per-dataset knobs (powerline frequency, native fs); callers just write `ECGPreprocessor.for_<dataset>()`.
- **PTB-XL preprocessing is brought onto the training manifold** — the biggest functional improvement.
- **Test suite** prevents silent regression and validates parity with the legacy implementation.
- **Net code reduction** of ~30%, with all dataset knowledge centralized.

---

*Implementation pending user approval. Estimated effort: ~2 hours including tests + per-script migration.*
