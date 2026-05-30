# Stanford 1D-ResNet vs ECGFounder — model comparison

**Date**: 2026-05-30
**Scope**: walk through the Stanford ECG model in `stanford/` and compare it with the
project's current model (ECGFounder Net1D), including a MIT-BIH head-to-head.

---

## 1. What each model is

| | **Stanford 1D-ResNet** (`stanford/`) | **ECGFounder** (current) |
|---|---|---|
| origin | Hannun, Rajpurkar, …, Ng — *Nature Medicine* 2019, "Cardiologist-level arrhythmia detection" (`awni/ecg`) | Shenda Hong et al. — ECG foundation model |
| paradigm | **task-specific net, trained from scratch** per dataset | **foundation model**, pretrained on millions of ECGs, then fine-tuned/probed |
| framework | Keras/TF (this copy ported to Keras 3 / TF 2.16) | PyTorch |
| designed for | continuous ambulatory **rhythm strips** (iRhythm Zio patch) | per-window multi-label **diagnosis** |

---

## 2. Architecture walkthrough

### Stanford (`stanford/ecg/network.py`)
A 34-layer **1D ResNet**, sequence-to-sequence:
1. **Input** — single lead, z-scored, truncated to a multiple of **256** samples, zero-padded per batch.
2. **Stem** — `Conv1D` (len-16) → BN → ReLU.
3. **16 residual blocks**, each: 2 conv layers (len-16, he-normal), pre-activation **BN→ReLU→Dropout(0.2)**;
   shortcut = **MaxPool** by the block stride, channels **zero-padded ×2** when widening.
   Filters **double every 4 blocks** (32→64→128→256); strides alternate 1,2,1,2… → **256× total downsampling**.
4. **Output** — `TimeDistributed(Dense(K)) → softmax`: **one label per 256 input samples** (≈ one rhythm tag per ~1.3 s).
5. **Train** — categorical cross-entropy, Adam, early-stop + ReduceLROnPlateau.

### ECGFounder (Net1D)
A deeper/wider **1D ResNet (Net1D)**, sequence-to-**one**:
- Single lead, fixed **10 s @ 500 Hz** (1×5000) input.
- Conv stages → **global pooling** → `Linear(1024, 150)` → **150 sigmoid** outputs (one multi-label vector **per record**).
- ~**30.8 M** params; used as a frozen backbone we adapt (linear probe / DualHead).

---

## 3. Side-by-side

| dimension | Stanford 1D-ResNet | ECGFounder |
|---|---|---|
| network family | 1D ResNet (34-layer) | 1D ResNet (Net1D) |
| **output granularity** | **sequence** — 1 label / ~1.3 s segment | **1 global vector / record** |
| **label type** | **softmax** (one mutually-exclusive class/segment) | **sigmoid** (multi-label, co-occurring) |
| **# classes** | **few** (4 cinc17 / ~12 iRhythm) | **150** diagnoses |
| training | **from scratch** on target data | **pretrained foundation** + fine-tune/probe |
| feature width | caps at **256** channels | **1024**-dim |
| input | variable length, dataset rate (200/300/360 Hz) | fixed 10 s @ 500 Hz |
| best fit | continuous Holter/patch monitoring | per-window multi-diagnosis |

### The differences that matter
1. **Sequence vs global output** — Stanford tags every ~1.3 s of a long strip (built for continuous monitoring); ECGFounder emits one label set per 10 s. For ambulatory continuous data (fzark/MOVE), the sequence design is the more natural fit.
2. **Softmax vs sigmoid** — Stanford forces one rhythm per segment; ECGFounder allows simultaneous diagnoses (it routinely co-fires ~8 heads — the reason for the scope/FP machinery).
3. **Scratch vs foundation** — Stanford learns a small label set from one dataset; ECGFounder is a transfer-learning backbone covering 150 labels.

---

## 4. MIT-BIH head-to-head

Both run on MIT-BIH — but **in fundamentally different settings**, which is the whole point:

| | Stanford 1D-ResNet | ECGFounder (single-lead) |
|---|---|---|
| relation to MIT-BIH | **TRAINED ON it** (train_split), eval on dev_split → **in-distribution** | **NEVER trained on it** → **zero-shot / transfer** |
| label space | 4-class softmax (SINUS/AF/Other/Noise), per ~1.3 s segment | 150-class multi-label, per record |
| result | **accuracy ≈ 1.00** — SINUS f1 **1.00** (n=1597), BIGEMINY f1 **0.96** (n=41) | AFib **ROC 0.950 / F1 0.86**; PVC **0.940 / 0.72**; NSR **0.839 / 0.78**; PAC 0.925 / 0.63 |

> **⚠ This is NOT "Stanford beats ECGFounder."** Stanford was **trained on MIT-BIH**, so ~100% on a held-out split of the *same* dataset (which is ~97% SINUS) is **in-distribution** performance. ECGFounder's numbers are **zero-shot** — it never saw MIT-BIH. They measure two different things: *fit-to-this-dataset* vs *transfer-to-an-unseen-dataset*. A fair comparison would require either training ECGFounder on MIT-BIH or evaluating Stanford on a dataset it was not trained on.

Two further non-comparabilities: the **label spaces differ** (4 mutually-exclusive rhythm classes vs 150 multi-label diagnoses), and the **output granularity differs** (per-1.3 s segment vs per-record). So per-class F1 is not directly head-to-head either.

**Reproduce**: `cd stanford && ./ecg_env/bin/python3 ecg/evaluate_trained_mitdb.py` (paths in `examples/mitdb/dev_split.json` were rewritten to the local `processed/` dir).

---

## 5. Verdict — when to use which

- **ECGFounder** — when you need **multi-label diagnosis across many conditions** and want to **transfer** to new datasets/devices without per-dataset training. The project's foundation model; everything (scope, fine-tune, DualHead, FP suppression) is built on it.
- **Stanford 1D-ResNet** — when you need **per-segment rhythm labels on a continuous strip** and can **train on your target distribution**. Its sequence output is conceptually the right tool for continuous ambulatory monitoring (the fzark/MOVE use case), but it's narrow (few classes, single-label, not pretrained).
- **Complementary, not competing**: Stanford = continuous single-rhythm timing on in-distribution data; ECGFounder = broad multi-label diagnosis with transfer. The most useful role for the Stanford model here is an **independent, in-distribution baseline** (e.g., MIT-BIH) — keeping in mind the train/test caveat above.

## Caveats
- Stanford `dev_split` is mostly SINUS; TRIGEMINY/VT had 0 support → those classes unmeasured.
- The Stanford checkpoint here was trained by a prior run (in `stanford/saved/`); not committed (126 MB each, dataset rule).
- ECGFounder MIT-BIH numbers are from `res/mitdb_singlelead/` (single-lead zero-shot eval).
