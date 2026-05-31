# Stanford 1D-ResNet vs ECGFounder — model comparison

**Date**: 2026-05-30
**Scope**: walk through the Stanford ECG model in `stanford/` and compare it with the
project's current model (ECGFounder Net1D), including a MIT-BIH head-to-head.

---

## 0. What the source papers say (paper-grounded facts)

This section anchors the comparison in the two primary papers (read in full), so the
numbers below are quotable rather than inferred.

### 0a. Hannun, Rajpurkar, …, Ng — *Nature Medicine* 2019 (the `stanford/` model)
- **Data**: 91,232 single-lead records from **53,549 patients** (abstract says 53,877),
  recorded by the **Zio patch** (iRhythm) — a **modified Lead II at 200 Hz**, median wear 10.6 days.
- **Classes**: **12 output rhythm classes** (10 arrhythmias + sinus rhythm + noise). **Not 14.**
- **Architecture**: 34-layer **1D ResNet** — 16 residual blocks × 2 conv layers, filter width 16,
  32·2ᵏ filters (k++ every 4th block), every alternate block subsamples ×2, pre-activation
  BN→ReLU→Dropout(0.2). Trained **de-novo** (He init, Adam lr 1e-3 ×10 decay, batch 128).
  They **tried LSTM/bidirectional recurrence and abandoned it** (no gain, slower).
- **Output granularity**: one softmax prediction **every 256 samples = 1.28 s** ("output interval")
  → **~23 predictions per 30 s record**. (The Executive Summary's "1 label/second, 30 per record"
  is wrong.)
- **Eval**: test set 328 records / 328 patients; gold standard = **3-cardiologist consensus
  committee**, human baseline = **6 individual** cardiologists; inter-annotator agreement 72.8 %.
- **Headline**: class-weighted **AUC 0.97** (seq 0.978 / set 0.977); **F1 0.837 > avg cardiologist
  0.780**; DNN sensitivity ≥ avg cardiologist for all classes at cardiologist-level specificity.
- **CinC-2017 generalization**: retrained on PhysioNet/CinC-2017 (4 classes: sinus/AF/noise/other)
  → F1 0.83. **This is the `cinc17` checkpoint we use.**
- **Their stated limitation** (verbatim, p.6): *"applying our algorithm sequentially across an ECG
  record of long duration would result in non-trivial false-positive diagnoses."* — exactly the
  long-recording FP problem our motion/SQI suppression + `report_recording.py` windowing address.
- Code open (`github.com/awni/ecg`), test set public, **training data proprietary** (iRhythm).

### 0b. Li, Aguirre, …, Westover, Hong — ECGFounder, 2025 (arXiv 2410.04133v4) — **our base model**
- **This paper describes `1_lead_ECGFounder.pth`.** It is not a competitor — it is the foundation
  our current stack is built on.
- **Data — HEEDB (Harvard-Emory)**: **10,771,552 ECGs / 1,818,247 subjects**, predominantly
  **10 s 12-lead clinical**. **150 labels** curated from 287 phrases parsed (regex) out of
  **Marquette 12SL** GE reports → this is our `tasks.txt` / 150-head space.
- **Architecture**: **Net1D on a RegNet backbone**, bottleneck blocks with **group convolutions +
  channel-wise attention** (temporal **and** cross-lead). Per-record **multi-label sigmoid** over 150 heads.
- **Training innovation — Positive-Unlabeled (PU) learning**: real labels are incomplete (missing ≠
  negative), so a custom loss **ℒ = −(γ−p)p², γ=1.5** down-weights confident negatives. AdamW,
  lr 1e-3, trainable temperature, batch 1024, ≤20 epochs.
- **Single-lead via LEAD AUGMENTATION** — *the key link to this repo*: take **Lead I** from each
  12-lead record and with 50 % probability inject one of **6 angularly-projected leads (−90°…+90°:
  aVL, −aVR, II, −III, aVF, −aVF)**. **Our "fuzzy" angle dataset (45/60/75/90°, 60°≡Lead II) is a
  faithful re-implementation of this exact technique** — and the basis of the scope linear-probe /
  DualHead fine-tune.
- **Fine-tuning**: **linear probing** (freeze backbone, train new head) or full fine-tune. Our scope
  probe = their linear-probing recipe.
- **Eval**: **same committee design as Hannun (they cite it)** — 3 cardiologists annotate 523 recent
  ECGs (20 label types), 4 more compare. Committee **AUROC 0.968, sens 0.971, spec 0.937, F1 0.677 >
  cardiologist 0.640**; >0.95 AUROC on 80 diagnoses. **External**: CODE-test avg **AUROC 0.981**
  (> S12L-ECG 0.980, CTN 0.963, ECG-SE-ResNet 0.963); **PTB-XL 0.924**. **Single-lead external**:
  NSR **0.975**, AF **0.957**. Table 1 (12-lead committee): AFib 0.996/F1 0.866; Sinus Tachy
  0.996/0.833; Sinus Brady 0.995/0.791; **VT 0.903/0.635**; Normal Sinus 0.969/0.952.
- **Downstream** (fine-tune on MIMIC-IV-ECG): age, sex, NT-proBNP, LVEF, CKD, CHD + **PPG-AFib
  (DeepBeat)** — beats ECG-SimCLR by 2–3 and ECG-ResNet by 4–6 AUROC points.
- Explicitly notes ECGFounder runs **cloud-side on data uploaded from wearables, not on-device** —
  matching our MOVE framing. Code + model + data open (`github.com/PKUDigitalHealth/ECGFounder`).

### 0c. Why this matters for the comparison
- **Stanford and ECGFounder are not rivals in our stack.** Stanford is the external
  sequence-labeling baseline; **ECGFounder is our base model**, and its paper validates the exact
  choices we depend on (150-label space, lead augmentation = fuzzy angles, linear probing = scope probe,
  committee evaluation).
- **The single-lead VT gap is real**: the paper reports strong **12-lead** VT (committee 0.903,
  PTB-XL 0.987), but our measured **single-lead** VT head is ~0.36 on fzark — the 12-lead VT strength
  does **not** survive single-lead, which is why the VT head false-fires on MOVE and why the FP
  machinery matters.

---

## 1. What each model is

| | **Stanford 1D-ResNet** (`stanford/`) | **ECGFounder** (current) |
|---|---|---|
| origin | Hannun, Rajpurkar, …, Ng — *Nature Medicine* 2019, "Cardiologist-level arrhythmia detection" (`awni/ecg`) | Li, Aguirre, …, Westover, Hong — ECGFounder, 2025 (arXiv 2410.04133; `PKUDigitalHealth/ECGFounder`) |
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
4. **Output** — `TimeDistributed(Dense(K)) → softmax`: **one label per 256 input samples** (= 1.28 s at 200 Hz → ~23 tags per 30 s record).
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
| **output granularity** | **sequence** — 1 label / 1.28 s segment (256 samp @ 200 Hz) | **1 global vector / record** |
| **label type** | **softmax** (one mutually-exclusive class/segment) | **sigmoid** (multi-label, co-occurring) |
| **# classes** | **few** (4 cinc17 / ~12 iRhythm) | **150** diagnoses |
| training | **from scratch** on target data | **pretrained foundation** + fine-tune/probe |
| feature width | caps at **256** channels | **1024**-dim |
| input | variable length, dataset rate (200/300/360 Hz) | fixed 10 s @ 500 Hz |
| best fit | continuous Holter/patch monitoring | per-window multi-diagnosis |

### The differences that matter
1. **Sequence vs global output** — Stanford tags every 1.28 s of a long strip (built for continuous monitoring); ECGFounder emits one label set per 10 s. For ambulatory continuous data (fzark/MOVE), the sequence design is the more natural fit.
2. **Softmax vs sigmoid** — Stanford forces one rhythm per segment; ECGFounder allows simultaneous diagnoses (it routinely co-fires ~8 heads — the reason for the scope/FP machinery).
3. **Scratch vs foundation** — Stanford learns a small label set from one dataset; ECGFounder is a transfer-learning backbone covering 150 labels.

---

## 3a. Label handling across databases (a core architectural difference)

**Stanford has no global label space.** Each checkpoint's classes are **auto-derived from its training
data** and frozen into the output layer at train time (`load.Preproc`):
```python
self.classes = sorted(set(l for label in labels for l in label))   # ← whatever strings are in the data
```
So `cinc17` → `['A','N','O','~']`, `mitdb` → `['BIGEMINY','SINUS','TRIGEMINY','VT']` — two different
checkpoints, two different vocabularies, **no shared ontology, no canonical index space**.

Consequently a **new database is handled one of two ways — they are alternatives, not steps:**

| path | what you do | labels come from | cost |
|---|---|---|---|
| **1. Retrain** | set `num_categories` + dataset paths, train (usually from scratch — the output layer/label set changed) | **auto-derived from the new dataset's annotations** (not hand-written) → a *new checkpoint* speaking that DB's vocabulary | full training run per dataset |
| **2. Reuse + hand-map** | keep an existing checkpoint, write a per-script `LABEL_MAP` dict translating the new DB's labels into the model's fixed classes | a **hand-written, lossy** dict (e.g. `{'SINUS':'N','AF':'A','VT':'O','BIGEMINY':'O',…}`, unknown→'O') | ad-hoc, not centralized, collapses classes |

(`evaluate_mitdb.py` literally hard-codes that `LABEL_MAP` to apply the cinc17 model to MIT-BIH.) There is
**no reusable, enforced cross-dataset mapping** — every cross-DB use reconciles labels by hand.

**ECGFounder inverts this.** One **fixed 150-label vocabulary** (`tasks.txt`, SHA-pinned) is defined *once*;
every dataset is **mapped *into* that fixed space** through a **centralized, version-controlled, import-time-enforced**
layer — `label_config` (`FZARK_ONTOLOGY`, `SCOPE_EVENT_TO_HEAD`, `MITDB_BEAT_MAP`, `PTBXL_ACTIVE_CLASSES`) with
`GLOBAL_LABEL_MAP.md` as the single source of truth. A new dataset needs a mapping entry, **not a retrained
backbone** (you add a linear probe / DualHead route at most).

| | **Stanford** | **ECGFounder** |
|---|---|---|
| label space | per-checkpoint, derived from training data | **one fixed 150-label space**, pinned once |
| new database | **retrain** (new checkpoint) *or* **hand-map** (lossy, per-script) | **map into the fixed space** (centralized); backbone untouched |
| cross-DB consistency | ad-hoc dicts, can disagree | centralized + **fails to import on drift** |
| add a new condition | needs data with it + retrain | already one of 150 heads, or map an existing head |

**Takeaway**: Stanford's "one model ⇄ one dataset's labels" forces per-dataset retraining or hand-mapping;
ECGFounder's fixed vocabulary + centralized mapping is the foundation-model advantage for multi-dataset work —
exactly why our cross-dataset evals (PTB-XL, MIT-BIH, fzark, MOVE) all reduce to entries in one label map.

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
