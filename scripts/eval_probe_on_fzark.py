"""A/B: does noise augmentation help on NOISY fzark? Evaluate base vs clean-probe vs
noise-augmented-probe on the fzark ambulatory cohort (the real noisy-deploy target).

For each scope head with a fzark event, positives = fzark TP windows of that event,
negatives = fp_doctor (flagged-then-removed) windows of that event → threshold-free AUROC
+ binary metrics at the head threshold. Windows are preprocessed ONCE (cached), then each
checkpoint is run over the same tensors.

Models:
  base        = checkpoint/1_lead_ECGFounder.pth
  clean-probe = checkpoint/1_lead_ECGFounder_6head_scope.pth
  noise-probe = checkpoint/1_lead_ECGFounder_6head_scope_noise.pth

Scope heads evaluable on fzark (have TP+FP events): AFib(5), Bradycardia(4),
Sinus Tachy(6), SV Run(93), V Run(98). Normal(2)/Pause(142) have no fzark TP cohort.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score

from preprocessing import ECGPreprocessor
from checkpoints import load_ecgfounder
from device_utils import resolve_device
from label_config import FZARK_LABEL_MAP, head_threshold

TP_DIR = "data/ecg_tp_fzark"
FP_DIR = "data/ecg_fp_doctor removed1"
OUT = "res/finetune_6head_v2"
CACHE = f"{OUT}/fzark_ab_cache.npz"          # gitignored (*.npz)
PER_CLASS, SEED = 500, 42
# scope heads that exist as fzark events
EVENTS = {"Atrial Fibrillation": 5, "Bradycardia": 4, "Sinus Tachycardia": 6,
          "Supraventricular Run": 93, "Ventricular Run": 98}
CKPTS = {
    "base":        "checkpoint/1_lead_ECGFounder.pth",
    "clean-probe": "checkpoint/1_lead_ECGFounder_6head_scope.pth",
    "noise-probe": "checkpoint/1_lead_ECGFounder_6head_scope_noise.pth",
}


def load_cohort(summary, data_dir):
    df = pd.read_csv(summary)
    df = pd.concat([g.sample(n=min(PER_CLASS, len(g)), random_state=SEED)
                    for _, g in df.groupby("Event Type")]).reset_index(drop=True)
    return df


def preprocess_cohort(df, data_dir, prep):
    X, keep = [], []
    for i, r in df.iterrows():
        p = os.path.join(data_dir, str(r["JSON File"]).replace("\\", "/"))
        try:
            X.append(prep.from_fzark_json(p).numpy().astype(np.float32)); keep.append(i)
        except Exception:
            pass
    return np.stack(X) if X else np.zeros((0, 1, 5000), np.float32), keep


def build_cache():
    prep = ECGPreprocessor.for_fzark()
    tp_df = load_cohort(f"{TP_DIR}/summary.csv", TP_DIR)
    fp_df = load_cohort(f"{FP_DIR}/summary.csv", FP_DIR)
    print(f"preprocessing fzark TP={len(tp_df)} + FP={len(fp_df)} windows (once) ...", flush=True)
    Xtp, ktp = preprocess_cohort(tp_df, TP_DIR, prep)
    Xfp, kfp = preprocess_cohort(fp_df, FP_DIR, prep)
    ev_tp = tp_df.loc[ktp, "Event Type"].to_numpy()
    ev_fp = fp_df.loc[kfp, "Event Type"].to_numpy()
    np.savez(CACHE, Xtp=Xtp, Xfp=Xfp, ev_tp=ev_tp, ev_fp=ev_fp)
    print(f"  cached → {CACHE}  (TP {Xtp.shape}, FP {Xfp.shape})")
    return Xtp, Xfp, ev_tp, ev_fp


def infer(ckpt, X, device):
    model = load_ecgfounder(device)
    if ckpt != CKPTS["base"]:
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    xb = torch.from_numpy(X).to(device); out = []
    with torch.no_grad():
        for s in range(0, len(xb), 256):
            out.append(torch.sigmoid(model(xb[s:s+256])).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 150), np.float32)


def main():
    device = resolve_device()
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        Xtp, Xfp, ev_tp, ev_fp = z["Xtp"], z["Xfp"], z["ev_tp"], z["ev_fp"]
        print(f"loaded cache {CACHE}  (TP {Xtp.shape}, FP {Xfp.shape})")
    else:
        Xtp, Xfp, ev_tp, ev_fp = build_cache()

    models = {name: ck for name, ck in CKPTS.items() if name == "base" or os.path.exists(ck)}
    probs = {}
    for name, ck in models.items():
        print(f"inferring {name} ({ck}) ...", flush=True)
        probs[name] = (infer(ck, Xtp, device), infer(ck, Xfp, device))

    pd.set_option("display.width", 200)
    print("\n" + "=" * 92)
    print("fzark A/B — AUROC by scope head (TP=event positives vs fp_doctor negatives)")
    print("=" * 92)
    hdr = f"{'event':<22}{'head':>5}{'n_pos':>7}{'n_neg':>7}" + "".join(f"{m:>13}" for m in models)
    if "clean-probe" in models and "noise-probe" in models:
        hdr += f"{'Δ(noise-clean)':>16}"
    print(hdr)
    rows = []
    for ev, h in EVENTS.items():
        mtp = (ev_tp == ev); mfp = (ev_fp == ev)
        npos, nneg = int(mtp.sum()), int(mfp.sum())
        if npos < 5 or nneg < 5:
            continue
        line = f"{ev:<22}{h:>5}{npos:>7}{nneg:>7}"
        aucs = {}
        for m in models:
            ptp, pfp = probs[m]
            y = np.r_[np.ones(npos), np.zeros(nneg)]
            s = np.r_[ptp[mtp, h], pfp[mfp, h]]
            aucs[m] = roc_auc_score(y, s)
            line += f"{aucs[m]:>13.4f}"
        if "clean-probe" in aucs and "noise-probe" in aucs:
            line += f"{aucs['noise-probe']-aucs['clean-probe']:>+16.4f}"
        print(line)
        rows.append(dict(event=ev, head=h, n_pos=npos, n_neg=nneg, **{f"auroc_{m}": aucs[m] for m in models}))
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/fzark_ab_auroc.csv", index=False)
    if "clean-probe" in models and "noise-probe" in models and len(df):
        dmean = (df["auroc_noise-probe"] - df["auroc_clean-probe"]).mean()
        print(f"\nmean Δ AUROC (noise-probe − clean-probe) on noisy fzark: {dmean:+.4f}")
    print(f"\nArtifacts → {OUT}/fzark_ab_auroc.csv")


if __name__ == "__main__":
    main()
