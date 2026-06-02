"""Device-specific L1 fine-tune on the AliveCor (Challenge 2017) dataset.

Challenge 2017 only labels AFib (A) among the 6 scope events, so this calibrates
the **AFib head (5)** of the ScopeProjection for the AliveCor device, warm-started
from the fzark L1 (fzarkSL.pth) with all other heads frozen (their fzark weights,
temperature, threshold are preserved exactly). Trains the AFib output on A (pos)
vs N (neg) center-windows of the TRAIN split; fits AFib temperature + decision
threshold (Youden-J on A-vs-N) on VAL.

Split: csv/challenge2017_split.csv (80/10/10). Backbone logits cached.
Writes res/scope_overlay/alivecorSL.pth.
Run:  python3 -m scripts.train_l1_alivecor
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_curve

from checkpoints import load_ecgfounder
from device_utils import resolve_device
from preprocessing import ECGPreprocessor
from overlay.scope_overlay import ScopeProjection, SCOPE_HEADS

DATA = "data/challenge2017/Challenge2017/raw_data.npy"
SPLIT = "csv/challenge2017_split.csv"
WARM = "res/scope_overlay/fzarkSL.pth"
OUT = "res/scope_overlay/alivecorSL.pth"
FEAT = "res/scope_overlay/feat_alivecor"
FS_IN, WIN = 300, 3000
AFIB = 5
AJ = SCOPE_HEADS.index(AFIB)


def center(sig):
    sig = np.asarray(sig, float)
    if len(sig) <= WIN:
        return np.pad(sig, (0, WIN - len(sig)))
    s = (len(sig) - WIN) // 2
    return sig[s:s + WIN]


def logits_for(df, backbone, device, tag):
    os.makedirs(FEAT, exist_ok=True)
    cache = f"{FEAT}/logits_{tag}.npy"
    if os.path.exists(cache) and len(np.load(cache)) == len(df):
        return np.load(cache)
    data = np.load(DATA, allow_pickle=True)
    prep = ECGPreprocessor(powerline_hz=60, normalize="winsorize")
    xs = []
    for idx in tqdm(df.idx.values, desc=f"{tag} prep"):
        try:
            xs.append(prep.process(center(data[idx]), fs_in=FS_IN))
        except Exception:
            xs.append(torch.zeros(1, 5000))
    out = []
    with torch.no_grad():
        for i in range(0, len(xs), 128):
            out.append(backbone(torch.stack(xs[i:i+128]).to(device)).cpu().numpy())
    logits = np.concatenate(out, 0).astype(np.float32)
    np.save(cache, logits)
    return logits


def main():
    device = resolve_device(); torch.manual_seed(0)
    backbone = load_ecgfounder(device); backbone.eval()
    sp = pd.read_csv(SPLIT)
    clean = sp[sp.label_str.isin(["A", "N"])].copy()             # AFib labels only
    tr = clean[clean.split == "train"].reset_index(drop=True)
    va = clean[clean.split == "val"].reset_index(drop=True)
    ytr = (tr.label_str == "A").values.astype(np.float32)
    yva = (va.label_str == "A").values.astype(np.float32)
    print(f"train A/N = {int(ytr.sum())}/{len(ytr)-int(ytr.sum())}  "
          f"val A/N = {int(yva.sum())}/{len(yva)-int(yva.sum())}")

    Xtr = torch.tensor(logits_for(tr, backbone, device, "train"))
    Xva = torch.tensor(logits_for(va, backbone, device, "val"))
    Ytr = torch.tensor(ytr); Yva = torch.tensor(yva)

    proj = ScopeProjection(); proj.load_state_dict(torch.load(WARM, map_location="cpu"))
    warm_w = proj.proj.weight.detach().clone(); warm_b = proj.proj.bias.detach().clone()
    warm_temp = proj.temperature.clone(); warm_dth = proj.decision_threshold.clone()

    # train ONLY the AFib output column (weight_decay=0 ⇒ zero-grad rows don't move)
    opt = torch.optim.Adam(proj.proj.parameters(), lr=1e-3, weight_decay=0.0)
    pw = torch.tensor([(len(Ytr) - Ytr.sum()) / Ytr.sum().clamp(min=1)])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    best, best_state, bad = -1.0, None, 0
    for ep in range(400):
        proj.train(); opt.zero_grad()
        z = proj.proj(Xtr)[:, AJ]
        loss = lossf(z, Ytr); loss.backward(); opt.step()
        proj.eval()
        with torch.no_grad():
            pv = torch.sigmoid(proj.proj(Xva)[:, AJ]).numpy()
        ap = average_precision_score(yva, pv)
        if ap > best:
            best, best_state, bad = ap, {k: v.clone() for k, v in proj.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= 30:
            break
    proj.load_state_dict(best_state)

    # restore every non-AFib head EXACTLY to fzark (freeze them)
    with torch.no_grad():
        w = proj.proj.weight.clone(); b = proj.proj.bias.clone()
        for j in range(len(SCOPE_HEADS)):
            if j != AJ:
                w[j] = warm_w[j]; b[j] = warm_b[j]
        proj.proj.weight.copy_(w); proj.proj.bias.copy_(b)

    # AFib temperature (grid) on val
    with torch.no_grad():
        zva = proj.proj(Xva)[:, AJ]
    bl, bt = 1e9, 1.0
    for t in torch.linspace(0.5, 3.0, 26):
        l = nn.functional.binary_cross_entropy_with_logits(zva / t, Yva).item()
        if l < bl:
            bl, bt = l, float(t)
    temp = warm_temp.clone(); temp[AJ] = bt
    proj.temperature.copy_(temp)

    # AFib decision threshold = Youden-J on val (A vs N)
    with torch.no_grad():
        pva = proj(Xva).numpy()[:, AJ]
    fpr, tpr, thr = roc_curve(yva, pva)
    tau = float(thr[np.argmax(tpr - fpr)])
    dth = warm_dth.clone(); dth[AJ] = tau
    proj.decision_threshold.copy_(dth)

    torch.save(proj.state_dict(), OUT)
    sens = float((pva[yva == 1] >= tau).mean()); spec = float((pva[yva == 0] < tau).mean())
    print(f"saved {OUT}")
    print(f"AFib head: temp={bt:.2f}  τ(Youden)={tau:.3f}  val sens={sens:.3f} spec={spec:.3f} AP={best:.3f}")
    print(f"other heads frozen at fzark values (weights/temp/threshold unchanged)")


if __name__ == "__main__":
    main()
