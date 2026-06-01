"""Train + evaluate the L1 ScopeProjection (overlay) on PTB-XL + fuzzy-lead.

Pipeline (see res/OVERLAY_DESIGN.md §3):
  features = frozen-backbone 150-logits of
      (a) PTB-XL lead-II        — from res/ptbxl_baseline_sqi/base_probs_full.npy
                                   (probs → logits), aligned to csv/ptbxl_label.csv
      (b) fuzzy derived-lead    — data/fuzzylead2/ptbxl/{train,val}_<angle>deg.npz,
                                   backbone run here and cached to res/scope_overlay/feat
  labels   = 6 scope heads {4,5,6,93,98,142} from the 150-vector labels
  split    = folds 1-8 train / 9 val / 10 test, by ecg_id (csv/ptbxl_fold_split.csv)

Trains the 150→6 linear projection (overlay.scope_overlay.ScopeProjection),
fits per-head temperature on val, evaluates fold-10 test vs the BASE backbone
head (raw scope logit). Writes res/scope_overlay/.

Run:  python3 -m scripts.train_scope_overlay            # all 4 angles
      python3 -m scripts.train_scope_overlay --angles 60
"""
from __future__ import annotations
import argparse
import csv
import ast
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

from checkpoints import load_ecgfounder
from device_utils import resolve_device
from overlay.scope_overlay import ScopeProjection, SCOPE_HEADS

BASE_PROBS = "res/ptbxl_baseline_sqi/base_probs_full.npy"
LABEL_CSV = "csv/ptbxl_label.csv"
FOLD_CSV = "csv/ptbxl_fold_split.csv"
FUZZY_DIR = "data/fuzzylead2/ptbxl"
OUT_DIR = "res/scope_overlay"
FEAT_DIR = f"{OUT_DIR}/feat"
ANGLES = [45, 60, 75, 90]
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fold_map() -> dict[int, int]:
    m = {}
    with open(FOLD_CSV) as f:
        for r in csv.DictReader(f):
            m[int(r["ecg_id"])] = int(r["strat_fold"])
    return m


def label_lookup() -> dict[int, np.ndarray]:
    """ecg_id → authoritative 150-vector from csv/ptbxl_label.csv.

    The fuzzy npz carry their OWN `labels` field in a different, sparse encoding
    (e.g. AFib=40 vs 1514) — DO NOT use it. Join fuzzy rows to this lookup so
    lead-II and fuzzy share one label source.
    """
    lut = {}
    with open(LABEL_CSV) as f:
        for r in csv.DictReader(f):
            eid = int(r["filename_lr"].split("/")[-1].split("_")[0])
            lut[eid] = np.asarray(ast.literal_eval(r["label"]), dtype=np.float32)
    return lut


def load_ptbxl_leadii():
    """Standard lead-II: 150-logits (from cached probs) + labels + ecg_id."""
    probs = np.load(BASE_PROBS)                       # (21799,150) sigmoid probs
    feats = logit(probs).astype(np.float32)
    labels, ecg_ids = [], []
    with open(LABEL_CSV) as f:
        for r in csv.DictReader(f):
            labels.append(ast.literal_eval(r["label"]))
            # ecg_id = trailing int of filename_lr "records100/00000/00001_lr"
            ecg_ids.append(int(r["filename_lr"].split("/")[-1].split("_")[0]))
    labels = np.asarray(labels, dtype=np.float32)
    assert feats.shape[0] == labels.shape[0] == len(ecg_ids)
    return feats, labels, np.asarray(ecg_ids, dtype=int)


def fuzzy_logits(npz_path: str, model, device) -> np.ndarray:
    """Run the frozen backbone on a fuzzy npz; cache 150-logits to FEAT_DIR."""
    cache = os.path.join(FEAT_DIR, os.path.basename(npz_path).replace(".npz", "_logits.npy"))
    if os.path.exists(cache):
        return np.load(cache)
    d = np.load(npz_path)
    ecg = torch.from_numpy(d["ecg"]).float()          # (N,1,5000)
    out = []
    with torch.no_grad():
        for i in range(0, len(ecg), 256):
            x = ecg[i:i + 256].to(device)
            out.append(model(x).cpu().numpy())        # raw logits
    L = np.concatenate(out, 0).astype(np.float32)
    os.makedirs(FEAT_DIR, exist_ok=True)
    np.save(cache, L)
    return L


def load_fuzzy(angles, model, device, lut):
    """Fuzzy derived-lead features; labels joined from `lut` (NOT npz labels)."""
    feats, labels, ecg_ids = [], [], []
    for a in angles:
        for split in ("train", "val"):
            # test fold-10 file is val_<a>deg_split.npz; train file is train_<a>deg.npz
            fn = f"{FUZZY_DIR}/{'val' if split=='val' else 'train'}_{a}deg" + \
                 ("_split.npz" if split == "val" else ".npz")
            if not os.path.exists(fn):
                continue
            d = np.load(fn)
            eids = d["ecg_ids"].astype(int)
            keep = np.array([e in lut for e in eids])
            feats.append(fuzzy_logits(fn, model, device)[keep])
            labels.append(np.stack([lut[int(e)] for e in eids[keep]]))
            ecg_ids.append(eids[keep])
            print(f"  fuzzy {os.path.basename(fn):<24} n={int(keep.sum())}/{len(eids)}")
    return (np.concatenate(feats, 0), np.concatenate(labels, 0),
            np.concatenate(ecg_ids, 0))


def split_idx(ecg_ids, fm):
    folds = np.array([fm.get(int(e), -1) for e in ecg_ids])
    return (np.where((folds >= 1) & (folds <= 8))[0],
            np.where(folds == 9)[0], np.where(folds == 10)[0])


# ── L1 TRAINING POLICY (2026-06-01, rev 2) ───────────────────────────────────
# Improve per-head SPECIFICITY by choosing the threshold that MAXIMIZES F1,
# subject to sensitivity dropping no more than MAX_SENS_DROP (default 0.05 = 5pp)
# below the BASELINE head at 0.5. Applied to the production L1 heads only
# (Brady/AFib/Tachy). Fit on validation; report PR-AUC and the max-F1 threshold.
# See res/scope_overlay/L1_TRAINING_POLICY.md.
POLICY_HEADS = (4, 5, 6)
# Per-head acceptable sensitivity-loss budget (pp). Currently uniform 5pp for
# all L1 heads (Brady reverted from 10pp → 5pp 2026-06-01). The map is retained
# so a head can be given a different budget without code changes. Heads not
# listed fall back to the --max-sens-drop arg.
POLICY_MAX_SENS_DROP = {4: 0.05, 5: 0.05, 6: 0.05}


def drop_for(h, default=0.05):
    return POLICY_MAX_SENS_DROP.get(h, default)


def fit_policy_thresholds(p_l1_va, base_va, Yva, scope_cols, max_drop, margin=0.02):
    """Per-head threshold = argmax F1 s.t. sens ≥ base_sens − drop(h) (+margin).

    drop(h) is the per-head budget (POLICY_MAX_SENS_DROP), else `max_drop`.
    Returns (thresholds, info) rows (head, base_sens, floor, tau*, sens, spec, f1, pr_auc)."""
    thr = np.full(len(scope_cols), 0.5, dtype=np.float32)
    info = []
    for i, h in enumerate(scope_cols):
        if h not in POLICY_HEADS:
            continue
        y = Yva[:, i]; npos = int(y.sum()); nneg = int((y == 0).sum())
        if npos == 0:
            continue
        base_sens = float(((base_va[:, i] >= 0.5) & (y == 1)).sum() / npos)
        floor = max(0.0, base_sens - drop_for(h, max_drop)) + margin
        p = p_l1_va[:, i]
        grid = np.unique(np.concatenate([p, np.linspace(0.01, 0.99, 99)]))
        best = (-1.0, 0.5, 0.0, 0.0)   # f1, tau, sens, spec
        for t in grid:
            f = p >= t
            tp = int((f & (y == 1)).sum()); fp = int((f & (y == 0)).sum())
            sens = tp / npos
            if sens < floor:
                continue
            ppv = tp / max(1, tp + fp)
            f1 = 2 * ppv * sens / max(1e-9, ppv + sens)
            spec = (nneg - fp) / max(1, nneg)
            if f1 > best[0]:
                best = (f1, float(t), sens, spec)
        pr = float(average_precision_score(y, p))
        thr[i] = float(np.clip(best[1], 0.01, 0.99))
        info.append((h, base_sens, floor, best[1], best[2], best[3], best[0], pr))
    return thr, info


def metrics(y, p):
    out = {}
    for i, h in enumerate(SCOPE_HEADS):
        yi, pi = y[:, i], p[:, i]
        npos = int(yi.sum())
        fired = pi >= 0.5
        tp = int((fired & (yi == 1)).sum()); fp = int((fired & (yi == 0)).sum())
        sens = tp / npos if npos else float("nan")
        ppv = tp / (tp + fp) if (tp + fp) else float("nan")
        try:
            au = roc_auc_score(yi, pi) if 0 < npos < len(yi) else float("nan")
            pr = average_precision_score(yi, pi) if npos else float("nan")
        except Exception:
            au = pr = float("nan")
        out[h] = dict(n_pos=npos, sens=sens, ppv=ppv, auroc=au, pr_auc=pr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", type=int, nargs="*", default=ANGLES)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--include-leadii", action="store_true", default=True)
    ap.add_argument("--ptbxl-only", action="store_true",
                    help="Train on PTB-XL lead-II ONLY (no fuzzy, no fzark).")
    ap.add_argument("--fuzzy-only", action="store_true",
                    help="Train on PTB-XL fuzzy derived-leads ONLY (no lead-II).")
    ap.add_argument("--out", default=f"{OUT_DIR}/scope_projection.pth",
                    help="Checkpoint output path.")
    ap.add_argument("--spec-opt", action="store_true", default=True,
                    help="Apply the L1 training policy: max-F1 per-head thresholds "
                         "under a max sensitivity-drop constraint.")
    ap.add_argument("--max-sens-drop", type=float, default=0.05,
                    help="Max allowed sensitivity drop vs baseline (pp). Default 0.05.")
    args = ap.parse_args()
    if args.ptbxl_only:
        args.angles = []
    os.makedirs(OUT_DIR, exist_ok=True)
    device = resolve_device()
    torch.manual_seed(0)
    print(f"Device: {device} | angles: {args.angles}")

    scope_cols = SCOPE_HEADS  # 150-vector columns for the 6 scope heads

    if args.fuzzy_only:
        print("PTB-XL FUZZY derived-leads ONLY (no lead-II).")
        model = load_ecgfounder(device); model.eval()
        feats, labels, ecg_ids = load_fuzzy(args.angles, model, device, label_lookup())
    elif args.angles:
        print("Loading PTB-XL lead-II features ...")
        f0, l0, e0 = load_ptbxl_leadii()
        print(f"  lead-II n={len(e0)}")
        print("Loading fuzzy derived-lead features (backbone fwd, cached) ...")
        model = load_ecgfounder(device); model.eval()
        f1, l1, e1 = load_fuzzy(args.angles, model, device, label_lookup())
        feats = np.concatenate([f0, f1], 0)
        labels = np.concatenate([l0, l1], 0)
        ecg_ids = np.concatenate([e0, e1], 0)
    else:
        print("PTB-XL lead-II ONLY (no fuzzy, no fzark).")
        feats, labels, ecg_ids = load_ptbxl_leadii()
    Y = labels[:, scope_cols]
    fm = fold_map()
    tr, va, te = split_idx(ecg_ids, fm)
    print(f"rows: total={len(feats)} train={len(tr)} val={len(va)} test={len(te)}")

    Xtr = torch.tensor(feats[tr]); Ytr = torch.tensor(Y[tr])
    Xva = torch.tensor(feats[va]); Yva = torch.tensor(Y[va])

    # per-head pos_weight (train); Pause (142) has 0 pos → weight 1, untrainable
    pos = Ytr.sum(0); neg = len(Ytr) - pos
    pw = torch.where(pos > 0, neg / pos.clamp(min=1), torch.ones_like(pos))

    proj = ScopeProjection()
    opt = torch.optim.Adam(proj.proj.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_val, best_state, patience, bad = -1, None, 20, 0
    for ep in range(args.epochs):
        proj.train(); opt.zero_grad()
        z = proj.proj(Xtr)                      # pre-sigmoid logits (B,6)
        loss = lossf(z, Ytr)
        loss.backward(); opt.step()
        proj.eval()
        with torch.no_grad():
            pv = torch.sigmoid(proj.proj(Xva)).numpy()
        # mean PR-AUC over heads with val positives
        prs = [average_precision_score(Yva[:, i], pv[:, i])
               for i in range(len(scope_cols)) if Yva[:, i].sum() > 0]
        mpr = float(np.mean(prs))
        if mpr > best_val:
            best_val, best_state, bad = mpr, {k: v.clone() for k, v in proj.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 20 == 0 or bad == 0:
            print(f"  ep{ep:>3} loss={loss.item():.4f} val_mPRAUC={mpr:.4f}{' *' if bad==0 else ''}")
        if bad >= patience:
            print(f"  early stop @ ep{ep}"); break
    proj.load_state_dict(best_state)

    # fit per-head temperature on val (minimise BCE) — simple grid
    with torch.no_grad():
        zv = proj.proj(Xva)
    temps = torch.ones(len(scope_cols))
    grid = torch.linspace(0.5, 3.0, 26)
    for i in range(len(scope_cols)):
        if Yva[:, i].sum() == 0:
            continue
        best_t, best_l = 1.0, 1e9
        for t in grid:
            l = nn.functional.binary_cross_entropy_with_logits(zv[:, i] / t, Yva[:, i]).item()
            if l < best_l:
                best_l, best_t = l, float(t)
        temps[i] = best_t
    proj.temperature.copy_(temps)

    # ── L1 TRAINING POLICY: spec-optimised per-head thresholds (fit on val) ──
    if args.spec_opt:
        with torch.no_grad():
            pva = proj(Xva).numpy()
        base_va = 1.0 / (1.0 + np.exp(-feats[va][:, scope_cols]))
        thr, info = fit_policy_thresholds(pva, base_va, Yva.numpy(), scope_cols, args.max_sens_drop)
        proj.decision_threshold.copy_(torch.tensor(thr))
        print(f"\nL1 policy thresholds (max-F1, max sens drop {args.max_sens_drop:.0%}) — val:")
        for h, bs, fl, t, se, sp, f1, pr in info:
            print(f"  head {h:>3} base_sens={bs:.3f} floor={fl:.3f} -> tau={t:.3f} | "
                  f"val sens={se:.3f} spec={sp:.3f} F1={f1:.3f} PR-AUC={pr:.3f}")

    torch.save(proj.state_dict(), args.out)

    # ── evaluate fold-10 test: L1 vs BASE backbone head ──────────────────────
    def eval_block(mask_idx, tag):
        X = torch.tensor(feats[mask_idx]); y = Y[mask_idx]
        with torch.no_grad():
            p_l1 = proj(X).numpy()                       # calibrated L1
        # base = sigmoid of the raw scope-head logit (feature columns == head idx)
        p_base = 1 / (1 + np.exp(-feats[mask_idx][:, scope_cols]))
        m_l1, m_base = metrics(y, p_l1), metrics(y, p_base)
        print(f"\n=== TEST [{tag}] n={len(mask_idx)} — BASE head vs L1 projection ===")
        print(f"{'head':>3} {'name':<22} {'npos':>5} | {'AUROC base→L1':>16} | {'PR-AUC base→L1':>17} | {'PPV base→L1':>15}")
        for h in SCOPE_HEADS:
            b, l = m_base[h], m_l1[h]
            print(f"{h:>3} {str(b['n_pos']):>5} | "
                  f"{b['auroc']:.3f}→{l['auroc']:.3f} | {b['pr_auc']:.3f}→{l['pr_auc']:.3f} | "
                  f"{b['ppv']:.3f}→{l['ppv']:.3f}   ({h})")
        return m_l1, m_base

    # lead-II rows are the leading block only when lead-II was loaded
    n_leadii = 0 if args.fuzzy_only else (21799 if not args.angles else 21799)
    is_leadii = np.zeros(len(feats), bool); is_leadii[:n_leadii] = True
    te_leadii = np.array([i for i in te if is_leadii[i]], dtype=int)
    te_fuzzy = np.array([i for i in te if not is_leadii[i]], dtype=int)
    res = {}
    if len(te_leadii):
        res["leadII"] = eval_block(te_leadii, "PTB-XL lead-II")
    if len(te_fuzzy):
        res["fuzzy"] = eval_block(te_fuzzy, "fuzzy derived-lead")

    # write metrics csv
    with open(f"{OUT_DIR}/metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test_set", "head", "n_pos", "model", "auroc", "pr_auc", "sens", "ppv"])
        for tag, (m_l1, m_base) in res.items():
            for h in SCOPE_HEADS:
                for mdl, m in (("base", m_base[h]), ("L1", m_l1[h])):
                    w.writerow([tag, h, m["n_pos"], mdl,
                                f"{m['auroc']:.4f}", f"{m['pr_auc']:.4f}",
                                f"{m['sens']:.4f}", f"{m['ppv']:.4f}"])
    print(f"\nsaved {OUT_DIR}/scope_projection.pth + metrics.csv | temps={temps.numpy().round(2)}")


if __name__ == "__main__":
    main()
