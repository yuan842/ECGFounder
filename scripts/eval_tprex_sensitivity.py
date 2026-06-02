"""ecg-tp_rex (Belgian Holter) — base ECGFounder single-lead sensitivity eval.

The tp_rex val split is a TP-only cohort (every row True Event=True), so the
faithful metric is per-class SENSITIVITY (recall) at the head threshold — there
are no real per-class negatives for ROC/PR. Rows whose JSON is not present on
disk are skipped and reported (so the number is honest about coverage).

Pipeline: ECGPreprocessor.for_fzark() → base backbone (1-lead) → sigmoid →
fire if prob[mapped head] >= threshold. Reports base@0.5 and, for scope heads,
the device scope threshold too.

Writes res/tprex_eval/tprex_sensitivity.csv + .md.
Run:  python3 -m scripts.eval_tprex_sensitivity
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from preprocessing import ECGPreprocessor
from checkpoints import load_ecgfounder
from device_utils import resolve_device
from label_config import FZARK_LABEL_MAP as LABEL_MAP
import label_config as L

DATA_DIR = "data/ecg-tp_rex"
VAL_CSV = "data/ecg-tp_rex/splits/val.csv"
OUT = "res/tprex_eval"


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(VAL_CSV)
    df["json_path"] = df["JSON File"].map(lambda r: os.path.join(DATA_DIR, str(r)))
    df["exists"] = df["json_path"].map(os.path.exists)
    have = df[df["exists"]].reset_index(drop=True)
    miss = df[~df["exists"]]
    print(f"val rows: {len(df)}  | files present: {len(have)}  | missing: {len(miss)}")

    device = resolve_device()
    model = load_ecgfounder(device); model.eval()
    prep = ECGPreprocessor.for_fzark()

    # preprocess + infer (skip unparseable)
    xs, ok = [], []
    for p in tqdm(have["json_path"], desc="preprocess"):
        try:
            xs.append(prep.from_fzark_json(p)); ok.append(True)
        except Exception as e:
            xs.append(torch.zeros(1, 5000)); ok.append(False)
    have["parsed"] = ok
    probs = np.zeros((len(xs), 150), np.float32)
    with torch.no_grad():
        for i in range(0, len(xs), 64):
            b = torch.stack(xs[i:i+64]).to(device)
            probs[i:i+64] = torch.sigmoid(model(b)).cpu().numpy()

    scope = set(L.scope_indices())
    rows = []
    for ev, g in have.groupby("Event Type"):
        h = LABEL_MAP.get(ev)
        idx = g.index.values
        parsed = g["parsed"].values
        n = int(parsed.sum())
        if h is None or n == 0:
            rows.append(dict(event=ev, head=h, n=n, mapped="unmapped" if h is None else "ok",
                             sens_base=np.nan, sens_scope=np.nan, mean_p=np.nan, median_p=np.nan))
            continue
        p = probs[idx[parsed], h]
        in_sc = h in scope
        thr = L.head_threshold(h) if in_sc else 0.5
        rows.append(dict(event=ev, head=h, n=n,
                         mapped=("scope" if in_sc else "base-only(out-of-scope head)"),
                         sens_base=float((p >= 0.5).mean()),
                         sens_scope=float((p >= thr).mean()) if in_sc else np.nan,
                         scope_thr=thr if in_sc else np.nan,
                         mean_p=float(p.mean()), median_p=float(np.median(p))))

    res = pd.DataFrame(rows).sort_values("n", ascending=False)
    res.to_csv(f"{OUT}/tprex_sensitivity.csv", index=False)

    md = ["# ecg-tp_rex — base ECGFounder single-lead sensitivity (TP-only cohort)",
          "",
          f"- Val rows: **{len(df)}**; files present on disk: **{len(have)}**; "
          f"missing (not evaluable): **{len(miss)}**.",
          "- Every row is `True Event=True` → metric is **sensitivity** (recall) at the "
          "head threshold; ROC/PR are not defined (no per-class negatives).",
          "- `sens_base` = fire rate at base@0.5; `sens_scope` = at the device scope "
          "threshold (scope heads only).",
          "",
          "## Files present per class (val split)", "",
          "| event | present/total |", "|---|---|"]
    pc = df.assign(e=df["exists"]).groupby("Event Type")["e"].agg(["sum", "count"])
    for ev, r in pc.iterrows():
        md.append(f"| {ev} | {int(r['sum'])}/{int(r['count'])} |")

    md += ["", "## Per-class sensitivity (evaluable classes)", "",
           "| event | head | n | routing | sens base@0.5 | sens @scope-τ | mean p | median p |",
           "|---|---:|---:|---|---:|---:|---:|---:|"]
    for _, r in res.iterrows():
        def f(x): return "—" if (isinstance(x, float) and np.isnan(x)) else (f"{x:.3f}" if isinstance(x, float) else str(x))
        md.append(f"| {r['event']} | {r['head']} | {r['n']} | {r['mapped']} | "
                  f"{f(r['sens_base'])} | {f(r.get('sens_scope', np.nan))} | "
                  f"{f(r['mean_p'])} | {f(r['median_p'])} |")
    if len(miss):
        md += ["", f"> ⚠ {len(miss)} records across "
               f"{miss['Event Type'].nunique()} classes "
               f"({', '.join(sorted(miss['Event Type'].unique()))}) have no JSON on disk "
               "— not evaluated. Stage the files to extend coverage."]

    with open(f"{OUT}/tprex_sensitivity.md", "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md)); print(f"\nwrote {OUT}/tprex_sensitivity.csv + .md")


if __name__ == "__main__":
    main()
