"""Challenge 2017 — AFib performance WITH vs WITHOUT the S0 SQI gate,
using the DEVICE-SPECIFIC AliveCor-fine-tuned L1 (alivecorSL.pth).

Routing = AliveCor L1 (AFib head fine-tuned on Challenge 2017; other heads frozen
at fzark) + L2 GT-matched. AFib fire threshold = the L1's fitted τ. The S0 SQI gate
(overlay.signal_quality_gate, device="alivecor" → default τ: NOISY ⇔ snr_proxy<=0.10)
runs per 10 s window on the RAW signal; a Noisy window is skipped (no detection).

Recording-level (any-window-fires):
  • WITHOUT gate : AFib fires if ANY window L2-fires            (= current production)
  • WITH gate    : AFib fires if any NON-gated window L2-fires; an all-Noisy
                   recording is "uninterpretable" (no detection).

Reported on the held-out TEST split. Binary metrics use clean A vs N; O/~ are
context. Also reports window-gate rate and per-group uninterpretable rate.

Writes res/challenge2017/SQI_GATE_COMPARE.md.
Run:  python3 -m scripts.eval_challenge2017_sqi_gate
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.signal import butter, filtfilt

from preprocessing import ECGPreprocessor
from checkpoints import load_ecgfounder
from device_utils import resolve_device
from overlay.scope_overlay import ScopeProjection, SCOPE_HEADS
from overlay.device_config import load_device_config
from overlay.arbiter import arbitrate, ArbiterConfig, AFIB
from overlay.types import ScopeScores
from overlay.signal_quality_gate import SignalQualityGate

DATA = "data/challenge2017/Challenge2017/raw_data.npy"
SPLIT = "csv/challenge2017_split.csv"
DEVCFG = "res/scope_overlay/device_configs/fzark.json"
OUT = "res/challenge2017/SQI_GATE_COMPARE.md"
FS_IN, WIN = 300, 3000
GATE_DEVICE = "alivecor"             # Challenge 2017 = AliveCor


def raw_snr(sig, fs=FS_IN):
    b, a = butter(2, [5.0 / (fs / 2), 25.0 / (fs / 2)], btype="bandpass")
    bp = filtfilt(b, a, sig)
    return float(np.var(bp) / (np.var(sig - bp) + 1e-9))


def windows(sig):
    n = len(sig) // WIN
    return [sig] if n == 0 else [sig[i*WIN:(i+1)*WIN] for i in range(n)]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    data = np.load(DATA, allow_pickle=True)
    sp = pd.read_csv(SPLIT)
    te = sp[sp.split == "test"].reset_index(drop=True)

    device = resolve_device()
    backbone = load_ecgfounder(device); backbone.eval()
    L1_CKPT = "res/scope_overlay/alivecorSL.pth"        # device-specific AliveCor L1
    l1 = ScopeProjection(); l1.load_state_dict(torch.load(L1_CKPT, map_location="cpu")); l1.eval()
    dc = load_device_config(DEVCFG)
    l1_heads = set(dc.l1_heads()); base_heads = frozenset(dc.base_heads())
    fire = {h: 0.5 for h in SCOPE_HEADS}; fire.update(dc.fire_thresholds())
    fire[AFIB] = float(l1.thresholds()[AFIB])           # AliveCor-fitted AFib τ
    cfg = ArbiterConfig(enabled=True, fire_threshold=fire, uncalibrated_heads=base_heads)
    gate = SignalQualityGate(enabled=True, device=GATE_DEVICE)
    prep = ECGPreprocessor(powerline_hz=60, normalize="winsorize")
    print(f"test={len(te)}  gate: snr {'<=' if gate.snr_inclusive else '<'} {gate.min_snr} ({GATE_DEVICE})")

    # build windows: backbone tensor + raw snr + owner
    xs, snrs, owner = [], [], []
    for ri in tqdm(range(len(te)), desc="prep"):
        sig = np.asarray(data[te.idx[ri]], dtype=np.float64)
        for w in windows(sig):
            try:
                xs.append(prep.process(w, fs_in=FS_IN))
            except Exception:
                xs.append(torch.zeros(1, 5000))
            snrs.append(raw_snr(w / 1000.0))      # S0 raw SNR (scale-free; /1000 mV harmless)
            owner.append(ri)
    owner = np.array(owner); snrs = np.array(snrs)

    # per-window L2 AFib fire + gate flag
    fire_w = np.zeros(len(xs), bool); gated_w = np.zeros(len(xs), bool)
    with torch.no_grad():
        for s in tqdm(range(0, len(xs), 128), desc="score", leave=False):
            b = torch.stack(xs[s:s+128]).to(device)
            logit = backbone(b)
            base = torch.sigmoid(logit).cpu().numpy()
            l1p = l1(logit.cpu()).numpy()
            for k in range(len(b)):
                pr = {h: (float(l1p[k, j]) if h in l1_heads else float(base[k, h]))
                      for j, h in enumerate(SCOPE_HEADS)} | {1: float(base[k, 1])}
                d = arbitrate(ScopeScores(probs=pr, nsr_score=pr[1]), cfg)
                fire_w[s+k] = d[AFIB].fired
                gated_w[s+k] = not gate.gate(None, sqi={"snr_proxy": float(snrs[s+k])}).passed

    # recording rollups
    n = len(te); lab = te.label_str.values
    fire_nogate = np.zeros(n, bool); fire_gate = np.zeros(n, bool)
    uninterp = np.zeros(n, bool)
    for ri in range(n):
        m = owner == ri
        fire_nogate[ri] = fire_w[m].any()
        surviving = m & ~gated_w
        uninterp[ri] = not surviving.any()                       # all windows gated
        fire_gate[ri] = fire_w[surviving].any()                  # fire among non-gated

    A, N, O, T = lab == "A", lab == "N", lab == "O", lab == "~"
    avn = A | N

    def metrics(rec_fire):
        sens = float(rec_fire[A].mean()); spec = 1 - float(rec_fire[N].mean())
        ppv = (rec_fire & A).sum() / max(1, (rec_fire & avn).sum())
        f1 = 2*ppv*sens/(ppv+sens) if (ppv+sens) else float("nan")
        return dict(sens=sens, spec=spec, ppv=ppv, f1=f1,
                    frO=float(rec_fire[O].mean()), frT=float(rec_fire[T].mean()))

    mng, mg = metrics(fire_nogate), metrics(fire_gate)
    win_gate_rate = {g: float(gated_w[np.isin(owner, np.where(lab == g)[0])].mean()) for g in "ANO~"}
    uninterp_rate = {g: float(uninterp[lab == g].mean()) for g in "ANO~"}

    md = ["# Challenge 2017 — AFib performance WITH vs WITHOUT the S0 SQI gate", "",
          "Routing = **device-specific AliveCor-fine-tuned L1** (AFib head fit on "
          f"Challenge 2017, others frozen at fzark; AFib τ={fire[AFIB]:.3f}) + L2.", "",
          f"Held-out TEST (n={n}: A={int(A.sum())}, N={int(N.sum())}, O={int(O.sum())}, "
          f"~={int(T.sum())}). Gate = S0 SQI, device `{GATE_DEVICE}`→default "
          f"(NOISY ⇔ snr_proxy {'<=' if gate.snr_inclusive else '<'} "
          f"{gate.min_snr}; baseline-drift gating off). Per-window gate; recording "
          "fires AFib if any non-gated window fires. Binary metrics: clean A vs N.", "",
          "## AFib detection — with vs without gate", "",
          "| metric | without gate | with gate | Δ |", "|---|---:|---:|---:|"]
    for key, l0 in (("sens", "Sensitivity (A)"), ("spec", "Specificity (N)"),
                    ("ppv", "PPV"), ("f1", "F1"),
                    ("frO", "Fire-rate Other ↓"), ("frT", "Fire-rate Noisy ↓")):
        md.append(f"| {l0} | {mng[key]:.3f} | {mg[key]:.3f} | {mg[key]-mng[key]:+.3f} |")
    md += ["",
           "## Gate activity (per group)", "",
           "| group | windows gated Noisy | recordings fully uninterpretable |",
           "|---|---:|---:|"]
    nm = {"A": "AFib", "N": "Normal", "O": "Other", "~": "Noisy"}
    for g in "AN O~".replace(" ", ""):
        md.append(f"| {nm[g]} | {win_gate_rate[g]:.1%} | {uninterp_rate[g]:.1%} |")
    md += ["",
           "## Read", "",
           f"- The gate skips noisy windows before detection. On **Noisy (~)** it gates "
           f"{win_gate_rate['~']:.0%} of windows, cutting their AFib fire-rate "
           f"{mng['frT']:.3f}→{mg['frT']:.3f} and Other {mng['frO']:.3f}→{mg['frO']:.3f}.",
           f"- AFib sensitivity {mng['sens']:.3f}→{mg['sens']:.3f}, Normal specificity "
           f"{mng['spec']:.3f}→{mg['spec']:.3f} (clean groups are barely gated: "
           f"AFib {win_gate_rate['A']:.1%}, Normal {win_gate_rate['N']:.1%} of windows).",
           "- 'Uninterpretable' = every window of a recording gated Noisy → no detection "
           "emitted (reported, not counted as a positive)."]

    with open(OUT, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md)); print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
