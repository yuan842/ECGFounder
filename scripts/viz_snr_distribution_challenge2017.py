"""S0 (raw) snr_proxy: distribution across AFib / Normal / Noisy on Challenge 2017,
plus a recommended cutoff that best separates Noisy from clean (AFib + Normal).

snr_proxy = var(5-25 Hz band) / var(out-of-band residual)  (sqi.compute_band_sqi),
on the center 10 s RAW window (S0, prior to preprocessing). Uses ALL records per
label (no model → no split needed): AFib 758, Normal 5076, Noisy 279.

Noisy windows have LOW snr_proxy → rule: **flag NOISY if snr_proxy < τ**. We sweep
τ and pick the Youden-optimal point (max sensitivity+specificity−1, prevalence-
independent), and report ROC-AUC of snr_proxy as a noise detector.

Outputs:
  docs/snr_distribution_challenge2017.png   — distribution (+ cutoff & reject region) | ROC
  res/challenge2017/SNR_S0_CI.md            — CIs, separation, cutoff recommendation
Run:  python3 -m scripts.viz_snr_distribution_challenge2017
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve

import sqi
from scripts.eval_sqi_stages_challenge2017 import center_window, WIN0, FS0, GROUPS, DATA, SPLIT

OUT_PNG = "docs/snr_distribution_challenge2017.png"
OUT_MD = "res/challenge2017/SNR_S0_CI.md"
COL = {"A": "#C0392B", "N": "#1C7293", "~": "#7A6F9B"}
NBOOT, SEED = 10000, 42


def boot_ci(x, fn, nboot=NBOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(x)
    stats = np.array([fn(x[rng.randint(0, n, n)]) for _ in range(nboot)])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def best_cutoff(noisy, clean):
    """Youden-optimal τ for rule 'noisy if snr < τ'. Returns dict + full sweep."""
    y = np.r_[np.ones(len(noisy)), np.zeros(len(clean))]
    s = np.r_[noisy, clean]
    auc = roc_auc_score(y, -s)
    grid = np.linspace(0.0, float(np.percentile(s, 99.5)), 800)
    sweep = []
    for tau in grid:
        pred = s < tau
        tp = np.sum(pred & (y == 1)); fn = np.sum(~pred & (y == 1))
        fp = np.sum(pred & (y == 0)); tn = np.sum(~pred & (y == 0))
        sens = tp / (tp + fn); spec = tn / (tn + fp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
        sweep.append((tau, sens, spec, sens + spec - 1, prec, f1))
    R = np.array(sweep)
    j = int(np.argmax(R[:, 3]))
    bal = int(np.argmin(np.abs(R[:, 1] - R[:, 2])))
    return dict(auc=auc, R=R, y=y, s=s,
                star=R[j], bal=R[bal])


def main():
    data = np.load(DATA, allow_pickle=True)
    sp = pd.read_csv(SPLIT)
    snr = {}
    for g in GROUPS:
        ids = sp[sp.label_str == g].idx.values
        vals = []
        for rid in ids:
            w = center_window(data[rid], WIN0)
            if np.std(w) < 1e-9:
                continue
            v = sqi.compute_band_sqi(w, FS0).get("snr_proxy", np.nan)
            if np.isfinite(v):
                vals.append(v)
        snr[g] = np.array(vals, float)
        print(f"{GROUPS[g]}: n={len(vals)}")

    clean = np.concatenate([snr["A"], snr["N"]])
    noisy = snr["~"]
    cut = best_cutoff(noisy, clean)
    tau_star, sens_s, spec_s, j_s, prec_s, f1_s = cut["star"]
    tau_bal, sens_b, spec_b, j_b, prec_b, f1_b = cut["bal"]

    # ── stats table ──
    rows = []
    for g in GROUPS:
        x = snr[g]
        rows.append((GROUPS[g], len(x), float(np.mean(x)), boot_ci(x, np.mean),
                     float(np.median(x)), boot_ci(x, np.median),
                     float(np.percentile(x, 25)), float(np.percentile(x, 75))))

    md = ["# Challenge 2017 — S0 (raw) snr_proxy: distribution, CIs & Noisy cutoff",
          "",
          "snr_proxy on the center 10 s RAW window (prior to preprocessing). 95% CIs are "
          f"bootstrap percentile intervals ({NBOOT:,} resamples, seed {SEED}).",
          "",
          "| group | n | mean | 95% CI (mean) | median | 95% CI (median) | IQR |",
          "|---|---:|---:|---|---:|---|---|"]
    for name, n, mean, cim, med, cimd, q1, q3 in rows:
        md.append(f"| {name} | {n} | {mean:.3f} | [{cim[0]:.3f}, {cim[1]:.3f}] | "
                  f"{med:.3f} | [{cimd[0]:.3f}, {cimd[1]:.3f}] | [{q1:.3f}–{q3:.3f}] |")

    md += ["", "## Pairwise separation (Mann–Whitney U, two-sided)", "",
           "| pair | U p-value |", "|---|---|"]
    for a, b in [("A", "N"), ("A", "~"), ("N", "~")]:
        _, p = mannwhitneyu(snr[a], snr[b], alternative="two-sided")
        md.append(f"| {GROUPS[a]} vs {GROUPS[b]} | {p:.2e} |")

    md += ["",
           "## Recommended cutoff — Noisy vs clean (AFib + Normal)",
           "",
           f"Rule: **flag NOISY if snr_proxy < τ**. snr_proxy as a noise detector has "
           f"**ROC-AUC = {cut['auc']:.3f}** (clean n={len(clean)}, noisy n={len(noisy)}).",
           "",
           "| criterion | τ | sensitivity (Noisy) | specificity (clean) | Youden J | precision | F1 |",
           "|---|---:|---:|---:|---:|---:|---:|",
           f"| **Youden-optimal (recommended)** | **{tau_star:.2f}** | {sens_s:.3f} | {spec_s:.3f} | {j_s:.3f} | {prec_s:.3f} | {f1_s:.3f} |",
           f"| balanced (sens≈spec) | {tau_bal:.2f} | {sens_b:.3f} | {spec_b:.3f} | {j_b:.3f} | {prec_b:.3f} | {f1_b:.3f} |",
           "",
           "### Sweep near the recommendation",
           "",
           "| τ | sens (Noisy) | spec (clean) | Youden J |", "|---:|---:|---:|---:|"]
    R = cut["R"]
    for tau in [tau_star - 0.2, tau_star, tau_star + 0.2, tau_star + 0.5, tau_star + 1.0]:
        if tau <= 0:
            continue
        i = int(np.argmin(np.abs(R[:, 0] - tau)))
        md.append(f"| {R[i,0]:.2f} | {R[i,1]:.3f} | {R[i,2]:.3f} | {R[i,3]:.3f} |")
    md += ["",
           f"> **Recommendation: gate on `snr_proxy < {tau_star:.2f}` at S0** — catches "
           f"{sens_s:.0%} of Noisy while keeping {spec_s:.0%} of clean AFib/Normal. "
           "Lower τ → fewer clean rejects; higher τ → more noise caught. The Noisy and "
           "clean distributions overlap (AUC≈" f"{cut['auc']:.2f}), so SNR alone is a useful "
           "but not perfect gate — combine with baseline-drift for a stronger rule."]
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")

    # ── figure: distribution + cutoff  |  ROC ──
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6),
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    fig.suptitle("Challenge 2017 — raw (S0) snr_proxy: AFib vs Normal vs Noisy + recommended Noisy cutoff",
                 fontsize=15, fontweight="bold")

    xmax = float(np.percentile(np.concatenate([snr[g] for g in GROUPS]), 99))
    xs = np.linspace(0, xmax, 400)
    for g in GROUPS:
        x = snr[g]
        axL.hist(x, bins=45, range=(0, xmax), density=True, alpha=0.22, color=COL[g])
        kde = gaussian_kde(x)
        axL.plot(xs, kde(xs), color=COL[g], lw=2.4, label=f"{GROUPS[g]} (n={len(x)})")
        axL.fill_between(xs, kde(xs), color=COL[g], alpha=0.10)
    axL.axvspan(0, tau_star, color="#888", alpha=0.10)
    axL.axvline(tau_star, color="black", ls="--", lw=2.2, label=f"recommended τ = {tau_star:.2f}")
    axL.text(tau_star + 0.05, axL.get_ylim()[1]*0.96, "← flag NOISY", fontsize=10,
             color="#333", va="top", ha="left", fontweight="bold")
    axL.set_title(f"Distributions (S0 raw) — reject region snr<τ shaded", fontsize=12)
    axL.set_xlabel("snr_proxy  (band 5–25 Hz power / residual)"); axL.set_ylabel("density")
    axL.set_xlim(0, xmax); axL.legend(fontsize=10.5, frameon=False); axL.grid(alpha=0.2)

    fpr, tpr, _ = roc_curve(cut["y"], -cut["s"])
    axR.plot(fpr, tpr, color="#0B5394", lw=2.4, label=f"ROC (AUC = {cut['auc']:.3f})")
    axR.plot([0, 1], [0, 1], color="#BBB", ls=":", lw=1)
    axR.scatter([1 - spec_s], [sens_s], color="#C0392B", s=85, zorder=5,
                label=f"τ* = {tau_star:.2f}\nsens {sens_s:.2f} · spec {spec_s:.2f}")
    axR.set_title("Noisy detector ROC (Noisy = positive)", fontsize=12)
    axR.set_xlabel("1 − specificity  (clean flagged Noisy)")
    axR.set_ylabel("sensitivity  (Noisy caught)")
    axR.set_xlim(0, 1); axR.set_ylim(0, 1.02); axR.legend(fontsize=10.5, frameon=False, loc="lower right")
    axR.grid(alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print("\n".join(md)); print(f"\nwrote {OUT_PNG} and {OUT_MD}")


if __name__ == "__main__":
    main()
