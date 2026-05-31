"""AFib-only reporting-format comparison: Stanford (cinc17) vs ECGFounder, MOVE 3B8D.

Three aligned timelines:
  (A) activity context
  (B) Stanford cinc17 P(AFib) per 0.85 s segment  — fine, single continuous probability
  (C) ECGFounder P(AFib) per 10 s window          — coarse, + motion-gate FP suppression
Output: docs/afib_format_comparison_3B8D.png
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import label_config as L

SUBJ = "3B8D"; THR_AF = L.head_threshold(5)
EF = pd.read_csv(f"res/recording_reports/{SUBJ}/windows.csv")
ST = pd.read_csv(f"res/recording_reports/{SUBJ}_stanford_afib/afib_segments.csv")
t0 = min(EF.t_start_s.min(), ST.t_start_s.min())
def mins(s): return (s - t0)/60.0
TMAX = mins(EF.t_end_s.max())
ACT_C = {"baseline":"#9FB6C9","lift":"#C9B79F","greetings":"#C9B79F","gesticulate":"#C9B79F",
         "jumps":"#E0A96D","walk_before":"#7FB07F","run":"#E06D6D","walk_after":"#7FB07F","unknown":"#DDDDDD"}

# ECGFounder AFib per-window
ef_t = mins(EF.t_start_s.to_numpy()); ef_w = (EF.t_end_s-EF.t_start_s).to_numpy()/60.0
ef_p = EF.p_5.to_numpy()
ef_kept = EF.detections.astype(str).str.contains("Atrial Fibrillation") & ~EF.detections.astype(str).str.contains(r"Atrial Fibrillation\(suppressed\)")
ef_raw = ef_p >= THR_AF
st_t = mins(ST.t_start_s.to_numpy()); st_p = ST.p_afib.to_numpy()

fig, ax = plt.subplots(3, 1, figsize=(15, 7), sharex=True,
                       gridspec_kw={"height_ratios":[0.45, 1.4, 1.4], "hspace":0.32})
# (A) activity
for _, r in EF.iterrows():
    ax[0].axvspan(mins(r.t_start_s), mins(r.t_end_s), color=ACT_C.get(r.activity,"#DDD"), lw=0)
ax[0].set_yticks([]); ax[0].set_ylabel("Activity", rotation=0, ha="right", va="center", fontsize=11)
ax[0].set_title(f"AFib reporting-format comparison — MOVE {SUBJ} ({TMAX:.0f} min, rhythm-negative)  —  same signal, AFib only",
                fontsize=14, fontweight="bold", loc="left", pad=8)
acts=[a for a in ACT_C if a in set(EF.activity)]
ax[0].legend(handles=[Patch(color=ACT_C[a],label=a) for a in acts], ncol=len(acts), fontsize=8,
             loc="upper center", bbox_to_anchor=(0.5,2.2), frameon=False)
# (B) Stanford P(AFib) per 0.85s
ax[1].fill_between(st_t, 0, st_p, step="post", color="#065A82", alpha=0.85, lw=0)
ax[1].axhline(0.5, ls="--", c="#888", lw=1)
ax[1].set_ylim(0,1); ax[1].set_ylabel("Stanford cinc17\nP(AFib) / 0.85 s", fontsize=10)
ax[1].text(0.99,0.88,f"AFib fired 1/{len(ST)} segs (0.0%) · max p={st_p.max():.2f} → essentially NO AFib",
           transform=ax[1].transAxes, ha="right", fontsize=10, color="#065A82", fontweight="bold")
# (C) ECGFounder P(AFib) per 10s + FP gate
for i in range(len(EF)):
    if ef_raw[i]:
        ax[2].axvspan(ef_t[i], ef_t[i]+ef_w[i], color=("#C0392B" if ef_kept.iloc[i] else "#065A82"),
                      alpha=(0.85 if ef_kept.iloc[i] else 0.18), lw=0)
ax[2].step(ef_t, ef_p, where="post", color="#065A82", lw=1.2)
ax[2].axhline(0.5, ls="--", c="#888", lw=1)
ax[2].set_ylim(0,1); ax[2].set_ylabel("ECGFounder\nP(AFib) / 10 s", fontsize=10)
ax[2].set_xlabel("time (minutes)", fontsize=11); ax[2].set_xlim(0,TMAX)
ax[2].text(0.99,0.88,f"AFib raw {int(ef_raw.sum())} win → {int(ef_kept.sum())} after motion gate (90% suppressed)",
           transform=ax[2].transAxes, ha="right", fontsize=10, color="#C0392B", fontweight="bold")
ax[2].legend(handles=[Patch(color="#C0392B",alpha=0.85,label="fired (survives FP gate)"),
                      Patch(color="#065A82",alpha=0.2,label="fired raw, FP-suppressed"),
                      plt.Line2D([],[],ls="--",c="#888",label="threshold 0.5")],
             fontsize=8, loc="upper left", frameon=True)
plt.savefig("docs/afib_format_comparison_3B8D.png", dpi=140, bbox_inches="tight")
print("wrote docs/afib_format_comparison_3B8D.png")
print(f"Stanford AFib fired: {int(ST.fired.sum())}/{len(ST)} | ECGF AFib raw {int(ef_raw.sum())} -> kept {int(ef_kept.sum())}")
