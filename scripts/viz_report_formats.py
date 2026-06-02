"""Visualize the two reporting formats on the same MOVE recording (subject 3B8D).

Three stacked timelines on a shared time axis:
  (A) Activity context (MOVE annotation)             — why some detections occur
  (B) Stanford: per-0.71 s rhythm sequence           — fine, single-rhythm-per-segment
  (C) ECGFounder: per-10 s multi-label window rows    — coarse, multi-label + FP suppression

Output: docs/report_format_comparison_3B8D.png
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import label_config as L

SUBJ = "3B8D"
EF = pd.read_csv(f"res/recording_reports/{SUBJ}/windows.csv")
ST = pd.read_csv(f"res/recording_reports/{SUBJ}_stanford/segments.csv")
t0 = min(EF.t_start_s.min(), ST.t_start_s.min())
def mins(s): return (s - t0) / 60.0
TMAX = mins(EF.t_end_s.max())

# ---- colors ----
ACT_C = {"baseline":"#9FB6C9","lift":"#C9B79F","greetings":"#C9B79F","gesticulate":"#C9B79F",
         "jumps":"#E0A96D","walk_before":"#7FB07F","run":"#E06D6D","walk_after":"#7FB07F","unknown":"#DDDDDD"}
ST_C  = {"SINUS":"#3C9A5F","BIGEMINY":"#E0A33D","TRIGEMINY":"#D2691E","VT":"#C0392B"}
# Scope heads only (Normal/NSR removed from DETECTION_SCOPE 2026-06-01 → no p_2 column).
EF_HEADS = [(6,"Sinus Tachy","#1C7293"),(5,"AFib","#065A82"),(98,"V Run (VT)","#C0392B"),
            (93,"SV Run","#8E6FB0"),(4,"Bradycardia","#5A8F69"),(142,"Pause","#B07F2A")]

fig, ax = plt.subplots(3, 1, figsize=(15, 8.2), sharex=True,
                       gridspec_kw={"height_ratios":[0.6, 0.9, 3.4], "hspace":0.28})

# (A) activity context
for _, r in EF.iterrows():
    ax[0].axvspan(mins(r.t_start_s), mins(r.t_end_s), color=ACT_C.get(r.activity,"#DDD"), lw=0)
ax[0].set_yticks([]); ax[0].set_ylabel("Activity", rotation=0, ha="right", va="center", fontsize=11)
ax[0].set_title(f"Reporting-format comparison on a MOVE recording (subject {SUBJ}, {TMAX:.0f} min)  —  same signal, two models",
                fontsize=14, fontweight="bold", loc="left", pad=10)
acts = [a for a in ACT_C if a in set(EF.activity)]
ax[0].legend(handles=[Patch(color=ACT_C[a], label=a) for a in acts], ncol=len(acts),
             fontsize=8, loc="upper center", bbox_to_anchor=(0.5,1.9), frameon=False)

# (B) Stanford per-0.71s sequence  (imshow of class index over time)
cls = ["SINUS","BIGEMINY","TRIGEMINY","VT"]
cidx = ST.label.map({c:i for i,c in enumerate(cls)}).to_numpy()
from matplotlib.colors import ListedColormap
cmap = ListedColormap([ST_C[c] for c in cls])
ax[1].imshow(cidx[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=3,
             extent=[mins(ST.t_start_s.min()), mins(ST.t_end_s.max()), 0, 1], interpolation="nearest")
ax[1].set_yticks([]); ax[1].set_ylabel("Stanford\n0.71 s/seg\n(1 rhythm)", rotation=0, ha="right", va="center", fontsize=10)
ax[1].text(0.5, -0.32, f"{len(ST)} segments — 100% SINUS, 0 arrhythmia (one rhythm per 0.71 s)",
           transform=ax[1].transAxes, ha="center", fontsize=9, color="#444")

# (C) ECGFounder per-10s multi-label rows (raw-fired light, after-FP solid)
def raw_fired(row,h): return row[f"p_{h}"] >= L.head_threshold(h)
def kept(row,h):
    d=str(row["detections"]); name={6:"Sinus Tachycardia",5:"Atrial Fibrillation",98:"Ventricular Run",
        93:"Supraventricular Run",4:"Bradycardia",142:"Pause",2:"Normal ECG"}[h]
    return (name in d) and (f"{name}(suppressed)" not in d)
rows=[h for h in EF_HEADS]
for yi,(h,lab,c) in enumerate(rows):
    y=len(rows)-1-yi
    for _, r in EF.iterrows():
        if raw_fired(r,h):
            x0=mins(r.t_start_s); w=mins(r.t_end_s)-x0
            if kept(r,h):
                ax[2].add_patch(plt.Rectangle((x0,y+0.12),w,0.76,color=c,lw=0))
            else:  # raw-fired but FP-suppressed
                ax[2].add_patch(plt.Rectangle((x0,y+0.12),w,0.76,facecolor=c,alpha=0.22,
                                              hatch="////",edgecolor=c,lw=0.3))
ax[2].set_ylim(0,len(rows)); ax[2].set_yticks([len(rows)-1-i+0.5 for i in range(len(rows))])
ax[2].set_yticklabels([lab for _,lab,_ in rows], fontsize=10)
ax[2].set_ylabel("ECGFounder — 10 s/window (multi-label)", fontsize=11)
ax[2].set_xlabel("time (minutes)", fontsize=11); ax[2].set_xlim(0, TMAX); ax[2].grid(axis="x", alpha=0.2)
ax[2].legend(handles=[Patch(color="#444",label="fired (after FP)"),
                      Patch(facecolor="#444",alpha=0.25,hatch="////",label="fired raw, FP-suppressed")],
             fontsize=9, loc="upper right", frameon=True)

plt.savefig("docs/report_format_comparison_3B8D.png", dpi=140, bbox_inches="tight")
print("wrote docs/report_format_comparison_3B8D.png")
print(f"Stanford: {len(ST)} segs @0.71s (all SINUS)  |  ECGFounder: {len(EF)} windows @10s, multi-label")
