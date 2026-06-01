"""L2 — deterministic multi-label rule arbiter.

Pure function: ScopeScores → {head: Decision}. No learned weights; thresholds
are versioned constants. SUPPRESS / MERGE only — never promotes a head (raising
sensitivity is L1's job). Implements res/ptbxl_cofiring/MULTI_LABEL_RULES.md.

Precedence (signal-quality stage lives in the SQG, upstream — NOT here):
  1. validity mask        (Pause needs adequate clean signal length)
  2. NSR contradiction    (high head-1 score + weak arrhythmia ⇒ suppress)
  3. AFib ▸ Tachy         (both fired ⇒ drop Tachy; AF-RVR, Tachy co-fire PPV 1%)
  4. mutual exclusion     (Brady⊕Tachy, Brady⊕AFib ⇒ keep higher-confidence)
  5. HR plausibility      (Bradycardia requires hr_bpm ≤ ~56)
  6. run cluster          (SVT+V-Run kept; collapsed to one alert by to_alerts())
"""
from __future__ import annotations

from label_config import DETECTION_SCOPE
from overlay.types import ScopeScores, Decision, RunAlert

SCOPE_HEADS: list[int] = sorted(DETECTION_SCOPE)   # [4, 5, 6, 93, 98, 142]

# head indices
BRADY, AFIB, TACHY, SVT, VRUN, PAUSE = 4, 5, 6, 93, 98, 142
NSR = 1   # out-of-scope feature head

# ── thresholds (TODO: calibrate from res/ptbxl_cofiring/*.csv) ──────────────
TAU_FIRE = 0.5      # L1 prob ≥ this ⇒ candidate firing
TAU_NSR = 0.5       # NSR-contradiction: head-1 score ≥ this AND …
TAU_WEAK = 0.70     # … arrhythmia score < this ⇒ suppress it
HR_BRADY_MAX = 56.3 # bpm; reclassified from the old SQI filter
RUN_DELTA = 0.20    # (reserved) max gap to treat SVT/V-Run as co-firing

MERGE_SVT_VT = True  # Decision 2 = merge for alerting (keep both in audit)


def arbitrate(s: ScopeScores) -> dict[int, Decision]:
    p = {h: float(s.probs.get(h, 0.0)) for h in SCOPE_HEADS}
    fired = {h: p[h] >= TAU_FIRE for h in SCOPE_HEADS}
    reason = {h: ("fired" if fired[h] else "below-threshold") for h in SCOPE_HEADS}

    def suppress(h: int, why: str) -> None:
        if fired[h]:
            fired[h] = False
            reason[h] = f"suppressed: {why}"

    # 1. validity mask (placeholder — Pause needs adequate clean signal length)
    #    left as a no-op until a length/quality signal is wired in.

    # 2. NSR contradiction (head-1 score is an FP feature, not a detection)
    if s.nsr_score >= TAU_NSR:
        for h in (AFIB, TACHY):
            if fired[h] and p[h] < TAU_WEAK:
                suppress(h, f"NSR-contradiction (nsr={s.nsr_score:.2f}, p={p[h]:.2f})")

    # 3. AFib ▸ Tachy  (AF with rapid ventricular response double-counted)
    if fired[AFIB] and fired[TACHY]:
        suppress(TACHY, "AFib▸Tachy (AF-RVR; Tachy is the FP member)")

    # 4. mutual exclusion — keep the higher-confidence head
    for a, b in ((BRADY, TACHY), (BRADY, AFIB)):
        if fired[a] and fired[b]:
            lo, hi = (a, b) if p[a] <= p[b] else (b, a)
            suppress(lo, f"mutual-exclusion vs {DETECTION_SCOPE[hi]} (p {p[lo]:.2f}<{p[hi]:.2f})")

    # 5. HR plausibility — Bradycardia requires a slow rate
    hr = s.context.get("hr_bpm")
    if fired[BRADY] and hr is not None and hr > HR_BRADY_MAX:
        suppress(BRADY, f"HR {hr:.0f}>{HR_BRADY_MAX} bpm (not bradycardic)")

    # 6. run cluster — keep BOTH heads in the record (collapse handled in to_alerts)
    if fired[SVT] and fired[VRUN]:
        reason[SVT] += " | run-cluster(origin uncertain)"
        reason[VRUN] += " | run-cluster(origin uncertain)"

    return {h: Decision(h, fired[h], p[h], reason[h]) for h in SCOPE_HEADS}


def to_alerts(decisions: dict[int, Decision]) -> tuple[list[Decision], RunAlert | None]:
    """Collapse the run cluster for *alerting* (Decision 2 = merge).

    Returns (non-run alerts, optional merged RunAlert). The original SVT/V-Run
    Decisions remain in `decisions` for GT-faithful evaluation and audit.
    """
    non_run = [d for h, d in decisions.items() if h not in (SVT, VRUN) and d.fired]
    svt, vrun = decisions.get(SVT), decisions.get(VRUN)
    run = None
    if MERGE_SVT_VT and ((svt and svt.fired) or (vrun and vrun.fired)):
        score = max(d.score for d in (svt, vrun) if d and d.fired)
        run = RunAlert(fired=True, score=score)   # acuity escalated to VT-level
    elif svt and svt.fired:
        non_run.append(svt)
    elif vrun and vrun.fired:
        non_run.append(vrun)
    return non_run, run


if __name__ == "__main__":
    # demo: AFib+Tachy double-fire with high NSR + a slow-HR bradycardia
    demo = ScopeScores(
        probs={AFIB: 0.92, TACHY: 0.61, BRADY: 0.80, SVT: 0.0, VRUN: 0.0, PAUSE: 0.0},
        nsr_score=0.0, context={"hr_bpm": 72.0},
    )
    for h, d in arbitrate(demo).items():
        print(f"  head {h:>3} {DETECTION_SCOPE[h]:<28} fired={d.fired!s:<5} {d.reason}")
