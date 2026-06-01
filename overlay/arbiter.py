"""L2 — deterministic multi-label rule arbiter.

Pure function: ScopeScores → {head: Decision}. No learned weights; thresholds
are versioned constants. SUPPRESS / MERGE only — never promotes a head (raising
sensitivity is L1's job). Implements res/ptbxl_cofiring/MULTI_LABEL_RULES.md.

The structural rules are a DIRECT IMAGE of the PTB-XL GT co-occurrence matrix
(res/ptbxl_cofiring/gt_cofire_counts.csv): the rate-rhythm heads are pairwise
mutually exclusive in GT (and physiologically), and SVT-Run≡V-Run are coupled.
See res/scope_overlay/L2_DESIGN.md.

Precedence (signal-quality stage lives in the SQG, upstream — NOT here):
  1. validity mask        (Pause needs adequate clean signal length)
  2. NSR contradiction    (high head-1 score + weak arrhythmia ⇒ suppress)
  3. exclusion groups     ({Brady,AFib,Tachy} pairwise GT=0 ⇒ keep one)
  4. HR plausibility      (Bradycardia requires hr_bpm ≤ ~56)
  5. couple groups        (SVT+V-Run co-fire kept; merged to one alert in to_alerts())
"""
from __future__ import annotations

from dataclasses import dataclass

from label_config import DETECTION_SCOPE
from overlay.types import ScopeScores, Decision, RunAlert

SCOPE_HEADS: list[int] = sorted(DETECTION_SCOPE)   # [4, 5, 6, 93, 98, 142]

# head indices
BRADY, AFIB, TACHY, SVT, VRUN, PAUSE = 4, 5, 6, 93, 98, 142
NSR = 1   # out-of-scope feature head

# ── GT-derived structure (PTB-XL gt_cofire_counts.csv) ───────────────────────
# Exclusion group: every pair within it has GT co-occurrence 0 (and is
# physiologically mutually exclusive) → at most one head may fire.
EXCLUSION_GROUPS: list[tuple[int, ...]] = [(BRADY, AFIB, TACHY)]
# Directional overrides applied first inside an exclusion group (winner, loser):
# AFib beats Tachy (AF-RVR; co-firing Tachy is the FP, realness PPV ~1%).
DIRECTIONAL: list[tuple[int, int]] = [(AFIB, TACHY)]
# Couple group: GT Jaccard 1.0 (co-extensive) → co-fire kept, merged for alerting.
COUPLE_GROUPS: list[tuple[int, ...]] = [(SVT, VRUN)]


@dataclass(frozen=True)
class ArbiterConfig:
    """L2 configuration. ``enabled`` is the master switch — DEFAULT OFF.

    When OFF, arbitrate()/to_alerts() are pure pass-through: candidate firings
    (prob ≥ fire_threshold) flow through unchanged, no suppression, no merge.
    When ON, each rule can be toggled. The exclusion/couple structure is derived
    from the PTB-XL GT co-occurrence matrix (module constants above).
    """
    enabled: bool = False          # ← master switch, kept OFF for now
    # float (same for all heads) OR dict[head]→threshold (per-head policy points)
    fire_threshold: "float | dict[int, float]" = 0.5
    # per-rule toggles (only consulted when enabled=True)
    nsr_contradiction: bool = True
    rate_exclusion: bool = True    # GT-derived mutual exclusion among rate heads
    hr_plausibility: bool = True
    merge_runs: bool = True
    # thresholds
    tau_nsr: float = 0.5
    tau_weak: float = 0.70
    hr_brady_max: float = 56.3

# default instance — L2 OFF
DEFAULT_CONFIG = ArbiterConfig()
# ready-to-use GT-matched config (L2 ON); detector stays OFF unless given this
GT_MATCHED_CONFIG = ArbiterConfig(enabled=True)


def arbitrate(s: ScopeScores, config: ArbiterConfig | None = None) -> dict[int, Decision]:
    cfg = config or DEFAULT_CONFIG
    ft = cfg.fire_threshold
    def _thr(h: int) -> float:
        return float(ft.get(h, 0.5)) if isinstance(ft, dict) else float(ft)
    p = {h: float(s.probs.get(h, 0.0)) for h in SCOPE_HEADS}
    fired = {h: p[h] >= _thr(h) for h in SCOPE_HEADS}

    # ── L2 OFF: pure pass-through (no rules) ─────────────────────────────────
    if not cfg.enabled:
        return {h: Decision(h, fired[h], p[h],
                            "fired (L2 off)" if fired[h] else "below-threshold")
                for h in SCOPE_HEADS}

    reason = {h: ("fired" if fired[h] else "below-threshold") for h in SCOPE_HEADS}

    def suppress(h: int, why: str) -> None:
        if fired[h]:
            fired[h] = False
            reason[h] = f"suppressed: {why}"

    def margin(h: int) -> float:           # confidence above this head's threshold
        return p[h] - _thr(h)

    # 1. validity mask (placeholder — Pause needs adequate clean signal length)

    # 2. NSR contradiction (head-1 score is an FP feature, not a detection)
    if cfg.nsr_contradiction and s.nsr_score >= cfg.tau_nsr:
        for h in (AFIB, TACHY):
            if fired[h] and p[h] < cfg.tau_weak:
                suppress(h, f"NSR-contradiction (nsr={s.nsr_score:.2f}, p={p[h]:.2f})")

    # 3. exclusion groups — GT pairwise 0 ⇒ at most one head fires.
    #    Directional overrides first (AFib▸Tachy), then keep best margin-over-threshold.
    if cfg.rate_exclusion:
        for group in EXCLUSION_GROUPS:
            for hi, lo in DIRECTIONAL:
                if hi in group and lo in group and fired[hi] and fired[lo]:
                    suppress(lo, f"{DETECTION_SCOPE[hi]}▸{DETECTION_SCOPE[lo]} (AF-RVR)")
            members = [h for h in group if fired[h]]
            if len(members) > 1:
                best = max(members, key=margin)
                for h in members:
                    if h != best:
                        suppress(h, f"excl: kept {DETECTION_SCOPE[best]} "
                                    f"(margin {margin(h):+.2f}<{margin(best):+.2f})")

    # 4. HR plausibility — Bradycardia requires a slow rate
    if cfg.hr_plausibility:
        hr = s.context.get("hr_bpm")
        if fired[BRADY] and hr is not None and hr > cfg.hr_brady_max:
            suppress(BRADY, f"HR {hr:.0f}>{cfg.hr_brady_max} bpm (not bradycardic)")

    # 5. couple groups — keep BOTH (collapse handled in to_alerts)
    for group in COUPLE_GROUPS:
        if all(fired[h] for h in group):
            for h in group:
                reason[h] += " | couple(origin uncertain)"

    return {h: Decision(h, fired[h], p[h], reason[h]) for h in SCOPE_HEADS}


def to_alerts(decisions: dict[int, Decision],
              config: ArbiterConfig | None = None) -> tuple[list[Decision], RunAlert | None]:
    """Collapse the run cluster for *alerting* (merge), when L2 is enabled.

    Returns (non-run alerts, optional merged RunAlert). The original SVT/V-Run
    Decisions remain in `decisions` for GT-faithful evaluation and audit. When
    L2 is OFF (or merge_runs=False) the run heads pass through as separate alerts.
    """
    cfg = config or DEFAULT_CONFIG
    non_run = [d for h, d in decisions.items() if h not in (SVT, VRUN) and d.fired]
    svt, vrun = decisions.get(SVT), decisions.get(VRUN)
    merge = cfg.enabled and cfg.merge_runs
    run = None
    if merge and ((svt and svt.fired) or (vrun and vrun.fired)):
        score = max(d.score for d in (svt, vrun) if d and d.fired)
        run = RunAlert(fired=True, score=score)   # acuity escalated to VT-level
    else:
        if svt and svt.fired:
            non_run.append(svt)
        if vrun and vrun.fired:
            non_run.append(vrun)
    return non_run, run


if __name__ == "__main__":
    # demo: all 3 rate heads fire (exclusion group) + SVT/VT couple
    demo = ScopeScores(
        probs={BRADY: 0.80, AFIB: 0.92, TACHY: 0.61, SVT: 0.7, VRUN: 0.7, PAUSE: 0.0},
        nsr_score=0.0, context={"hr_bpm": 72.0},
    )
    for tag, cfg in (("OFF (default)", DEFAULT_CONFIG), ("ON (GT-matched)", GT_MATCHED_CONFIG)):
        print(f"\nL2 {tag}:")
        d = arbitrate(demo, cfg)
        for h, dec in d.items():
            print(f"  head {h:>3} {DETECTION_SCOPE[h]:<28} fired={dec.fired!s:<5} {dec.reason}")
        non_run, run = to_alerts(d, cfg)
        print("  alerts:", [DETECTION_SCOPE[x.head] for x in non_run], "| run:", run)
