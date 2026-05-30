"""Alarm-type → ECGFounder-head map for the PhysioNet/CinC 2015 Challenge
("Reducing False Arrhythmia Alarms in the ICU").

Each ICU record is a triggered arrhythmia *alarm* with a known **type** and a
**true/false** verdict (was it a real arrhythmia or a false alarm?). That verdict is
the gold: a TRUE alarm of type T is a clean positive for T's head; a FALSE alarm of
type T is a HARD negative (it looked like T but wasn't).

This is the main open-access source for the two heads Chapman can't supply:
  • Asystole       → head 142 (Pause / asystole proxy)   ← Pause's best available label
  • Ventricular Tachycardia → head 98 (VT)               ← real sustained VT
  • Bradycardia (extreme)   → head 4  (bonus)
  • Tachycardia (extreme)   → head 6  (⚠ loose — not necessarily *sinus* tachy)
  • Ventricular Flutter/Fib → NOT in scope → unmapped (distinct from VT; left out)

CAVEATS (verify against the real headers with --report-unmapped):
  • Not 12-lead → NO frontal-plane angle derivation (typically II + V + ABP/PLETH).
    Use as a single-lead (Lead-II proxy) source; feeds the model at native lead only.
  • The true/false verdict location in the WFDB header varies by release; parse_alarm
    looks in the header comments and supports an external truth CSV fallback.
"""
from __future__ import annotations

HEAD_NAMES = {4: "SINUS BRADYCARDIA", 6: "SINUS TACHYCARDIA",
              98: "VENTRICULAR TACHYCARDIA", 142: "WITH SINUS PAUSE"}

# alarm-type token (lowercased, substring match) → founder head index.
# Order matters: check 'ventricular' tokens before generic 'tachycardia'.
ALARM_TO_HEAD: list[tuple[str, int]] = [
    ("asystole",                 142),   # Pause / asystole proxy
    ("ventricular_flutter",      None),  # VF/VFlutter — out of scope, explicitly skip
    ("ventricular_fib",          None),
    ("ventricular_tachycardia",  98),    # VT
    ("vtach",                    98),
    ("bradycardia",              4),     # extreme bradycardia
    ("tachycardia",              6),     # extreme tachycardia ⚠ (not necessarily sinus)
]

SCOPE_HEADS = sorted({h for _, h in ALARM_TO_HEAD if h is not None})

# known alarm-type display strings (for matching record names / headers loosely)
_TRUE_TOKENS  = ("true", "1", "real")
_FALSE_TOKENS = ("false", "0")


def alarm_to_head(alarm_text: str):
    """Return the founder head for an alarm-type string, or None if out-of-scope/unknown."""
    t = alarm_text.strip().lower().replace(" ", "_")
    for token, head in ALARM_TO_HEAD:
        if token in t:
            return head
    return None


def parse_alarm(header_comments: list[str], record_name: str = ""):
    """Extract (alarm_text, head, is_true) from WFDB header comments.

    is_true is True/False if a verdict token is found, else None (caller may supply a
    truth CSV). The alarm type is whichever comment matches a known alarm token; the
    record_name is a fallback (CinC training names sometimes encode the type).
    """
    blob = " ".join(header_comments or []).lower()
    alarm_text = ""
    for token, _ in ALARM_TO_HEAD:
        if token in blob:
            alarm_text = token
            break
    if not alarm_text:
        # fallback: scan the record name
        for token, _ in ALARM_TO_HEAD:
            if token in record_name.lower():
                alarm_text = token
                break
    head = alarm_to_head(alarm_text) if alarm_text else None
    # verdict
    is_true = None
    if any(f"true alarm" in c.lower() or c.strip().lower() in _TRUE_TOKENS
           for c in (header_comments or [])):
        is_true = True
    elif any("false alarm" in c.lower() or c.strip().lower() in _FALSE_TOKENS
             for c in (header_comments or [])):
        is_true = False
    return alarm_text, head, is_true


def labels_from_alarm(head, is_true, n_classes: int = 150):
    """150-dim label for one record.

    Positive at `head` ONLY if the alarm is a TRUE alarm. A FALSE alarm yields an
    all-zero vector → a HARD negative for that head (looked like the arrhythmia but
    wasn't). Returns None if the head is out of scope (record should be skipped).
    """
    import numpy as np
    if head is None:
        return None
    vec = np.zeros(n_classes, dtype=np.float32)
    if is_true:
        vec[head] = 1.0
    return vec


def _self_test():
    import numpy as np
    print("CinC-2015 alarm → founder head:")
    for token, head in ALARM_TO_HEAD:
        tag = "(skip — out of scope)" if head is None else f"→ head {head} {HEAD_NAMES.get(head,'')}"
        print(f"  {token:<26} {tag}")
    # synthetic records
    cases = [
        (["Ventricular_Tachycardia", "True alarm"], 98, True),
        (["Asystole", "False alarm"],               142, False),
        (["Ventricular_Flutter_Fib", "True alarm"], None, True),   # out of scope
    ]
    print("\nparse + label tests:")
    for comments, exp_head, exp_true in cases:
        at, head, is_true = parse_alarm(comments)
        vec = labels_from_alarm(head, is_true)
        pos = None if vec is None else int(vec.sum())
        print(f"  {comments} → alarm={at!r} head={head} true={is_true} pos={pos}")
        assert head == exp_head and is_true == exp_true
    # true VT → positive at 98; false asystole → all-zero (hard negative)
    assert labels_from_alarm(98, True)[98] == 1
    assert labels_from_alarm(142, False).sum() == 0
    print("\nself-test OK ✓")


if __name__ == "__main__":
    _self_test()
