# ECG Cloud Demo — Internal MVP

**Status**: planning only — design ratified 2026-05-28, implementation pending one more review pass.

Single-file Streamlit web demo for evaluating the current production ECG model (`DualHeadECGFounder` + v3.1 ontology) over HTTP. Internal use only — no auth, no PHI, no production deployment.

## Architecture in one paragraph

One EC2 instance runs a single Streamlit process on port 8000. Streamlit serves both the UI and the inference path from a single `app.py`; the model loads once via `@st.cache_resource`. No FastAPI, no JSON contract, no JS, no S3, no DB. Demo operator drags an ECG file onto the page, sees findings + waveform + 10-head probability chart in 300 ms–2 s.

## Why Streamlit (and not the prior FastAPI + Vanilla JS plan)

The original 2026-05-28 plan used a strict frontend/backend split (FastAPI service + static HTML/JS page). After a review pass, the project relaxed the separation requirement to optimize for **time-to-first-demo**. Streamlit collapses two layers into one file, removes the JSON contract entirely, and natively handles drag-drop / matplotlib / bar charts.

| Aspect | Prior plan (FastAPI + JS) | Current plan (Streamlit) |
|---|---|---|
| Time to working demo | ~5 hours | ~1.5 hours |
| Files involved | 9 | 1 (`app.py`) + 1 (`.streamlit/config.toml`) |
| Lines of code | ~500 | ~150 |
| Dependencies | FastAPI, uvicorn, python-multipart | streamlit |
| Drag-drop UI | hand-rolled HTML/JS | `st.file_uploader(...)` |
| Matplotlib display | base64 PNG + img src dance | `st.pyplot(fig)` |
| Probability bars | hand-rolled CSS | `st.bar_chart(...)` |
| Frontend/backend separation | strict | **relaxed (intentional)** |

The archived FastAPI plan lives at [_archived/](_archived/) and can be re-promoted if separation becomes a hard requirement again (external customer demos, multi-client backends, HIPAA-compliant production — see archived README for the full list of when-to-revisit triggers).

## Documents in this folder

Read in this order if you're starting fresh:

1. **[BEGINNER.md](BEGINNER.md)** — start here if you've never done AWS/cloud deployment before. Recommends running the demo on your laptop first (Phase 0), explains every AWS term in plain English, sets cost guardrails, and lists the common first-timer gotchas.
2. **[SETUP.md](SETUP.md)** — the install + ops guide. Covers tool dependencies (AWS, CLI, SSH, Streamlit), step-by-step provisioning, verification, day-2 operations, and troubleshooting.
3. **[STREAMLIT_APP.md](STREAMLIT_APP.md)** — the design doc. Covers UI flow, model service, file parsers, deployment, sizing, and error handling. Reference when reading the code.

Substantive decisions inherited from the archived plan (result composition policy, risk-tier color tokens, reliability pill design, multi-lead handling, sizing, exception-message surfacing) are restated in STREAMLIT_APP.md — no need to consult the archived docs to understand the current behavior.

## Project layout (after implementation)

```
app/                           ← new top-level dir for the demo
├── app.py                     ← the Streamlit app, ~150 LoC total
├── requirements.txt           ← streamlit + the existing repo's deps
└── .streamlit/
    └── config.toml            ← theme, server port, file-size limits

deploy/
└── ecg-demo.service           ← systemd unit (5 lines body)

docs/cloud_demo/
├── README.md                  ← this file
├── STREAMLIT_APP.md           ← the design doc
└── _archived/                 ← FastAPI + JS plan, for reference only
    ├── README.md
    ├── BACKEND.md
    └── FRONTEND.md
```

The Streamlit app **imports the existing algorithm modules in-place** — no fork, no copy. Same pattern as the FastAPI plan:

```python
from device_utils import resolve_device
from dual_head_ecgfounder import DualHeadECGFounder
from label_config import FZARK_ONTOLOGY
from preprocessing import ECGPreprocessor
```

## Quick start (post-implementation)

See [STREAMLIT_APP.md §"Deployment runbook"](STREAMLIT_APP.md#7-deployment-runbook). TL;DR: clone repo on a t3.large, `pip install streamlit`, `systemctl enable --now ecg-demo`. Open `http://<ec2-private-ip>:8000` from inside the VPN.

## Out of scope — when to NOT use this stack

| Use case | Why this stack is wrong | Where to escalate |
|---|---|---|
| Real patient data | No PHI handling, no BAA, no encryption-at-rest, no access log retention | Production architecture review with security |
| External / customer demos | No auth, no rate limiting, iconic Streamlit look may not pass brand review | Re-promote the archived FastAPI + JS plan, add Cognito + TLS |
| Multiple client types (Slack bot + CLI + web) | No stable JSON API surface | Re-promote the archived FastAPI plan; its API contract is reusable |
| High-volume continuous workload | Streamlit's threading model serializes per-session work | Move to async SQS + Fargate ("production hardening" path) |
| Long ECGs (>10 minutes) | Body size cap + synchronous run | Either bump limits or add background job pattern (Streamlit + `st.queue` is one option) |
| Persistence / audit trail | No DB, no result history | Add a SQLite layer or postgres + dedicated session table |

For any of the above, this demo is a starting point but the architecture changes. Do not bolt auth / TLS / scaling onto the demo — start a new design.

## Implementation phases

| Phase | Deliverable | Estimated effort |
|---|---|---|
| **0** (now) | These planning docs ([README.md](README.md) + [STREAMLIT_APP.md](STREAMLIT_APP.md)) | 1 hour ✅ |
| **1** | Implement `app/app.py` per STREAMLIT_APP.md | ~1.5 hours |
| **2** | Deploy on a `t3.large` EC2 + smoke test | 30 min |
| **3** | Internal demo + capture feedback | 1 hour |
| **4** (if needed) | Iterate on UX / output format based on demo feedback | TBD |

Total to working demo from a clean repo: **~3 hours** (vs ~6 hours for the prior FastAPI plan). The reduction is real and is the reason for the pivot.

---

*Plan ratified 2026-05-28. The decision to relax frontend/backend separation is documented and reversible — the archived FastAPI design at [_archived/](_archived/) can be re-promoted at any time if requirements change.*
