# Archived FastAPI + Vanilla JS Plan

These two documents describe an **earlier design** for the ECG cloud demo using a strict frontend/backend split (FastAPI service + static HTML/JS page). The plan was sound, but on 2026-05-28 the project relaxed the frontend/backend separation requirement to enable a faster, simpler Streamlit single-file implementation.

| File | Original purpose |
|---|---|
| [BACKEND.md](BACKEND.md) | FastAPI service design, including a versioned JSON API contract |
| [FRONTEND.md](FRONTEND.md) | Static single-page HTML/JS UI design, consuming the API contract |

The **current active plan** is at [../STREAMLIT_APP.md](../STREAMLIT_APP.md) — a monolithic Streamlit app that collapses both layers into one file.

## When to revisit these archived docs

Re-promote them out of `_archived/` if any of the following becomes true:

1. **External / customer-facing demos** — Streamlit's iconic look may not pass brand review; a custom HTML page does.
2. **Multiple client types** — if the same model needs to back a Slack bot, a CLI, AND a web page, a stable JSON API matters again.
3. **Independent UI/model ownership** — two engineers wanting non-overlapping ownership of UI vs model code benefit from the split.
4. **HIPAA-compliant production** — FastAPI's middleware ecosystem (rate limiting, auth, structured logging) is more mature than Streamlit's; production hardening is easier from the FastAPI baseline.

For the **internal MVP demo**, none of those apply, so Streamlit wins on time-to-first-demo.

## What's preserved from the FastAPI plan

Even though we're switching frameworks, several substantive decisions documented in BACKEND.md / FRONTEND.md carry forward to the Streamlit plan unchanged:

- **Result composition policy** (which heads to show, ordering rules, head-sharing dedup) — see BACKEND.md §4.5
- **Risk-tier color / reliability pill design** — see FRONTEND.md §6
- **Multi-lead Option A** (auto-extract Lead II, banner) — see FRONTEND.md §"Open questions" (item retired)
- **Sizing decision** (t3.large, not t3.medium) — see BACKEND.md §8 (corrected per reviewer)
- **Exception-message surfacing in errors** — see BACKEND.md §4.4 (corrected per reviewer)

The Streamlit plan inherits all of the above; only the *delivery mechanism* changed.
