# Frontend Design — ECG Cloud Demo

**Layer**: frontend (static HTML/JS, served by backend)
**Status**: planning only
**Companion**: [BACKEND.md](BACKEND.md)
**Top-level**: [README.md](README.md)

---

## 1. Purpose & scope

Single-page web UI that lets an internal demo operator:

1. Drag an ECG file onto the page (or click to browse)
2. See the preprocessed waveform rendered in ~1 second
3. See a list of detected findings with risk + reliability badges
4. See a 10-head probability bar chart with all v3.1 heads

Everything runs in the operator's browser; the only backend dependency is the contract defined in [BACKEND.md §"API contract"](BACKEND.md#4-api-contract).

## 2. Non-goals

- ❌ No JS framework (React, Vue, Svelte, etc.)
- ❌ No build step — no webpack, vite, npm install
- ❌ No TypeScript
- ❌ No authentication UI
- ❌ No history / list of past predictions
- ❌ No mobile-responsive design (desktop demo only)
- ❌ No internationalization

## 3. User flow

```
1. Page loads
   ├─ GET /healthz
   ├─ green dot ("backend ready") or red ("backend offline" / "warming")
   └─ Drop zone visible, empty result panel

2. User drags ECG file onto drop zone (or clicks to open file picker)
   ├─ Filename + size displayed below the drop zone
   ├─ "Analyzing…" spinner appears in result panel
   └─ POST /predict with file as multipart form-data

3. Response arrives (~300 ms – 2 s)
   ├─ Spinner replaced with the result panel
   ├─ Waveform PNG rendered at top (from response.waveform_png_b64)
   ├─ Findings list (descending probability, with risk + reliability chips)
   ├─ All-10-heads bar chart at the bottom (from all_v31_head_probabilities)
   └─ Metadata footer: filename, format, fs_hz, duration_s, inference_ms

4. User clicks "Run another"
   └─ Resets state, returns to step 2

5. Error path (any non-200 from /predict)
   └─ Result panel shows a red box with response.error.message
      (and response.error.code as a small monospace label)
```

## 4. Wireframe

```
┌──────────────────────────────────────────────────────────────────┐
│  ECG Founder — internal demo            ● backend ready          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│            ┌────────────────────────────────────────┐            │
│            │     drop an ECG file here              │            │
│            │     (.csv, .json, or .dat+.hea zip)    │            │
│            │     — or click to browse —             │            │
│            └────────────────────────────────────────┘            │
│                                                                  │
│            sample_001.csv  ·  47 KB                              │
│                                                                  │
├─────  Results  (only visible after a successful POST) ──────────┤
│                                                                  │
│  Waveform  (preprocessed: notch + bandpass + median + zscore)   │
│  ╱╲      ╱╲      ╱╲      ╱╲      ╱╲                            │
│ ╱  ╲────╱  ╲────╱  ╲────╱  ╲────╱  ╲                            │
│                                                                  │
│  Findings  (2 events ≥ 0.5)                                      │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ 🟠 ATRIAL FIBRILLATION    0.962   [HIGH]  [reliable]    │    │
│  │     Irregularly irregular atrial rhythm lacking…        │    │
│  ├──────────────────────────────────────────────────────────┤    │
│  │ 🟡 PVC (Isolated V Beat)  0.78    [LOW]   [severe-fp]   │    │
│  │     Ectopic beats originating from ventricles…          │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  All 10 V3.1 heads (probability)                                 │
│  ATRIAL FIBRILLATION       ████████████████████ 0.96             │
│  PVC                       ███████████████      0.78             │
│  ISB / PAC                 ██▌                  0.10             │
│  Bradycardia               █                    0.04             │
│  ST Elevation              ▌                    0.01             │
│  Sinus Tachycardia         ▌                    0.01             │
│  SV Couplet                ▌                    0.01             │
│  SV Run                    ▏                    0.00             │
│  V Run                     ▏                    0.00             │
│  Pause                                          0.00             │
│                                                                  │
│  Metadata: sample_001.csv · csv · 5000 samples @ 500 Hz · 10.0s │
│            · inference 287 ms                                    │
│                                                                  │
│            [ Run another ]                                       │
└──────────────────────────────────────────────────────────────────┘
```

## 5. Files

```
frontend/
├── index.html      # ~80 lines: structure + inline <style> + <script src=app.js>
├── app.js          # ~120 lines: drop-zone, fetch, render, error path
└── styles.css      # ~60 lines: flexbox layout, color tokens, badge styling
```

Total budget: **<300 lines** combined. Zero dependencies. No tooling.

Served at the FastAPI root via `app.mount("/", StaticFiles(directory='frontend', html=True))` — visiting `http://host:8000/` returns `index.html`.

## 6. Visual language

### 6.1 Risk-tier color (leading dot icon on each finding)

| Tier | Dot | Hex |
|---|---|---|
| critical | 🔴 | `#dc2626` |
| high | 🟠 | `#ea580c` |
| moderate | 🟡 | `#ca8a04` |
| low | 🟢 | `#16a34a` |

### 6.2 Reliability badge (pill)

| Reliability | Pill style | Tooltip |
|---|---|---|
| `reliable` | green fill, white text | "PPV ≥ 80% on production cohort" |
| `moderate_fp` | amber fill, dark text | "PPV 20–80%; downstream FP filter recommended" |
| `severe_fp` | red fill, white text | "⚠ PPV < 20% — many alerts from this head are false positives" |
| `insufficient_data` | gray fill, dark text | "n < 5 in fzark — reliability not assessable" |

### 6.3 Probability bar

Full-width horizontal bar inside its grid cell. Width = `probability × 100%`. Color tracks risk tier (same hex as the leading dot, ~30% opacity). Two-decimal numeric label to the right.

### 6.4 Color tokens (CSS variables)

```css
:root {
  --bg: #fafafa;
  --panel: #ffffff;
  --border: #e5e7eb;
  --text: #111827;
  --text-muted: #6b7280;

  --risk-critical: #dc2626;
  --risk-high:     #ea580c;
  --risk-moderate: #ca8a04;
  --risk-low:      #16a34a;

  --rel-reliable:    #16a34a;
  --rel-moderate-fp: #ca8a04;
  --rel-severe-fp:   #dc2626;
  --rel-insuff:      #6b7280;

  --status-ok:      #16a34a;
  --status-warming: #ca8a04;
  --status-error:   #dc2626;
}
```

All clinical-decision logic for choosing these colors lives in the backend's response — `risk_tier` and `reliability` come straight from `FZARK_ONTOLOGY`. Frontend looks them up in a static map.

## 7. API consumption

The frontend makes exactly **two** fetch calls. If you find yourself adding a third, you're probably violating the minimalist scope — push it to the backend instead.

### 7.1 `GET /healthz` on page load

```js
window.addEventListener('DOMContentLoaded', async () => {
  try {
    const r = await fetch('/healthz');
    const j = await r.json();
    setStatusBadge(j.status === 'ok' ? 'ready' : j.status);  // 'warming' | 'ok'
  } catch {
    setStatusBadge('error');
  }
});
```

`setStatusBadge('ready' | 'warming' | 'error')` toggles the dot color in the top-right corner.

### 7.2 `POST /predict` on file drop

```js
async function runPredict(file) {
  setResultPanel('loading', { filename: file.name });
  const fd = new FormData();
  fd.append('file', file);
  let res;
  try {
    res = await fetch('/predict', { method: 'POST', body: fd });
  } catch (netErr) {
    return setResultPanel('error', {
      code: 'network_error', message: String(netErr),
    });
  }
  const j = await res.json();
  if (j.error) return setResultPanel('error', j.error);
  setResultPanel('ok', j);
}
```

These two functions are the only contract surface. If the backend changes its JSON, the frontend breaks here — and only here.

## 8. Rendering pseudocode

```js
function setResultPanel(state, payload) {
  const root = document.querySelector('#result');
  root.dataset.state = state;        // 'idle' | 'loading' | 'ok' | 'error'
  if (state === 'ok')    renderOk(root, payload);
  if (state === 'error') renderError(root, payload);
}

function renderOk(root, r) {
  // 1. Waveform
  root.querySelector('#waveform').src = 'data:image/png;base64,' + r.waveform_png_b64;
  root.querySelector('#preprocessing').textContent = r.preprocessing_applied.join(' · ');

  // 2. Findings list
  const list = root.querySelector('#findings');
  list.innerHTML = '';
  for (const f of r.findings) {
    list.insertAdjacentHTML('beforeend', findingCardHTML(f));
  }
  root.querySelector('#findings-count').textContent =
    `${r.findings.length} event${r.findings.length !== 1 ? 's' : ''} ≥ 0.5`;

  // 3. All-10-heads bar chart
  const chart = root.querySelector('#all-heads');
  chart.innerHTML = '';
  const sorted = Object.entries(r.all_v31_head_probabilities)
                       .sort(([, a], [, b]) => b - a);
  for (const [name, p] of sorted) {
    chart.insertAdjacentHTML('beforeend', headBarHTML(name, p));
  }

  // 4. Metadata footer
  const meta = root.querySelector('#meta');
  meta.textContent =
    `${r.input.filename} · ${r.input.format} · ${r.input.n_samples} samples @ ` +
    `${r.input.fs_hz} Hz · ${r.input.duration_s.toFixed(1)} s · inference ${r.inference_ms} ms`;
}

function findingCardHTML(f) {
  return `
    <li class="finding finding--${f.risk_tier}">
      <span class="finding__dot"></span>
      <span class="finding__title">${f.event_label}</span>
      <span class="finding__prob">${f.probability.toFixed(2)}</span>
      <span class="badge badge--${f.risk_tier}">${f.risk_tier}</span>
      <span class="badge badge--${f.reliability}" title="${tooltipFor(f.reliability)}">${f.reliability.replace('_', ' ')}</span>
      <p class="finding__support">${escapeHtml(f.clinical_support)}</p>
    </li>`;
}
```

`escapeHtml` is a 6-line utility — don't pull in a templating library for this.

## 9. State machine

```
        ┌─────────┐  file dropped  ┌──────────┐
        │  idle   ├───────────────>│ loading  │
        └─────────┘                └─────┬────┘
              ▲                          │
              │ "run another"            │ /predict resolves
              │                          ▼
              │                    ┌──────────┐
              │     200 OK         │   ok     │
              ├────────────────────┤          │
              │                    └──────────┘
              │                          │
              │                          │ /predict 4xx/5xx
              │                          ▼
              │                    ┌──────────┐
              └────────────────────┤  error   │
                                   └──────────┘
```

State is held entirely in `<div id="result" data-state="…">`. CSS hides / shows panels by attribute selector — no JS framework needed.

## 10. Deployment

Backend serves `frontend/` directly. There is no separate frontend deployment.

```bash
# On the demo box, after editing any frontend file:
sudo systemctl restart ecg-demo

# In the browser, hard-reload (Cmd+Shift+R) to bust any browser cache.
```

For airgapped demos where you can't clone the repo on the demo box, `frontend/` totals ~12 KB across 3 files — rsync / scp directly:

```bash
rsync -av frontend/ ubuntu@demo-host:~/ECGFounder/frontend/
```

## 11. Browser support

| Browser | Supported |
|---|---|
| Chrome / Edge (Chromium) ≥ 100 | ✅ |
| Safari ≥ 16 | ✅ |
| Firefox ≥ 100 | ✅ |
| IE / legacy Edge | ❌ |
| Mobile Safari / Chrome Android | ⚠ usable but not designed for it |

Tested target for the live demo: **Chrome on macOS 13+** (the laptop driving the demo). Everything else is best-effort.

Features used:
- `fetch` (everywhere modern)
- HTML5 drag-and-drop API
- `FormData` multipart
- base64 data URIs for the waveform image
- CSS Grid + Flexbox

## 12. Accessibility (minimum bar for internal demo)

- Drop zone is keyboard-focusable; pressing Enter opens the file picker.
- Status badge uses both color AND text ("backend ready" / "backend warming").
- Reliability pills include `title` tooltip text (also readable by screen readers).
- Sufficient color contrast (all text ≥ AA on the chosen backgrounds).

Out of scope for the internal demo: full WCAG audit, ARIA live regions for the loading state, focus management between idle / loading / ok states.

## 13. Out-of-scope but worth noting

These are deliberate omissions; flag them if the demo grows up:

- **Multiple-file batch upload** — current design accepts one file at a time. Extension path: `for (const f of files)` loop, render results in tabs.
- **Saved-result share link** — not implemented (no backend storage). Adding requires backend `GET /results/{id}` plus a results table.
- **Export to PDF** — point users to browser's "Print to PDF" if asked.
- **Multi-lead display** — current design assumes a single derived 1-lead signal (matches `DualHeadECGFounder` input). For a true 12-lead Holter visualization, the frontend would need to render a separate trace per lead, and the API would need to expose them.

## 14. Open questions for the next review pass

1. **Format of the "Findings ≥ 0.5" empty state** — do we show "No findings" cleanly, or always render the top-3 heads regardless of threshold to give the demo operator *something* to point at? (Backend currently allows empty `findings[]`.)
2. **What happens to v3.1 head-shared events in the UI** — e.g. if the PVC head fires at 0.85, do we show one "PVC head" card or two cards (Isolated V Beat + V Couplet, both at 0.85)? Backend plan (BACKEND §4.5) emits one card per *event*. Confirm this is what we want visually.
3. **Should the demo operator be able to toggle "suppression on/off"** — i.e. show what the v2 FP filter does, side-by-side? Powerful demo moment but adds backend complexity (must run both pipelines and return both result sets). Defer to phase 5 if demand emerges.
4. **Branding** — VivaLink logo? "DEMO ONLY" watermark in the corner? Internal-only label in the top bar?

---

*This is a planning document. No code committed under `frontend/` yet — implementation in phase 2 per [README.md §"Implementation phases"](README.md#implementation-phases-when-greenlit-again).*
