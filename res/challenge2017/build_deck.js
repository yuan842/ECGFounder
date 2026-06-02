const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const FA = require("react-icons/fa");

// ── palette (Ocean / clinical-tech) ─────────────────────────────────────────
const NAVY = "11244A";      // dominant dark
const DEEP = "0A1A33";      // darker title bg
const BLUE = "0B5394";      // primary blue
const TEAL = "1C7293";      // secondary
const MINT = "2EC4B6";      // accent
const AMBER = "F4A259";     // stat accent
const CORAL = "E76F51";     // warning / FP
const INK = "1E293B";       // body text
const MUTE = "64748B";      // muted
const PAPER = "FFFFFF";
const ICE = "EAF1F8";       // light panel

async function icon(Comp, color = "#FFFFFF", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color, size: String(size) }));
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}
const shadow = () => ({ type: "outer", color: "000000", blur: 7, offset: 3, angle: 135, opacity: 0.16 });

(async () => {
  const p = new pptxgen();
  p.layout = "LAYOUT_WIDE";              // 13.3 x 7.5
  p.author = "ECGFounder team";
  p.title = "Out-of-Box AFib Validation — Challenge 2017";
  const W = 13.3, H = 7.5;

  const ic = {
    heart: await icon(FA.FaHeartbeat, "#" + MINT),
    shield: await icon(FA.FaShieldAlt, "#" + MINT),
    chart: await icon(FA.FaChartLine, "#" + MINT),
    flask: await icon(FA.FaFlask, "#" + MINT),
    layers: await icon(FA.FaLayerGroup, "#" + MINT),
    micro: await icon(FA.FaMicrochip, "#" + MINT),
    stetho: await icon(FA.FaStethoscope, "#" + MINT),
    bal: await icon(FA.FaBalanceScale, "#" + MINT),
    flag: await icon(FA.FaFlagCheckered, "#" + MINT),
    warn: await icon(FA.FaExclamationTriangle, "#" + AMBER),
    check: await icon(FA.FaCheckCircle, "#" + MINT),
    arrow: await icon(FA.FaArrowRight, "#" + MUTE),
  };

  // helpers
  const titleBar = (s, kicker, title) => {
    s.addText(kicker.toUpperCase(), { x: 0.6, y: 0.42, w: 12, h: 0.3, fontSize: 12, bold: true,
      color: TEAL, charSpacing: 3, fontFace: "Trebuchet MS", margin: 0 });
    s.addText(title, { x: 0.6, y: 0.72, w: 12.1, h: 0.7, fontSize: 30, bold: true, color: NAVY,
      fontFace: "Georgia", margin: 0 });
  };
  const footer = (s, n) => {
    s.addText("ECGFounder · Out-of-Box AFib Validation on PhysioNet/CinC Challenge 2017",
      { x: 0.6, y: H - 0.42, w: 10, h: 0.3, fontSize: 9, color: MUTE, fontFace: "Calibri", margin: 0 });
    s.addText(String(n), { x: W - 0.9, y: H - 0.42, w: 0.4, h: 0.3, fontSize: 9, color: MUTE, align: "right", margin: 0 });
  };
  const card = (s, x, y, w, h, fill = PAPER) =>
    s.addShape(p.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: "E2E8F0", width: 1 }, shadow: shadow() });

  // ════════════════════════════════ 1. TITLE ════════════════════════════════
  let s = p.addSlide(); s.background = { color: DEEP };
  s.addShape(p.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: 0.18, fill: { color: TEAL } });
  s.addImage({ data: ic.heart, x: 0.7, y: 1.5, w: 0.9, h: 0.9 });
  s.addText("OUT-OF-BOX EXTERNAL VALIDATION", { x: 0.72, y: 2.55, w: 11, h: 0.4, fontSize: 15,
    bold: true, color: MINT, charSpacing: 4, fontFace: "Trebuchet MS", margin: 0 });
  s.addText("Production Single-Lead AFib Detector\non PhysioNet/CinC Challenge 2017", {
    x: 0.7, y: 3.0, w: 12, h: 1.7, fontSize: 40, bold: true, color: PAPER, fontFace: "Georgia", lineSpacingMultiple: 1.05, margin: 0 });
  s.addText([
    { text: "Zero retraining · zero tuning · held-out test set", options: { color: "CBD5E1", breakLine: true } },
    { text: "Technical & Clinical Executive Briefing", options: { color: AMBER, bold: true } },
  ], { x: 0.72, y: 4.95, w: 11, h: 0.8, fontSize: 16, fontFace: "Calibri", lineSpacingMultiple: 1.25, margin: 0 });
  s.addText("2026-06-02", { x: 0.72, y: 6.5, w: 6, h: 0.3, fontSize: 12, color: MUTE, margin: 0 });

  // ════════════════════════════ 2. EXEC SUMMARY ═════════════════════════════
  s = p.addSlide(); s.background = { color: PAPER };
  titleBar(s, "Executive summary", "Strong out-of-box AFib performance on an unseen device");
  const stats = [
    ["88.2%", "Sensitivity", "AFib recordings caught", BLUE],
    ["93.5%", "Specificity", "normal recordings left alone", TEAL],
    ["0.95", "ROC-AUC", "overall discrimination", MINT],
    ["0.76", "F1 score", "balanced precision / recall", AMBER],
  ];
  stats.forEach((st, i) => {
    const x = 0.6 + i * 3.07;
    card(s, x, 1.65, 2.85, 1.95);
    s.addShape(p.shapes.RECTANGLE, { x, y: 1.65, w: 2.85, h: 0.12, fill: { color: st[3] } });
    s.addText(st[0], { x, y: 1.9, w: 2.85, h: 0.95, fontSize: 46, bold: true, color: NAVY, align: "center", fontFace: "Georgia", margin: 0 });
    s.addText(st[1], { x, y: 2.85, w: 2.85, h: 0.35, fontSize: 16, bold: true, color: st[3], align: "center", margin: 0 });
    s.addText(st[2], { x: x + 0.15, y: 3.2, w: 2.55, h: 0.35, fontSize: 11, color: MUTE, align: "center", margin: 0 });
  });
  s.addShape(p.shapes.RECTANGLE, { x: 0.6, y: 4.0, w: 12.1, h: 2.45, fill: { color: NAVY } });
  s.addShape(p.shapes.RECTANGLE, { x: 0.6, y: 4.0, w: 0.12, h: 2.45, fill: { color: AMBER } });
  s.addText("BOTTOM LINE", { x: 0.95, y: 4.25, w: 11, h: 0.35, fontSize: 13, bold: true, color: AMBER, charSpacing: 3, margin: 0 });
  s.addText([
    { text: "A detector calibrated only on our own ambulatory single-lead data generalizes to a different handheld device at ", options: {} },
    { text: "0.95 AUC", options: { bold: true, color: MINT } },
    { text: ". The production post-processing turns a raw model that fires indiscriminately (F1 ", options: {} },
    { text: "0.50", options: { bold: true, color: CORAL } },
    { text: ") into a deployable detector (F1 ", options: {} },
    { text: "0.76", options: { bold: true, color: MINT } },
    { text: ") with clinically usable specificity.", options: {} },
  ], { x: 0.95, y: 4.65, w: 11.5, h: 1.6, fontSize: 18, color: "F1F5F9", fontFace: "Calibri", lineSpacingMultiple: 1.25, valign: "top", margin: 0 });
  footer(s, 2);

  // ════════════════════════════ 3. WHY THIS STUDY ═══════════════════════════
  s = p.addSlide(); s.background = { color: PAPER };
  titleBar(s, "Motivation", "Why validate on an external dataset?");
  const why = [
    [ic.micro, "Calibrated on one device", "The production AFib head was tuned on the fzark ambulatory single-lead cohort. We must prove it holds on data it never saw."],
    [ic.flask, "An independent benchmark", "Challenge 2017: 8,528 publicly-labeled single-lead recordings from a different handheld device (AliveCor, 300 Hz)."],
    [ic.shield, "Zero-leakage generalization", "The model was never trained on this data — a true out-of-box test of real-world robustness."],
  ];
  why.forEach((r, i) => {
    const y = 1.75 + i * 1.6;
    card(s, 0.6, y, 12.1, 1.4, ICE);
    s.addShape(p.shapes.OVAL, { x: 0.95, y: y + 0.33, w: 0.74, h: 0.74, fill: { color: NAVY } });
    s.addImage({ data: r[0], x: 1.12, y: y + 0.5, w: 0.4, h: 0.4 });
    s.addText(r[1], { x: 2.1, y: y + 0.22, w: 10.2, h: 0.4, fontSize: 19, bold: true, color: NAVY, fontFace: "Georgia", margin: 0 });
    s.addText(r[2], { x: 2.1, y: y + 0.66, w: 10.3, h: 0.6, fontSize: 14, color: INK, fontFace: "Calibri", margin: 0 });
  });
  footer(s, 3);

  // ════════════════════════════ 4. DATA & METHOD ════════════════════════════
  s = p.addSlide(); s.background = { color: PAPER };
  titleBar(s, "Data & method", "Dataset, split, and recording-level detection");
  // left: dataset table
  s.addText("Challenge 2017 — 8,528 single-lead recordings (30–60 s)", { x: 0.6, y: 1.6, w: 6.2, h: 0.4, fontSize: 14, bold: true, color: TEAL, margin: 0 });
  const tRows = [
    [{ text: "Label", options: { bold: true, color: PAPER, fill: { color: NAVY } } },
     { text: "Meaning", options: { bold: true, color: PAPER, fill: { color: NAVY } } },
     { text: "n", options: { bold: true, color: PAPER, fill: { color: NAVY }, align: "right" } },
     { text: "Role", options: { bold: true, color: PAPER, fill: { color: NAVY } } }],
    ["N", "Normal sinus", "5,076", "neg (clean)"],
    ["A", "Atrial fibrillation", "758", "target"],
    ["O", "Other rhythm", "2,415", "context"],
    ["~", "Noisy", "279", "context"],
  ];
  s.addTable(tRows, { x: 0.6, y: 2.05, w: 6.2, colW: [0.8, 2.6, 1.1, 1.7], rowH: 0.46,
    fontSize: 13, fontFace: "Calibri", color: INK, valign: "middle", align: "left",
    border: { pt: 0.5, color: "D9E2EC" }, fill: { color: "F8FAFC" } });
  s.addText([
    { text: "Only A maps to a production scope event (AFib). ", options: { bold: true } },
    { text: "N is the clean negative; O and ~ are reported for context only (mixed / noise), excluded from binary metrics.", options: {} },
  ], { x: 0.6, y: 4.5, w: 6.2, h: 1.0, fontSize: 12.5, color: MUTE, fontFace: "Calibri", lineSpacingMultiple: 1.2, margin: 0 });

  // right: split + detection cards
  card(s, 7.1, 1.95, 5.6, 1.55, ICE);
  s.addText("STRATIFIED 80 / 10 / 10 SPLIT", { x: 7.35, y: 2.12, w: 5.2, h: 0.3, fontSize: 12, bold: true, color: TEAL, charSpacing: 2, margin: 0 });
  s.addText([
    { text: "Per-class, seed-fixed, frozen as a manifest. ", options: {} },
    { text: "Validation", options: { bold: true, color: NAVY } },
    { text: " = tuning context · ", options: {} },
    { text: "Test", options: { bold: true, color: NAVY } },
    { text: " = held-out report (852 recordings).", options: {} },
  ], { x: 7.35, y: 2.5, w: 5.15, h: 0.9, fontSize: 13.5, color: INK, fontFace: "Calibri", lineSpacingMultiple: 1.18, valign: "top", margin: 0 });
  card(s, 7.1, 3.7, 5.6, 1.8, ICE);
  s.addText("RECORDING-LEVEL DETECTION", { x: 7.35, y: 3.87, w: 5.2, h: 0.3, fontSize: 12, bold: true, color: TEAL, charSpacing: 2, margin: 0 });
  s.addText([
    { text: "Tile each recording into 10 s windows → preprocess (300→500 Hz, notch, band-pass, normalize) → score every window. ", options: {} },
    { text: "A recording is flagged AFib if any window fires", options: { bold: true, color: NAVY } },
    { text: " — identical to the production recording report.", options: {} },
  ], { x: 7.35, y: 4.25, w: 5.15, h: 1.15, fontSize: 13.5, color: INK, fontFace: "Calibri", lineSpacingMultiple: 1.18, valign: "top", margin: 0 });
  footer(s, 4);

  // ════════════════════════════ 5. THE MODEL ════════════════════════════════
  s = p.addSlide(); s.background = { color: PAPER };
  titleBar(s, "The production model", "A three-stage pipeline on a frozen foundation backbone");
  const stages = [
    [ic.micro, "Backbone", "ECGFounder 1-lead", "Frozen 150-head foundation model. Raw per-event probabilities.", BLUE],
    [ic.layers, "L1 · Calibration", "fzarkSL projection", "Re-calibrates data-rich heads (AFib, Brady, Tachy) to device operating points.", TEAL],
    [ic.bal, "L2 · Arbiter", "GT-matched rules", "Deterministic suppression of impossible co-firings. Cuts false alarms, no learned weights.", MINT],
  ];
  stages.forEach((st, i) => {
    const x = 0.6 + i * 4.27;
    card(s, x, 1.95, 3.9, 3.4);
    s.addShape(p.shapes.RECTANGLE, { x, y: 1.95, w: 3.9, h: 0.14, fill: { color: st[4] } });
    s.addShape(p.shapes.OVAL, { x: x + 1.55, y: 2.4, w: 0.8, h: 0.8, fill: { color: NAVY } });
    s.addImage({ data: st[0], x: x + 1.73, y: 2.58, w: 0.44, h: 0.44 });
    s.addText(st[1], { x, y: 3.35, w: 3.9, h: 0.4, fontSize: 19, bold: true, color: NAVY, align: "center", fontFace: "Georgia", margin: 0 });
    s.addText(st[2], { x, y: 3.78, w: 3.9, h: 0.3, fontSize: 13, italic: true, color: st[4], align: "center", margin: 0 });
    s.addText(st[3], { x: x + 0.3, y: 4.2, w: 3.3, h: 1.0, fontSize: 13, color: INK, align: "center", fontFace: "Calibri", lineSpacingMultiple: 1.18, margin: 0 });
  });
  [4.5, 8.77].forEach(x => s.addImage({ data: ic.arrow, x: x - 0.02, y: 3.5, w: 0.42, h: 0.42 }));
  s.addText("This study isolates each stage's contribution to the final result.", {
    x: 0.6, y: 5.7, w: 12.1, h: 0.5, fontSize: 14, italic: true, color: MUTE, align: "center", margin: 0 });
  footer(s, 5);

  // ════════════════════════════ 6. PRIMARY RESULTS ══════════════════════════
  s = p.addSlide(); s.background = { color: PAPER };
  titleBar(s, "Primary results · held-out TEST", "Raw head → L1 calibration → full production");
  s.addChart(p.charts.BAR, [
    { name: "Sensitivity", labels: ["Raw head @0.5", "+ L1 (L2-off)", "Full production"], values: [0.987, 0.895, 0.882] },
    { name: "Specificity", labels: ["Raw head @0.5", "+ L1 (L2-off)", "Full production"], values: [0.708, 0.876, 0.935] },
    { name: "F1", labels: ["Raw head @0.5", "+ L1 (L2-off)", "Full production"], values: [0.502, 0.657, 0.761] },
  ], {
    x: 0.6, y: 1.7, w: 7.5, h: 4.6, barDir: "col", barGrouping: "clustered",
    chartColors: [BLUE, TEAL, AMBER], chartArea: { fill: { color: PAPER } },
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK, dataLabelFontSize: 9, dataLabelFormatCode: "0.00",
    valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelColor: MUTE, catAxisLabelColor: INK, catAxisLabelFontSize: 11,
    valGridLine: { color: "EDF2F7", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "t", legendColor: INK, legendFontSize: 11,
  });
  // right callout panel
  card(s, 8.4, 1.95, 4.3, 4.05, NAVY);
  s.addShape(p.shapes.RECTANGLE, { x: 8.4, y: 1.95, w: 0.12, h: 4.05, fill: { color: AMBER } });
  s.addText("WHAT THE PIPELINE BUYS", { x: 8.7, y: 2.2, w: 3.8, h: 0.35, fontSize: 13, bold: true, color: AMBER, charSpacing: 2, margin: 0 });
  const gains = [
    ["Specificity", "0.71 → 0.94", "+23 pp false-positive reduction on normals"],
    ["Precision (PPV)", "0.34 → 0.67", "nearly doubled"],
    ["F1", "0.50 → 0.76", "undeployable → deployable"],
    ["Sensitivity cost", "0.99 → 0.88", "modest, intentional trade"],
  ];
  gains.forEach((g, i) => {
    const y = 2.65 + i * 0.83;
    s.addText(g[0], { x: 8.7, y, w: 3.8, h: 0.3, fontSize: 13, bold: true, color: "CBD5E1", margin: 0 });
    s.addText(g[1], { x: 8.7, y: y + 0.27, w: 3.8, h: 0.3, fontSize: 17, bold: true, color: i === 3 ? CORAL : MINT, margin: 0 });
    s.addText(g[2], { x: 8.7, y: y + 0.55, w: 3.8, h: 0.28, fontSize: 10, color: "94A3B8", italic: true, margin: 0 });
  });
  footer(s, 6);

  // ════════════════════════════ 7. L2 CONTRIBUTION ══════════════════════════
  s = p.addSlide(); s.background = { color: PAPER };
  titleBar(s, "Stage contribution", "L2 arbitration: more specificity at almost no sensitivity cost");
  s.addChart(p.charts.BAR, [
    { name: "L2-off (L1 only)", labels: ["Sensitivity (A)", "Specificity (N)", "Fire-rate Other ↓"], values: [0.895, 0.876, 0.282] },
    { name: "L2-on (full production)", labels: ["Sensitivity (A)", "Specificity (N)", "Fire-rate Other ↓"], values: [0.882, 0.935, 0.178] },
  ], {
    x: 0.6, y: 1.75, w: 7.6, h: 4.5, barDir: "col", barGrouping: "clustered",
    chartColors: [TEAL, MINT], chartArea: { fill: { color: PAPER } },
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK, dataLabelFontSize: 10, dataLabelFormatCode: "0.000",
    valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelColor: MUTE, catAxisLabelColor: INK, catAxisLabelFontSize: 11,
    valGridLine: { color: "EDF2F7", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "t", legendColor: INK, legendFontSize: 11,
  });
  const deltas = [
    ["Specificity", "+5.9 pp", MINT, "0.876 → 0.935"],
    ["Other fire-rate", "−10.4 pp", MINT, "0.282 → 0.178"],
    ["Sensitivity", "−1.3 pp", CORAL, "0.895 → 0.882"],
  ];
  deltas.forEach((d, i) => {
    const y = 1.95 + i * 1.35;
    card(s, 8.5, y, 4.2, 1.15);
    s.addText(d[0], { x: 8.75, y: y + 0.18, w: 3.7, h: 0.32, fontSize: 14, bold: true, color: NAVY, margin: 0 });
    s.addText(d[1], { x: 8.75, y: y + 0.46, w: 2.2, h: 0.5, fontSize: 26, bold: true, color: d[2], fontFace: "Georgia", margin: 0 });
    s.addText(d[3], { x: 10.7, y: y + 0.55, w: 1.8, h: 0.4, fontSize: 12, color: MUTE, align: "right", margin: 0 });
  });
  footer(s, 7);

  // ════════════════════════════ 8. KEY FINDING ══════════════════════════════
  s = p.addSlide(); s.background = { color: PAPER };
  titleBar(s, "Key technical finding", "The overlay calibrates the decision point — not the ranking");
  card(s, 0.6, 1.8, 5.85, 4.3, ICE);
  s.addImage({ data: ic.chart, x: 0.95, y: 2.1, w: 0.55, h: 0.55 });
  s.addText("Raw head ranks slightly better…", { x: 1.65, y: 2.12, w: 4.6, h: 0.5, fontSize: 16, bold: true, color: NAVY, fontFace: "Georgia", valign: "middle", margin: 0 });
  s.addText([
    { text: "Raw head ROC-AUC ", options: {} }, { text: "0.983", options: { bold: true, color: BLUE } },
    { text: "  vs  L1 ", options: {} }, { text: "0.946", options: { bold: true, color: TEAL } },
    { text: "  (test).", options: {} },
  ], { x: 0.95, y: 2.85, w: 5.2, h: 0.45, fontSize: 15, color: INK, margin: 0 });
  s.addText("…but the raw head is useless at its default 0.5 threshold — it fires on 29% of normal recordings.", {
    x: 0.95, y: 3.4, w: 5.25, h: 1.0, fontSize: 14, color: INK, fontFace: "Calibri", lineSpacingMultiple: 1.2, margin: 0 });
  s.addText([
    { text: "Implication: ", options: { bold: true, color: CORAL } },
    { text: "high AUC ≠ deployable. A usable operating point is what matters in production.", options: { italic: true } },
  ], { x: 0.95, y: 4.7, w: 5.25, h: 1.1, fontSize: 14, color: NAVY, fontFace: "Calibri", lineSpacingMultiple: 1.2, valign: "top", margin: 0 });

  card(s, 6.85, 1.8, 5.85, 4.3, NAVY);
  s.addShape(p.shapes.RECTANGLE, { x: 6.85, y: 1.8, w: 0.12, h: 4.3, fill: { color: MINT } });
  s.addText("WHAT THE OVERLAY DELIVERS", { x: 7.2, y: 2.1, w: 5.3, h: 0.35, fontSize: 13, bold: true, color: MINT, charSpacing: 2, margin: 0 });
  s.addText([
    { text: "L1", options: { bold: true, color: AMBER } },
    { text: "  maps the raw score to a usable threshold — specificity ", options: {} },
    { text: "0.876", options: { bold: true, color: MINT } },
    { text: " at τ = 0.54.", options: {} },
  ], { x: 7.2, y: 2.6, w: 5.3, h: 0.85, fontSize: 15, color: "E2E8F0", lineSpacingMultiple: 1.2, valign: "top", margin: 0 });
  s.addText([
    { text: "L2", options: { bold: true, color: AMBER } },
    { text: "  adds specificity on top → ", options: {} },
    { text: "0.935", options: { bold: true, color: MINT } },
    { text: ", changing the binary decision, not the score (so AUC is unchanged L2-off vs on).", options: {} },
  ], { x: 7.2, y: 3.55, w: 5.3, h: 1.1, fontSize: 15, color: "E2E8F0", lineSpacingMultiple: 1.2, valign: "top", margin: 0 });
  s.addText([
    { text: "Headroom: ", options: { bold: true, color: AMBER } },
    { text: "re-fitting L1 toward this distribution could recover ranking AUC for a multi-device operating point.", options: {} },
  ], { x: 7.2, y: 4.95, w: 5.3, h: 1.0, fontSize: 14, italic: true, color: "CBD5E1", lineSpacingMultiple: 1.2, valign: "top", margin: 0 });
  footer(s, 8);

  // ════════════════════════════ 9. CLINICAL ═════════════════════════════════
  s = p.addSlide(); s.background = { color: PAPER };
  titleBar(s, "Clinical interpretation", "What the numbers mean at the bedside");
  const cl = [
    [ic.heart, "Catches ~8 of 9 AFib", "88% sensitivity on an unseen handheld device — strong for ambulatory screening / triage."],
    [ic.shield, "Clears ~19 of 20 normals", "93.5% specificity keeps the false-alarm burden low, reducing clinician alert fatigue."],
    [ic.check, "Fewer spurious alerts", "−23 pp false-positive reduction vs the raw model means far fewer non-AFib alerts reach review."],
    [ic.warn, "“Other” fires ~18%", "Expected — that class contains AF-like and other arrhythmias; not a pure false-positive rate."],
  ];
  cl.forEach((r, i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 1.85 + Math.floor(i / 2) * 2.15;
    card(s, x, y, 5.85, 1.9);
    s.addShape(p.shapes.OVAL, { x: x + 0.3, y: y + 0.35, w: 0.85, h: 0.85, fill: { color: i === 3 ? "5A4216" : NAVY } });
    s.addImage({ data: r[0], x: x + 0.49, y: y + 0.54, w: 0.47, h: 0.47 });
    s.addText(r[1], { x: x + 1.35, y: y + 0.28, w: 4.3, h: 0.55, fontSize: 17, bold: true, color: NAVY, fontFace: "Georgia", valign: "middle", margin: 0 });
    s.addText(r[2], { x: x + 1.35, y: y + 0.85, w: 4.35, h: 0.9, fontSize: 12.5, color: INK, fontFace: "Calibri", lineSpacingMultiple: 1.15, valign: "top", margin: 0 });
  });
  footer(s, 9);

  // ════════════════════════════ 10. LIMITATIONS ═════════════════════════════
  s = p.addSlide(); s.background = { color: PAPER };
  titleBar(s, "Limitations & next steps", "Honest scope and where we go from here");
  card(s, 0.6, 1.8, 6.0, 4.3, "FCF3EC");
  s.addImage({ data: ic.warn, x: 0.9, y: 2.05, w: 0.5, h: 0.5 });
  s.addText("Limitations", { x: 1.5, y: 2.07, w: 4.8, h: 0.45, fontSize: 19, bold: true, color: "9A4A1E", fontFace: "Georgia", valign: "middle", margin: 0 });
  s.addText([
    { text: "Only AFib maps 1:1 to a production scope event — other arrhythmia heads not evaluable here.", options: { bullet: true, breakLine: true } },
    { text: "“Other” / “Noisy” are not clean negatives — shown as context, excluded from binary metrics.", options: { bullet: true, breakLine: true } },
    { text: "Recording-level any-window-fires rollup is sensitive to recording length.", options: { bullet: true, breakLine: true } },
    { text: "Modest positive count (76 AFib in test) — consistency with validation mitigates this.", options: { bullet: true } },
  ], { x: 0.95, y: 2.75, w: 5.4, h: 3.2, fontSize: 13.5, color: INK, fontFace: "Calibri", paraSpaceAfter: 10, valign: "top", margin: 0 });

  card(s, 6.9, 1.8, 5.8, 4.3, "E9F3F1");
  s.addImage({ data: ic.flag, x: 7.2, y: 2.05, w: 0.5, h: 0.5 });
  s.addText("Next steps", { x: 7.8, y: 2.07, w: 4.6, h: 0.45, fontSize: 19, bold: true, color: "0F6B5C", fontFace: "Georgia", valign: "middle", margin: 0 });
  s.addText([
    { text: "Optionally re-fit L1 for a multi-device operating point to recover ranking-AUC headroom.", options: { bullet: true, breakLine: true } },
    { text: "Extend validation to additional external single-lead datasets.", options: { bullet: true, breakLine: true } },
    { text: "Report confidence intervals with a larger positive sample.", options: { bullet: true, breakLine: true } },
    { text: "Stage remaining tp_rex classes to broaden multi-event coverage.", options: { bullet: true } },
  ], { x: 7.25, y: 2.75, w: 5.2, h: 3.2, fontSize: 13.5, color: INK, fontFace: "Calibri", paraSpaceAfter: 10, valign: "top", margin: 0 });
  footer(s, 10);

  // ════════════════════════════ 11. CONCLUSION ══════════════════════════════
  s = p.addSlide(); s.background = { color: DEEP };
  s.addShape(p.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: 0.18, fill: { color: TEAL } });
  s.addImage({ data: ic.flag, x: 0.7, y: 0.85, w: 0.7, h: 0.7 });
  s.addText("Conclusions", { x: 0.7, y: 1.6, w: 11, h: 0.7, fontSize: 34, bold: true, color: PAPER, fontFace: "Georgia", margin: 0 });
  const concl = [
    ["Generalizes out-of-box", "0.95 AUC, 88% / 94% sensitivity / specificity on an independent single-lead device — the calibration is not overfit to our device."],
    ["The overlay is essential", "Without L1 + L2 the raw model is undeployable (fires on 29% of normals). L2 alone buys +5.9 pp specificity at ~zero recall cost."],
    ["Clear path forward", "Re-fit L1 for multi-device deployment; broaden external validation; report CIs on a larger positive sample."],
  ];
  concl.forEach((c, i) => {
    const y = 2.55 + i * 1.35;
    s.addShape(p.shapes.OVAL, { x: 0.75, y: y + 0.05, w: 0.55, h: 0.55, fill: { color: MINT } });
    s.addText(String(i + 1), { x: 0.75, y: y + 0.05, w: 0.55, h: 0.55, fontSize: 22, bold: true, color: DEEP, align: "center", valign: "middle", fontFace: "Georgia", margin: 0 });
    s.addText(c[0], { x: 1.55, y: y, w: 11, h: 0.45, fontSize: 20, bold: true, color: MINT, fontFace: "Georgia", margin: 0 });
    s.addText(c[1], { x: 1.55, y: y + 0.45, w: 11.1, h: 0.75, fontSize: 14.5, color: "CBD5E1", fontFace: "Calibri", lineSpacingMultiple: 1.15, valign: "top", margin: 0 });
  });
  s.addText("Artifacts: PROD_AFIB.md · CHALLENGE2017_STUDY_REPORT.md · csv/challenge2017_split.csv · scripts/eval_challenge2017_prod.py",
    { x: 0.7, y: 7.0, w: 12, h: 0.3, fontSize: 10, italic: true, color: MUTE, margin: 0 });

  await p.writeFile({ fileName: "res/challenge2017/Challenge2017_AFib_Executive_Briefing.pptx" });
  console.log("wrote res/challenge2017/Challenge2017_AFib_Executive_Briefing.pptx");
})();
