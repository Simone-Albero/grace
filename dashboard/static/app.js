// Dashboard front-end: one page, one tab per stage. Charts are hand-made SVG, no dependencies.

/**
 * UI state.
 * @type {{dataset: string|null, model: string|null, split: "val"|"test",
 *   tab: "split"|"regions"|"complexity"|"train"|"failures"|"regress",
 *   mode: "count"|"row", cls: string|null, measure: string|null, region: number|null}}
 */
const state = {
  dataset: null, model: null, split: "test", tab: "split",
  mode: "count", cls: null, measure: null, region: null,
};
const MODEL_TABS = ["train", "failures", "regress"];

const last = { train: null, regions: null }; // last responses, for the controls that only redraw
let loadId = 0; // drops answers of superseded requests

const $ = (id) => document.getElementById(id);
const NS = { html: "http://www.w3.org/1999/xhtml", svg: "http://www.w3.org/2000/svg" };

// --- formatting ---

const pct = (v, d = 1) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);
const num = (v) => (v == null ? "–" : Number(v).toLocaleString("en-US"));
const dec = (v, d = 3) => (v == null ? "–" : Number(v).toFixed(d));
const short = (v) => (v == null ? "–" : String(Number(Number(v).toPrecision(3))));
const when = (iso) => (iso ? `${iso.replace("T", " ").slice(0, 16)} UTC` : "–");
const params = (p) => Object.entries(p).map(([k, v]) => `${k}=${v}`).join(", ") || "defaults";

// --- dom helpers ---

function make(ns, tag, attrs, kids) {
  const el = document.createElementNS(NS[ns], tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  el.append(...kids.flat(Infinity).filter((k) => k != null));
  return el;
}
const h = (tag, attrs = {}, ...kids) => make("html", tag, attrs, kids);
const sv = (tag, attrs = {}, ...kids) => make("svg", tag, attrs, kids);

function table(headers, rows) {
  return h(
    "table", {},
    h("thead", {}, h("tr", {}, headers.map((x) => h("th", {}, x)))),
    h("tbody", {}, rows.map((r) => h("tr", {}, r.map((c) => h("td", {}, c))))),
  );
}

/**
 * A table sorted by clicking a header (numbers descending first, text ascending first).
 * @param {HTMLElement} el container, replaced
 * @param {{key: string, label: string, format?: (v: any) => any}[]} columns
 * @param {object[]} rows one object per row, keyed by column key
 * @param {{onClick?: (row: object) => void, selected?: (row: object) => boolean}} [opts]
 */
function sortable(el, columns, rows, opts = {}) {
  let by = null, dir = 1;
  const draw = () => {
    const sorted = by == null ? rows : [...rows].sort((a, b) => {
      const x = a[by], y = b[by];
      if (x == null) return 1;
      if (y == null) return -1;
      return (typeof x === "number" ? y - x : String(x).localeCompare(String(y))) * dir;
    });
    const head = columns.map((c) => {
      const th = h("th", { class: "sortable" }, c.label);
      if (c.key === by) {
        const descending = typeof rows[0]?.[by] === "number" ? dir > 0 : dir < 0;
        th.setAttribute("aria-sort", descending ? "descending" : "ascending");
      }
      th.onclick = () => { dir = by === c.key ? -dir : 1; by = c.key; draw(); };
      return th;
    });
    const body = sorted.map((r) => {
      const tr = h("tr", {}, columns.map((c) => h("td", {}, (c.format ?? String)(r[c.key]))));
      if (opts.onClick) { tr.classList.add("clickable"); tr.onclick = () => opts.onClick(r); }
      if (opts.selected?.(r)) tr.classList.add("selected");
      return tr;
    });
    el.replaceChildren(h("table", {}, h("thead", {}, h("tr", {}, head)), h("tbody", {}, body)));
  };
  draw();
}

function tile(label, value, sub) {
  return h("div", { class: "tile" },
    h("div", { class: "label" }, label),
    h("div", { class: "value" }, value),
    sub ? h("div", { class: "sub" }, sub) : null);
}

function fill(id, ...kids) {
  $(id).replaceChildren(...kids.flat().filter((k) => k != null));
}

// --- data ---

/**
 * GET a JSON endpoint under /api.
 * @param {string} path e.g. "/synthetic_test/split"
 * @returns {Promise<object>} parsed body; rejects with the server "detail" on 404 (shown as the empty state)
 */
async function api(path) {
  const res = await fetch(`/api${path}`);
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error(typeof body?.detail === "string" ? body.detail : `request failed (${res.status})`);
  return body;
}

/** Fill the selectors from /api/options, restore tab/dataset/model from the URL hash, wire the controls, load the tab. */
async function init() {
  const opts = await api("/options");
  const fillSelect = (el, values) => el.replaceChildren(...values.map((v) => h("option", {}, v)));
  fillSelect($("dataset"), opts.datasets);
  fillSelect($("model"), opts.models);
  const [tab, dataset, model] = location.hash.slice(1).split("/").map(decodeURIComponent);
  if ([...document.querySelectorAll(".tabs button")].some((b) => b.dataset.tab === tab)) state.tab = tab;
  state.dataset = opts.datasets.includes(dataset) ? dataset : opts.datasets[0];
  state.model = opts.models.includes(model) ? model : opts.models[0];
  $("dataset").value = state.dataset;
  $("model").value = state.model;
  $("dataset").onchange = (e) => { state.dataset = e.target.value; state.cls = state.measure = state.region = null; load(); };
  $("model").onchange = (e) => { state.model = e.target.value; state.region = null; load(); };
  $("split").onchange = (e) => { state.split = e.target.value; load(); };
  $("rg-class").onchange = (e) => { state.cls = e.target.value; if (last.regions) drawCandidates(last.regions); };
  $("cx-measure").onchange = (e) => { state.measure = e.target.value; load(); };
  for (const b of document.querySelectorAll(".tabs button")) b.onclick = () => showTab(b.dataset.tab);
  for (const b of document.querySelectorAll("#tr-mode button")) {
    b.onclick = () => {
      state.mode = b.dataset.mode;
      for (const o of document.querySelectorAll("#tr-mode button")) o.setAttribute("aria-pressed", o === b);
      if (last.train) drawConfusion(last.train);
    };
  }
  showTab(state.tab);
}

/** Show a tab, hide the others; the model selector shows on the model stages, the split on "train". */
function showTab(tab) {
  state.tab = tab;
  for (const b of document.querySelectorAll(".tabs button")) b.setAttribute("aria-selected", b.dataset.tab === tab);
  for (const p of document.querySelectorAll(".panel")) p.hidden = p.id !== `tab-${tab}`;
  $("model-label").hidden = !MODEL_TABS.includes(tab);
  $("split-label").hidden = tab !== "train";
  load();
}

/** Fetch the data of state.tab and render it; on error show the message in the tab's empty state. */
async function load() {
  const id = ++loadId;
  const tab = state.tab;
  history.replaceState(null, "", `#${[tab, state.dataset, state.model].map(encodeURIComponent).join("/")}`);
  const panel = $(`tab-${tab}`);
  const empty = panel.querySelector(".empty");
  const d = `/${encodeURIComponent(state.dataset)}`, m = `model=${encodeURIComponent(state.model)}`;
  const path = {
    split: `${d}/split`,
    regions: `${d}/regions`,
    complexity: `${d}/complexity${state.measure ? `?measure=${encodeURIComponent(state.measure)}` : ""}`,
    train: `${d}/train?${m}&split=${state.split}`,
    failures: `${d}/failures?${m}`,
    regress: `${d}/regress?${m}`,
  }[tab];
  try {
    const data = await api(path);
    if (id !== loadId) return;
    panel.classList.remove("failed");
    empty.hidden = true;
    ({
      split: renderSplit, regions: renderRegions, complexity: renderComplexity,
      train: renderTrain, failures: renderFailures, regress: renderRegress,
    })[tab](data);
    renderRun(data.run);
  } catch (e) {
    if (id !== loadId) return;
    panel.classList.add("failed");
    empty.hidden = false;
    empty.textContent = e.message;
    renderRun(null);
  }
}

// --- tabs ---

/** Show the id and date of the displayed run. @param {{id: number, started: string, finished: string}|null} run */
function renderRun(run) {
  $("run-info").textContent = run ? `run #${run.id} · ${when(run.finished || run.started)}` : "";
}

/** Refill the model selector with the models that have a run of the current stage. @param {string[]} models */
function setModels(models) {
  $("model").replaceChildren(...models.map((m) => h("option", {}, m)));
  $("model").value = state.model;
}

/** Tiles, classes per split with inline bars, dropped classes and columns, features. @param {object} data response of /split */
function renderSplit(data) {
  const s = data.summary;
  const totals = ["train", "val", "test"].map((k) => data.classes.reduce((a, c) => a + c[k], 0));
  fill("sp-tiles",
    tile("rows read", num(s.n_raw)),
    tile("rows kept", num(s.n_kept), `${num(s.n_rare)} in rare classes`),
    tile("classes", num(s.n_classes), `${data.dropped_classes.length} dropped`),
    ["train", "val", "test"].map((k, i) => tile(k, num(totals[i]), pct(totals[i] / s.n_kept))));
  const top = Math.max(...data.classes.map((c) => c.train + c.val + c.test), 1);
  fill("sp-classes", table(
    ["class", "rows", "train", "val", "test"],
    data.classes.map((c) => {
      const n = c.train + c.val + c.test;
      return [c.label, h("span", {}, h("i", { class: "bar", style: `width:${Math.round((n / top) * 60)}px` }), num(n)),
        num(c.train), num(c.val), num(c.test)];
    })));
  fill("sp-dropped", data.dropped_classes.map((c) => h("span", { class: "chip" }, `${c.label} (${num(c.n_rows)})`)));
  fill("sp-constant", data.dropped_columns.map((c) => h("span", { class: "chip" }, c)));
  fill("sp-features",
    h("h3", {}, `numeric (${data.features.num.length})`),
    h("div", { class: "chips" }, data.features.num.map((c) => h("span", { class: "chip" }, c))),
    h("h3", { style: "margin-top:12px" }, `categorical (${data.features.cat.length})`),
    h("div", { class: "chips" }, data.features.cat.map((c) => h("span", { class: "chip" }, c))));
}

/** Tiles, granularity per class, candidates of a class, regions table. @param {object} data response of /regions */
function renderRegions(data) {
  last.regions = data;
  const s = data.stats;
  fill("rg-tiles",
    tile("regions", num(s.n_regions)),
    tile("algorithm", s.algorithm),
    tile("distance", s.distance),
    tile("graph nodes", num(s.n_graph_nodes)),
    tile("test rows per train row", dec(s.eval_ratio, 2)));
  sortable($("rg-classes"), [
    { key: "class", label: "class" },
    { key: "n_train", label: "train rows", format: num },
    { key: "n_regions", label: "regions", format: num },
    { key: "loss", label: "loss", format: (v) => dec(v, 4) },
    { key: "params", label: "chosen", format: params },
  ], data.classes, { onClick: (r) => { state.cls = r.class; $("rg-class").value = r.class; drawCandidates(data); } });
  const classes = data.classes.map((c) => c.class);
  if (!classes.includes(state.cls)) state.cls = classes[0];
  $("rg-class").replaceChildren(...classes.map((c) => h("option", {}, c)));
  $("rg-class").value = state.cls;
  drawCandidates(data);
  const sizes = data.regions.map((r) => r.n_test);
  const edges = Array.from({ length: 21 }, (_, i) => (Math.max(...sizes, 1) * i) / 20);
  histogram($("rg-sizes"), edges, [{
    name: "regions by test rows", color: "var(--accent)",
    values: edges.slice(0, -1).map((lo, i) => sizes.filter((v) => v >= lo && (v < edges[i + 1] || i === 19)).length),
  }], { format: num });
  sortable($("rg-table"), [
    { key: "region", label: "region" },
    { key: "class", label: "class" },
    { key: "n_train", label: "train", format: num },
    { key: "n_val", label: "val", format: num },
    { key: "n_test", label: "test", format: num },
  ], data.regions);
}

/** Loss of every candidate of state.cls against its clusters, the chosen one marked. @param {object} data response of /regions */
function drawCandidates(data) {
  const rows = data.candidates.filter((c) => c.class === state.cls);
  const chosen = data.classes.find((c) => c.class === state.cls);
  const isChosen = (c) => JSON.stringify(c.params) === JSON.stringify(chosen?.params);
  scatter($("rg-loss"), rows.map((c) => ({
    x: c.n_clusters, y: c.loss, selected: isChosen(c),
    tip: `${params(c.params)} → ${c.n_clusters} clusters, loss ${dec(c.loss, 5)}${isChosen(c) ? " (chosen)" : ""}`,
  })), { xLabel: "clusters after the merge", yLabel: "granularity loss", line: true });
}

/** Strip of the measure by class with the class value, and the summary of every measure. @param {object} data response of /complexity */
function renderComplexity(data) {
  state.measure = data.measure;
  $("cx-measure").replaceChildren(...data.measures.map((m) => h("option", {}, m)));
  $("cx-measure").value = data.measure;
  const classValue = Object.fromEntries(data.classes.map((c) => [c.class, c.value]));
  strip($("cx-strip"), data.classes.map((c) => c.class), data.regions.map((r) => ({
    group: r.class, y: r.value, tip: `region ${r.region} (${r.class}): ${short(r.value)}`,
  })), { marks: classValue, yLabel: data.measure });
  sortable($("cx-summary"), [
    { key: "measure", label: "measure" },
    { key: "mean", label: "mean", format: short },
    { key: "min", label: "min", format: short },
    { key: "max", label: "max", format: short },
    { key: "n_null", label: "undefined", format: num },
  ], data.summary, { onClick: (r) => { state.measure = r.measure; load(); } });
}

/** Metric tiles, F1 bars, confusion matrix, confidence histogram. @param {object} data response of /train */
function renderTrain(data) {
  last.train = data;
  setModels(data.models);
  const o = data.overall;
  fill("tr-tiles",
    tile("accuracy", dec(o.accuracy)), tile("macro F1", dec(o.macro_f1)), tile("weighted F1", dec(o.weighted_f1)));
  barsH($("tr-f1"), data.classes.map((c) => ({
    label: c.label, value: c.f1,
    detail: `P ${dec(c.precision, 2)} · R ${dec(c.recall, 2)} · n ${num(c.support)}`,
  })));
  drawConfusion(data);
  const c = data.confidence;
  histogram($("tr-confidence"), c.edges, [
    { name: "correct", color: "var(--accent)", values: c.correct },
    { name: "wrong", color: "var(--series-2)", values: c.wrong },
  ], { format: num });
}

function drawConfusion(data) {
  const { labels, counts } = data.confusion;
  const totals = counts.map((r) => r.reduce((a, b) => a + b, 0));
  const row = state.mode === "row";
  const values = row ? counts.map((r, i) => r.map((v) => (totals[i] ? v / totals[i] : 0))) : counts;
  heatmap($("tr-confusion"), labels, labels, values, {
    format: row ? (v) => (v ? `${Math.round(v * 100)}` : "") : (v) => (v ? num(v) : ""),
    tip: (i, j) => `true ${labels[i]} → predicted ${labels[j]}: ${num(counts[i][j])}` +
      (totals[i] ? ` (${pct(counts[i][j] / totals[i])} of row)` : ""),
  });
}

const SCORE_COLUMNS = [
  { key: "predictor", label: "predictor" },
  { key: "spearman", label: "Spearman ρ", format: dec },
  { key: "r2", label: "R²", format: dec },
  { key: "mae", label: "MAE", format: dec },
  { key: "mse", label: "MSE", format: (v) => dec(v, 4) },
];

/** Tiles, baseline scores and scatters, rate by class, regions table. @param {object} data response of /failures */
function renderFailures(data) {
  setModels(data.models);
  const s = data.stats;
  fill("fl-tiles",
    tile("regions used", `${num(s.n_used)} / ${num(s.n_regions)}`, `${num(s.n_empty)} without test rows, ${num(s.n_small)} too small`),
    tile("test error rate", pct(s.error_rate), `${pct(s.error_rate_used)} on the used regions`),
    tile("ATC threshold", dec(s.atc_threshold)));
  sortable($("fl-scores"), SCORE_COLUMNS, data.scores);
  const used = data.regions.filter((r) => r.used);
  const pair = ["mcp", "atc"].map((key) => {
    const el = h("div", {});
    scatter(el, used.map((r) => ({
      x: r[key], y: r.failure_rate,
      tip: `region ${r.region} (${r.class}): ${key.toUpperCase()} ${dec(r[key])}, observed ${dec(r.failure_rate)}, ${num(r.n_eval)} rows`,
    })), { xLabel: key.toUpperCase(), yLabel: "observed failure rate", diagonal: true, max: 1, width: 560 });
    return h("div", {}, h("h3", {}, key.toUpperCase()), el);
  });
  fill("fl-scatter", pair);
  const withRows = data.regions.filter((r) => r.n_eval > 0);
  const classes = [...new Set(data.regions.map((r) => r.class))];
  const top = Math.max(...withRows.map((r) => r.n_eval), 1);
  strip($("fl-strip"), classes, withRows.map((r) => ({
    group: r.class, y: r.failure_rate, r: 2 + 7 * Math.sqrt(r.n_eval / top),
    tip: `region ${r.region} (${r.class}): ${pct(r.failure_rate)} of ${num(r.n_eval)} test rows`,
  })), { yLabel: "observed failure rate", max: 1 });
  sortable($("fl-table"), [
    { key: "region", label: "region" },
    { key: "class", label: "class" },
    { key: "n_eval", label: "test rows", format: num },
    { key: "n_error", label: "errors", format: num },
    { key: "failure_rate", label: "failure rate", format: (v) => pct(v) },
    { key: "mcp", label: "MCP", format: (v) => pct(v) },
    { key: "atc", label: "ATC", format: (v) => pct(v) },
    { key: "used", label: "used", format: (v) => (v ? "yes" : "no") },
  ], data.regions);
}

/** Tiles, grouped score bars, OOF scatter, importances, folds; the selected region's explanation. @param {object} data response of /regress */
function renderRegress(data) {
  setModels(data.models);
  const by = Object.fromEntries(data.scores.map((s) => [s.predictor, s]));
  fill("rr-tiles",
    tile("regions", num(data.estimates.length)),
    ...["rf", "mcp", "atc"].map((p) => tile(`${p.toUpperCase()} Spearman ρ`, dec(by[p]?.spearman), `MAE ${dec(by[p]?.mae)}`)));
  const metrics = ["spearman", "r2", "mae"];
  groupedBars($("rr-scores"), metrics, [
    { name: "RF", color: "var(--accent)", values: metrics.map((k) => by.rf?.[k]) },
    { name: "MCP", color: "var(--series-2)", values: metrics.map((k) => by.mcp?.[k]) },
    { name: "ATC", color: "var(--series-3)", values: metrics.map((k) => by.atc?.[k]) },
  ]);
  scatter($("rr-scatter"), data.estimates.map((e) => ({
    x: e.predicted, y: e.observed, id: e.region, selected: e.region === state.region,
    tip: `region ${e.region} (${e.class}): estimated ${dec(e.predicted)}, observed ${dec(e.observed)}, fold ${e.fold}`,
  })), {
    xLabel: "RF estimate (out of fold)", yLabel: "observed failure rate", diagonal: true, max: 1,
    onClick: (p) => { state.region = p.id; renderRegress(data); },
  });
  const pair = [["mdi", "impurity (MDI)"], ["mean_abs_shap", "mean |SHAP|"]].map(([key, title]) => {
    const rows = [...data.importances].sort((a, b) => b[key] - a[key]).slice(0, 15);
    const el = h("div", {});
    barsH(el, rows.map((r) => ({ label: r.descriptor, value: r[key] })), { max: Math.max(...rows.map((r) => r[key]), 1e-9), format: (v) => dec(v, 4) });
    return h("div", {}, h("h3", {}, title), el);
  });
  fill("rr-importances", pair);
  sortable($("rr-folds"), [
    { key: "fold", label: "fold" },
    { key: "n_regions", label: "regions", format: num },
    { key: "spearman", label: "Spearman ρ", format: dec },
    { key: "r2", label: "R²", format: dec },
    { key: "mae", label: "MAE", format: dec },
    { key: "best_score", label: "inner R²", format: dec },
    { key: "params", label: "params", format: params },
  ], data.folds);
  if (state.region != null && data.estimates.some((e) => e.region === state.region)) drawRegion(state.region);
  else $("rr-region-card").hidden = true;
}

/** Waterfall of the TreeSHAP contributions and the descriptors of one region. @param {number} region */
async function drawRegion(region) {
  const d = `/${encodeURIComponent(state.dataset)}`;
  const data = await api(`${d}/regress/region?model=${encodeURIComponent(state.model)}&region=${region}`);
  if (state.region !== region) return;
  const r = data.region;
  $("rr-region-card").hidden = false;
  $("rr-region-title").textContent = `Region ${r.region} · ${r.class} · estimated ${dec(r.predicted)}, observed ${dec(r.observed)}`;
  waterfall($("rr-waterfall"), r.base, data.contributions.slice(0, 12), data.contributions.slice(12));
  const descriptor = (m) => data.contributions.find((c) => c.descriptor === `region_${m}`);
  sortable($("rr-descriptors"), [
    { key: "measure", label: "measure" },
    { key: "region", label: "region", format: short },
    { key: "class", label: "class", format: short },
    { key: "class_mean", label: "class mean", format: short },
    { key: "dataset_mean", label: "dataset mean", format: short },
    { key: "contribution", label: "SHAP (region_)", format: (v) => (v == null ? "" : dec(v, 4)) },
  ], data.descriptors.map((x) => ({ ...x, contribution: descriptor(x.measure)?.contribution ?? null })));
}

// --- charts ---

/**
 * Horizontal bars, one row per item, value labelled at the bar end.
 * @param {HTMLElement} el container, replaced
 * @param {{label: string, value: number, detail?: string, color?: string}[]} rows value in [0, max]
 * @param {{max?: number, format?: (v: number) => string}} [opts] full-length value (default 1), value label (default dec)
 */
function barsH(el, rows, opts = {}) {
  const max = opts.max ?? 1, format = opts.format ?? dec;
  const W = 800, labelW = 230, detailW = 190, rowH = 24, bar = 8;
  const track = W - labelW - detailW - 48;
  const kids = rows.flatMap((r, i) => {
    const y = i * rowH;
    const w = Math.max(0, ((r.value ?? 0) / max) * track);
    const g = sv("g", {},
      sv("rect", { x: 0, y, width: W, height: rowH, fill: "transparent" }),
      sv("text", { x: labelW - 12, y: y + rowH / 2 + 4, "text-anchor": "end", style: "fill:var(--text-2)" }, r.label),
      sv("rect", { x: labelW, y: y + (rowH - bar) / 2, width: w, height: bar, rx: 2, style: `fill:${r.color ?? "var(--accent)"}` }),
      sv("text", { x: labelW + w + 8, y: y + rowH / 2 + 4, style: "fill:var(--text)" }, format(r.value)),
      sv("text", { x: W, y: y + rowH / 2 + 4, "text-anchor": "end" }, r.detail ?? ""));
    g.onmousemove = (ev) => tip(ev, `${r.label}: ${dec(r.value, 4)}${r.detail ? ` · ${r.detail}` : ""}`);
    g.onmouseleave = () => tip(null);
    return g;
  });
  const axis = sv("line", { class: "axis", x1: labelW, x2: labelW, y1: 0, y2: rows.length * rowH });
  el.replaceChildren(sv("svg", { viewBox: `0 0 ${W} ${Math.max(rows.length * rowH, 1)}`, role: "img" }, axis, kids));
}

/** Smallest 1, 2, 5 or 10 times a power of ten that is >= v, so the ticks are round. */
function niceMax(v) {
  const p = 10 ** Math.floor(Math.log10(v));
  return [1, 2, 5, 10].map((m) => m * p).find((x) => x >= v);
}

/** Y-axis gridlines and tick labels from lo to hi. */
function yTicks(kids, y, lo, hi, m, W, format = short) {
  for (let k = 0; k <= 4; k++) {
    const v = lo + ((hi - lo) * k) / 4;
    kids.push(
      sv("line", { class: k ? "grid" : "axis", x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }),
      sv("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end" }, format(v)));
  }
}

/**
 * Clustered bins per series with a legend.
 * @param {HTMLElement} el container, replaced
 * @param {number[]} edges bins + 1 values
 * @param {{name: string, color: string, values: number[]}[]} series one value per bin
 * @param {{format?: (v: number) => string}} [opts] y axis and tooltip number format
 */
function histogram(el, edges, series, opts = {}) {
  const format = opts.format ?? short;
  const W = 800, H = 240, m = { l: 52, r: 24, t: 8, b: 26 };
  const bins = edges.length - 1;
  const top = niceMax(Math.max(...series.flatMap((s) => s.values), 1e-9));
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const bw = pw / bins;
  const y = (v) => m.t + ph - (v / top) * ph;
  const kids = [];
  yTicks(kids, y, 0, top, m, W, format);
  for (let k = 0; k <= 4; k++) {
    const i = Math.round((bins * k) / 4);
    kids.push(sv("text", { x: m.l + i * bw, y: H - 8, "text-anchor": "middle" }, short(edges[i])));
  }
  const slot = (bw - 4) / series.length;
  for (let i = 0; i < bins; i++) {
    series.forEach((s, j) => {
      kids.push(sv("rect", {
        x: m.l + i * bw + 2 + j * slot, y: y(s.values[i]), width: Math.max(1, slot - 2), height: ph - (y(s.values[i]) - m.t),
        rx: 2, style: `fill:${s.color}`,
      }));
    });
    const hit = sv("rect", { x: m.l + i * bw, y: m.t, width: bw, height: ph, fill: "transparent" });
    const text = `${short(edges[i])} – ${short(edges[i + 1])}: ` + series.map((s) => `${s.name} ${format(s.values[i])}`).join(" · ");
    hit.onmousemove = (ev) => tip(ev, text);
    hit.onmouseleave = () => tip(null);
    kids.push(hit);
  }
  el.replaceChildren(
    h("div", { class: "legend" }, series.map((s) => h("span", {}, h("i", { style: `background:${s.color}` }), s.name))),
    sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, kids));
}

/**
 * Heatmap with row and column labels, a one-hue sequential scale and a tooltip per cell.
 * @param {HTMLElement} el container, replaced
 * @param {string[]} rows row labels
 * @param {string[]} cols column labels
 * @param {number[][]} values rows x cols
 * @param {{format?: (v: number) => string, tip?: (i: number, j: number) => string}} [opts]
 */
function heatmap(el, rows, cols, values, opts = {}) {
  const format = opts.format ?? ((v) => dec(v, 2));
  const W = 1100, charW = 6.2;
  const left = Math.max(...rows.map((r) => r.length)) * charW + 12;
  const colLen = Math.max(...cols.map((c) => c.length)) * charW;
  const topPad = colLen * 0.87 + 10, right = colLen * 0.5 + 8;
  const cw = Math.min(64, Math.floor((W - left - right) / cols.length));
  const ch = Math.min(cw, 28);
  const flat = values.flat().filter((x) => x != null);
  const lo = Math.min(...flat), hi = Math.max(...flat);
  const kids = [];
  cols.forEach((c, j) => kids.push(
    sv("text", { transform: `translate(${left + j * cw + cw / 2}, ${topPad - 6}) rotate(-60)`, "text-anchor": "start" }, c)));
  rows.forEach((r, i) => {
    kids.push(sv("text", { x: left - 8, y: topPad + i * ch + ch / 2 + 4, "text-anchor": "end", style: "fill:var(--text-2)" }, r));
    cols.forEach((_, j) => {
      const v = values[i][j];
      const t = v == null ? 0 : hi > lo ? (v - lo) / (hi - lo) : 0;
      const x = left + j * cw, yy = topPad + i * ch;
      const cell = sv("rect", {
        class: "cell", x: x + 1, y: yy + 1, width: cw - 2, height: ch - 2, rx: 2,
        style: v == null ? "fill:var(--grid)" : `fill:color-mix(in srgb, var(--accent) ${Math.round(t * 100)}%, var(--surface))`,
      });
      cell.onmousemove = (ev) => tip(ev, opts.tip ? opts.tip(i, j) : `${rows[i]} × ${cols[j]}: ${v == null ? "–" : format(v)}`);
      cell.onmouseleave = () => tip(null);
      kids.push(cell);
      const label = v == null ? "" : format(v);
      if (label && cw >= 30) {
        kids.push(sv("text", {
          x: x + cw / 2, y: yy + ch / 2 + 4, "text-anchor": "middle", "pointer-events": "none",
          style: `fill:${t > 0.55 ? "#fff" : "var(--text-2)"}; font-size:10px`,
        }, label));
      }
    });
  });
  el.replaceChildren(h("div", { class: "table-wrap" },
    sv("svg", { viewBox: `0 0 ${W} ${topPad + rows.length * ch + 4}`, role: "img", style: "min-width:640px" }, kids)));
}

/**
 * Scatter of points with axes from 0; optionally a y = x diagonal, a joining line and clickable dots.
 * @param {HTMLElement} el container, replaced
 * @param {{x: number, y: number, tip: string, id?: any, selected?: boolean}[]} points
 * @param {{xLabel: string, yLabel: string, diagonal?: boolean, line?: boolean, max?: number,
 *   onClick?: (p: object) => void, width?: number}} opts max fixes both axes (else each from its data);
 *   width is the viewBox width (default 800, 560 for a chart in a .pair)
 */
function scatter(el, points, opts) {
  const W = opts.width ?? 800, H = W * 0.55, m = { l: 56, r: 12, t: 10, b: 40 };
  const pts = points.filter((p) => p.x != null && p.y != null);
  const xMax = opts.max ?? niceMax(Math.max(...pts.map((p) => p.x), 1e-9));
  const yMax = opts.max ?? niceMax(Math.max(...pts.map((p) => p.y), 1e-9));
  const xMin = opts.line ? Math.min(...pts.map((p) => p.x), 0) : 0;
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const x = (v) => m.l + ((v - xMin) / (xMax - xMin || 1)) * pw;
  const y = (v) => m.t + ph - (v / yMax) * ph;
  const kids = [];
  yTicks(kids, y, 0, yMax, m, W);
  for (let k = 0; k <= 4; k++) {
    const v = xMin + ((xMax - xMin) * k) / 4;
    kids.push(sv("text", { x: x(v), y: H - 22, "text-anchor": "middle" }, short(v)));
  }
  kids.push(
    sv("text", { x: m.l + pw / 2, y: H - 4, "text-anchor": "middle" }, opts.xLabel),
    sv("text", { transform: `translate(12, ${m.t + ph / 2}) rotate(-90)`, "text-anchor": "middle" }, opts.yLabel));
  if (opts.diagonal) {
    const top = Math.min(xMax, yMax);
    kids.push(sv("line", { class: "diagonal", x1: x(0), y1: y(0), x2: x(top), y2: y(top) }));
  }
  if (opts.line && pts.length > 1) {
    const sorted = [...pts].sort((a, b) => a.x - b.x);
    kids.push(sv("polyline", {
      points: sorted.map((p) => `${x(p.x)},${y(p.y)}`).join(" "),
      fill: "none", style: "stroke:var(--axis); stroke-width:1.5",
    }));
  }
  for (const p of pts) {
    const dot = sv("circle", {
      class: `dot${opts.onClick ? " clickable" : ""}${p.selected ? " selected" : ""}`,
      cx: x(p.x), cy: y(p.y), r: p.selected ? 6 : 4,
      style: `fill:${p.selected ? "var(--series-4)" : "var(--accent)"}; fill-opacity:0.75`,
    });
    dot.onmousemove = (ev) => tip(ev, p.tip);
    dot.onmouseleave = () => tip(null);
    if (opts.onClick) dot.onclick = () => opts.onClick(p);
    kids.push(dot);
  }
  el.replaceChildren(sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, kids));
}

/**
 * Dots by category along x, jittered; an optional mark per category drawn as a short bar.
 * @param {HTMLElement} el container, replaced
 * @param {string[]} groups categories, in x order
 * @param {{group: string, y: number, tip: string, r?: number}[]} points
 * @param {{yLabel: string, max?: number, marks?: Object<string, number>}} opts
 */
function strip(el, groups, points, opts) {
  const W = 900, H = 300, m = { l: 56, r: 12, t: 10, b: 60 };
  const pts = points.filter((p) => p.y != null);
  const marks = Object.values(opts.marks ?? {}).filter((v) => v != null);
  const lo = Math.min(0, ...pts.map((p) => p.y), ...marks);
  const hi = opts.max ?? niceMax(Math.max(...pts.map((p) => p.y), ...marks, 1e-9));
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const bw = pw / Math.max(groups.length, 1);
  const y = (v) => m.t + ph - ((v - lo) / (hi - lo || 1)) * ph;
  const kids = [];
  yTicks(kids, y, lo, hi, m, W);
  kids.push(sv("text", { transform: `translate(12, ${m.t + ph / 2}) rotate(-90)`, "text-anchor": "middle" }, opts.yLabel));
  groups.forEach((g, i) => {
    kids.push(sv("text", { transform: `translate(${m.l + i * bw + bw / 2}, ${H - m.b + 14}) rotate(30)`, "text-anchor": "start" }, g));
    const mark = opts.marks?.[g];
    if (mark != null) {
      const bar = sv("line", { x1: m.l + i * bw + bw * 0.2, x2: m.l + i * bw + bw * 0.8, y1: y(mark), y2: y(mark), style: "stroke:var(--series-4); stroke-width:2.5" });
      bar.onmousemove = (ev) => tip(ev, `class ${g}: ${short(mark)}`);
      bar.onmouseleave = () => tip(null);
      kids.push(bar);
    }
  });
  let seed = 1;
  const jitter = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; };
  for (const p of pts) {
    const i = groups.indexOf(p.group);
    const dot = sv("circle", {
      class: "dot", cx: m.l + i * bw + bw / 2 + jitter() * bw * 0.5, cy: y(p.y), r: p.r ?? 3.5,
      style: "fill:var(--accent); fill-opacity:0.6",
    });
    dot.onmousemove = (ev) => tip(ev, p.tip);
    dot.onmouseleave = () => tip(null);
    kids.push(dot);
  }
  const legend = opts.marks ? h("div", { class: "legend" },
    h("span", {}, h("i", { style: "background:var(--accent)" }), "region"),
    h("span", {}, h("i", { style: "background:var(--series-4)" }), "whole class")) : null;
  el.replaceChildren(...[legend, sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, kids)].filter(Boolean));
}

/**
 * Bars grouped by metric, one bar per series, with a legend.
 * @param {HTMLElement} el container, replaced
 * @param {string[]} metrics group labels
 * @param {{name: string, color: string, values: (number|null)[]}[]} series one value per metric
 */
function groupedBars(el, metrics, series) {
  const W = 800, H = 240, m = { l: 52, r: 8, t: 8, b: 26 };
  const values = series.flatMap((s) => s.values).filter((v) => v != null);
  const lo = Math.min(0, ...values), hi = Math.max(1, ...values);
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const gw = pw / metrics.length, slot = (gw * 0.7) / series.length;
  const y = (v) => m.t + ph - ((v - lo) / (hi - lo)) * ph;
  const kids = [];
  yTicks(kids, y, lo, hi, m, W);
  kids.push(sv("line", { class: "axis", x1: m.l, x2: W - m.r, y1: y(0), y2: y(0) }));
  metrics.forEach((metric, i) => {
    kids.push(sv("text", { x: m.l + i * gw + gw / 2, y: H - 8, "text-anchor": "middle" }, metric));
    series.forEach((s, j) => {
      const v = s.values[i];
      if (v == null) return;
      const bar = sv("rect", {
        x: m.l + i * gw + gw * 0.15 + j * slot, y: Math.min(y(v), y(0)), width: slot - 3,
        height: Math.abs(y(v) - y(0)), rx: 2, style: `fill:${s.color}`,
      });
      bar.onmousemove = (ev) => tip(ev, `${s.name} ${metric}: ${dec(v)}`);
      bar.onmouseleave = () => tip(null);
      kids.push(bar);
    });
  });
  el.replaceChildren(
    h("div", { class: "legend" }, series.map((s) => h("span", {}, h("i", { style: `background:${s.color}` }), s.name))),
    sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, kids));
}

/**
 * Waterfall from the base: one bar per contribution, the rest summed, then the total.
 * @param {HTMLElement} el container, replaced
 * @param {number} base expected value of the forest
 * @param {{descriptor: string, value: number|null, contribution: number}[]} shown largest first
 * @param {{contribution: number}[]} rest summed into one bar
 */
function waterfall(el, base, shown, rest) {
  const restSum = rest.reduce((a, c) => a + c.contribution, 0);
  const steps = shown.map((c) => ({ label: c.descriptor, delta: c.contribution, value: c.value }));
  if (rest.length) steps.push({ label: `${rest.length} other descriptors`, delta: restSum });
  const W = 800, labelW = 260, rowH = 22, m = { t: 8, b: 26 };
  let level = base;
  const spans = steps.map((s) => { const from = level; level += s.delta; return { ...s, from, to: level }; });
  const all = [base, level, ...spans.flatMap((s) => [s.from, s.to])];
  const lo = Math.min(...all), hi = Math.max(...all);
  const pw = W - labelW - 70;
  const x = (v) => labelW + ((v - lo) / (hi - lo || 1)) * pw;
  const H = m.t + (spans.length + 2) * rowH + m.b;
  const kids = [];
  const row = (i, label, from, to, color, text) => {
    const yy = m.t + i * rowH;
    kids.push(
      sv("text", { x: labelW - 10, y: yy + rowH / 2 + 4, "text-anchor": "end", style: "fill:var(--text-2)" }, label),
      sv("rect", { x: Math.min(x(from), x(to)), y: yy + 5, width: Math.max(1.5, Math.abs(x(to) - x(from))), height: rowH - 10, rx: 2, style: `fill:${color}` }),
      sv("text", { x: Math.max(x(from), x(to)) + 6, y: yy + rowH / 2 + 4 }, text));
  };
  row(0, "base", lo, base, "var(--axis)", dec(base));
  spans.forEach((s, i) => row(i + 1, s.label, s.from, s.to, s.delta >= 0 ? "var(--series-2)" : "var(--accent)",
    `${s.delta >= 0 ? "+" : ""}${dec(s.delta, 4)}${s.value != null ? ` (value ${short(s.value)})` : ""}`));
  row(spans.length + 1, "estimate before clip", lo, level, "var(--series-4)", dec(level));
  kids.push(sv("line", { class: "diagonal", x1: x(base), x2: x(base), y1: m.t, y2: H - m.b }));
  el.replaceChildren(
    h("div", { class: "legend" },
      h("span", {}, h("i", { style: "background:var(--series-2)" }), "raises the estimate"),
      h("span", {}, h("i", { style: "background:var(--accent)" }), "lowers it")),
    sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, kids));
}

/** Show #tooltip near the pointer with the given text, or hide it when text is null. @param {MouseEvent} ev @param {string|null} text */
function tip(ev, text) {
  const el = $("tooltip");
  el.hidden = text == null;
  if (text == null) return;
  el.textContent = text;
  const x = Math.min(ev.clientX + 14, window.innerWidth - el.offsetWidth - 8);
  el.style.left = `${Math.max(8, x)}px`;
  el.style.top = `${ev.clientY + 14}px`;
}

init().catch((e) => {
  for (const p of document.querySelectorAll(".panel")) {
    p.classList.add("failed");
    p.querySelector(".empty").hidden = false;
    p.querySelector(".empty").textContent = e.message;
  }
});
