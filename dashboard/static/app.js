// Dashboard front-end: one page, one tab per stage. Charts are hand-made SVG, no dependencies.

/**
 * UI state.
 * @type {{dataset: string|null, tab: "clean"|"regions"|"complexity", cls: string|null, measure: string|null}}
 */
const state = { dataset: null, tab: "clean", cls: null, measure: null };

const last = { regions: null }; // last responses, for the controls that only redraw
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
      const tr = h("tr", {}, columns.map((c) => h("td", {}, (c.format ?? String)(r[c.key], r))));
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
 * @param {string} path e.g. "/synthetic_test/clean"
 * @returns {Promise<object>} parsed body; rejects with the server "detail" on 404 (shown as the empty state)
 */
async function api(path) {
  const res = await fetch(`/api${path}`);
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error(typeof body?.detail === "string" ? body.detail : `request failed (${res.status})`);
  return body;
}

/** Fill the dataset selector from /api/options, restore tab/dataset from the URL hash, wire the controls, load the tab. */
async function init() {
  const opts = await api("/options");
  $("dataset").replaceChildren(...opts.datasets.map((v) => h("option", {}, v)));
  const [tab, dataset] = location.hash.slice(1).split("/").map(decodeURIComponent);
  if ([...document.querySelectorAll(".tabs button")].some((b) => b.dataset.tab === tab)) state.tab = tab;
  state.dataset = opts.datasets.includes(dataset) ? dataset : opts.datasets[0];
  $("dataset").value = state.dataset;
  $("dataset").onchange = (e) => { state.dataset = e.target.value; state.cls = state.measure = null; load(); };
  $("rg-class").onchange = (e) => { state.cls = e.target.value; if (last.regions) drawCandidates(last.regions); };
  $("cx-measure").onchange = (e) => { state.measure = e.target.value; load(); };
  for (const b of document.querySelectorAll(".tabs button")) b.onclick = () => showTab(b.dataset.tab);
  showTab(state.tab);
}

/** Show a tab, hide the others. */
function showTab(tab) {
  state.tab = tab;
  for (const b of document.querySelectorAll(".tabs button")) b.setAttribute("aria-selected", b.dataset.tab === tab);
  for (const p of document.querySelectorAll(".panel")) p.hidden = p.id !== `tab-${tab}`;
  load();
}

/** Fetch the data of state.tab and render it; on error show the message in the tab's empty state. */
async function load() {
  const id = ++loadId;
  const tab = state.tab;
  history.replaceState(null, "", `#${[tab, state.dataset].map(encodeURIComponent).join("/")}`);
  const panel = $(`tab-${tab}`);
  const empty = panel.querySelector(".empty");
  const d = `/${encodeURIComponent(state.dataset)}`;
  const path = {
    clean: `${d}/clean`,
    regions: `${d}/regions`,
    complexity: `${d}/complexity${state.measure ? `?measure=${encodeURIComponent(state.measure)}` : ""}`,
  }[tab];
  try {
    const data = await api(path);
    if (id !== loadId) return;
    panel.classList.remove("failed");
    empty.hidden = true;
    ({ clean: renderClean, regions: renderRegions, complexity: renderComplexity })[tab](data);
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

/** Tiles, classes with inline bars, dropped classes and columns, features. @param {object} data response of /clean */
function renderClean(data) {
  const s = data.summary;
  fill("sp-tiles",
    tile("rows read", num(s.n_raw)),
    tile("duplicates dropped", num(s.n_duplicate), pct(s.n_duplicate / s.n_raw)),
    tile("rows kept", num(s.n_kept), `${num(s.n_rare)} in rare classes`),
    tile("classes", num(s.n_classes), `${data.dropped_classes.length} dropped`),
    tile("constant columns", num(s.n_constant)));
  const top = Math.max(...data.classes.map((c) => c.n_rows), 1);
  fill("sp-classes", table(
    ["class", "rows", "share", "read", "duplicates"],
    data.classes.map((c) => [
      c.label,
      h("span", {}, h("i", { class: "bar", style: `width:${Math.round((c.n_rows / top) * 60)}px` }), num(c.n_rows)),
      pct(c.n_rows / s.n_kept),
      num(c.n_read),
      num(c.n_read - c.n_rows),
    ])));
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
    tile("graph nodes", num(s.n_graph_nodes)));
  sortable($("rg-classes"), [
    { key: "class", label: "class" },
    { key: "n_rows", label: "rows", format: num },
    { key: "n_regions", label: "regions", format: num },
    { key: "loss", label: "loss", format: (v) => dec(v, 4) },
    { key: "params", label: "chosen", format: params },
  ], data.classes, { onClick: (r) => { state.cls = r.class; $("rg-class").value = r.class; drawCandidates(data); } });
  const classes = data.classes.map((c) => c.class);
  if (!classes.includes(state.cls)) state.cls = classes[0];
  $("rg-class").replaceChildren(...classes.map((c) => h("option", {}, c)));
  $("rg-class").value = state.cls;
  drawCandidates(data);
  const sizes = data.regions.map((r) => r.n_rows);
  const edges = Array.from({ length: 21 }, (_, i) => (Math.max(...sizes, 1) * i) / 20);
  histogram($("rg-sizes"), edges, [{
    name: "regions by rows", color: "var(--accent)",
    values: edges.slice(0, -1).map((lo, i) => sizes.filter((v) => v >= lo && (v < edges[i + 1] || i === 19)).length),
  }], { format: num });
  sortable($("rg-table"), [
    { key: "region", label: "region" },
    { key: "class", label: "class" },
    { key: "n_rows", label: "rows", format: num },
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

// --- charts ---

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
 * Scatter of points with axes from 0; optionally a y = x diagonal, a joining line and clickable dots.
 * @param {HTMLElement} el container, replaced
 * @param {{x: number, y: number, tip: string, id?: any, selected?: boolean}[]} points
 * @param {{xLabel: string, yLabel: string, diagonal?: boolean, line?: boolean, max?: number,
 *   onClick?: (p: object) => void, width?: number}} opts max fixes both axes (else each from its data);
 *   width is the viewBox width (default 800)
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
