// Dashboard front-end: one page, three tabs (quality, cells, data). Charts are hand-made SVG, no dependencies.

/**
 * UI state.
 * @type {{dataset: string|null, tab: "quality"|"cells"|"data", cls: string|null, node: number|null,
 *   colorBy: string|null, qclass: string|null, qfilter: string|null, region: number|null, tau: number|null}}
 *   cls, node: the class whose tree is drawn and the node whose map is shown (null: its root), the tree drawn from the
 *   level that shows it (treeLevels); colorBy: what colours the tree and the map, null their subtrees, "any" any
 *   defect, else one of DEFECTS; qclass, qfilter: the regions listed, picked in the matrix (null: all), qfilter "none"
 *   or one of DEFECTS; region: the region whose detail is shown (null: none); tau: the threshold the defects of the
 *   quality and cells tabs are judged at (null: the run's own)
 */
const state = {
  dataset: null, tab: "quality", cls: null, node: null, colorBy: null,
  qclass: null, qfilter: null, region: null, tau: null,
};

const last = { cells: null, cellInfo: null, quality: null }; // last responses, for the controls that only redraw
let loadId = 0; // drops answers of superseded requests

const $ = (id) => document.getElementById(id);
const NS = { html: "http://www.w3.org/1999/xhtml", svg: "http://www.w3.org/2000/svg" };

// --- formatting ---

const pct = (v, d = 1) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);
// a share that is not rounded to 0: one decimal, one significant digit under 0.1%
const pctSmall = (v) => (v > 0 && v < 0.001 ? `${Number((v * 100).toPrecision(1))}%` : pct(v));
const num = (v) => (v == null ? "–" : Number(v).toLocaleString("en-US"));
const dec = (v, d = 3) => (v == null ? "–" : Number(v).toFixed(d));
const short = (v) => (v == null ? "–" : String(Number(Number(v).toPrecision(3))));
const when = (iso) => (iso ? `${iso.replace("T", " ").slice(0, 16)} UTC` : "–");
const plural = (n, one, many = `${one}s`) => `${num(n)} ${n === 1 ? one : many}`;
// the first names of a list, then how many more
const some = (names, k = 3) => names.slice(0, k).join(", ") + (names.length > k ? ` and ${names.length - k} more` : "");

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
 * @param {{key: string, label: string, format?: (v: any, row: object) => any, help?: string, wrap?: boolean}[]} columns
 *   help: the header's tooltip; wrap: the cells may wrap; text: left-aligned, as text
 * @param {object[]} rows one object per row, keyed by column key
 * @param {{onClick?: (row: object) => void, selected?: (row: object) => boolean, empty?: string, by?: string}} [opts]
 *   empty: the note shown instead of a table without rows; by: the key sorted on first (default: none, rows in order)
 */
function sortable(el, columns, rows, opts = {}) {
  if (!rows.length) { el.replaceChildren(h("p", { class: "note" }, opts.empty ?? "Nothing to show.")); return; }
  let by = opts.by ?? null, dir = 1;
  const draw = () => {
    const sorted = by == null ? rows : [...rows].sort((a, b) => {
      const x = a[by], y = b[by];
      if (x == null) return 1;
      if (y == null) return -1;
      return (typeof x === "number" ? y - x : String(x).localeCompare(String(y))) * dir;
    });
    const head = columns.map((c) => {
      const th = h("th", { class: `sortable${c.text || c.wrap ? " text" : ""}`, ...(c.help ? { title: c.help } : {}) }, c.label);
      if (c.key === by) {
        const descending = typeof rows[0]?.[by] === "number" ? dir > 0 : dir < 0;
        th.setAttribute("aria-sort", descending ? "descending" : "ascending");
      }
      th.onclick = () => { dir = by === c.key ? -dir : 1; by = c.key; draw(); };
      return th;
    });
    const body = sorted.map((r) => {
      const kind = (c) => [c.wrap && "wrap", c.text && "text"].filter(Boolean).join(" ");
      const tr = h("tr", {}, columns.map((c) => h("td", kind(c) ? { class: kind(c) } : {}, (c.format ?? String)(r[c.key], r))));
      if (opts.onClick) { tr.classList.add("clickable"); tr.onclick = () => opts.onClick(r); }
      if (opts.selected?.(r)) tr.classList.add("selected");
      return tr;
    });
    el.replaceChildren(h("table", {}, h("thead", {}, h("tr", {}, head)), h("tbody", {}, body)));
  };
  draw();
}

/** A sortable table in a new container. @returns {HTMLElement} */
function sortableIn(columns, rows, opts) {
  const el = h("div", { class: "table-wrap" });
  sortable(el, columns, rows, opts);
  return el;
}

function tile(label, value, sub) {
  return h("div", { class: "tile" },
    h("div", { class: "label" }, label),
    h("div", { class: "value" }, value),
    sub ? h("div", { class: "sub" }, sub) : null);
}

/** The hyperparameters of a run as a table: their names, then the values. @param {Object<string, any>} params */
function paramTable(params) {
  return table(Object.keys(params), [Object.values(params).map(String)]);
}

function fill(id, ...kids) {
  $(id).replaceChildren(...kids.flat().filter((k) => k != null));
}

// --- data ---

/** The tau of the request, as a query string part (null tau: none). @param {"?"|"&"} [first] */
const tauQuery = (first = "?") => (state.tau == null ? "" : `${first}tau=${state.tau}`);

/**
 * GET a JSON endpoint under /api.
 * @param {string} path e.g. "/cic_2018/clean"
 * @returns {Promise<object>} parsed body; rejects with the server "detail" on 404 (shown as the empty state)
 */
async function api(path) {
  const res = await fetch(`/api${path}`);
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error(typeof body?.detail === "string" ? body.detail : `request failed (${res.status})`);
  return body;
}

const TABS = ["quality", "cells", "data"];
const OLD_TABS = { clean: "data", prepare: "data" }; // links written before data held both stages

/** Fill the dataset selector from /api/options, restore tab/dataset from the URL hash, wire the controls, load the tab. */
async function init() {
  const opts = await api("/options");
  $("dataset").replaceChildren(...opts.datasets.map((v) => h("option", {}, v)));
  readHash(opts.datasets);
  // back, forward and a pasted link change only the hash: draw what it names
  window.onhashchange = async () => {
    if (location.hash === shownHash) return;
    readHash(opts.datasets);
    await showTab(state.tab);
    if (state.tab === "quality" && state.region != null) scrollUnderHeader($("qr-detail"));
  };
  $("dataset").onchange = (e) => {
    state.dataset = e.target.value;
    state.cls = state.node = state.colorBy = state.qclass = state.qfilter = state.region = state.tau = null;
    load();
  };
  $("qr-tau").oninput = (e) => { $("qr-tau-value").textContent = e.target.value; };
  $("qr-tau").onchange = (e) => setTau(Number(e.target.value));
  $("qr-tau-reset").onclick = () => setTau(null);
  $("rg-class").onchange = (e) => selectClass(e.target.value);
  $("rg-color").onchange = (e) => { state.colorBy = e.target.value || null; drawClass(); };
  for (const b of document.querySelectorAll(".tabs button")) b.onclick = () => showTab(b.dataset.tab);
  showTab(state.tab);
}

/**
 * Judge the defects at another tau and draw the tab again; the run's own tau is null, never a number.
 * @param {number|null} tau
 */
function setTau(tau) {
  state.tau = tau === last.quality.params.tau ? null : tau;
  load();
}

let shownHash = ""; // the hash last written by saveHash: a hashchange to it is our own

/**
 * Set state from the URL hash: tab, dataset, the quality tab's region or the cells tab's class and node, and tau
 * ("?tau=0.15" after them).
 * @param {string[]} datasets the known datasets; an unknown one falls back to the first
 */
function readHash(datasets) {
  const [path, query] = location.hash.slice(1).split("?");
  const [named, dataset, a, b] = path.split("/").map(decodeURIComponent);
  state.tau = Number(new URLSearchParams(query).get("tau")) || null;
  const tab = OLD_TABS[named] ?? named;
  if (TABS.includes(tab)) state.tab = tab;
  const next = datasets.includes(dataset) ? dataset : datasets[0];
  if (next !== state.dataset) state.qclass = state.qfilter = state.colorBy = null;
  state.dataset = next;
  if (tab === "quality") {
    const region = ["classes", "regions"].includes(a) ? b : a; // older links name a panel first
    state.region = region ? Number(region) : null;
  }
  if (tab === "cells") {
    state.cls = a || null;
    state.node = b ? Number(b) : null;
  }
  $("dataset").value = state.dataset;
}

/** Show a tab, hide the others. @returns {Promise<void>} once it is drawn */
function showTab(tab) {
  state.tab = tab;
  for (const b of document.querySelectorAll(".tabs button")) b.setAttribute("aria-selected", b.dataset.tab === tab);
  for (const p of document.querySelectorAll(".panel")) p.hidden = p.id !== `tab-${tab}`;
  return load();
}

/**
 * Keep tab, dataset and, on the quality tab the region, on the cells tab class and node, and tau, in the URL hash.
 * @param {boolean} [push] a step the back button returns from (opening a region or a node); else the entry is replaced
 */
function saveHash(push = false) {
  const parts = [state.tab, state.dataset];
  if (state.tab === "quality" && state.region != null) parts.push(state.region);
  if (state.tab === "cells" && state.cls != null) parts.push(state.cls, ...(state.node != null ? [state.node] : []));
  shownHash = `#${parts.map((p) => encodeURIComponent(p)).join("/")}${tauQuery()}`;
  if (shownHash === location.hash) return;
  if (push) history.pushState(null, "", shownHash);
  else history.replaceState(null, "", shownHash);
}

/**
 * The data of a tab: its stage's answer; data holds clean's and, under prepare, prepare's or {error} when it has not run.
 * @param {string} tab
 * @param {string} d the dataset's path, "/<dataset>"
 */
async function fetchTab(tab, d) {
  if (tab !== "data") return api(`${d}/${tab}${tauQuery()}`);
  const [clean, prepare] = await Promise.all([
    api(`${d}/clean`), api(`${d}/prepare`).catch((e) => ({ error: e.message }))]);
  return { ...clean, prepare };
}

/** Fetch the data of state.tab and render it; on error show the message in the tab's empty state. */
async function load() {
  const id = ++loadId;
  const tab = state.tab;
  saveHash();
  const panel = $(`tab-${tab}`);
  const empty = panel.querySelector(".empty");
  try {
    const data = await fetchTab(tab, `/${encodeURIComponent(state.dataset)}`);
    if (id !== loadId) return;
    panel.classList.remove("failed");
    empty.hidden = true;
    ({ quality: renderQuality, cells: renderCells, data: renderData })[tab](data);
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

/**
 * Hyperparameters, tiles, classes with inline bars, what was removed, provenance.
 * @param {object} data response of /clean, with prepare: the response of /prepare or {error}
 */
function renderData(data) {
  const s = data.summary;
  fill("sp-params", paramTable({ ...data.params, ...(data.prepare.params ?? {}) }));
  const features = tile("features", `${num(data.features.num.length)} + ${num(data.features.cat.length)}`,
    "numeric + categorical");
  features.title = `numeric: ${data.features.num.join(", ")}\ncategorical: ${data.features.cat.join(", ") || "none"}`;
  fill("sp-tiles",
    tile("CSV rows read", num(s.n_raw)),
    tile("copies", num(s.n_duplicate), `${pct(s.n_duplicate / s.n_raw)} of the CSV rows`),
    tile("distinct rows kept", num(s.n_kept),
      `${num(s.n_rare)} in rare classes` + (s.n_capped ? `, ${num(s.n_capped)} past max_class_rows` : "")),
    tile("classes", num(s.n_classes), `${data.dropped_classes.length} dropped`),
    features);
  const top = Math.max(...data.classes.map((c) => c.n_rows), 1);
  const capped = data.classes.some((c) => (c.n_distinct ?? c.n_rows) > c.n_rows);
  fill("sp-classes", table(
    ["class", "distinct rows", "share", "CSV rows", "copies", ...(capped ? ["sampled out"] : [])],
    data.classes.map((c) => [
      c.label,
      h("span", {}, h("i", { class: "bar", style: `width:${Math.round((c.n_rows / top) * 60)}px` }), num(c.n_rows)),
      pct(c.n_rows / s.n_kept),
      num(c.n_read),
      num(c.n_read - (c.n_distinct ?? c.n_rows)),
      ...(capped ? [num((c.n_distinct ?? c.n_rows) - c.n_rows)] : []),
    ])));
  fill("sp-dropped", data.dropped_classes.map((c) => h("span", { class: "chip" }, `${c.label} (${num(c.n_rows)})`)));
  fill("sp-constant", data.dropped_columns.map((c) => h("span", { class: "chip" }, c)));
  sortable($("sp-provenance"), [
    { key: "key", label: "key", text: true },
    { key: "column", label: "column", text: true },
    { key: "n_read", label: "CSV rows", format: num, help: "rows of the CSV of the kept classes, copies included" },
    { key: "n_empty", label: "empty", format: (v) => flagged(num(v), v > 0) },
    { key: "n_unparsable", label: "not a time", format: (v, r) => (r.key === "time" ? flagged(num(v), v > 0) : "–"),
      help: "times that time_format does not read: unknown, as the empty ones" },
    { key: "n_values", label: "distinct values", format: num },
  ], data.provenance, { empty: "The dataset names no provenance columns (a provenance block in its yaml)." });
}

/**
 * Hyperparameters, leaves per class, and the tree and map of a class. The cells of stage cells are the leaves of
 * their class's tree here: the quality tab calls cell what it merged them into.
 * @param {object} data response of /cells
 */
function renderCells(data) {
  last.cells = data;
  last.cellInfo = new Map(data.cells.map((c) => [c.cell, c]));
  fill("rg-params", paramTable(data.params));
  const sizes = new Map();
  for (const c of data.cells) sizes.set(c.class, [...(sizes.get(c.class) ?? []), c.n_rows]);
  const rows = data.classes.map((c) => {
    const v = (sizes.get(c.class) ?? []).sort((a, b) => a - b);
    return { ...c, median: v[Math.floor((v.length - 1) / 2)], min: v[0], max: v.at(-1) };
  });
  sortable($("rg-classes"), [
    { key: "class", label: "class", text: true },
    { key: "n_read", label: "CSV rows", format: (v, r) => h("span", { title: `${num(r.n_rows)} distinct rows` }, num(v)),
      help: HELP.read },
    { key: "n_cells", label: "leaves", format: num },
    { key: "median", label: "rows per leaf, median", format: num,
      help: "the distinct rows of its leaves (lower median): from min_rows (the rows that decide a share τ away " +
        "from τ or one half) to under twice it" },
    { key: "max", label: "least – most", format: (v, r) => `${num(r.min)} – ${num(v)}`,
      help: "the fewest and the most distinct rows of one of its leaves" },
  ], rows, { onClick: (r) => selectClass(r.class), selected: (r) => r.class === state.cls, by: "n_read" });
  const classes = data.classes.map((c) => c.class);
  if (!classes.includes(state.cls)) { state.cls = classes[0]; state.node = null; }
  $("rg-class").replaceChildren(...data.classes.map((c) =>
    h("option", { value: c.class }, `${c.class} · ${plural(c.n_cells, "leaf", "leaves")}`)));
  colorChoices(data.cells);
  drawClass();
}

/**
 * The options of "colour by": the subtrees, then, when quality_report judged the cells, any defect and each defect
 * some cell has, with its cells; hidden without a judgement. A colouring no longer offered falls back to subtrees.
 * @param {object[]} cells the cells of /cells
 */
function colorChoices(cells) {
  const judged = cells.some((c) => c.defects != null);
  const count = (d) => cells.filter((c) => hasDefect(c.cell, d)).length;
  const offered = judged ? ["any", ...DEFECTS].filter((d) => count(d)) : [];
  if (!offered.includes(state.colorBy)) state.colorBy = null;
  $("rg-color").replaceChildren(h("option", { value: "" }, "subtree"), ...offered.map((d) =>
    h("option", { value: d }, `${d === "any" ? "any defect" : d} · ${plural(count(d), "leaf", "leaves")}`)));
  $("rg-color").value = state.colorBy ?? "";
  $("rg-color").parentElement.hidden = !judged;
}

/** Whether quality_report gave a cell a defect: d one of DEFECTS, or "any". */
function hasDefect(cell, d) {
  const found = last.cellInfo.get(cell)?.defects ?? [];
  return d === "any" ? found.length > 0 : found.includes(d);
}

/** Show the tree of a class, mapped from its root. @param {string} cls */
function selectClass(cls) {
  state.cls = cls;
  state.node = null;
  drawClass();
}

/**
 * Show a node of the cell trees: its class becomes state.cls.
 * @param {number} node
 * @param {boolean} [push] a click: the back button returns to the node before
 */
function selectNode(node, push = false) {
  state.cls = last.cells.tree.find((n) => n.node === node).class;
  state.node = node;
  drawClass(push);
}

/**
 * Tree and map of state.cls; state.node falls back to the root of the class. The tree is drawn from the last of its
 * levels down to state.node, with the levels above it as a path; coloured by state.colorBy.
 * @param {boolean} [push] passed to saveHash
 */
function drawClass(push = false) {
  const t = classTree(last.cells.tree.filter((n) => n.class === state.cls));
  if (!t.byId.has(state.node)) state.node = t.root;
  $("rg-class").value = state.cls;
  saveHash(push);
  const levels = treeLevels(t, state.node);
  const view = treeView(t, levels.at(-1));
  const colors = state.colorBy ? defectColor(t, state.colorBy) : leafColor(t, state.node, view.open);
  drawPath(levels, t);
  treeDiagram($("rg-tree"), t, view, colors);
  drawRegionLink(t.byId.get(state.node));
  drawMap(t.nodes, colors);
}

/**
 * Under the map of a leaf, the cell of the quality tab it is part of, with a link to it; hidden for other nodes, or
 * when quality_report has not judged the leaves.
 * @param {object} n the cell_tree row of state.node
 */
function drawRegionLink(n) {
  const info = n.cell != null ? last.cellInfo.get(n.cell) : null;
  $("rg-region").hidden = info?.region == null;
  if (info?.region == null) return;
  const open = h("button", { class: "chip", type: "button" }, `open cell ${info.region} in quality`);
  open.onclick = () => {
    state.qclass = state.qfilter = null;
    location.hash = `#quality/${encodeURIComponent(state.dataset)}/${info.region}${tauQuery()}`;
  };
  fill("rg-region", `Leaf ${n.cell} is part of cell ${info.region}.`, open);
}

// categorical slots, in fixed order; validated on the dark surface: every adjacent pair passes (worst CVD ΔE 8.4),
// not every pair (orange and red, violet and blue lie close), so legends and tooltips always name the class
const SERIES = ["var(--accent)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)",
  "var(--series-6)", "var(--series-7)", "var(--series-8)"];

const TREE_NODES = 63; // most nodes a drawing of a tree holds; the nodes left closed are drawn on their own when opened

/**
 * One class tree, indexed.
 * @param {object[]} nodes the cell_tree rows of one class, in pre-order (node ids rise in pre-order)
 * @returns {{nodes: object[], kids: Map<number, number[]>, byId: Map<number, object>, cells: Map<number, number>,
 *   first: Map<number, number>, root: number}} kids: the children of every inner node, in pre-order; cells: the cells
 *   below every node (1 for a cell); first: its first cell in pre-order
 */
function classTree(nodes) {
  const kids = new Map();
  for (const n of nodes) if (n.parent >= 0) kids.set(n.parent, [...(kids.get(n.parent) ?? []), n.node]);
  const cells = new Map(), first = new Map();
  for (const n of [...nodes].reverse()) {
    const below = kids.get(n.node) ?? [];
    cells.set(n.node, below.length ? below.reduce((a, k) => a + cells.get(k), 0) : 1);
    first.set(n.node, below.length ? first.get(below[0]) : n.cell);
  }
  const byId = new Map(nodes.map((n) => [n.node, n]));
  return { nodes, kids, byId, cells, first, root: nodes.find((n) => n.parent < 0).node };
}

/**
 * Open a subtree from its top, the node with the most rows first (ties in pre-order), as the cut splits: a node
 * opened is replaced by its children, while the result fits.
 * @param {{kids: Map<number, number[]>, byId: Map<number, object>}} t from classTree
 * @param {number} top
 * @param {(leaves: number, opened: number) => boolean} fits whether so many leaves and nodes opened fit
 * @param {(node: number) => boolean} [openable] whether a node may be opened (default: it has children)
 * @returns {{leaves: number[], open: Set<number>}} the leaves, in pre-order, and the nodes opened
 */
function openLargest(t, top, fits, openable = (v) => t.kids.has(v)) {
  let leaves = [top];
  const open = new Set();
  for (;;) {
    const next = leaves.filter(openable).sort((a, b) => t.byId.get(b).n_rows - t.byId.get(a).n_rows || a - b)[0];
    if (next == null || !fits(leaves.length - 1 + t.kids.get(next).length, open.size + 1)) break;
    open.add(next);
    leaves = leaves.flatMap((v) => (v === next ? t.kids.get(v) : [v]));
  }
  return { leaves, open };
}

/**
 * The drawing of a class tree from a node: at most TREE_NODES nodes, the largest opened first. A leaf of the drawing
 * that has children is closed: a subtree of its own.
 * @returns {{top: number, leaves: number[], open: Set<number>}}
 */
function treeView(t, top) {
  return { top, ...openLargest(t, top, (leaves, opened) => leaves + opened <= TREE_NODES) };
}

/**
 * The levels of a class tree down to a node: the root, then the closed node of each level's drawing on the way to the
 * node, until the node is drawn and not closed; a closed node selected is opened, the last level.
 * @returns {number[]} the top of every level, the last the one drawn
 */
function treeLevels(t, node) {
  const path = new Set([node]);
  for (let n = t.byId.get(node); n.parent >= 0; n = t.byId.get(n.parent)) path.add(n.parent);
  const levels = [t.root];
  for (;;) {
    const top = levels.at(-1);
    if (top === node) return levels;
    const { leaves, open } = treeView(t, top);
    const below = leaves.find((v) => path.has(v)); // the leaf of the drawing over the node, when not opened
    if (open.has(node) || (below === node && !t.kids.has(node))) return levels;
    levels.push(below);
  }
}

/**
 * The colour groups of the selected node: its leaves when they fit the palette, else the fewest subtrees below it that
 * do, splitting the largest first, never past the leaves drawn. Groups keep pre-order and take the palette in that
 * fixed order.
 * @param {object} t from classTree
 * @param {number} selected the node whose rows are mapped
 * @param {Set<number>} open the nodes opened in the drawing (treeView)
 * @returns {{groups: {node: number, color: string}[], byCell: Map<number, {node: number, color: string}>}}
 */
function leafColor(t, selected, open) {
  const groups = openLargest(t, selected, (leaves) => leaves <= SERIES.length, (v) => open.has(v)).leaves;
  const colored = groups.map((node, i) => ({ node, color: SERIES[i] }));
  const byCell = new Map();
  for (const g of colored) for (const c of cellsBelow(t, g.node)) byCell.set(c, g);
  return { groups: colored, byCell };
}

/** The cells below a node of a class tree, itself for a cell. @param {object} t from classTree @param {number} node */
function cellsBelow(t, node) {
  const cells = [], stack = [node];
  while (stack.length) {
    const id = stack.pop();
    const n = t.byId.get(id);
    if (n.cell != null) cells.push(n.cell);
    stack.push(...(t.kids.get(id) ?? []));
  }
  return cells;
}

const WITH = "var(--warning)", WITHOUT = "var(--accent)"; // the cells with and without the defect coloured by

/**
 * The colours of a class tree by a defect: its cells with it in WITH, the others in WITHOUT; a closed node in WITH
 * when a cell below has it (with how many in its tooltip). Shaped as leafColor, with no group ringed.
 * @param {object} t from classTree
 * @param {string} d one of DEFECTS, or "any"
 * @returns {{groups: [], byCell: Map<number, {node: null, color: string}>, tintOf: (node: number) => string,
 *   tipOf: (node: number) => string, legend: {color: string, label: string}[], nameOf: (color: string) => string}}
 */
function defectColor(t, d) {
  const name = d === "any" ? "a defect" : d;
  const byCell = new Map(t.nodes.filter((n) => n.cell != null)
    .map((n) => [n.cell, { node: null, color: hasDefect(n.cell, d) ? WITH : WITHOUT }]));
  const found = (v) => cellsBelow(t, v).filter((c) => byCell.get(c).color === WITH).length;
  return {
    groups: [],
    byCell,
    tintOf: (v) => (found(v) ? WITH : WITHOUT),
    tipOf: (v) => `${num(found(v))} of ${plural(t.cells.get(v), "leaf", "leaves")} with ${name}`,
    legend: [{ color: WITH, label: `with ${name}` }, { color: WITHOUT, label: `without ${name}` }],
    nameOf: (color) => (color === WITH ? `leaves with ${name}` : `leaves without ${name}`),
  };
}

/**
 * Tooltip text of a tree node: its counts, the cell of a leaf, the cells below an inner node.
 * @param {object} n a cell_tree row
 * @param {object} [t] from classTree, for an inner node
 * @param {boolean} [closed] a closed node of the drawing
 */
function nodeTip(n, t, closed = false) {
  const what = n.cell != null ? `leaf ${n.cell}` : `node ${n.node} · ${plural(t.cells.get(n.node), "leaf", "leaves")}`;
  const info = n.cell != null ? last.cellInfo.get(n.cell) : null;
  const judged = info?.region == null ? ""
    : ` · part of cell ${info.region} · ${info.defects.length ? info.defects.join(", ") : "no defect"}`;
  return `${what} · ${num(n.n_read)} CSV rows · ${num(n.n_rows)} distinct${judged}${closed ? " · click to open" : ""}`;
}

/**
 * The levels above the drawing, each a button that draws it, then the one drawn; hidden for a tree of one level.
 * @param {number[]} levels from treeLevels
 * @param {object} t from classTree
 */
function drawPath(levels, t) {
  const el = $("rg-path");
  el.hidden = levels.length < 2;
  el.replaceChildren(...levels.flatMap((v, i) => {
    const n = t.byId.get(v);
    const name = `${n.parent < 0 ? `class ${n.class}` : `node ${n.node}`} · ${plural(t.cells.get(v), "leaf", "leaves")}`;
    const here = i === levels.length - 1;
    const crumb = here ? h("span", { class: "here", "aria-current": "location" }, name)
      : h("button", { type: "button", class: "chip" }, name);
    if (!here) crumb.onclick = () => selectNode(v, true);
    return i ? [h("span", { class: "sep", "aria-hidden": "true" }, "›"), crumb] : [crumb];
  }));
}

/**
 * Node-link drawing of one level of a class tree, its top on top (port of bruno's treeDiagram): leaves at 0, 1, 2, ...
 * in pre-order, a parent at the mean of its children, one row per depth; squares for cells, circles for inner nodes,
 * size by sqrt(n_rows) against the top; a closed node hangs a triangle over the number of its cells; cells below
 * state.node and closed nodes take their map colour, the other nodes stay neutral; n_rows under a node when the
 * spacing allows, rows and CSV rows in the tooltip; state.node outlined; a click selects a node, and opens a closed one.
 * @param {HTMLElement} el container, replaced
 * @param {object} t from classTree
 * @param {{top: number, leaves: number[], open: Set<number>}} view from treeView
 * @param {{groups: object[], byCell: Map<number, {node: number, color: string}>}} colors from leafColor or
 *   defectColor
 */
function treeDiagram(el, t, view, colors) {
  const shown = [...view.open, ...view.leaves].sort((a, b) => a - b).map((v) => t.byId.get(v)); // pre-order
  const posX = new Map(view.leaves.map((v, i) => [v, i]));
  for (const n of [...shown].reverse()) {
    const below = t.kids.get(n.node);
    if (view.open.has(n.node)) posX.set(n.node, below.reduce((a, c) => a + posX.get(c), 0) / below.length);
  }
  const next = view.leaves.length;
  const top = t.byId.get(view.top);
  const depth = Math.max(...shown.map((n) => n.depth)) - top.depth;
  const step = Math.max(14, 1100 / Math.max(next, 1)); // room per leaf
  const W = Math.max(1100, step * next), m = 30;
  const rowH = Math.max(22, Math.min(64, 640 / Math.max(depth, 1)));
  const H = depth * rowH + 2 * m + 30;
  const x = (id) => (next > 1 ? m + (posX.get(id) / (next - 1)) * (W - 2 * m) : W / 2);
  const y = (n) => m + (n.depth - top.depth) * rowH;
  const labelled = step >= 44;
  const groupOf = new Map(colors.groups.map((g) => [g.node, g.color]));
  const edges = shown.filter((n) => n !== top).map((n) => {
    const up = t.byId.get(n.parent);
    return sv("line", { class: "edge", x1: x(up.node), y1: y(up), x2: x(n.node), y2: y(n) });
  });
  const marks = shown.map((n) => {
    const size = Math.min(step * 0.9, 6 + 18 * Math.sqrt(n.n_rows / top.n_rows));
    const picked = n.node === state.node;
    const closed = t.kids.has(n.node) && !view.open.has(n.node);
    const cx = x(n.node), cy = y(n);
    const tint = colors.tintOf?.(n.node) // a closed node is in one group of leafColor
      ?? colors.byCell.get(t.first.get(n.node))?.color ?? "var(--axis)";
    const fill = n.cell != null ? tint : "var(--surface)";
    const ring = picked ? "var(--text)" : groupOf.get(n.node) ?? (n.cell != null ? "var(--surface)" : "var(--muted)");
    const style = `fill:${fill};stroke:${ring};stroke-width:${picked ? 2.5 : 1.5}`;
    const mark = n.cell != null
      ? sv("rect", { x: cx - size / 2, y: cy - size / 2, width: size, height: size, rx: 2, style })
      : sv("circle", { cx, cy, r: size / 2, style });
    const w = Math.min(step * 0.7, 16), y0 = cy + size / 2 + 2;
    const subtree = closed ? [
      sv("path", { class: "subtree", d: `M${cx},${y0}L${cx + w / 2},${y0 + 11}L${cx - w / 2},${y0 + 11}Z`, style: `fill:${tint}` }),
      sv("text", { class: "count", x: cx, y: y0 + 23, "text-anchor": "middle" }, num(t.cells.get(n.node))),
    ] : null;
    const label = !closed && (labelled || picked || groupOf.has(n.node))
      ? sv("text", { x: cx, y: cy + size / 2 + 12, "text-anchor": "middle" }, num(n.n_rows))
      : null;
    const g = sv("g", { class: "node" }, mark, subtree, label);
    const found = colors.tipOf && n.cell == null ? ` · ${colors.tipOf(n.node)}` : "";
    g.onmousemove = (ev) => tip(ev, nodeTip(n, t, closed) + found);
    g.onmouseleave = () => tip(null);
    g.onclick = () => selectNode(n.node, true);
    return g;
  });
  el.replaceChildren(h("div", { class: "table-wrap" },
    sv("svg", { class: "tree", viewBox: `0 0 ${W} ${H}`, width: W, role: "img", style: "max-width:none" }, edges, marks)));
}

let mapId = 0; // drops map answers of superseded requests

/**
 * Fetch the map of state.node and draw it (port of bruno's drawMap): its sampled rows in the plane of their own first
 * two principal components, coloured by their group of leafColor; one scale for both axes, framed on the 1st to 99th
 * percentiles; navigable with panZoom; answers of a node selected in the meantime are dropped.
 * @param {object[]} nodes the cell_tree rows of state.cls
 * @param {{groups: object[], byCell: Map<number, {node: number, color: string}>}} colors from leafColor or
 *   defectColor
 */
async function drawMap(nodes, colors) {
  const id = ++mapId;
  const el = $("rg-map");
  const sel = nodes.find((n) => n.node === state.node);
  const what = sel.parent < 0 ? `class ${sel.class}` : sel.cell != null ? `leaf ${sel.cell}` : `node ${sel.node}`;
  $("rg-map-title").textContent = `Map of ${what}`;
  let data;
  try {
    data = await api(`/${encodeURIComponent(state.dataset)}/cells/map?node=${sel.node}`);
  } catch (e) {
    if (id === mapId) el.replaceChildren(h("p", { class: "note" }, e.message));
    return;
  }
  if (id !== mapId) return;
  const byId = new Map(nodes.map((n) => [n.node, n]));
  const cellNode = new Map(nodes.filter((n) => n.cell != null).map((n) => [n.cell, n]));
  const swatch = (color, text) => h("span", {}, h("i", { style: `background:${color}` }), text);
  const legend = h("div", { class: "legend" }, colors.legend?.map((l) => swatch(l.color, l.label)) ??
    colors.groups.map((g) => {
      const n = byId.get(g.node);
      return swatch(g.color, `${n.cell != null ? `leaf ${n.cell}` : `node ${n.node}`} · ${num(n.n_read)} CSV rows`);
    }));
  const names = new Map(colors.groups.map((g) => {
    const n = byId.get(g.node);
    return [g.color, n.cell != null ? `leaf ${n.cell}` : `node ${n.node}`];
  }));
  const map = pointMap(data, (c) => colors.byCell.get(c)?.color ?? "var(--muted)", (c) => {
    const n = cellNode.get(c);
    return n ? nodeTip(n) : null;
  }, colors.nameOf ?? ((color) => names.get(color) ?? "other leaves"), `map-clip-${id}`);
  el.replaceChildren(legend, map);
}

const OTHER_CLASS = "var(--muted)"; // the colour of the cells past the palette

const LEVELS = [0.25, 0.5, 0.75, 0.95]; // the share of a group's rows inside each of its curves, innermost first
const LEVEL_OPACITY = [1, 0.75, 0.55, 0.35];
const CELL = 4; // side of a density cell, in viewBox units
let showPoints = false; // the maps draw their rows as points too (the switch above every map)

/**
 * Sampled rows in their PCA plane, one scale for both axes framed on the 1st to 99th percentiles, navigable with
 * panZoom (the plot of drawMap). Each group of cells (the cells drawn in one colour) is shown by its density: four
 * curves enclosing 25, 50, 75 and 95% of its rows, and a curve along each axis in the strips above and on the right,
 * scaled to its own peak. A point stands for rows / taken rows of its cell (data.sampled), so the cells sampled
 * down weigh as much as they hold. A switch adds the points, whose tooltip names their cell.
 * @param {{shares: number[], points: number[][], sampled: number[][]}} data response of a map endpoint
 * @param {(cell: number) => string} colorOf colour of a cell's group
 * @param {(cell: number) => string|null} tipOf tooltip of a cell's points
 * @param {(color: string) => string} nameOf name of the group drawn in a colour, for the tooltip of the curves
 * @param {string} clip a unique id for the clip paths
 * @returns {HTMLElement} the switch and the plot
 */
function pointMap(data, colorOf, tipOf, nameOf, clip) {
  const pts = data.points.map(([a, b, cell]) => ({ a, b, cell }));
  const weight = new Map(data.sampled.map(([cell, rows, taken]) => [cell, rows / taken]));
  const frame = (vals) => {
    const lo = percentile(vals, 1), hi = percentile(vals, 99), pad = 0.2 * (hi - lo || 1);
    return [lo - pad, hi + pad];
  };
  const [xa, xb] = frame(pts.map((p) => p.a)), [ya, yb] = frame(pts.map((p) => p.b));
  const W = 1100, H = 600, S = 64, G = 8; // S: depth of a strip, G: its gap to the plot
  const m = { l: 48, r: 12 + S + G, t: 12 + S + G, b: 36 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const k = Math.min(pw / (xb - xa), ph / (yb - ya)); // one scale for both axes
  const cx = (pw - k * (xb - xa)) / 2 + m.l, cy = (ph - k * (yb - ya)) / 2 + m.t;
  const bx = pts.map((p) => cx + (p.a - xa) * k), by = pts.map((p) => cy + (yb - p.b) * k);

  // the density grid spans the plot and half of it again on every side, where a pan can bring the eye
  const grid = { x0: m.l - pw / 2, y0: m.t - ph / 2, nx: Math.ceil((2 * pw) / CELL), ny: Math.ceil((2 * ph) / CELL) };
  const members = new Map();
  pts.forEach((p, i) => {
    const color = colorOf(p.cell);
    if (!members.has(color)) members.set(color, []);
    members.get(color).push(i);
  });
  const groups = [...members].map(([color, idx]) =>
    ({ color, ...density(idx.map((i) => bx[i]), idx.map((i) => by[i]),
      idx.map((i) => weight.get(pts[i].cell) ?? 1), grid) }));
  groups.sort((g1, g2) => g2.total - g1.total); // the small groups drawn last, on top

  const stroke = (color, i) => `fill:none;stroke:${color};stroke-width:1.5;opacity:${LEVEL_OPACITY[i]}`;
  const curves = sv("g", {}, groups.flatMap((g) => g.levels.map((t, i) => t == null ? null
    : sv("path", { d: isolines(g.cells, grid, t), style: stroke(g.color, i), "vector-effect": "non-scaling-stroke" }))
    .filter(Boolean)));
  const line = (pairs) => "M" + pairs.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join("L");
  const col = (i) => grid.x0 + (i + 0.5) * CELL, row = (j) => grid.y0 + (j + 0.5) * CELL;
  const top = sv("g", {}, groups.map((g) => sv("path", {
    d: line(g.alongX.map((v, i) => [col(i), 12 + S - v * (S - 4)])),
    style: `fill:none;stroke:${g.color};stroke-width:1.5`, "vector-effect": "non-scaling-stroke",
  })));
  const right = sv("g", {}, groups.map((g) => sv("path", {
    d: line(g.alongY.map((v, j) => [W - 12 - S + v * (S - 4), row(j)])),
    style: `fill:none;stroke:${g.color};stroke-width:1.5`, "vector-effect": "non-scaling-stroke",
  })));
  const dots = pts.map((p) => sv("circle", {
    r: 2.5, "data-cell": p.cell, style: `fill:${colorOf(p.cell)};opacity:0.8`,
  }));
  const dotLayer = sv("g", {}, dots);
  // an SVG element has no hidden property: display shows and hides the points
  const pointsShown = () => dotLayer.getAttribute("display") !== "none";
  dotLayer.setAttribute("display", showPoints ? "inline" : "none");

  const rect = (id, x, y, width, height) => sv("clipPath", { id }, sv("rect", { x, y, width, height }));
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", class: "zoomable", style: "min-width:640px" },
    sv("defs", {},
      rect(clip, m.l, m.t, pw, ph), rect(`${clip}-x`, m.l, 12, pw, S), rect(`${clip}-y`, W - 12 - S, m.t, S, ph)),
    sv("rect", { x: m.l, y: m.t, width: pw, height: ph, fill: "none", style: "stroke:var(--grid)" }),
    sv("line", { class: "axis", x1: m.l, x2: m.l + pw, y1: 12 + S, y2: 12 + S }),
    sv("line", { class: "axis", x1: W - 12 - S, x2: W - 12 - S, y1: m.t, y2: m.t + ph }),
    sv("g", { "clip-path": `url(#${clip})` }, curves, dotLayer),
    sv("g", { "clip-path": `url(#${clip}-x)` }, top),
    sv("g", { "clip-path": `url(#${clip}-y)` }, right),
    sv("text", { x: m.l + pw / 2, y: H - 10, "text-anchor": "middle" }, `PC 1 · ${pct(data.shares[0], 0)} of the variance`),
    sv("text", { transform: `translate(16, ${m.t + ph / 2}) rotate(-90)`, "text-anchor": "middle" },
      `PC 2 · ${pct(data.shares[1], 0)} of the variance`));

  let now = { k: 1, x: 0, y: 0 };
  const draw = (t) => {
    now = t;
    curves.setAttribute("transform", `matrix(${t.k} 0 0 ${t.k} ${t.x} ${t.y})`);
    top.setAttribute("transform", `matrix(${t.k} 0 0 1 ${t.x} 0)`);
    right.setAttribute("transform", `matrix(1 0 0 ${t.k} 0 ${t.y})`);
    if (!pointsShown()) return;
    dots.forEach((d, i) => {
      d.setAttribute("cx", (t.k * bx[i] + t.x).toFixed(1));
      d.setAttribute("cy", (t.k * by[i] + t.y).toFixed(1));
    });
  };
  svg.onmousemove = (ev) => {
    const cell = ev.target.getAttribute?.("data-cell");
    if (cell != null) return tip(ev, tipOf(Number(cell)));
    const p = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(svg.getScreenCTM().inverse());
    if (p.x < m.l || p.x > m.l + pw || p.y < m.t || p.y > m.t + ph) return tip(null);
    const i = Math.floor(((p.x - now.x) / now.k - grid.x0) / CELL), j = Math.floor(((p.y - now.y) / now.k - grid.y0) / CELL);
    if (i < 0 || j < 0 || i >= grid.nx || j >= grid.ny) return tip(null);
    const inside = groups.map((g) => [g, g.levels.findIndex((t) => t != null && g.cells[j * grid.nx + i] >= t)])
      .filter(([, level]) => level >= 0).sort((a, b) => a[1] - b[1]);
    tip(ev, inside.length ? inside.map(([g, level]) => `${nameOf(g.color)}: inside its ${pct(LEVELS[level], 0)} curve`)
      .join(" · ") : null);
  };
  svg.onmouseleave = () => tip(null);
  panZoom(svg, draw);

  const choice = (label, on) => {
    const b = h("button", { class: "chip", type: "button", "aria-pressed": String(showPoints === on) }, label);
    b.onclick = () => {
      showPoints = on;
      dotLayer.setAttribute("display", on ? "inline" : "none");
      for (const x of b.parentElement.children) x.setAttribute("aria-pressed", String(x === b));
      draw(now);
    };
    return b;
  };
  return h("div", {},
    h("div", { class: "chips map-switch" }, choice("curves", false), choice("curves and points", true)),
    h("div", { class: "table-wrap" }, svg));
}

const REACH = [0, 1, 2, 3, 4, 6, 8, 11, 16, 22, 32, 45, 64]; // half sides, in cells, of the squares tried by density

/**
 * Weighted density of some points on a grid of CELL-sided cells: their weights binned in the cells, then smoothed by
 * a Gaussian (three box blurs) as wide as the median reach of a point, the half side of the smallest square around it
 * holding n_eff^(-1/3) / π of the weight. For a Gaussian cloud that is Scott's bandwidth (sd · n_eff^(-1/6)); measured
 * around each point, it keeps the detail of thin or far apart clusters that the spread of the whole group would blur.
 * @param {number[]} xs
 * @param {number[]} ys
 * @param {number[]} ws weight of each point
 * @param {{x0: number, y0: number, nx: number, ny: number}} grid
 * @returns {{cells: Float32Array, levels: (number|null)[], alongX: number[], alongY: number[], total: number}}
 *   cells: mass per cell, row by row; levels: the density above which lie LEVELS of the total weight (null when the
 *   grid holds less); alongX, alongY: the density along each axis, scaled to its peak; total: the summed weight
 */
function density(xs, ys, ws, grid) {
  const { x0, y0, nx, ny } = grid;
  const total = ws.reduce((s, w) => s + w, 0);
  const nEff = total ** 2 / ws.reduce((s, w) => s + w * w, 0);
  const cells = new Float32Array(nx * ny), alongX = new Float32Array(nx), alongY = new Float32Array(ny);
  const at = xs.map((x, n) => {
    const i = Math.floor((x - x0) / CELL), j = Math.floor((ys[n] - y0) / CELL);
    if (i >= 0 && i < nx) alongX[i] += ws[n];
    if (j >= 0 && j < ny) alongY[j] += ws[n];
    if (i < 0 || i >= nx || j < 0 || j >= ny) return -1;
    cells[j * nx + i] += ws[n];
    return j * nx + i;
  });
  const buf = new Float32Array(Math.max(nx, ny));
  const sigma = Math.max(1, medianReach(cells, at, ws, (nEff ** (-1 / 3) / Math.PI) * total, grid, buf) + 0.5);
  const r = Math.max(1, Math.round((Math.sqrt(4 * sigma * sigma + 1) - 1) / 2)); // three such boxes make that Gaussian
  for (let pass = 0; pass < 3; pass++) {
    for (let j = 0; j < ny; j++) blur(cells, j * nx, 1, nx, r, buf);
    for (let i = 0; i < nx; i++) blur(cells, i, nx, ny, r, buf);
    blur(alongX, 0, 1, nx, r, buf);
    blur(alongY, 0, 1, ny, r, buf);
  }
  const sorted = Float32Array.from(cells).sort().reverse();
  const levels = LEVELS.map(() => null);
  let sum = 0, next = 0;
  for (let c = 0; c < sorted.length && next < LEVELS.length && sorted[c] > 0; c++) {
    sum += sorted[c];
    while (next < LEVELS.length && sum >= LEVELS[next] * total) levels[next++] = sorted[c];
  }
  const scaled = (a) => { const peak = Math.max(...a) || 1; return Array.from(a, (v) => v / peak); };
  return { cells, levels, alongX: scaled(alongX), alongY: scaled(alongY), total };
}

/**
 * The weighted median, over the points on the grid, of the smallest of REACH whose square around the point's cell
 * holds at least need of the weight binned in cells (the last of REACH when none does).
 * @param {Float32Array} cells weight per cell
 * @param {number[]} at cell of each point, -1 off the grid
 * @param {number[]} ws weight of each point
 * @param {number} need
 * @param {{nx: number, ny: number}} grid
 * @param {Float32Array} buf scratch of at least max(nx, ny)
 * @returns {number} in cells
 */
function medianReach(cells, at, ws, need, grid, buf) {
  const { nx, ny } = grid;
  const reach = at.map(() => REACH[REACH.length - 1]), open = new Set(at.flatMap((c, n) => (c < 0 ? [] : [n])));
  const box = new Float32Array(cells.length);
  for (const r of REACH) {
    box.set(cells);
    if (r) {
      for (let j = 0; j < ny; j++) blur(box, j * nx, 1, nx, r, buf);
      for (let i = 0; i < nx; i++) blur(box, i, nx, ny, r, buf);
    }
    const scale = (2 * r + 1) ** 2; // blur leaves the mean of the square, not its sum
    for (const n of open) if (box[at[n]] * scale >= need) { reach[n] = r; open.delete(n); }
    if (!open.size) break;
  }
  const on = at.flatMap((c, n) => (c < 0 ? [] : [n])).sort((a, b) => reach[a] - reach[b]);
  const half = on.reduce((s, n) => s + ws[n], 0) / 2;
  let sum = 0;
  for (const n of on) if ((sum += ws[n]) >= half) return reach[n];
  return 0;
}

/** One box blur of radius r, zero beyond the ends, of the n values a[off], a[off + stride], …, in place. */
function blur(a, off, stride, n, r, buf) {
  let s = 0;
  for (let i = 0; i < Math.min(r, n); i++) s += a[off + i * stride];
  for (let i = 0; i < n; i++) {
    if (i + r < n) s += a[off + (i + r) * stride];
    if (i - r - 1 >= 0) s -= a[off + (i - r - 1) * stride];
    buf[i] = s / (2 * r + 1);
  }
  for (let i = 0; i < n; i++) a[off + i * stride] = buf[i];
}

/**
 * The curve where a grid of cells crosses t (marching squares between the cells' centres), as SVG path data of
 * unjoined segments.
 * @param {Float32Array} cells
 * @param {{x0: number, y0: number, nx: number, ny: number}} grid
 * @param {number} t
 * @returns {string}
 */
function isolines(cells, grid, t) {
  const { x0, y0, nx, ny } = grid;
  const out = [];
  const seg = (p, q) => out.push(`M${p[0].toFixed(1)},${p[1].toFixed(1)}L${q[0].toFixed(1)},${q[1].toFixed(1)}`);
  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const a = cells[j * nx + i], b = cells[j * nx + i + 1];
      const c = cells[(j + 1) * nx + i + 1], d = cells[(j + 1) * nx + i];
      const code = (a >= t ? 8 : 0) | (b >= t ? 4 : 0) | (c >= t ? 2 : 0) | (d >= t ? 1 : 0);
      if (code === 0 || code === 15) continue;
      const x = x0 + (i + 0.5) * CELL, y = y0 + (j + 0.5) * CELL;
      const T = [x + (CELL * (t - a)) / (b - a), y], R = [x + CELL, y + (CELL * (t - b)) / (c - b)];
      const B = [x + (CELL * (t - d)) / (c - d), y + CELL], L = [x, y + (CELL * (t - a)) / (d - a)];
      switch (code) {
        case 1: case 14: seg(L, B); break;
        case 2: case 13: seg(B, R); break;
        case 3: case 12: seg(L, R); break;
        case 4: case 11: seg(T, R); break;
        case 6: case 9: seg(T, B); break;
        case 7: case 8: seg(L, T); break;
        case 5: seg(L, T); seg(B, R); break;
        case 10: seg(T, R); seg(L, B); break;
      }
    }
  }
  return out.join("");
}

/** p-th percentile (0-100) of values, linear between ranks. */
function percentile(values, p) {
  const v = [...values].sort((a, b) => a - b);
  const i = (v.length - 1) * p / 100, lo = Math.floor(i);
  return lo + 1 < v.length ? v[lo] + (v[lo + 1] - v[lo]) * (i - lo) : v[lo];
}

/**
 * Pan (drag) and zoom (pinch or ctrl + wheel) of an SVG, reset on double click (port of bruno's panZoom).
 * @param {SVGSVGElement} svg
 * @param {(t: {k: number, x: number, y: number}) => void} draw redraws the marks under the transform t
 */
function panZoom(svg, draw) {
  let t = { k: 1, x: 0, y: 0 }, drag = null, frame = 0;
  const render = () => { cancelAnimationFrame(frame); frame = requestAnimationFrame(() => draw(t)); };
  const at = (ev) => new DOMPoint(ev.clientX, ev.clientY).matrixTransform(svg.getScreenCTM().inverse()); // viewBox units
  svg.addEventListener("wheel", (ev) => {
    if (!ev.ctrlKey && !ev.metaKey) return; // a pinch arrives as ctrl + wheel
    ev.preventDefault();
    const p = at(ev), d = Math.max(-50, Math.min(50, ev.deltaY * (ev.deltaMode === 1 ? 20 : 1)));
    const k = Math.max(1, Math.min(64, t.k * Math.exp(-d * 0.01)));
    t = { k, x: p.x - ((p.x - t.x) * k) / t.k, y: p.y - ((p.y - t.y) * k) / t.k };
    render();
  }, { passive: false });
  svg.addEventListener("pointerdown", (ev) => {
    drag = { p: at(ev), t };
    svg.setPointerCapture(ev.pointerId);
    svg.classList.add("dragging");
  });
  svg.addEventListener("pointermove", (ev) => {
    if (!drag) return;
    const p = at(ev);
    t = { k: drag.t.k, x: drag.t.x + p.x - drag.p.x, y: drag.t.y + p.y - drag.p.y };
    render();
  });
  const end = () => { drag = null; svg.classList.remove("dragging"); };
  svg.addEventListener("pointerup", end);
  svg.addEventListener("pointercancel", end);
  svg.addEventListener("dblclick", () => { t = { k: 1, x: 0, y: 0 }; render(); });
  draw(t);
}

// The quality tab calls cell what quality_report calls a region (its id the region's): the leaves of a class's tree
// merged where their defects agree. It never shows the leaves.

// the defects of a cell, in the order of src.quality.DEFECTS
const DEFECTS = ["invalid_values", "duplicates", "label_conflict", "suspect_label", "class_overlap", "shortcut",
  "repetition", "off_scenario"];

/** The defects the last quality run measured: off_scenario needs a scenario (share null without it). */
const measuredDefects = () => DEFECTS.filter((d) => last.quality.stats[`share_${d}`] != null);
const RIVAL_COLUMNS = 6; // most rival classes compared feature by feature, the highest rival share first
const RIVAL_SHORTCUTS = 12; // most rival classes listed in a shortcut's evidence, the highest rival share first

/** What each defect means, one line: the glossary and the tooltip of its chips. */
const DEFINITION = {
  invalid_values: "at least one row holds an empty, unparsable, NaN or infinite value",
  duplicates: "at least one row comes more than once in the CSV, label included: copies can fall on both sides of a " +
    "train/test split",
  label_conflict: "at least one row is identical on every feature to a row of another class: no classifier can tell " +
    "them apart",
  suspect_label: "against a rival class, its rival share is at least one half beyond the error of sampling: the cell " +
    "lies inside the other class, as rows under a wrong label would",
  class_overlap: "against a rival class, its rival share is τ or more beyond the error of sampling, and it is no " +
    "suspect_label",
  shortcut: "against a rival class, the best rule on a single column between its class and the " +
    "rival class leaves under τ of the cell's rows and under τ of the rival class's rows on the wrong side, beyond the " +
    "error of sampling: one column alone tells the two classes apart where the cell lies",
  repetition: "the cell lies in a branch of its class's tree that holds at least τ of the class's CSV rows and whose " +
    "rows lie at most τ times as far from their centre as the dataset's: a large part of its class repeated in " +
    "little space",
  off_scenario: "at least τ of the cell's CSV rows break a rule of the scenario the dataset declares for its class " +
    "(a background port as destination, a host or a time the documentation does not give the attack), beyond the " +
    "error of sampling",
};

/** The other words of the quality tab, one line each. */
const TERMS = {
  "CSV rows": "the rows of the CSV, copies included, as a classifier is trained on them; every share is of them",
  cell: "a compact group of rows of one class, the unit every defect is measured on: neighbouring leaves of the " +
    "class's tree (the cells tab) with the same defects against the same classes and the same host",
  "rival class": "another class holding the nearest row of another class to one of the cell's rows; the leaves " +
    "that hold those rows are its rival leaves",
  "rival share": "the share of a cell's rows whose nearest row in the whole dataset (itself left out) is of a " +
    "rival class; a row as near a rival row as its own nearest counts one half",
  undecided: "a measure within the error of sampling of its threshold, at the run's confidence: the cell neither has " +
    "nor lacks the defect, and it never splits a cell from its neighbours",
  "clear ratio": "the share of the CSV rows with none of the defects",
};

/** Which rows of the CSV have each defect: the row itself for the first three, its cell for the others. */
const ROWS_WITH = {
  invalid_values: "hold a null or invalid value",
  duplicates: "repeat a row read before them (the first copy of a row does not)",
  label_conflict: "are identical to a row of another class",
  suspect_label: "lie in a cell with suspect_label",
  class_overlap: "lie in a cell with class_overlap",
  shortcut: "lie in a cell with shortcut",
  repetition: "lie in a cell with repetition",
  off_scenario: "lie in a cell with off_scenario",
};

/** One sentence per column of the quality tables, shown on hover of the header. */
const HELP = {
  read: "rows of the CSV, copies included (hover a number for the distinct rows)",
  why: "each defect of the cell with what was found",
  rule: "the column whose rule between the two classes errs least on its worse side, with its error on this cell's " +
    "rows and on the rival class's",
  cellError: "the share of this cell's rows the rule of the two classes sends to the rival's side",
  rivalError: "the share of the rival class's rows the rule sends to this cell's side",
};

/** The hyperparameters, the matrix, the cells and the glossary. @param {object} data response of /quality */
function renderQuality(data) {
  last.quality = data;
  fill("qr-params", paramTable(data.params));
  $("qr-tau").value = $("qr-tau-value").textContent = data.stats.tau;
  $("qr-tau-reset").hidden = state.tau == null;
  const term = (name, text) => [h("dt", {}, name), h("dd", {}, text)];
  fill("qr-glossary",
    Object.entries(TERMS).map(([k, v]) => term(k, v)),
    measuredDefects().map((d) => term(d, DEFINITION[d].replaceAll("τ", `τ = ${data.stats.tau}`))));
  if (!data.regions.some((r) => r.region === state.region)) state.region = null;
  drawMatrix();
  drawCellList();
}

/**
 * Pick the cells listed: those of a class (null: every class) with a defect ("none": without; null: any).
 * @param {string|null} cls
 * @param {string|null} filter
 */
function pick(cls, filter) {
  state.qclass = cls;
  state.qfilter = filter;
  state.region = null;
  saveHash(true);
  drawMatrix();
  drawCellList();
  scrollUnderHeader($("qr-list"));
}

/**
 * The classes by their clear ratio and their share of CSV rows with each defect, the most CSV rows first, then the
 * whole dataset: one hue, darker as the share grows. A share picks its cells on a click.
 */
function drawMatrix() {
  const data = last.quality, s = data.stats, measured = measuredDefects();
  const total = data.classes.reduce((a, c) => a + c.n_read, 0);
  const lines = [...data.classes].sort((a, b) => b.n_read - a.n_read || a.class.localeCompare(b.class))
    .map((c) => ({ cls: c.class, name: c.class, n_read: c.n_read, n_cells: c.n_regions, of: c }));
  lines.push({ cls: null, name: "dataset", n_read: total, n_cells: s.n_regions, of: s });
  // every cell is listed when nothing is picked: no share stands for that
  const picked = (cls, f) => state.qclass === cls && state.qfilter === f && (cls != null || f != null);
  const cell = (line, f, content, title, v) => {
    const td = h("td", { class: `pick${picked(line.cls, f) ? " picked" : ""}`, title, tabindex: "0" }, content);
    if (v != null && v > 0 && f !== "none") td.style.background = `color-mix(in srgb, var(--accent) ${Math.round(3 + 55 * v)}%, transparent)`;
    td.onclick = () => pick(line.cls, f);
    td.onkeydown = (e) => { if (e.key === "Enter") pick(line.cls, f); };
    return td;
  };
  const head = h("tr", {},
    h("th", { class: "text" }, "class"), h("th", { title: HELP.read }, "CSV rows"),
    h("th", { title: TERMS["clear ratio"] }, "clear ratio"),
    measured.map((d) => h("th", { title: DEFINITION[d] }, colName(d))));
  const body = lines.map((line) => h("tr", { class: line.cls == null ? "total" : "" },
    cell(line, null, line.name, `${line.name}: its ${plural(line.n_cells, "cell")}`),
    h("td", {}, num(line.n_read)),
    cell(line, "none", share(line.of.share_no_defect), `the cells of ${line.name} without defects`),
    measured.map((d) => {
      const v = line.of[`share_${d}`];
      return cell(line, d, v ? pctSmall(v) : h("span", { class: "muted" }, "0"),
        `${pctSmall(v)} of the CSV rows of ${line.name} ${ROWS_WITH[d]}: click for its cells with ${d}`, v);
    })));
  fill("qr-matrix", h("table", { class: "matrix" }, h("thead", {}, head), h("tbody", {}, body)));
}

/** A column name that may break after its underscores. @param {string} name */
function colName(name) {
  return h("span", {}, name.split("_").flatMap((part, i, all) => (i < all.length - 1 ? [`${part}_`, h("wbr")] : [part])));
}

/** Column names joined by "," as breakable names, comma separated. @param {string} names */
function colNames(names) {
  return h("span", {}, names.split(",").flatMap((n, i) => (i ? [", ", colName(n)] : [colName(n)])));
}

/** Whether a cell passes a filter of the list: null all, "none" no defect, else that defect. */
function passes(r, f) {
  if (f == null) return true;
  if (f === "none") return !r.defects.length;
  return r.defects.includes(f);
}

/**
 * Each defect of a cell with what was found, one line each; without a defect its highest rival share, muted.
 * @param {object} r a quality_regions row
 * @param {Map<number, object[]>} missingOf the null and invalid values of each cell, by column and kind
 */
function why(r, missingOf) {
  if (!r.defects.length) {
    return h("span", { class: "muted" },
      r.rival_share == null ? "no defect · no rival measured" : `no defect · highest rival share ${dec(r.rival_share, 2)} (${r.rival_class})`);
  }
  const found = {
    invalid_values: () => `${plural(r.missing_read, "CSV row")} with a null or invalid value, in ` +
      some([...new Set((missingOf.get(r.region) ?? []).map((m) => m.column))]),
    duplicates: () => `${pct(r.dup_share)} of its CSV rows are copies`,
    label_conflict: () => `${plural(r.conflict_read, "CSV row")} identical to rows of ${r.conflict_with.join(", ")}`,
    suspect_label: () => `inside ${r.label_with.join(", ")} (highest rival share ${interval(r)})`,
    class_overlap: () => `confused with ${r.overlap_with.join(", ")} (highest ${interval(r)})`,
    shortcut: () => `one column tells it from ${r.shortcut_with.join(", ")}`,
    repetition: () => `in ${pct(r.repetition_share)} of its class, spread ${dec(r.repetition_ratio, 3)} times the dataset`,
    off_scenario: () => "off the scenario of its class",
  };
  return h("div", { class: "why" }, r.defects.map((d) => h("div", {}, diagnosisChip(d), " ", found[d]())));
}

/** A rival share with its interval: "0.42 (0.31–0.53)". @param {object} r a row with rival_share, _low and _high */
const interval = (r) => (r.rival_share_low == null ? dec(r.rival_share, 2)
  : `${dec(r.rival_share, 2)} (${dec(r.rival_share_low, 2)}–${dec(r.rival_share_high, 2)})`);

/** The filter line and the cells picked, the most CSV rows first, then the detail of state.region. */
function drawCellList() {
  const data = last.quality;
  const missingOf = new Map();
  for (const m of data.missing) missingOf.set(m.region, [...(missingOf.get(m.region) ?? []), m]);
  const shown = data.regions.filter((r) => (state.qclass == null || r.class === state.qclass) && passes(r, state.qfilter));
  const of = state.qclass == null ? "of every class" : `of ${state.qclass}`;
  const what = state.qfilter == null ? "" : state.qfilter === "none" ? " without defects" : ` with ${state.qfilter}`;
  const all = h("button", { class: "chip", type: "button" }, "show every cell");
  all.onclick = () => pick(null, null);
  fill("qr-filter", `${plural(shown.length, "cell")} ${of}${what}`,
    state.qclass != null || state.qfilter != null ? all : null);
  sortable($("qr-cells"), [
    { key: "region", label: "cell" },
    { key: "class", label: "class", text: true },
    { key: "n_read", label: "CSV rows", format: (v, r) => h("span", { title: `${num(r.n_rows)} distinct rows` }, num(v)),
      help: HELP.read },
    { key: "defects", label: "defects", format: (v, r) => why(r, missingOf), help: HELP.why, wrap: true },
  ], shown, {
    onClick: (r) => selectCell(r.region),
    selected: (r) => r.region === state.region,
    empty: "No cell.",
    by: "n_read",
  });
  drawDetail();
}

/** Open the detail of a cell (null: close it). @param {number|null} cell its region id */
function selectCell(cell) {
  state.region = cell;
  saveHash(true);
  drawCellList();
  if (cell != null) scrollUnderHeader($("qr-detail"));
}

/** Scroll an element to the top of the window, below the sticky header. @param {HTMLElement} el */
function scrollUnderHeader(el) {
  const top = el.getBoundingClientRect().top + scrollY - document.querySelector(".top").offsetHeight - 12;
  window.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
}

/**
 * The rival classes of a cell, one per class, each by the measure of its leaf of highest rival share among those kept, with
 * the rival leaves of that class of every leaf kept; the highest rival share first.
 * @param {object} d the detail of the cell (/quality/region)
 * @param {(rival: object) => boolean} [keep] the measures to look at
 * @returns {{class: string, here: number, rules: Array<[string, number, number]>, cells: number[]}[]}
 */
function rivalClasses(d, keep = () => true) {
  const best = new Map(), cells = new Map();
  for (const r of d.rivals) {
    const rival = { class: r.rival_class, here: r.rival_share, rules: r.rules };
    if (!keep(rival)) continue;
    cells.set(rival.class, new Set([...(cells.get(rival.class) ?? []), ...r.rival_cells]));
    const was = best.get(rival.class);
    if (was == null || (rival.here ?? -1) > (was.here ?? -1)) best.set(rival.class, rival);
  }
  return [...best.values()].map((r) => ({ ...r, cells: [...cells.get(r.class)] }))
    .sort((a, b) => (b.here ?? -1) - (a.here ?? -1) || a.class.localeCompare(b.class));
}

let detailId = 0; // drops detail answers of superseded requests

/** One part of a cell's evidence: the defect's name, then its content. @param {string} d one of DEFECTS */
function evidence(d, ...kids) {
  return h("section", { class: "evidence" }, h("h3", {}, d), ...kids);
}

/**
 * Fetch the detail of state.region, as a cell, and draw the evidence of each of its defects, its map with the rivals
 * picked, and, closed, its rivals and where its rows come from.
 */
async function drawDetail() {
  const id = ++detailId;
  const card = $("qr-detail");
  const row = last.quality.regions.find((r) => r.region === state.region);
  card.hidden = row == null;
  if (row == null) return;
  const tau = last.quality.stats.tau;
  const close = h("button", { class: "chip close", type: "button", title: "close the detail" }, "close");
  close.onclick = () => selectCell(null);
  fill("qr-detail-title", `Cell ${row.region} · ${row.class}`, close);
  fill("qr-detail-head",
    `${num(row.n_read)} CSV rows (${num(row.n_rows)} distinct) · `,
    flagged(`${pct(row.dup_share)} copies`, row.defects.includes("duplicates")), " · ",
    row.defects.length ? chips(row.defects) : "no defect");
  $("qr-all").open = false;
  fill("qr-evidence", h("p", { class: "note" }, "loading…"));
  for (const el of ["qr-map", "qr-rivals", "qr-provenance", "qr-provenance-time"]) fill(el);
  const ds = encodeURIComponent(state.dataset);
  let d, profile;
  try {
    [d, profile] = await Promise.all([
      api(`/${ds}/quality/region?region=${row.region}${tauQuery("&")}`),
      api(`/${ds}/quality/profile?region=${row.region}${tauQuery("&")}`).catch((e) => ({ error: e.message })),
    ]);
  } catch (e) {
    if (id === detailId) fill("qr-evidence", h("p", { class: "note" }, e.message));
    return;
  }
  if (id !== detailId) return;
  // the rival classes of each defect, as the run judged them (beyond the error of sampling)
  const against = (names) => rivalClasses(d, (s) => names.includes(s.class));
  const label = against(row.label_with), overlap = against(row.overlap_with), shortcut = against(row.shortcut_with);
  const parts = {
    invalid_values: () => evidence("invalid_values", sortableIn([
      { key: "column", label: "column", text: true },
      { key: "kind", label: "kind", text: true, help: "empty, unparsable (not a number), nan or inf" },
      { key: "n_read", label: "CSV rows", format: (v, r) => h("span", { title: `${num(r.n_rows)} distinct rows` }, num(v)) },
    ], d.missing, { by: "n_read" })),
    duplicates: () => null, // its copies are in the head
    label_conflict: () => evidence("label_conflict", conflicts(d, row)),
    suspect_label: () => evidence("suspect_label", featureDeltas(profile, label.slice(0, RIVAL_COLUMNS))),
    class_overlap: () => evidence("class_overlap", featureDeltas(profile, overlap.slice(0, RIVAL_COLUMNS))),
    shortcut: () => evidence("shortcut", shortcutTable(shortcut.slice(0, RIVAL_SHORTCUTS), tau)),
    repetition: () => evidence("repetition", branchTable(d, row, tau)),
    off_scenario: () => evidence("off_scenario", scenarioTable(d)),
  };
  fill("qr-evidence", row.defects.map((x) => parts[x]()).filter(Boolean));
  const rivals = rivalClasses(d);
  sortable($("qr-rivals"), [
    { key: "class", label: "rival class", text: true, help: TERMS["rival class"] },
    { key: "here", label: "rival share here", format: (v) => dec(v, 3), help: TERMS["rival share"] },
    { key: "rules", label: "best one-column rule", wrap: true, help: HELP.rule,
      format: (v) => (v.length ? [colName(v[0][0]), ` · ${dec(v[0][1], 3)} here, ${dec(v[0][2], 3)} of the rival`] : "–") },
  ], rivals, { empty: "No class was measured against it." });
  drawCellProvenance(d);
  const involved = new Set([...label, ...overlap, ...shortcut].map((r) => r.class));
  drawCellMap(row, d, rivals, involved.size ? involved : new Set(rivals.slice(0, 1).map((r) => r.class)), id);
}

/**
 * Feature by feature, how far this cell lies from the rows of each rival class near it: the difference of their mean
 * numeric features (logged and standardized: in standard deviations of the dataset), darker as it grows; the most
 * frequent value of each categorical feature, its own muted where it is this cell's.
 * @param {object} profile response of /quality/profile, or {error}
 * @param {object[]} rivals from rivalClasses, the columns
 */
function featureDeltas(profile, rivals) {
  if (profile.error) return h("p", { class: "note" }, profile.error);
  const mine = profile.own.means, top = profile.own.top;
  const of = new Map(profile.rivals.map((r) => [r.class, r]));
  const shade = (v) => `background: color-mix(in srgb, var(--accent) ${Math.round(55 * Math.min(Math.abs(v) / 2, 1))}%, transparent)`;
  const head = h("tr", {}, h("th", { class: "text" }, "feature"), h("th", { title: "its mean, in standard deviations, or its most frequent value" }, "this cell"),
    rivals.map((r) => h("th", { title: `rival share ${dec(r.here, 3)}` }, r.class)));
  const numeric = profile.columns.map((c, i) => h("tr", {}, h("td", { class: "text" }, c), h("td", {}, dec(mine[i], 2)),
    rivals.map((r) => {
      const v = of.get(r.class)?.means[i];
      const delta = v == null || mine[i] == null ? null : v - mine[i];
      return h("td", delta == null ? {} : { style: shade(delta), title: `its mean ${dec(v, 2)}` },
        delta == null ? "–" : `${delta < 0 ? "−" : "+"}${dec(Math.abs(delta), 2)}`);
    })));
  const categorical = Object.keys(top).map((c) => h("tr", {}, h("td", { class: "text" }, c), h("td", {}, top[c] ?? "null"),
    rivals.map((r) => {
      const v = of.get(r.class)?.top[c];
      return h("td", v === top[c] ? { class: "muted" } : {}, v ?? "null");
    })));
  return h("div", { class: "table-wrap" },
    h("table", { class: "deltas" }, h("thead", {}, head), h("tbody", {}, numeric, categorical)));
}

/**
 * Per rival class this cell has a shortcut against, the columns whose rule between the two classes errs under τ on
 * both sides here: its error on this cell's rows and on the rival class's, the best first.
 * @param {object[]} rivals from rivalClasses, the shortcut ones
 * @param {number} tau
 * @returns {HTMLElement}
 */
function shortcutTable(rivals, tau) {
  if (!rivals.length) return h("p", { class: "note" }, "No rival class measured.");
  const lines = rivals.flatMap((r) => r.rules.filter(([, a, b]) => a < tau && b < tau)
    .map(([column, a, b]) => [r.class, colName(column), dec(a, 3), dec(b, 3)]));
  return h("div", { class: "table-wrap" }, table(
    ["rival class", "column", h("span", { title: HELP.cellError }, "error here"),
      h("span", { title: HELP.rivalError }, "error on the rival class")], lines));
}

/**
 * The share of the cell's CSV rows that break the scenario of its class, over its leaves: on any key, the one
 * off_scenario reads, then per key.
 * @param {object} d the detail of the cell (/quality/region)
 */
function scenarioTable(d) {
  const line = (name, prefix) => {
    let broken = 0, judged = 0;
    for (const c of d.cells) {
      const n = c[`${prefix}_n`] ?? 0;
      if (n && c[prefix] != null) { broken += c[prefix] * n; judged += n; }
    }
    return judged ? [name, pct(broken / judged), num(judged)] : null;
  };
  const lines = [line("any key", "scenario"),
    ...["dst_port", "src_ip", "time"].map((key) => line(key, `scenario_${key}`))].filter(Boolean);
  return h("div", { class: "table-wrap" }, table(["key", "CSV rows off the scenario", "CSV rows judged"], lines));
}

/**
 * The widest branch of its class the cell lies in that is a repetition: its share of the class's CSV rows and its
 * variance over the dataset's, against τ.
 * @param {object} d the detail of the cell (/quality/region)
 * @param {object} row its quality_regions row
 * @param {number} tau
 */
function branchTable(d, row, tau) {
  return h("div", { class: "table-wrap" }, table(
    ["share of its class in the branch", "variance of the branch over the dataset's", "dataset variance", "τ"],
    [[pct(row.repetition_share), dec(row.repetition_ratio, 3), dec(d.cells[0]?.variance_dataset, 2), String(tau)]]));
}

/** "YYYY-MM-DD HH:MM:SS → HH:MM:SS", the day written once when both times fall on it. */
const span = (from, to) => `${from.slice(0, 19)} → ${to.slice(0, 10) === from.slice(0, 10) ? to.slice(11, 19) : to.slice(0, 19)}`;

/** Per provenance column, over the cell's CSV rows: its distinct values, the most frequent, the rows per repeated value. */
function drawCellProvenance(d) {
  const time = d.time == null ? null
    : h("p", { class: "note" }, `time ${span(d.time.time_from.replace("T", " "), d.time.time_to.replace("T", " "))}`);
  fill("qr-provenance-time", time);
  sortable($("qr-provenance"), [
    { key: "key", label: "column", text: true },
    { key: "n_values", label: "distinct values", format: num },
    { key: "top_share", label: "most frequent", text: true, format: (v, r) => (r.top == null ? "–" : `${r.top} · ${pct(v)}`) },
    { key: "repeat", label: "CSV rows per repeated value", format: (v) => (v == null ? "–" : short(v)),
      help: "the mean CSV rows of a value, over the values of more than one" },
  ], d.provenance, { empty: "The dataset names no provenance columns (a provenance block in its yaml)." });
}

/**
 * The label conflict groups of a cell: per group the classes on its two sides, this cell's and the others, with their
 * CSV rows, then the values all its rows share.
 * @returns {HTMLElement}
 */
function conflicts(d, row) {
  if (!d.conflicts.length) return h("p", { class: "note" }, "No group of identical rows was kept.");
  const columns = Object.keys(d.conflicts[0].values);
  const value = (v) => (v == null ? h("span", { class: "muted" }, "null") : String(v));
  const sum = (rows) => rows.reduce((a, r) => a + r.n_read, 0);
  return h("div", { class: "table-wrap conflicts" }, table(
    ["class a", "class b", "CSV rows a", "CSV rows b", ...columns],
    d.conflicts.map((g) => {
      const a = g.rows.filter((r) => r.class === row.class), b = g.rows.filter((r) => r.class !== row.class);
      return [row.class, [...new Set(b.map((r) => r.class))].join(", "), num(sum(a)), num(sum(b)),
        ...columns.map((c) => value(g.values[c]))];
    })));
}

let qualityMapId = 0; // drops quality map answers of superseded requests

/**
 * Fetch the map of a cell and its rival leaves, then draw it with the rival classes picked by the chips above it, each
 * in its colour: this cell first, a rival class by its place among the rivals (so its colour stays as others are
 * picked).
 * @param {object} row the cell's quality_regions row
 * @param {object} d its detail (/quality/region)
 * @param {object[]} rivals from rivalClasses, every rival class
 * @param {Set<string>} picked the rival classes drawn first
 * @param {number} detail the detailId of the detail it belongs to
 */
async function drawCellMap(row, d, rivals, picked, detail) {
  const id = ++qualityMapId;
  const el = $("qr-map");
  el.replaceChildren(h("p", { class: "note" }, "loading…"));
  let data;
  try {
    data = await api(`/${encodeURIComponent(state.dataset)}/quality/map?region=${row.region}${tauQuery("&")}`);
  } catch (e) {
    if (id === qualityMapId) el.replaceChildren(h("p", { class: "note" }, e.message));
    return;
  }
  if (id !== qualityMapId || detail !== detailId) return;
  shuffle(data.points); // drawn interleaved: no cell covers the others where they mix
  // a leaf's group: this cell (0), or the place of its rival class among the rivals (1, 2, ...)
  const groupOf = new Map(d.cells.map((c) => [c.cell, 0]));
  rivals.forEach((r, i) => { for (const leaf of r.cells) groupOf.set(leaf, i + 1); });
  const colorOf = new Map([[0, SERIES[0]], ...rivals.map((r, i) => [i + 1, i + 1 < SERIES.length ? SERIES[i + 1] : OTHER_CLASS])]);
  const name = new Map([[0, `cell ${row.region} · ${row.class} (this one)`], ...rivals.map((r, i) => [i + 1, r.class])]);
  const box = h("div");
  const draw = () => {
    const shown = new Set([0, ...rivals.flatMap((r, i) => (picked.has(r.class) ? [i + 1] : []))]);
    const keep = (leaf) => shown.has(groupOf.get(leaf));
    const part = { shares: data.shares, points: data.points.filter((p) => keep(p[2])), sampled: data.sampled.filter((s) => keep(s[0])) };
    const legend = h("div", { class: "legend" }, [...shown].map((c) =>
      h("span", {}, h("i", { style: `background:${colorOf.get(c)}` }), name.get(c))));
    const byColor = new Map([...shown].map((c) => [colorOf.get(c), name.get(c)]));
    box.replaceChildren(legend, pointMap(part, (leaf) => colorOf.get(groupOf.get(leaf)) ?? OTHER_CLASS,
      (leaf) => name.get(groupOf.get(leaf)) ?? null, (color) => byColor.get(color) ?? "other cells", `quality-clip-${id}`));
  };
  const toggles = rivals.map((r, i) => {
    const b = h("button", { class: "chip", type: "button", "aria-pressed": String(picked.has(r.class)),
      title: `rival share ${dec(r.here, 3)}` }, h("i", { style: `background:${colorOf.get(i + 1)}` }), name.get(i + 1));
    b.onclick = () => {
      if (picked.has(r.class)) picked.delete(r.class); else picked.add(r.class);
      b.setAttribute("aria-pressed", String(picked.has(r.class)));
      draw();
    };
    return b;
  });
  el.replaceChildren(h("div", { class: "chips rivals" }, h("span", { class: "muted" }, "rivals"), toggles), box);
  draw();
}

/** Shuffle an array in place, the same way every time (a fixed seed), so a redraw looks the same. @param {any[]} a */
function shuffle(a) {
  let seed = 42;
  const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/** Defects as chips, "–" without. @param {string[]} v */
function chips(v) {
  return v.length ? h("span", { class: "chips" }, v.map(diagnosisChip)) : "–";
}

/** A defect as a chip, its definition on hover. @param {string} d */
function diagnosisChip(d) {
  return h("span", { class: "chip defect", title: `${d}: ${DEFINITION[d]}` }, d);
}

/** A value, in bold when it shows a defect. @param {string|Node} text @param {boolean} over */
function flagged(text, over) {
  return over ? h("b", { class: "over", title: "over its threshold" }, text) : text;
}

/** A share in [0, 1] as an inline bar and a percentage. */
function share(v) {
  if (v == null) return "–";
  return h("span", {}, h("i", { class: "bar", style: `width:${Math.round(v * 60)}px` }), pctSmall(v));
}

// --- charts ---

/** Smallest 1, 2, 5 or 10 times a power of ten that is >= v, so the ticks are round. */
function niceMax(v) {
  const p = 10 ** Math.floor(Math.log10(v));
  return [1, 2, 5, 10].map((m) => m * p).find((x) => x >= v);
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
