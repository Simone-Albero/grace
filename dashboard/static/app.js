// Dashboard front-end: one page, one tab per stage. Charts are hand-made SVG, no dependencies.

/**
 * UI state.
 * @type {{dataset: string|null, tab: "clean"|"prepare"|"cells"|"quality", cls: string|null, node: number|null,
 *   view: "classes"|"regions", qclass: string|null, qfilter: string|null, region: number|null}}
 *   node: the cell tree node whose map is shown (null: the root of state.cls), the tree drawn from the level that
 *   shows it (treeLevels); view: the quality panel shown; qclass,
 *   qfilter: the filters of the quality regions (null: all), qfilter "none" or one of DEFECTS; region: the quality
 *   region whose detail is shown (null: none)
 */
const state = {
  dataset: null, tab: "clean", cls: null, node: null,
  view: "regions", qclass: null, qfilter: null, region: null,
};

const last = { cells: null, quality: null }; // last responses, for the controls that only redraw
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
 * @param {{onClick?: (row: object) => void, selected?: (row: object) => boolean, empty?: string}} [opts]
 *   empty: the note shown instead of a table without rows
 */
function sortable(el, columns, rows, opts = {}) {
  if (!rows.length) { el.replaceChildren(h("p", { class: "note" }, opts.empty ?? "Nothing to show.")); return; }
  let by = null, dir = 1;
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
  readHash(opts.datasets);
  // back, forward and a pasted link change only the hash: draw what it names
  window.onhashchange = async () => {
    if (location.hash === shownHash) return;
    readHash(opts.datasets);
    await showTab(state.tab);
    if (state.tab === "quality" && state.view === "regions" && state.region != null) scrollUnderHeader($("qr-detail"));
  };
  $("dataset").onchange = (e) => {
    state.dataset = e.target.value;
    state.cls = state.node = state.qclass = state.qfilter = state.region = null;
    load();
  };
  $("rg-class").onchange = (e) => selectClass(e.target.value);
  $("rg-tree-nodes").textContent = TREE_NODES;
  $("qr-class").onchange = (e) => { state.qclass = e.target.value || null; drawRegionList(); };
  for (const b of document.querySelectorAll(".view-switch button")) b.onclick = () => showView(b.dataset.view, true);
  for (const b of document.querySelectorAll(".tabs button")) b.onclick = () => showTab(b.dataset.tab);
  showTab(state.tab);
}

let shownHash = ""; // the hash last written by saveHash: a hashchange to it is our own

/**
 * Set state from the URL hash: tab, dataset, and the cells tab's class and node or the quality tab's view and region.
 * @param {string[]} datasets the known datasets; an unknown one falls back to the first
 */
function readHash(datasets) {
  const [tab, dataset, cls, node] = location.hash.slice(1).split("/").map(decodeURIComponent);
  if ([...document.querySelectorAll(".tabs button")].some((b) => b.dataset.tab === tab)) state.tab = tab;
  const next = datasets.includes(dataset) ? dataset : datasets[0];
  if (next !== state.dataset) state.qclass = state.qfilter = null;
  state.dataset = next;
  if (tab === "quality") {
    state.view = cls === "classes" ? "classes" : "regions";
    state.region = node ? Number(node) : null;
  }
  if (tab === "cells") {
    state.cls = cls || null;
    state.node = node ? Number(node) : null;
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
 * Keep tab, dataset and, on the cells tab, class and node, on the quality tab view and region, in the URL hash.
 * @param {boolean} [push] a step the back button returns from (opening a region, switching panel); else the entry is replaced
 */
function saveHash(push = false) {
  const parts = [state.tab, state.dataset];
  if (state.tab === "quality") parts.push(state.view, ...(state.view === "regions" && state.region != null ? [state.region] : []));
  if (state.tab === "cells" && state.cls != null && state.node != null) {
    parts.push(state.cls, state.node);
  } else if (state.tab === "cells" && state.cls != null) parts.push(state.cls);
  shownHash = `#${parts.map((p) => encodeURIComponent(p)).join("/")}`;
  if (shownHash === location.hash) return;
  if (push) history.pushState(null, "", shownHash);
  else history.replaceState(null, "", shownHash);
}

/** Fetch the data of state.tab and render it; on error show the message in the tab's empty state. */
async function load() {
  const id = ++loadId;
  const tab = state.tab;
  saveHash();
  const panel = $(`tab-${tab}`);
  const empty = panel.querySelector(".empty");
  const d = `/${encodeURIComponent(state.dataset)}`;
  const path = {
    clean: `${d}/clean`,
    prepare: `${d}/prepare`,
    cells: `${d}/cells`,
    quality: `${d}/quality`,
  }[tab];
  try {
    const data = await api(path);
    if (id !== loadId) return;
    panel.classList.remove("failed");
    empty.hidden = true;
    ({ clean: renderClean, prepare: renderPrepare, cells: renderCells, quality: renderQuality })[tab](data);
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

/** Tiles, numeric columns with their scaling, categorical slots with their rows. @param {object} data response of /prepare */
function renderPrepare(data) {
  const s = data.summary;
  fill("pp-tiles",
    tile("rows", num(s.n_rows)),
    tile("numeric columns", num(s.n_num)),
    tile("categorical columns", num(s.n_cat)),
    tile("prepared columns", num(s.n_columns)),
    tile("numbers set to 0", num(s.n_missing), `${num(s.n_missing_rows)} rows · ${pct(s.n_missing_rows / s.n_rows)}`));
  sortable($("pp-num"), [
    { key: "column", label: "column" },
    { key: "mean", label: "mean (log)", format: (v) => dec(v, 4) },
    { key: "std", label: "std (log)", format: (v) => dec(v, 4) },
  ], data.columns.filter((c) => c.kind === "num"));
  sortable($("pp-cat"), [
    { key: "source", label: "column" },
    { key: "value", label: "value" },
    { key: "n_rows", label: "rows", format: num },
    { key: "n_rows", label: "share", format: (v) => pct(v / s.n_rows) },
    { key: "weight", label: "weight", format: (v) => dec(v, 3) },
    { key: "weight", label: "mismatch cost", format: (v) => dec(v == null ? null : 2 * v * v, 2) },
  ], data.columns.filter((c) => c.kind === "cat"));
}

/** Tiles, class tree and map, cells per class, cells. @param {object} data response of /cells */
function renderCells(data) {
  last.cells = data;
  const s = data.stats;
  const rows = data.classes.reduce((a, c) => a + c.n_rows, 0);
  fill("rg-tiles",
    tile("cells", num(s.n_cells), `${num(rows)} rows`),
    tile("rows per cell", `≤ ${num(s.max_rows)}`, `at least ${num(s.min_rows)}`),
    tile("tree built on", num(s.max_fit_samples), "rows per class at most"),
    tile("distance", s.distance));
  sortable($("rg-classes"), [
    { key: "class", label: "class" },
    { key: "n_rows", label: "rows", format: num },
    { key: "n_read", label: "CSV rows", format: num },
    { key: "n_cells", label: "cells", format: num },
  ], data.classes, { onClick: (r) => selectClass(r.class), selected: (r) => r.class === state.cls });
  const classes = data.classes.map((c) => c.class);
  if (!classes.includes(state.cls)) { state.cls = classes[0]; state.node = null; }
  $("rg-class").replaceChildren(...data.classes.map((c) =>
    h("option", { value: c.class }, `${c.class} · ${num(c.n_cells)} ${c.n_cells > 1 ? "cells" : "cell"}`)));
  const sizes = data.cells.map((c) => c.n_rows);
  const edges = Array.from({ length: 21 }, (_, i) => (Math.max(...sizes, 1) * i) / 20);
  histogram($("rg-sizes"), edges, [{
    name: "cells by rows", color: "var(--accent)",
    values: edges.slice(0, -1).map((lo, i) => sizes.filter((v) => v >= lo && (v < edges[i + 1] || i === 19)).length),
  }], { format: num });
  const leafOf = new Map(data.tree.filter((n) => n.cell != null).map((n) => [n.cell, n.node]));
  sortable($("rg-table"), [
    { key: "cell", label: "cell" },
    { key: "class", label: "class" },
    { key: "n_rows", label: "rows", format: num },
    { key: "n_read", label: "CSV rows", format: num },
  ], data.cells, {
    onClick: (r) => { selectNode(leafOf.get(r.cell), true); scrollUnderHeader($("rg-map-title")); },
  });
  drawClass();
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
 * levels down to state.node, with the levels above it as a path.
 * @param {boolean} [push] passed to saveHash
 */
function drawClass(push = false) {
  const t = classTree(last.cells.tree.filter((n) => n.class === state.cls));
  if (!t.byId.has(state.node)) state.node = t.root;
  $("rg-class").value = state.cls;
  saveHash(push);
  const levels = treeLevels(t, state.node);
  const view = treeView(t, levels.at(-1));
  const colors = leafColor(t, state.node, view.open);
  drawPath(levels, t);
  treeDiagram($("rg-tree"), t, view, colors);
  drawMap(t.nodes, colors);
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
  for (const g of colored) {
    const stack = [g.node];
    while (stack.length) {
      const id = stack.pop();
      const n = t.byId.get(id);
      if (n.cell != null) byCell.set(n.cell, g);
      stack.push(...(t.kids.get(id) ?? []));
    }
  }
  return { groups: colored, byCell };
}

/**
 * Tooltip text of a tree node: its counts, the cell of a leaf, the cells below an inner node.
 * @param {object} n a cell_tree row
 * @param {object} [t] from classTree, for an inner node
 * @param {boolean} [closed] a closed node of the drawing
 */
function nodeTip(n, t, closed = false) {
  const what = n.cell != null ? `cell ${n.cell}` : `node ${n.node} · ${num(t.cells.get(n.node))} cells`;
  return `${what} · ${num(n.n_rows)} rows · ${num(n.n_read)} CSV rows${closed ? " · click to open" : ""}`;
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
    const name = `${n.parent < 0 ? `class ${n.class}` : `node ${n.node}`} · ${num(t.cells.get(v))} cells`;
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
 * @param {{groups: object[], byCell: Map<number, {node: number, color: string}>}} colors from leafColor
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
    const tint = colors.byCell.get(t.first.get(n.node))?.color ?? "var(--axis)"; // a closed node is in one group
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
    g.onmousemove = (ev) => tip(ev, nodeTip(n, t, closed));
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
 * @param {{groups: object[], byCell: Map<number, {node: number, color: string}>}} colors from leafColor
 */
async function drawMap(nodes, colors) {
  const id = ++mapId;
  const el = $("rg-map");
  const sel = nodes.find((n) => n.node === state.node);
  const what = sel.parent < 0 ? `class ${sel.class}` : sel.cell != null ? `cell ${sel.cell}` : `node ${sel.node}`;
  $("rg-map-title").textContent = `Map of ${what}`;
  let data;
  try {
    data = await api(`/${encodeURIComponent(state.dataset)}/cells/map?node=${sel.node}`);
  } catch (e) {
    if (id === mapId) { el.replaceChildren(h("p", { class: "note" }, e.message)); $("rg-map-note").textContent = ""; }
    return;
  }
  if (id !== mapId) return;
  const byId = new Map(nodes.map((n) => [n.node, n]));
  const cellNode = new Map(nodes.filter((n) => n.cell != null).map((n) => [n.cell, n]));
  const leaves = nodes.filter((n) => colors.byCell.has(n.cell)).length;
  $("rg-map-note").textContent = `The rows of ${what} on the first two principal components of ` +
    `${num(data.points.length)} of them taken at random (at most 2,000 per cell, 20,000 in all), in the space of the ` +
    "cells. " +
    (colors.groups.length < leaves
      ? `Its ${num(leaves)} cells are coloured by the ${colors.groups.length} subtrees ringed in the tree; click one to go deeper. `
      : "Each cell has its colour. ") +
    MAP_HOWTO;
  const legend = h("div", { class: "legend" }, colors.groups.map((g) => {
    const n = byId.get(g.node);
    const name = n.cell != null ? `cell ${n.cell}` : `node ${n.node}`;
    return h("span", {}, h("i", { style: `background:${g.color}` }), `${name} · ${num(n.n_rows)} rows`);
  }));
  const names = new Map(colors.groups.map((g) => {
    const n = byId.get(g.node);
    return [g.color, n.cell != null ? `cell ${n.cell}` : `node ${n.node}`];
  }));
  const map = pointMap(data, (c) => colors.byCell.get(c)?.color ?? "var(--muted)", (c) => {
    const n = cellNode.get(c);
    return n ? nodeTip(n) : null;
  }, (color) => names.get(color) ?? "other cells", `map-clip-${id}`);
  el.replaceChildren(legend, map);
}

const OTHER_CLASS = "var(--muted)"; // the colour of the classes past the palette

/**
 * The colour of every class, in the fixed order of the classes; past the palette the classes share OTHER_CLASS.
 * @param {string[]} classes in class order: only those a view shows, so that they rarely outnumber the palette
 * @returns {Map<string, string>}
 */
function classColor(classes) {
  return new Map(classes.map((c, i) => [c, i < SERIES.length ? SERIES[i] : OTHER_CLASS]));
}

/**
 * Legend entries of some classes, in class order: one per class with a hue of its own, then one for those sharing
 * OTHER_CLASS that names them all, so no class is told by colour alone.
 * @param {Map<string, string>} colors from classColor
 * @param {Map<string, string>} shown the classes to list, each with its label
 * @returns {HTMLElement[]}
 */
function classLegend(colors, shown) {
  const entries = [...colors].filter(([c]) => shown.has(c));
  const swatch = (col, text) => h("span", {}, h("i", { style: `background:${col}` }), text);
  const rest = entries.filter(([, col]) => col === OTHER_CLASS).map(([c]) => shown.get(c));
  return [
    ...entries.filter(([, col]) => col !== OTHER_CLASS).map(([c, col]) => swatch(col, shown.get(c))),
    ...(rest.length ? [swatch(OTHER_CLASS, `other classes: ${rest.join(", ")}`)] : []),
  ];
}

/**
 * The name of the classes drawn in each colour, for the tooltip of a map's curves.
 * @param {Map<string, string>} colors from classColor
 * @returns {(color: string) => string}
 */
function colorClass(colors) {
  const names = new Map([...colors].map(([c, col]) => [col, c]));
  return (color) => (color === OTHER_CLASS ? "other classes" : names.get(color) ?? "other classes");
}

// how to read a map of pointMap, the end of every map's note
const MAP_HOWTO = "Each colour's curves enclose 25, 50, 75 and 95% of its rows (a sampled row counts for the rows of " +
  "its cell it stands for); the strips show its rows along each axis, scaled to its own peak. Hover the plot for " +
  "the curves you are inside, or a point for its cell; drag to pan, pinch or ctrl + scroll to zoom, double click " +
  "to reset.";

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
  dotLayer.hidden = !showPoints;

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
    if (dotLayer.hidden) return;
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
      dotLayer.hidden = !on;
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

// the defects of a cell or a region, in the order of src.quality.DEFECTS
const DEFECTS = ["invalid_values", "duplicates", "label_conflict", "class_overlap", "shortcut"];
const RIVAL_BARS = 12; // most pairs drawn as bars in a region's detail, the most confused first

/** One sentence per column of the quality tables, shown on hover of the header. */
const HELP = {
  rows: "distinct rows (as kept by stage clean)",
  read: "rows of the CSV, copies included",
  cells: "the cells merged into the region: neighbours in the tree of the class with the same defects between cells",
  defects: "the defects of the region (see How to read); none: it has none",
  missing: "rows with at least one empty, unparsable, NaN or infinite cell, and in which columns (invalid_values)",
  copies: "share of the CSV rows that repeat another row exactly (duplicates when above 0)",
  conflicts: "rows identical on every feature to rows of another class, and of which classes (label_conflict)",
  overlap: "the classes its cells have class_overlap against, then the highest confusion of its cells with a rival",
  shortcut: "the classes its cells have shortcut against: a separation that rests on a few columns",
  confusion: "share of the cell's rows the better of 1-NN and the best single-column rule confuses with the rival's " +
    "(class_overlap from τ), and that rival's class",
  nn: "share of rows whose nearest row is in the other cell (0 apart, about 0.5 mixed)",
  bayes: "least error any classifier could reach, from the pair's nn_error (Cover & Hart, approximate)",
  single: "balanced accuracy of the best rule on one column alone, and that column",
  key: "the columns removed one at a time, each the one whose removal raises the 1-NN error the most",
};

/**
 * Show a quality panel, classes or regions.
 * @param {"classes"|"regions"} view
 * @param {boolean} [push] a click: the back button returns to the panel before
 */
function showView(view, push = false) {
  state.view = view;
  for (const b of document.querySelectorAll(".view-switch button")) b.setAttribute("aria-pressed", String(b.dataset.view === view));
  $("qr-classes-view").hidden = view !== "classes";
  $("qr-regions-view").hidden = view !== "regions";
  saveHash(push);
  if (view === "regions" && last.quality) drawRegionList();
}

/**
 * The summary line, the How to read values, then the panel of state.view.
 * @param {object} data response of /quality
 */
function renderQuality(data) {
  last.quality = data;
  const s = data.stats;
  fill("qr-summary",
    h("b", {}, `${num(s.regions_no_defect)} of ${num(s.n_regions)} regions without defects`),
    ` · ${pctSmall(s.share_no_defect)} of the CSV rows with none · ${num(s.n_cells)} cells measured, ` +
    `${num(s.n_pairs)} pairs of a cell and a rival · τ = ${s.tau}`);
  for (const el of document.querySelectorAll("#tab-quality [data-stat]")) el.textContent = s[el.dataset.stat] ?? "–";
  drawClassTable();
  if (!data.regions.some((r) => r.region === state.region)) state.region = null;
  showView(state.view);
}

/** Which rows of the CSV have each defect: the row itself for the first three, its cell for the others. */
const ROWS_WITH = {
  invalid_values: "that hold a null or invalid value",
  duplicates: "that repeat a row read before them (the first copy of a row does not)",
  label_conflict: "identical to a row of another class",
  class_overlap: "in a cell with class_overlap",
  shortcut: "in a cell with shortcut",
};

/** The classes table: per defect, the share of the class's CSV rows with it, and with none. */
function drawClassTable() {
  sortable($("qr-classes"), [
    { key: "class", label: "class", wrap: true },
    { key: "n_rows", label: "rows", format: num, help: HELP.rows },
    { key: "n_read", label: "CSV rows", format: num, help: HELP.read },
    { key: "n_cells", label: "cells", format: num },
    { key: "n_regions", label: "regions", format: num },
    { key: "share_no_defect", label: "no defect", format: share,
      help: "share of the class's CSV rows with none of the five defects" },
    ...DEFECTS.map((d) => ({ key: `share_${d}`, label: d, format: pctSmall,
      help: `share of the class's CSV rows ${ROWS_WITH[d]}` })),
  ], last.quality.classes);
}

/** A column name that may break after its underscores. @param {string} name */
function colName(name) {
  return h("span", {}, name.split("_").flatMap((part, i, all) => (i < all.length - 1 ? [`${part}_`, h("wbr")] : [part])));
}

/** Column names joined by "," as breakable names, comma separated. @param {string} names */
function colNames(names) {
  return h("span", {}, names.split(",").flatMap((n, i) => (i ? [", ", colName(n)] : [colName(n)])));
}

/** Whether a quality region passes a filter of the regions panel: null all, "none" no defect, else that defect. */
function passes(r, f) {
  if (f == null) return true;
  if (f === "none") return !r.defects.length;
  return r.defects.includes(f);
}

/** The filters, the regions table with one column per defect, and the detail of state.region. */
function drawRegionList() {
  const data = last.quality;
  const missingOf = new Map();
  for (const m of data.missing) missingOf.set(m.region, [...(missingOf.get(m.region) ?? []), m]);
  const classes = data.classes.map((c) => c.class);
  if (!classes.includes(state.qclass)) state.qclass = null;
  $("qr-class").replaceChildren(h("option", { value: "" }, "all classes"), ...classes.map((c) => h("option", { value: c }, c)));
  $("qr-class").value = state.qclass ?? "";
  const ofClass = data.regions.filter((r) => state.qclass == null || r.class === state.qclass);
  fill("qr-filters", [null, "none", ...DEFECTS].map((f) => {
    const chip = h("button", { class: "chip", type: "button", "aria-pressed": String(state.qfilter === f) },
      `${f ?? "all"} (${num(ofClass.filter((r) => passes(r, f)).length)})`);
    chip.onclick = () => { state.qfilter = f; drawRegionList(); };
    return chip;
  }));
  const missingText = (v, r) => {
    if (!v) return "";
    const cells = (missingOf.get(r.region) ?? []).map((m) => `${m.column} ${m.kind}`);
    const more = cells.length > 3 ? ` and ${cells.length - 3} more` : "";
    return flagged(`${num(v)} ${v === 1 ? "row" : "rows"}: ${cells.slice(0, 3).join(", ")}${more}`, true);
  };
  const conflictText = (v, r) => (v
    ? flagged(`${num(v)} ${v === 1 ? "row" : "rows"} · with ${r.conflict_with.join(", ")}`, true)
    : "");
  const overlapText = (v, r) => {
    const worst = r.confusion == null ? "" : `highest confusion ${dec(r.confusion, 2)} (${r.rival_class})`;
    if (!v.length) return h("span", { class: "muted" }, worst || "no rival measured");
    return h("span", {}, flagged(`with ${v.join(", ")}`, true), h("span", { class: "muted" }, ` · ${worst}`));
  };
  sortable($("qr-regions"), [
    { key: "region", label: "region" },
    { key: "class", label: "class", text: true },
    { key: "n_cells", label: "cells", format: num, help: HELP.cells },
    { key: "n_rows", label: "rows", format: num, help: HELP.rows },
    { key: "n_read", label: "CSV rows", format: num, help: HELP.read },
    { key: "defects", label: "defects", format: chips, help: HELP.defects, wrap: true },
    { key: "missing_rows", label: "null / invalid", format: missingText, help: HELP.missing, wrap: true },
    { key: "dup_share", label: "copies", format: (v) => flagged(pct(v), v > 0), help: HELP.copies },
    { key: "conflict_rows", label: "label conflicts", format: conflictText, help: HELP.conflicts, wrap: true },
    { key: "overlap_with", label: "class_overlap", format: overlapText, help: HELP.overlap, wrap: true },
    { key: "shortcut_with", label: "shortcut", format: (v) => (v.length ? flagged(`with ${v.join(", ")}`, true) : ""),
      help: HELP.shortcut, wrap: true },
  ], ofClass.filter((r) => passes(r, state.qfilter)), {
    onClick: (r) => selectRegion(r.region),
    selected: (r) => r.region === state.region,
    empty: "No region matches these filters.",
  });
  drawDetail();
}

/** Open the detail of a region (null: close it). @param {number|null} region */
function selectRegion(region) {
  state.region = region;
  saveHash(true);
  drawRegionList();
  if (region != null) scrollUnderHeader($("qr-detail"));
}

/** Scroll an element to the top of the window, below the sticky header. @param {HTMLElement} el */
function scrollUnderHeader(el) {
  const top = el.getBoundingClientRect().top + scrollY - document.querySelector(".top").offsetHeight - 12;
  window.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
}

let detailId = 0; // drops detail answers of superseded requests

/** Fetch the detail of state.region and draw its cells, their rivals, bars, map, identical rows and null values. */
async function drawDetail() {
  const id = ++detailId;
  const card = $("qr-detail");
  const row = last.quality.regions.find((r) => r.region === state.region);
  card.hidden = row == null;
  if (row == null) return;
  const tau = last.quality.stats.tau;
  const close = h("button", { class: "chip close", type: "button", title: "close the detail" }, "close");
  close.onclick = () => selectRegion(null);
  fill("qr-detail-title", `Region ${row.region} · ${row.class}`, close);
  fill("qr-detail-head",
    `${num(row.n_cells)} ${row.n_cells === 1 ? "cell" : "cells"} · ${num(row.n_rows)} rows · ${num(row.n_read)} CSV rows · ` +
    `${pct(row.dup_share)} copies · `,
    row.defects.length ? h("span", {}, "defects: ", chips(row.defects)) : "no defect");
  for (const el of ["qr-cells", "qr-neighbours", "qr-bars", "qr-conflicts", "qr-missing"]) {
    fill(el, h("p", { class: "note" }, "loading…"));
  }
  const ds = encodeURIComponent(state.dataset);
  let d;
  try {
    d = await api(`/${ds}/quality/region?region=${row.region}`);
  } catch (e) {
    if (id === detailId) fill("qr-cells", h("p", { class: "note" }, e.message));
    return;
  }
  if (id !== detailId) return;
  const confusionOf = (v, r) => (v == null ? "–" : h("span", {}, flagged(dec(v, 3), v >= tau), ` with ${r.rival_class}`));
  sortable($("qr-cells"), [
    { key: "cell", label: "cell" },
    { key: "n_rows", label: "rows", format: num, help: HELP.rows },
    { key: "n_read", label: "CSV rows", format: num, help: HELP.read },
    { key: "confusion", label: "highest confusion", format: confusionOf, help: HELP.confusion, wrap: true },
    { key: "shortcut_columns", label: "shortcut columns",
      format: (v, r) => (v ? h("span", {}, flagged(colNames(v), r.confusion_without >= tau), ` → ${dec(r.confusion_without, 2)}`) : "–"),
      help: "against the rival under τ whose confusion rises the most as key columns are removed: those columns, and " +
        "its confusion without them", wrap: true },
    { key: "defects", label: "defects", format: chips, wrap: true },
  ], d.cells);
  const sides = d.pairs.map((p) => {
    const [s, o] = p.region_a === row.region ? ["a", "b"] : ["b", "a"];
    return {
      cell: p[`cell_${s}`], rival: p[`cell_${o}`], class: p[`class_${o}`], region: p[`region_${o}`],
      n_conflicts: p.n_conflicts, n_conflicts_read: p.n_conflicts_read,
      here: p[`confusion_${s}`], there: p[`confusion_${o}`], without: p[`confusion_without_${s}`],
      nn: p[`nn_error_${s}`], bayes: p.bayes_error_low,
      acc: p[`single_column_acc_${s}`], column: p.single_column, key: p.key_columns,
    };
  }).sort((a, b) => (b.here ?? -1) - (a.here ?? -1) || a.cell - b.cell || a.rival - b.rival);
  const steps = (v, r) => (r.here == null ? "–"
    : h("span", {}, [r.here, ...v].map((e, i) => (i ? ` → ${dec(e, 2)}` : dec(e, 2))).join(""),
      r.here < tau && v.some((e) => e >= tau) ? h("span", {}, " ", diagnosisChip("shortcut")) : ""));
  sortable($("qr-neighbours"), [
    { key: "cell", label: "cell", help: "this region's cell" },
    { key: "rival", label: "rival cell" },
    { key: "class", label: "rival class", wrap: true },
    { key: "region", label: "rival region" },
    { key: "here", label: "confusion here / there",
      format: (v, r) => (v == null ? "–" : h("span", {}, flagged(dec(v, 3), v >= tau), ` / ${dec(r.there, 3)}`)),
      help: "share of this cell's rows / of the rival's rows confused with the other side (class_overlap from τ)" },
    { key: "without", label: "confusion without key columns", format: steps, wrap: true,
      help: "this cell's confusion with every column, then after removing the key columns one at a time" },
    { key: "key", label: "key columns", format: (v) => (v ? colNames(v) : "–"), help: HELP.key, wrap: true },
    { key: "nn", label: "nn_error here", format: (v) => dec(v, 3), help: HELP.nn },
    { key: "acc", label: "single column here", format: (v, r) => (r.column == null ? "–" : h("span", {}, `${pct(v)} `, colName(r.column))), help: HELP.single, wrap: true },
    { key: "bayes", label: "bayes_error_low", format: (v) => dec(v, 3), help: HELP.bayes },
    { key: "n_conflicts", label: "identical rows",
      format: (v, r) => (v ? h("span", { title: `${num(r.n_conflicts_read)} CSV rows` }, flagged(num(v), true)) : "0"),
      help: "rows of the two cells identical on every feature (hover: CSV rows)" },
  ], sides, { onClick: (r) => selectRegion(r.region), empty: "No cell of another class was measured against its cells." });
  const judged = sides.filter((s) => s.here != null);
  $("qr-bars-note").textContent = (judged.length > RIVAL_BARS ? `The ${RIVAL_BARS} most confused of its ${num(judged.length)} pairs, ` : "Every pair, ") +
    "this region's side. A shortcut is a first bar short of τ and a later one past it: the classes mix once those " +
    "columns are gone.";
  errorBars($("qr-bars"), judged.slice(0, RIVAL_BARS), tau);
  drawConflicts(d, row);
  if (d.missing.length) {
    sortable($("qr-missing"), [
      { key: "column", label: "column" },
      { key: "kind", label: "kind", help: "empty, unparsable (not a number), nan or inf", text: true },
      { key: "n_rows", label: "rows", format: num, help: HELP.rows },
      { key: "n_read", label: "CSV rows", format: num, help: HELP.read },
    ], d.missing);
  } else fill("qr-missing", h("p", { class: "note" }, "None: every cell of the region is a valid value."));
  drawQualityMap(row, d, id);
}

/**
 * One group of horizontal bars per pair: this region's cell's confusion with every column, then after removing the key
 * columns one at a time, darker as more are removed (one hue, ordered); one axis from 0, τ dashed; a tooltip per bar.
 * @param {HTMLElement} el container, replaced
 * @param {object[]} sides the pairs drawn, in display order
 * @param {number} tau quality.tau
 */
function errorBars(el, sides, tau) {
  if (!sides.length) { el.replaceChildren(h("p", { class: "note" }, "No pair measured: too few rows.")); return; }
  const k = Math.max(...sides.map((s) => s.without.length));
  const opacity = Array.from({ length: k + 1 }, (_, i) => 0.35 + (0.65 * i) / Math.max(k, 1));
  const top = Math.min(1, niceMax(Math.max(tau * 2, ...sides.flatMap((s) => [s.here, ...s.without]))));
  const bar = 7, gap = 2, rowH = (k + 1) * (bar + gap) + 12;
  const W = Math.max(560, el.clientWidth || 1100); // drawn at the width it is shown, so text keeps its size
  const m = { l: 230, r: 24, t: 22, b: 26 }, H = m.t + m.b + rowH * sides.length;
  const x = (v) => m.l + (Math.min(v, top) / top) * (W - m.l - m.r);
  const kids = [];
  for (let t = 0; t <= 4; t++) {
    const v = (top * t) / 4;
    kids.push(sv("line", { class: t ? "grid" : "axis", x1: x(v), x2: x(v), y1: m.t - 4, y2: H - m.b }),
      sv("text", { x: x(v), y: H - 8, "text-anchor": "middle" }, short(v)));
  }
  kids.push(sv("line", { x1: x(tau), x2: x(tau), y1: m.t - 10, y2: H - m.b, style: "stroke:var(--text-2);stroke-dasharray:4 4" }),
    sv("text", { x: x(tau) + 4, y: m.t - 10 }, `τ = ${tau}`));
  const stepName = (s, i) => (i ? `without ${s.key.split(",").slice(0, i).join(", ")}` : "every column");
  sides.forEach((s, row) => {
    const y0 = m.t + row * rowH;
    const name = `cell ${s.cell} vs ${s.class} ${s.rival}`;
    kids.push(sv("text", { x: m.l - 8, y: y0 + bar + 2, "text-anchor": "end" }, name));
    [s.here, ...s.without].forEach((v, i) => {
      const y = y0 + i * (bar + gap);
      const w = Math.max(0, x(v) - x(0));
      const r = Math.min(3, w, bar / 2); // rounded data end, square at the baseline
      const mark = sv("path", {
        d: `M${x(0)},${y}h${w - r}a${r},${r} 0 0 1 ${r},${r}v${bar - 2 * r}a${r},${r} 0 0 1 ${-r},${r}h${-(w - r)}z`,
        style: `fill:var(--accent);fill-opacity:${opacity[i]}`,
      });
      const hit = sv("rect", { x: x(0), y, width: W - m.l - m.r, height: bar + gap, fill: "transparent" });
      hit.onmousemove = (ev) => tip(ev, `${name} · confusion ${stepName(s, i)}: ${dec(v, 3)}`);
      hit.onmouseleave = () => tip(null);
      kids.push(mark, hit);
    });
  });
  el.replaceChildren(
    h("div", { class: "legend" }, opacity.map((o, i) => h("span", {},
      h("i", { style: `background:var(--accent);opacity:${o}` }),
      i === 0 ? "every column" : i === 1 ? "without the first key column" : `without the first ${i} key columns`))),
    sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "confusion per pair, as key columns are removed" }, kids));
}

/** The label conflict groups of the region: one table row per group, its classes then its feature values. */
function drawConflicts(d, row) {
  if (!d.conflicts.length) {
    $("qr-conflicts-note").textContent = "";
    fill("qr-conflicts", h("p", { class: "note" }, "None: no row of this region is identical to a row of another class."));
    return;
  }
  const some = d.conflicts.length < d.n_groups ? `The ${d.conflicts.length} with the most CSV rows; each` : "Each";
  $("qr-conflicts-note").textContent = `${num(row.conflict_rows)} rows of this region (${num(row.conflict_read)} CSV rows) ` +
    `fall in ${num(d.n_groups)} ${d.n_groups === 1 ? "group" : "groups"} of rows identical on every feature but labelled ` +
    `with different classes. ${some} line is one group: which classes carry it, then the values all its rows share.`;
  const columns = Object.keys(d.conflicts[0].values);
  const value = (v) => (v == null ? h("span", { class: "muted" }, "null") : String(v));
  fill("qr-conflicts", table(
    ["class (cell, region) × CSV rows", "CSV rows", ...columns],
    d.conflicts.map((g) => [
      h("span", { class: "cell-wrap" },
        g.rows.map((r) => `${r.class} (cell ${r.cell}, region ${r.region}) ×${num(r.n_read)}`).join(" · ")),
      num(g.n_read),
      ...columns.map((c) => value(g.values[c])),
    ])));
}

let qualityMapId = 0; // drops quality map answers of superseded requests

/**
 * Fetch and draw the map of a region's cells and of their rival cells, coloured by class.
 * @param {object} row its quality_regions row
 * @param {object} d its detail (/quality/region): the class and region of every cell shown
 * @param {number} detail the detailId of the detail it belongs to
 */
async function drawQualityMap(row, d, detail) {
  const id = ++qualityMapId;
  $("qr-map-title").textContent = `Map of region ${row.region} and its rivals`;
  $("qr-map-note").textContent = "";
  fill("qr-map", h("p", { class: "note" }, "loading…"));
  let data;
  try {
    data = await api(`/${encodeURIComponent(state.dataset)}/quality/map?region=${row.region}`);
  } catch (e) {
    if (id === qualityMapId) fill("qr-map", h("p", { class: "note" }, e.message));
    return;
  }
  if (id !== qualityMapId || detail !== detailId) return;
  shuffle(data.points); // drawn interleaved: no cell covers the others where they mix
  const of = new Map(d.cells.map((c) => [c.cell, { class: c.class, region: row.region }]));
  for (const p of d.pairs) {
    for (const k of ["a", "b"]) if (!of.has(p[`cell_${k}`])) of.set(p[`cell_${k}`], { class: p[`class_${k}`], region: p[`region_${k}`] });
  }
  const order = last.quality.classes.map((c) => c.class);
  // in the dataset's class order, so a class keeps its colour from one region's map to the next
  const shown = [...new Set(data.cells.map((c) => of.get(c).class))];
  shown.sort((a, b) => order.indexOf(a) - order.indexOf(b));
  const colors = classColor(shown);
  const rivals = data.cells.filter((c) => of.get(c).region !== row.region);
  const labels = new Map(shown.map((c) => {
    if (c === row.class) return [c, `${c} · region ${row.region} (this one)`];
    const k = rivals.filter((r) => of.get(r).class === c).length;
    return [c, `${c} · ${k} rival ${k === 1 ? "cell" : "cells"}`];
  }));
  $("qr-map-note").textContent = `The rows of region ${row.region} (${num(row.n_cells)} ` +
    `${row.n_cells === 1 ? "cell" : "cells"}) and of its ${num(rivals.length)} rival ` +
    `${rivals.length === 1 ? "cell" : "cells"} by class, on the first two principal components of ` +
    `${num(data.points.length)} of them taken at random (at most 2,000 per cell, 20,000 in all). ${MAP_HOWTO}`;
  const map = pointMap(data, (c) => colors.get(of.get(c)?.class) ?? OTHER_CLASS, (c) => {
    const x = of.get(c);
    return x ? `cell ${c} · ${x.class} · region ${x.region}` : null;
  }, colorClass(colors), `quality-clip-${id}`);
  fill("qr-map", h("div", { class: "legend" }, classLegend(colors, labels)), map);
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

/** A defect as a chip: the warning sign and its name. @param {string} d */
function diagnosisChip(d) {
  return h("span", { class: "chip flag" }, h("b", { class: "sign", "aria-hidden": "true" }, "▲"), d);
}

/** A value, with the warning sign when it shows a defect. @param {string|Node} text @param {boolean} over */
function flagged(text, over) {
  return over ? h("span", { class: "flag", title: "a defect" },
    h("b", { class: "sign", "aria-label": "a defect" }, "▲"), text) : text;
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
