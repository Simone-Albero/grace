// Dashboard front-end: one page, one tab per stage. Charts are hand-made SVG, no dependencies.

/**
 * UI state.
 * @type {{dataset: string|null, tab: "clean"|"prepare"|"regions"|"complexity", cls: string|null, node: number|null,
 *   overlap: number|null, measure: string|null}} node: the region tree node whose map is shown (null: the root of
 *   state.cls); overlap: the overlap tree node whose map is shown (null: its root)
 */
const state = { dataset: null, tab: "clean", cls: null, node: null, overlap: null, measure: null };

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
  const [tab, dataset, cls, node, overlap] = location.hash.slice(1).split("/").map(decodeURIComponent);
  if ([...document.querySelectorAll(".tabs button")].some((b) => b.dataset.tab === tab)) state.tab = tab;
  state.dataset = opts.datasets.includes(dataset) ? dataset : opts.datasets[0];
  if (state.tab === "regions") {
    state.cls = cls || null;
    state.node = node ? Number(node) : null;
    state.overlap = overlap ? Number(overlap) : null;
  }
  $("dataset").value = state.dataset;
  $("dataset").onchange = (e) => { state.dataset = e.target.value; state.cls = state.node = state.overlap = state.measure = null; load(); };
  $("rg-class").onchange = (e) => selectClass(e.target.value);
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

/** Keep tab, dataset and, on the regions tab, class, node and overlap node in the URL hash. */
function saveHash() {
  const parts = [state.tab, state.dataset];
  if (state.tab === "regions" && state.cls != null && state.node != null) {
    parts.push(state.cls, state.node, ...(state.overlap == null ? [] : [state.overlap]));
  } else if (state.tab === "regions" && state.cls != null) parts.push(state.cls);
  history.replaceState(null, "", `#${parts.map((p) => encodeURIComponent(p)).join("/")}`);
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
    regions: `${d}/regions`,
    complexity: `${d}/complexity${state.measure ? `?measure=${encodeURIComponent(state.measure)}` : ""}`,
  }[tab];
  try {
    const data = await api(path);
    if (id !== loadId) return;
    panel.classList.remove("failed");
    empty.hidden = true;
    ({ clean: renderClean, prepare: renderPrepare, regions: renderRegions, complexity: renderComplexity })[tab](data);
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

/** Tiles, class tree and map, granularity per class, regions. @param {object} data response of /regions */
function renderRegions(data) {
  last.regions = data;
  const s = data.stats;
  const voted = data.classes.reduce((a, c) => a + c.n_voted, 0);
  const rows = data.classes.reduce((a, c) => a + c.n_rows, 0);
  fill("rg-tiles",
    tile("regions", num(s.n_regions)),
    tile("rows voted", num(voted), `${pct(voted / rows)} of ${num(rows)}`),
    tile("distance", s.distance),
    tile("graph nodes", num(s.n_graph_nodes)));
  sortable($("rg-classes"), [
    { key: "class", label: "class" },
    { key: "n_rows", label: "rows", format: num },
    { key: "n_read", label: "CSV rows", format: num },
    { key: "n_regions", label: "regions", format: num },
    { key: "n_voted", label: "voted", format: num },
    { key: "penalty", label: "penalty", format: short },
    { key: "loss", label: "loss", format: (v) => dec(v, 4) },
  ], data.classes, { onClick: (r) => selectClass(r.class), selected: (r) => r.class === state.cls });
  const classes = data.classes.map((c) => c.class);
  if (!classes.includes(state.cls)) { state.cls = classes[0]; state.node = null; }
  $("rg-class").replaceChildren(...data.classes.map((c) =>
    h("option", { value: c.class }, `${c.class} · ${c.n_regions} ${c.n_regions > 1 ? "regions" : "region"}`)));
  const sizes = data.regions.map((r) => r.n_rows);
  const edges = Array.from({ length: 21 }, (_, i) => (Math.max(...sizes, 1) * i) / 20);
  histogram($("rg-sizes"), edges, [{
    name: "regions by rows", color: "var(--accent)",
    values: edges.slice(0, -1).map((lo, i) => sizes.filter((v) => v >= lo && (v < edges[i + 1] || i === 19)).length),
  }], { format: num });
  const leafOf = new Map(data.tree.filter((n) => n.region != null).map((n) => [n.region, n.node]));
  sortable($("rg-table"), [
    { key: "region", label: "region" },
    { key: "class", label: "class" },
    { key: "n_rows", label: "rows", format: num },
    { key: "n_read", label: "CSV rows", format: num },
    { key: "hardness_mean", label: "hardness mean", format: (v) => dec(v, 3) },
    { key: "hardness_std", label: "hardness sd", format: (v) => dec(v, 3) },
  ], data.regions, {
    onClick: (r) => { selectNode(leafOf.get(r.region)); $("rg-map-title").scrollIntoView({ behavior: "smooth", block: "start" }); },
  });
  drawClass();
  drawOverlaps();
}

/** Show the tree of a class, mapped from its root. @param {string} cls */
function selectClass(cls) {
  state.cls = cls;
  state.node = null;
  drawClass();
}

/** Show a node of the region trees: its class becomes state.cls. @param {number} node */
function selectNode(node) {
  state.cls = last.regions.tree.find((n) => n.node === node).class;
  state.node = node;
  drawClass();
}

/** Tree, map and candidates of state.cls; state.node falls back to the root of the class. */
function drawClass() {
  const nodes = last.regions.tree.filter((n) => n.class === state.cls);
  if (!nodes.some((n) => n.node === state.node)) state.node = nodes.find((n) => n.parent < 0)?.node ?? null;
  $("rg-class").value = state.cls;
  saveHash();
  const colors = leafColor(nodes, state.node);
  treeDiagram($("rg-tree"), nodes, colors);
  drawCandidates(last.regions);
  drawMap(nodes, colors);
}

/** Loss of the cut of state.cls at every penalty against its regions, the chosen cut marked. @param {object} data response of /regions */
function drawCandidates(data) {
  const rows = data.candidates.filter((c) => c.class === state.cls);
  const chosen = data.classes.find((c) => c.class === state.cls);
  const isChosen = (c) => c.n_clusters === chosen?.n_regions;
  scatter($("rg-loss"), rows.map((c) => ({
    x: c.n_clusters, y: c.loss, selected: isChosen(c),
    tip: `penalty ${short(c.penalty)} → ${num(c.n_clusters)} regions, loss ${dec(c.loss, 5)}${isChosen(c) ? " (chosen)" : ""}`,
  })), { xLabel: `regions of ${state.cls}`, yLabel: "granularity loss", line: true });
}

// categorical slots for the map, in fixed order; validated on the dark surface (series-8 reads gray: left out)
const SERIES = ["var(--accent)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)",
  "var(--series-6)", "var(--series-7)"];

/** Children of every node of one class tree, in pre-order. @param {object[]} nodes @returns {Map<number, number[]>} */
function childrenOf(nodes) {
  const kids = new Map();
  for (const n of nodes) if (n.parent >= 0) kids.set(n.parent, [...(kids.get(n.parent) ?? []), n.node]);
  return kids;
}

/**
 * The colour groups of the selected node: its leaves when they fit the palette, else the fewest subtrees below it that
 * do, splitting the largest first. Groups keep pre-order and take the palette in that fixed order.
 * @param {object[]} nodes the region_tree rows of one class, in pre-order
 * @param {number} selected the node whose rows are mapped
 * @returns {{groups: {node: number, color: string}[], byRegion: Map<number, {node: number, color: string}>}}
 */
function leafColor(nodes, selected) {
  const kids = childrenOf(nodes);
  const byId = new Map(nodes.map((n) => [n.node, n]));
  const order = new Map(nodes.map((n, i) => [n.node, i]));
  let groups = [selected];
  for (;;) {
    const open = groups.filter((g) => kids.has(g)).sort((a, b) => byId.get(b).n_rows - byId.get(a).n_rows);
    if (!open.length || groups.length + 1 > SERIES.length) break;
    groups = groups.flatMap((g) => (g === open[0] ? kids.get(g) : [g]));
  }
  groups.sort((a, b) => order.get(a) - order.get(b));
  const colored = groups.map((node, i) => ({ node, color: SERIES[i] }));
  const byRegion = new Map();
  for (const g of colored) {
    const stack = [g.node];
    while (stack.length) {
      const id = stack.pop();
      const n = byId.get(id);
      if (n.region != null) byRegion.set(n.region, g);
      stack.push(...(kids.get(id) ?? []));
    }
  }
  return { groups: colored, byRegion };
}

/** Tooltip text of a tree node: only its counts (and the region of a leaf). */
function nodeTip(n) {
  return `${n.region != null ? `region ${n.region} · ` : ""}${num(n.n_rows)} rows · ${num(n.n_read)} CSV rows`;
}

/**
 * Node-link drawing of one class tree, root on top (port of bruno's treeDiagram): leaves at 0, 1, 2, ... in pre-order,
 * a parent at the mean of its children, one row per depth; squares for leaves, circles for internal nodes, size by
 * sqrt(n_rows); leaves below state.node take their map colour, the other nodes stay neutral; n_rows under a node when
 * the spacing allows, rows and CSV rows in the tooltip; state.node outlined; a click selects a node.
 * @param {HTMLElement} el container, replaced
 * @param {object[]} nodes the region_tree rows of one class, in pre-order
 * @param {{groups: object[], byRegion: Map<number, {node: number, color: string}>}} colors from leafColor
 */
function treeDiagram(el, nodes, colors) {
  const kids = childrenOf(nodes);
  const posX = new Map();
  let next = 0;
  const place = (id) => {
    const below = kids.get(id) ?? [];
    below.forEach(place);
    posX.set(id, below.length ? below.reduce((a, c) => a + posX.get(c), 0) / below.length : next++);
  };
  nodes.filter((n) => n.parent < 0).forEach((n) => place(n.node));
  const depth = Math.max(...nodes.map((n) => n.depth));
  const step = Math.max(14, 1100 / Math.max(next, 1)); // room per leaf; wider trees scroll
  const W = Math.max(1100, step * next), m = 30;
  const rowH = Math.max(22, Math.min(64, 640 / Math.max(depth, 1)));
  const H = depth * rowH + 2 * m + 14;
  const x = (id) => (next > 1 ? m + (posX.get(id) / (next - 1)) * (W - 2 * m) : W / 2);
  const y = (d) => m + d * rowH;
  const byId = new Map(nodes.map((n) => [n.node, n]));
  const biggest = Math.max(...nodes.map((n) => n.n_rows), 1);
  const labelled = step >= 44;
  const groupOf = new Map(colors.groups.map((g) => [g.node, g.color]));
  const edges = nodes.filter((n) => byId.has(n.parent)).map((n) =>
    sv("line", { class: "edge", x1: x(n.parent), y1: y(byId.get(n.parent).depth), x2: x(n.node), y2: y(n.depth) }));
  const marks = nodes.map((n) => {
    const size = Math.min(step * 0.9, 6 + 18 * Math.sqrt(n.n_rows / biggest));
    const picked = n.node === state.node;
    const fill = n.region != null ? colors.byRegion.get(n.region)?.color ?? "var(--axis)" : "var(--surface)";
    const ring = picked ? "var(--text)" : groupOf.get(n.node) ?? (n.region != null ? "var(--surface)" : "var(--muted)");
    const style = `fill:${fill};stroke:${ring};stroke-width:${picked ? 2.5 : 1.5}`;
    const mark = n.region != null
      ? sv("rect", { x: x(n.node) - size / 2, y: y(n.depth) - size / 2, width: size, height: size, rx: 2, style })
      : sv("circle", { cx: x(n.node), cy: y(n.depth), r: size / 2, style });
    const label = labelled || picked || groupOf.has(n.node)
      ? sv("text", { x: x(n.node), y: y(n.depth) + size / 2 + 12, "text-anchor": "middle" }, num(n.n_rows))
      : null;
    const g = sv("g", { class: "node" }, mark, label);
    g.onmousemove = (ev) => tip(ev, nodeTip(n));
    g.onmouseleave = () => tip(null);
    g.onclick = () => selectNode(n.node);
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
 * @param {object[]} nodes the region_tree rows of state.cls
 * @param {{groups: object[], byRegion: Map<number, {node: number, color: string}>}} colors from leafColor
 */
async function drawMap(nodes, colors) {
  const id = ++mapId;
  const el = $("rg-map");
  const sel = nodes.find((n) => n.node === state.node);
  const what = sel.parent < 0 ? `class ${sel.class}` : sel.region != null ? `region ${sel.region}` : `node ${sel.node}`;
  $("rg-map-title").textContent = `Map of ${what}`;
  let data;
  try {
    data = await api(`/${encodeURIComponent(state.dataset)}/regions/map?node=${sel.node}`);
  } catch (e) {
    if (id === mapId) { el.replaceChildren(h("p", { class: "note" }, e.message)); $("rg-map-note").textContent = ""; }
    return;
  }
  if (id !== mapId) return;
  const byId = new Map(nodes.map((n) => [n.node, n]));
  const regionNode = new Map(nodes.filter((n) => n.region != null).map((n) => [n.region, n]));
  const leaves = nodes.filter((n) => colors.byRegion.has(n.region)).length;
  $("rg-map-note").textContent = `${num(data.points.length)} random rows of ${what} (at most 2,000 per region, ` +
    `20,000 in all), on the first two principal components of these rows, in the space of the regions. ` +
    (colors.groups.length < leaves
      ? `Its ${num(leaves)} regions are coloured by the ${colors.groups.length} subtrees ringed in the tree; click one to go deeper. `
      : "Each region has its colour. ") +
    "Hover a point for its region; drag to pan, pinch or ctrl + scroll to zoom, double click to reset.";
  const legend = h("div", { class: "legend" }, colors.groups.map((g) => {
    const n = byId.get(g.node);
    const name = n.region != null ? `region ${n.region}` : `node ${n.node}`;
    return h("span", {}, h("i", { style: `background:${g.color}` }), `${name} · ${num(n.n_rows)} rows`);
  }));
  const svg = pointMap(data, (r) => colors.byRegion.get(r)?.color ?? "var(--muted)", (r) => {
    const n = regionNode.get(r);
    return n ? nodeTip(n) : null;
  }, `map-clip-${id}`);
  el.replaceChildren(legend, h("div", { class: "table-wrap" }, svg));
}

/**
 * The colour of every class, in the fixed order of the classes; past the palette the classes share the muted "other".
 * @param {string[]} classes in class order
 * @returns {Map<string, string>}
 */
function classColor(classes) {
  return new Map(classes.map((c, i) => [c, i < SERIES.length ? SERIES[i] : "var(--muted)"]));
}

/** Show a node of the overlap tree: its tree, map, branches and contacts. @param {number} node */
function selectOverlap(node) {
  state.overlap = node;
  drawOverlaps();
}

/** Overlap tree, the map and the class branches of state.overlap (null: the root), and the contacts table. */
function drawOverlaps() {
  const data = last.regions;
  const nodes = data.overlap_tree;
  if (!nodes.length) {
    fill("rg-ov-tree", h("p", { class: "note" }, "This regions run has no overlap tree: run regions again."));
    for (const id of ["rg-ov-branches", "rg-ov-map", "rg-overlaps"]) fill(id);
    $("rg-ov-note").textContent = "";
    return;
  }
  const regionClass = new Map(data.regions.map((r) => [r.region, r.class]));
  const colors = classColor(data.classes.map((c) => c.class));
  if (!nodes.some((n) => n.node === state.overlap)) state.overlap = nodes.find((n) => n.parent < 0).node;
  saveHash();
  overlapDiagram($("rg-ov-tree"), nodes, colors, regionClass);
  const sel = nodes.find((n) => n.node === state.overlap);
  fill("rg-ov-branches", branchesOf(data.tree, regionsBelow(nodes, sel.node), regionClass).map((b) => {
    const where = b.node.parent < 0 ? "the root" : b.node.region != null ? `region ${b.node.region}` : `node ${b.node.node}`;
    const chip = h("button", { class: "chip", type: "button", title: "show it in the tree of its class" },
      h("i", { style: `background:${colors.get(b.class)}` }),
      `${b.class}: ${where} · ${b.n_regions} of its ${b.n_below} ${b.n_below > 1 ? "regions" : "region"}`);
    chip.onclick = () => { selectNode(b.node.node); $("rg-tree").scrollIntoView({ behavior: "smooth", block: "start" }); };
    return chip;
  }));
  drawOverlapMap(sel, colors, regionClass);
  const ratio = (v) => (v == null ? "∞" : dec(v, 2));
  const classNode = new Map(data.tree.map((n) => [n.node, n]));
  const branch = (v) => (classNode.get(v)?.region != null ? `region ${classNode.get(v).region}` : `node ${v}`);
  sortable($("rg-overlaps"), [
    { key: "node", label: "node" },
    { key: "class_a", label: "class a" },
    { key: "branch_a", label: "branch a", format: branch },
    { key: "n_rows_a", label: "rows a", format: num },
    { key: "class_b", label: "class b" },
    { key: "branch_b", label: "branch b", format: branch },
    { key: "n_rows_b", label: "rows b", format: num },
    { key: "distance", label: "distance", format: (v) => dec(v, 3) },
    { key: "ratio", label: "ratio", format: ratio },
    { key: "height", label: "height", format: short },
  ], [...data.overlaps].sort((a, b) => (a.ratio ?? Infinity) - (b.ratio ?? Infinity)), {
    onClick: (r) => { selectOverlap(r.node); $("rg-ov-tree").scrollIntoView({ behavior: "smooth", block: "start" }); },
    selected: (r) => r.node === state.overlap,
  });
}

/** The regions of the leaves below a node of a tree (region_tree or overlap_tree rows). @returns {number[]} */
function regionsBelow(nodes, id) {
  const kids = childrenOf(nodes);
  const byId = new Map(nodes.map((n) => [n.node, n]));
  const regions = [], stack = [id];
  while (stack.length) {
    const v = stack.pop();
    if (byId.get(v).region != null) regions.push(byId.get(v).region);
    stack.push(...(kids.get(v) ?? []));
  }
  return regions;
}

/**
 * Dendrogram of the overlap tree, leaves at the bottom in pre-order, a node at the log of its height; a branch takes
 * its class colour while it holds one class, neutral above; a dot at every node where classes meet, sized by
 * sqrt(n_rows); state.overlap outlined; hover for counts, classes, height and ratio; a click selects a node.
 * @param {HTMLElement} el container, replaced
 * @param {object[]} nodes the overlap_tree rows, in pre-order
 * @param {Map<string, string>} colors from classColor
 * @param {Map<number, string>} regionClass the class of every region
 */
function overlapDiagram(el, nodes, colors, regionClass) {
  const kids = childrenOf(nodes);
  const byId = new Map(nodes.map((n) => [n.node, n]));
  const posX = new Map(), held = new Map(), rows = new Map();
  let next = 0;
  for (const n of nodes) if (n.region != null) posX.set(n.node, next++); // pre-order: leaves left to right
  for (const n of [...nodes].reverse()) {
    const below = kids.get(n.node) ?? [];
    if (!below.length) {
      held.set(n.node, new Set([n.class]));
      rows.set(n.node, new Map([[n.class, n.n_rows]]));
      continue;
    }
    posX.set(n.node, below.reduce((a, c) => a + posX.get(c), 0) / below.length);
    held.set(n.node, new Set(below.flatMap((c) => [...held.get(c)])));
    const count = new Map();
    for (const c of below) for (const [k, v] of rows.get(c)) count.set(k, (count.get(k) ?? 0) + v);
    rows.set(n.node, count);
  }
  // where two classes meet: each side holds a class the other does not
  const meets = (n) => {
    const [l, r] = (kids.get(n.node) ?? []).map((c) => held.get(c));
    return l && [...l].some((k) => !r.has(k)) && [...r].some((k) => !l.has(k));
  };
  const heights = nodes.filter((n) => n.region == null).map((n) => n.height).filter((v) => v > 0);
  const lo = heights.length ? Math.log10(Math.min(...heights)) : 0;
  const hi = heights.length ? Math.log10(Math.max(...heights)) : 1;
  const step = Math.max(8, 1100 / Math.max(next, 1)); // room per leaf; wider trees scroll
  const m = { l: 56, r: 16, t: 16, b: 24 };
  const W = Math.max(1100, m.l + m.r + step * next), H = 440;
  const x = (id) => (next > 1 ? m.l + 8 + (posX.get(id) / (next - 1)) * (W - m.l - m.r - 16) : W / 2);
  const base = H - m.b; // the leaves
  const top = m.t, bottom = base - 18; // the merges, log of their height
  const y = (n) => (n.region != null ? base
    : hi > lo ? bottom - ((Math.log10(Math.max(n.height, 10 ** lo)) - lo) / (hi - lo)) * (bottom - top) : (top + bottom) / 2);
  const color = (n) => (n.class != null ? colors.get(n.class) ?? "var(--muted)" : "var(--muted)");
  const grid = [];
  for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) {
    const v = bottom - ((e - lo) / (hi - lo || 1)) * (bottom - top);
    grid.push(sv("line", { class: "grid", x1: m.l, x2: W - m.r, y1: v, y2: v }),
      sv("text", { x: m.l - 8, y: v + 4, "text-anchor": "end" }, short(10 ** e)));
  }
  grid.push(sv("text", { transform: `translate(14, ${(top + bottom) / 2}) rotate(-90)`, "text-anchor": "middle" },
    "merge height (log)"));
  const edges = nodes.filter((n) => n.parent >= 0).map((n) => {
    const up = byId.get(n.parent);
    return sv("path", {
      d: `M${x(n.node)},${y(n)}V${y(up)}H${x(up.node)}`, fill: "none",
      style: `stroke:${color(n)};stroke-width:${n.class != null ? 1.5 : 1.25}`,
    });
  });
  const biggest = Math.max(...nodes.map((n) => n.n_rows), 1);
  const nodeTipOf = (n) => {
    const mix = [...rows.get(n.node)].sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${num(v)}`).join(", ");
    if (n.region != null) return `region ${n.region} · ${n.class} · ${num(n.n_rows)} rows · ${num(n.n_read)} CSV rows`;
    const ratio = n.ratio == null ? "∞" : dec(n.ratio, 2);
    return `node ${n.node} · ${mix} rows · height ${short(n.height)} · sides ${dec(n.distance, 3)} apart, ratio ${ratio}`;
  };
  const marks = nodes.map((n) => {
    const picked = n.node === state.overlap;
    const leaf = n.region != null, meet = !leaf && meets(n);
    const r = leaf ? 3 : meet ? 3 + 5 * Math.sqrt(n.n_rows / biggest) : n.class == null ? 2.5 : 0;
    const mark = leaf
      ? sv("rect", { x: x(n.node) - 3, y: base - 3, width: 6, height: 6, rx: 1, style: `fill:${color(n)}` })
      : r ? sv("circle", { cx: x(n.node), cy: y(n), r, style: `fill:${meet ? "var(--text-2)" : "var(--muted)"};stroke:var(--surface);stroke-width:1.5` }) : null;
    const ring = picked ? sv("circle", { cx: x(n.node), cy: y(n), r: Math.max(r, 3) + 4, fill: "none", style: "stroke:var(--text);stroke-width:2" }) : null;
    const hit = sv("circle", { cx: x(n.node), cy: y(n), r: Math.max(r + 3, step / 2, 6), fill: "transparent" });
    const g = sv("g", { class: "node" }, mark, ring, hit);
    g.onmousemove = (ev) => tip(ev, nodeTipOf(n));
    g.onmouseleave = () => tip(null);
    g.onclick = () => selectOverlap(n.node);
    return g;
  });
  const present = [...colors].filter(([c]) => [...regionClass.values()].includes(c));
  const legend = h("div", { class: "legend" }, present.map(([c, col]) => h("span", {}, h("i", { style: `background:${col}` }), c)),
    h("span", {}, h("i", { style: "background:var(--text-2);border-radius:50%" }), "classes meet"));
  const wrap = h("div", { class: "table-wrap" },
    sv("svg", { class: "tree", viewBox: `0 0 ${W} ${H}`, width: W, role: "img", style: "max-width:none" }, grid, edges, marks));
  el.replaceChildren(legend, wrap);
  if (byId.has(state.overlap)) wrap.scrollLeft = x(state.overlap) - wrap.clientWidth / 2; // the selected node in view
}

/**
 * The branch of each class tree below an overlap node: per class, the lowest region_tree node above all the regions of
 * the class under it.
 * @param {object[]} tree the region_tree rows
 * @param {number[]} regions the regions under the overlap node
 * @param {Map<number, string>} regionClass the class of every region
 * @returns {{class: string, node: object, n_regions: number}[]} in class order
 */
function branchesOf(tree, regions, regionClass) {
  const byId = new Map(tree.map((n) => [n.node, n]));
  const leafOf = new Map(tree.filter((n) => n.region != null).map((n) => [n.region, n.node]));
  const above = (id) => { const path = []; for (let v = id; v >= 0; v = byId.get(v).parent) path.push(v); return path; };
  const byClass = new Map();
  for (const r of regions) byClass.set(regionClass.get(r), [...(byClass.get(regionClass.get(r)) ?? []), r]);
  const order = new Map([...new Set(tree.map((n) => n.class))].map((c, i) => [c, i]));
  return [...byClass].map(([cls, rs]) => {
    const path = above(leafOf.get(rs[0]));
    let common = new Set(path);
    for (const r of rs.slice(1)) { const a = new Set(above(leafOf.get(r))); common = new Set([...common].filter((v) => a.has(v))); }
    const node = byId.get(path.find((v) => common.has(v)));
    return { class: cls, node, n_regions: rs.length, n_below: regionsBelow(tree, node.node).length };
  }).sort((a, b) => order.get(a.class) - order.get(b.class));
}

/**
 * Fetch the map of state.overlap and draw it, its rows coloured by class (as drawMap otherwise).
 * @param {object} sel the overlap_tree row of state.overlap
 * @param {Map<string, string>} colors from classColor
 * @param {Map<number, string>} regionClass the class of every region
 */
async function drawOverlapMap(sel, colors, regionClass) {
  const id = ++overlapMapId;
  const el = $("rg-ov-map");
  const what = sel.parent < 0 ? "every class" : sel.region != null ? `region ${sel.region}` : `node ${sel.node}`;
  $("rg-ov-title").textContent = `Map of ${what}`;
  let data;
  try {
    data = await api(`/${encodeURIComponent(state.dataset)}/regions/overlap_map?node=${sel.node}`);
  } catch (e) {
    if (id === overlapMapId) { el.replaceChildren(h("p", { class: "note" }, e.message)); $("rg-ov-note").textContent = ""; }
    return;
  }
  if (id !== overlapMapId) return;
  const regionRow = new Map(last.regions.regions.map((r) => [r.region, r]));
  const rows = new Map();
  for (const r of regionsBelow(last.regions.overlap_tree, sel.node)) {
    rows.set(regionClass.get(r), (rows.get(regionClass.get(r)) ?? 0) + regionRow.get(r).n_rows);
  }
  $("rg-ov-note").textContent = `${num(data.points.length)} random rows of ${what} (at most 2,000 per region, ` +
    "20,000 in all), on the first two principal components of these rows, in the space of the regions, coloured by " +
    "class. Hover a point for its region; drag to pan, pinch or ctrl + scroll to zoom, double click to reset.";
  const legend = h("div", { class: "legend" }, [...colors].filter(([c]) => rows.has(c)).map(([c, col]) =>
    h("span", {}, h("i", { style: `background:${col}` }), `${c} · ${num(rows.get(c))} rows`)));
  const svg = pointMap(data, (r) => colors.get(regionClass.get(r)) ?? "var(--muted)", (r) => {
    const row = regionRow.get(r);
    return row ? `region ${r} · ${row.class} · ${num(row.n_rows)} rows · ${num(row.n_read)} CSV rows` : null;
  }, `overlap-clip-${id}`);
  el.replaceChildren(legend, h("div", { class: "table-wrap" }, svg));
}

let overlapMapId = 0; // drops overlap map answers of superseded requests

/**
 * Sampled rows in their PCA plane, one scale for both axes framed on the 1st to 99th percentiles, navigable with
 * panZoom (the plot of drawMap).
 * @param {{shares: number[], points: number[][]}} data response of a map endpoint
 * @param {(region: number) => string} colorOf fill of a region's points
 * @param {(region: number) => string|null} tipOf tooltip of a region's points
 * @param {string} clip a unique id for the clip path
 * @returns {SVGSVGElement}
 */
function pointMap(data, colorOf, tipOf, clip) {
  const pts = data.points.map(([a, b, region]) => ({ a, b, region }));
  const frame = (vals) => {
    const lo = percentile(vals, 1), hi = percentile(vals, 99), pad = 0.2 * (hi - lo || 1);
    return [lo - pad, hi + pad];
  };
  const [xa, xb] = frame(pts.map((p) => p.a)), [ya, yb] = frame(pts.map((p) => p.b));
  const W = 1100, H = 520, m = { l: 48, r: 12, t: 12, b: 36 };
  const k = Math.min((W - m.l - m.r) / (xb - xa), (H - m.t - m.b) / (yb - ya)); // one scale for both axes
  const cx = (W - m.l - m.r - k * (xb - xa)) / 2 + m.l, cy = (H - m.t - m.b - k * (yb - ya)) / 2 + m.t;
  const X = (v) => cx + (v - xa) * k, Y = (v) => cy + (yb - v) * k;
  const dots = pts.map((p) => sv("circle", {
    r: 2.5, "data-region": p.region, style: `fill:${colorOf(p.region)};opacity:0.8`,
  }));
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", class: "zoomable", style: "min-width:640px" },
    sv("defs", {}, sv("clipPath", { id: clip }, sv("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b }))),
    sv("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "none", style: "stroke:var(--grid)" }),
    sv("g", { "clip-path": `url(#${clip})` }, dots),
    sv("text", { x: (W + m.l) / 2, y: H - 10, "text-anchor": "middle" }, `PC 1 · ${pct(data.shares[0], 0)} of the variance`),
    sv("text", { transform: `translate(16, ${(H - m.b + m.t) / 2}) rotate(-90)`, "text-anchor": "middle" },
      `PC 2 · ${pct(data.shares[1], 0)} of the variance`));
  svg.onmousemove = (ev) => {
    const region = ev.target.getAttribute?.("data-region");
    tip(ev, region == null ? null : tipOf(Number(region)));
  };
  svg.onmouseleave = () => tip(null);
  const bx = pts.map((p) => X(p.a)), by = pts.map((p) => Y(p.b));
  panZoom(svg, (t) => dots.forEach((d, i) => {
    d.setAttribute("cx", (t.k * bx[i] + t.x).toFixed(1));
    d.setAttribute("cy", (t.k * by[i] + t.y).toFixed(1));
  }));
  return svg;
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
