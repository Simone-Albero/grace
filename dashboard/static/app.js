// Dashboard front-end: one page, one tab per stage. Charts are hand-made SVG, no dependencies.

/**
 * UI state.
 * @type {{dataset: string|null, tab: "clean"|"prepare"|"regions"|"quality", cls: string|null, node: number|null,
 *   overlap: number|null, view: "classes"|"regions", qclass: string|null, qfilter: string|null,
 *   region: number|null}} node: the region tree node whose map is shown (null: the root of state.cls); overlap: the
 *   overlap zones node whose map is shown (null: the root of the first zone); view: the quality panel shown; qclass,
 *   qfilter: the filters of the quality regions (null: all), qfilter "solid", "flagged" or one of ISSUES; region: the
 *   quality region whose detail is shown (null: none)
 */
const state = {
  dataset: null, tab: "clean", cls: null, node: null, overlap: null,
  view: "regions", qclass: null, qfilter: null, region: null,
};

const last = { regions: null, quality: null }; // last responses, for the controls that only redraw
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
    state.cls = state.node = state.overlap = state.qclass = state.qfilter = state.region = null;
    load();
  };
  $("rg-class").onchange = (e) => selectClass(e.target.value);
  $("qr-class").onchange = (e) => { state.qclass = e.target.value || null; drawRegionList(); };
  for (const b of document.querySelectorAll(".view-switch button")) b.onclick = () => showView(b.dataset.view, true);
  for (const b of document.querySelectorAll(".tabs button")) b.onclick = () => showTab(b.dataset.tab);
  showTab(state.tab);
}

let shownHash = ""; // the hash last written by saveHash: a hashchange to it is our own

/**
 * Set state from the URL hash: tab, dataset, and the regions tab's class, node and overlap or the quality tab's view and region.
 * @param {string[]} datasets the known datasets; an unknown one falls back to the first
 */
function readHash(datasets) {
  const [tab, dataset, cls, node, overlap] = location.hash.slice(1).split("/").map(decodeURIComponent);
  if ([...document.querySelectorAll(".tabs button")].some((b) => b.dataset.tab === tab)) state.tab = tab;
  const next = datasets.includes(dataset) ? dataset : datasets[0];
  if (next !== state.dataset) state.qclass = state.qfilter = null;
  state.dataset = next;
  if (tab === "quality") {
    state.view = cls === "classes" ? "classes" : "regions";
    state.region = node ? Number(node) : null;
  }
  if (tab === "regions") {
    state.cls = cls || null;
    state.node = node ? Number(node) : null;
    state.overlap = overlap ? Number(overlap) : null;
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
 * Keep tab, dataset and, on the regions tab, class, node and overlap node, on the quality tab view and region, in the URL hash.
 * @param {boolean} [push] a step the back button returns from (opening a region, switching panel); else the entry is replaced
 */
function saveHash(push = false) {
  const parts = [state.tab, state.dataset];
  if (state.tab === "quality") parts.push(state.view, ...(state.view === "regions" && state.region != null ? [state.region] : []));
  if (state.tab === "regions" && state.cls != null && state.node != null) {
    parts.push(state.cls, state.node, ...(state.overlap == null ? [] : [state.overlap]));
  } else if (state.tab === "regions" && state.cls != null) parts.push(state.cls);
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
    regions: `${d}/regions`,
    quality: `${d}/quality`,
  }[tab];
  try {
    const data = await api(path);
    if (id !== loadId) return;
    panel.classList.remove("failed");
    empty.hidden = true;
    ({ clean: renderClean, prepare: renderPrepare, regions: renderRegions, quality: renderQuality })[tab](data);
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

// a region found again on other halves at Jaccard >= STABLE is stable, under DOUBTFUL not to be trusted (Hennig 2007)
const STABLE = 0.75;
const DOUBTFUL = 0.6;

/** The stability of a region, marked when doubtful or unreliable. @param {number|null} v */
function stability(v) {
  if (v == null) return "–";
  const text = dec(v, 2);
  if (v >= STABLE) return text;
  const word = v >= DOUBTFUL ? "doubtful" : "unreliable";
  return h("span", { class: "flag", title: `${word}: found again on other halves at Jaccard ${text}` },
    h("b", { class: "sign", "aria-label": word }, "▲"), `${text} ${word}`);
}

/** Tiles, class tree and map, granularity per class, regions. @param {object} data response of /regions */
function renderRegions(data) {
  last.regions = data;
  const s = data.stats;
  const voted = data.classes.reduce((a, c) => a + c.n_voted, 0);
  const rows = data.classes.reduce((a, c) => a + c.n_rows, 0);
  const measured = data.regions.filter((r) => r.stability != null);
  const shaky = measured.filter((r) => r.stability < STABLE);
  const lost = shaky.filter((r) => r.stability < DOUBTFUL);
  fill("rg-tiles",
    tile("regions", num(s.n_regions)),
    tile("rows voted", num(voted), `${pct(voted / rows)} of ${num(rows)}`),
    tile("split separation", short(s.split_separation), "sd along the line, less 2 standard errors"),
    ...(measured.length ? [tile("regions not stable", num(shaky.length),
      `${num(lost.length)} unreliable (Jaccard < ${DOUBTFUL}), ${num(shaky.length - lost.length)} doubtful`)] : []),
    tile("distance", s.distance),
    ...(s.n_zones == null ? [] : [tile("overlap zones", num(s.n_zones),
      `${num(data.overlaps.length)} ${data.overlaps.length === 1 ? "pair" : "pairs"} of regions under ${s.overlap_separation} sd`)]));
  sortable($("rg-classes"), [
    { key: "class", label: "class" },
    { key: "n_rows", label: "rows", format: num },
    { key: "n_read", label: "CSV rows", format: num },
    { key: "n_regions", label: "regions", format: num },
    { key: "n_pieces", label: "pieces cut", format: num },
    { key: "n_voted", label: "voted", format: num },
    { key: "split_separation", label: "sewn at",
      format: (v) => flagged(short(v), v > s.split_separation) },
    { key: "stability", label: "stability", format: stability },
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
    { key: "stability", label: "stability", format: stability },
  ], data.regions, {
    onClick: (r) => { selectNode(leafOf.get(r.region)); scrollUnderHeader($("rg-map-title")); },
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
  drawCurve(last.regions);
  drawMap(nodes, colors);
}

/** The regions of state.cls at every threshold of its curve, the one used marked. @param {object} data response of /regions */
function drawCurve(data) {
  const rows = data.candidates.filter((c) => c.class === state.cls);
  const used = data.stats.split_separation;
  scatter($("rg-curve"), rows.map((c) => ({
    x: c.threshold, y: c.n_regions, selected: c.threshold === used,
    tip: `separation ${short(c.threshold)} → ${num(c.n_regions)} ${c.n_regions === 1 ? "region" : "regions"}${c.threshold === used ? " (used)" : ""}`,
  })), { xLabel: `split separation (${state.cls})`, yLabel: "regions", line: true });
}

// categorical slots, in fixed order; validated on the dark surface: every adjacent pair passes (worst CVD ΔE 8.4),
// not every pair (orange and red, violet and blue lie close), so legends and tooltips always name the class
const SERIES = ["var(--accent)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)",
  "var(--series-6)", "var(--series-7)", "var(--series-8)"];

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
  $("rg-map-note").textContent = `The rows of ${what} on the first two principal components of ` +
    `${num(data.points.length)} of them taken at random (at most 2,000 per region, 20,000 in all), in the space of the ` +
    "regions. " +
    (colors.groups.length < leaves
      ? `Its ${num(leaves)} regions are coloured by the ${colors.groups.length} subtrees ringed in the tree; click one to go deeper. `
      : "Each region has its colour. ") +
    MAP_HOWTO;
  const legend = h("div", { class: "legend" }, colors.groups.map((g) => {
    const n = byId.get(g.node);
    const name = n.region != null ? `region ${n.region}` : `node ${n.node}`;
    return h("span", {}, h("i", { style: `background:${g.color}` }), `${name} · ${num(n.n_rows)} rows`);
  }));
  const names = new Map(colors.groups.map((g) => {
    const n = byId.get(g.node);
    return [g.color, n.region != null ? `region ${n.region}` : `node ${n.node}`];
  }));
  const map = pointMap(data, (r) => colors.byRegion.get(r)?.color ?? "var(--muted)", (r) => {
    const n = regionNode.get(r);
    return n ? nodeTip(n) : null;
  }, (color) => names.get(color) ?? "other regions", `map-clip-${id}`);
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
  "its region it stands for); the strips show its rows along each axis, scaled to its own peak. Hover the plot for " +
  "the curves you are inside, or a point for its region; drag to pan, pinch or ctrl + scroll to zoom, double click " +
  "to reset.";

/** Show a node of the overlap zones: its zone, map, branches and pairs. @param {number} node */
function selectOverlap(node) {
  state.overlap = node;
  drawOverlaps();
}

/** The overlap zones, the map and the class branches of state.overlap (null: the first zone), and the pairs table. */
function drawOverlaps() {
  const data = last.regions;
  const nodes = data.zones, tau = data.stats.overlap_separation;
  const clear = (message) => {
    fill("rg-ov-tree", h("p", { class: "note" }, message));
    for (const id of ["rg-ov-branches", "rg-ov-map", "rg-overlaps"]) fill(id);
    $("rg-ov-title").textContent = "Map";
    $("rg-ov-note").textContent = "";
  };
  if (tau == null) return clear("This regions run has no overlap zones: run regions again.");
  if (!nodes.length) {
    return clear(`No two regions of different classes lie closer than ${tau} sd along the line between their centroids: ` +
      "no class overlaps another.");
  }
  const regionClass = new Map(data.regions.map((r) => [r.region, r.class]));
  const inZones = new Set(nodes.filter((n) => n.region != null).map((n) => n.class));
  const colors = classColor(data.classes.map((c) => c.class).filter((c) => inZones.has(c)));
  if (!nodes.some((n) => n.node === state.overlap)) state.overlap = nodes.find((n) => n.parent < 0).node;
  saveHash();
  zoneDiagram($("rg-ov-tree"), nodes, tau, colors, regionClass);
  const sel = nodes.find((n) => n.node === state.overlap);
  fill("rg-ov-branches", branchesOf(data.tree, regionsBelow(nodes, sel.node), regionClass).map((b) => {
    const where = b.node.parent < 0 ? "the root" : b.node.region != null ? `region ${b.node.region}` : `node ${b.node.node}`;
    const chip = h("button", { class: "chip", type: "button", title: "show it in the tree of its class" },
      h("i", { style: `background:${colors.get(b.class)}` }),
      `${b.class}: ${where} · ${b.n_regions} of its ${b.n_below} ${b.n_below > 1 ? "regions" : "region"}`);
    chip.onclick = () => { selectNode(b.node.node); scrollUnderHeader($("rg-tree")); };
    return chip;
  }));
  drawOverlapMap(sel, colors, regionClass);
  const meet = new Map(data.overlaps.map((r) => [r, meetOf(nodes, r.region_a, r.region_b)]));
  sortable($("rg-overlaps"), [
    { key: "zone", label: "zone" },
    { key: "region_a", label: "region a" },
    { key: "class_a", label: "class a" },
    { key: "region_b", label: "region b" },
    { key: "class_b", label: "class b" },
    { key: "distance", label: "distance", format: (v) => dec(v, 3) },
    { key: "spread_a", label: "sd a", format: (v) => dec(v, 3) },
    { key: "spread_b", label: "sd b", format: (v) => dec(v, 3) },
    { key: "separation", label: "separation", format: (v) => dec(v, 2) },
  ], [...data.overlaps].sort((a, b) => a.separation - b.separation), {
    onClick: (r) => { selectOverlap(meet.get(r)); scrollUnderHeader($("rg-ov-tree")); },
    selected: (r) => meet.get(r) === state.overlap,
  });
}

/**
 * The lowest node of the overlap zones above two regions of one zone.
 * @param {object[]} nodes the overlap_zones rows, zone by zone in pre-order
 * @param {number} a a region
 * @param {number} b a region of the zone of a
 * @returns {number} the node id
 */
function meetOf(nodes, a, b) {
  const byId = new Map(nodes.map((n) => [n.node, n]));
  const leafOf = new Map(nodes.filter((n) => n.region != null).map((n) => [n.region, n.node]));
  const above = new Set();
  for (let v = leafOf.get(a); v >= 0; v = byId.get(v).parent) above.add(v);
  let v = leafOf.get(b);
  while (!above.has(v)) v = byId.get(v).parent;
  return v;
}

/** The regions of the leaves below a node of a tree (region_tree or overlap_zones rows). @returns {number[]} */
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
 * Dendrogram of the overlap zones, side by side in zone order with a gap between them, leaves at the bottom in
 * pre-order and a merge at its separation (linear, 0 at the leaves, the threshold dashed at the top); leaves take the
 * colour of their class, branches are neutral; a dot at every merge sized by sqrt(n_rows); state.overlap outlined; hover
 * for counts, classes, separation and the pair that made it; a click selects a node.
 * @param {HTMLElement} el container, replaced
 * @param {object[]} nodes the overlap_zones rows, zone by zone in pre-order
 * @param {number} tau the overlap_separation of the run
 * @param {Map<string, string>} colors from classColor
 * @param {Map<number, string>} regionClass the class of every region
 */
function zoneDiagram(el, nodes, tau, colors, regionClass) {
  const kids = childrenOf(nodes);
  const byId = new Map(nodes.map((n) => [n.node, n]));
  const posX = new Map(), rows = new Map();
  let next = 0;
  for (const n of nodes) {
    if (n.parent < 0 && n.node !== nodes[0].node) next += 1; // a free slot between zones
    if (n.region != null) posX.set(n.node, next++); // pre-order: leaves left to right
  }
  for (const n of [...nodes].reverse()) {
    const below = kids.get(n.node) ?? [];
    if (!below.length) { rows.set(n.node, new Map([[n.class, n.n_rows]])); continue; }
    posX.set(n.node, below.reduce((a, c) => a + posX.get(c), 0) / below.length);
    const count = new Map();
    for (const c of below) for (const [k, v] of rows.get(c)) count.set(k, (count.get(k) ?? 0) + v);
    rows.set(n.node, count);
  }
  const step = Math.max(14, 1100 / Math.max(next, 1)); // room per slot; wider forests scroll
  const m = { l: 56, r: 16, t: 20, b: 40 };
  const W = Math.max(1100, m.l + m.r + step * next), H = 380;
  const x = (id) => (next > 1 ? m.l + 8 + (posX.get(id) / (next - 1)) * (W - m.l - m.r - 16) : W / 2);
  const base = H - m.b; // the leaves
  const top = m.t, bottom = base - 18; // the merges, 0 to tau
  const yOf = (v) => bottom - (v / tau) * (bottom - top);
  const y = (n) => (n.region != null ? base : yOf(n.separation));
  const grid = [0, 0.25, 0.5, 0.75].map((f) => {
    const v = yOf(f * tau);
    return [sv("line", { class: "grid", x1: m.l, x2: W - m.r, y1: v, y2: v }),
      sv("text", { x: m.l - 8, y: v + 4, "text-anchor": "end" }, dec(f * tau, 2))];
  }).flat();
  grid.push(sv("line", { x1: m.l, x2: W - m.r, y1: top, y2: top, style: "stroke:var(--text-2);stroke-dasharray:4 4" }),
    sv("text", { x: m.l - 8, y: top + 4, "text-anchor": "end" }, dec(tau, 2)),
    sv("text", { x: m.l + 4, y: top - 6 }, `overlap_separation ${tau}: nothing joins above`),
    sv("text", { transform: `translate(14, ${(top + bottom) / 2}) rotate(-90)`, "text-anchor": "middle" },
      "separation (sd)"));
  const zoneLabels = nodes.filter((n) => n.parent < 0).map((n) =>
    sv("text", { x: x(n.node), y: H - 8, "text-anchor": "middle" }, `zone ${n.zone}`));
  const color = (n) => (n.region != null ? colors.get(n.class) ?? "var(--muted)" : "var(--muted)");
  const edges = nodes.filter((n) => n.parent >= 0).map((n) => {
    const up = byId.get(n.parent);
    return sv("path", {
      d: `M${x(n.node)},${y(n)}V${y(up)}H${x(up.node)}`, fill: "none",
      style: `stroke:${color(n)};stroke-width:${n.region != null ? 1.5 : 1.25}`,
    });
  });
  const biggest = Math.max(...nodes.map((n) => n.n_rows), 1);
  const side = (r) => `region ${r} (${regionClass.get(r)})`;
  const nodeTipOf = (n) => {
    if (n.region != null) return `region ${n.region} · ${n.class} · ${num(n.n_rows)} rows · ${num(n.n_read)} CSV rows`;
    const mix = [...rows.get(n.node)].sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${num(v)}`).join(", ");
    return `zone ${n.zone} · node ${n.node} · ${mix} rows · ${side(n.region_a)} and ${side(n.region_b)} ` +
      `${dec(n.separation, 2)} sd apart`;
  };
  const marks = nodes.map((n) => {
    const picked = n.node === state.overlap;
    const leaf = n.region != null;
    const r = leaf ? 3 : 3 + 5 * Math.sqrt(n.n_rows / biggest);
    const mark = leaf
      ? sv("rect", { x: x(n.node) - 3, y: base - 3, width: 6, height: 6, rx: 1, style: `fill:${color(n)}` })
      : sv("circle", { cx: x(n.node), cy: y(n), r, style: "fill:var(--text-2);stroke:var(--surface);stroke-width:1.5" });
    const ring = picked ? sv("circle", { cx: x(n.node), cy: y(n), r: r + 4, fill: "none", style: "stroke:var(--text);stroke-width:2" }) : null;
    const hit = sv("circle", { cx: x(n.node), cy: y(n), r: Math.max(r + 3, step / 2, 6), fill: "transparent" });
    const g = sv("g", { class: "node" }, mark, ring, hit);
    g.onmousemove = (ev) => tip(ev, nodeTipOf(n));
    g.onmouseleave = () => tip(null);
    g.onclick = () => selectOverlap(n.node);
    return g;
  });
  const present = new Map(nodes.filter((n) => n.region != null).map((n) => [n.class, n.class]));
  const legend = h("div", { class: "legend" }, classLegend(colors, present),
    h("span", {}, h("i", { style: "background:var(--text-2);border-radius:50%" }), "regions of different classes meet"));
  const wrap = h("div", { class: "table-wrap" },
    sv("svg", { class: "tree", viewBox: `0 0 ${W} ${H}`, width: W, role: "img", style: "max-width:none" },
      grid, zoneLabels, edges, marks));
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
 * @param {object} sel the overlap_zones row of state.overlap
 * @param {Map<string, string>} colors from classColor
 * @param {Map<number, string>} regionClass the class of every region
 */
async function drawOverlapMap(sel, colors, regionClass) {
  const id = ++overlapMapId;
  const el = $("rg-ov-map");
  const what = sel.region != null ? `region ${sel.region}` : sel.parent < 0 ? `zone ${sel.zone}` : `node ${sel.node}`;
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
  for (const r of regionsBelow(last.regions.zones, sel.node)) {
    rows.set(regionClass.get(r), (rows.get(regionClass.get(r)) ?? 0) + regionRow.get(r).n_rows);
  }
  $("rg-ov-note").textContent = `The rows of ${what} by class, on the first two principal components of ` +
    `${num(data.points.length)} of them taken at random (at most 2,000 per region, 20,000 in all), in the space of ` +
    `the regions. ${MAP_HOWTO}`;
  const legend = h("div", { class: "legend" },
    classLegend(colors, new Map([...rows].map(([c, n]) => [c, `${c} · ${num(n)} rows`]))));
  const map = pointMap(data, (r) => colors.get(regionClass.get(r)) ?? OTHER_CLASS, (r) => {
    const row = regionRow.get(r);
    return row ? `region ${r} · ${row.class} · ${num(row.n_rows)} rows · ${num(row.n_read)} CSV rows` : null;
  }, colorClass(colors), `overlap-clip-${id}`);
  el.replaceChildren(legend, map);
}

let overlapMapId = 0; // drops overlap map answers of superseded requests

const LEVELS = [0.25, 0.5, 0.75, 0.95]; // the share of a group's rows inside each of its curves, innermost first
const LEVEL_OPACITY = [1, 0.75, 0.55, 0.35];
const CELL = 4; // side of a density cell, in viewBox units
let showPoints = false; // the maps draw their rows as points too (the switch above every map)

/**
 * Sampled rows in their PCA plane, one scale for both axes framed on the 1st to 99th percentiles, navigable with
 * panZoom (the plot of drawMap). Each group of regions (the regions drawn in one colour) is shown by its density: four
 * curves enclosing 25, 50, 75 and 95% of its rows, and a curve along each axis in the strips above and on the right,
 * scaled to its own peak. A point stands for rows / taken rows of its region (data.sampled), so the regions sampled
 * down weigh as much as they hold. A switch adds the points, whose tooltip names their region.
 * @param {{shares: number[], points: number[][], sampled: number[][]}} data response of a map endpoint
 * @param {(region: number) => string} colorOf colour of a region's group
 * @param {(region: number) => string|null} tipOf tooltip of a region's points
 * @param {(color: string) => string} nameOf name of the group drawn in a colour, for the tooltip of the curves
 * @param {string} clip a unique id for the clip paths
 * @returns {HTMLElement} the switch and the plot
 */
function pointMap(data, colorOf, tipOf, nameOf, clip) {
  const pts = data.points.map(([a, b, region]) => ({ a, b, region }));
  const weight = new Map(data.sampled.map(([region, rows, taken]) => [region, rows / taken]));
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
    const color = colorOf(p.region);
    if (!members.has(color)) members.set(color, []);
    members.get(color).push(i);
  });
  const groups = [...members].map(([color, idx]) =>
    ({ color, ...density(idx.map((i) => bx[i]), idx.map((i) => by[i]),
      idx.map((i) => weight.get(pts[i].region) ?? 1), grid) }));
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
    r: 2.5, "data-region": p.region, style: `fill:${colorOf(p.region)};opacity:0.8`,
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
    const region = ev.target.getAttribute?.("data-region");
    if (region != null) return tip(ev, tipOf(Number(region)));
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

// the problems of a region and the verdicts of a pair, in the order of src.quality.ISSUES and VERDICTS
const ISSUES = ["missing", "duplicates", "label_conflict", "class_overlap", "shortcut"];
const VERDICTS = ["label_conflict", "class_overlap", "shortcut", "robust"];

/** One sentence per column of the quality tables, shown on hover of the header. */
const HELP = {
  rows: "distinct rows (as kept by stage clean)",
  read: "rows of the CSV, copies included",
  status: "solid: none of the problems; otherwise the problems found (see How to read)",
  missing: "rows with at least one empty, unparsable, NaN or infinite cell, and in which columns",
  copies: "share of the CSV rows that repeat another row exactly",
  conflicts: "rows identical on every feature to rows of another class, and of which classes",
  neighbours: "the neighbour this region's rows confuse the most (its verdict and nn_error), then its pairs by verdict",
  nn: "share of rows whose nearest row is in the other region (0 apart, about 0.5 mixed)",
  bayes: "least error any classifier could reach, from nn_error (Cover & Hart, approximate)",
  shortcut: "the columns whose removal makes the classes mix: the separation depends on them",
  without: "nn_error once the shortcut columns are removed",
  single: "balanced accuracy of the best rule on one column alone, and that column",
  verdict: "label_conflict, class_overlap, shortcut or robust (see How to read)",
  reasons: "near: centroids close; label_conflict: shares identical rows; nearest: the closest region of another class",
  sep: "distance between the centroids, in standard deviations of their rows along the line between them",
  pdd: "mean cosine similarity of random pairs of rows of the class",
  tc: "share of random pairs of rows of the class that are functionally identical (cosine > 0.95)",
  solid: "share of the class's CSV rows in solid regions",
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
    h("b", {}, `${num(s.n_solid)} of ${num(s.n_regions)} regions solid`),
    ` · ${pct(s.solid_share)} of the CSV rows · ${num(s.n_pairs)} pairs of regions of different classes measured`);
  for (const el of document.querySelectorAll("#tab-quality [data-stat]")) {
    const v = el.dataset.stat.split(".").reduce((o, k) => o?.[k], s);
    el.textContent = v ?? "–";
  }
  drawClassTable();
  if (!data.regions.some((r) => r.region === state.region)) state.region = null;
  showView(state.view);
}

/** The classes table. */
function drawClassTable() {
  const data = last.quality, th = data.stats.thresholds;
  const nRegions = new Map();
  for (const r of data.regions) nRegions.set(r.class, (nRegions.get(r.class) ?? 0) + 1);
  sortable($("qr-classes"), [
    { key: "class", label: "class", wrap: true },
    { key: "n_rows", label: "rows", format: num, help: HELP.rows },
    { key: "n_read", label: "CSV rows", format: num, help: HELP.read },
    { key: "n_regions", label: "regions", format: num },
    { key: "solid_share", label: "solid", format: (v) => pct(v), help: HELP.solid },
    { key: "dup_share", label: "copies", format: (v) => pct(v), help: HELP.copies },
    { key: "conflict_share", label: "label conflicts", format: (v) => flagged(pct(v, 2), v > 0),
      help: "share of the class's CSV rows identical to rows of another class" },
    { key: "missing_share", label: "null / invalid", format: (v) => flagged(pct(v, 2), v > 0),
      help: "share of the class's CSV rows with an empty, unparsable, NaN or infinite cell" },
    { key: "nn_error", label: "nn_error", format: (v) => flagged(dec(v, 3), v >= th.nn_error),
      help: "nn_error of its regions against the neighbour each confuses the most, weighted by rows" },
    { key: "single_column_acc", label: "best single column",
      format: (v, r) => (v == null ? "–" : flagged(h("span", {}, `${pct(v)} `, colName(r.single_column)), v >= th.single_column_acc)),
      help: "the best single-column rule over its pairs that at most one column separates", wrap: true },
    { key: "pdd", label: "pdd", format: (v) => flagged(dec(v, 3), v >= th.pdd), help: HELP.pdd },
    { key: "tc", label: "tc", format: (v) => flagged(pct(v), v >= th.tc), help: HELP.tc },
    { key: "issues", label: "class problems", format: chips, help: "low_diversity (pdd) and collapse (tc)", wrap: true },
  ], data.classes.map((c) => ({ ...c, n_regions: nRegions.get(c.class) ?? 0 })));
}

/** A column name that may break after its underscores. @param {string} name */
function colName(name) {
  return h("span", {}, name.split("_").flatMap((part, i, all) => (i < all.length - 1 ? [`${part}_`, h("wbr")] : [part])));
}

/** Column names joined by "," as breakable names, comma separated. @param {string} names */
function colNames(names) {
  return h("span", {}, names.split(",").flatMap((n, i) => (i ? [", ", colName(n)] : [colName(n)])));
}

/** Whether a quality region passes a filter of the regions panel. */
function passes(r, f) {
  if (f == null) return true;
  if (f === "solid") return r.status === "solid";
  if (f === "flagged") return r.status !== "solid";
  return (r.issues ?? "").split(",").includes(f);
}

/** The filters, the regions table with one column per check, and the detail of state.region. */
function drawRegionList() {
  const data = last.quality, th = data.stats.thresholds;
  const classOf = new Map(data.regions.map((r) => [r.region, r.class]));
  const missingOf = new Map();
  for (const m of data.missing) missingOf.set(m.region, [...(missingOf.get(m.region) ?? []), m]);
  const conflictWith = new Map();
  for (const p of data.pairs.filter((p) => p.n_conflicts > 0)) {
    for (const [own, other] of [[p.region_a, p.class_b], [p.region_b, p.class_a]]) {
      conflictWith.set(own, new Set([...(conflictWith.get(own) ?? []), other]));
    }
  }
  const classes = data.classes.map((c) => c.class);
  if (!classes.includes(state.qclass)) state.qclass = null;
  $("qr-class").replaceChildren(h("option", { value: "" }, "all classes"), ...classes.map((c) => h("option", { value: c }, c)));
  $("qr-class").value = state.qclass ?? "";
  const ofClass = data.regions.filter((r) => state.qclass == null || r.class === state.qclass);
  fill("qr-filters", [null, "solid", "flagged", ...ISSUES].map((f) => {
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
    ? flagged(`${num(v)} ${v === 1 ? "row" : "rows"} · with ${[...(conflictWith.get(r.region) ?? [])].sort().join(", ")}`, true)
    : "");
  const neighbourText = (v, r) => {
    if (r.rival == null) return h("span", { class: "muted" }, "no neighbour measured");
    const judged = r.verdict ?? "not judged (too few rows)";
    const on = r.verdict === "shortcut" && r.shortcut_columns ? ` on ${r.shortcut_columns}` : "";
    const worst = `${judged}${on} with ${classOf.get(r.rival)} ${r.rival} · nn_error ${dec(v, 2)}`;
    const counts = VERDICTS.filter((k) => r[`pairs_${k}`]).map((k) => `${r[`pairs_${k}`]} ${k}`).join(", ");
    return h("span", {}, r.verdict && r.verdict !== "robust" ? flagged(worst, true) : worst,
      h("span", { class: "muted" }, ` · pairs: ${counts || "none judged"}`));
  };
  sortable($("qr-regions"), [
    { key: "region", label: "region" },
    { key: "class", label: "class", text: true },
    { key: "n_rows", label: "rows", format: num, help: HELP.rows },
    { key: "n_read", label: "CSV rows", format: num, help: HELP.read },
    { key: "status", label: "status", format: (v, r) => (v === "solid" ? "solid" : chips(r.issues)), help: HELP.status, wrap: true },
    { key: "missing_rows", label: "null / invalid", format: missingText, help: HELP.missing, wrap: true },
    { key: "dup_share", label: "copies", format: (v) => flagged(pct(v), v >= th.dup_share), help: HELP.copies },
    { key: "conflict_rows", label: "label conflicts", format: conflictText, help: HELP.conflicts, wrap: true },
    { key: "nn_error", label: "neighbours", format: neighbourText, help: HELP.neighbours, wrap: true },
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

/** Fetch the detail of state.region and draw its neighbours, bars, map, identical rows and null values. */
async function drawDetail() {
  const id = ++detailId;
  const card = $("qr-detail");
  const row = last.quality.regions.find((r) => r.region === state.region);
  card.hidden = row == null;
  if (row == null) return;
  const th = last.quality.stats.thresholds;
  const close = h("button", { class: "chip close", type: "button", title: "close the detail" }, "close");
  close.onclick = () => selectRegion(null);
  fill("qr-detail-title", `Region ${row.region} · ${row.class}`, close);
  fill("qr-detail-head",
    `${num(row.n_rows)} rows · ${num(row.n_read)} CSV rows · ${pct(row.dup_share)} copies · `,
    row.status === "solid" ? "solid" : h("span", {}, "flagged: ", chips(row.issues)));
  for (const el of ["qr-neighbours", "qr-bars", "qr-conflicts", "qr-missing"]) fill(el, h("p", { class: "note" }, "loading…"));
  const ds = encodeURIComponent(state.dataset);
  let d;
  try {
    d = await api(`/${ds}/quality/region?region=${row.region}`);
  } catch (e) {
    if (id === detailId) fill("qr-neighbours", h("p", { class: "note" }, e.message));
    return;
  }
  if (id !== detailId) return;
  const sides = d.pairs.map((p) => {
    const [s, o] = p.region_a === row.region ? ["a", "b"] : ["b", "a"];
    return {
      region: p[`region_${o}`], class: p[`class_${o}`], reasons: p.reasons, sep: p.sep,
      n_conflicts: p.n_conflicts, n_conflicts_read: p.n_conflicts_read,
      here: p[`nn_error_${s}`], there: p[`nn_error_${o}`], nn_error: p.nn_error, bayes: p.bayes_error_low,
      shortcut_columns: p.shortcut_columns, without: p.nn_error_without, rise: p.nn_error_rise,
      acc: p.single_column_acc, column: p.single_column, verdict: p.verdict,
    };
  }).sort((a, b) => (b.here ?? -1) - (a.here ?? -1) || a.region - b.region);
  sortable($("qr-neighbours"), [
    { key: "region", label: "region" },
    { key: "class", label: "class", wrap: true },
    { key: "verdict", label: "verdict", format: verdictChip, help: HELP.verdict, text: true },
    { key: "nn_error", label: "nn_error", format: (v) => flagged(dec(v, 3), v >= th.nn_error),
      help: "the pair's nn_error, the mean of its two sides: what the verdict compares with the threshold" },
    { key: "here", label: "here / there",
      format: (v, r) => (v == null ? "–" : `${dec(v, 3)} / ${dec(r.there, 3)}`),
      help: "nn_error of this region's rows / of the neighbour's rows" },
    { key: "bayes", label: "bayes_error_low", format: (v) => dec(v, 3), help: HELP.bayes },
    { key: "n_conflicts", label: "identical rows",
      format: (v, r) => (v ? h("span", { title: `${num(r.n_conflicts_read)} CSV rows` }, flagged(num(v), true)) : "0"),
      help: "rows of the two regions identical on every feature (hover: CSV rows)" },
    { key: "shortcut_columns", label: "shortcut_columns", help: HELP.shortcut, wrap: true,
      format: (v, r) => (!v ? "–" : r.verdict === "shortcut" ? flagged(colNames(v), true)
        : h("span", { class: "muted", title: "removed to test the pair: the separation does not hang on them" },
          "removed: ", colNames(v))) },
    { key: "without", label: "nn_error without", format: (v) => dec(v, 3), help: HELP.without },
    { key: "acc", label: "single column", format: (v, r) => (r.column == null ? "–" : h("span", {}, `${pct(v)} `, colName(r.column))), help: HELP.single, wrap: true },
    { key: "sep", label: "separation", format: short, help: HELP.sep },
    { key: "reasons", label: "why measured", format: (v) => v.join(", "), help: HELP.reasons, wrap: true },
  ], sides, { onClick: (r) => selectRegion(r.region), empty: "No region of another class was measured against this one." });
  errorBars($("qr-bars"), sides, th.nn_error);
  drawConflicts(d, row);
  if (d.missing.length) {
    sortable($("qr-missing"), [
      { key: "column", label: "column" },
      { key: "kind", label: "kind", help: "empty, unparsable (not a number), nan or inf", text: true },
      { key: "n_rows", label: "rows", format: num, help: HELP.rows },
      { key: "n_read", label: "CSV rows", format: num, help: HELP.read },
    ], d.missing);
  } else fill("qr-missing", h("p", { class: "note" }, "None: every cell of the region is a valid value."));
  drawQualityMap(row, id);
}

/**
 * Two horizontal bars per neighbour: the pair's nn_error with every column (series 1) and without its shortcut
 * columns (series 2), on one axis from 0, the threshold dashed; a tooltip per bar.
 * @param {HTMLElement} el container, replaced
 * @param {object[]} sides the neighbours, in display order
 * @param {number} tau thresholds.nn_error
 */
function errorBars(el, sides, tau) {
  const judged = sides.filter((s) => s.nn_error != null);
  if (!judged.length) { el.replaceChildren(h("p", { class: "note" }, "No pair judged: too few rows.")); return; }
  const top = Math.min(1, niceMax(Math.max(tau * 2, ...judged.map((s) => Math.max(s.nn_error, s.without)))));
  const bar = 9, gap = 2, rowH = 2 * bar + gap + 12;
  const W = Math.max(560, el.clientWidth || 1100); // drawn at the width it is shown, so text keeps its size
  const m = { l: 170, r: 24, t: 22, b: 26 }, H = m.t + m.b + rowH * judged.length;
  const x = (v) => m.l + (Math.min(v, top) / top) * (W - m.l - m.r);
  const kids = [];
  for (let k = 0; k <= 4; k++) {
    const v = (top * k) / 4;
    kids.push(sv("line", { class: k ? "grid" : "axis", x1: x(v), x2: x(v), y1: m.t - 4, y2: H - m.b }),
      sv("text", { x: x(v), y: H - 8, "text-anchor": "middle" }, short(v)));
  }
  kids.push(sv("line", { x1: x(tau), x2: x(tau), y1: m.t - 10, y2: H - m.b, style: "stroke:var(--text-2);stroke-dasharray:4 4" }),
    sv("text", { x: x(tau) + 4, y: m.t - 10 }, `τ = ${tau}`));
  judged.forEach((s, i) => {
    const y0 = m.t + i * rowH;
    kids.push(sv("text", { x: m.l - 8, y: y0 + bar + 4, "text-anchor": "end" }, `${s.class} ${s.region}`));
    [[s.nn_error, "var(--accent)", "every column"],
      [s.without, "var(--series-2)", s.shortcut_columns ? `without ${s.shortcut_columns.split(",").join(", ")}` : "without the columns removed"]]
      .forEach(([v, color, what], j) => {
        const y = y0 + j * (bar + gap);
        const w = Math.max(0, x(v) - x(0));
        const r = Math.min(4, w, bar / 2); // rounded data end, square at the baseline
        const mark = sv("path", {
          d: `M${x(0)},${y}h${w - r}a${r},${r} 0 0 1 ${r},${r}v${bar - 2 * r}a${r},${r} 0 0 1 ${-r},${r}h${-(w - r)}z`,
          style: `fill:${color}`,
        });
        const hit = sv("rect", { x: x(0), y, width: W - m.l - m.r, height: bar, fill: "transparent" });
        hit.onmousemove = (ev) => tip(ev, `${s.class} ${s.region} · nn_error ${what}: ${dec(v, 3)} · ${s.verdict}`);
        hit.onmouseleave = () => tip(null);
        kids.push(mark, hit);
      });
  });
  el.replaceChildren(
    h("div", { class: "legend" },
      h("span", {}, h("i", { style: "background:var(--accent)" }), "every column"),
      h("span", {}, h("i", { style: "background:var(--series-2)" }), "without the columns removed (shortcut_columns)")),
    sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "nn_error per neighbour, with and without the shortcut columns" }, kids));
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
    ["class (region) × CSV rows", "CSV rows", ...columns],
    d.conflicts.map((g) => [
      h("span", { class: "cell-wrap" }, g.rows.map((r) => `${r.class} (${r.region}) ×${num(r.n_read)}`).join(" · ")),
      num(g.n_read),
      ...columns.map((c) => value(g.values[c])),
    ])));
}

let qualityMapId = 0; // drops quality map answers of superseded requests

/**
 * Fetch and draw the map of a region and its neighbours, coloured by class.
 * @param {object} row its quality_regions row
 * @param {number} detail the detailId of the detail it belongs to
 */
async function drawQualityMap(row, detail) {
  const id = ++qualityMapId;
  $("qr-map-title").textContent = `Map of region ${row.region} and its neighbours`;
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
  shuffle(data.points); // drawn interleaved: no region covers the others where they mix
  const byRegion = new Map(last.quality.regions.map((r) => [r.region, r]));
  const order = last.quality.classes.map((c) => c.class);
  // in the dataset's class order, so a class keeps its colour from one region's map to the next
  const shown = [...new Set(data.regions.map((r) => byRegion.get(r).class))];
  shown.sort((a, b) => order.indexOf(a) - order.indexOf(b));
  const colors = classColor(shown);
  const labels = new Map(shown.map((c) => [c, `${c} · ${data.regions.filter((r) => byRegion.get(r).class === c)
    .map((r) => (r === row.region ? `region ${r} (this one)` : `region ${r}`)).join(", ")}`]));
  const n = data.regions.length - 1;
  $("qr-map-note").textContent = `The rows of region ${row.region} and of its ${n} ` +
    `${n === 1 ? "neighbour" : "neighbours"} by class, on the first two principal components of ` +
    `${num(data.points.length)} of them taken at random (at most 2,000 per region, 20,000 in all). ${MAP_HOWTO}`;
  const map = pointMap(data, (r) => colors.get(byRegion.get(r).class) ?? OTHER_CLASS, (r) => {
    const x = byRegion.get(r);
    return x ? `region ${r} · ${x.class} · ${num(x.n_rows)} rows · ${num(x.n_read)} CSV rows` : null;
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

/** A verdict as a chip, flagged unless robust. @param {string|null} v */
function verdictChip(v) {
  if (v == null) return "not judged";
  return v === "robust" ? "robust" : diagnosisChip(v);
}

/** Problems joined by "," as chips. @param {string|null} v */
function chips(v) {
  return v ? h("span", { class: "chips" }, v.split(",").map(diagnosisChip)) : "–";
}

/** A diagnosis as a chip: the warning sign and its name. @param {string} d */
function diagnosisChip(d) {
  return h("span", { class: "chip flag" }, h("b", { class: "sign", "aria-hidden": "true" }, "▲"), d);
}

/** A value, with the warning sign when it reaches its threshold (quality.thresholds). @param {string} text @param {boolean} over */
function flagged(text, over) {
  return over ? h("span", { class: "flag", title: "over its threshold" },
    h("b", { class: "sign", "aria-label": "over the threshold" }, "▲"), text) : text;
}

/** A share in [0, 1] as an inline bar and a percentage, flagged when over its threshold. */
function share(v, over, digits = 1) {
  if (v == null) return "–";
  return h("span", {}, h("i", { class: "bar", style: `width:${Math.round(v * 60)}px` }), flagged(pct(v, digits), over));
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
