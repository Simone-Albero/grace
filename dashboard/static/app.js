// Dashboard front-end: one page, one tab per stage. Charts are hand-made SVG, no dependencies.

/**
 * UI state.
 * @type {{dataset: string|null, tab: "clean"|"prepare"|"regions"|"quality", cls: string|null, node: number|null,
 *   overlap: number|null, diagnosis: string|null}} node: the region tree node whose map is shown (null: the root of
 *   state.cls); overlap: the overlap zones node whose map is shown (null: the root of the first zone); diagnosis:
 *   the filter of the quality pairs, one of FILTERS (null: every pair)
 */
const state = { dataset: null, tab: "clean", cls: null, node: null, overlap: null, diagnosis: null };

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
  $("dataset").onchange = (e) => { state.dataset = e.target.value; state.cls = state.node = state.overlap = state.diagnosis = null; load(); };
  $("rg-class").onchange = (e) => selectClass(e.target.value);
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
    chip.onclick = () => { selectNode(b.node.node); $("rg-tree").scrollIntoView({ behavior: "smooth", block: "start" }); };
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
    onClick: (r) => { selectOverlap(meet.get(r)); $("rg-ov-tree").scrollIntoView({ behavior: "smooth", block: "start" }); },
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
  $("rg-ov-note").textContent = `${num(data.points.length)} random rows of ${what} (at most 2,000 per region, ` +
    "20,000 in all), on the first two principal components of these rows, in the space of the regions, coloured by " +
    "class. Hover a point for its region; drag to pan, pinch or ctrl + scroll to zoom, double click to reset.";
  const legend = h("div", { class: "legend" },
    classLegend(colors, new Map([...rows].map(([c, n]) => [c, `${c} · ${num(n)} rows`]))));
  const svg = pointMap(data, (r) => colors.get(regionClass.get(r)) ?? OTHER_CLASS, (r) => {
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

/** The filters of the quality pairs: the verdicts (quality.verdict), then the pairs a single column separates. */
const FILTERS = ["conflict", "overlap", "shortcut", "robust", "one feature"];

/**
 * Tiles, the classes table, the pairs table with its verdict filter, the regions table with their status.
 * @param {object} data response of /quality
 */
function renderQuality(data) {
  const s = data.stats, th = s.thresholds;
  const oneFeature = (p) => p.one_feature >= th.one_feature && p.n_separating <= 1;
  const matches = (p, f) => (f === "one feature" ? oneFeature(p) : p.verdict === f);
  const total = (key) => data.regions.reduce((a, r) => a + r[key], 0);
  const rows = total("n_rows"), read = total("n_read");
  const conflictRead = total("conflict_read"), missingRead = total("missing_read");
  fill("qr-tiles",
    tile("solid regions", `${num(s.n_solid)} of ${num(s.n_regions)}`, `${pct(s.solid_share)} of the CSV rows`),
    tile("copies removed", num(read - rows), `${pct((read - rows) / read)} of ${num(read)} CSV rows`),
    tile("rows in a label conflict", num(conflictRead), `${pct(conflictRead / read, 2)} of the CSV rows`),
    tile("rows with a null or invalid value", num(missingRead), `${pct(missingRead / read, 2)} of the CSV rows`),
    tile("pairs measured", num(s.n_pairs), `${num(s.n_near)} near · ${num(s.n_conflict)} conflict · ${num(s.n_nearest)} nearest`),
    tile("shortcut pairs", num(s.pairs_shortcut),
      `${num(s.pairs_overlap)} overlap · ${num(s.pairs_conflict)} conflict · ${num(s.pairs_robust)} robust`));
  $("qr-candidate").textContent = `${s.candidate_separation} sd`;
  $("qr-rows").textContent = num(s.max_rows_per_region);
  $("qr-error").textContent = th.error;
  $("qr-group").textContent = s.max_group;

  sortable($("qr-classes"), [
    { key: "class", label: "class" },
    { key: "n_rows", label: "rows", format: num },
    { key: "n_read", label: "CSV rows", format: num },
    { key: "dup_share", label: "copies", format: (v) => share(v, false) },
    { key: "conflict_share", label: "conflicts", format: (v) => share(v, v > 0, 2) },
    { key: "missing_share", label: "null / invalid", format: (v) => share(v, v > 0, 2) },
    { key: "err", label: "1-NN error", format: (v) => flagged(dec(v, 3), v >= th.error) },
    { key: "one_feature", label: "one feature", format: (v, r) => (v == null ? "–" : flagged(`${pct(v)} ${r.one_feature_col}`, v >= th.one_feature)) },
    { key: "pdd", label: "diversity (pdd)", format: (v) => flagged(dec(v, 3), v >= th.pdd) },
    { key: "tc", label: "collapse (tc)", format: (v) => flagged(pct(v), v >= th.tc) },
    { key: "issues", label: "class issues", format: chips },
    { key: "solid_share", label: "solid", format: (v) => share(v, false) },
    { key: "pairs_shortcut", label: "shortcut pairs", format: num },
    { key: "pairs_overlap", label: "overlap pairs", format: num },
    { key: "n_pairs", label: "pairs", format: num },
  ], data.classes);

  const draw = () => {
    const shown = state.diagnosis ? data.pairs.filter((p) => matches(p, state.diagnosis)) : data.pairs;
    fill("qr-filters", h("div", { class: "chips" }, [null, ...FILTERS].map((f) => {
      const n = f == null ? data.pairs.length : data.pairs.filter((p) => matches(p, f)).length;
      const chip = h("button", { class: "chip", type: "button", "aria-pressed": String(state.diagnosis === f) },
        `${f ?? "all pairs"} (${num(n)})`);
      chip.onclick = () => { state.diagnosis = f; draw(); };
      return chip;
    })));
    sortable($("qr-pairs"), [
      { key: "region_a", label: "region a" },
      { key: "class_a", label: "class a" },
      { key: "region_b", label: "region b" },
      { key: "class_b", label: "class b" },
      { key: "reasons", label: "measured as", format: (v) => v.join(", ") },
      { key: "verdict", label: "verdict", format: verdictChip },
      { key: "sep", label: "separation", format: short },
      { key: "n_conflicts", label: "conflicts", format: num },
      { key: "err_a", label: "1-NN error a", format: (v) => flagged(dec(v, 3), v >= th.error) },
      { key: "err_b", label: "1-NN error b", format: (v) => flagged(dec(v, 3), v >= th.error) },
      { key: "removed", label: "columns removed", format: (v) => v || "–" },
      { key: "err_without", label: "error without them", format: (v) => dec(v, 3) },
      { key: "delta", label: "rise", format: (v) => flagged(dec(v, 3), v >= th.error) },
      { key: "one_feature", label: "one feature", format: (v, r) => flagged(`${pct(v)} ${r.one_feature_col ?? ""}`, oneFeature(r)) },
      { key: "n_separating", label: "columns that separate", format: num },
    ], shown);
  };
  draw();

  const missing = {};
  for (const m of data.missing) (missing[m.region] ??= []).push(`${m.column} ${m.kind} ×${num(m.n_rows)}`);
  sortable($("qr-regions"), [
    { key: "region", label: "region" },
    { key: "class", label: "class" },
    { key: "status", label: "status", format: (v, r) => (v === "solid" ? "solid" : chips(r.issues)) },
    { key: "n_rows", label: "rows", format: num },
    { key: "n_read", label: "CSV rows", format: num },
    { key: "n_duplicates", label: "copies", format: num },
    { key: "dup_share", label: "copies share", format: (v) => share(v, v >= th.dup_share) },
    { key: "conflict_rows", label: "rows in a conflict", format: num },
    { key: "missing_rows", label: "rows null / invalid", format: (v, r) => (v ? h("span", { title: (missing[r.region] ?? []).join("\n") }, flagged(num(v), true)) : "0") },
    { key: "rival", label: "closest rival", format: (v) => (v == null ? "–" : String(v)) },
    { key: "err", label: "1-NN error", format: (v) => flagged(dec(v, 3), v >= th.error) },
    { key: "one_feature", label: "one feature", format: (v, r) => (v == null ? "–" : `${pct(v)} ${r.one_feature_col ?? ""}`) },
    { key: "verdict", label: "verdict", format: verdictChip },
    { key: "depends_on", label: "rests on", format: (v) => v || "–" },
    { key: "pairs_shortcut", label: "shortcut pairs", format: num },
    { key: "pairs_overlap", label: "overlap pairs", format: num },
    { key: "pairs_conflict", label: "conflict pairs", format: num },
  ], data.regions);
}

/** A verdict as a chip, flagged unless robust. @param {string|null} v */
function verdictChip(v) {
  if (v == null) return "–";
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
