// Draws the figures of the page with Plotly, in the theme of the system.
(function () {
  const dark = window.matchMedia("(prefers-color-scheme: dark)");
  // Each light-theme color of dashboard/charts.py with its dark-theme step.
  const DARK_STEPS = {
    "#2a78d6": "#3987e5",
    "#eb6834": "#d95926",
    "#1baf7a": "#199e70",
    "#eda100": "#c98500",
    "#e87ba4": "#d55181",
    "#6250d6": "#9085e9",
    "#e34948": "#e66767",
    "#0b0b0b": "#f0efec",
    "#ffffff": "#0b0b0b",
    // The diverging scale of the tree and the map, which keeps its order in
    // the dark theme: its neutral middle goes dark and its ends brighten.
    "#2166ac": "#5aa9f0",
    "#92c5de": "#2c6a9a",
    "#d9d9d9": "#4a4a47",
    "#f4a582": "#9c4f35",
    "#b2182b": "#f26b62",
  };
  const MARGIN = { l: 56, r: 16, t: 28, b: 48 };

  function token(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  // The neutral middle of a diverging colorscale, whose order the dark theme keeps.
  const DIVERGING_NEUTRAL = "#d9d9d9";

  function themed(figure) {
    if (!dark.matches) return figure;
    const copy = JSON.parse(JSON.stringify(figure), (key, value) =>
      key === "colorscale" &&
      Array.isArray(value) &&
      !value.some(([, color]) => color === DIVERGING_NEUTRAL)
        ? value.map(([at], i) => [at, value[value.length - 1 - i][1]])
        : value
    );
    return JSON.parse(
      JSON.stringify(copy).replace(/#[0-9a-f]{6}/gi, (hex) => DARK_STEPS[hex.toLowerCase()] || hex)
    );
  }

  // Zooms the tree and the map of the tree tab on a subtree; "" is the whole class.
  function focusNode(value) {
    const form = document.querySelector("form.filters input[name=node]")?.form;
    if (!form) return;
    form.elements.node.value = value;
    form.dispatchEvent(new Event("change", { bubbles: true }));
  }

  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-node]");
    if (button) focusNode(button.dataset.node);
  });
  // The ids of a tree mean nothing in another class's, so a new class starts whole.
  document.addEventListener(
    "change",
    (event) => {
      if (event.target.name === "class_id" && event.target.form?.elements.node) {
        event.target.form.elements.node.value = "";
      }
    },
    true
  );

  function draw(el) {
    el._figure = el._figure || JSON.parse(el.dataset.figure);
    const figure = themed(el._figure);
    const layout = Object.assign({ hovermode: "closest", legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom" } }, figure.layout);
    layout.margin = Object.assign({}, MARGIN, figure.layout.margin);
    layout.paper_bgcolor = "rgba(0,0,0,0)";
    layout.plot_bgcolor = "rgba(0,0,0,0)";
    layout.font = { family: "system-ui, -apple-system, Segoe UI, sans-serif", size: 12, color: token("--ink-2") };
    for (const axis of ["xaxis", "yaxis"]) layout[axis] = layout[axis] || {};
    for (const name of Object.keys(layout)) {
      if (/^[xy]axis\d*$/.test(name)) {
        layout[name] = Object.assign(
          { gridcolor: token("--grid"), zerolinecolor: token("--axis"), linecolor: token("--axis"), automargin: true },
          layout[name]
        );
      }
    }
    Plotly.react(el, figure.data, layout, {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
    });
    if (!el.dataset.ready) {
      el.dataset.ready = "1";
      el.on("plotly_click", (event) => {
        const target = event.points[0].customdata;
        if (typeof target === "string" && target.startsWith("/")) window.location = target;
        else if (target && target.node !== undefined) focusNode(target.node);
      });
    }
  }

  function drawAll(root) {
    const charts = [...root.querySelectorAll(".chart[data-figure]")];
    if (root.matches && root.matches(".chart[data-figure]")) charts.push(root);
    charts.forEach(draw);
  }

  htmx.onLoad(drawAll);
  dark.addEventListener("change", () => drawAll(document));
})();
