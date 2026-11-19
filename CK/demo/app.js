import {
  buildSymmetricGraph,
  getGraphEdges,
  generateMethod,
  interpolateEdge,
  interpolateSimplex,
  loadEcoli,
  makeTeachingDataset,
  makeTopologyDataset,
} from "./core.mjs";

const METHODS = [
  { id: "random", title: "Random oversampling", subtitle: "duplicate minority points" },
  { id: "smote", title: "SMOTE", subtitle: "sample from graph edges" },
  { id: "simplicial", title: "Simplicial SMOTE", subtitle: "sample from local simplices" },
];

const COLORS = {
  majority: "#9aa8b8",
  minority: "#345e9b",
  holdout: "#c68b3b",
  synthetic: "#bd5968",
  edge: "#a9b9ca",
  simplex: "#d8e4f2",
};

const state = {
  dataset: makeTopologyDataset("moons"),
  datasetKey: "moons",
  options: { k: 5, p: 2, budget: 80, seed: 2207 },
  results: new Map(),
  selected: null,
  tour: {
    step: 0,
    playing: false,
    timer: null,
    ticks: 0,
    fixture: makeTeachingDataset(),
    edgeLambda: 0.5,
    simplexWeights: [0.2, 0.5, 0.3],
  },
};

const elements = {
  dataset: document.querySelector("#dataset-select"),
  k: document.querySelector("#k-input"),
  p: document.querySelector("#p-input"),
  budget: document.querySelector("#budget-input"),
  seed: document.querySelector("#seed-input"),
  generate: document.querySelector("#generate-button"),
  status: document.querySelector("#status"),
  summary: document.querySelector("#dataset-summary"),
  source: document.querySelector("#dataset-source"),
  samples: document.querySelector("#sample-summary"),
  simplex: document.querySelector("#simplex-summary"),
  grid: document.querySelector("#comparison-grid"),
  detailTitle: document.querySelector("#detail-title"),
  detailMethod: document.querySelector("#detail-method"),
  detailContent: document.querySelector("#detail-content"),
  tourCanvas: document.querySelector("#tour-canvas-wrap"),
  tourStepLabel: document.querySelector("#tour-step-label"),
  tourProgress: document.querySelector("#tour-progress"),
  tourKicker: document.querySelector("#tour-kicker"),
  tourHeading: document.querySelector("#tour-heading"),
  tourCopy: document.querySelector("#tour-copy"),
  tourFormula: document.querySelector("#tour-formula"),
  tourObservation: document.querySelector("#tour-observation"),
  tourPlay: document.querySelector("#tour-play"),
  tourRestart: document.querySelector("#tour-restart"),
  tourNext: document.querySelector("#tour-next"),
  tourStepButtons: [...document.querySelectorAll(".tour-step-button")],
  budgetChart: document.querySelector("#budget-chart"),
  conclusion: document.querySelector("#conclusion"),
};

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
  }[character]));
}

function readOptions() {
  return {
    k: Math.max(2, Math.min(9, Number(elements.k.value) || 5)),
    p: Math.max(1, Math.min(4, Number(elements.p.value) || 2)),
    budget: Math.max(12, Math.min(260, Number(elements.budget.value) || 80)),
    seed: Math.max(1, Number(elements.seed.value) || 2207),
  };
}

function formatPercent(value) {
  return `${(value * 100).toFixed(0)}%`;
}

function projectPoints(dataset) {
  if (dataset.dimensions === 2) return dataset;
  const project = (point) => [point[0], point[1]];
  return {
    ...dataset,
    minority: dataset.minority.map(project),
    observedMinority: dataset.observedMinority.map(project),
    holdoutMinority: dataset.holdoutMinority.map(project),
    majority: dataset.majority.map(project),
  };
}

function plotBounds(dataset) {
  const allPoints = [...dataset.majority, ...dataset.minority];
  const xs = allPoints.map(([x]) => x);
  const ys = allPoints.map(([, y]) => y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const paddingX = (xMax - xMin || 1) * 0.1;
  const paddingY = (yMax - yMin || 1) * 0.1;
  return {
    xMin: xMin - paddingX, xMax: xMax + paddingX,
    yMin: yMin - paddingY, yMax: yMax + paddingY,
  };
}

function pointToSvg(point, bounds, width, height) {
  return [
    ((point[0] - bounds.xMin) / (bounds.xMax - bounds.xMin)) * width,
    height - ((point[1] - bounds.yMin) / (bounds.yMax - bounds.yMin)) * height,
  ];
}

const TOUR_STEPS = [
  {
    kicker: "Step 1 · The problem",
    heading: "The minority class is sparse",
    copy: "The blue points are minority examples the model needs to learn. When they are few and scattered, simply repeating the data does not create new support regions.",
    formula: "minority points → insufficient coverage",
    observation: "Goal: create new points while preserving local geometry.",
  },
  {
    kicker: "Step 2 · Random",
    heading: "Random oversampling only copies old points",
    copy: "Random oversampling increases the count, but every new point duplicates a minority parent. No new region of space is opened.",
    formula: "x_new = x_parent",
    observation: "The count increases, but diversity does not.",
  },
  {
    kicker: "Step 3 · SMOTE",
    heading: "SMOTE only moves along one edge",
    copy: "Choose two nearby minority points and interpolate between them. A synthetic point can move along the segment connecting two parents, but cannot enter the interior of a triangle.",
    formula: "x_new = (1 − λ)A + λB",
    observation: "λ changes the position along a single edge.",
  },
  {
    kicker: "Step 4 · Simplicial SMOTE",
    heading: "From an edge to a simplex",
    copy: "If three points form a clique in the k-NN graph, they define a 2-simplex. Dirichlet weights create a new point in the interior of the triangle.",
    formula: "x_new = λ₁A + λ₂B + λ₃C; Σλᵢ = 1",
    observation: "This is new geometric space that SMOTE cannot reach.",
  },
  {
    kicker: "Step 5 · Compare",
    heading: "Same budget, different coverage",
    copy: "Now run all three methods on the same data with the same sample count. Look for duplicate points, edge segments, and filled simplex regions.",
    formula: "same data · same seed · same budget",
    observation: "Coverage is an empirical question, not an assumption.",
  },
  {
    kicker: "Step 6 · Conclusion",
    heading: "What the paper aims to address",
    copy: "Simplicial SMOTE models the minority class with local simplices instead of only line segments. This may increase coverage, but the result depends on the geometry, k, and p.",
    formula: "edge model → simplex model",
    observation: "Simplicial SMOTE does not always outperform SMOTE.",
  },
];

function tourPoint(point, bounds, color, radius, className = "") {
  const [x, y] = pointToSvg(point, bounds, 620, 345);
  return `<circle class="tour-point ${className}" cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="${radius}" fill="${color}"></circle>`;
}

function renderTourSvg() {
  const dataset = state.tour.fixture;
  const bounds = plotBounds(dataset);
  const step = state.tour.step;
  const graph = buildSymmetricGraph(dataset.observedMinority, 3);
  const edges = getGraphEdges(graph).map(([from, to]) => {
    const a = pointToSvg(dataset.observedMinority[from], bounds, 620, 345);
    const b = pointToSvg(dataset.observedMinority[to], bounds, 620, 345);
    const highlighted = (step === 2 && [from, to].sort((x, y) => x - y).join("-") === "0-1")
      || (step === 3 && [[0, 1], [0, 3], [1, 3]].some(([x, y]) => x === from && y === to));
    return `<line class="tour-edge ${highlighted ? "is-highlighted" : ""}" x1="${a[0].toFixed(2)}" y1="${a[1].toFixed(2)}" x2="${b[0].toFixed(2)}" y2="${b[1].toFixed(2)}"></line>`;
  }).join("");
  const triangle = [0, 1, 3].map((index) => pointToSvg(dataset.observedMinority[index], bounds, 620, 345));
  const simplex = step >= 3
    ? `<polygon class="tour-simplex" points="${triangle.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" ")}"></polygon>`
    : "";
  const majority = dataset.majority.map((point) => tourPoint(point, bounds, COLORS.majority, 3)).join("");
  const minority = dataset.observedMinority.map((point, index) => tourPoint(
    point,
    bounds,
    COLORS.minority,
    5,
    (step === 3 && [0, 1, 3].includes(index)) ? "is-parent" : "",
  )).join("");
  let synthetic = "";
  if (step === 1) {
    synthetic = [0, 1, 3, 3, 1, 0].map((parent, index) => tourPoint(
      dataset.observedMinority[parent],
      bounds,
      COLORS.synthetic,
      4,
      index > 2 ? "is-duplicate" : "",
    )).join("");
  } else if (step === 2) {
    const point = interpolateEdge(dataset.observedMinority, [0, 1], state.tour.edgeLambda);
    synthetic = tourPoint(point, bounds, COLORS.synthetic, 7, "tour-synthetic");
  } else if (step >= 3) {
    const point = interpolateSimplex(dataset.observedMinority, [0, 1, 3], state.tour.simplexWeights);
    synthetic = tourPoint(point, bounds, COLORS.synthetic, 7, "tour-synthetic");
  }
  return `<svg class="tour-canvas" viewBox="0 0 620 345" role="img" aria-label="Illustration of step ${step + 1}">
    <rect width="620" height="345" fill="#fbfcfd"></rect>
    ${edges}${simplex}<g>${majority}</g><g>${minority}</g><g>${synthetic}</g>
  </svg>`;
}

function renderTour() {
  const step = TOUR_STEPS[state.tour.step];
  elements.tourCanvas.innerHTML = renderTourSvg();
  elements.tourStepLabel.textContent = `Step ${state.tour.step + 1} / ${TOUR_STEPS.length}`;
  elements.tourProgress.style.width = `${((state.tour.step + 1) / TOUR_STEPS.length) * 100}%`;
  elements.tourKicker.textContent = step.kicker;
  elements.tourHeading.textContent = step.heading;
  elements.tourCopy.textContent = step.copy;
  elements.tourFormula.textContent = state.tour.step === 2
    ? `x_new = (1 − λ)A + λB; λ = ${state.tour.edgeLambda.toFixed(2)}`
    : state.tour.step === 3
      ? `x_new = λ₁A + λ₂B + λ₃C; λ = [${state.tour.simplexWeights.map((value) => value.toFixed(2)).join(", ")}]`
      : step.formula;
  elements.tourObservation.textContent = step.observation;
  elements.tourPlay.textContent = state.tour.playing ? "Pause step" : "Play step";
  elements.tourNext.textContent = state.tour.step === TOUR_STEPS.length - 1 ? "Replay tour" : "Next";
  elements.tourStepButtons.forEach((button, index) => {
    button.classList.toggle("is-active", index === state.tour.step);
    if (index === state.tour.step) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  });
}

function stopTour() {
  state.tour.playing = false;
  if (state.tour.timer) window.clearInterval(state.tour.timer);
  state.tour.timer = null;
  state.tour.ticks = 0;
  renderTour();
}

function setTourStep(step) {
  stopTour();
  state.tour.step = Math.max(0, Math.min(TOUR_STEPS.length - 1, step));
  renderTour();
}

function playTour() {
  if (state.tour.playing) {
    stopTour();
    return;
  }
  state.tour.playing = true;
  state.tour.timer = window.setInterval(() => {
    state.tour.ticks += 1;
    const phase = (Math.sin(state.tour.ticks / 5) + 1) / 2;
    if (state.tour.step === 2) {
      state.tour.edgeLambda = phase;
      renderTour();
    }
    if (state.tour.step === 3) {
      state.tour.simplexWeights = [
        0.12 + phase * 0.35,
        0.55 - phase * 0.25,
        0.33 - phase * 0.1,
      ];
      renderTour();
    }
    if (state.tour.ticks % 26 !== 0) return;
    if (state.tour.step >= TOUR_STEPS.length - 1) {
      stopTour();
      return;
    }
    state.tour.step += 1;
    renderTour();
  }, 2600);
  renderTour();
}

function svgCircle(point, bounds, color, radius, className = "") {
  const [x, y] = pointToSvg(point, bounds, 560, 330);
  return `<circle class="${className}" cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="${radius}" fill="${color}"></circle>`;
}

function renderEdges(result, dataset, bounds, method) {
  if (method === "random") return "";
  const seen = new Set();
  const edges = [];
  result.adjacency.forEach((neighbors, index) => {
    neighbors.forEach((neighbor) => {
      const key = index < neighbor ? `${index}-${neighbor}` : `${neighbor}-${index}`;
      if (seen.has(key)) return;
      seen.add(key);
      const from = pointToSvg(dataset.observedMinority[index], bounds, 560, 330);
      const to = pointToSvg(dataset.observedMinority[neighbor], bounds, 560, 330);
      edges.push(`<line x1="${from[0].toFixed(2)}" y1="${from[1].toFixed(2)}" x2="${to[0].toFixed(2)}" y2="${to[1].toFixed(2)}"></line>`);
    });
  });
  return `<g class="graph-edges">${edges.join("")}</g>`;
}

function renderSimplex(result, dataset, bounds, method) {
  if (method !== "simplicial") return "";
  const simplex = result.generated.find(({ dimension }) => dimension >= 2);
  if (!simplex) return "";
  const points = simplex.parents.map((parent) => pointToSvg(dataset.observedMinority[parent], bounds, 560, 330));
  return `<polygon class="simplex-region" points="${points.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" ")}"></polygon>`;
}

function renderPlot(method, result) {
  const dataset = projectPoints(state.dataset);
  const bounds = plotBounds(dataset);
  const majority = dataset.majority.map((point) => svgCircle(point, bounds, COLORS.majority, 2.2)).join("");
  const observed = dataset.observedMinority.map((point) => svgCircle(point, bounds, COLORS.minority, 3)).join("");
  const holdout = dataset.holdoutMinority.map((point) => svgCircle(point, bounds, COLORS.holdout, 3)).join("");
  const synthetic = result.generated.map(({ point }, index) => {
    const [x, y] = pointToSvg(state.dataset.dimensions === 2 ? point : [point[0], point[1]], bounds, 560, 330);
    return `<circle class="plot-hit synthetic-hit" tabindex="0" role="button" aria-label="Synthetic point ${index + 1}" data-method="${method}" data-index="${index}" cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="4.1" fill="${COLORS.synthetic}"></circle>`;
  }).join("");
  return `<svg class="plot" viewBox="0 0 560 330" role="img" aria-label="${escapeHtml(METHODS.find(({ id }) => id === method).title)} plot">
    <rect x="0" y="0" width="560" height="330" fill="#fbfcfd"></rect>
    ${renderSimplex(result, dataset, bounds, method)}
    ${renderEdges(result, dataset, bounds, method)}
    <g>${majority}</g><g>${holdout}</g><g>${observed}</g><g>${synthetic}</g>
  </svg>`;
}

function methodMetricMarkup(result) {
  return `<div class="method-metrics">
    <div><span>coverage</span><strong>${formatPercent(result.metrics.coverage)}</strong></div>
    <div><span>intrusion</span><strong>${formatPercent(result.metrics.intrusion)}</strong></div>
    <div><span>runtime</span><strong>${result.elapsed.toFixed(1)} ms</strong></div>
  </div>`;
}

function renderBudgetChart() {
  const budgets = [24, 48, 80, 120];
  const width = 760;
  const height = 170;
  const padding = { left: 34, right: 18, top: 18, bottom: 27 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  const colors = { random: COLORS.holdout, smote: COLORS.minority, simplicial: COLORS.synthetic };
  const lines = METHODS.map((method) => {
    const values = budgets.map((budget) => generateMethod(state.dataset, method.id, {
      ...state.options,
      budget,
    }).metrics.coverage);
    const points = values.map((value, index) => [
      padding.left + (index / (budgets.length - 1)) * chartWidth,
      padding.top + (1 - value) * chartHeight,
    ]);
    return {
      method,
      values,
      points,
      path: points.map(([x, y], index) => `${index ? "L" : "M"} ${x.toFixed(1)} ${y.toFixed(1)}`).join(" "),
    };
  });
  const labels = lines.map(({ method, points }) => `
    <path class="chart-line" d="${lines.find(({ method: item }) => item.id === method.id).path}" stroke="${colors[method.id]}"></path>
    <text class="chart-label" x="${(points.at(-1)[0] - 48).toFixed(1)}" y="${(points.at(-1)[1] - 7).toFixed(1)}">${method.id}</text>
  `).join("");
  const xLabels = budgets.map((budget, index) => {
    const x = padding.left + (index / (budgets.length - 1)) * chartWidth;
    return `<text class="chart-label" x="${x.toFixed(1)}" y="${height - 7}" text-anchor="middle">${budget}</text>`;
  }).join("");
  elements.budgetChart.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Coverage theo synthetic budget">
    <line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${height - padding.bottom}" stroke="#dfe5ec"></line>
    <line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" stroke="#dfe5ec"></line>
    <text class="chart-label" x="4" y="${padding.top + 4}">100%</text>
    <text class="chart-label" x="14" y="${height - padding.bottom + 4}">0%</text>
    ${labels}${xLabels}
  </svg>`;
}

function renderConclusion() {
  const results = METHODS.map((method) => ({
    ...method,
    metrics: state.results.get(method.id).metrics,
  }));
  const bestCoverage = [...results].sort((a, b) => b.metrics.coverage - a.metrics.coverage)[0];
  const bestDiversity = [...results].sort((a, b) => b.metrics.diversity - a.metrics.diversity)[0];
  elements.conclusion.innerHTML = `<strong>Read the result:</strong> in the current setting,
    <strong>${bestCoverage.title}</strong> has the highest holdout coverage
    (${formatPercent(bestCoverage.metrics.coverage)}), while
    <strong>${bestDiversity.title}</strong> has the highest diversity
    (${formatPercent(bestDiversity.metrics.diversity)}). This is a geometric proxy for the demo;
    change the dataset or <code>k/p</code> to see how the conclusion can change.`;
}

function renderComparison() {
  elements.grid.innerHTML = METHODS.map((method) => {
    const result = state.results.get(method.id);
    return `<article class="method-panel">
      <div class="method-header">
        <span class="method-title">${method.title}</span>
        <span class="method-subtitle">${method.subtitle}</span>
      </div>
      <div class="plot-wrap">${renderPlot(method.id, result)}</div>
      ${methodMetricMarkup(result)}
    </article>`;
  }).join("");
  elements.grid.querySelectorAll(".synthetic-hit").forEach((node) => {
    node.addEventListener("click", () => selectPoint(node.dataset.method, Number(node.dataset.index)));
    node.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectPoint(node.dataset.method, Number(node.dataset.index));
      }
    });
  });
}

function selectPoint(method, index) {
  const item = state.results.get(method).generated[index];
  state.selected = { method, index };
  elements.detailTitle.textContent = `Synthetic point ${index + 1}`;
  elements.detailMethod.textContent = METHODS.find(({ id }) => id === method).title;
  elements.detailContent.innerHTML = `
    <p>Dimension simplex: <strong>${item.dimension}</strong> · ${item.parents.length} parent vertices</p>
    <div class="weight-list">
      ${item.weights.map((weight, parentIndex) => `<span class="weight-chip">parent ${item.parents[parentIndex] + 1}: <strong>${weight.toFixed(3)}</strong></span>`).join("")}
    </div>
    <p class="muted">The weights are non-negative and sum to ${item.weights.reduce((sum, value) => sum + value, 0).toFixed(3)}. This is a convex combination in ${state.dataset.dimensions}D space.</p>
  `;
}

async function loadDataset(key) {
  if (key === "ecoli") {
    elements.status.textContent = "Loading Ecoli…";
    return loadEcoli();
  }
  return makeTopologyDataset(key, state.options.seed);
}

async function generate() {
  elements.status.classList.remove("error");
  elements.status.textContent = "Generating samples…";
  elements.generate.disabled = true;
  try {
    state.options = readOptions();
    state.dataset = await loadDataset(state.datasetKey);
    state.results = new Map(METHODS.map((method) => [
      method.id,
      generateMethod(state.dataset, method.id, state.options),
    ]));
    const simplicial = state.results.get("simplicial");
    elements.summary.textContent = state.dataset.name;
    elements.source.textContent = state.dataset.source;
    elements.samples.textContent = `${state.dataset.observedMinority.length} / ${state.dataset.holdoutMinority.length}`;
    elements.simplex.textContent = simplicial.adjacency
      ? `${simplicial.generated.filter(({ dimension }) => dimension >= 2).length} samples p≥2`
      : "—";
    renderComparison();
    renderBudgetChart();
    renderConclusion();
    elements.status.textContent = `Generated ${state.options.budget} synthetic points per method`;
  } catch (error) {
    elements.status.classList.add("error");
    elements.status.textContent = error instanceof Error ? error.message : "Could not generate data.";
  } finally {
    elements.generate.disabled = false;
  }
}

elements.dataset.addEventListener("change", () => {
  state.datasetKey = elements.dataset.value;
  generate();
});
elements.generate.addEventListener("click", generate);
elements.tourPlay.addEventListener("click", playTour);
elements.tourRestart.addEventListener("click", () => setTourStep(0));
elements.tourNext.addEventListener("click", () => (
  setTourStep(state.tour.step === TOUR_STEPS.length - 1 ? 0 : state.tour.step + 1)
));
elements.tourStepButtons.forEach((button) => {
  button.addEventListener("click", () => setTourStep(Number(button.dataset.tourStep)));
});
document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, select, textarea, button")) return;
  if (event.key === "ArrowRight") setTourStep(state.tour.step + 1);
  if (event.key === "ArrowLeft") setTourStep(state.tour.step - 1);
  if (event.key === " ") {
    event.preventDefault();
    playTour();
  }
});
renderTour();
generate();
