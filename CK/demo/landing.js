// Local two-port run.sh keeps API on :8000; ngrok/same-origin share uses "".
const API =
  location.hostname === "127.0.0.1" || location.hostname === "localhost"
    ? "http://127.0.0.1:8000"
    : "";
const STORAGE_KEY = "ssmote-demo-dataset";
try {
  localStorage.removeItem("ssmote-demo-budget");
} catch {
  /* ignore */
}

const FALLBACK = [
  {
    id: "fraud",
    name: "Fraud amount × risk",
    blurb: "Normal transactions cluster around the origin; fraud forms two clusters.",
    howto: "Synthetic 2D: Normal around the origin (majority) plus two Normal clusters (minority). Not real transaction data.",
    features: ["transaction amount", "merchant risk"],
    group: "teaching",
    n_train: 214,
    n_test: 134,
    n_minority: 34,
    n_majority: 180,
    imbalance_ratio: 180 / 34,
  },
  {
    id: "moons",
    name: "Two moons",
    blurb: "Paper §4.1: two moon arcs with n⁺=50, n⁻=300.",
    howto: "Paper synthetic: minority on one moon arc, majority on the other (50 / 300 train).",
    features: ["x", "y"],
    group: "paper",
    n_train: 350,
    n_test: 150,
    n_minority: 50,
    n_majority: 300,
    imbalance_ratio: 6,
  },
  {
    id: "swiss_roll",
    name: "Swiss roll",
    blurb: "Paper §4.1: minority on a Swiss-roll manifold projected to 2D.",
    howto: "Paper synthetic: minority = Swiss roll (x,z); majority fills the bounding box (50 / 300 train).",
    features: ["x", "z"],
    group: "paper",
    n_train: 350,
    n_test: 150,
    n_minority: 50,
    n_majority: 300,
    imbalance_ratio: 6,
  },
  {
    id: "gaussian_circle",
    name: "Gaussian in circle",
    blurb: "Paper §4.1: Gaussian minority blob inside a circular majority ring.",
    howto: "Paper synthetic: majority on an outer ring; minority ~ Normal at the origin (50 / 300 train).",
    features: ["x", "y"],
    group: "paper",
    n_train: 350,
    n_test: 150,
    n_minority: 50,
    n_majority: 300,
    imbalance_ratio: 6,
  },
  {
    id: "circles",
    name: "Circle in circle",
    blurb: "Paper §4.1: concentric circles — minority on the outer ring.",
    howto: "Paper synthetic: inner disk = majority; outer ring = minority (50 / 300 train).",
    features: ["x", "y"],
    group: "paper",
    n_train: 350,
    n_test: 150,
    n_minority: 50,
    n_majority: 300,
    imbalance_ratio: 6,
  },
];

const grid = document.querySelector("#dataset-grid");
const selectedName = document.querySelector("#selected-name");
const selectedBlurb = document.querySelector("#selected-blurb");
const selectedStats = document.querySelector("#selected-stats");
const startDemo = document.querySelector("#start-demo");
const statusEl = document.querySelector("#landing-status");
const scenarioSample = document.querySelector("#scenario-sample");

const state = {
  datasets: FALLBACK,
  selected: localStorage.getItem(STORAGE_KEY) || "fraud",
  previews: new Map(),
};

function fmtRatio(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)} : 1` : "—";
}

function datasetById(id) {
  return state.datasets.find((item) => item.id === id) || state.datasets[0];
}

function fullM(item) {
  return Math.max(0, Number(item.n_majority) - Number(item.n_minority));
}

function demoHref() {
  const params = new URLSearchParams();
  params.set("dataset", state.selected);
  params.set("t", String(Date.now()));
  return `./demo.html?${params.toString()}`;
}

function paintSelection() {
  const item = datasetById(state.selected);
  const m = fullM(item);
  selectedName.textContent = item.name;
  selectedBlurb.innerHTML = `${item.blurb}<br><span class="landing-howto">${item.howto || ""}</span>`;
  selectedStats.innerHTML = `
    <div><dt>Train</dt><dd>${item.n_train}</dd></div>
    <div><dt>Test</dt><dd>${item.n_test}</dd></div>
    <div><dt>n⁺ minority</dt><dd>${item.n_minority}</dd></div>
    <div><dt>n⁻ majority</dt><dd>${item.n_majority}</dd></div>
    <div><dt>m = n⁻ − n⁺</dt><dd>${m}</dd></div>
    <div><dt>Imbalance</dt><dd>${fmtRatio(item.imbalance_ratio)}</dd></div>
    <div class="landing-stats-wide"><dt>Features</dt><dd>${item.features.join(" · ")}</dd></div>
  `;
  if (scenarioSample) {
    scenarioSample.innerHTML = `
      Oversample the minority class with <code>k = 5</code>, <code>p = 3</code>,
      and <code>seed = 2207</code>; generate
      <code>m = n⁻ − n⁺ = ${item.n_majority} − ${item.n_minority} = ${m}</code> points.
    `;
  }
  const glossary = document.querySelector("#scenario-glossary");
  if (glossary) {
    glossary.innerHTML = `
      <code>n⁺</code> = minority = <strong>${item.n_minority}</strong> ·
      <code>n⁻</code> = majority = <strong>${item.n_majority}</strong> ·
      <code>m = n⁻ − n⁺</code> = <strong>${m}</strong> to balance the two classes.
    `;
  }
  startDemo.href = demoHref();
}

function selectDataset(id) {
  state.selected = id;
  localStorage.setItem(STORAGE_KEY, id);
  paintSelection();
  renderCards();
  paintAllPreviews();
}

function renderCards() {
  grid.innerHTML = state.datasets
    .map((item) => {
      const active = item.id === state.selected;
      const badge =
        item.group === "paper"
          ? '<span class="dataset-pill dataset-pill-muted">Paper §4.1</span>'
          : '<span class="dataset-pill dataset-pill-muted">Teaching</span>';
      return `
        <button
          class="dataset-card${active ? " is-active" : ""}"
          type="button"
          role="option"
          aria-selected="${active ? "true" : "false"}"
          data-dataset="${item.id}"
        >
          <div class="dataset-card-head">
            <strong>${item.name}</strong>
            ${active ? '<span class="dataset-pill">Selected</span>' : badge}
          </div>
          <p>${item.blurb}</p>
          <canvas class="dataset-preview" width="280" height="140" aria-hidden="true" data-preview="${item.id}"></canvas>
          <div class="dataset-meta">
            <span>${item.n_minority} / ${item.n_majority}</span>
            <span>${fmtRatio(item.imbalance_ratio)}</span>
          </div>
        </button>
      `;
    })
    .join("");

  grid.querySelectorAll("[data-dataset]").forEach((button) => {
    button.addEventListener("click", () => selectDataset(button.dataset.dataset));
  });
}

function drawPoints(canvas, points) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f7f9fc";
  ctx.fillRect(0, 0, width, height);
  if (!points?.length) return;

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const pad = 12;
  const sx = (width - pad * 2) / Math.max(maxX - minX, 1e-6);
  const sy = (height - pad * 2) / Math.max(maxY - minY, 1e-6);

  for (const point of points) {
    const x = pad + (point.x - minX) * sx;
    const y = height - (pad + (point.y - minY) * sy);
    ctx.beginPath();
    ctx.fillStyle = point.label === 1 ? "#345e9b" : "#9aa8b8";
    ctx.arc(x, y, point.label === 1 ? 2.4 : 1.7, 0, Math.PI * 2);
    ctx.fill();
  }
}

function paintAllPreviews() {
  for (const item of state.datasets) {
    const canvas = grid.querySelector(`canvas[data-preview="${item.id}"]`);
    if (!canvas) continue;
    drawPoints(canvas, state.previews.get(item.id) || []);
  }
}

async function loadPreview(id) {
  try {
    const response = await fetch(`${API}/api/fraud/scenario?dataset=${encodeURIComponent(id)}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    state.previews.set(id, data.train || []);
    const index = state.datasets.findIndex((item) => item.id === id);
    if (index >= 0) {
      state.datasets[index] = {
        ...state.datasets[index],
        name: data.name || state.datasets[index].name,
        blurb: data.blurb || state.datasets[index].blurb,
        howto: data.howto || state.datasets[index].howto,
        features: data.features || state.datasets[index].features,
        group: data.group || state.datasets[index].group,
        n_train: data.n_train,
        n_test: data.n_test,
        n_minority: data.n_minority,
        n_majority: data.n_majority,
        imbalance_ratio: data.imbalance_ratio,
      };
    }
    return true;
  } catch (error) {
    statusEl.textContent = error instanceof Error ? error.message : "Could not load preview";
    statusEl.classList.add("error");
    return false;
  }
}

async function boot() {
  renderCards();
  paintSelection();
  try {
    const response = await fetch(`${API}/api/fraud/datasets`);
    if (response.ok) {
      const payload = await response.json();
      if (Array.isArray(payload.datasets) && payload.datasets.length) {
        state.datasets = payload.datasets;
      }
    }
  } catch {
    statusEl.textContent = "Backend is not ready — using fallback stats.";
  }
  if (!state.datasets.some((item) => item.id === state.selected)) {
    state.selected = state.datasets[0].id;
  }
  renderCards();
  paintSelection();
  const ok = await Promise.all(state.datasets.map((item) => loadPreview(item.id)));
  if (ok.some(Boolean)) {
    statusEl.textContent = "";
    statusEl.classList.remove("error");
    renderCards();
    paintSelection();
    paintAllPreviews();
  }
}

boot();
