// Local two-port run.sh keeps API on :8000; ngrok/same-origin share uses "".
const API =
  location.hostname === "127.0.0.1" || location.hostname === "localhost"
    ? "http://127.0.0.1:8000"
    : "";
const STORAGE_KEY = "ssmote-demo-dataset";
const DATASET_IDS = new Set(["fraud", "moons", "swiss_roll", "gaussian_circle", "circles"]);
const params = new URLSearchParams(window.location.search);
const requestedDataset = params.get("dataset") || localStorage.getItem(STORAGE_KEY) || "fraud";
const DATASET = DATASET_IDS.has(requestedDataset) ? requestedDataset : "fraud";
localStorage.setItem(STORAGE_KEY, DATASET);
try {
  localStorage.removeItem("ssmote-demo-budget");
} catch {
  /* ignore */
}

const METHODS = [
  // Labels match paper §4.1 table: Imbalanced / Gaussian / Random / Global / SMOTE / Simplicial.
  { id: "original", label: "Imbalanced", note: "No oversampling" },
  { id: "gaussian", label: "Gaussian", note: "Fit minority distribution" },
  { id: "random", label: "Random", note: "Duplicate minority points" },
  { id: "global", label: "Global", note: "Convex combination of a random minority pair" },
  { id: "smote", label: "SMOTE", note: "Interpolate along edges" },
  { id: "simplicial", label: "Simplicial SMOTE", note: "Generate inside simplices" },
];
const COLORS = {
  bg: "#eef1f5",
  panel: "#ffffff",
  panelBorder: "#e5e7eb",
  normal: "#64748b",
  fraud: "#dc2626",
  test: "#9333a3",
  synthetic: "#059669",
  ink: "#1f2937",
  muted: "#6b7280",
  accent: "#4b6b9a",
  grid: "#d5dbe6",
  simplex: "#c45b6a33",
  good: "#3d7a55",
  // Shared light chrome
  sideBg: "#f8f9fb",
  sideBorder: "#e5e7eb",
  sideInk: "#1f2937",
  sideMuted: "#6b7280",
  sideCard: "#ffffff",
  sideCardDone: "#f3f4f6",
  sideCardCurrent: "#eef2f7",
  sideBtn: "#ffffff",
  sideBtnHot: "#eef2f7",
  calloutBg: "#ffffffee",
};
const PHASES = ["select-parent", "connect-neighborhood", "show-weights", "interpolate", "commit-sample"];
const PHASE_LABELS = {
  "select-parent": "Select a cluster from the pool",
  "connect-neighborhood": "Connect the neighborhood (visual only)",
  "show-weights": "Draw e with RNG → normalize λ",
  interpolate: "Interpolate the new point",
  "commit-sample": "Commit synthetic sample",
};
const PHASE_LABELS_BY_METHOD = {
  random: {
    "select-parent": "Select a minority parent to duplicate",
    "commit-sample": "Commit duplicated sample",
  },
  global: {
    "select-parent": "Select a random minority pair",
    "show-weights": "Draw e with RNG → normalize λ",
    interpolate: "Interpolate along the pair",
    "commit-sample": "Commit synthetic sample",
  },
  gaussian: {
    "select-parent": "Draw from minority covariance",
    "commit-sample": "Commit Gaussian sample",
  },
};
const PHASE_DURATIONS = {
  "select-parent": 2200,
  "connect-neighborhood": 2000,
  "show-weights": 2800,
  interpolate: 2600,
  "commit-sample": 2300,
};
const PHASE_NAMES = PHASES.map((id) => PHASE_LABELS[id]);
const DURATIONS = PHASES.map((id) => PHASE_DURATIONS[id]);

function usesNeighborhoodGraph(methodId = state.active) {
  return methodId === "smote" || methodId === "simplicial";
}

function phaseLabel(methodId, phaseId) {
  return PHASE_LABELS_BY_METHOD[methodId]?.[phaseId] || PHASE_LABELS[phaseId] || phaseId;
}

function phasesFor(methodId, result) {
  const item = result?.timeline?.[0];
  if (item?.phases?.length) return item.phases;
  if (methodId === "random" || methodId === "gaussian") return ["select-parent", "commit-sample"];
  if (methodId === "global") return ["select-parent", "show-weights", "interpolate", "commit-sample"];
  return PHASES;
}

function activePhaseList(result = state.results.get(state.active)) {
  return phasesFor(state.active, result);
}

function activePhaseName(result = state.results.get(state.active)) {
  return activePhaseList(result)[state.phaseIndex] || PHASES[state.phaseIndex];
}

function fmtPoint(point) {
  const p = pointObject(point);
  return `(${Number(p.x).toFixed(2)}, ${Number(p.y).toFixed(2)})`;
}

function poolChoiceLabel(item) {
  const size = Number(item?.pool_size) || 0;
  const index = Number(item?.pool_index) || 0;
  if (size <= 0) return "pool —";
  const reuse = Number(item?.parent_reuse_count) || 0;
  return `cluster #${index + 1}/${size}${reuse > 1 ? ` · reused ${reuse}×` : ""}`;
}

function drawPlotCallout(plot, lines, alphas) {
  const boxX = plot.x + 14;
  const boxY = plot.y + 14;
  const boxW = Math.min(420, plot.w - 28);
  const boxH = 16 + lines.length * 18;
  ctx.fillStyle = COLORS.calloutBg;
  roundRect(boxX, boxY, boxW, boxH, 10);
  ctx.fill();
  ctx.strokeStyle = COLORS.accent;
  ctx.lineWidth = 1.4;
  ctx.stroke();
  lines.forEach((line, lineIndex) => {
    const alpha = alphas?.[lineIndex] ?? 1;
    if (alpha <= 0) return;
    ctx.fillStyle = lineIndex === 0 ? COLORS.accent : lineIndex === 1 ? COLORS.sideInk : COLORS.good;
    ctx.font = lineIndex <= 1 ? "700 13px system-ui" : "600 12px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.globalAlpha = alpha;
    ctx.fillText(line, boxX + 12, boxY + 10 + lineIndex * 18);
  });
  ctx.globalAlpha = 1;
}

function lambdaCalcLines(item) {
  const parents = item?.parents || [];
  const weights = item?.weights || [];
  const draws = item?.exp_draws || [];
  const d = Math.max(parents.length, 1);
  const ones = Array.from({ length: d }, () => "1").join(",");
  const sumE = draws.reduce((acc, value) => acc + Number(value), 0);
  if (!draws.length) {
    return parents.map((parent, index) => `λ_${parent + 1} = ${Number(weights[index]).toFixed(3)}`);
  }
  const lines = [
    `1) RNG: eⱼ ~ Exp(1)  (Dir ${ones})`,
  ];
  parents.forEach((parent, index) => {
    lines.push(`   e_${parent + 1} ← ${Number(draws[index]).toFixed(3)}`);
  });
  lines.push(`2) Σe = ${sumE.toFixed(3)}`);
  lines.push(`3) λⱼ = eⱼ / Σe`);
  parents.forEach((parent, index) => {
    const e = Number(draws[index]);
    const w = Number(weights[index]);
    lines.push(`   λ_${parent + 1} = ${e.toFixed(3)}/${sumE.toFixed(3)} = ${w.toFixed(3)}`);
  });
  lines.push(`Σλ = ${weights.reduce((a, v) => a + Number(v), 0).toFixed(3)}`);
  return lines;
}

function stepCalcLines(item, stepIndex) {
  if (!item) return ["—"];
  const parents = item.parents || [];
  const weights = item.weights || [];
  const parentPoints = (item.parent_points || []).map(pointObject);
  const labels = parents.map((parent) => `P${parent + 1}`);
  const phaseName = (item.phases || PHASES)[stepIndex] || PHASES[stepIndex];
  const method = state.active;

  if (phaseName === "select-parent") {
    if (method === "gaussian") {
      return ["N(μ, Σ) over minority train", `sample → ${fmtPoint(item.point)}`];
    }
    if (!parents.length) return ["No parent selected"];
    if (method === "random") {
      const parent = parentPoints[0];
      return [
        `parent → ${labels[0] || "P"}`,
        parent ? `copy ${fmtPoint(parent)}` : "duplicate minority point",
      ].filter(Boolean);
    }
    if (method === "global") {
      const lines = [`random minority pair (${parents.length})`];
      labels.forEach((label, index) => {
        const point = parentPoints[index];
        lines.push(point ? `${label} = ${fmtPoint(point)}` : label);
      });
      return lines;
    }
    const size = Number(item.pool_size) || 0;
    const lines = [
      `pool size = ${size || "—"}`,
      item.parent_reused ? `same parent simplex · used ${item.parent_reuse_count}×` : "",
      `random → ${poolChoiceLabel(item)}`,
    ];
    labels.forEach((label, index) => {
      const point = parentPoints[index];
      lines.push(point ? `${label} = ${fmtPoint(point)}` : label);
    });
    return lines.filter(Boolean);
  }

  if (phaseName === "connect-neighborhood") {
    if (parents.length < 2) {
      return ["Visual only (no new point)", `Neighborhood: ${labels.join(", ") || "—"}`];
    }
    const edges = ["Visual only (no new point)"];
    for (let i = 0; i < parents.length; i += 1) {
      for (let j = i + 1; j < parents.length; j += 1) {
        const a = parentPoints[i];
        const b = parentPoints[j];
        if (!a || !b) continue;
        const dist = Math.hypot(a.x - b.x, a.y - b.y);
        edges.push(`${labels[i]}↔${labels[j]}  d=${dist.toFixed(2)}`);
      }
    }
    return edges.length > 1 ? edges : ["Visual only (no new point)", `Connect ${labels.join(" – ")}`];
  }

  if (phaseName === "show-weights") {
    if (!weights.length) return ["λ not available"];
    return lambdaCalcLines(item);
  }

  if (phaseName === "interpolate") {
    if (!weights.length || !parentPoints.length) return ["Waiting for barycentric weights"];
    const terms = labels.map((label, index) => `${Number(weights[index]).toFixed(2)}·${label}`);
    const result = {
      x: parentPoints.reduce((acc, point, index) => acc + point.x * Number(weights[index] || 0), 0),
      y: parentPoints.reduce((acc, point, index) => acc + point.y * Number(weights[index] || 0), 0),
    };
    return [`P = ${terms.join(" + ")}`, `P = ${fmtPoint(result)}`];
  }

  if (method === "gaussian") return [`Commit ${fmtPoint(item.point)}`, "no parent links"];
  if (method === "random") {
    return [`Commit ${fmtPoint(item.point)}`, `duplicate of ${labels[0] || "parent"}`];
  }
  return [
    `Commit ${fmtPoint(item.point)}`,
    `parents [${labels.join(", ")}]`,
    `dim=${item.dimension}`,
  ];
}

const state = {
  results: new Map(),
  active: "original",
  selectedTest: null,
  selectedSynthetic: null,
  running: false,
  playing: false,
  itemIndex: 0,
  phaseIndex: 0,
  phaseElapsed: 0,
  speed: 0.001,
  visibleSynthetic: 0,
  raf: 0,
  lastFrame: 0,
  camera: null,
  cameraTarget: null,
  focusEnabled: false,
  viewport: { width: 1, height: 1, dpr: 1 },
  dragging: false,
  dragMoved: false,
  dragStart: null,
  hitRegions: [],
  hoverId: null,
  status: "Not run",
  dataset: DATASET,
  animTime: 0,
  commitBurst: 0,
};

document.head.insertAdjacentHTML("beforeend", `<link rel="stylesheet" href="./fraud.css">`);
document.body.innerHTML = `<canvas id="fraud-canvas" aria-label="Fraud oversampling single canvas"></canvas>`;

const canvas = document.querySelector("#fraud-canvas");
const ctx = canvas.getContext("2d");
// Debug/test hook for focus + camera state.
window.__fraudDemo = state;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const easeInOut = (t) => (t < 0.5 ? 4 * t ** 3 : 1 - ((-2 * t + 2) ** 3) / 2);
const easeOut = (t) => 1 - (1 - t) ** 3;
const pointObject = (p) => (Array.isArray(p) ? { x: p[0], y: p[1] } : p);

function drawGlow(sx, sy, radius, color, alpha = 0.35) {
  const gradient = ctx.createRadialGradient(sx, sy, 0, sx, sy, radius);
  gradient.addColorStop(0, color);
  gradient.addColorStop(1, "rgba(0,0,0,0)");
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(sx, sy, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawDashedProgress(from, to, t, color, width = 2) {
  const x = from[0] + (to[0] - from[0]) * clamp(t, 0, 1);
  const y = from[1] + (to[1] - from[1]) * clamp(t, 0, 1);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash([7, 6]);
  ctx.lineDashOffset = -state.animTime * 0.08;
  ctx.globalAlpha = 0.95;
  ctx.beginPath();
  ctx.moveTo(from[0], from[1]);
  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
  return [x, y];
}

function parentConnectionPairs(count) {
  const pairs = [];
  for (let i = 0; i < count; i += 1) {
    for (let j = i + 1; j < count; j += 1) pairs.push([i, j]);
  }
  return pairs;
}

/** 2σ covariance ellipse of minority points in data space (for Gaussian visuals). */
function minorityCovarianceEllipse(points) {
  if (!points || points.length < 2) return null;
  const n = points.length;
  let mx = 0;
  let my = 0;
  for (const p of points) {
    mx += p.x;
    my += p.y;
  }
  mx /= n;
  my /= n;
  let cxx = 0;
  let cxy = 0;
  let cyy = 0;
  for (const p of points) {
    const dx = p.x - mx;
    const dy = p.y - my;
    cxx += dx * dx;
    cxy += dx * dy;
    cyy += dy * dy;
  }
  const denom = Math.max(n - 1, 1);
  cxx /= denom;
  cxy /= denom;
  cyy /= denom;
  const trace = cxx + cyy;
  const det = cxx * cyy - cxy * cxy;
  const disc = Math.sqrt(Math.max(0, (trace * trace) / 4 - det));
  const l1 = Math.max(trace / 2 + disc, 0);
  const l2 = Math.max(trace / 2 - disc, 0);
  let angle = 0;
  if (Math.abs(cxy) > 1e-12) angle = Math.atan2(l1 - cxx, cxy);
  else if (cxx < cyy) angle = Math.PI / 2;
  return {
    cx: mx,
    cy: my,
    rx: Math.sqrt(l1) * 2,
    ry: Math.sqrt(l2) * 2,
    angle,
  };
}

function drawCovarianceEllipse(ellipse, result, view, alpha = 0.55) {
  if (!ellipse) return;
  const center = toScreen({ x: ellipse.cx, y: ellipse.cy }, result, view);
  const right = toScreen({ x: ellipse.cx + Math.cos(ellipse.angle) * ellipse.rx, y: ellipse.cy + Math.sin(ellipse.angle) * ellipse.rx }, result, view);
  const up = toScreen(
    {
      x: ellipse.cx + Math.cos(ellipse.angle + Math.PI / 2) * ellipse.ry,
      y: ellipse.cy + Math.sin(ellipse.angle + Math.PI / 2) * ellipse.ry,
    },
    result,
    view,
  );
  const rx = Math.hypot(right[0] - center[0], right[1] - center[1]);
  const ry = Math.hypot(up[0] - center[0], up[1] - center[1]);
  const screenAngle = Math.atan2(right[1] - center[1], right[0] - center[0]);
  ctx.save();
  ctx.translate(center[0], center[1]);
  ctx.rotate(screenAngle);
  ctx.beginPath();
  ctx.ellipse(0, 0, Math.max(rx, 1), Math.max(ry, 1), 0, 0, Math.PI * 2);
  ctx.fillStyle = "#05966922";
  ctx.globalAlpha = alpha;
  ctx.fill();
  ctx.strokeStyle = COLORS.synthetic;
  ctx.lineWidth = 1.6;
  ctx.setLineDash([6, 4]);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
  ctx.globalAlpha = 1;
}

function drawConnectionEdge(a, b, t, color, width = 2.4) {
  const tip = drawDashedProgress(a, b, t, color, width);
  if (t > 0.02 && t < 0.999) {
    drawGlow(tip[0], tip[1], 12, color, 0.5);
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.arc(tip[0], tip[1], 3.4, 0, Math.PI * 2);
    ctx.fill();
  }
  if (t >= 1) {
    // traveling particle along completed edge
    const travel = (state.animTime * 0.00055) % 1;
    const px = a[0] + (b[0] - a[0]) * travel;
    const py = a[1] + (b[1] - a[1]) * travel;
    drawGlow(px, py, 10, color, 0.4);
    ctx.beginPath();
    ctx.fillStyle = "#fff";
    ctx.arc(px, py, 2.4, 0, Math.PI * 2);
    ctx.fill();
  }
  return tip;
}
const roundRect = (x, y, w, h, r) => {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
};

function layout() {
  const w = state.viewport.width;
  const h = state.viewport.height;
  const pad = 12;
  const sideW = Math.min(360, Math.max(280, w * 0.28));
  return {
    pad,
    plot: {
      x: pad,
      y: pad,
      w: w - pad * 2 - sideW - 10,
      h: h - pad * 2,
    },
    side: {
      x: w - pad - sideW,
      y: pad,
      w: sideW,
      h: h - pad * 2,
    },
  };
}

function allPoints(result) {
  return [...(result.train || []), ...(result.test || [])];
}

function worldBounds(result) {
  const points = allPoints(result);
  if (!points.length) return { xMin: -1, xMax: 1, yMin: -1, yMax: 1 };
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  return {
    xMin: Math.min(...xs) - 0.35,
    xMax: Math.max(...xs) + 0.35,
    yMin: Math.min(...ys) - 0.35,
    yMax: Math.max(...ys) + 0.35,
  };
}

function fitCamera(result) {
  const bounds = worldBounds(result);
  return {
    x: (bounds.xMin + bounds.xMax) / 2,
    y: (bounds.yMin + bounds.yMax) / 2,
    scale: 1,
  };
}

function focusCameraForItem(result, item) {
  if (!item) return fitCamera(result);
  const points = [
    ...(item.parent_points || []).map(pointObject),
    pointObject(item.point),
  ];
  if (!points.length) return fitCamera(result);
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const bounds = worldBounds(result);
  const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 0.12);
  const overall = Math.max(bounds.xMax - bounds.xMin, bounds.yMax - bounds.yMin, 1);
  return {
    x: (Math.min(...xs) + Math.max(...xs)) / 2,
    y: (Math.min(...ys) + Math.max(...ys)) / 2,
    scale: clamp(overall / (span * 1.05), 4.0, 12),
  };
}

function syncFocusTarget(result) {
  if (!state.focusEnabled || !result?.timeline?.length) {
    state.cameraTarget = null;
    return;
  }
  const item = activeItem(result);
  state.cameraTarget = focusCameraForItem(result, item);
}

function updateFocusCamera(delta) {
  if (!state.focusEnabled || !state.camera || !state.cameraTarget || state.dragging) return;
  const ease = 1 - Math.exp(-(delta || 16) * 0.008);
  state.camera.x += (state.cameraTarget.x - state.camera.x) * ease;
  state.camera.y += (state.cameraTarget.y - state.camera.y) * ease;
  state.camera.scale += (state.cameraTarget.scale - state.camera.scale) * ease;
}

function plotView(result, plot) {
  const bounds = worldBounds(result);
  const camera = state.camera || fitCamera(result);
  const baseScale =
    Math.min(plot.w / (bounds.xMax - bounds.xMin), plot.h / (bounds.yMax - bounds.yMin)) * 0.88;
  return { plot, bounds, camera, scale: baseScale * camera.scale };
}

function toScreen(point, result, view) {
  return [
    view.plot.x + view.plot.w / 2 + (point.x - view.camera.x) * view.scale,
    view.plot.y + view.plot.h / 2 - (point.y - view.camera.y) * view.scale,
  ];
}

function fromScreen(sx, sy, result, plot) {
  const view = plotView(result, plot);
  return {
    x: view.camera.x + (sx - (plot.x + plot.w / 2)) / view.scale,
    y: view.camera.y - (sy - (plot.y + plot.h / 2)) / view.scale,
  };
}

function addHit(region) {
  state.hitRegions.push(region);
}

function hitAt(x, y) {
  for (let i = state.hitRegions.length - 1; i >= 0; i -= 1) {
    const r = state.hitRegions[i];
    if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) return r;
  }
  return null;
}

function resize() {
  const dpr = window.devicePixelRatio || 1;
  state.viewport = { width: window.innerWidth, height: window.innerHeight, dpr };
  canvas.width = Math.floor(window.innerWidth * dpr);
  canvas.height = Math.floor(window.innerHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function activeItem(result) {
  return result?.timeline?.[state.itemIndex] || null;
}

function drawPanel(x, y, w, h, alpha = 0.92, light = false) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = light ? COLORS.sideBg : COLORS.panel;
  roundRect(x, y, w, h, 12);
  ctx.fill();
  ctx.strokeStyle = light ? COLORS.sideBorder : COLORS.panelBorder;
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.restore();
}

function drawButton(id, x, y, w, h, label, active = false) {
  const hot = state.hoverId === id;
  ctx.save();
  ctx.fillStyle = active ? COLORS.accent : hot ? COLORS.sideBtnHot : COLORS.sideBtn;
  roundRect(x, y, w, h, 8);
  ctx.fill();
  ctx.strokeStyle = active ? COLORS.accent : COLORS.sideBorder;
  ctx.stroke();
  ctx.fillStyle = active ? "#ffffff" : COLORS.sideInk;
  ctx.font = `600 ${Math.min(14, Math.max(12, h * 0.45))}px system-ui`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, x + w / 2, y + h / 2 + 0.5);
  ctx.restore();
  addHit({ id, x, y, w, h, type: "button" });
}

function drawChip(id, x, y, w, h, title, active) {
  const hot = state.hoverId === id;
  ctx.save();
  ctx.fillStyle = active ? COLORS.sideCardCurrent : hot ? COLORS.sideBtnHot : COLORS.sideCard;
  roundRect(x, y, w, h, 7);
  ctx.fill();
  ctx.strokeStyle = active ? COLORS.accent : COLORS.sideBorder;
  ctx.lineWidth = active ? 1.5 : 1;
  ctx.stroke();
  ctx.fillStyle = active ? COLORS.accent : COLORS.sideInk;
  ctx.font = "700 13px system-ui";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(title, x + w / 2, y + h / 2 + 0.5);
  ctx.restore();
  addHit({ id, x, y, w, h, type: "method", method: id.replace("method-", "") });
}

function wrapText(text, x, y, maxWidth, lineHeight) {
  const words = String(text).split(" ");
  let line = "";
  let yy = y;
  let lines = 0;
  words.forEach((word) => {
    const test = line ? `${line} ${word}` : word;
    if (ctx.measureText(test).width > maxWidth && line) {
      ctx.fillText(line, x, yy);
      line = word;
      yy += lineHeight;
      lines += 1;
    } else line = test;
  });
  if (line) {
    ctx.fillText(line, x, yy);
    lines += 1;
  }
  return lines;
}

function drawSideMenu(ui, result) {
  const { x, y, w, h } = ui.side;
  drawPanel(x, y, w, h, 1, true);

  let cy = y + 12;
  const inner = 12;
  const contentW = w - inner * 2;

  // Back to landing (own row so hit target is not covered)
  drawButton("home", x + inner, cy, contentW, 30, "← Datasets", false);
  cy += 36;
  // One primary action: load data if needed, otherwise play/pause animation
  const hasTimeline = !!result?.timeline?.length;
  const runLabel = state.running
    ? "Loading…"
    : !hasTimeline
      ? "Load data"
      : state.playing
        ? "❚❚ Pause"
        : "▶ Run";
  drawButton("run", x + inner, cy, contentW * 0.55, 30, runLabel, hasTimeline && state.playing);
  ctx.fillStyle = COLORS.sideMuted;
  ctx.font = "12px system-ui";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillText(state.status, x + inner + contentW, cy + 15);
  ctx.textAlign = "left";
  cy += 36;
  ctx.fillStyle = COLORS.sideMuted;
  ctx.font = "12px system-ui";
  ctx.fillText(`Dataset: ${state.dataset}`, x + inner, cy);
  cy += 18;

  // Legend compact
  const items = [
    [COLORS.normal, "Majority"],
    [COLORS.fraud, "Minority"],
    [COLORS.test, "Test"],
    [COLORS.synthetic, "Synthetic"],
  ];
  items.forEach(([color, label], i) => {
    const lx = x + inner + i * (contentW / 4);
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.arc(lx + 5, cy + 5, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = COLORS.sideMuted;
    ctx.font = "12px system-ui";
    ctx.textBaseline = "middle";
    ctx.fillText(label, lx + 13, cy + 5);
  });
  cy += 22;

  // Methods
  ctx.fillStyle = COLORS.sideMuted;
  ctx.font = "700 11px system-ui";
  ctx.textBaseline = "top";
  ctx.fillText("METHOD", x + inner, cy);
  cy += 15;
  const chipH = 28;
  const chipGap = 6;
  const chipW = (contentW - chipGap) / 2;
  METHODS.forEach((method, index) => {
    const col = index % 2;
    const row = Math.floor(index / 2);
    drawChip(
      `method-${method.id}`,
      x + inner + col * (chipW + chipGap),
      cy + row * (chipH + chipGap),
      chipW,
      chipH,
      method.label,
      state.active === method.id,
    );
  });
  cy += Math.ceil(METHODS.length / 2) * (chipH + chipGap) + 6;

  // Controls
  ctx.fillStyle = COLORS.sideMuted;
  ctx.font = "700 11px system-ui";
  ctx.fillText("CONTROLS", x + inner, cy);
  cy += 15;
  const bh = 26;
  const gap = 4;
  const row1 = [
    { id: "zoom-out", label: "−" },
    { id: "zoom-in", label: "+" },
    { id: "fit", label: "Fit" },
    { id: "focus", label: "Focus" },
  ];
  const row2 = [
    { id: "back", label: "‹" },
    { id: "forward", label: "›" },
    { id: "restart", label: "↻" },
  ];
  const ctrlW1 = (contentW - gap * 3) / 4;
  row1.forEach((c, i) => {
    drawButton(
      c.id,
      x + inner + i * (ctrlW1 + gap),
      cy,
      ctrlW1,
      bh,
      c.label,
      c.id === "focus" && state.focusEnabled,
    );
  });
  cy += bh + 5;
  const ctrlW2 = (contentW - gap * 2) / 3;
  row2.forEach((c, i) => {
    drawButton(c.id, x + inner + i * (ctrlW2 + gap), cy, ctrlW2, bh, c.label, false);
  });
  cy += bh + 5;

  const speedGap = 4;
  const speedBtnW = (contentW - speedGap * 2) / 3;
  const speedRow = [
    { id: "speed-down", label: "Slower" },
    { id: "speed-stop", label: "Stop" },
    { id: "speed-up", label: "Faster" },
  ];
  speedRow.forEach((c, i) => {
    drawButton(
      c.id,
      x + inner + i * (speedBtnW + speedGap),
      cy,
      speedBtnW,
      bh,
      c.label,
      c.id === "speed-stop" && !state.playing,
    );
  });
  cy += bh + 10;

  // Full step list for current sample (clipped so it cannot overflow outcome)
  const metricsH = 170;
  const stepsTop = cy;
  const stepsBottom = y + h - metricsH - 8;
  ctx.fillStyle = COLORS.sideMuted;
  ctx.font = "700 11px system-ui";
  ctx.fillText("STEPS", x + inner, cy);
  cy += 15;

  ctx.save();
  ctx.beginPath();
  ctx.rect(x + 1, stepsTop, w - 2, Math.max(0, stepsBottom - stepsTop));
  ctx.clip();

  if (!result || !result.timeline?.length) {
    ctx.fillStyle = COLORS.sideMuted;
    ctx.font = "13px system-ui";
    const emptyMsg = !result
      ? "Click Run to load data."
      : state.active === "original" || !(result.n_synthetic > 0)
        ? "This method does not generate synthetic samples."
        : "Animation steps will appear when the method exposes a timeline.";
    wrapText(emptyMsg, x + inner, cy, contentW, 16);
    cy += 40;
  } else {
    const item = activeItem(result);
    ctx.fillStyle = COLORS.accent;
    ctx.font = "700 12px system-ui";
    ctx.fillText(`Sample ${state.itemIndex + 1}/${result.timeline.length}`, x + inner, cy);
    ctx.fillStyle = COLORS.sideMuted;
    ctx.textAlign = "right";
    ctx.fillText(
      `p=${item?.parents?.length ?? "—"} vertices · dim ${item?.dimension ?? "—"}`,
      x + inner + contentW,
      cy,
    );
    ctx.textAlign = "left";
    cy += 18;

    const phaseIds = item?.phases || activePhaseList(result);
    phaseIds.forEach((phaseId, index) => {
      const name = phaseLabel(state.active, phaseId);
      const done = index < state.phaseIndex;
      const current = index === state.phaseIndex;
      // Compact non-current steps to avoid side-panel overflow.
      const lines = current || done ? stepCalcLines(item, index) : [];
      const showLines = current ? lines : done ? lines.slice(0, 1) : [];
      const lineH = 14;
      const titleH = 16;
      const padY = 8;
      const stepH = padY * 2 + titleH + showLines.length * lineH;
      const boxY = cy;
      if (boxY > stepsBottom) return;

      ctx.fillStyle = current ? COLORS.sideCardCurrent : done ? COLORS.sideCardDone : COLORS.sideCard;
      roundRect(x + inner, boxY, contentW, stepH, 7);
      ctx.fill();
      ctx.strokeStyle = current ? COLORS.accent : COLORS.sideBorder;
      ctx.lineWidth = current ? 1.5 : 1;
      ctx.stroke();

      ctx.beginPath();
      ctx.fillStyle = done ? COLORS.good : current ? COLORS.accent : "#9aa3af";
      ctx.arc(x + inner + 10, boxY + padY + 7, 3.5, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = current ? COLORS.sideInk : COLORS.sideMuted;
      ctx.font = "700 12px system-ui";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(`${index + 1}. ${name}`, x + inner + 20, boxY + padY);

      if (done) {
        ctx.fillStyle = COLORS.good;
        ctx.font = "11px system-ui";
        ctx.textAlign = "right";
        ctx.fillText("done", x + inner + contentW - 8, boxY + padY);
        ctx.textAlign = "left";
      }

      const lineMaxW = contentW - 28;
      showLines.forEach((line, lineIndex) => {
        ctx.fillStyle = current ? COLORS.accent : done ? "#4b5563" : "#9aa3af";
        ctx.font = "600 12px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.textAlign = "left";
        let text = String(line);
        while (text.length > 1 && ctx.measureText(text).width > lineMaxW) text = `${text.slice(0, -2)}…`;
        ctx.fillText(text, x + inner + 20, boxY + padY + titleH + lineIndex * lineH);
      });

      addHit({
        id: `step-${index}`,
        x: x + inner,
        y: boxY,
        w: contentW,
        h: stepH,
        type: "step",
        index,
      });
      cy += stepH + 4;
    });
  }
  ctx.restore();

  // Outcome metrics at bottom of side panel (fixed band)
  const metricsTop = y + h - metricsH;
  ctx.fillStyle = COLORS.sideMuted;
  ctx.font = "700 11px system-ui";
  ctx.textAlign = "left";
  ctx.fillText("OUTCOME", x + inner, metricsTop);

  if (!result) {
    ctx.fillStyle = COLORS.sideMuted;
    ctx.font = "13px system-ui";
    ctx.fillText("No data yet", x + inner, metricsTop + 20);
    return;
  }

  const metrics = result.metrics;
  const minorityCount = result.test.filter((p) => p.label === 1).length;
  const caught = minorityCount - metrics.false_negatives;
  ctx.fillStyle = COLORS.accent;
  ctx.font = "800 30px system-ui";
  ctx.fillText(String(caught), x + inner, metricsTop + 24);
  ctx.fillStyle = COLORS.sideMuted;
  ctx.font = "13px system-ui";
  ctx.fillText(`/ ${minorityCount} minority test`, x + inner + 52, metricsTop + 36);

  const rows = [
    ["Recall", `${(metrics.recall * 100).toFixed(0)}%`],
    ["MCC", metrics.mcc.toFixed(2)],
    ["F1", metrics.f1.toFixed(2)],
    ["FN", String(metrics.false_negatives)],
    ["FP", String(metrics.false_positives)],
  ];
  rows.forEach(([label, value], i) => {
    const ry = metricsTop + 58 + i * 22;
    ctx.fillStyle = COLORS.sideMuted;
    ctx.font = "13px system-ui";
    ctx.textAlign = "left";
    ctx.fillText(label, x + inner, ry);
    ctx.fillStyle = COLORS.sideInk;
    ctx.textAlign = "right";
    ctx.fillText(value, x + inner + contentW, ry);
  });
}

function drawPlot(ui, result) {
  const { plot } = ui;
  drawPanel(plot.x, plot.y, plot.w, plot.h, 1, true);
  ctx.save();
  ctx.beginPath();
  ctx.rect(plot.x + 1, plot.y + 1, plot.w - 2, plot.h - 2);
  ctx.clip();

  if (!result) {
    ctx.fillStyle = COLORS.sideMuted;
    ctx.font = "14px system-ui";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("Run to begin", plot.x + plot.w / 2, plot.y + plot.h / 2);
    ctx.restore();
    return;
  }

  const view = plotView(result, plot);

  // grid
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  ctx.globalAlpha = 0.9;
  const step = 0.5;
  for (let gx = Math.ceil(view.bounds.xMin / step) * step; gx <= view.bounds.xMax; gx += step) {
    const a = toScreen({ x: gx, y: view.bounds.yMin }, result, view);
    const b = toScreen({ x: gx, y: view.bounds.yMax }, result, view);
    ctx.beginPath();
    ctx.moveTo(...a);
    ctx.lineTo(...b);
    ctx.stroke();
  }
  for (let gy = Math.ceil(view.bounds.yMin / step) * step; gy <= view.bounds.yMax; gy += step) {
    const a = toScreen({ x: view.bounds.xMin, y: gy }, result, view);
    const b = toScreen({ x: view.bounds.xMax, y: gy }, result, view);
    ctx.beginPath();
    ctx.moveTo(...a);
    ctx.lineTo(...b);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // soft boundary (kept light so points stay on top visually)
  result.boundary.forEach((cell) => {
    const [sx, sy] = toScreen({ x: cell.x, y: cell.y }, result, view);
    const size = Math.max(4, view.scale * 0.16);
    ctx.fillStyle = cell.probability > 0.5 ? "#dbeafe" : "#f1f5f9";
    ctx.globalAlpha = 0.22 + Math.abs(cell.probability - 0.5) * 0.2;
    ctx.fillRect(sx - size / 2, sy - size / 2, size, size);
  });
  ctx.globalAlpha = 1;

  const minority = result.minority || result.train.filter((p) => p.label === 1);
  const item = activeItem(result);
  const method = state.active;
  const phaseName = activePhaseName(result);
  // Map named phases onto the original 0..4 ranks so short timelines still work.
  const phase = ({
    "select-parent": 0,
    "connect-neighborhood": 1,
    "show-weights": 2,
    interpolate: 3,
    "commit-sample": 4,
  })[phaseName] ?? state.phaseIndex;
  const progress = state.phaseElapsed / (PHASE_DURATIONS[phaseName] || DURATIONS[state.phaseIndex] || 1);
  const activeParents = new Set(item?.parents || []);
  const showKnn = usesNeighborhoodGraph(method);

  // Base k-NN edges only for local neighborhood methods (SMOTE / Simplicial).
  if (showKnn) {
    (result.edges || []).forEach(([a, b]) => {
      if (!minority[a] || !minority[b]) return;
      const active = activeParents.has(a) && activeParents.has(b);
      if (active) return;
      const p1 = toScreen(minority[a], result, view);
      const p2 = toScreen(minority[b], result, view);
      ctx.strokeStyle = "#94a3b8";
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 1.35;
      ctx.beginPath();
      ctx.moveTo(...p1);
      ctx.lineTo(...p2);
      ctx.stroke();
    });
    (result.edges || []).forEach(([a, b]) => {
      if (!minority[a] || !minority[b]) return;
      if (!(activeParents.has(a) && activeParents.has(b))) return;
      const p1 = toScreen(minority[a], result, view);
      const p2 = toScreen(minority[b], result, view);
      ctx.strokeStyle = COLORS.accent;
      ctx.globalAlpha = 1;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(...p1);
      ctx.lineTo(...p2);
      ctx.stroke();
    });
    ctx.globalAlpha = 1;
  }

  if (method === "gaussian") {
    drawCovarianceEllipse(minorityCovarianceEllipse(minority), result, view, item ? 0.7 : 0.45);
  }

  result.train
    .filter((p) => !p.label)
    .forEach((p) => {
      const [sx, sy] = toScreen(p, result, view);
      ctx.beginPath();
      ctx.fillStyle = COLORS.normal;
      ctx.globalAlpha = item ? 0.72 : 0.9;
      ctx.arc(sx, sy, item ? 3.2 : 3.6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#334155";
      ctx.globalAlpha = item ? 0.35 : 0.45;
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  ctx.globalAlpha = 1;

  const baseR = item ? 3.2 : 3.6;
  minority.forEach((p, index) => {
    const [sx, sy] = toScreen(p, result, view);
    const active = activeParents.has(index);
    ctx.beginPath();
    ctx.fillStyle = COLORS.fraud;
    ctx.globalAlpha = 1;
    // Same radius as majority so count imbalance stays visible; active only via stroke.
    ctx.arc(sx, sy, baseR, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = active ? "#ffffff" : "#7f1d1d";
    ctx.lineWidth = active ? 2 : 1;
    ctx.stroke();
  });
  ctx.globalAlpha = 1;

  result.test.forEach((p, index) => {
    if (!p.label) return;
    const [sx, sy] = toScreen(p, result, view);
    const selected = state.selectedTest === index;
    ctx.beginPath();
    ctx.fillStyle = COLORS.test;
    ctx.arc(sx, sy, selected ? 7.5 : 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#9a3412";
    ctx.lineWidth = selected ? 2 : 1.5;
    ctx.stroke();
    addHit({ id: `test-${index}`, x: sx - 8, y: sy - 8, w: 16, h: 16, type: "test", index });
  });

  if (item) {
    const parents = (item.parent_points || []).map(pointObject);
    const pulse = 0.5 + 0.5 * Math.sin(state.animTime * 0.006);
    const phaseT = easeOut(clamp(progress, 0, 1));
    // Random/Gaussian: no neighborhood mesh. Global: pair only. Local: full graph overlays.
    const showParentLinks = method === "global" || showKnn;
    const showSimplexFill = method === "simplicial" && parents.length > 2;
    const showWeights = method === "global" || showKnn;
    const showInterpolate = method === "global" || showKnn;

    // Phase 0: method-aware callout
    if (phase === 0) {
      let lines;
      if (method === "gaussian") {
        lines = ["Gaussian N(μ, Σ)", "ellipse = minority covariance", progress > 0.4 ? `sample → ${fmtPoint(item.point)}` : "sampling…"];
      } else if (method === "random") {
        lines = [
          "Duplicate one minority parent",
          progress > 0.28 ? (parents[0] ? `P${(item.parents[0] ?? 0) + 1} = ${fmtPoint(parents[0])}` : "parent selected") : "selecting…",
        ];
      } else if (method === "global") {
        lines = [
          "Random minority pair",
          ...((progress > 0.4 ? parents : []).map((parent, index) => {
            const label = `P${(item.parents[index] ?? index) + 1}`;
            return `${label} = ${fmtPoint(parent)}`;
          })),
        ];
      } else {
        lines = [
          `Pool size = ${Number(item.pool_size) || "—"}`,
          item.parent_reused ? `same parent simplex · used ${item.parent_reuse_count}×` : "",
          progress > 0.28 ? `random → ${poolChoiceLabel(item)}` : "random → …",
          ...((progress > 0.55 ? parents : []).map((parent, index) => {
            const label = `P${(item.parents[index] ?? index) + 1}`;
            return `${label} = ${fmtPoint(parent)}`;
          })).filter(Boolean),
        ];
      }
      lines = lines.filter(Boolean);
      drawPlotCallout(
        plot,
        lines,
        lines.map((_, lineIndex) => (lineIndex === 0 ? 1 : clamp((progress - 0.25) / 0.35, 0, 1))),
      );
    }

    // Phase 2: λ math for Global / SMOTE / Simplicial
    if (phase === 2 && showWeights && item.weights?.length) {
      drawPlotCallout(plot, lambdaCalcLines(item));
    }

    // Parent highlight (Random / Global / local). Gaussian has no parents.
    parents.forEach((parent, index) => {
      const appear =
        phase > 0
          ? 1
          : progress < 0.55
            ? 0
            : easeOut(clamp(((progress - 0.55) / 0.45) * parents.length - index, 0, 1));
      if (appear <= 0) return;
      const [sx, sy] = toScreen(parent, result, view);
      const radius = 5 + 4 * appear;
      drawGlow(sx, sy, 18 + 10 * pulse, COLORS.fraud, 0.18 + 0.12 * appear);
      ctx.beginPath();
      ctx.strokeStyle = COLORS.fraud;
      ctx.globalAlpha = 0.35 + 0.4 * pulse;
      ctx.lineWidth = 2;
      ctx.arc(sx, sy, radius + 6 + 5 * pulse * appear, 0, Math.PI * 2);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.fillStyle = COLORS.fraud;
      ctx.arc(sx, sy, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.6;
      ctx.stroke();
      ctx.fillStyle = COLORS.sideInk;
      ctx.font = "800 15px system-ui";
      ctx.textAlign = "left";
      ctx.globalAlpha = appear;
      ctx.fillText(`P${item.parents[index] + 1}`, sx + 11, sy - 12);
      ctx.globalAlpha = 1;
    });

    // Parent-parent connections: Global pair edge, or local neighborhood mesh.
    // Skip Random/Gaussian. For Global (no connect-neighborhood phase), show from weights onward.
    if (showParentLinks && parents.length >= 2 && (showKnn ? phase >= 1 : phase >= 2)) {
      const screens = parents.map((parent) => toScreen(parent, result, view));
      const pairs = parentConnectionPairs(parents.length);
      const pairCount = Math.max(pairs.length, 1);

      pairs.forEach(([i, j], pairIndex) => {
        let edgeT = 1;
        if (phase === 1) {
          const local = progress * pairCount - pairIndex;
          edgeT = easeOut(clamp(local, 0, 1));
        }
        if (edgeT <= 0) return;
        const a = screens[i];
        const b = screens[j];
        ctx.save();
        ctx.strokeStyle = COLORS.accent;
        ctx.globalAlpha = 0.18 + 0.2 * edgeT;
        ctx.lineWidth = 6;
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(a[0] + (b[0] - a[0]) * edgeT, a[1] + (b[1] - a[1]) * edgeT);
        ctx.stroke();
        ctx.restore();
        drawConnectionEdge(a, b, edgeT, COLORS.accent, 2.6);

        if (edgeT > 0.15) {
          const midX = a[0] + (b[0] - a[0]) * Math.min(edgeT, 1) * 0.5;
          const midY = a[1] + (b[1] - a[1]) * Math.min(edgeT, 1) * 0.5;
          ctx.fillStyle = COLORS.sideInk;
          ctx.font = "800 13px system-ui";
          ctx.textAlign = "center";
          ctx.globalAlpha = edgeT;
          ctx.fillText(`P${item.parents[i] + 1}–P${item.parents[j] + 1}`, midX, midY - 10);
          ctx.globalAlpha = 1;
        }
      });
    }

    // Simplicial only: simplex fill
    if (showSimplexFill && phase >= 2) {
      const fillT = phase === 2 ? phaseT : 1;
      ctx.beginPath();
      parents.forEach((parent, i) => {
        const [sx, sy] = toScreen(parent, result, view);
        if (i) ctx.lineTo(sx, sy);
        else ctx.moveTo(sx, sy);
      });
      ctx.closePath();
      ctx.globalAlpha = 0.2 + 0.55 * fillT;
      ctx.fillStyle = COLORS.simplex;
      ctx.fill();
      ctx.globalAlpha = 0.5 + 0.5 * fillT;
      ctx.strokeStyle = COLORS.accent;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Weight labels for Global / SMOTE / Simplicial
    if (showWeights && phase >= 2 && item.weights?.length) {
      parents.forEach((parent, index) => {
        const pop = phase === 2 ? easeOut(clamp((progress * parents.length - index) / 1.1, 0, 1)) : 1;
        if (pop <= 0) return;
        const [sx, sy] = toScreen(parent, result, view);
        ctx.save();
        ctx.globalAlpha = pop;
        ctx.fillStyle = COLORS.good;
        ctx.font = "800 14px ui-monospace, Menlo, monospace";
        ctx.textAlign = "left";
        ctx.fillText(`λ=${Number(item.weights[index] || 0).toFixed(2)}`, sx + 11, sy + 16);
        ctx.restore();
      });
    }

    // Interpolate spokes for Global / SMOTE / Simplicial only
    const target = parents.length
      ? {
          x: parents.reduce((s, p, i) => s + p.x * (item.weights[i] || 0), 0),
          y: parents.reduce((s, p, i) => s + p.y * (item.weights[i] || 0), 0),
        }
      : pointObject(item.point);
    if (showInterpolate && phase >= 3 && parents.length) {
      const t = phase === 3 ? easeInOut(clamp(progress, 0, 1)) : 1;
      const targetScreen = toScreen(target, result, view);
      const screens = parents.map((parent) => toScreen(parent, result, view));
      screens.forEach((a) => {
        const edgeT = phase === 3 ? t : 1;
        ctx.save();
        ctx.strokeStyle = COLORS.synthetic;
        ctx.globalAlpha = 0.12 + 0.2 * edgeT;
        ctx.lineWidth = 5;
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(a[0] + (targetScreen[0] - a[0]) * edgeT, a[1] + (targetScreen[1] - a[1]) * edgeT);
        ctx.stroke();
        ctx.restore();
        drawConnectionEdge(a, targetScreen, edgeT, COLORS.synthetic, 1.8);
      });
      const [sx, sy] = targetScreen;
      const appear = phase === 3 ? t : 1;
      drawGlow(sx, sy, 26 + 8 * pulse, COLORS.synthetic, 0.45 * appear);
      ctx.beginPath();
      ctx.strokeStyle = COLORS.synthetic;
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.7 * appear;
      ctx.arc(sx, sy, 12 + 5 * Math.sin(state.animTime * 0.01), 0, Math.PI * 2);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.fillStyle = COLORS.synthetic;
      ctx.globalAlpha = appear;
      ctx.arc(sx, sy, 4 + 3 * appear, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#145c3a";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Commit burst (all methods that generate samples)
    if (phase >= 4) {
      const [sx, sy] = toScreen(pointObject(item.point), result, view);
      const burst = phase === 4 ? phaseT : 1;
      for (let ring = 0; ring < 3; ring += 1) {
        ctx.beginPath();
        ctx.strokeStyle = COLORS.synthetic;
        ctx.globalAlpha = (1 - burst) * 0.55 + 0.15;
        ctx.lineWidth = 2;
        ctx.arc(sx, sy, 10 + burst * (18 + ring * 14), 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      drawGlow(sx, sy, 34, COLORS.synthetic, 0.35 + 0.25 * pulse);
    }
  }

  result.synthetic.slice(0, state.visibleSynthetic).forEach((sample, index) => {
    const [sx, sy] = toScreen({ x: sample.point[0], y: sample.point[1] }, result, view);
    const active = index === state.itemIndex;
    const selected = state.selectedSynthetic === index;
    const justBorn = active && activePhaseName(result) === "commit-sample";
    const birth = justBorn ? easeOut(clamp(progress, 0, 1)) : 1;
    const radius = (active || selected ? 7.5 : 5.4) * (0.4 + 0.6 * birth);
    if (active) drawGlow(sx, sy, 26, COLORS.synthetic, 0.3);
    ctx.beginPath();
    ctx.fillStyle = COLORS.synthetic;
    ctx.globalAlpha = (active || selected ? 1 : 0.85) * birth;
    ctx.arc(sx, sy, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#065f46";
    ctx.lineWidth = active || selected ? 2 : 1.4;
    ctx.stroke();
    ctx.globalAlpha = 1;
    addHit({
      id: `syn-${index}`,
      x: sx - 8,
      y: sy - 8,
      w: 16,
      h: 16,
      type: "synthetic",
      index,
    });
  });

  ctx.restore();
}

function draw() {
  const result = state.results.get(state.active);
  const ui = layout();
  state.hitRegions = [];
  ctx.clearRect(0, 0, state.viewport.width, state.viewport.height);
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, state.viewport.width, state.viewport.height);
  drawPlot(ui, result);
  drawSideMenu(ui, result);
}

function setTimelinePosition(position, play = false) {
  const result = state.results.get(state.active);
  if (!result?.timeline?.length) return;
  const nPhases = activePhaseList(result).length;
  const total = result.timeline.length * nPhases;
  const cursor = clamp(Math.round(position), 0, total - 1);
  state.itemIndex = Math.floor(cursor / nPhases);
  state.phaseIndex = cursor % nPhases;
  state.phaseElapsed = 0;
  state.visibleSynthetic = state.itemIndex + (state.phaseIndex === nPhases - 1 ? 1 : 0);
  state.playing = play;
  syncFocusTarget(result);
}

function resetTimeline(result, shouldPlay = false) {
  state.itemIndex = 0;
  state.phaseIndex = 0;
  state.phaseElapsed = 0;
  state.lastFrame = 0;
  state.playing = shouldPlay && !!result.timeline?.length;
  state.visibleSynthetic = 0;
  state.camera = state.focusEnabled
    ? focusCameraForItem(result, result.timeline?.[0])
    : fitCamera(result);
  state.cameraTarget = state.camera;
  state.selectedSynthetic = null;
  syncFocusTarget(result);
}

function advanceTimeline(result) {
  const nPhases = activePhaseList(result).length;
  if (state.phaseIndex === nPhases - 1) {
    state.visibleSynthetic = Math.max(state.visibleSynthetic, state.itemIndex + 1);
    state.itemIndex += 1;
    state.phaseIndex = 0;
  } else {
    state.phaseIndex += 1;
  }
  if (state.itemIndex >= result.timeline.length) {
    state.itemIndex = Math.max(0, result.timeline.length - 1);
    state.phaseIndex = nPhases - 1;
    state.playing = false;
    state.visibleSynthetic = result.timeline.length;
    state.status = "Completed";
  }
}

function animate(timestamp) {
  const result = state.results.get(state.active);
  const delta = state.lastFrame ? Math.min(48, timestamp - state.lastFrame) : 0;
  state.lastFrame = timestamp;
  state.animTime += delta;
  if (result?.timeline?.length && state.playing) {
    state.phaseElapsed += delta * state.speed;
    const phaseName = activePhaseName(result);
    const duration = reducedMotion ? 1 : (PHASE_DURATIONS[phaseName] || DURATIONS[state.phaseIndex] || 1);
    if (state.phaseElapsed >= duration) {
      state.phaseElapsed = 0;
      advanceTimeline(result);
      syncFocusTarget(result);
    }
  }
  syncFocusTarget(result);
  updateFocusCamera(delta);
  draw();
  state.raf = requestAnimationFrame(animate);
}

function runBody(methodId) {
  return {
    method: methodId,
    dataset: state.dataset,
    seed: 2207,
    k: 5,
    p: 3,
    budget_pct: 1,
  };
}

async function runAll() {
  if (state.running) return;
  state.running = true;
  state.status = "Running…";
  try {
    for (const method of METHODS) {
      const response = await fetch(`${API}/api/fraud/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(runBody(method.id)),
      });
      if (!response.ok) throw new Error(`Backend ${response.status}`);
      const data = await response.json();
      state.results.set(method.id, data);
      state.active = method.id;
      resetTimeline(data, false);
      state.status = `${method.label} loaded`;
      await new Promise((r) => setTimeout(r, 250));
    }
    state.active = "simplicial";
    resetTimeline(state.results.get("simplicial"), false);
    state.status = "Ready";
  } catch (error) {
    state.status = error instanceof Error ? error.message : "Could not run the demo";
  } finally {
    state.running = false;
  }
}

function handleAction(id) {
  const result = state.results.get(state.active);
  if (id === "home") {
    window.location.assign(`./index.html?from=demo&t=${Date.now()}`);
    return;
  }
  if (id === "run") {
    if (state.running) return;
    if (!result?.timeline?.length) return runAll();
    state.playing = !state.playing;
    return;
  }
  if (id.startsWith("method-")) {
    const method = id.replace("method-", "");
    if (!state.results.has(method)) return;
    state.active = method;
    resetTimeline(state.results.get(method), false);
    return;
  }
  if (!result) return;
  if (id === "zoom-in") return zoomAt(1.2);
  if (id === "zoom-out") return zoomAt(0.84);
  if (id === "fit") {
    state.focusEnabled = false;
    state.camera = fitCamera(result);
    state.cameraTarget = state.camera;
    return;
  }
  if (id === "focus") {
    state.focusEnabled = !state.focusEnabled;
    if (state.focusEnabled) {
      syncFocusTarget(result);
      if (state.cameraTarget) state.camera = { ...state.cameraTarget };
    }
    return;
  }
  if (id === "restart") return resetTimeline(result, false);
  if (id === "back") {
    const nPhases = activePhaseList(result).length;
    return setTimelinePosition(state.itemIndex * nPhases + state.phaseIndex - 1);
  }
  if (id === "forward") {
    const nPhases = activePhaseList(result).length;
    return setTimelinePosition(state.itemIndex * nPhases + state.phaseIndex + 1);
  }
  if (id === "speed-down" || id === "speed-up") {
    const speeds = [0.001, 0.01, 1];
    let index = speeds.indexOf(state.speed);
    if (index < 0) index = 0;
    index = id === "speed-up" ? Math.min(speeds.length - 1, index + 1) : Math.max(0, index - 1);
    state.speed = speeds[index];
    return;
  }
  if (id === "speed-stop") {
    state.playing = false;
    return;
  }
  if (id.startsWith("step-")) {
    const step = Number(id.replace("step-", ""));
    if (!Number.isFinite(step) || !result.timeline?.length) return;
    const nPhases = activePhaseList(result).length;
    setTimelinePosition(state.itemIndex * nPhases + step, false);
  }
}

function unfocusCamera() {
  state.focusEnabled = false;
  state.cameraTarget = null;
}

function zoomAt(factor, sx = state.viewport.width / 2, sy = state.viewport.height / 2) {
  const result = state.results.get(state.active);
  if (!result || !state.camera) return;
  unfocusCamera();
  const ui = layout();
  const before = fromScreen(sx, sy, result, ui.plot);
  state.camera.scale = clamp(state.camera.scale * factor, 0.45, 14);
  const after = fromScreen(sx, sy, result, ui.plot);
  state.camera.x += before.x - after.x;
  state.camera.y += before.y - after.y;
}

function panBy(dxScreen, dyScreen) {
  const result = state.results.get(state.active);
  if (!result || !state.camera) return;
  unfocusCamera();
  const ui = layout();
  const view = plotView(result, ui.plot);
  state.camera.x += dxScreen / view.scale;
  state.camera.y -= dyScreen / view.scale;
}

canvas.addEventListener("pointermove", (event) => {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  if (state.dragging && state.camera) {
    const result = state.results.get(state.active);
    if (!result) return;
    const ui = layout();
    const view = plotView(result, ui.plot);
    const dx = event.clientX - state.dragStart.x;
    const dy = event.clientY - state.dragStart.y;
    if (!state.dragMoved && dx * dx + dy * dy > 4) {
      state.dragMoved = true;
      unfocusCamera();
    }
    state.camera.x = state.dragStart.cameraX - dx / view.scale;
    state.camera.y = state.dragStart.cameraY + dy / view.scale;
    return;
  }
  const hit = hitAt(x, y);
  state.hoverId = hit?.id || null;
  canvas.classList.toggle("is-hot", !!hit && hit.type !== "plot");
});

canvas.addEventListener("pointerdown", (event) => {
  if (event.button !== 0) return;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const hit = hitAt(x, y);
  if (hit?.type === "button" || hit?.type === "method" || hit?.type === "step") {
    handleAction(hit.id);
    return;
  }
  if (hit?.type === "synthetic") {
    state.selectedSynthetic = hit.index;
    state.playing = false;
    return;
  }
  if (hit?.type === "test") {
    state.selectedTest = hit.index;
    return;
  }
  const result = state.results.get(state.active);
  if (!result || !state.camera) return;
  const ui = layout();
  if (x < ui.plot.x || y < ui.plot.y || x > ui.plot.x + ui.plot.w || y > ui.plot.y + ui.plot.h) return;
  state.dragging = true;
  state.dragMoved = false;
  state.dragStart = { x: event.clientX, y: event.clientY, cameraX: state.camera.x, cameraY: state.camera.y };
  canvas.setPointerCapture(event.pointerId);
  canvas.classList.add("is-dragging");
});

canvas.addEventListener("pointerup", (event) => {
  state.dragging = false;
  state.dragMoved = false;
  canvas.classList.remove("is-dragging");
  try {
    canvas.releasePointerCapture(event.pointerId);
  } catch {
    /* ignore */
  }
});

// Capture on window so trackpad scroll is not lost to the browser/page.
window.addEventListener(
  "wheel",
  (event) => {
    const rect = canvas.getBoundingClientRect();
    if (
      event.clientX < rect.left ||
      event.clientX > rect.right ||
      event.clientY < rect.top ||
      event.clientY > rect.bottom
    ) {
      return;
    }
    event.preventDefault();
    unfocusCamera();
    // Pinch-zoom (ctrl/cmd + wheel). Two-finger / mouse wheel pans.
    if (event.ctrlKey || event.metaKey) {
      zoomAt(
        event.deltaY < 0 ? 1.12 : 0.9,
        event.clientX - rect.left,
        event.clientY - rect.top,
      );
      return;
    }
    panBy(event.deltaX, event.deltaY);
  },
  { passive: false, capture: true },
);

window.addEventListener("resize", () => {
  resize();
});

resize();
state.raf = requestAnimationFrame(animate);
runAll();
