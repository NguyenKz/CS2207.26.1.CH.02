const TAU = Math.PI * 2;

export function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function randomNormal(random) {
  const radius = Math.sqrt(-2 * Math.log(Math.max(random(), Number.EPSILON)));
  return radius * Math.cos(TAU * random());
}

function distanceSquared(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const delta = a[i] - b[i];
    sum += delta * delta;
  }
  return sum;
}

export function nearestNeighbors(points, index, k) {
  return points
    .map((point, candidate) => ({
      candidate,
      distance: candidate === index ? Infinity : distanceSquared(points[index], point),
    }))
    .sort((left, right) => left.distance - right.distance)
    .slice(0, Math.min(k, points.length - 1))
    .map(({ candidate }) => candidate);
}

export function buildSymmetricGraph(points, k) {
  const adjacency = points.map(() => new Set());
  points.forEach((_, index) => {
    nearestNeighbors(points, index, k).forEach((neighbor) => {
      adjacency[index].add(neighbor);
      adjacency[neighbor].add(index);
    });
  });
  return adjacency;
}

function combinations(values, size, start = 0, chosen = [], result = []) {
  if (chosen.length === size) {
    result.push([...chosen]);
    return result;
  }
  for (let index = start; index <= values.length - (size - chosen.length); index += 1) {
    chosen.push(values[index]);
    combinations(values, size, index + 1, chosen, result);
    chosen.pop();
  }
  return result;
}

function isClique(candidate, adjacency) {
  for (let left = 0; left < candidate.length; left += 1) {
    for (let right = left + 1; right < candidate.length; right += 1) {
      if (!adjacency[candidate[left]].has(candidate[right])) return false;
    }
  }
  return true;
}

export function enumerateSimplices(adjacency, maxDimension) {
  const simplices = [];
  const maxSize = Math.max(2, maxDimension + 1);

  function visit(candidate, available) {
    if (candidate.length >= 2) simplices.push([...candidate]);
    if (candidate.length === maxSize) return;
    available.forEach((vertex, offset) => {
      const nextAvailable = available.slice(offset + 1).filter(
        (other) => adjacency[vertex].has(other),
      );
      visit([...candidate, vertex], nextAvailable);
    });
  }

  visit([], adjacency.map((_, index) => index));

  // Keep maximal cliques in the selected p-skeleton, matching Algorithm 1.
  return simplices.filter((simplex) => !simplices.some(
    (larger) => larger.length > simplex.length
      && simplex.every((vertex) => larger.includes(vertex)),
  ));
}

export function getGraphEdges(adjacency) {
  const edges = [];
  adjacency.forEach((neighbors, from) => {
    neighbors.forEach((to) => {
      if (from < to) edges.push([from, to]);
    });
  });
  return edges;
}

function sampleDirichlet(size, random) {
  const values = Array.from({ length: size }, () => -Math.log(Math.max(random(), Number.EPSILON)));
  const total = values.reduce((sum, value) => sum + value, 0);
  return values.map((value) => value / total);
}

function makeSyntheticPoint(points, parents, weights) {
  const dimensions = points[0].length;
  const point = Array(dimensions).fill(0);
  parents.forEach((parent, index) => {
    point.forEach((_, dimension) => {
      point[dimension] += weights[index] * points[parent][dimension];
    });
  });
  return point;
}

export function randomOversample(points, count, random) {
  return Array.from({ length: count }, () => {
    const parent = Math.floor(random() * points.length);
    return { point: [...points[parent]], parents: [parent], weights: [1], dimension: 0 };
  });
}

export function smote(points, count, k, random) {
  return Array.from({ length: count }, () => {
    const parent = Math.floor(random() * points.length);
    const neighbor = nearestNeighbors(points, parent, k)[Math.floor(random() * Math.min(k, points.length - 1))];
    const weight = random();
    return {
      point: points[parent].map((value, dimension) => value + weight * (points[neighbor][dimension] - value)),
      parents: [parent, neighbor],
      weights: [1 - weight, weight],
      dimension: 1,
    };
  });
}

export function simplicialSmote(points, count, k, dimension, random) {
  const adjacency = buildSymmetricGraph(points, k);
  const simplices = enumerateSimplices(adjacency, dimension);
  if (!simplices.length) return smote(points, count, k, random);

  return Array.from({ length: count }, () => {
    const parents = simplices[Math.floor(random() * simplices.length)];
    const weights = sampleDirichlet(parents.length, random);
    return {
      point: makeSyntheticPoint(points, parents, weights),
      parents,
      weights,
      dimension: parents.length - 1,
    };
  });
}

function makeMoonPoints(count, offset, random) {
  return Array.from({ length: count }, (_, index) => {
    const angle = Math.PI * (index / Math.max(count - 1, 1));
    const noiseX = randomNormal(random) * 0.035;
    const noiseY = randomNormal(random) * 0.035;
    return offset === 0
      ? [Math.cos(angle) + noiseX, Math.sin(angle) + noiseY]
      : [1 - Math.cos(angle) + noiseX, 0.35 - Math.sin(angle) + noiseY];
  });
}

function makeRingPoints(count, innerRadius, outerRadius, random) {
  return Array.from({ length: count }, () => {
    const angle = random() * TAU;
    const radius = innerRadius + random() * (outerRadius - innerRadius);
    return [radius * Math.cos(angle), radius * Math.sin(angle)];
  });
}

export function makeTopologyDataset(kind = "moons", seed = 2207) {
  const random = mulberry32(seed);
  const minority = kind === "ring"
    ? makeRingPoints(72, 0.72, 0.98, random)
    : makeMoonPoints(72, 0, random);
  const majority = kind === "ring"
    ? Array.from({ length: 240 }, () => {
      const angle = random() * TAU;
      const radius = 1.25 + random() * 0.55;
      return [radius * Math.cos(angle), radius * Math.sin(angle)];
    })
    : makeMoonPoints(240, 1, random).map(([x, y]) => [x + randomNormal(random) * 0.12, y + randomNormal(random) * 0.12]);

  const holdoutSize = Math.floor(minority.length * 0.2);
  return {
    name: kind === "ring" ? "Concentric ring" : "Two moons",
    dimensions: 2,
    minority,
    observedMinority: minority.slice(0, minority.length - holdoutSize),
    holdoutMinority: minority.slice(minority.length - holdoutSize),
    majority,
    source: "Synthetic, fixed seed",
  };
}

export function makeTeachingDataset() {
  const minority = [
    [-0.86, -0.28], [-0.04, -0.34], [0.73, -0.17],
    [-0.34, 0.48], [0.42, 0.64], [1.02, 0.48],
    [-0.95, 0.92], [0.16, 1.02], [0.88, 1.01],
  ];
  const majority = [
    [-1.55, -0.92], [-1.18, -0.95], [-0.72, -0.98], [-0.25, -0.94],
    [0.2, -0.95], [0.66, -0.93], [1.14, -0.89], [1.5, -0.62],
    [-1.54, -0.35], [1.54, -0.18], [-1.48, 0.15], [1.48, 0.25],
    [-1.52, 0.58], [1.51, 0.72], [-1.36, 1.02], [-0.96, 1.37],
    [-0.48, 1.42], [0.04, 1.45], [0.55, 1.42], [1.05, 1.37],
    [-0.7, 0.05], [0.85, 0.04], [-1.2, 0.55], [1.22, 0.98],
  ];
  return {
    name: "Teaching triangle",
    dimensions: 2,
    minority,
    observedMinority: minority,
    holdoutMinority: [],
    majority,
    teachingEdge: [0, 1],
    teachingTriangle: [0, 1, 3],
    source: "Deterministic teaching fixture",
  };
}

export function interpolateEdge(points, parents, lambda) {
  const [first, second] = parents;
  return points[first].map((value, dimension) => (
    value + lambda * (points[second][dimension] - value)
  ));
}

export function interpolateSimplex(points, parents, weights) {
  return points[parents[0]].map((_, dimension) => parents.reduce(
    (sum, parent, index) => sum + weights[index] * points[parent][dimension],
    0,
  ));
}

function parseEcoli(text) {
  const rows = text.trim().split(/\r?\n/).map((line) => (
    line.includes(",") ? line.trim().split(",") : line.trim().split(/\s+/)
  ));
  const points = rows
    .filter((row) => row.length >= 8 && row[0] !== "ID")
    .map((row) => {
      const hasId = row.length >= 9 && Number.isNaN(Number(row[0]));
      const valueStart = hasId ? 1 : 0;
      return {
        values: row.slice(valueStart, valueStart + 7).map(Number),
        label: row[valueStart + 7],
      };
    });
  if (!points.length || points.some(({ values }) => values.some(Number.isNaN))) {
    throw new Error("Could not parse the Ecoli format.");
  }
  const means = points[0].values.map((_, index) => points.reduce((sum, row) => sum + row.values[index], 0) / points.length);
  const scales = points[0].values.map((_, index) => Math.sqrt(
    points.reduce((sum, row) => sum + (row.values[index] - means[index]) ** 2, 0) / points.length,
  ) || 1);
  const normalized = points.map(({ values, label }) => ({
    values: values.map((value, index) => (value - means[index]) / scales[index]),
    label,
  }));
  const minority = normalized.filter(({ label }) => label === "imU").map(({ values }) => values);
  const majority = normalized.filter(({ label }) => label !== "imU").map(({ values }) => values);
  const holdoutSize = Math.max(1, Math.floor(minority.length * 0.2));
  return {
    name: "Ecoli — imU vs rest",
    dimensions: 7,
    minority,
    observedMinority: minority.slice(0, minority.length - holdoutSize),
    holdoutMinority: minority.slice(minority.length - holdoutSize),
    majority,
    source: "UCI Ecoli, z-score normalized; imU is the minority class",
  };
}

export async function loadEcoli() {
  const urls = [
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/ecoli.csv",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data",
  ];
  for (const url of urls) {
    try {
      const response = await fetch(url);
      if (response.ok) return parseEcoli(await response.text());
    } catch {
      // Try the next public mirror.
    }
  }
  throw new Error("Could not load Ecoli from UCI or a mirror. Check your network connection.");
}

function nearestDistance(point, candidates) {
  return Math.sqrt(Math.min(...candidates.map((candidate) => distanceSquared(point, candidate))));
}

function percentile(values, fraction) {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))] || 0;
}

export function calculateMetrics(dataset, generated, originalMinority = dataset.observedMinority) {
  const observedDistances = originalMinority.map((point, index) => nearestDistance(
    point,
    originalMinority.filter((_, candidate) => candidate !== index),
  ));
  const radius = percentile(observedDistances, 0.6) * 1.15;
  const coverage = dataset.holdoutMinority.length
    ? dataset.holdoutMinority.filter((point) => nearestDistance(point, generated.map(({ point: value }) => value)) <= radius).length / dataset.holdoutMinority.length
    : 0;
  const intrusion = generated.length
    ? generated.filter(({ point }) => {
      const nearestMajority = nearestDistance(point, dataset.majority);
      const nearestMinority = nearestDistance(point, originalMinority);
      return nearestMajority < nearestMinority;
    }).length / generated.length
    : 0;
  const nearestMinorityDistances = generated.map(({ point }) => nearestDistance(point, originalMinority));
  const occupiedCells = new Set(generated.map(({ point }) => point.map((value) => Math.round(value * 8)).join(","))).size;
  const duplicateRate = generated.length ? 1 - occupiedCells / generated.length : 0;
  return {
    coverage,
    intrusion,
    meanDistance: nearestMinorityDistances.reduce((sum, value) => sum + value, 0) / Math.max(generated.length, 1),
    diversity: generated.length ? occupiedCells / generated.length : 0,
    duplicateRate,
    radius,
  };
}

export function validateGenerated(points, adjacency) {
  points.forEach(({ point, parents, weights }) => {
    if (weights.some((weight) => weight < -1e-10) || Math.abs(weights.reduce((sum, value) => sum + value, 0) - 1) > 1e-8) {
      throw new Error("Invalid barycentric weights.");
    }
    parents.forEach((parent) => {
      if (!point.every((value, dimension) => Number.isFinite(value) && Number.isFinite(points[0]?.point?.[dimension] ?? value))) {
        throw new Error("Synthetic point is not finite.");
      }
      if (parent < 0 || parent >= adjacency.length) throw new Error("Parent index out of range.");
    });
    for (let left = 0; left < parents.length; left += 1) {
      for (let right = left + 1; right < parents.length; right += 1) {
        if (!adjacency[parents[left]].has(parents[right])) throw new Error("Parent vertices do not form a clique.");
      }
    }
  });
  return true;
}

export function generateMethod(dataset, method, options) {
  const random = mulberry32(options.seed);
  const start = performance.now();
  const generators = {
    random: () => randomOversample(dataset.observedMinority, options.budget, random),
    smote: () => smote(dataset.observedMinority, options.budget, options.k, random),
    simplicial: () => simplicialSmote(dataset.observedMinority, options.budget, options.k, options.p, random),
  };
  const generated = generators[method]();
  const adjacency = buildSymmetricGraph(dataset.observedMinority, options.k);
  const elapsed = performance.now() - start;
  if (method === "simplicial") validateGenerated(generated, adjacency);
  return {
    generated,
    adjacency,
    edges: getGraphEdges(adjacency),
    simplices: enumerateSimplices(adjacency, options.p),
    metrics: calculateMetrics(dataset, generated),
    elapsed,
  };
}
