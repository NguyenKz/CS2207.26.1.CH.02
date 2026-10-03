import { forwardRef, memo, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactElement } from "react";

import { PredictDatasetView } from "./DatasetPanel";

const API_BASE = "http://localhost:6788";
const DRAWING_SIZE = 128;
const DRAWING_COUNT = DRAWING_SIZE * DRAWING_SIZE;

type ModelMeta = {
  id: string;
  name: string;
  kind: "linear" | "ann";
  architecture: number[];
  activations: string[];
  parameter_count: number;
  test_accuracy: number;
  validation_accuracy: number | null;
  source?: string;
  source_url?: string;
};

type PredictMeta = {
  ready: boolean;
  dataset: {
    name: string;
    description: string;
    sample_count: number;
    input_shape: [number, number];
    feature_count: number;
    class_count: number;
    pixel_min: number;
    pixel_max: number;
  };
  preprocessing: { name: string };
  test_indices: number[];
  default_sample_index: number;
  primary_model_id: string | null;
  training: {
    fit_samples: number;
    validation_samples: number;
    test_samples: number;
    total_samples: number;
    augmented_fit_samples: number;
    augmentation_factor: number;
    epochs: number;
    batch_size: number;
  };
  models: ModelMeta[];
};

type LayerTrace = {
  index: number;
  name: string;
  kind: string;
  activation: string;
  z: number[];
  h: number[];
  neuron_count: number;
};

type ModelResult = ModelMeta & {
  predicted_class: number;
  confidence: number;
  probabilities: number[];
  layers: LayerTrace[];
};

type PredictResult = {
  pixels: number[][];
  features_normalized: number[];
  sample_index: number | null;
  true_label: number | null;
  models: ModelResult[];
  preprocessing?: PreprocessingInfo;
};

type PreprocessingInfo = {
  source_size: [number, number];
  threshold: number;
  bounding_box: { x: number; y: number; width: number; height: number };
  cropped_size: [number, number];
  square_size: number;
  normalized_pixels: number[][];
};

type PredictStatus = "loading" | "ready" | "predicting" | "error";

function formatPercent(value: number | null): string {
  return value == null ? "n/a" : `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value: number, digits = 3): string {
  return Number.isFinite(value) ? value.toFixed(digits) : "n/a";
}

function formatInteger(value: number): string {
  return value.toLocaleString("en-US");
}

function flattenPixels(pixels: number[][]): number[] {
  return pixels.flat().map((value) => Math.max(0, Math.min(1, value)));
}

function defaultPixels(pixelSize: number): number[] {
  return Array.from({ length: pixelSize * pixelSize }, () => 0);
}

function blankDrawing(): number[] {
  return Array.from({ length: DRAWING_COUNT }, () => 0);
}

function sampleToDrawing(pixels: number[], pixelSize: number): number[] {
  return Array.from({ length: DRAWING_COUNT }, (_, index) => {
    const sourceY = Math.min(pixelSize - 1, Math.floor(Math.floor(index / DRAWING_SIZE) * pixelSize / DRAWING_SIZE));
    const sourceX = Math.min(pixelSize - 1, Math.floor((index % DRAWING_SIZE) * pixelSize / DRAWING_SIZE));
    return pixels[sourceY * pixelSize + sourceX] ?? 0;
  });
}

function normalizedColor(value: number): string {
  const alpha = Math.max(0.04, Math.min(1, value));
  return `rgba(23, 32, 39, ${alpha})`;
}

const NormalizedPreview = memo(function NormalizedPreview({
  pixels,
  pixelSize,
}: {
  pixels: number[] | null;
  pixelSize: number;
}): ReactElement {
  return (
    <div className="predict-normalized-block">
      <div className="predict-normalized-heading">
        <span className="section-kicker">NORMALIZED INPUT · {pixelSize}×{pixelSize}</span>
        <small>{pixels ? "sent to model" : "appears after Predict"}</small>
      </div>
      <div
        className="predict-normalized-grid"
        role="img"
        aria-label={`${pixelSize} by ${pixelSize} normalized digit input`}
        style={{ gridTemplateColumns: `repeat(${pixelSize}, minmax(0, 1fr))` }}
      >
        {(pixels ?? defaultPixels(pixelSize)).map((value, index) => (
          <span
            className="predict-normalized-cell"
            key={index}
            title={`Pixel ${index + 1}: ${value}`}
            style={{ backgroundColor: normalizedColor(value) }}
          />
        ))}
      </div>
    </div>
  );
});

type FreehandCanvasHandle = {
  save: () => void;
};

type FreehandCanvasProps = {
  drawing: number[];
  onCommit: (drawing: number[]) => void;
  downloadName: string;
};

const FreehandCanvas = memo(forwardRef<FreehandCanvasHandle, FreehandCanvasProps>(function FreehandCanvas({
  drawing,
  onCommit,
  downloadName,
}, ref): ReactElement {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawingRef = useRef(drawing);
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);
  const drawingActiveRef = useRef(false);

  function paintBackground(context: CanvasRenderingContext2D): void {
    context.fillStyle = "#172027";
    context.fillRect(0, 0, DRAWING_SIZE, DRAWING_SIZE);
  }

  function paintDrawing(context: CanvasRenderingContext2D, values: number[]): void {
    paintBackground(context);
    const cellSize = 1;
    for (let index = 0; index < DRAWING_COUNT; index += 1) {
      const value = values[index] ?? 0;
      if (value <= 0) continue;
      const x = index % DRAWING_SIZE;
      const y = Math.floor(index / DRAWING_SIZE);
      context.fillStyle = `rgba(251, 250, 246, ${Math.max(0.08, value)})`;
      context.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
    }
  }

  useEffect(() => {
    drawingRef.current = drawing;
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (context) paintDrawing(context, drawing);
  }, [drawing]);

  function getPoint(event: PointerEvent<HTMLCanvasElement>): { x: number; y: number } {
    const canvas = event.currentTarget;
    const bounds = canvas.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(DRAWING_SIZE - 1, (event.clientX - bounds.left) / bounds.width * DRAWING_SIZE)),
      y: Math.max(0, Math.min(DRAWING_SIZE - 1, (event.clientY - bounds.top) / bounds.height * DRAWING_SIZE)),
    };
  }

  function drawStroke(event: PointerEvent<HTMLCanvasElement>): void {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    const point = getPoint(event);
    const previous = lastPointRef.current ?? point;
    context.strokeStyle = "#fbfaf6";
    // Match MNIST stroke mass after 28×28 normalization.
    context.lineWidth = 5;
    context.lineCap = "round";
    context.lineJoin = "round";
    context.beginPath();
    context.moveTo(previous.x, previous.y);
    context.lineTo(point.x, point.y);
    context.stroke();
    lastPointRef.current = point;
  }

  function readCanvas(): number[] {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return drawingRef.current;
    const image = context.getImageData(0, 0, DRAWING_SIZE, DRAWING_SIZE).data;
    const backgroundBrightness = (23 + 32 + 39) / 3;
    const strokeBrightness = (251 + 250 + 246) / 3;
    const brightnessRange = strokeBrightness - backgroundBrightness;
    return Array.from({ length: DRAWING_COUNT }, (_, index) => {
      const brightness = (image[index * 4] + image[index * 4 + 1] + image[index * 4 + 2]) / 3;
      return Math.max(0, Math.min(1, (brightness - backgroundBrightness) / brightnessRange));
    });
  }

  function finishStroke(): void {
    if (!drawingActiveRef.current) return;
    drawingActiveRef.current = false;
    lastPointRef.current = null;
    const nextDrawing = readCanvas();
    drawingRef.current = nextDrawing;
    onCommit(nextDrawing);
  }

  function saveDrawing(): void {
    const canvas = canvasRef.current;
    if (!canvas || drawing.every((value) => value === 0)) return;
    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${downloadName}-${Date.now()}.png`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
    }, "image/png");
  }

  useImperativeHandle(ref, () => ({ save: saveDrawing }), [drawing, downloadName]);

  return (
    <>
      <canvas
        ref={canvasRef}
        className="predict-drawing-canvas"
        width={DRAWING_SIZE}
        height={DRAWING_SIZE}
        role="img"
        aria-label="Freehand digit drawing canvas"
        tabIndex={0}
        onPointerDown={(event) => {
          event.preventDefault();
          event.currentTarget.setPointerCapture(event.pointerId);
          drawingActiveRef.current = true;
          drawStroke(event);
        }}
        onPointerMove={(event) => {
          if (drawingActiveRef.current) drawStroke(event);
        }}
        onPointerUp={finishStroke}
        onPointerCancel={finishStroke}
      />
    </>
  );
}));

function ProbabilityBars({ probabilities, compact = false }: { probabilities: number[]; compact?: boolean }): ReactElement {
  return (
    <div className={`predict-probability-bars ${compact ? "is-compact" : ""}`} aria-label="Digit probabilities">
      {probabilities.map((probability, index) => (
        <div className="predict-probability-row" key={index}>
          <span>{index}</span>
          <span className="predict-probability-track">
            <span style={{ width: `${Math.max(1, probability * 100)}%` }} />
          </span>
          <code>{formatPercent(probability)}</code>
        </div>
      ))}
    </div>
  );
}

function ModelCard({
  model,
  trueLabel,
  selected,
  onSelect,
}: {
  model: ModelResult | ModelMeta;
  trueLabel: number | null;
  selected: boolean;
  onSelect: () => void;
}): ReactElement {
  const result = "predicted_class" in model ? model : null;
  const isCorrect = result && trueLabel != null ? result.predicted_class === trueLabel : null;

  return (
    <article className={`predict-model-card ${selected ? "is-selected" : ""}`}>
      <button
        className="predict-model-select"
        type="button"
        aria-pressed={selected}
        onClick={onSelect}
      >
        <span className="predict-model-topline">
          {model.kind === "ann" && <span className="section-kicker">ANN MODEL</span>}
          <span className="predict-model-state">{selected ? "Selected" : "Inspect"}</span>
        </span>
        <strong>{model.name}</strong>
        {model.source && <small className="predict-baseline-source">Official scikit-learn baseline</small>}
        <code>{model.architecture.join(" → ")}</code>
        <span className="predict-model-meta">
          {formatInteger(model.parameter_count)} parameters · test {formatPercent(model.test_accuracy)}
        </span>
      </button>
      <div className={`predict-model-result ${isCorrect === true ? "is-correct" : isCorrect === false ? "is-wrong" : ""}`}>
        {result ? (
          <>
            <div className="predict-model-prediction">
              <strong>{result.predicted_class}</strong>
              <code>{formatPercent(result.confidence)}</code>
            </div>
            <ProbabilityBars probabilities={result.probabilities} compact />
          </>
        ) : (
          <p className="predict-model-empty">Run Predict to see output probabilities.</p>
        )}
      </div>
    </article>
  );
}

function PredictInsights({ meta }: { meta: PredictMeta }): ReactElement {
  const [activeTab, setActiveTab] = useState<"training" | "benchmark">("training");
  const training = meta.training;
  const benchmarkModels = [...meta.models].sort((left, right) =>
    (right.validation_accuracy ?? -1) - (left.validation_accuracy ?? -1)
    || right.test_accuracy - left.test_accuracy,
  );

  return (
    <section className="predict-insights" aria-label="Training and benchmark details">
      <nav className="predict-insight-tabs" role="tablist" aria-label="Prediction details">
        {(["training", "benchmark"] as const).map((tab) => (
          <button
            key={tab}
            className={`predict-insight-tab ${activeTab === tab ? "is-active" : ""}`}
            type="button"
            role="tab"
            aria-selected={activeTab === tab}
            aria-controls={`predict-insight-${tab}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab === "training" ? "TRAINING" : "BENCHMARK"}
          </button>
        ))}
      </nav>

      {activeTab === "training" ? (
        <div id="predict-insight-training" className="predict-insight-panel" role="tabpanel">
          <div className="predict-insight-heading">
            <div>
              <span className="section-kicker">TRAINING RUN</span>
              <strong>MNIST 28×28 model set</strong>
            </div>
          </div>
          <div className="predict-training-grid">
            <div><span>Train</span><strong>{formatInteger(training.fit_samples)}</strong></div>
            <div><span>Validation</span><strong>{formatInteger(training.validation_samples)}</strong></div>
            <div><span>Test</span><strong>{formatInteger(training.test_samples)}</strong></div>
            <div><span>Total</span><strong>{formatInteger(training.total_samples)}</strong></div>
            <div><span>Epochs</span><strong>{formatInteger(training.epochs)}</strong></div>
            <div><span>Batch size</span><strong>{formatInteger(training.batch_size)}</strong></div>
          </div>
        </div>
      ) : (
        <div id="predict-insight-benchmark" className="predict-insight-panel" role="tabpanel">
          <div className="predict-insight-heading">
            <div>
              <span className="section-kicker">MODEL BENCHMARK</span>
              <strong>Validation and test accuracy</strong>
            </div>
            <small>higher is better</small>
          </div>
          <div className="predict-benchmark-list">
            {benchmarkModels.map((model) => (
              <div className={`predict-benchmark-row ${meta.primary_model_id === model.id ? "is-primary" : ""}`} key={model.id}>
                <div className="predict-benchmark-model">
                  <strong>{model.name}</strong>
                  <code>{model.architecture.join(" → ")}</code>
                </div>
                <div><span>Val</span><strong>{formatPercent(model.validation_accuracy)}</strong></div>
                <div><span>Test</span><strong>{formatPercent(model.test_accuracy)}</strong></div>
                <div><span>Params</span><strong>{formatInteger(model.parameter_count)}</strong></div>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function ActivationStrip({ layer, values = layer.h }: { layer: LayerTrace; values?: number[] }): ReactElement {
  const maxAbs = Math.max(...values.map((value) => Math.abs(value)), 1);
  return (
    <div className="predict-activation-strip">
      <div className="predict-activation-heading">
        <strong>{layer.name}</strong>
        <code>{layer.activation} · {layer.neuron_count} units</code>
      </div>
      <div className="predict-activation-values">
        {values.map((value, index) => {
          const positive = value >= 0;
          const intensity = Math.max(0.16, Math.abs(value) / maxAbs);
          return (
            <div className="predict-activation-cell" key={index} title={`z=${formatNumber(layer.z[index])}, h=${formatNumber(value)}`}>
              <span
                className={positive ? "is-positive" : "is-negative"}
                style={{ opacity: intensity }}
              />
              <small>{index + 1}</small>
              <code>{formatNumber(value, 2)}</code>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function PredictPanel(): ReactElement {
  const [activeDemoTab, setActiveDemoTab] = useState<"prediction" | "dataset">("prediction");
  const [meta, setMeta] = useState<PredictMeta | null>(null);
  const [pixels, setPixels] = useState<number[]>([]);
  const [drawing, setDrawing] = useState<number[]>(blankDrawing);
  const [normalizedPixels, setNormalizedPixels] = useState<number[] | null>(null);
  const [sampleIndex, setSampleIndex] = useState<number | null>(null);
  const [sampleLabel, setSampleLabel] = useState<number | null>(null);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [selectedModelId, setSelectedModelId] = useState("logistic");
  const [status, setStatus] = useState<PredictStatus>("loading");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const predictionSequenceRef = useRef(0);
  const canvasControlRef = useRef<FreehandCanvasHandle>(null);
  const pixelSize = meta?.dataset.input_shape[0] ?? 8;

  const selectedModelResult = useMemo(
    () => result?.models.find((model) => model.id === selectedModelId) ?? result?.models[0] ?? null,
    [result, selectedModelId],
  );

  async function fetchSample(index: number): Promise<void> {
    setErrorMessage(null);
    const response = await fetch(`${API_BASE}/predict/sample?index=${index}`);
    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      throw new Error(detail?.detail ?? `Could not load sample (${response.status})`);
    }
    const sample = (await response.json()) as { index: number; pixels: number[][]; label: number };
    const nextPixels = flattenPixels(sample.pixels);
    setPixels(nextPixels);
    setDrawing(sampleToDrawing(nextPixels, sample.pixels.length));
    setNormalizedPixels(nextPixels);
    setSampleIndex(sample.index);
    setSampleLabel(sample.label);
    setResult(null);
    setStatus("ready");
  }

  useEffect(() => {
    let cancelled = false;
    async function load(): Promise<void> {
      try {
        const response = await fetch(`${API_BASE}/predict/meta`);
        if (!response.ok) {
          const detail = await response.json().catch(() => null);
          throw new Error(detail?.detail ?? `Could not load model artifact (${response.status})`);
        }
        const data = (await response.json()) as PredictMeta;
        if (cancelled) return;
        setMeta(data);
        setSelectedModelId(data.primary_model_id ?? data.models[0]?.id ?? "logistic");
        if (data.default_sample_index !== undefined) await fetchSample(data.default_sample_index);
      } catch (error) {
        if (!cancelled) {
          setStatus("error");
          setErrorMessage(error instanceof Error ? error.message : String(error));
        }
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  const commitDrawing = useCallback((nextDrawing: number[]): void => {
    setDrawing(nextDrawing);
    setNormalizedPixels(null);
    setSampleIndex(null);
    setSampleLabel(null);
    setResult(null);
    setStatus("ready");
  }, []);

  useEffect(() => {
    if (sampleIndex !== null || drawing.every((value) => value === 0)) return undefined;
    const timer = window.setTimeout(() => {
      void handlePredict(drawing);
    }, 1000);
    return () => {
      window.clearTimeout(timer);
      predictionSequenceRef.current += 1;
    };
  }, [drawing, sampleIndex]);

  async function handleRandomSample(): Promise<void> {
    if (!meta?.test_indices.length) return;
    const randomPosition = Math.floor(Math.random() * meta.test_indices.length);
    try {
      await fetchSample(meta.test_indices[randomPosition]);
    } catch (error) {
      setStatus("error");
      setErrorMessage(error instanceof Error ? error.message : String(error));
    }
  }

  async function handlePredict(nextDrawing?: number[]): Promise<void> {
    const sequence = predictionSequenceRef.current + 1;
    predictionSequenceRef.current = sequence;
    setStatus("predicting");
    setErrorMessage(null);
    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(nextDrawing !== undefined ? { drawing: nextDrawing } : sampleIndex == null ? { drawing } : { pixels, sample_index: sampleIndex }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(detail?.detail ?? `Prediction failed (${response.status})`);
      }
      const prediction = (await response.json()) as PredictResult;
      if (sequence !== predictionSequenceRef.current) return;
      setResult(prediction);
      setNormalizedPixels(flattenPixels(prediction.pixels));
      setSelectedModelId((current) => prediction.models.some((model) => model.id === current) ? current : prediction.models[0].id);
      setStatus("ready");
    } catch (error) {
      if (sequence !== predictionSequenceRef.current) return;
      setStatus("error");
      setErrorMessage(error instanceof Error ? error.message : String(error));
    }
  }

  function clearPixels(): void {
    setPixels(defaultPixels(pixelSize));
    setDrawing(blankDrawing());
    setNormalizedPixels(null);
    setSampleIndex(null);
    setSampleLabel(null);
    setResult(null);
    setStatus("ready");
  }

  const displayLabel = sampleLabel == null ? "Custom drawing" : `True label · ${sampleLabel}`;
  const drawingFileName = `digit-${sampleLabel == null ? "custom" : `sample-${sampleIndex ?? "unknown"}-label-${sampleLabel}`}-${selectedModelResult ? `${selectedModelId}-predicted-${selectedModelResult.predicted_class}` : "unpredicted"}`;

  function handleDemoTabKeyDown(event: KeyboardEvent<HTMLButtonElement>): void {
    if (!["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const nextTab = event.key === "Home"
      ? "prediction"
      : event.key === "End"
        ? "dataset"
        : event.key === "ArrowRight" || event.key === "ArrowDown"
          ? "dataset"
          : "prediction";
    setActiveDemoTab(nextTab);
    window.requestAnimationFrame(() => document.getElementById(`demo-tab-${nextTab}`)?.focus());
  }

  return (
    <div className={`predict-shell ${activeDemoTab === "dataset" ? "is-dataset" : ""}`}>
      <nav className="predict-subtabs" role="tablist" aria-label="Demo views">
        <button
          id="demo-tab-prediction"
          className={`predict-subtab ${activeDemoTab === "prediction" ? "is-active" : ""}`}
          type="button"
          role="tab"
          aria-selected={activeDemoTab === "prediction"}
          aria-controls="prediction-board-panel"
          onKeyDown={handleDemoTabKeyDown}
          onClick={() => setActiveDemoTab("prediction")}
        >
          PREDICTION BOARD
        </button>
        <button
          id="demo-tab-dataset"
          className={`predict-subtab ${activeDemoTab === "dataset" ? "is-active" : ""}`}
          type="button"
          role="tab"
          aria-selected={activeDemoTab === "dataset"}
          aria-controls="dataset-browser-panel"
          onKeyDown={handleDemoTabKeyDown}
          onClick={() => setActiveDemoTab("dataset")}
        >
          DATASET
        </button>
      </nav>

      {activeDemoTab === "dataset" ? (
        <div id="dataset-browser-panel" role="tabpanel" aria-label="MNIST dataset browser">
          <PredictDatasetView />
        </div>
      ) : (
        <div id="prediction-board-panel" role="tabpanel" aria-label="Prediction board">
      {status === "loading" && (
        <section className="panel-surface predict-state" aria-live="polite">
          <span className="section-kicker">LOADING MODEL ARTIFACT</span>
          <h2>Preparing the four digit readers</h2>
          <p>Loading the saved weights and normalized MNIST test samples.</p>
        </section>
      )}

      {status === "error" && (
        <section className="panel-surface predict-state predict-state-error" role="alert">
          <span className="section-kicker">PREDICT UNAVAILABLE</span>
          <h2>Could not prepare the demo</h2>
          <p>{errorMessage}</p>
          <p>Run the offline export command, then restart the backend.</p>
        </section>
      )}

      {meta && status !== "loading" && status !== "error" && (
        <div className="predict-layout">
          <aside className="predict-input-column" aria-label="Digit input controls">
            <section className="panel-surface predict-input-panel">
              <div className="section-heading compact">
                <div>
                  <div className="section-kicker">INPUT CANVAS</div>
                </div>
                <p className="predict-input-label">{displayLabel}</p>
              </div>
              <div className="predict-pixel-frame">
                <FreehandCanvas ref={canvasControlRef} drawing={drawing} onCommit={commitDrawing} downloadName={drawingFileName} />
                <div className="predict-pixel-scale"><span>freehand</span><span>128×128 canvas</span><span>ink</span></div>
                <NormalizedPreview pixels={normalizedPixels} pixelSize={pixelSize} />
                {result?.preprocessing && (
                  <p className="predict-preprocess-meta">
                    Crop {result.preprocessing.bounding_box.width}×{result.preprocessing.bounding_box.height}
                    {" · "}square {result.preprocessing.square_size}px{ " · " }resize {pixelSize}×{pixelSize}
                  </p>
                )}
              </div>
              <div className="predict-sample-actions">
                <button className="button button-quiet" type="button" onClick={() => void handleRandomSample()}>Random sample</button>
              </div>
              <div className="predict-control-actions">
                <button className="button button-primary" type="button" onClick={() => void handlePredict()} disabled={status === "predicting"}>
                  {status === "predicting" ? "Predicting…" : "Predict this digit"}
                </button>
                <button className="button button-quiet" type="button" onClick={clearPixels}>Clear grid</button>
                <button
                  className="button button-quiet predict-icon-button"
                  type="button"
                  aria-label="Save drawing as PNG"
                  title="Save drawing as PNG"
                  onClick={() => canvasControlRef.current?.save()}
                  disabled={drawing.every((value) => value === 0)}
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                    <path d="M12 3v11m0 0 4-4m-4 4-4-4M4 20h16" />
                  </svg>
                </button>
              </div>
              <p className="predict-input-note">Draw freely. The backend removes noise, crops the ink, squares it and resizes it to {pixelSize}×{pixelSize} before prediction.</p>
              <PredictInsights meta={meta} />
            </section>
          </aside>

          <main className="predict-result-column">
            <section className="panel-surface predict-overview" aria-label="Prediction overview">
              <div className="section-heading compact">
                <div>
                  <div className="section-kicker">PREDICTION BOARD</div>
                </div>
                <p>{result ? `Predicted from ${result.sample_index == null ? "custom pixels" : `test sample ${result.sample_index}`}` : "Choose a sample, then run one forward pass."}</p>
              </div>
              <div className="predict-model-grid">
                {meta.models.map((model) => {
                  const resultModel = result?.models.find((item) => item.id === model.id);
                  return (
                    <ModelCard
                      key={model.id}
                      model={resultModel ?? model}
                      trueLabel={result?.true_label ?? sampleLabel}
                      selected={selectedModelId === model.id}
                      onSelect={() => setSelectedModelId(model.id)}
                    />
                  );
                })}
              </div>
            </section>

            {selectedModelResult && (
              <section className="panel-surface predict-flow-panel" aria-label="Layer activations">
                <div className="predict-activation-panel">
                  <div className="section-kicker">LAYER ACTIVATIONS</div>
                  <div className="predict-activation-list">
                    {selectedModelResult.layers.map((layer) => (
                      <ActivationStrip
                        layer={layer}
                        values={layer.kind === "output" ? selectedModelResult.probabilities : layer.h}
                        key={layer.index}
                      />
                    ))}
                  </div>
                </div>
              </section>
            )}
          </main>
        </div>
      )}
        </div>
      )}
    </div>
  );
}
