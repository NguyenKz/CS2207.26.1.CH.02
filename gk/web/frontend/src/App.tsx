import { useMemo, useRef, useState, type ReactElement } from "react";

import { PredictPanel } from "./PredictPanel";

const ACTIVATIONS = [
  "tanh",
  "sigmoid",
  "relu",
  "leaky_relu",
  "identity",
] as const;
const CLASS_COUNT = 10;
const SAMPLES_PER_CLASS = 1000;
const DEFAULT_SAMPLE_COUNT = CLASS_COUNT * SAMPLES_PER_CLASS;
const INPUT_FEATURE_OPTIONS = [8, 32, 128, 512, 1024] as const;
const HIDDEN_NEURON_OPTIONS = [8, 16, 32, 64, 128] as const;

type ActivationName = (typeof ACTIVATIONS)[number];
type RunState = "idle" | "connecting" | "running" | "completed" | "cancelled" | "error";
type ConnectionState = "disconnected" | "connecting" | "connected";

export type RunConfig = {
  epochs: number;
  learning_rate: number;
  difficulty: number;
  sample_count: number;
  batch_size: number;
  train_percentage: number;
  validation_percentage: number;
  test_percentage: number;
  input_feature_count: number;
  hidden_neuron_count: number;
  random_seed: number;
  early_stopping: boolean;
  early_stopping_patience: number;
  early_stopping_min_delta: number;
  activations: ActivationName[];
};

type Metric = {
  epoch: number;
  totalEpochs: number;
  trainingLoss: number | null;
  validationLoss: number | null;
  validationAccuracy: number | null;
  status: "idle" | "running" | "completed" | "cancelled" | "early_stopped";
};

type LossPoint = {
  epoch: number;
  trainingLoss: number;
  validationLoss: number;
};

type SummaryResult = {
  activation: ActivationName;
  epochs_completed: number;
  test_accuracy: number;
  training_loss: number | null;
  validation_loss: number | null;
  validation_accuracy: number | null;
  stopped_early?: boolean;
};

type SocketMessage = {
  type: string;
  run_id?: string;
  activation?: ActivationName;
  epoch?: number;
  total_epochs?: number;
  training_loss?: number;
  validation_loss?: number;
  validation_accuracy?: number;
  status?: Metric["status"] | "completed";
  message?: string;
  duration_ms?: number;
  difficulty?: number;
  sample_count?: number;
  class_count?: number;
  samples_per_class?: number;
  batch_size?: number;
  train_percentage?: number;
  validation_percentage?: number;
  test_percentage?: number;
  results?: SummaryResult[];
  dataset?: {
    training: number[];
    validation: number[];
    testing: number[];
  };
};

const DEFAULT_CONFIG: RunConfig = {
  epochs: 600,
  learning_rate: 0.08,
  difficulty: 0.7,
  sample_count: DEFAULT_SAMPLE_COUNT,
  batch_size: 32,
  train_percentage: 70,
  validation_percentage: 15,
  test_percentage: 15,
  input_feature_count: 32,
  hidden_neuron_count: 32,
  random_seed: 42,
  early_stopping: false,
  early_stopping_patience: 40,
  early_stopping_min_delta: 0.001,
  activations: [...ACTIVATIONS],
};

const ACTIVATION_LABELS: Record<ActivationName, string> = {
  tanh: "tanh",
  sigmoid: "Sigmoid",
  relu: "ReLU",
  leaky_relu: "Leaky ReLU",
  identity: "Identity",
};

/** One-line teaching cue: formula + what to watch while curves move. */
const ACTIVATION_CUES: Record<ActivationName, { formula: string; watch: string }> = {
  tanh: { formula: "tanh(z)", watch: "symmetric · can saturate" },
  sigmoid: { formula: "σ(z)", watch: "0 to 1 · saturates early" },
  relu: { formula: "max(0, z)", watch: "fast · negative inputs become 0" },
  leaky_relu: { formula: "max(αz, z)", watch: "ReLU with a small negative gradient" },
  identity: { formula: "z", watch: "no nonlinearity" },
};

const ACTIVATION_COLORS: Record<ActivationName, string> = {
  tanh: "#d05a45",
  sigmoid: "#2c7a7b",
  relu: "#b7862c",
  leaky_relu: "#6b5ca5",
  identity: "#435466",
};

const INITIAL_METRIC = (totalEpochs: number): Metric => ({
  epoch: 0,
  totalEpochs,
  trainingLoss: null,
  validationLoss: null,
  validationAccuracy: null,
  status: "idle",
});

function createInitialMetrics(totalEpochs: number): Record<ActivationName, Metric> {
  return Object.fromEntries(
    ACTIVATIONS.map((activation) => [activation, INITIAL_METRIC(totalEpochs)]),
  ) as Record<ActivationName, Metric>;
}

function createInitialHistory(): Record<ActivationName, LossPoint[]> {
  return Object.fromEntries(
    ACTIVATIONS.map((activation) => [activation, []]),
  ) as unknown as Record<ActivationName, LossPoint[]>;
}

function formatLoss(value: number | null): string {
  return value === null ? "n/a" : value.toFixed(4);
}

function formatAccuracy(value: number | null | undefined): string {
  return value == null ? "n/a" : `${(value * 100).toFixed(1)}%`;
}

function parameterCount(inputFeatureCount: number, hiddenNeuronCount: number): number {
  return inputFeatureCount * hiddenNeuronCount + hiddenNeuronCount
    + hiddenNeuronCount * CLASS_COUNT + CLASS_COUNT;
}

function difficultyDescription(difficulty: number): string {
  if (difficulty < 0.34) return "Low noise · classes are easy to separate";
  if (difficulty < 0.67) return "Moderate noise · harder boundaries";
  return "High noise · classes overlap";
}

function StatusMark({ status }: { status: Metric["status"] }): ReactElement {
  const labels = {
    idle: "Idle",
    running: "Running",
    completed: "Completed",
    cancelled: "Cancelled",
    early_stopped: "Early stopped",
  };
  return (
    <span className={`status status-${status}`}>
      <span className="status-mark" aria-hidden="true" />
      {labels[status]}
    </span>
  );
}

function Sparkline({
  points,
  color,
  label,
}: {
  points: LossPoint[];
  color: string;
  label: string;
}): ReactElement {
  const values = points.flatMap((point) => [point.trainingLoss, point.validationLoss]);
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 1;
  const range = Math.max(max - min, 0.001);
  const width = 280;
  const height = 48;
  const toPoint = (value: number, index: number): string => {
    const x = points.length === 1 ? width : (index / (points.length - 1)) * width;
    const y = height - ((value - min) / range) * (height - 6) - 3;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  };
  const trainingPolyline = points.map((point, index) => toPoint(point.trainingLoss, index)).join(" ");
  const validationPolyline = points.map((point, index) => toPoint(point.validationLoss, index)).join(" ");

  return (
    <div className="sparkline-frame">
      <svg className="sparkline" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label={`${label}: train loss solid, validation loss dashed`}>
        <line x1="0" y1="0" x2="0" y2={height - 1} className="sparkline-axis" vectorEffect="non-scaling-stroke" />
        <line x1="0" y1={height - 1} x2={width} y2={height - 1} className="sparkline-baseline" vectorEffect="non-scaling-stroke" />
        {trainingPolyline && <polyline points={trainingPolyline} fill="none" stroke={color} strokeWidth="2.2" vectorEffect="non-scaling-stroke" />}
        {validationPolyline && <polyline points={validationPolyline} fill="none" stroke={color} strokeWidth="1.6" strokeDasharray="4 3" opacity="0.58" vectorEffect="non-scaling-stroke" />}
      </svg>
      <span className="sparkline-max-label">{max.toFixed(2)}</span>
      <span className="sparkline-y-label">loss</span>
      <span className="sparkline-x-label">epoch</span>
      <span className="sparkline-key"><span className="sparkline-key-train" /> train <span className="sparkline-key-validation" /> val</span>
    </div>
  );
}

function LossChart({
  histories,
  visibleActivations,
}: {
  histories: Record<ActivationName, LossPoint[]>;
  visibleActivations: Set<ActivationName>;
}): ReactElement {
  const width = 900;
  const height = 300;
  const padding = { top: 24, right: 24, bottom: 34, left: 50 };
  const allValues = ACTIVATIONS.flatMap((activation) =>
    histories[activation].flatMap((point) => [point.trainingLoss, point.validationLoss]),
  );
  const maxValue = Math.max(...allValues, 1);
  const minValue = Math.min(...allValues, 0);
  const xRange = width - padding.left - padding.right;
  const yRange = height - padding.top - padding.bottom;
  const maxEpoch = Math.max(...ACTIVATIONS.map((activation) => histories[activation].length), 1);
  const toPoint = (value: number, index: number): string => {
    const x = padding.left + (index / Math.max(maxEpoch - 1, 1)) * xRange;
    const y = padding.top + (1 - (value - minValue) / Math.max(maxValue - minValue, 0.001)) * yRange;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  };

  return (
    <div className="chart-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label="Live loss chart">
        <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} className="chart-axis" />
        <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} className="chart-axis" />
        <text x={padding.left - 12} y={padding.top + 4} className="chart-label" textAnchor="end">{maxValue.toFixed(2)}</text>
        <text x={padding.left - 12} y={height - padding.bottom + 4} className="chart-label" textAnchor="end">{minValue.toFixed(2)}</text>
        <text x="13" y={height / 2} className="chart-axis-title" textAnchor="middle" transform={`rotate(-90 13 ${height / 2})`}>Loss (cross-entropy)</text>
        <text x={padding.left} y={height - 10} className="chart-label">epoch 0</text>
        <text x={width - padding.right} y={height - 10} className="chart-label" textAnchor="end">epoch {maxEpoch}</text>
        <text x={width / 2} y={height - 1} className="chart-axis-title" textAnchor="middle">Epoch</text>
        {ACTIVATIONS.map((activation) => {
          if (!visibleActivations.has(activation)) return null;
          const points = histories[activation];
          const trainingPoints = points.map((point, index) => toPoint(point.trainingLoss, index)).join(" ");
          const validationPoints = points.map((point, index) => toPoint(point.validationLoss, index)).join(" ");
          return (
            <g key={activation}>
              {trainingPoints && <polyline points={trainingPoints} fill="none" stroke={ACTIVATION_COLORS[activation]} strokeWidth="2.2" />}
              {validationPoints && <polyline points={validationPoints} fill="none" stroke={ACTIVATION_COLORS[activation]} strokeWidth="1.6" strokeDasharray="5 4" opacity="0.58" />}
            </g>
          );
        })}
      </svg>
      <div className="chart-key-note"><span className="solid-key" /> training loss <span className="dashed-key" /> validation loss</div>
    </div>
  );
}

function PipelineSizeSelect({
  value,
  options,
  disabled,
  ariaLabel,
  onChange,
}: {
  value: number;
  options: readonly number[];
  disabled: boolean;
  ariaLabel: string;
  onChange: (value: number) => void;
}): ReactElement {
  return (
    <label className={`pipeline-size ${disabled ? "is-disabled" : ""}`}>
      <span className="pipeline-size-prefix">(n,&nbsp;</span>
      <select
        value={value}
        disabled={disabled}
        aria-label={ariaLabel}
        onChange={(event) => onChange(Number(event.target.value))}
      >
        {options.map((option) => (
          <option key={option} value={option}>{option}</option>
        ))}
      </select>
      <span className="pipeline-size-suffix">)</span>
    </label>
  );
}

function NetworkPipeline({
  inputFeatureCount,
  hiddenNeuronCount,
  totalParameters,
  currentEpoch,
  totalEpochs,
  disabled,
  onInputFeatureCountChange,
  onHiddenNeuronCountChange,
}: {
  inputFeatureCount: number;
  hiddenNeuronCount: number;
  totalParameters: number;
  currentEpoch: number;
  totalEpochs: number;
  disabled: boolean;
  onInputFeatureCountChange: (value: number) => void;
  onHiddenNeuronCountChange: (value: number) => void;
}): ReactElement {
  const steps: Array<{
    number: string;
    title: string;
    body: ReactElement;
    note: string;
  }> = [
    {
      number: "01",
      title: "Input",
      body: (
        <PipelineSizeSelect
          value={inputFeatureCount}
          options={INPUT_FEATURE_OPTIONS}
          disabled={disabled}
          ariaLabel="Input feature count"
          onChange={onInputFeatureCountChange}
        />
      ),
      note: "dataset input",
    },
    {
      number: "02",
      title: "Hidden layer",
      body: (
        <div className="pipeline-hidden-body">
          <PipelineSizeSelect
            value={hiddenNeuronCount}
            options={HIDDEN_NEURON_OPTIONS}
            disabled={disabled}
            ariaLabel="Hidden neuron count"
            onChange={onHiddenNeuronCountChange}
          />
          <code>W₁x + b₁ → f(·)</code>
        </div>
      ),
      note: "weighted sum → activation",
    },
    {
      number: "03",
      title: "Output",
      body: <code className="pipeline-shape">(n, {CLASS_COUNT})</code>,
      note: "softmax · 10 classes",
    },
    {
      number: "04",
      title: "Loss",
      body: <code className="pipeline-shape">scalar</code>,
      note: "cross-entropy",
    },
    {
      number: "05",
      title: "Update",
      body: <code className="pipeline-shape">weights</code>,
      note: "backprop + GD",
    },
  ];

  return (
    <section className="pipeline-panel" aria-labelledby="pipeline-title">
      <div className="section-kicker">ANN BASICS</div>
      <div className="pipeline-heading">
        <div>
          <h2 id="pipeline-title">One epoch, step by step</h2>
          <p>Inputs become activations, loss measures error, and updates change the weights.</p>
        </div>
        <div className="pipeline-metrics">
          <div className="epoch-counter"><span>Epoch</span><strong>{currentEpoch}</strong><small>/ {totalEpochs}</small></div>
          <div className="parameter-counter"><span>Total parameters</span><strong>{totalParameters.toLocaleString("en-US")}</strong></div>
        </div>
      </div>
      <div className="pipeline-steps">
        {steps.map((step, index) => (
          <div className="pipeline-step" key={step.number}>
            <span className="pipeline-number">{step.number}</span>
            <strong>{step.title}</strong>
            {step.body}
            <small>{step.note}</small>
            {index < steps.length - 1 && <span className="pipeline-arrow" aria-hidden="true">→</span>}
          </div>
        ))}
      </div>
    </section>
  );
}

function ActivationCard({
  activation,
  metric,
  history,
}: {
  activation: ActivationName;
  metric: Metric;
  history: LossPoint[];
}): ReactElement {
  const progress = metric.totalEpochs ? (metric.epoch / metric.totalEpochs) * 100 : 0;
  const maxTrainingLoss = history.length ? Math.max(...history.map((point) => point.trainingLoss)) : null;
  const cue = ACTIVATION_CUES[activation];
  return (
    <article className={`activation-card activation-${activation}`}>
      <div className="card-topline">
        <div className="activation-title">
          <span className="activation-swatch" style={{ backgroundColor: ACTIVATION_COLORS[activation] }} />
          <h3>{ACTIVATION_LABELS[activation]}</h3>
        </div>
        <StatusMark status={metric.status} />
      </div>
      <p className="activation-cue">
        <code>{cue.formula}</code>
        <span>{cue.watch}</span>
      </p>
      <div className="progress-track" aria-label={`${ACTIVATION_LABELS[activation]} progress`}>
        <span style={{ width: `${progress}%`, backgroundColor: ACTIVATION_COLORS[activation] }} />
      </div>
      <div className="card-epoch"><strong>{metric.epoch}</strong><span>/ {metric.totalEpochs} epochs</span><span className="card-max-loss">max loss {formatLoss(maxTrainingLoss)}</span></div>
      <Sparkline points={history} color={ACTIVATION_COLORS[activation]} label={`${ACTIVATION_LABELS[activation]} training loss`} />
      <div className="metric-grid">
        <div><span>train loss</span><strong>{formatLoss(metric.trainingLoss)}</strong></div>
        <div><span>validation loss</span><strong>{formatLoss(metric.validationLoss)}</strong></div>
        <div><span>validation accuracy</span><strong>{formatAccuracy(metric.validationAccuracy)}</strong></div>
      </div>
    </article>
  );
}

function App(): ReactElement {
  const [activeTab, setActiveTab] = useState<"train" | "predict">("train");
  const [runConfig, setRunConfig] = useState<RunConfig>(DEFAULT_CONFIG);
  const [runState, setRunState] = useState<RunState>("idle");
  const [connectionState, setConnectionState] = useState<ConnectionState>("disconnected");
  const [metrics, setMetrics] = useState<Record<ActivationName, Metric>>(() => createInitialMetrics(DEFAULT_CONFIG.epochs));
  const [histories, setHistories] = useState<Record<ActivationName, LossPoint[]>>(() => createInitialHistory());
  const [visibleActivations, setVisibleActivations] = useState<Set<ActivationName>>(() => new Set(ACTIVATIONS));
  const [summary, setSummary] = useState<{ durationMs: number; results: SummaryResult[] } | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const controlsDisabled = runState === "running" || runState === "connecting";
  const splitPercentageTotal = runConfig.train_percentage + runConfig.validation_percentage + runConfig.test_percentage;
  const splitIsValid = Math.abs(splitPercentageTotal - 100) < 0.001;
  const totalParameters = parameterCount(runConfig.input_feature_count, runConfig.hidden_neuron_count);

  const currentEpoch = Math.max(...ACTIVATIONS.map((activation) => metrics[activation].epoch));
  const statusText = useMemo(() => {
    if (runState === "running") return "Training realtime";
    if (runState === "completed") return "Run complete";
    if (runState === "cancelled") return "Run cancelled";
    if (runState === "error") return "Needs attention";
    return "Ready to inspect";
  }, [runState]);

  const handleMessage = (message: SocketMessage) => {
    if (message.type === "run_started") {
      setRunState("running");
      setErrorMessage(null);
      if (message.difficulty !== undefined) {
        setRunConfig((current) => ({ ...current, difficulty: message.difficulty! }));
      }
      setRunConfig((current) => ({
        ...current,
        ...(message.sample_count === undefined ? {} : { sample_count: message.sample_count }),
        ...(message.batch_size === undefined ? {} : { batch_size: message.batch_size }),
        ...(message.train_percentage === undefined ? {} : { train_percentage: message.train_percentage }),
        ...(message.validation_percentage === undefined ? {} : { validation_percentage: message.validation_percentage }),
        ...(message.test_percentage === undefined ? {} : { test_percentage: message.test_percentage }),
      }));
      return;
    }
    if (message.type === "epoch_update" && message.activation) {
      const activation = message.activation;
      setMetrics((current) => ({
        ...current,
        [activation]: {
          ...current[activation],
          epoch: message.epoch ?? current[activation].epoch,
          totalEpochs: message.total_epochs ?? current[activation].totalEpochs,
          trainingLoss: message.training_loss ?? current[activation].trainingLoss,
          validationLoss: message.validation_loss ?? current[activation].validationLoss,
          validationAccuracy: message.validation_accuracy ?? current[activation].validationAccuracy,
          status: message.status === "early_stopped" ? "early_stopped" : "running",
        },
      }));
      if (message.training_loss !== undefined && message.validation_loss !== undefined) {
        setHistories((current) => ({
          ...current,
          [activation]: [...current[activation], { epoch: message.epoch ?? 0, trainingLoss: message.training_loss!, validationLoss: message.validation_loss! }],
        }));
      }
      return;
    }
    if (message.type === "run_completed" || message.type === "run_cancelled") {
      const finalStatus = message.type === "run_completed" ? "completed" : "cancelled";
      setRunState(finalStatus);
      setSummary({ durationMs: message.duration_ms ?? 0, results: message.results ?? [] });
      const resultByActivation = new Map((message.results ?? []).map((result) => [result.activation, result]));
      setMetrics((current) => Object.fromEntries(
        ACTIVATIONS.map((activation) => [
          activation,
          {
            ...current[activation],
            status: finalStatus === "completed" && resultByActivation.get(activation)?.stopped_early
              ? "early_stopped"
              : finalStatus,
          },
        ]),
      ) as Record<ActivationName, Metric>);
      return;
    }
    if (message.type === "error") {
      setRunState("error");
      setErrorMessage(message.message ?? "Backend returned an unknown error.");
    }
  };

  const sendStart = () => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    socket.send(JSON.stringify({ type: "start", config: runConfig }));
  };

  const handleStart = () => {
    if (runState === "running" || runState === "connecting") return;
    setMetrics(createInitialMetrics(runConfig.epochs));
    setHistories(createInitialHistory());
    setSummary(null);
    setErrorMessage(null);
    setRunState("connecting");
    const existingSocket = socketRef.current;
    if (existingSocket?.readyState === WebSocket.OPEN) {
      sendStart();
      return;
    }
    const socket = new WebSocket("ws://localhost:6788/ws/train");
    socketRef.current = socket;
    setConnectionState("connecting");
    socket.onopen = () => {
      setConnectionState("connected");
      sendStart();
    };
    socket.onmessage = (event) => handleMessage(JSON.parse(event.data) as SocketMessage);
    socket.onerror = () => {
      setRunState("error");
      setErrorMessage("Could not connect to the backend. Start FastAPI on port 6788.");
    };
    socket.onclose = () => setConnectionState("disconnected");
  };

  const handleCancel = () => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: "cancel" }));
    }
  };

  const handleReset = () => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: "reset" }));
      socketRef.current.close();
    }
    setRunState("idle");
    setConnectionState("disconnected");
    setMetrics(createInitialMetrics(runConfig.epochs));
    setHistories(createInitialHistory());
    setSummary(null);
    setErrorMessage(null);
  };

  const toggleActivation = (activation: ActivationName) => {
    setVisibleActivations((current) => {
      const next = new Set(current);
      if (next.has(activation)) next.delete(activation);
      else next.add(activation);
      return next;
    });
  };

  const bestValidation = summary?.results.reduce<SummaryResult | null>((best, result) => {
    if (!best || (result.validation_accuracy ?? -1) > (best.validation_accuracy ?? -1)) return result;
    return best;
  }, null);

  return (
    <main className="app-shell">
      <header className="topbar app-topbar">
        <div className="brand-lockup">
          <div className="brand-mark">ANN</div>
          <div><strong>Training Lab</strong><span>ANN from scratch · review · demo</span></div>
        </div>
        <nav className="tabs" aria-label="Demo sections">
          <button
            className={`tab ${activeTab === "train" ? "tab-active" : ""}`}
            type="button"
            onClick={() => setActiveTab("train")}
          >
            Review
          </button>
          <button
            className={`tab ${activeTab === "predict" ? "tab-active" : ""}`}
            type="button"
            onClick={() => setActiveTab("predict")}
          >
            Demo
          </button>
        </nav>
        <div className={`connection-pill connection-${connectionState}`}>
          <span className="status-mark" /> {activeTab === "train" ? connectionState : "http demo"}
        </div>
      </header>

      {activeTab === "predict" ? <PredictPanel /> : (
        <>
          <section className="lesson-header" aria-label="Training lab header">
            <div className="lesson-body">
              <div className="lesson-intro">
                <div className="lesson-copy">
                  <div className="section-kicker">ANN REVIEW</div>
                  <h1>ANN Review</h1>
                  <span className="review-status"><span className="status-mark" aria-hidden="true" />{statusText}</span>
                  <p className="intro-copy">
                    Review the key parts of an ANN: inputs, hidden layers, activations, loss, and weight updates.
                  </p>
                </div>
              </div>

              <section className="control-panel" aria-label="Training controls">
                <label className="early-stopping-field"><input type="checkbox" checked={runConfig.early_stopping} disabled={controlsDisabled} onChange={(event) => setRunConfig({ ...runConfig, early_stopping: event.target.checked })} /><span>Early stopping</span></label>
                <label className="difficulty-field">Noise <output>{Math.round(runConfig.difficulty * 100)}%</output><input className="difficulty-range" type="range" min="0" max="100" step="1" value={Math.round(runConfig.difficulty * 100)} disabled={controlsDisabled} onChange={(event) => setRunConfig({ ...runConfig, difficulty: Number(event.target.value) / 100 })} /><span className="difficulty-endpoints"><span>Easy</span><span>Hard</span></span><small>{difficultyDescription(runConfig.difficulty)}</small></label>
                <div className="dataset-settings" aria-label="Dataset settings">
                  <label>Total<input type="number" min="30" max="10000" step="10" value={runConfig.sample_count} disabled={controlsDisabled} onChange={(event) => setRunConfig({ ...runConfig, sample_count: Number(event.target.value) })} /></label>
                  <label>Train %<input type="number" min="1" max="98" value={runConfig.train_percentage} disabled={controlsDisabled} onChange={(event) => setRunConfig({ ...runConfig, train_percentage: Number(event.target.value) })} /></label>
                  <label>Val %<input type="number" min="1" max="98" value={runConfig.validation_percentage} disabled={controlsDisabled} onChange={(event) => setRunConfig({ ...runConfig, validation_percentage: Number(event.target.value) })} /></label>
                  <label>Test %<input type="number" min="1" max="98" value={runConfig.test_percentage} disabled={controlsDisabled} onChange={(event) => setRunConfig({ ...runConfig, test_percentage: Number(event.target.value) })} /></label>
                  <small className="dataset-count-note">default: {CLASS_COUNT} classes × {SAMPLES_PER_CLASS} = {DEFAULT_SAMPLE_COUNT}</small>
                  <small className={splitIsValid ? "split-valid" : "split-invalid"}>sum {splitPercentageTotal}%</small>
                </div>
                <label className="batch-field">Batch size<input type="number" min="1" max={runConfig.sample_count} value={runConfig.batch_size} disabled={controlsDisabled} onChange={(event) => setRunConfig({ ...runConfig, batch_size: Number(event.target.value) })} /></label>
                <label className="epochs-field">Epochs<input type="number" min="1" max="5000" value={runConfig.epochs} disabled={controlsDisabled} onChange={(event) => setRunConfig({ ...runConfig, epochs: Number(event.target.value) })} /></label>
                <label className="learning-field">Learning rate<input type="number" min="0.001" max="1" step="0.01" value={runConfig.learning_rate} disabled={controlsDisabled} onChange={(event) => setRunConfig({ ...runConfig, learning_rate: Number(event.target.value) })} /></label>
                <div className="control-actions"><button className="button button-primary" type="button" onClick={handleStart} disabled={controlsDisabled || !splitIsValid}>Start training</button><button className="button button-quiet" type="button" onClick={handleCancel} disabled={runState !== "running"}>Cancel</button><button className="button button-quiet" type="button" onClick={handleReset}>Reset</button></div>
                {errorMessage && <p className="error-note" role="alert">{errorMessage}</p>}
              </section>
            </div>
          </section>

          <NetworkPipeline
            inputFeatureCount={runConfig.input_feature_count}
            hiddenNeuronCount={runConfig.hidden_neuron_count}
            totalParameters={totalParameters}
            currentEpoch={currentEpoch}
            totalEpochs={runConfig.epochs}
            disabled={controlsDisabled}
            onInputFeatureCountChange={(value) => setRunConfig({ ...runConfig, input_feature_count: value })}
            onHiddenNeuronCountChange={(value) => setRunConfig({ ...runConfig, hidden_neuron_count: value })}
          />

          <div className="training-stage">
            <section className="section-block" aria-labelledby="lanes-title">
              <div className="section-heading"><div><div className="section-kicker">ACTIVATION COMPARISON</div><h2 id="lanes-title">Five functions, one starting point</h2></div><p>Compare the shape, loss, and validation accuracy.</p></div>
              <div className="activation-grid">
                {ACTIVATIONS.map((activation) => <ActivationCard key={activation} activation={activation} metric={metrics[activation]} history={histories[activation]} />)}
              </div>
            </section>

            <section className="lower-grid">
              <div className="chart-panel panel-surface"><div className="section-heading compact"><div><div className="section-kicker">LOSS OVER TIME</div><h2>Which curve drops first?</h2></div><p>Click a name to hide or show it. Try ReLU vs Sigmoid, then Identity vs the rest.</p></div><div className="legend-row">{ACTIVATIONS.map((activation) => <button key={activation} type="button" className={`legend-item ${visibleActivations.has(activation) ? "legend-visible" : "legend-hidden"}`} onClick={() => toggleActivation(activation)}><span style={{ backgroundColor: ACTIVATION_COLORS[activation] }} />{ACTIVATION_LABELS[activation]}</button>)}</div><LossChart histories={histories} visibleActivations={visibleActivations} /></div>
              {summary && <section className="summary-panel panel-surface" aria-labelledby="summary-title"><div className="section-heading"><div><div className="section-kicker">RUN SUMMARY</div><h2 id="summary-title">Which model wins on validation?</h2></div><p>{summary.durationMs} ms · {summary.results.length} activation functions · highlight = highest validation accuracy</p></div><div className="summary-grid">{summary.results.map((result) => <div className={`summary-row ${bestValidation?.activation === result.activation ? "summary-highlight" : ""}`} key={result.activation}><span className="activation-swatch" style={{ backgroundColor: ACTIVATION_COLORS[result.activation] }} /><strong>{ACTIVATION_LABELS[result.activation]}</strong><span>v loss <b>{formatLoss(result.validation_loss)}</b></span><span>v acc <b>{formatAccuracy(result.validation_accuracy)}</b></span><span>test acc <b>{formatAccuracy(result.test_accuracy)}</b></span></div>)}</div><p className="summary-note">Accuracy matters more than loss for classification. Test accuracy is measured after training. ReLU and Leaky ReLU often learn quickly; Sigmoid and tanh can saturate; Identity has no nonlinearity.</p></section>}
            </section>
          </div>
        </>
      )}

      <footer className="footer-note"><span>Source: self-built NumPy ANN</span><span>Validation guides the run · test is reported at the end</span></footer>
    </main>
  );
}

export default App;
