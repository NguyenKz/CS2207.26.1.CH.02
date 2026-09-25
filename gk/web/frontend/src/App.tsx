import { useMemo, useRef, useState, type ReactElement } from "react";

const ACTIVATIONS = [
  "tanh",
  "sigmoid",
  "relu",
  "leaky_relu",
  "softplus",
  "identity",
] as const;

type ActivationName = (typeof ACTIVATIONS)[number];
type RunState = "idle" | "connecting" | "running" | "completed" | "cancelled" | "error";
type ConnectionState = "disconnected" | "connecting" | "connected";

type RunConfig = {
  epochs: number;
  delay_seconds: number;
  learning_rate: number;
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
  results?: SummaryResult[];
  dataset?: {
    training: number[];
    validation: number[];
    testing: number[];
  };
};

const DEFAULT_CONFIG: RunConfig = {
  epochs: 2000,
  delay_seconds: 0.001,
  learning_rate: 0.05,
  hidden_neuron_count: 8,
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
  softplus: "Softplus",
  identity: "Identity",
};

const ACTIVATION_COLORS: Record<ActivationName, string> = {
  tanh: "#d05a45",
  sigmoid: "#2c7a7b",
  relu: "#b7862c",
  leaky_relu: "#6b5ca5",
  softplus: "#377d5f",
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
  return value === null ? "—" : value.toFixed(4);
}

function formatAccuracy(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function StatusMark({ status }: { status: Metric["status"] }): ReactElement {
  const labels = {
    idle: "Idle",
    running: "Đang chạy",
    completed: "Hoàn tất",
    cancelled: "Đã dừng",
    early_stopped: "Dừng sớm",
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

function NetworkPipeline({ currentEpoch, totalEpochs }: { currentEpoch: number; totalEpochs: number }): ReactElement {
  const steps = [
    ["01", "Input", "(n, 4)", "bốn feature"],
    ["02", "Weighted sum", "(n, 8)", "X @ W₁ + b₁"],
    ["03", "Activation", "(n, 8)", "tanh / ReLU / ..."],
    ["04", "Output", "(n, 3)", "softmax probabilities"],
    ["05", "Loss", "scalar", "cross-entropy"],
    ["06", "Update", "weights", "backprop + gradient descent"],
  ];

  return (
    <section className="pipeline-panel" aria-labelledby="pipeline-title">
      <div className="section-kicker">THE MECHANISM</div>
      <div className="pipeline-heading">
        <div>
          <h2 id="pipeline-title">Một epoch, nhìn từ bên trong</h2>
          <p>Không có bước nào bị ẩn sau một API model có sẵn.</p>
        </div>
        <div className="epoch-counter"><span>Epoch</span><strong>{currentEpoch}</strong><small>/ {totalEpochs}</small></div>
      </div>
      <div className="pipeline-steps">
        {steps.map(([number, title, shape, detail], index) => (
          <div className="pipeline-step" key={number}>
            <span className="pipeline-number">{number}</span>
            <strong>{title}</strong>
            <code>{shape}</code>
            <small>{detail}</small>
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
  return (
    <article className={`activation-card activation-${activation}`}>
      <div className="card-topline">
        <div className="activation-title">
          <span className="activation-swatch" style={{ backgroundColor: ACTIVATION_COLORS[activation] }} />
          <h3>{ACTIVATION_LABELS[activation]}</h3>
        </div>
        <StatusMark status={metric.status} />
      </div>
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
  const [runConfig, setRunConfig] = useState<RunConfig>(DEFAULT_CONFIG);
  const [runState, setRunState] = useState<RunState>("idle");
  const [connectionState, setConnectionState] = useState<ConnectionState>("disconnected");
  const [metrics, setMetrics] = useState<Record<ActivationName, Metric>>(() => createInitialMetrics(DEFAULT_CONFIG.epochs));
  const [histories, setHistories] = useState<Record<ActivationName, LossPoint[]>>(() => createInitialHistory());
  const [visibleActivations, setVisibleActivations] = useState<Set<ActivationName>>(() => new Set(ACTIVATIONS));
  const [summary, setSummary] = useState<{ durationMs: number; results: SummaryResult[] } | null>(null);
  const [datasetShapes, setDatasetShapes] = useState<string>("90 train · 30 validation · 30 test");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);

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
      if (message.dataset) {
        setDatasetShapes(`${message.dataset.training[0]} train · ${message.dataset.validation[0]} validation · ${message.dataset.testing[0]} test`);
      }
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
      setErrorMessage("Không thể kết nối backend. Hãy chạy FastAPI ở port 6788.");
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
    if (!best || (result.validation_loss ?? Infinity) < (best.validation_loss ?? Infinity)) return result;
    return best;
  }, null);

  return (
    <main className="app-shell">
      <section className="lesson-header" aria-label="Training lab header">
        <header className="topbar">
          <div className="brand-lockup">
            <div className="brand-mark">ANN</div>
            <div><strong>Training Lab</strong><span>NumPy from scratch · Iris</span></div>
          </div>
          <nav className="tabs" aria-label="Demo sections">
            <button className="tab tab-active" type="button">Train</button>
            <button className="tab tab-disabled" type="button" disabled>Predict <span>coming soon</span></button>
          </nav>
          <div className={`connection-pill connection-${connectionState}`}><span className="status-mark" /> {connectionState}</div>
        </header>

        <div className="lesson-body">
          <div className="lesson-intro">
            <div className="lesson-copy">
              <div className="section-kicker">LESSON 01 / TRAINING</div>
              <h1>Watch the network learn.</h1>
              <p className="intro-copy">Six activation functions. One dataset. Every forward pass, loss and weight update visible as it happens.</p>
            </div>
            <div className="intro-meta"><span className="meta-label">MODEL</span><strong>4 → 8 → 3</strong><span>{datasetShapes}</span></div>
          </div>

          <section className="control-panel" aria-label="Training controls">
            <div className="control-heading"><span className="section-kicker">CONTROL ROOM</span><strong>{statusText}</strong><label className="toggle-field"><input type="checkbox" checked={runConfig.early_stopping} disabled={runState === "running" || runState === "connecting"} onChange={(event) => setRunConfig({ ...runConfig, early_stopping: event.target.checked })} /><span>Early stopping</span><small>Dừng nếu validation loss không giảm ≥ {runConfig.early_stopping_min_delta} trong {runConfig.early_stopping_patience} epoch liên tiếp.</small></label></div>
            <label>Epochs<input type="number" min="1" max="5000" value={runConfig.epochs} disabled={runState === "running" || runState === "connecting"} onChange={(event) => setRunConfig({ ...runConfig, epochs: Number(event.target.value) })} /></label>
            <label>Delay per epoch<select value={runConfig.delay_seconds} disabled={runState === "running" || runState === "connecting"} onChange={(event) => setRunConfig({ ...runConfig, delay_seconds: Number(event.target.value) })}><option value="0.001">0.001s</option><option value="0.01">0.01s</option><option value="0.05">0.05s</option><option value="0.1">0.1s</option></select></label>
            <label>Learning rate<input type="number" min="0.001" max="1" step="0.01" value={runConfig.learning_rate} disabled={runState === "running" || runState === "connecting"} onChange={(event) => setRunConfig({ ...runConfig, learning_rate: Number(event.target.value) })} /></label>
            <div className="control-actions"><button className="button button-primary" type="button" onClick={handleStart} disabled={runState === "running" || runState === "connecting"}>Start training</button><button className="button button-quiet" type="button" onClick={handleCancel} disabled={runState !== "running"}>Cancel</button><button className="button button-quiet" type="button" onClick={handleReset}>Reset</button></div>
            {errorMessage && <p className="error-note" role="alert">{errorMessage}</p>}
          </section>
        </div>
      </section>

      <NetworkPipeline currentEpoch={currentEpoch} totalEpochs={runConfig.epochs} />

      <div className="training-stage">
        <section className="section-block" aria-labelledby="lanes-title">
          <div className="section-heading"><div><div className="section-kicker">LIVE COMPARISON</div><h2 id="lanes-title">Six learners, same starting line</h2></div><p>Solid line = train loss · dashed line = validation loss</p></div>
          <div className="activation-grid">
            {ACTIVATIONS.map((activation) => <ActivationCard key={activation} activation={activation} metric={metrics[activation]} history={histories[activation]} />)}
          </div>
        </section>

        <section className="lower-grid">
          <div className="chart-panel panel-surface"><div className="section-heading compact"><div><div className="section-kicker">LOSS OVER TIME</div><h2>Which curves are moving?</h2></div><p>Click a label to isolate a learner.</p></div><div className="legend-row">{ACTIVATIONS.map((activation) => <button key={activation} type="button" className={`legend-item ${visibleActivations.has(activation) ? "legend-visible" : "legend-hidden"}`} onClick={() => toggleActivation(activation)}><span style={{ backgroundColor: ACTIVATION_COLORS[activation] }} />{ACTIVATION_LABELS[activation]}</button>)}</div><LossChart histories={histories} visibleActivations={visibleActivations} /></div>
          {summary && <section className="summary-panel panel-surface" aria-labelledby="summary-title"><div className="section-heading"><div><div className="section-kicker">RUN SUMMARY</div><h2 id="summary-title">The run is measurable, not magical.</h2></div><p>{summary.durationMs} ms · {summary.results.length} activation functions</p></div><div className="summary-grid">{summary.results.map((result) => <div className={`summary-row ${bestValidation?.activation === result.activation ? "summary-highlight" : ""}`} key={result.activation}><span className="activation-swatch" style={{ backgroundColor: ACTIVATION_COLORS[result.activation] }} /><strong>{ACTIVATION_LABELS[result.activation]}</strong><span>v loss <b>{formatLoss(result.validation_loss)}</b></span><span>v acc <b>{formatAccuracy(result.validation_accuracy)}</b></span><span>t acc <b>{formatAccuracy(result.test_accuracy)}</b></span></div>)}</div><p className="summary-note">Highlighted means lowest validation loss in this run on Iris. It is not a universal ranking of activation functions.</p></section>}
        </section>
      </div>

      <footer className="footer-note"><span>Source: self-built NumPy ANN</span><span>Validation guides the run · test is reported at the end</span></footer>
    </main>
  );
}

export default App;
