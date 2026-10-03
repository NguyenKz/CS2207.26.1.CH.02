import { useCallback, useEffect, useMemo, useRef, useState, type ReactElement, type RefObject } from "react";

import type { RunConfig } from "./App";

const API_BASE = "http://localhost:6788";
const PAGE_SIZE = 60;
const CLASS_LABELS = Array.from({ length: 10 }, (_, index) => index);
const CLASS_COLORS = ["#d05a45", "#2c7a7b", "#b7862c", "#6b5ca5", "#377d5f", "#435466", "#b05a82", "#668e9b", "#a16d3d", "#596b46"];

type DatasetTab = "predict" | "train";
type SplitName = "all" | "training" | "validation" | "testing";
type DatasetTrainConfig = Pick<
  RunConfig,
  | "difficulty"
  | "sample_count"
  | "train_percentage"
  | "validation_percentage"
  | "test_percentage"
  | "input_feature_count"
  | "random_seed"
>;

type PredictSample = {
  id: string;
  source: "train" | "test";
  index: number;
  split: "training" | "validation" | "testing";
  label: number;
  image_url: string;
};

type PredictMeta = {
  dataset: {
    name: string;
    description: string;
    sample_count: number;
    training_sample_count?: number;
    test_sample_count?: number;
    input_shape: [number, number];
    feature_count: number;
    class_count: number;
    pixel_min: number;
    pixel_max: number;
    original_input_shape?: [number, number];
    source_url?: string;
  };
  class_counts: number[];
  split_counts: Record<Exclude<SplitName, "all">, number>;
  image_ready: boolean;
  representatives: PredictSample[];
};

type PredictPage = {
  items: PredictSample[];
  offset: number;
  limit: number;
  total: number;
  has_more: boolean;
};

type ScatterPoint = {
  index: number;
  label: number;
  split: Exclude<SplitName, "all">;
  x: number;
  y: number;
};

type TrainPreview = {
  sample_count: number;
  class_count: number;
  class_names: string[];
  feature_names: string[];
  class_counts: number[];
  split_counts: Record<Exclude<SplitName, "all">, number>;
  feature_x: number;
  feature_y: number;
  points: ScatterPoint[];
};

type TrainSample = {
  index: number;
  label: number;
  split: Exclude<SplitName, "all">;
  features: number[];
};

type TrainPage = {
  items: TrainSample[];
  offset: number;
  limit: number;
  total: number;
  has_more: boolean;
};

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

function formatCount(value: number): string {
  return value.toLocaleString("en-US");
}

function DatasetImage({ sample, detail = false }: { sample: PredictSample; detail?: boolean }): ReactElement {
  return (
    <div className={`dataset-image-frame ${detail ? "is-detail" : ""}`}>
      <img
        src={`${API_BASE}${sample.image_url}`}
        alt={`Digit ${sample.label}, ${sample.split} sample ${sample.index}`}
        loading="lazy"
        width={28}
        height={28}
      />
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }): ReactElement {
  return (
    <div className="dataset-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StateMessage({
  title,
  detail,
  error = false,
  onRetry,
}: {
  title: string;
  detail: string;
  error?: boolean;
  onRetry?: () => void;
}): ReactElement {
  return (
    <div className={`dataset-state ${error ? "dataset-state-error" : ""}`} role={error ? "alert" : undefined}>
      <strong>{title}</strong>
      <p>{detail}</p>
      {onRetry && <button className="button button-quiet" type="button" onClick={onRetry}>Retry</button>}
    </div>
  );
}

function SampleTile({
  sample,
  selected,
  onSelect,
}: {
  sample: PredictSample;
  selected: boolean;
  onSelect: () => void;
}): ReactElement {
  return (
    <button
      className={`dataset-digit-tile ${selected ? "is-selected" : ""}`}
      type="button"
      aria-pressed={selected}
      aria-label={`Class ${sample.label}, sample ${sample.index}`}
      onClick={onSelect}
    >
      <DatasetImage sample={sample} />
      <span><strong>{sample.label}</strong><small>{sample.split} · #{sample.index}</small></span>
    </button>
  );
}

function LoadMore({
  hasMore,
  loading,
  onLoad,
  endLabel,
  scrollRootRef,
}: {
  hasMore: boolean;
  loading: boolean;
  onLoad: () => void;
  endLabel: string;
  scrollRootRef?: RefObject<HTMLElement | null>;
}): ReactElement {
  const sentinelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = sentinelRef.current;
    if (!node || !hasMore || loading || typeof IntersectionObserver === "undefined") return undefined;
    const observer = new IntersectionObserver((entries) => {
      if (entries[0]?.isIntersecting) onLoad();
    }, { root: scrollRootRef?.current ?? null, rootMargin: "240px" });
    observer.observe(node);
    return () => observer.disconnect();
  }, [hasMore, loading, onLoad, scrollRootRef]);

  if (!hasMore) return <p className="dataset-end-note">{endLabel}</p>;
  return (
    <div className="dataset-load-more">
      <button className="button button-quiet" type="button" onClick={onLoad} disabled={loading}>
        {loading ? "Loading more samples..." : "Load more samples"}
      </button>
      <div ref={sentinelRef} aria-hidden="true" />
    </div>
  );
}

function ClassSelector({
  counts,
  selectedClass,
  onSelect,
}: {
  counts: number[];
  selectedClass: number | null;
  onSelect: (value: number | null) => void;
}): ReactElement {
  return (
    <div className="dataset-class-selector" aria-label="Filter dataset by class">
      <button className={`dataset-class-button ${selectedClass === null ? "is-selected" : ""}`} type="button" aria-pressed={selectedClass === null} onClick={() => onSelect(null)}>
        <strong>All</strong><small>{formatCount(counts.reduce((sum, value) => sum + value, 0))}</small>
      </button>
      {CLASS_LABELS.map((label) => (
        <button
          className={`dataset-class-button ${selectedClass === label ? "is-selected" : ""}`}
          type="button"
          aria-pressed={selectedClass === label}
          key={label}
          onClick={() => onSelect(label)}
        >
          <strong>{label}</strong><small>{formatCount(counts[label] ?? 0)}</small>
        </button>
      ))}
    </div>
  );
}

export function PredictDatasetView(): ReactElement {
  const [meta, setMeta] = useState<PredictMeta | null>(null);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [selectedClass, setSelectedClass] = useState<number | null>(null);
  const [samples, setSamples] = useState<PredictSample[]>([]);
  const [selectedSample, setSelectedSample] = useState<PredictSample | null>(null);
  const [sampleError, setSampleError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [total, setTotal] = useState(0);
  const offsetRef = useRef(0);
  const metaRequestRef = useRef<AbortController | null>(null);
  const requestRef = useRef<AbortController | null>(null);
  const loadingRef = useRef(false);
  const browserPanelRef = useRef<HTMLElement | null>(null);

  const loadMeta = useCallback(async (): Promise<void> => {
    metaRequestRef.current?.abort();
    const controller = new AbortController();
    metaRequestRef.current = controller;
    setMetaError(null);
    try {
      const data = await fetchJson<PredictMeta>(`${API_BASE}/dataset/predict/meta`, { signal: controller.signal });
      if (controller.signal.aborted || metaRequestRef.current !== controller) return;
      setMeta(data);
      setSelectedSample(data.representatives[0] ?? null);
    } catch (error) {
      if (!isAbortError(error)) setMetaError(error instanceof Error ? error.message : String(error));
    } finally {
      if (metaRequestRef.current === controller) metaRequestRef.current = null;
    }
  }, []);

  useEffect(() => {
    void loadMeta();
    return () => {
      metaRequestRef.current?.abort();
      requestRef.current?.abort();
    };
  }, [loadMeta]);

  const loadSamples = useCallback(async (reset: boolean): Promise<void> => {
    if (loadingRef.current && !reset) return;
    if (reset) requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    loadingRef.current = true;
    setLoadingMore(true);
    setSampleError(null);
    const offset = reset ? 0 : offsetRef.current;
    if (reset) {
      offsetRef.current = 0;
      setSamples([]);
      setSelectedSample(null);
      setHasMore(true);
    }
    try {
      const params = new URLSearchParams({ offset: String(offset), limit: String(PAGE_SIZE) });
      if (selectedClass !== null) params.set("label", String(selectedClass));
      const data = await fetchJson<PredictPage>(`${API_BASE}/dataset/predict/samples?${params}`, { signal: controller.signal });
      if (controller.signal.aborted || requestRef.current !== controller) return;
      setSamples((current) => reset ? data.items : [...current, ...data.items]);
      setSelectedSample((current) => current ?? data.items[0] ?? null);
      offsetRef.current = data.offset + data.items.length;
      setHasMore(data.has_more);
      setTotal(data.total);
    } catch (error) {
      if (!isAbortError(error)) setSampleError(error instanceof Error ? error.message : String(error));
    } finally {
      if (requestRef.current === controller) {
        loadingRef.current = false;
        setLoading(false);
        setLoadingMore(false);
      }
    }
  }, [selectedClass]);

  useEffect(() => {
    void loadSamples(true);
  }, [loadSamples]);

  if (metaError && !meta) {
    return <StateMessage title="Could not load the Predict dataset" detail={metaError} error onRetry={() => { void loadMeta(); void loadSamples(true); }} />;
  }
  if (!meta) return <StateMessage title="Reading the Predict dataset..." detail="Preparing metadata and representative samples." />;

  const selectedLabel = selectedClass === null ? "all classes" : `class ${selectedClass}`;
  return (
    <div className="dataset-view">
      <section className="panel-surface dataset-summary-bar">
        <div className="dataset-summary-title">
          <div className="section-kicker">PREDICT DATASET</div>
          <h2>{meta.dataset.name}</h2>
        </div>
        <div className="dataset-metric-grid dataset-predict-metric-grid">
          <Metric label="All samples" value={formatCount(meta.dataset.sample_count)} />
          <Metric label="Classes" value={String(meta.dataset.class_count)} />
          <Metric label="Image size" value={`${meta.dataset.input_shape[0]}×${meta.dataset.input_shape[1]}`} />
        </div>
      </section>

      <div className="dataset-predict-layout">
        <aside className="panel-surface dataset-class-filter-panel">
          <ClassSelector counts={meta.class_counts} selectedClass={selectedClass} onSelect={setSelectedClass} />
        </aside>

        <div className="dataset-predict-main">
          <section ref={browserPanelRef} className="panel-surface dataset-browser-panel">
            <div className="section-heading compact">
              <div><div className="section-kicker">SAMPLE BROWSER</div><h2>{formatCount(total)} samples in {selectedLabel}</h2></div>
            </div>
            {sampleError && <div className="dataset-inline-error" role="alert">{sampleError} <button type="button" onClick={() => void loadSamples(samples.length === 0)}>Retry</button></div>}
            {loading && !samples.length ? <StateMessage title="Loading samples..." detail="Only one small page loads at a time to keep the page responsive." /> : samples.length ? (
              <>
                <div className="dataset-sample-grid">
                  {samples.map((sample) => <SampleTile key={sample.id} sample={sample} selected={selectedSample?.id === sample.id} onSelect={() => setSelectedSample(sample)} />)}
                </div>
                <LoadMore
                  hasMore={hasMore}
                  loading={loadingMore}
                  onLoad={() => void loadSamples(false)}
                  endLabel="All filtered samples are shown."
                  scrollRootRef={browserPanelRef}
                />
              </>
            ) : <StateMessage title="No samples" detail="The current filter returned no samples." />}
          </section>
        </div>
      </div>
    </div>
  );
}

function ScatterPlot({
  points,
  visibleClasses,
  featureX,
  featureY,
}: {
  points: ScatterPoint[];
  visibleClasses: Set<number>;
  featureX: string;
  featureY: string;
}): ReactElement {
  const width = 760;
  const height = 360;
  const padding = { left: 54, right: 18, top: 18, bottom: 42 };
  const visiblePoints = points.filter((point) => visibleClasses.has(point.label));
  const xValues = visiblePoints.map((point) => point.x);
  const yValues = visiblePoints.map((point) => point.y);
  const xMin = xValues.length ? Math.min(...xValues) : 0;
  const xMax = xValues.length ? Math.max(...xValues) : 1;
  const yMin = yValues.length ? Math.min(...yValues) : 0;
  const yMax = yValues.length ? Math.max(...yValues) : 1;
  const xRange = Math.max(xMax - xMin, 0.001);
  const yRange = Math.max(yMax - yMin, 0.001);
  const x = (value: number) => padding.left + ((value - xMin) / xRange) * (width - padding.left - padding.right);
  const y = (value: number) => height - padding.bottom - ((value - yMin) / yRange) * (height - padding.top - padding.bottom);
  return (
    <div className="dataset-scatter-wrap">
      <svg className="dataset-scatter" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${featureY} by ${featureX} for the Train dataset`}>
        <line className="dataset-scatter-axis" x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} />
        <line className="dataset-scatter-axis" x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} />
        {visiblePoints.map((point) => <circle key={point.index} cx={x(point.x)} cy={y(point.y)} r="3.2" fill={CLASS_COLORS[point.label]} fillOpacity=".68"><title>{`class ${point.label} · ${point.split} · ${point.index}`}</title></circle>)}
        <text className="dataset-scatter-label" x={width / 2} y={height - 8} textAnchor="middle">{featureX}</text>
        <text className="dataset-scatter-label" x="14" y={height / 2} textAnchor="middle" transform={`rotate(-90 14 ${height / 2})`}>{featureY}</text>
      </svg>
      {!visiblePoints.length && <p className="dataset-empty-copy">Enable at least one class to view the scatter plot.</p>}
    </div>
  );
}

function TrainDatasetView({ trainConfig }: { trainConfig: DatasetTrainConfig }): ReactElement {
  const [preview, setPreview] = useState<TrainPreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(true);
  const [featureX, setFeatureX] = useState(0);
  const [featureY, setFeatureY] = useState(1);
  const [visibleClasses, setVisibleClasses] = useState<Set<number>>(() => new Set(CLASS_LABELS));
  const [classFilter, setClassFilter] = useState<number | null>(null);
  const [splitFilter, setSplitFilter] = useState<SplitName>("all");
  const [samples, setSamples] = useState<TrainSample[]>([]);
  const [selectedSample, setSelectedSample] = useState<TrainSample | null>(null);
  const [sampleError, setSampleError] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [total, setTotal] = useState(0);
  const offsetRef = useRef(0);
  const previewRequestRef = useRef<AbortController | null>(null);
  const sampleRequestRef = useRef<AbortController | null>(null);
  const loadingRef = useRef(false);
  const configPayload = useMemo(() => ({ ...trainConfig }), [trainConfig]);

  const loadPreview = useCallback(async (): Promise<void> => {
    previewRequestRef.current?.abort();
    const controller = new AbortController();
    previewRequestRef.current = controller;
    setPreviewLoading(true);
    setPreviewError(null);
    try {
      const data = await fetchJson<TrainPreview>(`${API_BASE}/dataset/train/preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...configPayload, feature_x: featureX, feature_y: featureY }),
        signal: controller.signal,
      });
      if (controller.signal.aborted || previewRequestRef.current !== controller) return;
      setPreview(data);
      setFeatureX(data.feature_x);
      setFeatureY(data.feature_y);
    } catch (error) {
      if (!isAbortError(error)) setPreviewError(error instanceof Error ? error.message : String(error));
    } finally {
      if (previewRequestRef.current === controller) setPreviewLoading(false);
    }
  }, [configPayload, featureX, featureY]);

  useEffect(() => {
    void loadPreview();
    return () => previewRequestRef.current?.abort();
  }, [loadPreview]);

  const loadSamples = useCallback(async (reset: boolean): Promise<void> => {
    if (loadingRef.current && !reset) return;
    if (reset) sampleRequestRef.current?.abort();
    const controller = new AbortController();
    sampleRequestRef.current = controller;
    loadingRef.current = true;
    setLoadingMore(true);
    setSampleError(null);
    const offset = reset ? 0 : offsetRef.current;
    if (reset) {
      offsetRef.current = 0;
      setSamples([]);
      setSelectedSample(null);
      setHasMore(true);
    }
    try {
      const data = await fetchJson<TrainPage>(`${API_BASE}/dataset/train/samples`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...configPayload, class_label: classFilter, split: splitFilter, offset, limit: PAGE_SIZE }),
        signal: controller.signal,
      });
      if (controller.signal.aborted || sampleRequestRef.current !== controller) return;
      setSamples((current) => reset ? data.items : [...current, ...data.items]);
      setSelectedSample((current) => current ?? data.items[0] ?? null);
      offsetRef.current = data.offset + data.items.length;
      setHasMore(data.has_more);
      setTotal(data.total);
    } catch (error) {
      if (!isAbortError(error)) setSampleError(error instanceof Error ? error.message : String(error));
    } finally {
      if (sampleRequestRef.current === controller) {
        loadingRef.current = false;
        setLoadingMore(false);
      }
    }
  }, [classFilter, configPayload, splitFilter]);

  useEffect(() => {
    void loadSamples(true);
    return () => sampleRequestRef.current?.abort();
  }, [loadSamples]);

  if (previewError && !preview) {
    return <StateMessage title="Could not load the Train dataset" detail={previewError} error onRetry={() => { void loadPreview(); void loadSamples(true); }} />;
  }
  if (!preview) return <StateMessage title="Rebuilding the Train dataset..." detail="The data is regenerated from the current Train tab configuration." />;

  const toggleClass = (label: number): void => {
    setVisibleClasses((current) => {
      const next = new Set(current);
      if (next.has(label)) next.delete(label); else next.add(label);
      return next;
    });
  };
  const visibleFeatureNames = preview.feature_names;
  const previewColumns = visibleFeatureNames.slice(0, Math.min(6, visibleFeatureNames.length));
  const splitLabels: Array<{ value: SplitName; label: string }> = [
    { value: "all", label: "All splits" },
    { value: "training", label: "Training" },
    { value: "validation", label: "Validation" },
    { value: "testing", label: "Testing" },
  ];
  return (
    <div className="dataset-view">
      <section className="panel-surface dataset-overview-panel">
        <div className="section-heading compact">
          <div><div className="section-kicker">TRAIN DATASET</div><h2>Generated classification data</h2></div>
          <p>Preview regenerated from the current Train configuration.</p>
        </div>
        <div className="dataset-metric-grid">
          <Metric label="Total samples" value={formatCount(preview.sample_count)} />
          <Metric label="Classes" value={String(preview.class_count)} />
          <Metric label="Input features" value={String(preview.feature_names.length)} />
          <Metric label="Noise" value={`${Math.round(trainConfig.difficulty * 100)}%`} />
        </div>
        <div className="dataset-split-row">
          {Object.entries(preview.split_counts).map(([split, count]) => <span key={split}><strong>{formatCount(count)}</strong> {split}</span>)}
        </div>
      </section>

      <section className="dataset-train-layout">
        <div className="panel-surface dataset-scatter-panel">
          <div className="section-heading compact">
            <div><div className="section-kicker">FEATURE MAP</div><h2>Class separation</h2></div>
            <p>{preview.points.length} representative points</p>
          </div>
          <div className="dataset-feature-selectors">
            <label>X axis<select value={featureX} onChange={(event) => setFeatureX(Number(event.target.value))}>{visibleFeatureNames.map((name, index) => <option key={name} value={index} disabled={index === featureY}>{name}</option>)}</select></label>
            <label>Y axis<select value={featureY} onChange={(event) => setFeatureY(Number(event.target.value))}>{visibleFeatureNames.map((name, index) => <option key={name} value={index} disabled={index === featureX}>{name}</option>)}</select></label>
          </div>
          {previewLoading && <p className="dataset-loading-note">Updating projection...</p>}
          <ScatterPlot points={preview.points} visibleClasses={visibleClasses} featureX={visibleFeatureNames[featureX]} featureY={visibleFeatureNames[featureY]} />
          <div className="dataset-class-legend" aria-label="Toggle classes in scatter plot">
            {CLASS_LABELS.map((label) => <button key={label} className={visibleClasses.has(label) ? "is-visible" : "is-hidden"} type="button" aria-pressed={visibleClasses.has(label)} onClick={() => toggleClass(label)}><span style={{ backgroundColor: CLASS_COLORS[label] }} />class_{label}</button>)}
          </div>
        </div>
        <div className="panel-surface dataset-vector-panel">
          <div className="section-kicker">SELECTED VECTOR</div>
          {selectedSample ? (
            <>
              <h2>Sample #{selectedSample.index}</h2>
              <p>class_{selectedSample.label} · {selectedSample.split}</p>
              <div className="dataset-vector-grid">{selectedSample.features.map((value, index) => <div key={index}><small>{preview.feature_names[index]}</small><code>{value.toFixed(3)}</code></div>)}</div>
            </>
          ) : <p className="dataset-empty-copy">Choose a row in the table to view the full vector.</p>}
        </div>
      </section>

      <section className="panel-surface dataset-browser-panel">
        <div className="section-heading compact">
          <div><div className="section-kicker">SAMPLE TABLE</div><h2>{formatCount(total)} filtered samples</h2></div>
          <p>Shows a subset of features. The detail panel keeps the full vector.</p>
        </div>
        <div className="dataset-table-filters">
          <label>Class<select value={classFilter ?? "all"} onChange={(event) => setClassFilter(event.target.value === "all" ? null : Number(event.target.value))}><option value="all">All classes</option>{CLASS_LABELS.map((label) => <option value={label} key={label}>class_{label}</option>)}</select></label>
          <label>Split<select value={splitFilter} onChange={(event) => setSplitFilter(event.target.value as SplitName)}>{splitLabels.map((item) => <option value={item.value} key={item.value}>{item.label}</option>)}</select></label>
        </div>
        {sampleError && <div className="dataset-inline-error" role="alert">{sampleError} <button type="button" onClick={() => void loadSamples(samples.length === 0)}>Retry</button></div>}
        {samples.length ? (
          <>
            <div className="dataset-table-wrap">
              <table className="dataset-sample-table"><thead><tr><th>Index</th><th>Class</th><th>Split</th>{previewColumns.map((name) => <th key={name}>{name}</th>)}</tr></thead><tbody>{samples.map((sample) => <tr key={sample.index} className={selectedSample?.index === sample.index ? "is-selected" : ""} tabIndex={0} onClick={() => setSelectedSample(sample)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setSelectedSample(sample); } }}><td>{sample.index}</td><td>class_{sample.label}</td><td>{sample.split}</td>{previewColumns.map((_name, index) => <td key={index}>{sample.features[index]?.toFixed(2)}</td>)}</tr>)}</tbody></table>
            </div>
            <LoadMore hasMore={hasMore} loading={loadingMore} onLoad={() => void loadSamples(false)} endLabel="All filtered samples are shown." />
          </>
        ) : loadingMore ? <StateMessage title="Loading samples..." detail="Preparing the first page." /> : <StateMessage title="No samples" detail="The current filter returned no samples." />}
      </section>
    </div>
  );
}

export function DatasetPanel({ trainConfig }: { trainConfig: DatasetTrainConfig }): ReactElement {
  const [activeDatasetTab, setActiveDatasetTab] = useState<DatasetTab>("predict");
  return (
    <div className="dataset-shell">
      <section className="lesson-header dataset-hero" aria-label="Dataset overview">
        <div className="lesson-body">
          <div className="lesson-copy">
            <div className="section-kicker">DATASET / READ THE INPUT</div>
            <h1>See the data before the model sees it</h1>
            <p className="intro-copy">Browse the real samples used by Predict and the generated vectors used by Train.</p>
          </div>
        </div>
      </section>
      <nav className="dataset-subtabs" aria-label="Dataset views">
        <button type="button" role="tab" aria-selected={activeDatasetTab === "predict"} className={activeDatasetTab === "predict" ? "is-active" : ""} onClick={() => setActiveDatasetTab("predict")}>Predict dataset</button>
        <button type="button" role="tab" aria-selected={activeDatasetTab === "train"} className={activeDatasetTab === "train" ? "is-active" : ""} onClick={() => setActiveDatasetTab("train")}>Train dataset</button>
      </nav>
      {activeDatasetTab === "predict" ? <PredictDatasetView /> : <TrainDatasetView trainConfig={trainConfig} />}
    </div>
  );
}
