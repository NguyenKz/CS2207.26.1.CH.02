import { Fragment, useEffect, useMemo, useRef, useState, type ReactElement } from "react";

const ACTIVATIONS = [
  "tanh",
  "sigmoid",
  "relu",
  "leaky_relu",
  "softplus",
  "identity",
] as const;

type ActivationName = (typeof ACTIVATIONS)[number];

type StudioLayer = {
  id: string;
  neurons: number;
  activation: ActivationName;
};

type HiddenNeuron = {
  index: number;
  z: number;
  h: number;
  weights: number[];
  bias: number;
};

type OutputNeuron = {
  index: number;
  logit: number;
  probability: number;
  weights: number[];
  bias: number;
};

type ForwardTrace = {
  features_raw: number[];
  features_normalized: number[];
  activation: ActivationName;
  hidden_neuron_count: number;
  hidden_layers: Array<{ neurons: number; activation: ActivationName }>;
  layers: Array<{
    index: number;
    activation: ActivationName;
    neuron_count: number;
    neurons: HiddenNeuron[];
  }>;
  hidden: HiddenNeuron[];
  output: OutputNeuron[];
  predicted_class: number;
};

type BuildResult = {
  model_id: string;
  activation: string;
  hidden_neuron_count: number;
  epochs_trained: number;
  final_training_loss: number;
  validation_loss: number;
  validation_accuracy: number;
  hidden_layers?: Array<{ neurons: number; activation: ActivationName }>;
};

type Point = { x: number; y: number; label: string; value: number | null };

const API_BASE = "http://localhost:6788";

const ACTIVATION_LABELS: Record<ActivationName, string> = {
  tanh: "tanh",
  sigmoid: "Sigmoid",
  relu: "ReLU",
  leaky_relu: "Leaky ReLU",
  softplus: "Softplus",
  identity: "Identity",
};

const DEFAULT_STUDIO_LAYERS: StudioLayer[] = [
  { id: "hidden-1", neurons: 8, activation: "tanh" },
];

function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value == null || Number.isNaN(value)) return "n/a";
  return value.toFixed(digits);
}

function formatInputValue(value: number): number {
  return Number(value.toFixed(3));
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function valueFill(value: number | null, maxAbs: number, kind: "input" | "hidden" | "output"): string {
  if (value == null || maxAbs <= 0) {
    if (kind === "hidden") return "rgba(208, 90, 69, 0.22)";
    if (kind === "output") return "rgba(44, 122, 123, 0.24)";
    return "#e4dfd5";
  }
  const intensity = clamp(Math.abs(value) / maxAbs, 0.15, 1);
  if (kind === "output") {
    return `rgba(44, 122, 123, ${0.25 + intensity * 0.7})`;
  }
  if (value < 0) {
    return `rgba(107, 92, 165, ${0.25 + intensity * 0.7})`;
  }
  return `rgba(208, 90, 69, ${0.25 + intensity * 0.7})`;
}

function useElementWidth() {
  const elementRef = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    const updateWidth = () => setWidth(Math.round(element.getBoundingClientRect().width));
    updateWidth();

    const observer = new ResizeObserver(updateWidth);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return [elementRef, width] as const;
}

function NetworkGraph({
  trace,
  placeholderHidden,
  selectedHidden,
  onSelectHidden,
}: {
  trace: ForwardTrace | null;
  placeholderHidden: number;
  selectedHidden: number;
  onSelectHidden: (index: number) => void;
}): ReactElement {
  const hiddenCount = trace?.hidden_neuron_count ?? placeholderHidden;
  const [graphRef, containerWidth] = useElementWidth();
  const isMobile = containerWidth > 0 && containerWidth < 640;
  const width = isMobile ? Math.max(containerWidth, 300) : Math.max(containerWidth, 760);
  const denseDesktop = !isMobile && hiddenCount > 16;
  const compactDesktop = !isMobile && !denseDesktop;
  const nodeRadius = isMobile ? 16 : denseDesktop ? 8 : 12;
  const desktopHiddenColumns = Math.min(hiddenCount, 4);
  const desktopHiddenRows = Math.ceil(hiddenCount / desktopHiddenColumns);
  const desktopHiddenRowGap = denseDesktop ? 24 : 52;
  const layerTop = isMobile ? 0 : denseDesktop ? 40 : 46;
  const layerBottom = isMobile ? 0 : denseDesktop ? 14 : 14;
  const desktopHeight = Math.max(
    denseDesktop ? 220 : 280,
    layerTop + Math.max(desktopHiddenRows - 1, 0) * desktopHiddenRowGap + layerBottom,
  );
  const desktopHiddenBlockHeight = Math.max(desktopHiddenRows - 1, 0) * desktopHiddenRowGap;
  const desktopHiddenStartY = denseDesktop
    ? layerTop + Math.max((desktopHeight - layerTop - layerBottom - desktopHiddenBlockHeight) / 2, 0)
    : layerTop + (desktopHeight - layerTop - layerBottom - desktopHiddenBlockHeight) / 2;
  const mobileColumns = hiddenCount > 12 ? 3 : 2;
  const mobileRows = Math.ceil(hiddenCount / mobileColumns);
  const mobileHiddenStartY = 170;
  const mobileHiddenRowGap = 60;
  const mobileOutputY = mobileHiddenStartY + Math.max(mobileRows - 1, 0) * mobileHiddenRowGap + 104;
  const height = isMobile ? mobileOutputY + 64 : desktopHeight;

  const layerPositions = (count: number, x: number): Point[] => {
    const usable = height - layerTop - layerBottom;
    if (count === 1) {
      return [{ x, y: layerTop + usable / 2, label: "", value: null }];
    }
    return Array.from({ length: count }, (_, index) => ({
      x,
      y: layerTop + (index / (count - 1)) * usable,
      label: "",
      value: null,
    }));
  };

  const inputValues = trace?.features_normalized ?? [null, null, null, null];
  const hiddenValues = trace?.hidden.map((neuron) => neuron.h) ?? Array.from({ length: hiddenCount }, () => null);
  const outputValues = trace?.output.map((neuron) => neuron.probability) ?? [null, null, null];

  const inputNodes = (isMobile
    ? Array.from({ length: 4 }, (_, index) => ({
        x: 32 + (index / 3) * (width - 64),
        y: 78,
        label: "",
        value: null,
      }))
    : layerPositions(4, 90)
  ).map((point, index) => ({
    ...point,
    label: `x${index + 1}`,
    value: inputValues[index],
  }));

  const hiddenNodes = (isMobile
    ? Array.from({ length: hiddenCount }, (_, index) => ({
        x: ((index % mobileColumns) + 0.5) * (width / mobileColumns),
        y: mobileHiddenStartY + Math.floor(index / mobileColumns) * mobileHiddenRowGap,
        label: "",
        value: null,
      }))
    : Array.from({ length: hiddenCount }, (_, index) => {
        const column = index % desktopHiddenColumns;
        const row = Math.floor(index / desktopHiddenColumns);
        const gridStartX = width * 0.31;
        const gridWidth = width * 0.38;
        return {
          x:
            desktopHiddenColumns === 1
              ? width / 2
              : gridStartX + (column / (desktopHiddenColumns - 1)) * gridWidth,
          y: desktopHiddenStartY + row * desktopHiddenRowGap,
          label: "",
          value: null,
        };
      }))
  .map((point, index) => ({
    ...point,
    label: `H${index + 1}`,
    value: hiddenValues[index],
  }));

  const outputNodes = (isMobile
    ? Array.from({ length: 3 }, (_, index) => ({
        x: 32 + (index / 2) * (width - 64),
        y: mobileOutputY,
        label: "",
        value: null,
      }))
    : layerPositions(3, width - 90)
  ).map((point, index) => ({
    ...point,
    label: `O${index}`,
    value: outputValues[index],
  }));

  const maxInputAbs = Math.max(...inputValues.map((value) => Math.abs(value ?? 0)), 0.001);
  const maxHiddenAbs = Math.max(...hiddenValues.map((value) => Math.abs(value ?? 0)), 0.001);
  const maxOutputAbs = Math.max(...outputValues.map((value) => Math.abs(value ?? 0)), 0.001);

  const maxInWeight = trace
    ? Math.max(...trace.hidden.flatMap((neuron) => neuron.weights.map(Math.abs)), 0.001)
    : 1;
  const maxOutWeight = trace
    ? Math.max(...trace.output.flatMap((neuron) => neuron.weights.map(Math.abs)), 0.001)
    : 1;

  const edgeEnds = (from: { x: number; y: number }, to: { x: number; y: number }) =>
    isMobile
      ? { x1: from.x, y1: from.y + nodeRadius, x2: to.x, y2: to.y - nodeRadius }
      : { x1: from.x + nodeRadius, y1: from.y, x2: to.x - nodeRadius, y2: to.y };

  const nodeLabelY = (node: Point) => node.y - nodeRadius - 10;

  return (
    <div className="network-graph-wrap" ref={graphRef}>
      <svg
        className="network-graph"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="ANN graph from input to hidden to output"
      >
        <text
          x={isMobile ? width / 2 : 90}
          y={isMobile ? 26 : compactDesktop || denseDesktop ? 16 : 26}
          className="graph-layer-label"
          textAnchor="middle"
        >
          Input
        </text>
        <text
          x={isMobile ? width / 2 : width / 2}
          y={isMobile ? 136 : compactDesktop || denseDesktop ? 16 : 26}
          className="graph-layer-label"
          textAnchor="middle"
        >
          Hidden · {trace?.activation ?? "…"}
        </text>
        <text
          x={isMobile ? width / 2 : width - 90}
          y={isMobile ? mobileOutputY - 38 : compactDesktop || denseDesktop ? 16 : 26}
          className="graph-layer-label"
          textAnchor="middle"
        >
          Softmax
        </text>

        {inputNodes.flatMap((inputNode, inputIndex) =>
            hiddenNodes.map((hiddenNode, hiddenIndex) => {
              const weight = trace?.hidden[hiddenIndex].weights[inputIndex] ?? 0;
              const selected = Boolean(trace) && selectedHidden === hiddenIndex;
              const strength = Math.abs(weight) / maxInWeight;
              const ends = edgeEnds(inputNode, hiddenNode);
              return (
                <line
                  key={`ih-${inputIndex}-${hiddenIndex}`}
                  {...ends}
                  className={selected ? "edge edge-hot" : "edge edge-dim"}
                  stroke={trace ? (weight >= 0 ? "#d05a45" : "#6b5ca5") : "#d05a45"}
                  strokeWidth={trace ? 0.6 + strength * (selected ? 3.2 : 1.4) : 0.9}
                  opacity={trace ? (selected ? 0.35 + strength * 0.55 : 0.08 + strength * 0.12) : 0.2}
                  strokeDasharray={trace ? undefined : "3 4"}
                />
              );
            }),
          )}

        {hiddenNodes.flatMap((hiddenNode, hiddenIndex) =>
            outputNodes.map((outputNode, outputIndex) => {
              const weight = trace?.output[outputIndex].weights[hiddenIndex] ?? 0;
              const selected = Boolean(trace) && selectedHidden === hiddenIndex;
              const strength = Math.abs(weight) / maxOutWeight;
              const winner = trace?.predicted_class === outputIndex;
              const ends = edgeEnds(hiddenNode, outputNode);
              return (
                <line
                  key={`ho-${hiddenIndex}-${outputIndex}`}
                  {...ends}
                  className={selected || winner ? "edge edge-hot" : "edge edge-dim"}
                  stroke={trace ? (weight >= 0 ? "#2c7a7b" : "#6b5ca5") : "#2c7a7b"}
                  strokeWidth={trace ? 0.6 + strength * (selected ? 3.2 : 1.4) : 0.9}
                  opacity={
                    trace
                      ? selected
                        ? 0.35 + strength * 0.55
                        : winner
                          ? 0.18 + strength * 0.2
                          : 0.08 + strength * 0.12
                      : 0.2
                  }
                  strokeDasharray={trace ? undefined : "3 4"}
                />
              );
            }),
          )}

        {inputNodes.map((node, index) => (
          <g key={`in-${index}`}>
            <circle
              cx={node.x}
              cy={node.y}
              r={nodeRadius}
              fill={valueFill(node.value, maxInputAbs, "input")}
              stroke="#172027"
              strokeWidth={1.2}
            />
            <text x={node.x} y={nodeLabelY(node)} className="graph-node-label" textAnchor="middle">
              {node.label}
            </text>
            <text x={node.x} y={node.y + 4} className="graph-node-value" textAnchor="middle">
              {formatNumber(node.value, 2)}
            </text>
          </g>
        ))}

        {hiddenNodes.map((node, index) => {
          const selected = selectedHidden === index;
          return (
            <g
              key={`h-${index}`}
              className="graph-hit"
              onClick={() => onSelectHidden(index)}
              onKeyDown={(event) => {
                if (trace && (event.key === "Enter" || event.key === " ")) {
                  event.preventDefault();
                  onSelectHidden(index);
                }
              }}
              role={trace ? "button" : undefined}
              tabIndex={trace ? 0 : undefined}
              aria-label={trace ? `Select hidden neuron H${index + 1}` : undefined}
              style={{ cursor: trace ? "pointer" : "default" }}
            >
              <circle
                cx={node.x}
                cy={node.y}
                r={selected ? nodeRadius + 4 : nodeRadius}
                fill={valueFill(node.value, maxHiddenAbs, "hidden")}
                stroke={selected ? "#d05a45" : "#172027"}
                strokeWidth={selected ? 2.4 : 1.2}
              />
              <text x={node.x} y={nodeLabelY(node) - (selected ? 4 : 0)} className="graph-node-label" textAnchor="middle">
                {node.label}
              </text>
              <text x={node.x} y={node.y + 4} className="graph-node-value" textAnchor="middle">
                {formatNumber(node.value, 2)}
              </text>
            </g>
          );
        })}

        {outputNodes.map((node, index) => {
          const winner = trace?.predicted_class === index;
          return (
            <g key={`out-${index}`}>
              <circle
                cx={node.x}
                cy={node.y}
                r={winner ? nodeRadius + 4 : nodeRadius}
                fill={valueFill(node.value, maxOutputAbs, "output")}
                stroke={winner ? "#2c7a7b" : "#172027"}
                strokeWidth={winner ? 2.4 : 1.2}
              />
              <text x={node.x} y={nodeLabelY(node) - (winner ? 4 : 0)} className="graph-node-label" textAnchor="middle">
                {node.label}
              </text>
              <text x={node.x} y={node.y + 4} className="graph-node-value" textAnchor="middle">
                {formatNumber(node.value, 2)}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="graph-legend">
        <span><i className="swatch swatch-pos" /> positive weight / activation</span>
        <span><i className="swatch swatch-neg" /> negative</span>
        <span><i className="swatch swatch-out" /> output probability</span>
        <span>Line thickness ∝ |weight| · click a hidden node</span>
      </div>
    </div>
  );
}

type GraphLayer = {
  key: string;
  label: string;
  size: number;
  activation: string;
  kind: "input" | "hidden" | "output";
  layerIndex: number | null;
  values: Array<number | null>;
};

function LayeredNetworkGraph({
  trace,
  studioLayers,
  selectedLayer,
  selectedHidden,
  onSelectHidden,
}: {
  trace: ForwardTrace | null;
  studioLayers: StudioLayer[];
  selectedLayer: number;
  selectedHidden: number;
  onSelectHidden: (layerIndex: number, neuronIndex: number) => void;
}): ReactElement {
  const [graphRef, containerWidth] = useElementWidth();
  const isMobile = containerWidth > 0 && containerWidth < 640;
  const width = isMobile ? Math.max(containerWidth, 300) : Math.max(containerWidth, 760);
  const graphLayers: GraphLayer[] = [
    {
      key: "input",
      label: "Input",
      size: 4,
      activation: "features",
      kind: "input",
      layerIndex: null,
      values: trace?.features_normalized ?? [null, null, null, null],
    },
    ...studioLayers.map((layer, index) => ({
      key: layer.id,
      label: `Hidden ${index + 1}`,
      size: trace?.layers[index]?.neuron_count ?? layer.neurons,
      activation: trace?.layers[index]?.activation ?? layer.activation,
      kind: "hidden" as const,
      layerIndex: index,
      values: trace?.layers[index]?.neurons.map((neuron) => neuron.h)
        ?? Array.from({ length: layer.neurons }, () => null),
    })),
    {
      key: "output",
      label: "Softmax",
      size: 3,
      activation: "probability",
      kind: "output",
      layerIndex: null,
      values: trace?.output.map((neuron) => neuron.probability) ?? [null, null, null],
    },
  ];

  const maxLayerSize = Math.max(...graphLayers.map((layer) => layer.size), 1);
  const dense = !isMobile && maxLayerSize > 16;
  const nodeRadius = isMobile ? 14 : dense ? 8 : 13;
  const desktopNodeColumns = (_layer: GraphLayer) => 1;
  const desktopRowGap = dense ? 22 : 40;
  const maxRows = Math.max(
    ...graphLayers.map((layer) => Math.ceil(layer.size / desktopNodeColumns(layer))),
    1,
  );
  const layerTop = dense ? 40 : 46;
  const layerBottom = dense ? 14 : 18;
  const height = isMobile
    ? graphLayers.reduce((total, layer) => {
        const columns = 1;
        return total + 82 + Math.max(Math.ceil(layer.size / columns) - 1, 0) * 42;
      }, 36)
    : Math.max(dense ? 220 : 280, layerTop + (maxRows - 1) * desktopRowGap + layerBottom);

  const layerX = (layerIndex: number) =>
    graphLayers.length === 1
      ? width / 2
      : 70 + (layerIndex / (graphLayers.length - 1)) * (width - 140);

  const nodePositions = graphLayers.map((layer, layerIndex) => {
    if (isMobile) {
      const previousLayers = graphLayers.slice(0, layerIndex);
      const y = previousLayers.reduce((total, previousLayer) => {
        const columns = 1;
        return total + 82 + Math.max(Math.ceil(previousLayer.size / columns) - 1, 0) * 42;
      }, 65 + layerIndex * 0);
      const columns = 1;
      return Array.from({ length: layer.size }, (_, index) => ({
        x: ((index % columns) + 0.5) * (width / columns),
        y: y + Math.floor(index / columns) * 42,
      }));
    }

    const columns = desktopNodeColumns(layer);
    const rows = Math.ceil(layer.size / columns);
    const blockHeight = Math.max(rows - 1, 0) * desktopRowGap;
    const startY = layerTop + (height - layerTop - layerBottom - blockHeight) / 2;
    const gridWidth = Math.min(140, Math.max(48, width / (graphLayers.length * 2.8)));
    return Array.from({ length: layer.size }, (_, index) => {
      const column = index % columns;
      const row = Math.floor(index / columns);
      return {
        x: columns === 1
          ? layerX(layerIndex)
          : layerX(layerIndex) + (column / (columns - 1) - 0.5) * gridWidth,
        y: startY + row * desktopRowGap,
      };
    });
  });

  const edgeEnds = (from: { x: number; y: number }, to: { x: number; y: number }) =>
    isMobile
      ? { x1: from.x, y1: from.y + nodeRadius, x2: to.x, y2: to.y - nodeRadius }
      : { x1: from.x + nodeRadius, y1: from.y, x2: to.x - nodeRadius, y2: to.y };

  const weightFor = (targetLayerIndex: number, targetNeuronIndex: number, sourceNeuronIndex: number) => {
    if (!trace) return 0;
    if (targetLayerIndex < studioLayers.length) {
      return trace.layers[targetLayerIndex]?.neurons[targetNeuronIndex]?.weights[sourceNeuronIndex] ?? 0;
    }
    return trace.output[targetNeuronIndex]?.weights[sourceNeuronIndex] ?? 0;
  };

  const maxWeight = Math.max(
    ...(trace?.layers.flatMap((layer) => layer.neurons.flatMap((neuron) => neuron.weights.map(Math.abs))) ?? []),
    ...(trace?.output.flatMap((neuron) => neuron.weights.map(Math.abs)) ?? []),
    0.001,
  );

  return (
    <div className="network-graph-wrap" ref={graphRef}>
      <svg
        className="network-graph"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="ANN graph showing every layer from input to output"
      >
        {graphLayers.map((layer, index) => {
          const firstNode = nodePositions[index][0];
          return (
            <text
              key={`layer-label-${layer.key}`}
              x={isMobile ? width / 2 : layerX(index)}
              y={isMobile ? firstNode.y - nodeRadius - 22 : 18}
              className="graph-layer-label"
              textAnchor="middle"
            >
              {isMobile ? `${layer.label} · ${layer.activation}` : layer.label}
            </text>
          );
        })}

        {graphLayers.slice(0, -1).flatMap((sourceLayer, sourceLayerIndex) => {
          const targetLayer = graphLayers[sourceLayerIndex + 1];
          return nodePositions[sourceLayerIndex].flatMap((sourceNode, sourceIndex) =>
            nodePositions[sourceLayerIndex + 1].map((targetNode, targetIndex) => {
              const targetIsOutput = targetLayer.kind === "output";
              const targetHiddenLayer = targetIsOutput ? studioLayers.length : targetLayer.layerIndex ?? 0;
              const weight = weightFor(targetHiddenLayer, targetIndex, sourceIndex);
              const strength = Math.abs(weight) / maxWeight;
              const selected = targetLayer.layerIndex === selectedLayer && targetIndex === selectedHidden;
              const winner = targetIsOutput && trace?.predicted_class === targetIndex;
              const ends = edgeEnds(sourceNode, targetNode);
              const stroke = targetIsOutput
                ? weight >= 0 ? "#2c7a7b" : "#6b5ca5"
                : weight >= 0 ? "#d05a45" : "#6b5ca5";
              return (
                <line
                  key={`edge-${sourceLayer.key}-${sourceIndex}-${targetLayer.key}-${targetIndex}`}
                  {...ends}
                  className={selected || winner ? "edge edge-hot" : "edge edge-dim"}
                  stroke={trace ? stroke : targetIsOutput ? "#2c7a7b" : "#d05a45"}
                  strokeWidth={trace ? 0.7 + strength * (selected ? 3 : 1.5) : 1}
                  opacity={trace ? selected ? 0.45 + strength * 0.5 : winner ? 0.25 : 0.07 + strength * 0.12 : 0.2}
                  strokeDasharray={trace ? undefined : "3 4"}
                />
              );
            }),
          );
        })}

        {graphLayers.flatMap((layer, layerIndex) =>
          nodePositions[layerIndex].map((node, neuronIndex) => {
            const selected = layer.layerIndex === selectedLayer && neuronIndex === selectedHidden;
            const winner = layer.kind === "output" && trace?.predicted_class === neuronIndex;
            const maxAbs = Math.max(...layer.values.map((value) => Math.abs(value ?? 0)), 0.001);
            const kind = layer.kind === "output" ? "output" : layer.kind === "input" ? "input" : "hidden";
            const label = layer.kind === "input"
              ? `x${neuronIndex + 1}`
              : layer.kind === "output"
                ? `O${neuronIndex}`
                : `H${neuronIndex + 1}`;
            return (
              <g
                key={`node-${layer.key}-${neuronIndex}`}
                className={layer.kind === "hidden" ? "graph-hit" : undefined}
                onClick={() => layer.layerIndex != null && onSelectHidden(layer.layerIndex, neuronIndex)}
                onKeyDown={(event) => {
                  if (layer.layerIndex != null && trace && (event.key === "Enter" || event.key === " ")) {
                    event.preventDefault();
                    onSelectHidden(layer.layerIndex, neuronIndex);
                  }
                }}
                role={layer.kind === "hidden" && trace ? "button" : undefined}
                tabIndex={layer.kind === "hidden" && trace ? 0 : undefined}
                aria-label={layer.kind === "hidden" && trace
                  ? layer.layerIndex === 0
                    ? `Select hidden neuron H${neuronIndex + 1}`
                    : `Select hidden neuron layer ${layer.layerIndex! + 1} H${neuronIndex + 1}`
                  : undefined}
                style={{ cursor: layer.kind === "hidden" && trace ? "pointer" : "default" }}
              >
                <circle
                  cx={node.x}
                  cy={node.y}
                  r={selected || winner ? nodeRadius + 3 : nodeRadius}
                  fill={valueFill(layer.values[neuronIndex], maxAbs, kind)}
                  stroke={selected ? "#d05a45" : winner ? "#2c7a7b" : "#172027"}
                  strokeWidth={selected || winner ? 2.4 : 1.2}
                />
                <text x={node.x} y={node.y - nodeRadius - 7} className="graph-node-label" textAnchor="middle">
                  {label}
                </text>
                <text x={node.x} y={node.y + 4} className="graph-node-value" textAnchor="middle">
                  {formatNumber(layer.values[neuronIndex], 2)}
                </text>
              </g>
            );
          }),
        )}
      </svg>
      <div className="graph-legend">
        <span><i className="swatch swatch-pos" /> positive weight / activation</span>
        <span><i className="swatch swatch-neg" /> negative</span>
        <span><i className="swatch swatch-out" /> output probability</span>
        <span>Line thickness ∝ |weight| · click any hidden neuron</span>
      </div>
    </div>
  );
}

function ActivationBars({
  title,
  labels,
  values,
  highlightIndex,
  mode,
}: {
  title: string;
  labels: string[];
  values: number[];
  highlightIndex?: number;
  mode: "signed" | "prob";
}): ReactElement {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const width = 420;
  const height = 200;
  const left = 42;
  const right = 18;
  const top = 18;
  const bottom = 28;
  const chartWidth = width - left - right;
  const chartHeight = height - top - bottom;
  const maxAbs =
    mode === "prob"
      ? 1
      : Math.max(...values.map((value) => Math.abs(value)), 0.001);
  const zeroY = mode === "prob" ? top + chartHeight : top + chartHeight / 2;
  const barGap = values.length > 20 ? 3 : 6;
  const barWidth = Math.max(4, chartWidth / Math.max(values.length, 1) - barGap);
  const labelStep = values.length > 20 ? Math.ceil(values.length / 8) : values.length > 10 ? 2 : 1;
  const activeIndex = hoveredIndex ?? highlightIndex ?? null;
  const gridValues = mode === "prob" ? [1, 0.5, 0] : [maxAbs, 0, -maxAbs];

  return (
    <div className="activation-chart panel-surface">
      <div className="chart-heading">
        <div className="section-kicker">{title}</div>
        <span className="chart-unit-note">{values.length} units · hover to inspect</span>
      </div>
      <div className="chart-stage">
        <svg viewBox={`0 0 ${width} ${height}`} className="activation-chart-svg" role="img" aria-label={title}>
          {gridValues.map((gridValue) => {
            const gridY =
              mode === "prob"
                ? top + chartHeight - gridValue * chartHeight
                : zeroY - (gridValue / maxAbs) * (chartHeight / 2);
            return (
              <line
                key={gridValue}
                x1={left}
                y1={gridY}
                x2={left + chartWidth}
                y2={gridY}
                className={gridValue === 0 ? "chart-axis chart-zero" : "chart-grid"}
              />
            );
          })}
        <line
          x1={left}
          y1={top}
          x2={left}
          y2={top + chartHeight}
          className="chart-axis"
        />
        {mode === "signed" && (
          <>
            <text x={left - 8} y={top + 4} className="chart-label" textAnchor="end">
              +{maxAbs.toFixed(1)}
            </text>
            <text x={left - 8} y={top + chartHeight + 4} className="chart-label" textAnchor="end">
              -{maxAbs.toFixed(1)}
            </text>
          </>
        )}
        {mode === "prob" && (
          <>
            <text x={left - 8} y={top + 4} className="chart-label" textAnchor="end">
              1
            </text>
            <text x={left - 8} y={top + chartHeight + 4} className="chart-label" textAnchor="end">
              0
            </text>
          </>
        )}
        {values.map((value, index) => {
          const x = left + index * (barWidth + barGap) + barGap / 2;
          let barHeight: number;
          let y: number;
          if (mode === "prob") {
            barHeight = value * chartHeight;
            y = top + chartHeight - barHeight;
          } else {
            barHeight = (Math.abs(value) / maxAbs) * (chartHeight / 2);
            y = value >= 0 ? zeroY - barHeight : zeroY;
          }
          const highlighted = highlightIndex === index;
          const hovered = activeIndex === index;
          const fill =
            mode === "prob"
              ? highlighted
                ? "#1f6f70"
                : "#2c7a7b"
              : value < 0
                ? "#6b5ca5"
                : highlighted
                  ? "#b44735"
                  : "#d05a45";
          return (
            <g
              key={labels[index]}
              className={hovered ? "chart-bar-group chart-bar-active" : "chart-bar-group"}
              onMouseEnter={() => setHoveredIndex(index)}
              onMouseLeave={() => setHoveredIndex(null)}
            >
              <rect
                x={x}
                y={y}
                width={barWidth}
                height={Math.max(barHeight, 1)}
                fill={fill}
                opacity={hovered || highlighted || mode === "prob" ? 0.95 : 0.72}
                rx={barWidth >= 7 ? 1.5 : 0}
                tabIndex={0}
                role="img"
                aria-label={`${labels[index]}: ${formatNumber(value, 3)}`}
                onFocus={() => setHoveredIndex(index)}
                onBlur={() => setHoveredIndex(null)}
              />
              {index % labelStep === 0 && (
                <text
                  x={x + barWidth / 2}
                  y={height - 8}
                  className="chart-label"
                  textAnchor="middle"
                >
                  {labels[index]}
                </text>
              )}
              {(values.length <= 12 || hovered || highlighted) && (
                <text
                  x={x + barWidth / 2}
                  y={y - 4}
                  className="chart-value-label"
                  textAnchor="middle"
                >
                  {formatNumber(value, 2)}
                </text>
              )}
            </g>
          );
        })}
        </svg>
        {activeIndex != null && values[activeIndex] != null && (
          <div className="chart-tooltip" role="status">
            <strong>{labels[activeIndex]}</strong>
            <span>{formatNumber(values[activeIndex], 3)}</span>
          </div>
        )}
      </div>
    </div>
  );
}

function ModelArchitecturePreview({
  layers,
  trace,
  selectedLayer,
  onSelectLayer,
}: {
  layers: StudioLayer[];
  trace: ForwardTrace | null;
  selectedLayer: number;
  onSelectLayer: (index: number) => void;
}): ReactElement {
  return (
    <div className="studio-architecture-preview" aria-label="Model architecture">
      <div className="studio-architecture-track">
        <div className="studio-layer-block studio-layer-input">
          <span className="studio-layer-index">INPUT</span>
          <strong>4</strong>
          <small>features</small>
        </div>
        {layers.map((layer, index) => {
          const liveLayer = trace?.layers[index];
          const active = selectedLayer === index;
          return (
            <Fragment key={layer.id}>
              <span className="studio-layer-arrow" aria-hidden="true">→</span>
              <button
                type="button"
                className={`studio-layer-block studio-layer-hidden${active ? " studio-layer-selected" : ""}`}
                onClick={() => onSelectLayer(index)}
                aria-pressed={active}
              >
                <span className="studio-layer-index">HIDDEN {index + 1}</span>
                <strong>{layer.neurons}</strong>
                <small>{liveLayer?.activation ?? layer.activation}</small>
              </button>
            </Fragment>
          );
        })}
        <span className="studio-layer-arrow" aria-hidden="true">→</span>
        <div className="studio-layer-block studio-layer-output">
          <span className="studio-layer-index">OUTPUT</span>
          <strong>3</strong>
          <small>softmax</small>
        </div>
      </div>
    </div>
  );
}

export function InspectPanel(): ReactElement {
  const [studioLayers, setStudioLayers] = useState<StudioLayer[]>(DEFAULT_STUDIO_LAYERS);
  const [epochs, setEpochs] = useState(200);
  const [learningRate, setLearningRate] = useState(0.05);
  const [features, setFeatures] = useState<[number, number, number, number]>([0, 0, 0, 0]);
  const [modelId, setModelId] = useState<string | null>(null);
  const [buildInfo, setBuildInfo] = useState<BuildResult | null>(null);
  const [trace, setTrace] = useState<ForwardTrace | null>(null);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHidden, setSelectedHidden] = useState(0);
  const [busy, setBusy] = useState<"idle" | "build" | "forward" | "sample">("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [sampleLabel, setSampleLabel] = useState<number | null>(null);

  const selectedLayerTrace = trace?.layers[selectedLayer] ?? null;
  const selectedStudioLayer = studioLayers[selectedLayer] ?? studioLayers[0];
  const selectedNeuron = selectedLayerTrace?.neurons[selectedHidden] ?? null;

  function markStudioDirty(): void {
    setModelId(null);
    setBuildInfo(null);
    setTrace(null);
    setSampleLabel(null);
  }

  function updateStudioLayer(id: string, patch: Partial<Omit<StudioLayer, "id">>): void {
    setStudioLayers((current) =>
      current.map((layer) => (layer.id === id ? { ...layer, ...patch } : layer)),
    );
    markStudioDirty();
  }

  function addStudioLayer(): void {
    if (studioLayers.length >= 4) return;
    setStudioLayers((current) => [
      ...current,
      {
        id: `hidden-${Date.now()}`,
        neurons: 8,
        activation: "relu",
      },
    ]);
    markStudioDirty();
  }

  function removeStudioLayer(id: string): void {
    if (studioLayers.length <= 1) return;
    setStudioLayers((current) => current.filter((layer) => layer.id !== id));
    setSelectedLayer((current) => Math.min(current, studioLayers.length - 2));
    markStudioDirty();
  }

  async function handleBuild(): Promise<void> {
    setBusy("build");
    setErrorMessage(null);
    setTrace(null);
    try {
      const response = await fetch(`${API_BASE}/inspect/build`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          activation: studioLayers[0].activation,
          hidden_neuron_count: studioLayers[0].neurons,
          hidden_layers: studioLayers.map(({ neurons, activation }) => ({ neurons, activation })),
          epochs,
          learning_rate: learningRate,
          sample_count: 300,
          difficulty: 0.5,
          random_seed: 42,
        }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(detail?.detail ?? `Build failed (${response.status})`);
      }
      const result = (await response.json()) as BuildResult;
      setModelId(result.model_id);
      setBuildInfo(result);
      setSelectedLayer(0);
      setSelectedHidden(0);

      const sampleResponse = await fetch(
        `${API_BASE}/inspect/sample?model_id=${encodeURIComponent(result.model_id)}`,
      );
      if (sampleResponse.ok) {
        const sample = (await sampleResponse.json()) as {
          features: number[];
          label: number;
        };
        const nextFeatures: [number, number, number, number] = [
          formatInputValue(sample.features[0]),
          formatInputValue(sample.features[1]),
          formatInputValue(sample.features[2]),
          formatInputValue(sample.features[3]),
        ];
        setFeatures(nextFeatures);
        setSampleLabel(sample.label);
        await runForward(result.model_id, nextFeatures);
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy("idle");
    }
  }

  async function runForward(
    activeModelId: string,
    nextFeatures: number[],
  ): Promise<void> {
    setBusy("forward");
    setErrorMessage(null);
    try {
      const response = await fetch(`${API_BASE}/inspect/forward`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_id: activeModelId,
          features: nextFeatures,
        }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(detail?.detail ?? `Forward failed (${response.status})`);
      }
      const result = (await response.json()) as ForwardTrace;
      setTrace(result);
      if (selectedLayer >= result.layers.length) {
        setSelectedLayer(0);
      }
      if (selectedHidden >= (result.layers[selectedLayer]?.neurons.length ?? 0)) {
        setSelectedHidden(0);
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy("idle");
    }
  }

  async function handleForward(): Promise<void> {
    if (!modelId) {
      setErrorMessage("Build a model first.");
      return;
    }
    await runForward(modelId, [...features]);
  }

  async function handleLoadSample(): Promise<void> {
    if (!modelId) {
      setErrorMessage("Build a model first.");
      return;
    }
    setBusy("sample");
    setErrorMessage(null);
    try {
      const response = await fetch(
        `${API_BASE}/inspect/sample?model_id=${encodeURIComponent(modelId)}`,
      );
      if (!response.ok) {
        throw new Error(`Sample failed (${response.status})`);
      }
      const sample = (await response.json()) as {
        features: number[];
        label: number;
      };
      const nextFeatures: [number, number, number, number] = [
        formatInputValue(sample.features[0]),
        formatInputValue(sample.features[1]),
        formatInputValue(sample.features[2]),
        formatInputValue(sample.features[3]),
      ];
        setFeatures(nextFeatures);
      setSampleLabel(sample.label);
      await runForward(modelId, nextFeatures);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : String(error));
      setBusy("idle");
    }
  }

  function updateFeature(index: number, value: number): void {
    setFeatures((current) => {
      const next = [...current] as [number, number, number, number];
      next[index] = value;
      return next;
    });
    setSampleLabel(null);
  }

  const formulaTerms = useMemo(() => {
    if (!selectedNeuron) return null;
    return selectedNeuron.weights.map((weight, index) => ({ weight, index }));
  }, [selectedNeuron]);

  return (
    <div className="inspect-shell">
      <section className="lesson-header inspect-hero" aria-label="Inspect lab header">
        <div className="lesson-body inspect-lesson-body">
          <div className="lesson-intro">
            <div className="lesson-copy">
              <div className="section-kicker">LESSON 02 / INSPECT</div>
              <h1>ANN Model Studio</h1>
              <p className="intro-copy">
                Design layers, train one model, and inspect every forward pass from input → hidden → softmax.
              </p>
            </div>
          </div>
        </div>
      </section>

      <div className="inspect-layout">
        <aside className="inspect-controls-column" aria-label="Inspect controls and input">
          <section className="control-panel inspect-control-panel">
            <div className="control-heading">
              <span className="section-kicker">MODEL STUDIO</span>
              <strong>{modelId ? "Model ready" : "Design the network"}</strong>
            </div>
            <div className="studio-editor" aria-label="Hidden layer editor">
              <div className="studio-editor-heading">
                <span>Hidden layers</span>
                <small>{studioLayers.length}/4 layers</small>
              </div>
              <div className="studio-layer-editor-list">
                {studioLayers.map((layer, index) => (
                  <div className="studio-layer-editor" key={layer.id}>
                    <div className="studio-layer-editor-topline">
                      <strong>Layer {index + 1}</strong>
                      <button
                        className="studio-remove-layer"
                        type="button"
                        onClick={() => removeStudioLayer(layer.id)}
                        disabled={busy !== "idle" || studioLayers.length <= 1}
                        aria-label={`Remove hidden layer ${index + 1}`}
                      >
                        ×
                      </button>
                    </div>
                    <div className="studio-layer-editor-fields">
                      <label>
                        Neurons
                        <input
                          type="number"
                          min={1}
                          max={32}
                          value={layer.neurons}
                          disabled={busy !== "idle"}
                          onChange={(event) =>
                            updateStudioLayer(layer.id, { neurons: Number(event.target.value) })
                          }
                        />
                      </label>
                      <label>
                        Activation
                        <select
                          value={layer.activation}
                          disabled={busy !== "idle"}
                          onChange={(event) =>
                            updateStudioLayer(layer.id, {
                              activation: event.target.value as ActivationName,
                            })
                          }
                        >
                          {ACTIVATIONS.map((name) => (
                            <option key={name} value={name}>
                              {ACTIVATION_LABELS[name]}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                  </div>
                ))}
              </div>
              <button
                className="button button-quiet studio-add-layer"
                type="button"
                onClick={addStudioLayer}
                disabled={busy !== "idle" || studioLayers.length >= 4}
              >
                + Add hidden layer
              </button>
            </div>
            <label>
              Epochs
              <input
                type="number"
                min={1}
                max={5000}
                value={epochs}
                disabled={busy !== "idle"}
                onChange={(event) => setEpochs(Number(event.target.value))}
              />
            </label>
            <label>
              Learning rate
              <input
                type="number"
                min={0.001}
                max={1}
                step={0.01}
                value={learningRate}
                disabled={busy !== "idle"}
                onChange={(event) => setLearningRate(Number(event.target.value))}
              />
            </label>
            <div className="control-actions">
              <button
                className="button button-primary"
                type="button"
                onClick={() => void handleBuild()}
                disabled={busy !== "idle"}
              >
                {busy === "build" ? "Building…" : "Build model"}
              </button>
            </div>
            {buildInfo && (
              <p className="inspect-build-meta">
                {studioLayers.map((layer) => `${layer.neurons} ${ACTIVATION_LABELS[layer.activation]}`).join(" → ")}
                {" · "}
                {buildInfo.epochs_trained} epochs · val acc{" "}
                {(buildInfo.validation_accuracy * 100).toFixed(1)}%
              </p>
            )}
            {errorMessage && (
              <p className="inspect-error-note" role="alert">
                {errorMessage}
              </p>
            )}
          </section>

          <section className="panel-surface inspect-input-panel" aria-label="Input features">
            <div className="section-heading compact">
              <div>
                <div className="section-kicker">INPUT</div>
                <h2>Four features</h2>
              </div>
              <p>
                {sampleLabel == null
                  ? "Edit values or load a holdout sample."
                  : `Holdout sample · true label class_${sampleLabel}`}
              </p>
            </div>
            <div className="inspect-feature-grid">
              {features.map((value, index) => (
                <label key={index}>
                  x{index + 1}
                  <input
                    type="number"
                    step="0.01"
                    value={value}
                    disabled={!modelId || busy !== "idle"}
                    onChange={(event) => updateFeature(index, Number(event.target.value))}
                  />
                </label>
              ))}
            </div>
            <div className="control-actions">
              <button
                className="button button-primary"
                type="button"
                onClick={() => void handleForward()}
                disabled={!modelId || busy !== "idle"}
              >
                {busy === "forward" ? "Running…" : "Run forward pass"}
              </button>
              <button
                className="button button-quiet"
                type="button"
                onClick={() => void handleLoadSample()}
                disabled={!modelId || busy !== "idle"}
              >
                Load sample
              </button>
            </div>
            {trace && (
              <div className="inspect-norm-row">
                <span>Normalized</span>
                {trace.features_normalized.map((value, index) => (
                  <code key={index}>{formatNumber(value)}</code>
                ))}
              </div>
            )}
          </section>
        </aside>

        <main className="inspect-visual-column">
          <section className="panel-surface inspect-network" aria-label="Network diagram">
            <div className="section-heading compact">
              <div>
                <div className="section-kicker">MODEL MAP + TRACE</div>
                <h2>
                  {[4, ...studioLayers.map((layer) => layer.neurons), 3].join(" → ")}
                </h2>
              </div>
              <p>
                {trace
                  ? `Predicted class_${trace.predicted_class} · layer ${selectedLayer + 1} selected`
                  : "Design preview · build to reveal learned weights."}
              </p>
            </div>
            <ModelArchitecturePreview
              layers={studioLayers}
              trace={trace}
              selectedLayer={selectedLayer}
              onSelectLayer={(index) => {
                setSelectedLayer(index);
                setSelectedHidden(0);
              }}
            />
            <LayeredNetworkGraph
              trace={trace}
              studioLayers={studioLayers}
              selectedLayer={selectedLayer}
              selectedHidden={selectedHidden}
              onSelectHidden={(layerIndex, neuronIndex) => {
                setSelectedLayer(layerIndex);
                setSelectedHidden(neuronIndex);
              }}
            />
          </section>

          <section className="panel-surface inspect-detail" aria-label="Selected neuron">
            <div className="section-heading compact">
              <div>
                <div className="section-kicker">NEURON DETAIL · LAYER {selectedLayer + 1}</div>
                <h2>H{(selectedNeuron?.index ?? selectedHidden) + 1}</h2>
              </div>
              <p>z = w·a + b, then {selectedLayerTrace?.activation ?? selectedStudioLayer.activation}.</p>
            </div>
            {selectedNeuron && trace && formulaTerms ? (
              <div className="inspect-formula">
                <p className="formula-line">
                  z<sub>{selectedNeuron.index + 1}</sub> ={" "}
                  {formulaTerms.map(({ weight, index }) => (
                    <span key={index}>
                      {index > 0 ? " + " : ""}
                      ({formatNumber(weight)})·{selectedLayer === 0 ? "x" : "a"}<sub>{index + 1}</sub>
                    </span>
                  ))}
                  {" + "}({formatNumber(selectedNeuron.bias)})
                </p>
                <p className="formula-line">
                  z<sub>{selectedNeuron.index + 1}</sub> ={" "}
                  <strong>{formatNumber(selectedNeuron.z)}</strong>
                </p>
                <p className="formula-line">
                  h<sub>{selectedNeuron.index + 1}</sub> = {selectedLayerTrace?.activation ?? "activation"}(z
                  <sub>{selectedNeuron.index + 1}</sub>) ={" "}
                  <strong>{formatNumber(selectedNeuron.h)}</strong>
                </p>
                <div className="inspect-weight-table">
                  {selectedNeuron.weights.map((weight, index) => (
                    <div key={index}>
                      <span>w{index + 1}</span>
                      <code>{formatNumber(weight, 4)}</code>
                    </div>
                  ))}
                  <div>
                    <span>bias</span>
                    <code>{formatNumber(selectedNeuron.bias, 4)}</code>
                  </div>
                </div>
              </div>
            ) : (
              <p className="inspect-empty">Build a model and run a forward pass to inspect a neuron.</p>
            )}
          </section>
        </main>
      </div>
    </div>
  );
}
