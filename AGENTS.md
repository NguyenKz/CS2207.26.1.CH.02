# Repository Agent Instructions

## Project context

- This repository contains coursework and experiments, including the moderation-dashboard demo under `CK/demo/`.
- For the demo's current UI and behavior, use `CK/demo/methods_refactored.html` as the reference implementation.
- Before changing that demo, read `CK/demo/UI_AGENT_RULES.md`; its UI requirements and acceptance criteria are authoritative for work in `CK/demo/`.

## UI work in `CK/demo/`

- Treat this as a compact, information-first dashboard, not a long-scrolling landing page. Prioritize the comparison panels and visualizations in the initial desktop viewport.
- Keep desktop header and controls compact; use side-by-side comparison panels when space permits. On small screens, prioritize readable content and accessible scrolling over zero-scroll.
- Avoid hiding overflow to simulate a fit. Keep long lists scrollable within their panel where appropriate, and ensure controls and important information remain accessible.
- Preserve the existing methods, comparator and budget controls, Heatmap/2D/List modes, Reset, reviewed state, KPI meanings, labels, data, and calculations unless the task explicitly requests a behavior change.
- Keep mock/demo status clear. Do not silently alter data or metric semantics to improve presentation; report inconsistencies separately.
- Make interactions keyboard-accessible; do not rely on color or hover alone to convey essential information.
- Verify real viewport behavior, not just source or a full-page screenshot. Follow the required viewport sizes and regression checks in `CK/demo/UI_AGENT_RULES.md`; report which checks were actually performed and any remaining issues.

## General repository work

- Keep changes scoped to the requested task and follow existing project conventions.
- Do not make unrelated cleanup or change coursework results/data without an explicit request.
