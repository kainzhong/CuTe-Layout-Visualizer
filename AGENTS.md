# CuTe Layout Visualizer — Codex guide

This repository is a zero-build, browser-only visualizer for CUTLASS CuTe layouts, layout operations, Copy atoms/TiledCopy partitions, TMA, and warp MMA. The JavaScript is a behavioral port of CuTe/pycute: correctness means matching CuTeDSL, not merely producing a plausible picture.

## Start here

- Read this file before changing code.
- For feature-specific invariants or historical rationale, read the relevant section of [`docs/architecture.md`](docs/architecture.md). Do not load that entire long document unless the task is broad; its headings cover basis layouts, UI infrastructure, testing, Copy/ldmatrix/TMA, MMA, partitioning, tab wiring, and presets.
- `README.md` describes user-visible behavior. `tests/README.md` explains the differential harness.
- Preserve unrelated worktree changes. This project is often developed with several features in flight.

## Runtime architecture

There is no build step, framework, package dependency, or module system. Open `index.html` directly in a browser.

Load order is significant:

```text
cute.js -> layout.js -> ui.js -> tabs/*.js -> inline initialization in index.html
```

- `cute.js`: pure layout parsing/evaluation, scaled-basis coordinates, and colors; no DOM.
- `layout.js`: JavaScript port of pycute layout/int-tuple operations plus project helpers.
- `ui.js`: shared SVG builders, controls, tab/scope framework, deep-link import/export, fullscreen/export, and Copy-pane infrastructure.
- `tabs/*.js`: one global-script file per feature. Each owns its template, mutable state, render function, presets, and small feature-specific helpers.
- `tests/`: Node harness, CuTeDSL corpus/generator, committed oracle, unit tests, DOM smoke tests, and optional GPU execution checks.

All browser symbols are globals. `layout.js` loads after `cute.js` and wins name collisions. Before introducing a top-level function, `const`, `let`, or `class`, search both core files and the tab files. The test harness rejects duplicate top-level lexical names.

## Non-negotiable correctness rules

1. Treat CuTeDSL/CUTLASS as the oracle for ported math. If `cutlass.cute` can express the operation, add a differential case rather than a hand-written expected result.
2. Keep computation separable from the DOM. A render function should parse/read inputs, call a DOM-free computation, then render the result. Pure helpers such as `tmaComputeAtom`, `tpComputePartition`, `psdComputePartition`, and `pabcComputePartition` are the model.
3. Do not hand-edit `tests/reference.json`. Change `tests/cases.json` and `tests/gen_reference.py`, then run `npm run reference` in an environment with `nvidia-cutlass-dsl`.
4. A feature is not finished when only its math passes. `tests/dom_smoke.js` must also cover its defaults, presets, render path, element IDs, visualization hosts, and shareable URL round trip.
5. Every shipped default and preset must render without an error-level diagnostic. Deliberately invalid examples belong in unit tests; an educational warning may be a preset only if the visualization still renders.
6. Do not advertise a new hardware Op based only on construction or tracing. Run `tests/gpu_check.py` on suitable hardware before adding it to a picker. The normal suite intentionally remains GPU-free.

## Commands

```bash
npm test                         # full Node suite; required for JS/UI changes
node tests/run.js <filter>       # focused differential/unit/DOM sections
node tests/run.js --verbose      # print passing assertions
npm run reference                # regenerate oracle; needs nvidia-cutlass-dsl
npm run reference:check          # regenerate and fail on drift
python3 tests/gpu_check.py        # optional GPU gate for newly exposed Ops
```

Run the narrowest relevant test while iterating and `npm test` before handoff. A passing run currently includes differential math, unit validations, and `dom_smoke`; do not silently skip any section.

## Layout and parser invariants

- `parseLayout(str)` accepts ordinary integer-strided layouts. Only pass `{ basis: true }` in operations that CuTe defines for scaled-basis strides.
- Basis-strided layouts are valid for operations that only scale/add/compare strides: composition, coalesce/filter/slice, logical divide and zipped/tiled/flat divide, and local tile.
- Reject basis strides for operations that order strides: complement, inverses, logical product, and product variants. Tilers remain ordinary even when layout A is basis-strided.
- `parseLayout` is not a Tiler parser. CuTe tilers can contain independent modes such as `(8:1, 16:2)`; use `mtcParseTiler` and keep its Python twin in `tests/gen_reference.py` synchronized.
- The app draws 2-D grids. Any layout-syntax field that may flatten outer rank greater than 2 must use the shared rank-warning convention, unless the feature genuinely renders all modes and the architecture document records why it is exempt.

## UI and DOM conventions

- Use `layoutInputField(...)` for layout/shape syntax fields and `statusDivs(prefix)` for standard error/warning hosts. Call `updateRankWarning(...)` from render functions where applicable.
- A `.viz-box` must contain a child SVG host with an id:

  ```html
  <div class="viz-box"><div id="${id}-result-svg"></div></div>
  ```

  Never put the host id on `.viz-box`; rendering into it would delete the fullscreen/export button.
- Reuse shared builders and controls from `ui.js`. Do not fork SVG, zoom, copy-pane, dtype, swizzle, tooltip, highlighting, fullscreen, or export logic into a tab without a concrete feature-specific need.
- `buildTVSVG` defaults to a column-major cell index. A row-major or transposed view must supply its own `cellIndex` so value labels continue to represent the layout codomain.
- Draw boundaries spanning cells with `buildColoredLayoutSVG`'s overlay callback, not doubled per-cell strokes.
- A select whose options depend on another control must be repopulated before assigning its imported/preset value. Assigning a value absent from the old option list silently fails in the browser.
- Keep reference help in `infoIcon(...)` tooltips; tooltip content is plain text.
- Inline handlers are part of the public UI. Preset handlers must remain executable JavaScript and use the tab's `setX('${id}', ...)` shape so `dom_smoke` can drive them. In template literals, encode a newline needed inside an onclick string as `\\n`, not a literal newline.
- Use `.view-toggle` for a tab-wide alternative visualization and `.seg-control` for mutually exclusive modes. Do not combine `.seg-control` with `.mode-btn-group`.

## Shared behavior that must stay shared

- Copy movement legality lives in `COPY_OP_MOVES`; Copy tabs must not offer arbitrary memory-space pairs.
- SIMT Copy Op definitions live in `SIMT_COPY_OPS` so `make_copy_atom`, `make_tiled_copy`, and `make_tiled_copy_tv` cannot drift.
- Copy SRC/DST pane behavior lives in `copyPanes` and related `ui.js` helpers. `partition_sd` and `tma_partition` intentionally do not use those panes because they visualize partitions, not transfers.
- Visualization title badges name the operand's actual memory space, using the shared GMEM/SMEM/RMEM/TMEM palette in `style.css`. Copy badges follow the selected move; `partition_sd` badges follow the selected S/D tensor. Current warp MMA A/B/C fragments are all RMEM. Do not infer a partitioned tensor's space from its layout or recolor SVG ownership cells to indicate memory space.
- Thread highlighting uses `readHighlightTid` where the UI means "focus this thread." `make_tiled_mma` is intentionally different: its focus unit follows the TVs/warps mode and masks other units.
- MMA atom layouts are a lookup table derived from CuTeDSL/CUTLASS hardware traits, not an algebraic derivation. Extend the table from the oracle and add exhaustive cases.
- The MMA alternative view reorders existing panes and transposes B only for display; it must not change the underlying layout.

## Adding or changing a tab

For a new tab, complete all of these in one change:

1. Add `tabs/<feature>.js` with `generate<Feature>TabContent(id)`, a render function, state, presets, and an export helper.
2. Use shared layout inputs/status hosts and correctly nested visualization hosts.
3. Register its tab markup and `data-scope` in `generateTabContent`.
4. Add it to `switchInnerTab`'s `modeIndex`; DOM order and index must agree.
5. Add it to `TAB_RENDER_FN`, set `data-tab`, and name the primary action `Render` so Cmd/Ctrl+Enter works.
6. Add its generator to `generateTabContent` and its script tag to `index.html` after `ui.js` and before inline initialization.
7. Add `FEATURE_SPEC`, `applyKeyParam`, and export wiring. If inputs are Op-dependent, rebuild controls before assigning imported values.
8. Add focused styles only where existing shared classes cannot express the layout.
9. Add CuTeDSL cases and runner/generator support, unit cases only for non-oracle input/validation behavior, DOM smoke coverage, and GPU validation for a newly exposed Op.
10. Update `README.md` for user-visible behavior and `docs/architecture.md` for durable implementation rationale or non-obvious invariants.

When adding a tab to a new scope, also add the scope button, `data-scope` filtering, active accent styles, and deep-link synchronization. See the "Scopes" section in `docs/architecture.md`.

## Testing anatomy

- `tests/cases.json`: shared input corpus, written as CuTe layout strings so parsing is exercised.
- `tests/gen_reference.py`: runs the matching CuTeDSL calls and produces the oracle.
- `tests/reference.json`: committed generated output; never edit directly.
- `tests/run.js`: evaluates the JS port and compares layout strings, size/cosize, and every domain point.
- `tests/unit.js`: parser behavior and validations for which CuTeDSL cannot be the oracle.
- `tests/dom_smoke.js`: real templates and render functions under a DOM shim; catches silent handler, ID, default, preset, and deep-link failures.
- `tests/harness.js`: loads browser globals into Node and detects lexical-name collisions.
- `tests/gpu_check.py`: compiles and executes offered hardware Ops; optional because it requires a GPU.

Pointwise agreement is the decisive layout check. Two printed layouts can differ while representing the same map; a mismatched value is always a real defect.

## Scope boundaries

- `make_copy_atom`: what one instruction moves. No tensor layout, partition, coalescing, or bank analysis.
- `make_tiled_copy`: primitive `(layout_tv, Tiler_MN)` construction and tile coverage.
- `make_tiled_copy_tv`: derives those primitive arguments from thread/value layouts.
- `partition_sd`: one thread's TiledCopy partition over a tensor.
- `make_tiled_tma_atom`: one TMA instruction, descriptor/box, SMEM mapping, and returned GMEM coordinate tensor; no per-thread TV view.
- `tma_partition`: divides tensors into instruction-sized TMA chunks; it moves no data.
- `tv`: access-pattern questions such as GMEM coalescing and SMEM bank behavior.
- `make_mma_atom`: one warp-level instruction's A/B/C fragments.
- `make_tiled_mma`: replication of the atom across warps and permutation.
- `partition_abc`: one thread's TiledMMA partition over A/B/C tensors.
- `make_fragment_abc`: register layout produced by `MmaAtom.make_fragment_A/B/C` from an already partitioned warp MMA operand. It labels source offsets, not runtime values; Hopper descriptors and Blackwell TMEM are outside this tab.

If a proposed control cannot change a drawn cell, prefer a label or explanatory note. Keep constructor parameters, derived tiling, tensor partitioning, and access analysis in their respective layers instead of growing an all-purpose tab.
