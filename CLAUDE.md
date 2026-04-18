# layout-visualizer

Interactive browser tool for visualizing CUTLASS CuTe layouts, TV layouts, and layout operations.

## File structure

```
index.html     Minimal HTML shell — <header>, tab bar, <script> tags, and init
style.css      All CSS (no dynamic values, purely static)
cute.js        CuTe layout math, parser, and color utilities (pure logic, no DOM)
layout.js      Port of python/pycute/layout.py + int_tuple.py, plus raked_product / make_layout_tv
ui.js          Shared UI infrastructure: SVG builders, tab framework, URL import/export,
               and shared helpers (showErr, layoutInputField, statusDivs, ...)
tabs/
  layout.js    "Layout" tab — generateLayoutTabContent + renderLayout + state + helpers
  tv.js        "TV Layout" tab
  composition.js   "Composition & Complement" tab (with complement-toggle)
  complement.js    Standalone "Complement" tab
  divide.js        "Logical Divide" tab
  zipped.js        "Zipped Divide" tab
```

## Dependency graph

```
cute.js   (standalone, zero DOM references)
layout.js (pycute port; uses its own globals, overlaps harmlessly with cute.js on `product`/`crd2idx`)
  ↓
ui.js     (shared UI infra; depends on cute.js and layout.js globals; touches DOM)
  ↓
tabs/*.js (each tab depends on ui.js shared helpers and on cute.js / layout.js)
  ↓
<inline init script at end of index.html>  addOuterTab() + applyKeyParam(firstTabId)
```

Load order is enforced by the `<script>` tags at the bottom of `index.html`: cute.js, layout.js, ui.js, then every `tabs/*.js`, then the inline init script.

No module system — all functions are plain globals on `window`. The `onclick` attributes in the HTML templates reference those globals.

## What lives where

### cute.js — math / parser / colors
- **Layout arithmetic**: `product`, `productEach`, `unflatten`, `crd2idx`, `layoutAt`, `evalLayoutFlat`, `evalModeAt`, `autoStride`
- **Parser**: `parseValue`, `topLevelColon`, `parseLayout` — accepts `shape:stride` with `:` at top level only; rejects colons inside parens
- **Colors**: `BW_COLORS`, `TV_COLORS`, `HIGHLIGHT_COLORS`, `colorBW`, `colorTV`, `colorHighlight`, `textOnBG`

### layout.js — pycute port + CuTe helpers
Port of `python/pycute/int_tuple.py` and `python/pycute/layout.py`, plus a few helpers ported from `include/cute/layout.hpp` and `python/CuTeDSL/cutlass/cute/core.py`.
- **Int tuple helpers**: `is_int`, `is_tuple`, `flatten`, `product`, `prefix_product`, `crd2idx`, `idx2crd`, `shape_div`, `slice_`, `has_none`
- **Layout class**: `new Layout(shape, stride)`, `.call(...args)`, `.mode(i)`, `.rank()`, `.size()`, `.cosize()`
- **Layout functions**: `make_layout`, `coalesce`, `filter`, `composition`, `complement`, `right_inverse`, `left_inverse`, `logical_divide`, `logical_product`, `zipped_divide`, `tiled_divide`, `zipped_product`, `tiled_product`, `slice_and_offset`
- **Extra**: `product_each`, `zip_tuple`, `zip_layouts`, `append_layout`, `raked_product`, `make_layout_tv`, `isBijective`

### ui.js — shared UI infrastructure only
- **SVG builders**: `buildLayoutSVG`, `buildTVSVG`, `buildHighlightedLayoutSVG`, `buildGridSVG`, `errSVG`
- **SVG helpers**: `cellSize`, `svgFitStyle`, `cellTextSVG`, `buildCellLines`, `toModeSet`
- **Zoom**: `applyZoomState`, `toggleZoom`
- **Tab framework**: `generateTabContent` (orchestrator — calls each tab's `generateXTabContent`), `addOuterTab`, `switchOuterTab`, `closeOuterTab`, `switchInnerTab` (its `modeIndex` maps tab names to DOM order)
- **Shared helpers**: `showErr`, `showWarn`, `isHighRankLayout`, `collectHighRank`, `updateRankWarning`, `updateModeBtns`, `updateOuterTabLabel`, `downloadSVG`
- **Input components**: `layoutInputField`, `statusDivs` — ALWAYS use these for layout inputs (see "Layout input convention" below)
- **Layout-string utilities**: `stripTrivialTrailing`, `formatLayoutStr`
- **URL import/export infrastructure**: `FEATURE_SPEC`, `parseKeyParam`, `applyKeyParam` (dispatches on feature name to call the right `renderX`), `exportURL`

### tabs/*.js — one file per tab
Each tab file holds everything specific to that feature:
- The HTML template generator `generateXTabContent(id)`
- The mutable state object (e.g. `layoutState`, `tvState`, `compState`, `cplState`, `ldState`)
- The main `renderX(tabId)` function
- Per-grid helpers if any (e.g. `renderCompGrid`, `renderCplGrid`)
- Mode/toggle helpers (e.g. `setLayoutMode`, `toggleComplement`)
- Preset helper `setX` and export helper `exportX`

### style.css
All styling. Key sections: outer tab bar, inner tabs, controls panel, viz-box, composition 2x2 grid, mode-btn-group, error/warning message boxes.

## Display modes

Each SVG grid supports 3 cell-label modes via the value/index/coord button group:
- **value**: `layout(i)` — the mapped output value
- **index**: flat 1-D coordinate (`m + n*M`, column-major)
- **coord**: `(m, n)` — the 2D grid coordinates

## Shareable URLs (export / import)

The URL accepts `?key=<feature>[-<method>]-<input1>[-<input2>]` to deep-link into a visualization. Examples:
```
?key=layout-(10,10):(1,10)
?key=composition-(4,4):(4,1)-(2,2):(1,2)
?key=tv-1-(32,4):(1,32)-(8,16):(16,1)
?key=tv-2-(2,3):(3,1)-(2,2):(2,1)
?key=complement-(2,2):(1,2)-(4,4):(1,4)
?key=logical_divide-(12,32):(32,1)-3:1\n8:1
```
- Parsing is in `parseKeyParam()` (driven by `FEATURE_SPEC` in ui.js).
- Rendering is in `applyKeyParam()` (dispatches to the tab's render function).
- Export buttons live next to each Render button and call `exportURL(btnId, feature, ...inputs)`.

## Conventions for adding a NEW tab

Every new tab MUST:

1. **Live in its own file** under `tabs/yourtab.js`.
2. **Define `generateYourTabContent(id)`** that returns the HTML template (uses `${id}` interpolation). Start with `<div id="${id}-tab-yourtab" class="panel">...</div>`.
3. **Use `layoutInputField(...)` and `statusDivs(prefix)`** from ui.js for ALL layout inputs (never hand-roll `<input>` blocks for layout strings).
4. **Call `updateRankWarning(warnId, [[label, val], ...])`** in the render function for every layout input, so the rank-warning appears if the user enters a rank > 2 layout.
5. **Support URL import/export**:
   - Add an entry to `FEATURE_SPEC` in ui.js: `{ yourtab: { inputs: N } }` (or include `methods: [...]` if multiple input methods).
   - Add a `case 'yourtab':` branch to `applyKeyParam` that populates the inputs and calls your `renderX`.
   - Add an "Export URL" button in the tab HTML that calls your `exportX(tabId)`, which wraps `exportURL(btnId, 'yourtab', ...inputs)`.
6. **Add preset buttons** in the tab HTML that call a `setX(tabId, ...)` helper (which sets the inputs and calls `renderX`).

Then wire it into the shell:

7. **Register the tab button** in `generateTabContent` in ui.js: add `<div class="tab" onclick="switchInnerTab('${id}', 'yourtab')">Your Tab</div>` to the tab bar.
8. **Add to `modeIndex`** in `switchInnerTab` in ui.js: `{ ..., yourtab: <next index> }`.
9. **Call your generator** from `generateTabContent`: append `${generateYourTabContent(id)}`.
10. **Load the script** in `index.html`: `<script src="tabs/yourtab.js"></script>` (must appear after ui.js, before the inline init).
11. **Document it in this CLAUDE.md** under "What lives where → tabs/*.js".

## Layout input convention (rank warnings)

The visualizer renders 2D grids, so any layout with outer rank > 2 (e.g. `(2,3,4):(1,2,6)`) renders value-correct but structurally-misleading. **Every tab that accepts layout-syntax input MUST surface this caveat** using the shared input component:

1. **HTML**: use `layoutInputField({ id, label, value, hint?, textarea?, rows?, placeholder? })` for every layout/shape input. Do NOT hand-roll `<input>`/`<textarea>` blocks for layout inputs.
2. **HTML (status)**: after the inputs, emit `statusDivs(prefix)` which renders the standard error + rank-warning pair (`${prefix}-error`, `${prefix}-warning`).
3. **JS (render)**: in the render function, call `updateRankWarning('${prefix}-warning', [[label1, val1], [label2, val2], ...])` with every layout input. It shows an amber note naming the offending fields when any has rank > 2.

Both helpers live in `ui.js` near `showErr`. Example:

```js
// HTML (inside generateYourTabContent):
${layoutInputField({ id: `${id}-mything-input`, label: 'My Layout', value: '(4,4):(1,4)' })}
${layoutInputField({ id: `${id}-mything-other`, label: 'Other', value: '' })}
${statusDivs(`${id}-mything`)}

// JS (inside renderMyThing):
const a = document.getElementById(`${tabId}-mything-input`).value;
const b = document.getElementById(`${tabId}-mything-other`).value;
updateRankWarning(`${tabId}-mything-warning`, [['My Layout', a], ['Other', b]]);
```

This is enforced by convention, not by the framework — if you add a new layout-input without these calls, the warning simply won't appear and the user won't know their rank-3 input is being flattened.

## Adding presets

Edit the relevant `tabs/*.js` file and add preset buttons to its `generateXTabContent` output. Each preset button should call the tab's `setX(tabId, ...)` helper. For multi-line tiler presets, use `\\n` in the string (the template literal produces `\n` in the HTML, which JS then interprets as a newline).
