# layout-visualizer

Interactive browser tool for visualizing CUTLASS CuTe layouts, TV layouts, and layout composition.

## File structure

```
index.html   Minimal HTML shell — just <header>, tab bar, and <script> tags
style.css    All CSS (no dynamic values, purely static)
cute.js      CuTe layout math, parser, and color utilities (pure logic, no DOM)
ui.js        SVG generation, DOM manipulation, tab management, composition logic
layout.js    Port of python/pycute/layout.py + int_tuple.py, plus raked_product / make_layout_tv. Loaded by index.html.
```

## Dependency graph

```
cute.js   (standalone, zero DOM references)
layout.js (pycute port; uses its own globals, overlaps harmlessly with cute.js on `product`/`crd2idx`)
  ↓
ui.js     (depends on cute.js and layout.js globals; touches DOM)
  ↓
index.html loads cute.js, layout.js, ui.js in order
```

No module system — all functions are plain globals on `window`. The `onclick` attributes in HTML (both static in `index.html` and dynamic in `generateTabContent()`) reference global functions from `ui.js`.

## What lives where

### cute.js — look here for math/parser/color bugs
- **Layout arithmetic**: `product`, `productEach`, `unflatten`, `crd2idx`, `layoutAt`, `evalLayoutFlat`, `evalModeAt`, `autoStride`
- **Parser**: `parseValue`, `topLevelColon`, `parseLayout` — accepts `shape:stride` with `:` at top level only; rejects colons inside parens
- **Colors**: `BW_COLORS`, `TV_COLORS`, `HIGHLIGHT_COLORS`, `colorBW`, `colorTV`, `colorHighlight`, `textOnBG`

### ui.js — look here for rendering/UI bugs
- **SVG builders**: `buildLayoutSVG`, `buildTVSVG`, `buildHighlightedLayoutSVG`, `buildGridSVG`, `errSVG`
- **Tab management**: `generateTabContent` (the big HTML template), `addOuterTab`, `switchOuterTab`, `closeOuterTab`, `switchInnerTab`
- **Layout tab**: `renderLayout`, `setLayoutMode`, `setL`
- **TV tab**: `renderTV`, `buildLegend`, `setTV`, `setTileStride`
- **Composition tab**: `renderComposition`, `renderCompGrid`, `setCompMode`, `setComp`, `toggleComplement` — handles single-layout and tiler (by-mode) composition, plus optional `complement(R, A)` visualization with anchor-cell edges and faded shifted-tile copies
- **Complement tab**: `renderComplementFeature`, `renderCplGrid`, `setCplMode`, `setCpl`, `exportCpl` — standalone `complement(L, size(cotarget))` visualization; cotarget is a layout or shape that defines the codomain to cover
- **Utilities**: `showErr`, `updateOuterTabLabel`, `updateModeBtns`, `downloadSVG`

### layout.js — pycute port + make_layout_tv
Port of `python/pycute/int_tuple.py` and `python/pycute/layout.py`, plus a few helpers ported from `include/cute/layout.hpp` and `python/CuTeDSL/cutlass/cute/core.py`. Loaded by `index.html`.
- **Int tuple helpers**: `is_int`, `is_tuple`, `flatten`, `product`, `prefix_product`, `crd2idx`, `idx2crd`, `shape_div`, `slice_`, `has_none`
- **Layout class**: `new Layout(shape, stride)`, `.call(...args)` (Python `__call__`), `.mode(i)` (Python `__getitem__`), `.rank()`, `.size()`, `.cosize()`
- **Layout functions**: `make_layout`, `coalesce`, `filter`, `composition`, `complement`, `right_inverse`, `left_inverse`, `logical_divide`, `logical_product`, `zipped_divide`, `tiled_divide`, `zipped_product`, `tiled_product`, `slice_and_offset`
- **Extra**: `product_each`, `zip_tuple`, `zip_layouts`, `append_layout`, `raked_product`, `make_layout_tv`

### style.css — look here for layout/styling issues
All styling. Key sections: outer tab bar, inner tabs, controls panel, viz-box, composition 2x2 grid, mode-btn-group.

## Adding presets
Edit `generateTabContent()` in `ui.js`. Preset buttons call `setL()`, `setTV()`, or `setComp()`. For tiler presets, use `\\n` in the B string (the template literal produces `\n` in the HTML, which JS interprets as a newline).

## Display modes
Each SVG grid supports 3 cell-label modes via the value/index/coord button group:
- **value**: `layout(i)` — the mapped output value
- **index**: flat 1-D coordinate (`m + n*M`, column-major)
- **coord**: `(m, n)` — the 2D grid coordinates

## Shareable URLs (export / import)
The URL accepts `?key=<feature>[-<method>]-<input1>[-<input2>]` to deep-link into a visualization. Example:
```
?key=composition-(4,4):(4,1)-(2,2):(1,2)
?key=tv-1-(32,4):(1,32)-(8,16):(16,1)
?key=tv-2-(2,3):(3,1)-(2,2):(2,1)
```
- Parsing is in `parseKeyParam()` (driven by the `FEATURE_SPEC` table) and rendering is in `applyKeyParam()`.
- Export buttons live next to each Render button and call `exportURL(btnId, feature, ...inputs)`.

**Convention for new features: every feature MUST support both import (via `FEATURE_SPEC` + `applyKeyParam`) and export (an "Export URL" button + an `exportXxx` function).** When a feature has multiple input methods (like TV Layout does with direct TV+Tile vs thr/val), use a numeric method suffix: `tv-1`, `tv-2`, etc. List the allowed methods under `methods: [...]` in `FEATURE_SPEC`.

## Layout input convention (rank warnings)

The visualizer renders 2D grids, so any layout with outer rank > 2 (e.g. `(2,3,4):(1,2,6)`) renders value-correct but structurally-misleading. **Every tab that accepts layout-syntax input MUST surface this caveat** using the shared input component:

1. **HTML**: use `layoutInputField({ id, label, value, hint?, textarea?, rows?, placeholder? })` for every layout/shape input. Do NOT hand-roll `<input>`/`<textarea>` blocks for layout inputs.
2. **HTML (status)**: after the inputs, emit `statusDivs(prefix)` which renders the standard error + rank-warning pair (`${prefix}-error`, `${prefix}-warning`).
3. **JS (render)**: in the render function, call `updateRankWarning('${prefix}-warning', [[label1, val1], [label2, val2], ...])` with every layout input. It shows an amber note naming the offending fields when any has rank > 2.

Both helpers live in `ui.js` near `showErr`. Example:

```js
// HTML (inside generateTabContent):
${layoutInputField({ id: `${id}-mything-input`, label: 'My Layout', value: '(4,4):(1,4)' })}
${layoutInputField({ id: `${id}-mything-other`, label: 'Other', value: '' })}
${statusDivs(`${id}-mything`)}

// JS (inside renderMyThing):
const a = document.getElementById(`${tabId}-mything-input`).value;
const b = document.getElementById(`${tabId}-mything-other`).value;
updateRankWarning(`${tabId}-mything-warning`, [['My Layout', a], ['Other', b]]);
```

This is enforced by convention, not by the framework — if you add a new layout-input without these calls, the warning simply won't appear and the user won't know their rank-3 input is being flattened.
