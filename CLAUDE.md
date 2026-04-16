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
- **Composition tab**: `renderComposition`, `renderCompGrid`, `setCompMode`, `setComp` — handles both single-layout and tiler (by-mode) composition
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
