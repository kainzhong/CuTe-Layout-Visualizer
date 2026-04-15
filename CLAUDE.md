# layout-visualizer

Interactive browser tool for visualizing CUTLASS CuTe layouts, TV layouts, and layout composition.

## File structure

```
index.html   Minimal HTML shell — just <header>, tab bar, and <script> tags
style.css    All CSS (no dynamic values, purely static)
cute.js      CuTe layout math, parser, and color utilities (pure logic, no DOM)
ui.js        SVG generation, DOM manipulation, tab management, composition logic
layout.js    Faithful port of python/pycute/layout.py + int_tuple.py (standalone, not loaded by index.html)
```

## Dependency graph

```
cute.js  (standalone, zero DOM references)
  ↓
ui.js    (depends on all cute.js globals; touches DOM)
  ↓
index.html loads cute.js first, then ui.js
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

### layout.js — faithful port of pycute Layout
Port of `python/pycute/int_tuple.py` and `python/pycute/layout.py`. Not loaded by `index.html` — it's a standalone library.
- **Int tuple helpers**: `is_int`, `is_tuple`, `flatten`, `product`, `prefix_product`, `crd2idx`, `idx2crd`, `shape_div`, `slice_`, `has_none`
- **Layout class**: `new Layout(shape, stride)`, `.call(...args)` (Python `__call__`), `.mode(i)` (Python `__getitem__`), `.rank()`, `.size()`, `.cosize()`
- **Layout functions**: `make_layout`, `coalesce`, `filter`, `composition`, `complement`, `right_inverse`, `left_inverse`, `logical_divide`, `logical_product`, `zipped_divide`, `tiled_divide`, `zipped_product`, `tiled_product`, `slice_and_offset`

### style.css — look here for layout/styling issues
All styling. Key sections: outer tab bar, inner tabs, controls panel, viz-box, composition 2x2 grid, mode-btn-group.

## Adding presets
Edit `generateTabContent()` in `ui.js`. Preset buttons call `setL()`, `setTV()`, or `setComp()`. For tiler presets, use `\\n` in the B string (the template literal produces `\n` in the HTML, which JS interprets as a newline).

## Display modes
Each SVG grid supports 3 cell-label modes via the value/index/coord button group:
- **value**: `layout(i)` — the mapped output value
- **index**: flat 1-D coordinate (`m + n*M`, column-major)
- **coord**: `(m, n)` — the 2D grid coordinates
