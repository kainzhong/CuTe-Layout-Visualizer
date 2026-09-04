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
tests/         Differential tests against CuTeDSL — see "Testing" below. `npm test`.
tabs/
  layout.js    "Layout" tab — generateLayoutTabContent + renderLayout + state + helpers
  tv.js        "TV Layout" tab
  swizzle.js   "Swizzle" tab — BASICS scope; two stacked grids (base + Swizzle<B,M,S> applied), bottom cells show "a → b"
  composition.js   "Composition & Complement" tab (with complement-toggle)
  complement.js    Standalone "Complement" tab
  divide.js        "Logical Divide" tab
  zipped.js        "Zipped / Tiled / Flat Divide" tab (dropdown picks result form)
  local_tile.js    "local_tile" tab — `zipped_divide` + a slice. Greys out everything the coord
                   did NOT select, one colour per surviving tile. Takes the optional `proj`
                   (`(1, x, 1)`), which dices tiler and coord first. Prefix `lt`.
  product.js       "Logical Product" tab
  zipped_product.js "Zipped / Tiled / Flat Product" tab (single-layout tiler, dropdown picks result form)
  blocked_product.js "Blocked Product" tab (rank-preserving matrix tiling)
  raked_product.js  "Raked Product" tab (block-interleaved, scattered)
  make_copy_atom.js     "make_copy_atom" tab (COPY scope) — mirrors cute.make_copy_atom(op, dtype,
                        num_bits_per_copy). Section 1 picks the Op and shows its CONSTRUCTOR params;
                        section 2 takes make_copy_atom's own arguments. Two visualizations, chosen by
                        `MCA_OPS[key].kind`: the SIMT Ops draw one 1xN value grid, the three
                        `warp.LdMatrix*` Ops draw the src/dst TV layouts over the atom's tile.
                        Presets are filtered to the selected Op. Prefix `mca`.
  make_tiled_copy.js    "make_tiled_copy" tab (COPY scope) — the PRIMITIVE constructor: you supply
                        layout_tv and Tiler_MN directly. Also holds everything shared with the
                        make_tiled_copy_tv tab (mtcAtomSection/mtcReadAtom, mtcVizSection,
                        mtcBuildTileLookup, mtcCoverageCheck, mtcVectorizationCheck, mtcRenderTileViz,
                        mtcRenderThreadPanel, mtcReadHighlight, setMtcMode) — prefix `mtc`.
  make_tiled_copy_tv.js "make_tiled_copy_tv" tab (COPY scope) — the DERIVED constructor: thr_layout
                        + val_layout in, (layout_tv, Tiler_MN) out, then the same viz — prefix `mtv`.
  make_tiled_tma_atom.js "make_tiled_tma_atom" tab (COPY scope) — the TMA atom for
                        `cpasync.CopyBulkTensorTileG2SOp` (num_multicast fixed at 1). Ports the C++
                        pipeline in include/cute/atom/copy_traits_sm90_tma.hpp; draws the box over the
                        CTA tile (SRC) against SMEM (DST), the cuTensorMapEncodeTiled argument list,
                        and the returned coordinate tensor. Self-contained — shares nothing with the
                        mtc/mtv tabs but ui.js's copy panes. Prefix `tma`.
  make_mma_atom.js      "make_mma_atom" tab (MMA scope) — mirrors cute.make_mma_atom(op) for the three
                        DENSE `cute.nvgpu.warp` WarpMmaOp subclasses. Draws the A/B/C operand fragments as
                        three TV grids over three DIFFERENT tiles — (M,K), (N,K), (M,N). A lookup
                        TABLE, not a derivation: CUTLASS defines these as hand-written MMA_Traits
                        specializations. Prefix `mma`.
  tma_partition.js      "tma_partition" tab (COPY scope) — CuTe's "VectorCopy Partitioner"
                        (copy_traits_sm90_tma.hpp:1409). Splits mode 0 of every tensor into
                        (TMA, TMA_Iter) using the atom's NumValSrc for the chunk size and the SMEM
                        layout for the ORDER. Inputs mirror the real signature — atom, cta_coord,
                        cta_layout, smem_tensor, gmem_tensor — with the tensors given pre-grouping
                        and `group_modes(x, 0, 2)` applied here, as the caller does. Draws tAsA and
                        tAgA as two stacked layouts — NOT copy panes; it moves nothing. Prefix `tp`.
```

## Dependency graph

```
cute.js   (standalone, zero DOM references)
layout.js (pycute port; uses its own globals, overlaps harmlessly with cute.js on `product`/`crd2idx`)
          ⚠ layout.js loads AFTER cute.js, so on a name clash layout.js WINS, silently. A new
            global in cute.js must not collide with anything in layout.js — `crd2crd` already
            cost one debugging round (cute.js's basis evaluator is named `crd2basis` for this
            reason). Grep both files before adding a global.
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
- **Parser**: `parseValue`, `topLevelColon`, `topLevelCompose`, `parseLayout(str, opts)` — accepts `shape:stride` with `:` at top level only; rejects colons inside parens
- **Coordinate (basis) layouts**: `makeBasis`, `isBasis`, `hasBasisStride`, `basisRank`, `crd2basis`, `basisAt` — CuTe's scaled-basis strides (`k@i`), the thing TMA and identity/predication tensors need. Opt-in: `parseLayout(str, { basis: true })`; without the flag a `k@i` stride or an `<origin> o <layout>` prefix throws, so every other tab keeps its "strides are integers" assumption instead of silently producing NaN
- **Colors**: `BW_COLORS`, `TV_COLORS`, `HIGHLIGHT_COLORS`, `TV_DISABLED_FG`, `colorBW`, `colorTV`, `colorHighlight`, `textOnBG`

### layout.js — pycute port + CuTe helpers
Port of `python/pycute/int_tuple.py` and `python/pycute/layout.py`, plus a few helpers ported from `include/cute/layout.hpp` and `python/CuTeDSL/cutlass/cute/core.py`.
- **Int tuple helpers**: `is_int`, `is_tuple`, `flatten`, `product`, `prefix_product`, `crd2idx`, `idx2crd`, `shape_div`, `slice_`, `has_none`
- **Layout class**: `new Layout(shape, stride)`, `.call(...args)`, `.mode(i)`, `.rank()`, `.size()`, `.cosize()`
- **Layout functions**: `make_layout`, `coalesce`, `filter`, `composition`, `complement`, `right_inverse`, `left_inverse`, `logical_divide`, `logical_product`, `zipped_divide`, `tiled_divide`, `zipped_product`, `tiled_product`, `slice_and_offset`
- **Extra**: `product_each`, `zip_tuple`, `zip_layouts`, `append_layout`, `raked_product`, `make_layout_tv`, `isBijective`

### Coordinate (TMA / basis-strided) layouts in layout operations

A coordinate tensor's strides are scaled basis elements (`1@0`), so its codomain is a *coordinate*,
not a 1-D offset. CuTe supports exactly the operations that only ever **scale and add** such strides,
and rejects the ones that need them **ordered**. The split is not a matter of taste — it falls out of
two lines in `include/cute/layout.hpp`:

| op | basis-strided A? | why |
|---|---|---|
| `composition(A, B)` | **yes** | lhs strides are only multiplied (`composition_impl:1049`) |
| `logical_divide`, `zipped`/`tiled`/`flat_divide` | **yes** | `= composition(A, (tiler, complement(TILER)))` — complement is of the **tiler**, never of A (`:1562`) |
| `local_tile` | **yes** | `zipped_divide` + a slice |
| `coalesce`, `filter`, `slice_` | **yes** | compare strides for equality only |
| `complement(A)` | **no** | does `min(stride)` and `min_stride / result_stride` (`:1199-1202`) |
| `right_inverse` / `left_inverse` | **no** | sorts by stride |
| `logical_product` and every product variant | **no** | `= (A, composition(complement(A), B))` (`:1656`) |

`layout.js` implements this with `stride_mul` / `stride_eq` / `stride_is_zero` / `has_basis_stride`,
and `reject_basis(op, layout)` guards `complement` and `right_inverse` — so the product family fails
with the *reason* rather than producing NaN. `composition`'s only two contacts with A's strides are
`stride_mul` calls; everything else in it is integer arithmetic on B's side.

Verified by differential test: for `A = (32,64):(1@0,1@1)` against the integer `(32,64):(1,32)`
describing the same map, all 2048 points agree after `composition`, `logical_divide`,
`zipped_divide`, `tiled_divide`, `flat_divide` and `coalesce`.

**Every tab whose operation CuTe defines for basis strides accepts them**: Layout, Composition,
Logical Divide, Zipped / Tiled / Flat Divide, Local Tile. The rule for adding another is simply
whether the op appears in the "yes" half of the table above; if it does, pass `{ basis: true }` to
`parseLayout` and the rest already works.

The grid builders do the heavy lifting, so tabs mostly did not need per-cell changes. `basisNdimOf(stride)`
returns the output rank (0 for ordinary strides) and `cellValueAt(shape, stride, m, n, nd, origin)`
returns `{ label, key }` — a tuple label coloured by output axis 0, or an integer offset.
`buildLayoutSVG`, `buildTiledLayoutSVG` and `buildHighlightedLayoutSVG` all route through it;
`buildGridSVG` additionally accepts coordinate *cells*, since composing a coordinate layout produces
coordinates. `formatLayoutStr` prints `k@i` (it used to throw `x.map is not a function` on them —
every tab that prints a layout string goes through it).

Two things that are NOT free and had to be handled per tab:
- **Composition's tiler branch** sums per-mode contributions (`r0Vals[i] + r1Vals[j]`). For a
  coordinate layout that is a componentwise vector sum, so the tab carries `modeVal` / `addVals`
  helpers that switch on `aNd`.
- **Composition's complement toggle** computes `complement(R, size(A))`, which is undefined here. The
  button is `disabled` with the reason in its label rather than left to throw from three frames down.

Tilers stay ordinary in every case — `complement()` is taken *of the tiler*, so a basis-strided tiler
would be undefined even where a basis-strided A is fine.

### Coordinate layouts in the Layout tab

The Layout tab is the ONE tab that passes `{ basis: true }` to `parseLayout`. It accepts:

```
(4, 5):(1@0, 1@1)          identity — coord in, same coord out
(3, 4):(1@1, 1@0)          transpose — i@1 + j@0 = (j,i)
(8, 8):(4@0, 1@1)          scaled basis
(2, 2) o (4, 4):(1@0,1@1)  CuTe's coordinate-tensor printout, origin included
(3, 4):(1@0, 3@0)          every stride on axis 0 == an ordinary layout
```

`buildBasisLayoutSVG` (ui.js) draws these: each cell shows the COORDINATE it maps to, coloured by
output axis 0 so identity gives horizontal bands and a transpose gives vertical ones. The `origin`
from the `o` prefix is added to every cell — it is the iterator's value, the constant `slice_and_offset`
accumulates and a layout structurally cannot hold. Bijectivity / `right_inverse` are skipped for these,
since there is no 1-D offset to invert. Nested bases (`k@i@j`) are rejected with a clear message: they
produce hierarchical coordinates this 2-D grid cannot draw.

### ui.js — shared UI infrastructure only
- **SVG builders**: `buildLayoutSVG`, `buildTVSVG`, `buildHighlightedLayoutSVG`, `buildGridSVG`, `buildColoredLayoutSVG`, `errSVG`
- **`buildTVSVG`'s `opts` (9th arg)** — `{ dimTids, cellIndex }`. `dimTids` is a Set of thread ids
  drawn in `TV_DISABLED_FG` (translucent, so it stays legible over the live thread's hue): threads the
  layout maps but whose contribution the instruction **discards**. Several disabled threads aliasing
  onto a live one is *broadcast*, not a collision, so the red multi-claim stroke fires only when two
  **enabled** threads claim a cell — get this wrong and every `ldmatrix.x1` cell looks like an error.
  `cellIndex(m, n)` overrides the `value` overlay, which defaults to the col-major flat position;
  a tile that is row-major in its own codomain (the ldmatrix atom) must pass its own or the overlay
  prints a number that is not the layout's output. `cellTextSVG`'s `fg` accordingly accepts an array
  parallel to `lines`, which is what lets one cell mix live and disabled entries.
- **Drawing ACROSS cells**: `buildColoredLayoutSVG`'s `opts.overlay(geom)` returns raw SVG appended after the cells, with `geom = { cs, margin, W, H, M, N }`. A tile boundary is a *line*, not a property of the cells beside it — faking one with per-cell strokes gives a doubled, fuzzy edge. `tma_partition` uses it for its red tile outlines.
- **SVG helpers**: `cellSize`, `svgFitStyle`, `cellTextSVG`, `buildCellLines`, `toModeSet`
- **Zoom**: `applyZoomState`, `toggleZoom`
- **Copy SRC/DST panes**: `COPY_OP_MOVES`, `copyMoveField`, `syncCopyMoves`, `copyMove`, `setCopyMove`, `updateCopyPaneTitles`, `initCopyPanes`, `copyDirButtons`, `copyPanes`, `setCopyDir`, `copyDir`, `toggleCopyZoom` — the side-by-side view shared by all four Copy tabs. Both SVGs are always in the DOM; `data-dir` on `.copy-panes` decides visibility, so SRC/DST/BOTH is pure CSS and needs no re-render. In BOTH mode the panes are equal flex children, which halves each SVG's width while `width:100%;height:auto` preserves its ratio. Note `attachVizFullscreenButtons` iterates **every** `.viz-box` inside a `.comp-viz-item`, not just the first — the Copy tabs put two panes in one item, and taking the first left DST without a button.
- **One atom renderer, three tabs**: `simtAtomPaneHTML(side, layoutStr, elements, dtype)` draws the
  `(1, N):(0, 1)` value strip. `make_copy_atom` uses it for its panes; `make_tiled_copy` /
  `make_tiled_copy_tv` draw the SAME strip in a `comp-viz-item` **above** their tile viz, so the unit
  and the tiling of that unit are visible together. Every cell is `colorTV(0)` — the colour T0's first
  invocation gets in the tile grid below — so the two pictures line up rather than merely coexist.
  The atom panes use prefix `<p>-atom` and have no movement picker of their own; `mtcReadAtom` calls
  `updateCopyPaneTitles(tabId, '<p>-atom', p)` so they follow the tab's selection. `mtcClearPanes`
  clears both pairs, since a stale atom strip over a blank tile would read as "the atom is fine, the
  tiling failed".
- **One Op table, three pickers**: `SIMT_COPY_OPS` + `simtCopyOpOptions(sel)`. Every Op whose Atom is
  `ThrID = 1:0` with `ValLayoutSrc == ValLayoutDst == (1, N):(0, 1)` — `CopyUniversalOp`,
  `cpasync.CopyG2SOp`, and the four directional `CopyG2ROp` / `CopyR2GOp` / `CopyS2ROp` / `CopyR2SOp`.
  `make_copy_atom` spreads it into `MCA_OPS` (adding `kind: 'simt'`); `make_tiled_copy` /
  `make_tiled_copy_tv` alias it as `MTC_OPS`, and `mtcAtomSection` generates its `<option>`s from it.
  It lives in ui.js **because the duplication had already drifted** — the four directional Ops were
  added to `make_copy_atom`'s private table and silently did not appear in the other two tabs. Adding
  an Op here now offers it in all three with no template edit. The ldmatrix Ops stay out of `MTC_OPS`
  on purpose: they are warp-collective with divergent src/dst, so `mtcReadAtom`'s `AtomNumThr = 1`
  would be a lie: supporting them means teaching those tabs a second atom shape, not a dropdown entry.
- **Memory movement is constrained, not free.** `COPY_OP_MOVES` lists the legal `(src, dst)` pairs per Op key; section 0 of each Copy tab is a `<select>` built from it, and the pane titles are read-only labels driven by that select. `cpasync.CopyG2SOp` has exactly one entry (GMEM→SMEM) so its picker is `disabled`. `CopyUniversalOp` gets the six cross-space pairs over GMEM/SMEM/RMEM — **TMEM is deliberately excluded**, because it isn't thread-addressable (tcgen05's Ld/St src layout is stride-0 across all 32 lanes), so reaching it requires a tcgen05 Op. `tma_g2s` (the bulk-tensor load) has one entry too, for the same reason as `cpasync`. The four
directional SIMT Ops (`g2r`, `r2g`, `s2r`, `r2s`) likewise have exactly one each — the direction is in
the Op's name and is baked into its MLIR type (`!cute_nvgpu.atom.g2r<...>`), not chosen at
Atom-construction time. `ldmatrix` likewise has exactly one, SMEM→RMEM: the source operand is a `.shared::cta` address and the destination is the register file, both fixed by the instruction encoding — there is no GMEM form (that is cp.async / TMA) and no RMEM→SMEM form (that is `stmatrix`, a different Op). `syncCopyMoves(tabId, p, opKey)` rebuilds the options on every Op change and keeps the current selection when the new Op still permits it; `initCopyPanes(tabId)` runs from `addOuterTab` so the picker and titles are right before the first Render.
- **Collapsible viz**: `attachVizCollapsibles(root)` — injects a ▾/▸ chevron into each `.comp-viz-item` / `.visualization` header in `root`; clicking it folds just that viz's body (viz-box / legend / description). Called once from `addOuterTab` per new panel; the collapsed state lives on the item's class and survives in-place re-renders.
- **Render shortcut**: `TAB_RENDER_FN` (inner-tab name -> render function name), `renderActiveInnerTab`, `attachRenderHints`, `renderShortcutLabel`, `isMacPlatform` — a single document-level keydown listener maps Cmd/Ctrl+Enter to the visible tab's Render
- **Initial render**: `renderAllTabs(tabId, activeTab)` — called from `addOuterTab`, renders every tab once with its shipped defaults so switching to a tab shows a picture instead of an empty box. Relies on the "defaults and presets must render cleanly" rule. The active tab renders LAST so its `updateOuterTabLabel` wins, and each render is try/caught so one bad tab can't blank the rest.
- **Tab framework**: `generateTabContent` (orchestrator — calls each tab's `generateXTabContent`), `addOuterTab`, `switchOuterTab`, `closeOuterTab`, `switchInnerTab` (its `modeIndex` maps tab names to DOM order)
- **Shared helpers**: `showErr`, `showWarn`, `isHighRankLayout`, `collectHighRank`, `updateRankWarning`, `updateModeBtns`, `updateOuterTabLabel`, `downloadSVG`
- **Input components**: `layoutInputField`, `statusDivs` — ALWAYS use these for layout inputs (see "Layout input convention" below)
- **Element-type / swizzle helpers**: `DTYPE_BITS`, `dtypeOptions(selected)`, `applySwizzleOffset(x, sw)`, `parseSwizzleSpec(raw)` — shared by the TV, Copy_Atom, make_tiled_copy and TMA tabs. `parseSwizzleSpec` accepts `Sw<B,M,S>`, `Swizzle<B,M,S>` and a bare `B, M, S`; `applySwizzleOffset` swizzles an **element** index, which is NOT the unit CuTe prints a TMA swizzle in — see the TMA section below
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
?key=tv-1-(32,4):(1,32)-(8,16)
?key=tv-2-(2,3):(3,1)-(2,2):(2,1)
?key=complement-(2,2):(1,2)-(4,4):(1,4)
?key=logical_divide-(12,32):(32,1)-3:1\n8:1
?key=local_tile-(16, 16):(16, 1)-(4, 4)-(1, 2)
?key=local_tile-(32, 64):(64, 1)-(8, 16, 4)-(1, 2, _)-(1, X, 1)   # proj is the optional 4th
?key=zipped_product-(2,2):(1,2)-(2,2):(1,2)
?key=blocked_product-(2,2):(1,2)-(3,3):(1,3)
?key=raked_product-(2,2):(1,2)-(3,3):(1,3)
?key=make_copy_atom-universal-128-half_t
?key=make_copy_atom-ldmatrix-128-half_t-4-1   # ldmatrix adds num_matrices, transpose
?key=make_copy_atom-ldmatrix16x16x8b-128-int8_t-2-1-6   # the 8b Ops add unpack_bits
?key=make_tiled_copy-cpasync-128-half_t-((8,16),8):((128,1),16)-(16, 64)
?key=make_tiled_copy_tv-cpasync-128-half_t-(16,8):(8,1)-(1,8):(1,1)
?key=swizzle-(8, 8):(8, 1)-3, 0, 3
?key=make_mma_atom-f16bf16-half_t-float-16   # op, ab_dtype, acc_dtype, K
?key=make_mma_atom-tf32-na-na-8             # 'na' where the Op takes no dtype
?key=make_tiled_tma_atom-half_t-(256, 128):(128, 1)-3,4,3-(64, 64):(64, 1)-(64, 64)
?key=tma_partition-1024-float-3,4,3-(8, 32):(32, 1)-(4, 2)
```
- Parsing is in `parseKeyParam()` (driven by `FEATURE_SPEC` in ui.js).
- A feature may declare `optional: N` alongside `inputs`, accepting `inputs .. inputs+N` values. That
  is how `make_copy_atom` grew `num_matrices` / `transpose` for ldmatrix, then `unpack_bits` for the
  8-bit Ops, **without invalidating the shorter links already shared**; `exportMCA` emits each part of
  the tail only for the Ops that have it, so a `CopyUniversalOp` link stays 3 inputs and an
  `LdMatrix8x8x16bOp` link stays 5.
- **A `<select>` whose options are rebuilt per Op must be repopulated BEFORE the value is assigned.**
  `sel.value = '2'` on a select still holding the previous Op's options is a silent no-op, and the
  render then runs with a stale parameter. `setMCA` and `applyKeyParam` both call `mcaRenderOpParams`
  first for exactly this reason.
- Rendering is in `applyKeyParam()` (dispatches to the tab's render function).
- Export buttons live next to each Render button and call `exportURL(btnId, feature, ...inputs)`.

## Testing

**This repo is a PORT, so the tests are differential and CuTeDSL is the oracle.**
`layout.js` ports pycute / `include/cute/layout.hpp`; the Copy tabs port
`cutlass.cute`'s copy constructors. A port is correct exactly when it agrees with the
thing it was ported from — so nothing here is checked against a hand-written expectation
when a `cutlass.cute` call can produce it instead.

```
npm test                      # node only; reference.json is committed
node tests/run.js tma         # only sections matching a filter
npm run reference             # regenerate reference.json (needs nvidia-cutlass-dsl)
npm run reference:check       # fail if it drifted
```

Layout:

```
tests/cases.json        the shared corpus — inputs as CuTe layout STRINGS, so the
                        parser is exercised on the way in
                        Sections: layout_ops, basis_ops, make_layout_tv, make_tiled_copy,
                        copy_atom, ldmatrix_atom, mma_atom, tma_atom, tma_partition,
                        swizzle, local_tile
tests/gen_reference.py  runs cases.json through CuTeDSL -> tests/reference.json
tests/reference.json    committed golden output (never hand-edit)
tests/harness.js        loads the browser globals into node
tests/run.js            runs cases.json through the JS port and diffs
tests/unit.js           only the parts with no CuTeDSL analogue
tests/dom_smoke.js      the tabs' RENDER paths against a DOM shim — every preset,
                        renderAllTabs, and every ?key= round-trip. Part of `npm test`.
tests/gpu_check.py      OPTIONAL, needs a GPU. Compiles and EXECUTES each Op a
                        tab offers -- the gate for adding one. Not in `npm test`.
```

For every layout-valued result the suite compares the printed layout, its `size`/
`cosize`, **and its value at every point of the domain**. The pointwise check is the one
that matters: two different strings can describe the same map, but a divergent value is
always a real bug.

No GPU is needed to regenerate. Every layout in the corpus is static, so the generator
runs entirely at **trace time** inside one `@cute.jit` function. It tolerates the MLIR
compilation of that combined trace failing (tracing a hundred unrelated TMA descriptors
in one function does that) but only after verifying every case was evaluated first.

### The math is not the tab

`run.js` diffs the math against CuTeDSL. It cannot see the other half of a tab: a mistyped element id,
a preset that throws, a `<select>` whose options are rebuilt after its value is assigned, a pane that
never receives an SVG. Those stay invisible until someone opens the page — and for `applyKeyParam`,
until someone opens a shared **link**, which is worse because it throws during init outside any
try/catch.

`tests/dom_smoke.js` runs the real render functions against a DOM shim rich enough to execute them,
and fails on a throw, a populated error box, an element id no template declared, or an empty
visualization pane. It drives `initCopyPanes` + `renderAllTabs` (exactly what `addOuterTab` does),
every preset button found in the generated markup, and every `?key=` form through `parseKeyParam`.
It is Node-only and part of `npm test`.

It has already earned its place twice: it caught `make_mma_atom`'s K `<select>` reading `''` on the
first render (the options are built by `mmaSyncControls`, and the code was leaning on the browser
auto-selecting option 0 after an `innerHTML` swap), and it caught `updateCopyPaneTitles` blind-probing
for an atom pane pair that only two of the four Copy tabs have.

Two things it needs and a new tab must not break: controls are seeded from the markup's declared
defaults (`value="…"`, `<textarea>` bodies, `<option selected>`), and every preset button's handler
must be `setX('${id}', ...)` so the driver can find it.

**Handler arguments are obtained by EVALUATING the handler with its target stubbed, never by regex.**
The markup is two decodings away from the values a browser passes: character references resolve
first, then the JS parser resolves backslash escapes. A regex gets both layers wrong — it hands back
`&lt;` where the browser hands back `<`, and the two characters `\n` where the browser hands back a
newline. `handlerArgs` runs the onclick through `vm.runInContext` instead, which is what the browser
does, so both layers are right for free.

**Silence is the failure mode to design against here.** Two `local_tile` presets were dead for four
commits — a `\n` in the template literal put a real newline inside the onclick's single-quoted
string, so the browser raised a SyntaxError and the button did nothing — and dom_smoke did not
notice, because its matcher's `.` does not cross a newline and it *skipped* what it could not parse.
Two checks close that off, and neither should be removed: **every inline handler must parse as JS**
(405 of them, covering handlers the driver never invokes), and **every preset button must be
accounted for** as having actually run, rather than merely the ones a pattern liked.

### Only ship an Op a user can actually use

Three things can be true of a DSL Op, and only the third earns it a place in a tab:

1. it **constructs** — the dataclass accepts your parameters
2. it **traces** — `make_*_atom` returns layouts
3. it **lowers and runs** — MLIR → NVVM → PTX, and the numbers come out right

`cases.json` + `gen_reference.py` can only ever reach (2): the reference generator runs
entirely at trace time and explicitly *tolerates* MLIR compilation failing. So a tab can
be fully green and still be advertising an instruction nobody can execute.

`tests/gpu_check.py` closes that gap and is the gate for adding an Op. It is optional and
NOT part of `npm test` — the standard suite stays GPU-free on purpose — but run it before
adding an entry, and re-run it after a CuTeDSL upgrade. Verified on a GB200 (sm_100): all
ten `make_mma_atom` configurations compile and compute correctly.

**Do not trust the DSL's own support matrix.** `cute/nvgpu/warp/mma.py:47-77` carries a
commented table with a `DSL` column marking each instruction Y/N. It is **stale**: TF32 is
marked `N` with no DSL class, yet `MmaTF32Op` exists, traces, lowers and computes correctly
(rel err 6.8e-4 on a GB200 — exactly tf32's 10-bit mantissa, which is itself the proof the
hardware really did tf32 math rather than silently widening to fp32). Trust the hardware
over the docstring. The table is still useful as a *map of what does not exist yet* — F64
`m8n8k4` and `m16n8k{4,8,16}`, all the integer types (I8/U8/I4/U4/B1), TF32 sparse — none
of which have a DSL class at all, so none can be offered.

It also marks `MmaF16BF16SparseOp` as `[I]` = internal release only, which is a second,
independent reason it is excluded beyond the drawability argument below.

### When you add a tab, you add tests — this is not optional

A tab that ports a CuTe operation and has no case in `tests/cases.json` is
undifferentiated from a tab that gets it wrong. The checklist:

1. **Find the `cutlass.cute` entry point** your tab models. Almost everything is
   reachable at trace time with no GPU — `cute.composition`, `cute.make_layout_tv`,
   `cute.make_tiled_copy`, `cpasync.make_tiled_tma_atom`, `cpasync.tma_partition` all
   work from plain `@cute.jit`.
2. **Extract the DOM-free computation** out of `renderX` if it is not already separable.
   The render function reads the DOM and writes HTML; the math in the middle should be a
   pure function taking parsed inputs and returning the results. `tmaComputeAtom` and
   `tpComputePartition` are the pattern — `tpComputePartition` was carved out of
   `renderTmaPartition` for exactly this reason.
3. **Add cases to `tests/cases.json`** — its own section if the tab needs one, with a
   `_comment_<section>` saying what the oracle is and why.
4. **Add the section to `SECTIONS` in `tests/gen_reference.py`** and a matching block in
   `tests/run.js`.
5. **Run `npm run reference` and commit `reference.json` alongside the code.**
6. **Only what CuTeDSL cannot express goes in `tests/unit.js`** — the string parsers, and
   the validation CuTe skips (see below). Everything else belongs in `cases.json`, because
   a case with an oracle stays true when CUTLASS changes and a hand-written expectation
   does not.

Also add the tab's presets as cases where they are cheap: a preset that renders is not
the same as a preset that renders the *right* thing.

### What CuTeDSL cannot be the oracle for

Two categories, both in `tests/unit.js`:

- **Input handling.** CuTe's inputs are C++ types; this tool's are text. `parseLayout`,
  `mtcParseTiler`, `parseSwizzleSpec`, `tmaParseSmemField`, and `tmaSwizzleInfo`'s
  byte→element conversion have nothing to diff against.
- **Validation CuTe skips.** `mtcCoverageCheck`, `mtcVectorizationCheck`,
  `mtcRequireCompact`, the TMA host-`assert()` issues, and `tpComputePartition`'s
  permutation / divisibility guards all exist *because* CuTe compiles and runs those
  configurations while being silently wrong (see "Validation that CuTe itself skips").
  By construction the DSL cannot be the oracle; the expectations encode the C++
  preconditions instead. `mcaLdmatrixAtom`'s two refusals belong here for the same
  reason: `num_matrices ∉ {1,2,4}` the DSL *does* reject, but a `.trans` on a
  non-16-bit element and an element wider than 64 bits it accepts and silently
  recasts.

One thing `run.js` does check against the DSL here: for every `tma_atom` case CuTeDSL
accepted, the tab must report **no** error-level issue. A validation that fires on valid
input is as broken as one that misses invalid input.

`mma_atom` is EXHAUSTIVE over its parameter domains (16 cases) for a reason beyond
coverage: it is what pins the tab's load-bearing claim that the three layouts depend
only on `(op family, K)`. `run.js` groups the cases by that key and requires every
member of a group to produce identical layout strings, so if a future CUTLASS made
`acc_dtype` matter, the suite would say so rather than the tab quietly lying. It also
checks each operand is a bijection onto the tile the tab draws it over — `size` and
`cosize` both equal `M*K` / `N*K` / `M*N` — since a grid of the wrong shape is exactly
the failure a layout string alone would not reveal.

`ldmatrix_atom` does the same trick for the facts the tab draws but the DSL does not name:
the tile the panes are placed into must equal `cosize` on **both** sides (a tile bigger or
smaller would draw empty cells or silently wrap); `size/cosize` on the src side must equal
`32/liveLanes`, i.e. the greyed-out lanes are exactly the broadcast factor; and
`liveLanes === tile rows`, which is the `matrixBytes/16` identity above. All checked against
CuTeDSL's own `size` and `cosize` rather than against a guess. The section also pins each
Op's parameter domains, since a tab offering an out-of-domain `num_matrices` would produce
nothing but error boxes.

The `unpack_bits` cases exist for a claim the tab makes in prose: that the parameter selects
the `LdsmSzPattern` and changes no layout. Those cases pass the argument to CuTeDSL and diff
the result against a derivation that never received it, so the claim is enforced rather than
asserted.

### The harness, and the load-order hazard

`tests/harness.js` concatenates every source file into one script and copies the
top-level `const`/`let`/`class` names onto `globalThis` — those never reach the global
object the way function declarations do, so they cannot be read out of a per-file vm
context. It also **fails loudly on a duplicate top-level lexical name across files**,
which in the browser would silently shadow. That is the same hazard the dependency-graph
warning above describes, now enforced rather than remembered.

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

7. **Register the tab button** in `generateTabContent` in ui.js with a `data-scope` — e.g. `<div class="tab" data-scope="operations" onclick="switchInnerTab('${id}', 'yourtab')">Your Tab</div>`. Pick the scope the tab belongs to (see "Scopes" below).
8. **Add to `modeIndex`** in `switchInnerTab` in ui.js: `{ ..., yourtab: <next index> }`. The index must match the tab's order inside `.tab-bar`.
8b. **Add to `TAB_RENDER_FN`** in ui.js: `{ ..., yourtab: 'renderYourTab' }`, and give the tab's `<div class="tab">` a `data-tab="yourtab"` attribute. This is what Cmd/Ctrl+Enter uses to find the render function of whichever tab is visible. Name the tab's main button exactly `Render` — `attachRenderHints` keys off that text to inject the keyboard hint, and skips variants like "Render Inverse" that are toggles rather than the tab's main render.
9. **Call your generator** from `generateTabContent`: append `${generateYourTabContent(id)}`.
10. **Load the script** in `index.html`: `<script src="tabs/yourtab.js"></script>` (must appear after ui.js, before the inline init).
11. **Document it in this CLAUDE.md** under "What lives where → tabs/*.js".
12. **Add differential tests** — see "Testing" above. A new tab without cases in
    `tests/cases.json` is not finished.

## Scopes (tab groups)

The tab bar is grouped into **scopes** so it doesn't become a wall of buttons. Each scope is a named bucket; only one scope's tabs are visible at a time. Current scopes:

- `basics` — Layout, TV Layout, Swizzle. Accent color: blue (`#3b82f6`).
- `operations` — the CuTe layout-algebra tabs (composition, complement, divide/product variants, `local_tile`). Accent color: purple (`#a855f7`).

  **`local_tile` uses CuTe's tiler convention, not the Divide tabs'.** A single line without a top-level
  colon is a *Shape* tiler — `(8, 32)` means one tiler mode per tensor mode, exactly as
  `local_tile(data, Shape<_32,_64>{}, coord)` does. Reading it as a single layout tiler (what the
  Zipped / Divide tabs do) collapses the "rest" mode to rank 1 and makes a 2-D coord impossible; that
  bug ate every preset on the first pass. A line *with* a colon is still a single layout tiler, and
  several lines are still one per mode. It also uses `parseValue`, not `parseLayout`, because the
  latter pads a bare `8` out to `(8,1)` and would silently invent a second tiler mode.

  **`proj` is pure pre-processing, and the tab implements it as exactly that.**
  `local_tile(t, tiler, coord, proj)` is *defined* as
  `local_tile(t, dice(proj, tiler), dice(proj, coord))` (tensor_impl.hpp:1049), so `ltDiceTiler`
  runs before anything else and everything downstream — `zipped_divide`, the slice, both grids,
  the rank warning — sees only the projected tiler. It is what lets one
  `cta_tiler = (BLK_M, BLK_N, BLK_K)` and one `cta_coord = (m, n, _)` tile all three GEMM operands.
  `proj` and `coord` decide independent things per dimension: `proj` decides whether the tiler mode
  exists at all (a number **selects** it, `x` / `_` / `None` **masks it out**), and `coord` matters
  only for the survivors, where an index picks a tile and `_` keeps the mode as the result's k-loop
  mode. A masked dimension's coord component is thrown away **unread** — that is the point, since
  `cta_coord`'s `m` and `n` are concrete and gA must discard `n` while gB discards `m`. Verified
  against CuTeDSL: `(1, 2, _)`, `(1, 999, _)` and `(1, _, _)` under `(1, x, 1)` all give the same
  layout and origin.
  The rank warning deliberately reports the **projected** tiler: a rank-3 tiler is the normal case
  with a proj, so warning about the pre-dice form would be noise.

  **Both fields normalize on Render** (`projNorm` / `coordNorm`, computed in `ltComputeLocalTile` so
  the strings are unit-testable, written back by `renderLocalTile`). Every selected dimension comes
  back `1` and every masked one `_`, because `dice` never reads the value — `(2, X, 7)` and
  `(1, _, 1)` are the *same* projection and one spelling stops that looking like a distinction the
  tool tracks. The coord is rewritten **in the masked slots only**: a number there is discarded
  unread, so leaving it on screen would imply it still selects something; surviving slots keep
  exactly what was typed. Without a proj nothing is masked and neither field is touched — in
  particular a blank coord stays blank rather than expanding to `(_, _)`. Normalization runs only on
  **success**, so a rejected input stays on screen to be fixed, and only from the Render path —
  neither field has an `oninput` handler, so it cannot fight typing. `dom_smoke` asserts the
  write-back lands in the right box, which the pure test cannot see.

  **The field takes VALUES, and CUTLASS C++'s `Step<_1, X, _1>` is deliberately refused.** `_1` is
  `Int<1>` — a template parameter, not something anyone types into a text box — so the grammar is
  `(1, x, 1)` or the DSL's `(1, None, 1)`. `ltParseProj` still *recognizes* both C++ spellings well
  enough to name them in the error, because `Step<_1, X, _1>` is exactly what gets pasted out of a
  GEMM kernel and "Cannot read proj component" would be a poor answer to it. `parse_lt_coord` in
  `tests/gen_reference.py` is the Python twin of this grammar and must stay in step; the refusals are
  pinned in `tests/unit.js`, since CuTeDSL takes C++ types and cannot be the oracle for text.

  **A rest mode can itself be a tuple**, which a single layout tiler routinely produces
  (`(2,(2,16))`). A scalar coord component into such a mode is a *flat* index into it — CuTe's
  `crd2idx` does `idx2crd` on it first — so the "which tiles survived" scan decomposes the rest
  index col-major keeping each mode flat, rather than comparing against `idx2crd`'s nested
  coordinate. Comparing structurally selected **zero** tiles and left `kept[0]` undefined.
  CuTeDSL is not the oracle for this one: its MLIR `crd2idx` cannot infer a result type for that
  coord/layout pair at all, so the case lives in `tests/unit.js`.
- `copy` — the copy-construction pipeline: `make_copy_atom` (one instruction), then `make_tiled_copy` / `make_tiled_copy_tv` (replicate it over a tile), plus `make_tiled_tma_atom` and `tma_partition` (the TMA path, which bypasses threads entirely). Accent color: emerald (`#10b981`).
- `mma` — the MMA side: `make_mma_atom`. Accent color: amber (`#f59e0b`). The natural next tabs are `make_tiled_mma` (the analogue of `make_tiled_copy`) and `make_tiled_copy_A/B`, which is where the copy and MMA scopes finally meet.

### How scopes are wired

- **Markup**: `generateTabContent` emits a `<div class="tab-nav" data-scope="...">` wrapping the scope selector (`.tab-scope-bar` with `.tab-scope-btn` pills) and the tab bar (`.tab-bar` with one `<div class="tab" data-scope="...">` per tab). The `data-scope` attribute on each tab assigns it to a group.
- **CSS**: `.tab-bar[data-scope="X"] .tab[data-scope="Y"] { display: none; }` hides tabs that don't belong to the current scope. Accent colors also key off `data-scope` — e.g. `.tab[data-scope="operations"].active` is purple.
- **JS**: `switchTabGroup(tabId, scope)` flips the attribute on `.tab-nav` + `.tab-bar`, toggles the `.active` class on the scope buttons, and auto-activates the first tab in the new scope if the previously-active tab is now hidden. `switchInnerTab` also syncs the scope attribute + scope-button active state so URL deep-links (and programmatic tab switches) keep the selector in sync with the visible tab.

### Adding a new scope (e.g. `mma`, `copy`)

1. **Add a scope button** to `.tab-scope-bar` in `generateTabContent` with a unique `data-scope` value and a count of tabs it will hold:
   ```html
   <div class="tab-scope-btn" data-scope="mma" onclick="switchTabGroup('${id}', 'mma')">
     <span class="tab-scope-icon">⬡</span>MMA
     <span class="tab-scope-count">N</span>
   </div>
   ```
2. **Tag every new tab** with your scope — `data-scope="mma"` on the `.tab` div.
3. **Add accent-color CSS** in `style.css` for the new scope:
   - Left-stripe color: `.tab-nav[data-scope="mma"]::before { background: <color>; }`
   - Active scope-button: `.tab-scope-btn[data-scope="mma"].active { ... }`
   - Active tab within that scope: `.tab[data-scope="mma"].active { ... }`
   - Optional: active-count badge tint inside the active scope button.
4. **Filter CSS** — add one line in `style.css` for the new scope:
   ```css
   .tab-bar[data-scope="mma"] .tab:not([data-scope="mma"]) { display: none; }
   ```
   This hides any tab that doesn't belong to `mma` while the `mma` scope is active. No other CSS edits needed.

Adding a tab to an existing scope is just step 2 with the existing scope name — no CSS changes needed.

## Where copy-related concerns live

The copy pipeline is deliberately split across three tabs, one concern each. Keep it that way when
adding new copy atoms.

| Concern | Tab | Scope | Why there |
|---|---|---|---|
| What one instruction moves | `make_copy_atom` | `copy` | A property of the Op + dtype alone |
| How the atom is replicated over a tile, given (layout_tv, Tiler_MN) | `make_tiled_copy` | `copy` | The primitive `_make_tiled_copy` takes exactly these two |
| Deriving (layout_tv, Tiler_MN) from a thr/val pair | `make_tiled_copy_tv` | `copy` | One of several ways to compute the primitive's arguments |
| What one TMA instruction moves, and the descriptor behind it | `make_tiled_tma_atom` | `copy` | A property of (tensor, smem layout, tiler) — no threads involved |
| How a tile is split into instruction-sized chunks | `tma_partition` | `copy` | Needs only the atom's element count and the SMEM order |
| Whether the access pattern is coalesced / bank-conflict-free | `tv` | `basics` | A property of (TV layout, **data layout**) — no atom involved |

### The six SIMT Ops produce one identical Atom

`CopyUniversalOp`, `cpasync.CopyG2SOp` and the four directional Ops (`CopyG2ROp`, `CopyR2GOp`,
`CopyS2ROp`, `CopyR2SOp`) all give `ThrID = 1:0` and `ValLayoutSrc == ValLayoutDst == (1, N):(0, 1)`.
They share `kind: 'simt'` and the same render path, on purpose — the picture is of the data movement,
and these Ops move data identically.

**What the directional Ops add is memory attributes, and those change no cell.** They are
`make_copy_atom` kwargs, not Op fields: `memory_order`, `memory_scope`, `l1c_evict_priority`,
`shared_space` on all of them, plus `l2_prefetch_size` / `load_cache_mode` / `invariant` on the load
side and `store_cache_mode` on the store side. (Those asymmetries are the tell that this is not one
parameter set gated by direction — prefetch and `ld.global.nc` have no store analogue, and SMEM has no
L1/L2 policy, so the two shared-memory Ops get only the sync knobs.) Measured across all 32 settings of
`CopyG2ROp`'s: **1** distinct `(ThrID, src, dst)` triple, **26** distinct MLIR atom types. So
`MCA_OPS[key].attrs` NAMES them in the params box rather than rendering controls — a control that
cannot change a cell is noise, the same call as `unpack_bits` not being a parameter of
`mcaLdmatrixAtom`.

Two things the attributes *do* change, worth not forgetting: the emitted PTX, and the fact that
`CopyG2RTrait` / `CopyR2GTrait` override `unpack()` to accept a **runtime** `cache_policy` (an `Int64`
from `createpolicy`) which `CopyUniversalTrait` cannot.

**Three cross-parameter rules the DSL enforces**, found by writing the `attrs` test cases and watching
them fail — worth knowing before adding more:

- `invariant` requires every other memory feature at its default
- `load_cache_mode` must be `ALWAYS` unless `memory_order` is `WEAK`
- `store_cache_mode` must be `WRITE_BACK` unless `memory_order` is `WEAK`

A one-attribute-at-a-time sweep never hits these; they only bite on combinations.

`CopyS2ROp` shares SMEM→RMEM with the whole ldmatrix family but is the single-thread load, so its
layouts are completely different — same `COPY_OP_MOVES` entry, opposite ends of the tab.

`make_copy_atom` keeps the DSL's two-step split visible: an Op is constructed with its own fields
first, then `make_copy_atom(op, dtype, num_bits_per_copy)` turns it into an Atom. `MCA_OPS[key]`
carries `params` (the constructor params) and `kind` (which derivation and which viz). The two SIMT
Ops have no params, which is why section 1 states that instead of rendering controls;
`warp.LdMatrix8x8x16bOp` has two and they appear there. `tcgen05.Ld*` (repeat, pack) slots into the
same area.

All four Copy tabs render into `-<prefix>-src-svg` / `-<prefix>-dst-svg` rather than a single host.
For `CopyUniversalOp` and `cpasync.CopyG2SOp` the two panes are identical, because
`ValLayoutSrc == ValLayoutDst`; they diverge for shuffling atoms such as `ldmatrix`, where the
src layout is an *addressing* pattern (which lane points where) and the dst layout is the register
outcome. Keep the two render paths separate even while they agree.

### make_copy_atom's LdMatrix family

Three Ops, all warp-collective SMEM→RMEM loads with `ThrID = 32:1` and a genuinely divergent
src/dst pair. `MCA_LDSM_SPECS[key]` carries the geometry and the parameter domains;
`mcaLdmatrixAtom(opKey, elemBits, numMatrices, transpose)` is the one DOM-free derivation for all
three. `MCA_OPS` entries just name a spec via `ldsm`.

| Op | matrix | unitBits | bytes/matrix | num_matrices | transpose | unpack_bits |
|---|---|---|---|---|---|---|
| `LdMatrix8x8x16bOp` | 8x8 | 16 | 128 | 1, 2, 4 | optional | **rejected** |
| `LdMatrix16x8x8bOp` | 16x8 | 8 | 128 | 2, 4 | **required** | None, 4, 6 |
| `LdMatrix16x16x8bOp` | 16x16 | 8 | 256 | 1, 2 | **required** | None, 4, 6 |

The domains are enforced by `__post_init__`, so `mcaSyncLdsmControls` rebuilds the `num_matrices`
options per Op, pins-and-disables `transpose` where it is mandatory, and hides `unpack_bits` where it
is rejected — an out-of-domain control could only ever produce an error box.

**`matrixBytes/16` is the load-bearing constant.** A lane addresses 16 B, so it is simultaneously the
rows per matrix and the lanes one matrix consumes, which is why `liveLanes === tile rows` always
(asserted in the tests). 8x8 of 16-bit and 16x8 of 8-bit are both 128 B and therefore both give
`8*k` rows; 16x16 of 8-bit is 256 B and gives `16*k`.

**`LdMatrix16x8x8bOp` is the one that is NOT a re-parameterization.** It has no direct PTX form: the
DSL lowers it to `.m16n16` plus address and value permutations chosen to match
`stmatrix.m16n8.trans`. That permutation is baked into `ValLayoutSrc`, which is a **rank-4 thread
mode** `((2,2,4,2), R):((R, 8R, 2R, 0 or 16R), 1)` rather than the `(live, broadcast)` pair every
other Op has. Concretely the lane→row map is permuted — `T0→r0 T1→r1 T2→r8 T3→r9 T4→r2 T5→r3
T6→r10 …` — which the src pane now shows directly, and which is exactly why this Op earns a
visualization rather than a table row.

**`unpack_bits` changes NO layout.** It selects the `LdsmSzPattern` the DSL hands to MLIR (`u8` /
`u4x16p64to8` / `u6x16p32to8`), i.e. the PTX qualifier and the packed source container — 16x4b with
64b padding, or 16x6b with 32b padding, widened into 8-bit registers. Verified identical across all
**156** accepted combinations of the two 8-bit Ops, which is why it is not a parameter of
`mcaLdmatrixAtom` at all; the render path reads it only to label the instruction and to say plainly
that the picture is unchanged. Pinned by `unpack_bits` cases in `tests/cases.json`.

**`num_matrices` decides how many lanes' addresses the hardware CONSUMES, not how much a lane
addresses.** A lane always covers one 128-bit row; it consumes `(matrixBytes/16) * num_matrices` of
the 32 lanes.
The lanes it does not consume still execute the instruction and still hand over an address operand —
`.sync.aligned` requires the whole warp converged, and there is no membermask on this instruction —
so `ValLayoutSrc` has to be a total function over the warp. CuTe writes that as a **stride-0** second
thread mode, aliasing the ignored lanes onto live ones: in-bounds, branchless, and it makes
`size` exceed `cosize` by exactly the broadcast factor. Those lanes are the ones drawn in
`TV_DISABLED_FG`.

**Src and dst are not paired by `(t, v)`.** At `.x1` a lane has 8 source slots against 2 destination
slots, so there is no slot-to-slot correspondence to draw; the two layouts relate only through the
codomain, covering the same cells and nothing more. Do not add a connector between the panes. Src is
*which lane points at which 16 B row*; dst is *which lane ends up holding which element*; the data
crosses lanes inside the instruction, so a lane never receives the row it addressed.

**Everything the derivation returns is the instruction's fixed geometry divided through by the
element width** — which is what `copy_internal_type` does in the DSL, and why `Float32` gives an 8x4
tile where `Float16` gives 8x8 for the same instruction. The scale is `unitBits / elemBits`, so the
same three structural branches appear for every Op, just around a different unit:

- element **is** the transpose unit — the base case (`half_t` for the b16 Op, `int8_t` for the 8b ones)
- element **smaller** — a leading value mode of `unitBits/e` appears, because the transpose moves
  whole units so the elements packed inside one stay adjacent
- element **larger** — an element spans several units, which the transpose separates, so the
  transposed thread mode drops **below 32 lanes**

The tab reports that last case rather than hiding it: CuTeDSL recasts happily and returns a layout,
but the transpose is `unitBits` wide whatever `copy_internal_type` says. Elements wider than 64 bits
are refused outright — a 128-bit row would hold fewer than 2 of them and the matrix degenerates.
Note the 8-bit Ops can never hit the "element smaller" branch, since 8 bits is the narrowest entry
in `DTYPE_BITS`.

**`num_bits_per_copy` is ignored by this Op** (`_make_trait` never reads it). The field is *disabled*
rather than hidden, so the absence is legible rather than mysterious.

**Still missing from the family**: `LdMatrix8x16x8bOp` (the third 8-bit Op — `transpose` is
*rejected* on it, `num_matrices` is `{1,2,4}`, and without unpacking the DSL notes it is equivalent
to `8x8x16b`), and the whole `StMatrix*` set, which is the same shape of work in the RMEM→SMEM
direction and would need `COPY_OP_MOVES` entries pointing the other way.

**The atom's tile is ROW-major in its own codomain** — flat = `m*N + n`,
`(matrixBytes/16)*num_matrices` rows of `128/elemBits` elements, one lane-addressed row per grid row.
That is why the viz passes `cellIndex` to `buildTVSVG`; the TV tab's tile is col-major and the
default would print the wrong number.

**Presets are data, not markup** (`MCA_PRESETS` + `mcaPresetButtons`), because they are FILTERED to
the selected Op by `mcaSyncPresets`. Twenty buttons for five unrelated instructions is a wall, and
only one Op's presets can ever apply. Adding an Op means adding rows, not editing the template.

Concretely: a copy-atom tab must NOT take a tensor layout, must NOT compute a partition, and must NOT
offer coalescing or bank-conflict checks. Those belong to the TV Layout tab, which already accepts a
data layout and renders both checks over the same grid.

Coalescing is measured per *instruction*, so the GMEM check needs a vector width — but it is
**derived, never asked for**. `tvVectorFeasibility(...).maxWidth(numV, cap)` returns the largest
divisor of `numV` for which every thread's W consecutive vids sit at W consecutive addresses (CuTe's
`upcast<W>` precondition). The value layout and the data layout already determine this between them.

**Vector width and `tensor_dtype` are one quantity, not two: bytes moved per thread per
instruction.** CuTe says so directly — a 128-bit copy of `half_t` recasts the tensor to a 128-bit
type, and the per-thread layout becomes one element wide. "4 adjacent `int8_t`" and "1 `uint32_t`"
are the same instruction, and the tool agrees (both report 1 issue / 1 sector / coalesced). The
consequence is that the hardware ceiling is a *byte* ceiling: `TV_MAX_ACCESS_BITS = 128`, the widest
single load on any current architecture. The derived width is capped there, so a thread owning 16
adjacent `half_t` gets 2 instructions rather than one fictional 256-bit access. Do not remove the
cap — without it the summary happily reported "a 1024-bit access".

Each check owns its own `tensor_dtype`, alongside its own data layout — they are independent
analyses of independent tensors. SMEM needs it because a bank is 4 bytes wide, so element size
decides which bank a cell lands in; GMEM needs it because sectors are counted in bytes. Do not
re-merge them into one shared selector.

The two checks are separate `<details>` sections in the TV tab, each with its own data-layout input
(`-tv-gmem-input` / `-tv-smem-input`), its own `tensor_dtype` (`-tv-gmem-dtype` / `-tv-smem-dtype`), and its own toggle button. They are
mutually exclusive at render time — turning one on clears the other. Render paths are
`renderTVCoalescedSVG` / `renderTVBankSVG`, sharing `tvResolveData(tabId, which, M, N, checkName)` and
`tvBuildGrid`; `setTVDataMajor(tabId, which, major)` fills whichever box's Row/Col-major button was clicked.

### make_tiled_copy vs make_tiled_copy_tv

`_make_tiled_copy(atom, layout_tv, tiler_mn)` (`cutlass/cute/atom.py:1011`) is the *only* function
that builds a `TiledCopy`. Every public constructor computes those two arguments some other way and
calls it. The two tabs mirror that split: `make_tiled_copy` takes them literally,
`make_tiled_copy_tv` derives them via `raked_product` + `right_inverse`. Keep future constructors
(`make_cotiled_copy`, `make_tiled_copy_A/B/C/S/D`) as further `copy` tabs feeding the same viz.

**Tiler_MN is NOT required to match layout_tv's shape.** They describe different things —
`layout_tv`'s shape is `(num_threads, num_values)`, `Tiler_MN` is the `(M, N)` tile; they are linked
only through the codomain, since `layout_tv(tid, vid)` is a flat col-major index into the tile. CuTe's
only static asserts (`copy_atom.hpp:205-206`) relate `layout_tv` to the *atom*, never to the tiler.
Two ways the sizes legitimately differ:

- **Broadcast** — a stride-0 mode in `layout_tv` makes several threads read the same element, so
  `size(layout_tv)` is a *multiple* of `size(Tiler_MN)`. `make_tiled_copy_A/B(atom, tiled_mma)` does
  exactly this. Measured on an SM100: `MmaF16BF16Op` with a `(2,2,1)` atom layout gives
  `tv_layout_A_tiled` of size 1024 against a 32x16 = 512 tile — every cell claimed exactly twice.
  `mtcHasBroadcast` detects it via a zero in `flatten(layout_tv.stride)`.
- **Tiler_MN is a Tiler, not a shape** — its modes are *independent*, each an extent or a whole
  layout (`(8:1, 16:2)` selects a strided sub-tile), and the rank need not be 2. This means
  `parseLayout` is the WRONG parser for a tiler: it demands a single top-level `shape:stride` split
  and explicitly rejects a colon inside parens, so it would refuse `(8:1, 16:2)` — which is exactly
  how CuTeDSL prints a tiler (`tc.tiler_mn`), i.e. you could not paste one back. Use `mtcParseTiler`,
  which splits top-level commas and reads each mode separately. Tiler strides are accepted and
  reported but do not change the picture: they say where the tile's cells sit in the *tensor*, and
  this tab never involves a tensor.

### Highlighting a thread

All three tabs that draw a TV partition — `tv`, `make_tiled_copy`, `make_tiled_copy_tv` — take a
**Highlight thread** id, live on `oninput`, and must dim the same way: a dimmed cell keeps its
`T`/`V` labels in `#bbb` on `#f0f0f0`, the palette `buildTVSVG` uses. Highlighting is a **focus, not
a mask** — erasing the text instead (which `mtcRenderTileViz` used to do) made the rest of the tile
unreadable the moment a thread was picked, so you could no longer see what the highlighted thread was
interleaved *with*, which is the whole reason to pick one. An unclaimed cell is `—` in the same grey,
again as the TV tab draws it.

The mtc/mtv side adds two things the TV tab has no analogue for, and both should stay: `mtcReadHighlight`
**warns** on an out-of-range id and shows all threads, rather than silently ignoring it (the tile's
thread count is derived, so a stale id after an Op change is easy to hit); and `mtcRenderThreadPanel`
prints that thread's atom-by-atom `(m, n)` coordinates below the grid, which is where `FrgX > 1`
becomes legible.

`tests/dom_smoke.js` drives all three setters with an in-range id, an out-of-range one and a clear —
the fields ship empty, so nothing else in the suite reaches those handlers.

### Validation that CuTe itself skips

Everything below was verified against CuTeDSL on an SM100 — each failing case compiles and runs, and
is silently wrong.

- `mtcRequireAtomDivides` — `TiledNumVal % AtomNumVal == 0`. This one CuTe *does* assert. The message
  enumerates every `num_bits_per_copy` that would divide the given `TiledNumVal`.
- `mtcCoverageCheck` — `layout_tv` must fill `Tiler_MN` exactly. Tiler too big → the uncovered
  elements are never copied. Too small → `layout_tv` indexes into the next tile. Duplicate claims are
  an error **only when no stride is 0**; with a stride-0 mode they are intended replication, and the
  check instead requires the multiplicity to be uniform across cells. Getting this wrong rejects a
  legitimate `make_tiled_copy_A`.
- `mtcRequireCompact` (tv tab only) — `cosize() === size()` on thr_layout and val_layout.
  `make_layout_tv`'s docstring states this as a precondition and enforces nothing; `right_inverse` of
  a non-injective layout returns a *partial* inverse. The message distinguishes `cosize > size`
  (holes) from `cosize < size` (colliding coordinates).
- `mtcVectorizationCheck` — T0's first `AtomNumVal` vids must be a stride-1 run along one tile axis,
  which also tells you which major-ness the src/dst tensor needs. Note this depends on the tiler's
  *shape*, not just its size: reshaping `(8,16)` → `(16,8)` keeps the cell count but changes the
  strides `zipped_divide` produces and can break vectorization.

### make_tiled_tma_atom

TMA is not a per-thread copy: one thread issues the instruction with a logical *coordinate* and the
TMA unit does address generation, bounds handling and the swizzled SMEM write. So this tab has no TV
layout and no thread panel — `mtcRenderThreadPanel` has no analogue here, and the `TiledCopy`
wrapper's `layout_TV` is degenerate (`cta_t_map = 1` without multicast). The tab shares nothing with
the mtc/mtv tabs except ui.js's SRC/DST panes.

**The box is inferred, never given.** `cta_tiler` picks the region; `smem_layout` picks the order and
swizzle; the descriptor falls out of the two. The pipeline is a port of
`include/cute/atom/copy_traits_sm90_tma.hpp` — `construct_tma_gbasis` (:735), `coalesce_256` (:676),
`fill_tma_gmem_shape_stride` (:858), `make_tma_copy_desc` (:929), `get_tma_tensor` (:152). The DSL
entry point (`cpasync/helpers.py:419`) only packs arguments for MLIR, so C++ is the reference.

**`sidx2gmode` is inferred numerically, not symbolically.** CuTe computes
`coalesce(composition(cta_v_map, right_inverse(slayout)))` over scaled-basis strides. Porting basis
composition into layout.js would be a large piece of machinery for a tool that only ever draws 2-D,
so `tmaInferBasisModes` reads the modes off the *function* instead: each mode's stride is the first
coordinate delta, extended while the block repeats. A composition of layouts is a layout, so the
greedy decomposition is exact, and because neighbouring modes merge only when their deltas agree the
result is already coalesced. A delta that moves along two axes at once means the composition is
undefined — that is a real error, not an inference failure. **This is why the tab requires a flat
rank-2 GMEM tensor**: it keeps every basis axis a plain 0 or 1.

**The swizzle is a picker, not text.** `CUtensorMapSwizzle` is a closed enum (NONE / 32B / 64B /
128B), so section 3 offers a `<select>` labelled with both spellings — the `Sw<B,M,S>` CuTe prints
and the byte width people think in. An inline `Sw<..> o ..` prefix in the smem_layout box is still
parsed, so a CuTe printout pastes verbatim, and when present it **overrides** the picker and says so
in an amber note. Do not remove the inline path (paste-verbatim was deliberate) and do not make it
silent (two controls disagreeing with no explanation is worse than either alone). The M == 6 (64B
base) form is deliberately absent from the dropdown: `get_tma_swizzle_base` constrains only B for it
and CUTLASS's assert message there is a copy-paste of the M == 5 one, so the correct S is not
knowable from the source — it still works if typed inline.

**The TMA swizzle triple is in BYTES; `applySwizzleOffset` is in ELEMENTS.** `upcast` on a
`ComposedLayout` deliberately leaves the swizzle alone (`include/cute/pointer_flagged.hpp:73`), so
`GMMA::Layout_K_SW128_Atom<half_t>` prints as `Sw<3,4,3>` while its layout is `(8,64):(64,1)` in
elements: base `2^M` bytes, width `2^B · 2^M` bytes. That is why `get_tma_swizzle_bits`
(`copy_traits_sm90_tma_swizzle.hpp:48`) can demand `M == 4` regardless of element type. Drawing one
over an element grid needs `M - log2(bytes per element)` — `Sw<3,4,3>` on `half_t` draws as
`Sw<3,3,3>`, which is the form the **Swizzle** tab takes. `tmaSwizzleInfo` does the conversion once
and reports both forms; do not "simplify" it away, and do not feed a byte-form triple to
`applySwizzleOffset`.

**Violations are reported, not thrown.** Most TMA constraints are host-side `assert()`s inside
`make_tma_copy_desc` — they vanish in a release build and are invisible from the DSL. Since the box
is still well-defined when they fail, they land in `issues[]` and are painted into the descriptor
panel next to the picture that caused them. Only genuinely uncomputable inputs throw: size mismatch
between smem layout and tiler, a non-permutation smem layout (`right_inverse` would silently return
a *partial* inverse and shrink the box), a non-composable smem/tiler pair, and `smem_rank == 0`
("Could not find a common tile-gmem vectorization"). This is what lets the majorness-mismatch preset
exist at all — see the preset rule below.

**Cell labels follow the same rule as every other tab: the input is always shown, `value` overlays
the mapping's output.** Here the input is the element's GMEM coordinate `(i, j)` — what the TMA
instruction actually carries — and it appears on *both* panes, so the same tuple found twice is the
copy: where the element starts and where it lands. `value` adds the mapped address per space: the
linear offset from the tensor base on SRC, and the *composed* SMEM layout's output on DST, swizzle
included, since that is what `slayout(coord)` returns. It defaults OFF, like every other tab's mode
toggle. To see the swizzle's effect, flip the swizzle picker to `none` and compare — that A/B is what
the picker being a first-class control buys.

Three earlier attempts here were wrong and are worth not repeating. (1) An accumulating toggle set
printed the *same number twice* on DST — `value` as the pre-swizzle offset and `index` as the SMEM
index, which are equal since `cells[i].smemIdx === i`. (2) Suppressing that duplicate per-cell only
hid it. (3) A `logical` / `physical` single-select fixed the duplication but invented a second
vocabulary for what `value` already means everywhere else in this repo. The lesson is that DST's
extra label must come from the *other* space (the GMEM coordinate), never a second SMEM-derived
quantity. A box-coordinate label was also dropped: colour already says which box row a cell is in,
and an element's index *within* an instruction is not something anyone looks up per cell.

**A swizzle needs a tile big enough to reach its source bits.** `Sw<3,4,3>` XORs byte-offset bits
`[7,10)` into `[4,7)`, so it does nothing at all until the SMEM tile spans 8 rows of 128B. In
elements that is 8x64 `half_t` but only 8x8 `uint128_t` — same swizzle, same 128B rows, a 64th of
the cells. This is why the small presets use wide element types, and why the DST pane header reports
how many cells the swizzle actually moved: `0 cells moved` means the picture is indistinguishable
from `SWIZZLE_NONE` and the tile, not the swizzle, is the problem. Keep the tab's defaults small —
`renderAllTabs` builds every tab on panel creation, and a 64x64 tile is ~8k SVG nodes per pane.

The floors here are **byte** alignments, not tile sizes: the innermost box row must be a multiple of
16B (and no wider than the swizzle), and every non-innermost gmem stride must be a multiple of 16B.
For `half_t` the smallest legal atom is a 1x8 tile moving 16 B. The `128` in "128B swizzle" is a
*ceiling* on the box row, never a minimum.

**Viz order is box, then TMA tensor, then descriptor — and the descriptor ships folded.** It is
reference material (what `cuTensorMapEncodeTiled` was actually handed), not something you read on
every render, so its `.comp-viz-item` carries `collapsed` in the template. `attachVizCollapsibles`
now reads that class when choosing the chevron glyph, so any tab can ship a viz folded the same way.
Note the constraint violations are reported *inside* that folded panel, which is why the amber
warning under the Render button names the count — otherwise a fold would hide them.

**Draw the whole result; never a silent sample of it.** The TMA-tensor panel briefly carried a cell
budget that showed a corner, so a `(32,64)` tensor drew as `16x16` — a quarter of it, in the wrong
aspect, reading as a square tensor. `ui.js`'s `MAX_CELLS` already refuses anything genuinely
undrawable *with a message*, which beats a partial picture in both directions. The same reasoning
removed `tma_partition`'s `TP_MAX_CELLS`: atom resolution keeps realistic inputs small, and where it
cannot, an honest refusal is better than a fraction of the answer.

**The returned coordinate tensor is the GMEM-side one, never "the SRC one".** It is the source only
because the single Op supported so far is a load: `Copy_Traits<SM90_TMA_STORE>` has the same
`get_tma_tensor` (`copy_traits_sm90_tma.hpp:389`) built over the GMEM tensor, where GMEM is the
*destination*. The viz is labelled "TMA tensor (GMEM)" for that reason — naming it after the
direction would silently become wrong the moment the store Op is added, which is on this tab's own
roadmap. SMEM never has one; it is an ordinary layout over a flat buffer.

### tma_partition

Its own tab rather than a panel on `make_tiled_tma_atom`, because it is a separate call people reach
for on its own; a **flow collection** composing `make_tiled_tma_atom` + `local_tile` +
`tma_partition` is the eventual home for the end-to-end story.

**Its inputs mirror the real signature**, `tma_partition(atom, cta_coord, cta_layout, smem_tensor,
gmem_tensor)` — two *tensors*, not a tensor plus a list of Rest extents. They are entered **already
grouped**, i.e. in the `(TMATile, Rest...)` form the function actually receives: **mode 0 IS the
tile**, everything after it is Rest. Do not re-group inside the tab. An earlier version applied
`group_modes(x, 0, 2)` itself, which folded a stage mode into the tile — `((32,128),2)` came out as an
8192-element tile and tripped a spurious size assert on perfectly valid input.

One parser ambiguity to know: `parseValue` unwraps single-element parens, so a no-Rest tensor written
`((4,16))` is indistinguishable from `(4,16)`. The rule is therefore **a tuple mode 0 means "tile,
then Rest"; a scalar mode 0 means the whole layout is the tile** — which keeps the simple
`(4,16):(16,1)` form working while `((32,128),2):((128,1),4096)` reads correctly.

**Hue = which Rest element, brightness = which pass — on BOTH panels.** In tAgA a Rest element is a
tile; in tAsA it is a *stage*, since an SMEM tensor's Rest is the stage mode. Pinning tAsA at
`colorTV(0)` made two stages — two separate atom coverages — look identical. The hue index carries no
meaning beyond identity (it cycles every 8); it only separates.

**A cell is labelled with the coordinate that SELECTS it** (`tpSliceLabel`) — `[None, (0,0)]`, which
is what you would type at the copy. Mode 0 is the atom, so it prints `None` when the tile is a single
pass and `(None, i)` when it takes several; each Rest mode then contributes its own coordinate,
nested exactly as the mode is, so a Rest of `(8,15)` gives `(7,14)` rather than a flat 119. Invented
labels like `#0` / `st0` were unreadable without the source. `value` adds the GMEM coordinate (tAgA)
or the SMEM offset (tAsA) the region starts at.

**Cells are placed by GMEM COORDINATE, not by Rest-mode index** (`tpPlaceByCoord`). The panel has to
look like the tensor, with an atom's region merged into one cell — so a Rest mode that walks axis 1
must produce *columns*, whatever position it holds in the mode list. With tile strides `(1@1,1@0)` and
Rest `(8,15):(32@1,128@0)`, the 8 steps go along axis 1 and the 15 along axis 0, so the picture is
15x8 in atoms; index-ordered placement drew it 8x15, i.e. transposed. Cells landing off the atom
lattice, or an ordinary integer GMEM tensor with no axes at all, fall back to index placement.

**The drawing unit is one ATOM's footprint, not one element** (`tpAtomBlock`). Instruction `i` owns
SMEM offsets `[i*N, (i+1)*N)`, which for any sane SMEM layout is a rectangular band; the bands tile
the tile regularly, so one cell can stand for a whole pass. That is what makes real inputs drawable
*without* sampling: 120 tiles of a 4096-element tile is **120 cells**, against ~500k at element
resolution. When the footprint is not a clean rectangle `tpAtomBlock` returns null and the panel falls
back to elements, where `MAX_CELLS` applies as it does everywhere else.

The atom section presents the atom the way CuTe prints it — `ThrID: 1:0`,
`TV Layout Src = Dst = (1,N):(0,1)`, `Value type` — and asks for N, not a bit count. **You do not hand
a TMA atom `num_bits_per_tma`**; `make_tiled_tma_atom` derives it from the box it inferred, so the tab
reports the bit equivalent as output. Src == Dst holds for TMA load, multicast *and* store
(`copy_traits_sm90_tma.hpp:103, 261, 363`); mode 0's stride is degenerate (size-1 mode, so C++ prints
`1` and CuTeDSL prints `0` for the same map) and both copy tabs print the DSL's form.

`tAsA` and `tAgA` differ only in Rest — normally `((TMA,TMA_Iter))` against
`((TMA,TMA_Iter), 4, 2)` — because SMEM is one tile buffer while GMEM still carries its tile indices.
The grouping drawn in both panels is identical, since `layout_V` is built from `layout<0>(stensor)`
alone: GMEM contributes nothing to the ordering, it is only made to follow it.

**It does NOT use the SRC/DST copy panes**, and must not: `tma_partition` moves no data, so a
`SRC → DST` pair would assert a memory movement that is not happening. It draws the two *results*
stacked instead — `tAsA` above `tAgA` — and both in **tile coordinates**, sharing one colour map:
which instruction owns each element. `tAsA` is one tile, so it is a single colour when the tile is
exactly one atom and banded when it holds several. `tAgA` draws **every** tile of Rest, each
partitioned identically, with red boundaries between them — nothing is sliced, and that repetition is
the point, since `tma_partition` touches mode 0 only. This is the only Copy-scope tab without
`copyPanes`, which is why `tp` is absent from `initCopyPanes`'s prefix list.

**The results are computed, not described.** `tpComputePartition(sp, gp, numValSrc)`
holds the whole derivation, DOM-free, so it can be diffed against CuTeDSL;
`renderTmaPartition` only parses the inputs and draws what comes back. The tab runs the
actual pipeline —
`layout_V = logical_divide(right_inverse(smem tile), Layout<NumValSrc>)`, then
`coalesce(tensor.compose(layout_V), Shape<Shape<_1,_1>>)` per tensor — and prints the resulting
layouts with their strides. Verified character-for-character against CuTeDSL:

```
tAsA = ((4096,1),2):((1,0),4096)
tAgA = (((128,32),1),(12,2)):(((1@0,1@1),0),(32@1,128@0))
```

An earlier version synthesized the mode-0 string as `(NumValSrc, iters)`, which is wrong whenever
mode 0 does **not** coalesce to a flat pair — a transposed GMEM tile gives `((128,32),1)`, because its
sub-modes sit on different basis axes and `coalesce` cannot merge them. Only the SMEM side is
reliably flat. The `Shape<Shape<_1,_1>>` profile maps onto this port's `coalesce(layout, [1, 1])`.

Section 4 prints **only `tAsA` and `tAgA`** — the derivation (`inv_smem_layout`, `layout_v`,
`layout_V`) lives in the hint. A result box nobody reads is worse than a short one; the same applies
to the other tabs' result boxes when they grow.

**A stale id in `applyKeyParam` is invisible until someone opens a link** and then throws during
init, since it is not inside a try/catch. Renaming an input means updating three places — the tab
template, the tab's `exportX`, and `applyKeyParam`. This audit catches the miss:

```python
# every id applyKeyParam writes must exist in some tab template
ids  = re.findall(r'getElementById\(`\$\{tabId\}-([a-z0-9_-]+)`\)', applyKeyParam_body)
# NB: also resolve ids built as `${id}-${p}-suffix` (mtcAtomSection), or you get false positives
```

`tAgA`'s tiles are drawn in **canonical blocks** (tile `(r0,r1)` at rows `r0*T0`, cols `r1*T1`) rather
than at their true stride positions, so the picture stays a grid whatever the Rest strides are; the
cell labels evaluate the actual GMEM layout, so the truth is always in the numbers. Keep both panels
wide and short — they are stacked, so height is the scarce axis, which is why Rest defaults to 2x2
rather than 4x2.

The swizzle picker is present but changes nothing, on purpose: `get_nonswizzle_portion` strips it
first, exactly as when the box was derived. Two checks CuTe skips: the SMEM layout must be a
permutation (`right_inverse` of a non-permutation is a *partial* inverse, so `layout_v` would not
cover the tile), and `NumValSrc` must divide the tile size (`logical_divide` has to be exact).

Out of scope for now, each a clean follow-on: multicast (`domain_offset` is always zero while
`cta_layout = (1)`), TMA store, im2col, `gather4`/`scatter4`, `internal_type` recasts, and
rank > 2 tensors.

## The MMA scope

### make_mma_atom

`cute.make_mma_atom(op)` — note it takes **only the op**. Unlike `make_copy_atom` there is no dtype
argument: an MMA's input and accumulator types are part of the instruction, so they live on the Op.

**An MMA Atom is the exact analogue of a Copy Atom**, with one structural difference that drives the
whole tab. A Copy Atom has two TV layouts (`src`, `dst`, plus `ref`) over **one** tile. An MMA Atom
has three — `A`, `B`, `C` — over **three different tiles**, because the instruction reads two operands
and accumulates into a third:

| | over | for m16n8k16 f16 |
|---|---|---|
| `tv_layout_A` | (M, K) | 16x16, 8 elements/lane |
| `tv_layout_B` | (N, K) | 8x16, 4 elements/lane |
| `tv_layout_C` | (M, N) | 16x8, 4 elements/lane |

The viz mirrors that shape: **A and B sit side by side** in the two columns `.comp-results` already
provides, since they share the K axis and differ only in the M/N extent — B visibly having none is the
point. **C spans the full width below them** via `.comp-viz-span` (the general form of the older
`.comp-viz-complement` rule), because it is over a different pair of axes entirely and is what the
epilogue has to deal with.

All three are **bijections** — `size == cosize` — so every cell is owned by exactly one lane and there
is no broadcast at the atom level. That is asserted for all 28 cases, and it is why the viz passes
neither `dimTids` nor `cellIndex` to `buildTVSVG`: nothing is disabled, and these tiles are col-major
in their own codomain, which is the builder's default. (Contrast the ldmatrix atom, which needs both.)

**This tab is a TABLE, and that is the correct port.** CUTLASS defines these layouts as hand-written
`MMA_Traits` specializations — literal `Layout<Shape<...>, Stride<...>>` typedefs, one per instruction
(`include/cute/atom/mma_traits_sm80.hpp:78`). There is no arithmetic to port, because an MMA atom is a
hardware fact rather than a computation: it is the PTX register signature transcribed into layout form.
`mma.sync.aligned.m16n8k16` takes `{d0..d3}, {a0..a3}, {b0,b1}, {c0..c3}`, and that *is* 8/8/4 elements
per lane in f16. Do not try to derive `MMA_ATOM_TABLE`; extend it from CuTeDSL and add cases.

**The layouts depend only on `(op family, K)`** — 16 valid parameter combinations collapse to 6 distinct
triples. `ab_dtype` does not matter (half_t vs bfloat16_t, e4m3 vs e5m2 are identical) and neither does
`acc_dtype` (an f16 accumulator packs the same four values into two registers instead of four — that
changes the register count, not the map). The layouts are stated in units of ELEMENTS, and how many
elements pack into a 32-bit register is fixed by the family.
`run.js` enforces this rather than trusting it: any two cases sharing an `op|K` key must produce
identical layout strings. `MMA_C_LAYOUT` is a separate constant because C is invariant across
*everything* here, K included.

**Scope: `cute.nvgpu.warp`, and within it the three DENSE `WarpMmaOp` subclasses** — `MmaF16BF16Op`,
`MmaTF32Op`, `MmaFP8Op`. Two separate exclusions, for two different reasons:

- `MmaF16BF16SparseOp` is the fourth `WarpMmaOp` subclass and traces perfectly well, but it is left out
  because **the tab could not draw what distinguishes it**: `MmaAtom` exposes no metadata (E) layout,
  only A/B/C, and its A layout is over the *logical* (M, K) tile — cosize `M*K` — so the 2:4
  compression is invisible in the only thing shown. Re-adding it is a `MMA_ATOM_TABLE` entry plus a
  `sparse_metadata_format` control (which changes no layout, and neither does `acc_dtype`). The
  layouts, traced from CuTeDSL, are:

  ```
  'f16bf16_sparse|16': A '((4,8),(4,2)):((64,1),(16,8))'        B '((4,8),(2,2)):((16,1),(8,64))'
  'f16bf16_sparse|32': A '((4,8),(4,2,2)):((64,1),(16,8,256))'  B '((4,8),(2,4)):((16,1),(8,64))'
  ```

  C is `MMA_C_LAYOUT` as always. `kDomain` is `[16, 32]`; `sparse_metadata_format` is
  `{TID, REG_OFFSET}` from `cutlass.cute.nvgpu.warp.mma.SparseMetadataFormat`.
- `MmaMXF4Op` / `MmaMXF8Op` / `MmaMXF4NVF4Op` / `MmaMXF8F6F4Op` subclass `MmaOp` *directly*, not
  `WarpMmaOp` — `issubclass(MmaMXF4Op, WarpMmaOp)` is `False` — and are sm_120-only, so constructing
  one on an sm_100 box raises `expects arch to be one of [sm_120a, ...]`. No oracle could be
  generated, so they are absent rather than shipped untested.

The `WarpMmaOp` subclass check is the structural boundary; "can the three grids show what makes this
Op different" is the editorial one. Both matter when deciding whether a new Op belongs here.

Further out, Hopper `wgmma` and Blackwell `tcgen05` are a different shape of problem and do NOT belong
in this tab: `wgmma` reads A/B from SMEM through descriptors (its Op takes an `OperandSource`), and
`tcgen05` accumulates in TMEM, which is not thread-addressable. The "three register fragments" reading
that makes this tab work stops holding for both.

**Per-Op `<select>`s must be repopulated before their value is assigned**, and `mmaSyncControls`'s
`fill()` assigns the fallback EXPLICITLY (`sel.value = ok ? want : String(values[0])`) rather than
leaning on the browser auto-selecting option 0 after an `innerHTML` swap. Without that the first
render reads `''` and `parseInt` gives `NaN` — which `renderAllTabs` hits immediately, since the
template ships those selects empty. Same hazard and same shape of fix as `syncCopyMoves` and
`mcaRenderOpParams`.

**Where this meets the Copy scope.** `tv_layout_A` is exactly what `make_tiled_copy_A(atom, tiled_mma)`
hands to a copy as its `layout_tv`, and since every ldmatrix trait sets `RefLayout = DstLayout`, the
copy's destination *is* this fragment. So the ldmatrix story in `make_copy_atom` reads backwards from
here: the MMA declares the fragment, and `right_inverse(Ref) o Src` solves for which lane must supply
which address. The MMA atom is the fixed point.

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

**Every preset — and every tab's default input values — must render cleanly.** `addOuterTab` now
renders all 18 tabs on creation via `renderAllTabs`, so a tab whose defaults error would show its
error box before the user has touched anything.

**Every preset must render cleanly.** Presets are there to show working configurations, not to
demo error messages — never ship one that lands in the error box. If a case is worth teaching
because it is subtly wrong (e.g. a `val_layout` that can't vectorize), it belongs as a preset only
if the tab still renders and reports it as an inline diagnostic.

Edit the relevant `tabs/*.js` file and add preset buttons to its `generateXTabContent` output. Each preset button should call the tab's `setX(tabId, ...)` helper.

**For multi-line tiler presets, use `\\n` in the string** — the template literal then emits the two
characters `\n` into the attribute, and the onclick's own string literal reads them as a newline. A
single `\n` emits a *real* newline into the attribute, which is a SyntaxError inside a single-quoted
JS string: the browser reports it and the button silently does nothing. Two presets shipped that way.
`dom_smoke` now parses every inline handler and requires every preset button to have run, so this
cannot recur silently.
