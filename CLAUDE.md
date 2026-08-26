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
  swizzle.js   "Swizzle" tab — BASICS scope; two stacked grids (base + Swizzle<B,M,S> applied), bottom cells show "a → b"
  composition.js   "Composition & Complement" tab (with complement-toggle)
  complement.js    Standalone "Complement" tab
  divide.js        "Logical Divide" tab
  zipped.js        "Zipped / Tiled / Flat Divide" tab (dropdown picks result form)
  local_tile.js    "local_tile" tab — `zipped_divide` + a slice. Greys out everything the coord
                   did NOT select, one colour per surviving tile. Prefix `lt`.
  product.js       "Logical Product" tab
  zipped_product.js "Zipped / Tiled / Flat Product" tab (single-layout tiler, dropdown picks result form)
  blocked_product.js "Blocked Product" tab (rank-preserving matrix tiling)
  raked_product.js  "Raked Product" tab (block-interleaved, scattered)
  make_copy_atom.js     "make_copy_atom" tab (COPY scope) — mirrors cute.make_copy_atom(op, dtype,
                        num_bits_per_copy). Section 1 picks the Op and shows its CONSTRUCTOR params
                        (both current Ops have none); section 2 takes make_copy_atom's own arguments.
                        Visualizes the Copy_Atom ONLY — one 1xN value grid out. Prefix `mca`.
  make_tiled_copy.js    "make_tiled_copy" tab (COPY scope) — the PRIMITIVE constructor: you supply
                        layout_tv and Tiler_MN directly. Also holds everything shared with the
                        make_tiled_copy_tv tab (mtcAtomSection/mtcReadAtom, mtcVizSection,
                        mtcBuildTileLookup, mtcCoverageCheck, mtcVectorizationCheck, mtcRenderTileViz,
                        mtcRenderThreadPanel, setMtcMode) — prefix `mtc`.
  make_tiled_copy_tv.js "make_tiled_copy_tv" tab (COPY scope) — the DERIVED constructor: thr_layout
                        + val_layout in, (layout_tv, Tiler_MN) out, then the same viz — prefix `mtv`.
  make_tiled_tma_atom.js "make_tiled_tma_atom" tab (COPY scope) — the TMA atom for
                        `cpasync.CopyBulkTensorTileG2SOp` (num_multicast fixed at 1). Ports the C++
                        pipeline in include/cute/atom/copy_traits_sm90_tma.hpp; draws the box over the
                        CTA tile (SRC) against SMEM (DST), the cuTensorMapEncodeTiled argument list,
                        and the returned coordinate tensor. Self-contained — shares nothing with the
                        mtc/mtv tabs but ui.js's copy panes. Prefix `tma`.
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
- **Colors**: `BW_COLORS`, `TV_COLORS`, `HIGHLIGHT_COLORS`, `colorBW`, `colorTV`, `colorHighlight`, `textOnBG`

### layout.js — pycute port + CuTe helpers
Port of `python/pycute/int_tuple.py` and `python/pycute/layout.py`, plus a few helpers ported from `include/cute/layout.hpp` and `python/CuTeDSL/cutlass/cute/core.py`.
- **Int tuple helpers**: `is_int`, `is_tuple`, `flatten`, `product`, `prefix_product`, `crd2idx`, `idx2crd`, `shape_div`, `slice_`, `has_none`
- **Layout class**: `new Layout(shape, stride)`, `.call(...args)`, `.mode(i)`, `.rank()`, `.size()`, `.cosize()`
- **Layout functions**: `make_layout`, `coalesce`, `filter`, `composition`, `complement`, `right_inverse`, `left_inverse`, `logical_divide`, `logical_product`, `zipped_divide`, `tiled_divide`, `zipped_product`, `tiled_product`, `slice_and_offset`
- **Extra**: `product_each`, `zip_tuple`, `zip_layouts`, `append_layout`, `raked_product`, `make_layout_tv`, `isBijective`

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
- **SVG builders**: `buildLayoutSVG`, `buildTVSVG`, `buildHighlightedLayoutSVG`, `buildGridSVG`, `errSVG`
- **SVG helpers**: `cellSize`, `svgFitStyle`, `cellTextSVG`, `buildCellLines`, `toModeSet`
- **Zoom**: `applyZoomState`, `toggleZoom`
- **Copy SRC/DST panes**: `COPY_OP_MOVES`, `copyMoveField`, `syncCopyMoves`, `copyMove`, `setCopyMove`, `updateCopyPaneTitles`, `initCopyPanes`, `copyDirButtons`, `copyPanes`, `setCopyDir`, `copyDir`, `toggleCopyZoom` — the side-by-side view shared by all four Copy tabs. Both SVGs are always in the DOM; `data-dir` on `.copy-panes` decides visibility, so SRC/DST/BOTH is pure CSS and needs no re-render. In BOTH mode the panes are equal flex children, which halves each SVG's width while `width:100%;height:auto` preserves its ratio. Note `attachVizFullscreenButtons` iterates **every** `.viz-box` inside a `.comp-viz-item`, not just the first — the Copy tabs put two panes in one item, and taking the first left DST without a button.
- **Memory movement is constrained, not free.** `COPY_OP_MOVES` lists the legal `(src, dst)` pairs per Op key; section 0 of each Copy tab is a `<select>` built from it, and the pane titles are read-only labels driven by that select. `cpasync.CopyG2SOp` has exactly one entry (GMEM→SMEM) so its picker is `disabled`. `CopyUniversalOp` gets the six cross-space pairs over GMEM/SMEM/RMEM — **TMEM is deliberately excluded**, because it isn't thread-addressable (tcgen05's Ld/St src layout is stride-0 across all 32 lanes), so reaching it requires a tcgen05 Op. `tma_g2s` (the bulk-tensor load) has one entry too, for the same reason as `cpasync`. `syncCopyMoves(tabId, p, opKey)` rebuilds the options on every Op change and keeps the current selection when the new Op still permits it; `initCopyPanes(tabId)` runs from `addOuterTab` so the picker and titles are right before the first Render.
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
?key=zipped_product-(2,2):(1,2)-(2,2):(1,2)
?key=blocked_product-(2,2):(1,2)-(3,3):(1,3)
?key=raked_product-(2,2):(1,2)-(3,3):(1,3)
?key=make_copy_atom-universal-128-half_t
?key=make_tiled_copy-cpasync-128-half_t-((8,16),8):((128,1),16)-(16, 64)
?key=make_tiled_copy_tv-cpasync-128-half_t-(16,8):(8,1)-(1,8):(1,1)
?key=swizzle-(8, 8):(8, 1)-3, 0, 3
?key=make_tiled_tma_atom-half_t-(256, 128):(128, 1)-3,4,3-(64, 64):(64, 1)-(64, 64)
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

7. **Register the tab button** in `generateTabContent` in ui.js with a `data-scope` — e.g. `<div class="tab" data-scope="operations" onclick="switchInnerTab('${id}', 'yourtab')">Your Tab</div>`. Pick the scope the tab belongs to (see "Scopes" below).
8. **Add to `modeIndex`** in `switchInnerTab` in ui.js: `{ ..., yourtab: <next index> }`. The index must match the tab's order inside `.tab-bar`.
8b. **Add to `TAB_RENDER_FN`** in ui.js: `{ ..., yourtab: 'renderYourTab' }`, and give the tab's `<div class="tab">` a `data-tab="yourtab"` attribute. This is what Cmd/Ctrl+Enter uses to find the render function of whichever tab is visible. Name the tab's main button exactly `Render` — `attachRenderHints` keys off that text to inject the keyboard hint, and skips variants like "Render Inverse" that are toggles rather than the tab's main render.
9. **Call your generator** from `generateTabContent`: append `${generateYourTabContent(id)}`.
10. **Load the script** in `index.html`: `<script src="tabs/yourtab.js"></script>` (must appear after ui.js, before the inline init).
11. **Document it in this CLAUDE.md** under "What lives where → tabs/*.js".

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
- `copy` — the copy-construction pipeline: `make_copy_atom` (one instruction), then `make_tiled_copy` / `make_tiled_copy_tv` (replicate it over a tile), plus `make_tiled_tma_atom` (the TMA path, which bypasses threads entirely). Accent color: emerald (`#10b981`).

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
| Whether the access pattern is coalesced / bank-conflict-free | `tv` | `basics` | A property of (TV layout, **data layout**) — no atom involved |

`make_copy_atom` keeps the DSL's two-step split visible: an Op is constructed with its own fields
first, then `make_copy_atom(op, dtype, num_bits_per_copy)` turns it into an Atom. `MCA_OPS[key].params`
is the list of constructor params — empty for both current Ops, which is why section 1 says so rather
than rendering controls. Adding `warp.LdMatrix*` (num_matrices, transpose) or `tcgen05.Ld*` (repeat,
pack) means filling that list, not restructuring the form.

All four Copy tabs render into `-<prefix>-src-svg` / `-<prefix>-dst-svg` rather than a single host.
For `CopyUniversalOp` and `cpasync.CopyG2SOp` the two panes are identical, because
`ValLayoutSrc == ValLayoutDst`; they only diverge for shuffling atoms such as `ldmatrix`, where the
src layout is an *addressing* pattern (which lane points where) and the dst layout is the register
outcome. Keep the two render paths separate even while they agree.

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

**The returned coordinate tensor is the GMEM-side one, never "the SRC one".** It is the source only
because the single Op supported so far is a load: `Copy_Traits<SM90_TMA_STORE>` has the same
`get_tma_tensor` (`copy_traits_sm90_tma.hpp:389`) built over the GMEM tensor, where GMEM is the
*destination*. The viz is labelled "TMA tensor (GMEM)" for that reason — naming it after the
direction would silently become wrong the moment the store Op is added, which is on this tab's own
roadmap. SMEM never has one; it is an ordinary layout over a flat buffer.

Out of scope for now, each a clean follow-on: multicast, TMA store, im2col, `gather4`/`scatter4`,
`internal_type` recasts, rank > 2 tensors, and `tma_partition`.

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
renders all 15 tabs on creation via `renderAllTabs`, so a tab whose defaults error would show its
error box before the user has touched anything.

**Every preset must render cleanly.** Presets are there to show working configurations, not to
demo error messages — never ship one that lands in the error box. If a case is worth teaching
because it is subtly wrong (e.g. a `val_layout` that can't vectorize), it belongs as a preset only
if the tab still renders and reports it as an inline diagnostic.

Edit the relevant `tabs/*.js` file and add preset buttons to its `generateXTabContent` output. Each preset button should call the tab's `setX(tabId, ...)` helper. For multi-line tiler presets, use `\\n` in the string (the template literal produces `\n` in the HTML, which JS then interprets as a newline).
