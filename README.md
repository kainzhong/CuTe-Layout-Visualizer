# CuTe Layout Visualizer

An interactive browser-based tool for visualizing [CUTLASS CuTe](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md) layouts, thread-value (TV) layouts, and the core CuTe layout-algebra operations.

**Try now:** https://kainzhong.github.io/CuTe-Layout-Visualizer/


Know which address you are reading and tell if the load is vectorized / coalseced instantly!
<img width="1477" height="851" alt="image" src="https://github.com/user-attachments/assets/cd6693d3-5fe9-4351-a82c-3737ae8dc36f" />

## Why not just `print_latex`?

CUTLASS ships `cute::print_latex(...)`, which dumps a LaTeX snippet you then have to paste into a `.tex` file, compile with `pdflatex`, and open in a PDF viewer every time you want to look at a layout. And all it ever shows you is *one static layout*.

This tool runs in your browser and lets you:

- **Visualize operations, not just layouts.** `composition`, `complement`, `logical_divide`, `zipped_divide`, `logical_product`, and `zipped_product` each get their own tab that renders the inputs **and** the result side-by-side, with coloring that makes the relationship between them obvious.
- **Edit inputs live.** Change a shape or a stride and hit Render — no rebuild, no LaTeX toolchain, no PDF reader.
- **Toggle display modes per cell** between value (`layout(i)`), index (1D flat coord), and coord (`(m,n)`) to match whatever mental model you're working in.
- **Share a URL** to a specific visualization and have a colleague open it in one click.
- **Open multiple tabs** so you can compare layouts or operations side by side.

| Live editing, link sharing, multi-tab support | Illustrative display, more than just `print_latex` |
| --- | --- | 
| <img width="1478" height="842" alt="image" src="https://github.com/user-attachments/assets/427ade71-c0a9-4189-9e02-aa3c3a0af7df" /> | <img width="1482" height="851" alt="image" src="https://github.com/user-attachments/assets/c689b69b-9e0e-4bd2-b9c0-457f02bdd2fa" /> |



## Features

### Layouts

- **Layout** — Render any CuTe layout specified as `shape:stride`, including nested/hierarchical modes (e.g. `((2,4),(2,4)):((1,8),(2,16))`). Auto-stride: omit the stride to default to column-major.
- **TV Layout** — Map a thread-value layout onto a tile to see how threads and values are distributed across a 2D grid. Supports two input methods: direct `(TV_Layout, Tile)` or derived-from `(Thread_Layout, Value_Layout)` via `make_layout_tv`. Click a thread to isolate its cells.

### Operations

Every operation tab shows the inputs and the result as linked visualizations, not just the algebra.

- **Composition & Complement** — Render `A`, `B`, `B_complement`, and `composition(A, B)`. Toggle the complement view to see how `complement(B, size(A))` fills in the remaining layout. Cells from the first tile in A are edge-highlighted in amber; the complement layout mirrors those highlights so the correspondence is immediate.
- **Complement** (standalone) — Render a layout and its complement against a given cotarget size. Useful for isolating what the complement operation actually produces before involving composition.
- **Logical Divide** — `logical_divide(A, tiler)` with full tile coloring: cells belonging to the same tile share a color across A and the result. Supports single-layout and multi-line (by-mode) tilers. For 2-mode tilers, the two axes use distinct accent colors (red for mode-0, deep blue for mode-1) so the row-axis and column-axis selections are visually separable.
- **Zipped / Tiled / Flat Divide** — `zipped_divide(A, tiler)` and its two reshape-only siblings `tiled_divide` (unpacks outer one level) and `flat_divide` (every mode flat). All three produce the same set of cells at the same positions; a dropdown picks which textual form to display while the visualization stays identical. Coloring matches Logical Divide so you can see that these are pure rearrangements of the same cells.
- **Logical Product** — `logical_product(A, tiler)` with slide-based coloring: A is the block being reproduced and gets color 0; each "slide" of A across the tiler produces the next tile, colored in the next shade. Supports single-layout and multi-line (by-mode) tilers.
- **Zipped / Tiled / Flat Product** — `zipped_product(A, tiler)` and its two reshape-only siblings `tiled_product` (unpacks outer one level) and `flat_product` (every mode flat). All three produce the same set of cells at the same positions; a dropdown lets you see the *textual* layout for each variant while the visualization stays identical. Each column is one copy of A → column `k` gets color `k`, column 0 matches A.
- **Blocked Product** — `blocked_product(A, tiler)`, the rank-preserving cousin of `logical_product`: each output axis carries `(block_i, tile_i)` merged, so copies of A are laid down as contiguous sub-blocks of a bigger matrix (grid is `size(A_i) * size(tiler_i)` per axis). This is what you want when building a matrix tile from a per-thread block and a thread layout — it's also the primitive behind `tile_to_shape`. A gets color 0, and each block-copy at tile position `(t0, t1)` gets color `t0 + t1 * size(tiler[0])`.
- **Raked Product** — `raked_product(A, tiler)`, the interleaved twin of `blocked_product`. Same set of cells, same 2D shape, but the zip order is reversed — tile-mode first, block-mode second — so cells of a single copy of A are scattered across the output tile at stride `size(tiler_i)` along each axis instead of clumped into a contiguous sub-block. This is the primitive behind `make_layout_tv`: it's why each thread's values are spread across the tile for coalesced memory access. Same coloring scheme as Blocked Product — compare the two tabs with identical inputs to see the scattered-vs-clumped difference at a glance.

### Workspace

- **Scoped navigation** — Tabs are grouped into scopes so the tab bar doesn't turn into a wall of buttons as more features are added. The current scopes are:
  - **Basics** (blue) — Layout, TV Layout.
  - **Layout Operations** (purple) — Composition, Complement, Logical Divide, Zipped / Tiled / Flat Divide, Logical Product, Zipped / Tiled / Flat Product, Blocked Product, Raked Product.
  - **Copy** (emerald) — CopyUniversalOp / cpasync.CopyG2SOp (one tab covers both since their `Copy_Traits` are byte-identical; 4-section menu: atom / tile / partition / highlight-thread, with a layout viz per section).

  Click a scope at the top of the nav card to swap in its tabs. The active scope has a color accent (left stripe + active-tab highlight) so you always know which section you're in. Deep-link URLs auto-flip to the right scope. Scopes are designed to be extended — future groups like **MMA** can be added without cluttering the existing ones.
- **Multiple tabs** — Open several independent workspaces side by side. Each tab is fully self-contained.
- **Shareable URLs** — Every operation has an "Export URL" button that copies a deep link to the current visualization. Paste it into chat or a doc and the recipient lands on the same view.
- **Zoom** — Click "Zoom in" on any panel to fit by the shortest side (useful for very wide or very tall layouts).
- **Presets** — Built-in examples per tab covering the common patterns.
- **Rank warning** — Layouts with outer rank > 2 still render (flattened to 2D), but the tool surfaces a warning so you know the structure is being collapsed.

## Usage

Enter a CuTe layout string using the standard notation:

```
(shape_0, shape_1):(stride_0, stride_1)
```

Nested shapes and strides are supported:

```
((2,4),(2,4)):((1,8),(2,16))
```

Pick the tab that matches what you want to visualize, fill in the inputs, and click **Render**. For operation tabs, the result is computed client-side using a JavaScript port of [`python/pycute/layout.py`](../python/pycute/layout.py) — the same algebra CUTLASS itself uses, just rendered in your browser instead of compiled into a PDF.

## URL format

The `?key=...` query parameter deep-links to a specific visualization:

```
?key=layout-(10,10):(1,10)
?key=tv-1-(32,4):(1,32)-(8,16)
?key=tv-2-(2,3):(3,1)-(2,2):(2,1)
?key=composition-(4,4):(4,1)-(2,2):(1,2)
?key=complement-(2,2):(1,2)-(4,4):(1,4)
?key=logical_divide-(12,32):(32,1)-3:1\n8:1
?key=zipped_divide-(12,32):(32,1)-3:1\n8:1
?key=logical_product-(2,2):(1,2)-(2,2):(1,2)
?key=zipped_product-(2,2):(1,2)-(2,2):(1,2)
?key=blocked_product-(2,2):(1,2)-(3,3):(1,3)
?key=raked_product-(2,2):(1,2)-(3,3):(1,3)
```

## Local development

No build step, no server. Just open `index.html` in your browser &mdash; everything is plain HTML/CSS/JS with zero dependencies.

See `CLAUDE.md` for architecture notes (file layout, adding a new tab, input conventions, URL scheme).

## TODO

### Copy Atoms

`CopyUniversalOp` and `cpasync.CopyG2SOp` are covered by a single merged tab (their `Copy_Traits` are byte-identical, so one visualization suffices). Four more tabs, each with dropdown-driven variants, cover everything non-trivial that's left. Rough order by implementation complexity:

1. **ldmatrix / stmatrix** (warp copy) — single tab with:
   - Direction toggle: load (`ldmatrix`, src = shuffled smem) vs store (`stmatrix`, dst = shuffled smem).
   - Transpose picker: `N` (no transpose) / `T` (transpose).
   - Count picker: `x1` / `x2` / `x4` / `x8`.
   - Dtype picker: `u32` (SM75/SM90), `u16` (SM75/SM90), `u8` / sub-byte (SM100 additions).

   All SM75 LDSM, SM90 STSM, and SM100 LDSM/STSM variants fold into this one tab. Same pipeline as the existing Copy tabs — direct extension, easiest to build first.

2. **TMA bulk tensor** (`cpasync.CopyBulk*`) — single tab with:
   - Variant picker: `LOAD` / `LOAD_MULTICAST` / `STORE` / `REDUCE_ADD`. Optionally include the non-tensor `BULK_COPY_G2S` / `BULK_COPY_S2G` as extra items.
   - CTA picker: 1-CTA (SM90) vs 2-CTA (SM100 `SM100_TMA_2SM_LOAD*`; `ThrID = 2`).
   - Existing `thr_layout` / `val_layout` / tensor inputs drive the tile shape.

   Quirk: TMA traits carry a runtime `TmaDescriptor`, but the TV layouts are static so the atom → TiledCopy → partition pipeline still applies.

3. **tcgen05 TMEM load / store** — single tab collapses all ~160 `SM100_TMEM_LOAD_*` / `SM100_TMEM_STORE_*` variants:
   - Direction toggle: load (TMEM → regs) vs store (regs → TMEM).
   - DP picker: `16dp` / `32dp` (rows per warp).
   - Width picker: `64b` / `128b` / `256b`.
   - Repeat picker: `1` / `2` / `4` / `8` / `16` / `32` / `64`.
   - `_16b` packing toggle.

   Most combinatorial space of the four. Needs the shared SVG builder (`buildColoredLayoutSVG` and the tile-lookup helpers) to generalize beyond the rank-2 `(1, elements)` atom-val-layout assumption — TMEM atoms have rank ≥ 3 atom layouts like `(tid, (dp, bit, rep))`.

4. **UTCCP** (tcgen05 multicast into TMEM) — single tab, smallest family:
   - Shape picker: `128dp{256,128}bit`, `4dp256bit`, `4x32dp128bit`, `2x64dp128bitlw{0213,0123}` (4-5 options).
   - 1-CTA / 2-CTA toggle.

   Structurally distinct from TMA and from TMEM load/store, so a separate tab rather than merging.

**Coverage rationale**: the `Copy_Traits` families with *trivial* layouts (`ThrID = Layout<_1>`, `SrcLayout = DstLayout = Layout<Shape<_1, bits>>`) — all four `SM80_CP_ASYNC_*` variants, `SM75_U32x1_MOVM_T`, `SM100_LOAD/STORE_256bit_CACHE_NOALLOCATION` — are pixel-identical to `UniversalCopy` and don't need their own tabs. They fold into the existing `CopyUniversalOp` / `cpasync.CopyG2SOp` coverage.

### MMA Atoms

TBD.
