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
- **TV Layout** — Map a thread-value layout onto a tile to see how threads and values are distributed across a 2D grid. Supports two input methods: direct `(TV_Layout, Tile)` or derived-from `(Thread_Layout, Value_Layout)` via `make_layout_tv`. The tab opens pre-filled with a thr/val pair (`(4,8):(8,1)` × `(2,2):(2,1)`) and the TV layout / tile it produces, so the derivation is visible from the first screen rather than starting from a bare TV layout. Click a thread to isolate its cells. This is also where the **memory-access checks** live, as two fully independent collapsible sections, each with its own data layout and `tensor_dtype` (a GMEM tile and its SMEM staging buffer are the same tile with different strides, so they get separate inputs):
  - **Check Coalesced Read (GMEM)** — colors the grid by warp-wide memory issue (one color = all threads of a warp accessing the same block of *vector-width* consecutive vids) and labels every cell with its physical offset from the GMEM layout. The vector width is derived from the value layout and the data layout — the widest run of adjacent addresses each thread owns — so there is nothing to configure. A summary line counts the distinct 32-byte sectors each issue touches against the theoretical minimum, so "is this coalesced" gets a number, not a squint.
  - **Check Bank Conflict (SMEM)** — appends the 32-bank SMEM bank id to each cell, with an optional bank filter and a `Swizzle<B, M, S>` applied to the offset before the bank is computed.

  Both checks are properties of the *(TV layout, data layout)* pair rather than of any copy atom, which is why they live here rather than in the Copy tabs.
- **Coordinate (TMA) layouts** — The Layout tab also accepts CuTe's scaled-basis strides, `k@i`, which make a layout map a coordinate to a *coordinate* instead of a 1-D offset. That's what TMA and identity/predication tensors are built from, and it's why `(3,4):(1@1,1@0)` is a transpose while `(4,5):(1@0,1@1)` is the identity. You can paste a coordinate-tensor printout verbatim, origin and all — `(2,2) o (4,4):(1@0,1@1)` — and the origin offsets every cell. Cells show the output coordinate, coloured by output axis 0, so which logical mode feeds which output dimension is obvious at a glance.
- **Swizzle** — Visualize a CuTe `Swizzle<B, M, S>` as a before/after pair over a base layout. Top grid shows the raw base layout (cell = logical offset `a`); bottom grid shows the same coords with each cell labelled `a → b` where `b = a ⊕ (((a >> (M+S)) & ((1<<B)-1)) << M)`. Bottom-cell colour is keyed to the swizzled offset `b`, so same colour = same post-swizzle address bucket — makes conflict-avoidance patterns visible at a glance.

### Operations

Every operation tab shows the inputs and the result as linked visualizations, not just the algebra.

- **Composition & Complement** — Render `A`, `B`, `B_complement`, and `composition(A, B)`. Toggle the complement view to see how `complement(B, size(A))` fills in the remaining layout. Cells from the first tile in A are edge-highlighted in amber; the complement layout mirrors those highlights so the correspondence is immediate.
- **Complement** (standalone) — Render a layout and its complement against a given cotarget size. Useful for isolating what the complement operation actually produces before involving composition.
- **Logical Divide** — `logical_divide(A, tiler)` with full tile coloring: cells belonging to the same tile share a color across A and the result. Supports single-layout and multi-line (by-mode) tilers. For 2-mode tilers, the two axes use distinct accent colors (red for mode-0, deep blue for mode-1) so the row-axis and column-axis selections are visually separable.
- **Zipped / Tiled / Flat Divide** — `zipped_divide(A, tiler)` and its two reshape-only siblings `tiled_divide` (unpacks outer one level) and `flat_divide` (every mode flat). All three produce the same set of cells at the same positions; a dropdown picks which textual form to display while the visualization stays identical. Coloring matches Logical Divide so you can see that these are pure rearrangements of the same cells.
- **Logical Product** — `logical_product(A, tiler)` with slide-based coloring: A is the block being reproduced and gets color 0; each "slide" of A across the tiler produces the next tile, colored in the next shade. Supports single-layout and multi-line (by-mode) tilers.
- **Zipped / Tiled / Flat Product** — `zipped_product(A, tiler)` and its two reshape-only siblings `tiled_product` (unpacks outer one level) and `flat_product` (every mode flat). All three produce the same set of cells at the same positions; a dropdown lets you see the *textual* layout for each variant while the visualization stays identical. Each column is one copy of A → column `k` gets color `k`, column 0 matches A.
- **local_tile** — `local_tile(A, tiler, coord)`, which is just `zipped_divide` followed by a slice: cut A into tiles, keep the tile modes whole, index into the "rest" with your coord. Since the point of it is *picking*, the visualization greys out everything the coord discarded and colours only what survives — one colour per surviving tile, so an `_` in the coord reads as several tiles rather than one big selection. Also prints the base offset separately from the layout, because slicing produces both and a layout structurally cannot carry a constant. Two things it makes concrete: `_` **keeps** a mode rather than picking (`(None, None)` selects no tile at all — it exposes the tile indices as modes, which is what the TMA flow does before `tma_partition`), and mode 0 is **unpacked one level, not flattened** — `((2,2),(4,2))` comes back as two modes that are each still tuples, and only a tile whose sub-modes are already scalars looks flat. Either way mode 0 stops being one mode, which is why TMA code has to call `group_modes(x, 0, 2)` to fold back what `local_tile` just unfolded.
- **Blocked Product** — `blocked_product(A, tiler)`, the rank-preserving cousin of `logical_product`: each output axis carries `(block_i, tile_i)` merged, so copies of A are laid down as contiguous sub-blocks of a bigger matrix (grid is `size(A_i) * size(tiler_i)` per axis). This is what you want when building a matrix tile from a per-thread block and a thread layout — it's also the primitive behind `tile_to_shape`. A gets color 0, and each block-copy at tile position `(t0, t1)` gets color `t0 + t1 * size(tiler[0])`.
- **Raked Product** — `raked_product(A, tiler)`, the interleaved twin of `blocked_product`. Same set of cells, same 2D shape, but the zip order is reversed — tile-mode first, block-mode second — so cells of a single copy of A are scattered across the output tile at stride `size(tiler_i)` along each axis instead of clumped into a contiguous sub-block. This is the primitive behind `make_layout_tv`: it's why each thread's values are spread across the tile for coalesced memory access. Same coloring scheme as Blocked Product — compare the two tabs with identical inputs to see the scattered-vs-clumped difference at a glance.

### Workspace

- **Scoped navigation** — Tabs are grouped into scopes so the tab bar doesn't turn into a wall of buttons as more features are added. The current scopes are:
  - **Basics** (blue) — Layout, TV Layout, Swizzle.
  - **Layout Operations** (purple) — Composition, Complement, Logical Divide, Zipped / Tiled / Flat Divide, Logical Product, Zipped / Tiled / Flat Product, Blocked Product, Raked Product.
  - **Copy** (emerald) — the copy-construction pipeline, four tabs mirroring CuTe's own layering.
    - **make_copy_atom** — build one Copy_Atom. Pick the Op (`CopyUniversalOp` or `cpasync.CopyG2SOp`; neither takes constructor parameters), then give `make_copy_atom` its `tensor_dtype` and `num_bits_per_copy`. Shows the one-thread / N-contiguous-value shape a single instruction moves.
    - **make_tiled_copy** — the primitive. You supply `layout_tv` and `Tiler_MN` yourself. (They need *not* have matching shapes — `layout_tv` is `(num_threads, num_values)`, `Tiler_MN` is the `(M, N)` tile.)
    - **make_tiled_copy_tv** — the derived form. Give it a `thr_layout` × `val_layout` and it runs `raked_product` → `right_inverse`, prints `layout_mn` / `Tiler_MN` / `layout_tv`, and hands them to the above.
    - **make_tiled_tma_atom** — the TMA path, which skips threads entirely. Give it a GMEM tensor, an SMEM layout, a swizzle (a picker, since `CUtensorMapSwizzle` is a closed enum — each option labelled with both the `Sw<B,M,S>` CuTe prints and the byte width you think in) and a CTA tiler; it derives the TMA **box** — you never specify it — and shows three things: the box laid over the CTA tile against its SMEM destination, the literal `cuTensorMapEncodeTiled` argument list, and the coordinate tensor the function returns.

    All four tabs render as a **SRC → DST** pair: a `SRC / DST / BOTH` toggle above the diagram, each pane titled with its memory space (GMEM / SMEM / RMEM / TMEM) and a small `→` between them. In BOTH mode the two grids sit side by side at half width, same aspect ratio. The memory movement is a constrained picker in section 0, not a free choice: it lists only the pairings the selected Op supports, and the pane titles follow it. `cpasync.CopyG2SOp` offers exactly one (GMEM→SMEM, so the picker is disabled); `CopyUniversalOp` offers the six cross-space pairs over GMEM/SMEM/RMEM — TMEM is excluded because it isn't thread-addressable and needs a tcgen05 Op. For both current Ops `ValLayoutSrc == ValLayoutDst`, so the panes match; they diverge for shuffling atoms like `ldmatrix`.

    TMA is the odd one out and deliberately so: one thread issues the instruction with a logical *coordinate* and the TMA unit does address generation, bounds handling and the swizzled SMEM write, so there is no TV layout and no per-thread view. What replaces it is the box. Presets run from a 64-cell tile up to a full 64x64 GEMM stage; the small ones use wide element types on purpose, because a swizzle does nothing until the tile spans 8 rows of 128 B — 8x64 in `half_t`, but only 8x8 in `uint128_t`. The DST pane reports how many cells the swizzle actually moved, so "0 moved" tells you the tile is too small rather than leaving you to wonder. Two more things the tab makes visible that are otherwise hard to see: the swizzle triple CuTe prints (`Sw<3,4,3>`) is in **bytes**, so it is shown alongside its element-grid equivalent (`Sw<3,3,3>` for `half_t`) — the form the Swizzle tab takes; and the TMA constraints are host-side `assert()`s that a release build removes, so violations (majorness mismatch, box extent > 256, unaligned strides, a box row wider than the swizzle) are reported inline next to the picture that caused them rather than replacing it with an error.

    The two tiled tabs draw how the TiledCopy covers one tile — color per thread, brightness per atom invocation — and run the checks CuTe *documents but never enforces*: that `layout_tv` fills its tiler, that thr/val layouts are compact, and that the atom's values land on a stride-1 run of one tile axis. A stride-0 mode in `layout_tv` is recognised as deliberate broadcast (several threads reading one element, as `make_tiled_copy_A` produces) rather than flagged as overlap.

  Click a scope at the top of the nav card to swap in its tabs. The active scope has a color accent (left stripe + active-tab highlight) so you always know which section you're in. Deep-link URLs auto-flip to the right scope. Scopes are designed to be extended — future groups like **MMA** can be added without cluttering the existing ones.
- **Cmd+Enter to render** — `⌘↵` on macOS, `Ctrl+↵` elsewhere, renders whichever tab is visible without scrolling down to the button. Works from inside any input, including the multi-line tiler boxes. The hint under each Render button shows the right key for your platform.
- **Multiple tabs** — Open several independent workspaces side by side. Each tab is fully self-contained.
- **Shareable URLs** — Every operation has an "Export URL" button that copies a deep link to the current visualization. Paste it into chat or a doc and the recipient lands on the same view.
- **Zoom** — Click "Zoom in" on any panel to fit by the shortest side (useful for very wide or very tall layouts).
- **Everything is drawn on load** — every tab renders its default inputs when the page opens, so switching tabs shows a working example immediately rather than an empty box. Every shipped default is a valid configuration.
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

2. **TMA bulk tensor** (`cpasync.CopyBulk*`) — *partly built*: the **make_tiled_tma_atom** tab covers `CopyBulkTensorTileG2SOp` (plain `.tile` load, `num_multicast = 1`) over a flat rank-2 tensor. Still to come, each a clean extension of the same pipeline:
   - Variant picker: `LOAD_MULTICAST` (`num_multicast > 1` truncates the box), `STORE`, `REDUCE_ADD`. Optionally the non-tensor `BULK_COPY_G2S` / `BULK_COPY_S2G`.
   - CTA picker: 1-CTA (SM90) vs 2-CTA (SM100 `SM100_TMA_2SM_LOAD*`; `ThrID = 2`).
   - im2col mode, `gather4` / `scatter4`, `internal_type` recasts, rank > 2 tensors.
   - `tma_partition`, which splits a tile into `(TMA, REST)` when the box doesn't cover it in one instruction (the tab already reports the instruction count).

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
