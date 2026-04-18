# CuTe Layout Visualizer

An interactive browser-based tool for visualizing [CUTLASS CuTe](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md) layouts, thread-value (TV) layouts, and the core CuTe layout-algebra operations.

**Try now:** https://kainzhong.github.io/CuTe-Layout-Visualizer/

## Why not just `print_latex`?

CUTLASS ships `cute::print_latex(...)`, which dumps a LaTeX snippet you then have to paste into a `.tex` file, compile with `pdflatex`, and open in a PDF viewer every time you want to look at a layout. And all it ever shows you is *one static layout*.

This tool runs in your browser and lets you:

- **Visualize operations, not just layouts.** `composition`, `complement`, `logical_divide`, `zipped_divide`, `logical_product`, and `zipped_product` each get their own tab that renders the inputs **and** the result side-by-side, with coloring that makes the relationship between them obvious.
- **Edit inputs live.** Change a shape or a stride and hit Render — no rebuild, no LaTeX toolchain, no PDF reader.
- **Toggle display modes per cell** between value (`layout(i)`), index (1D flat coord), and coord (`(m,n)`) to match whatever mental model you're working in.
- **Share a URL** to a specific visualization and have a colleague open it in one click.
- **Open multiple tabs** so you can compare layouts or operations side by side.

## Features

### Layouts

- **Layout** — Render any CuTe layout specified as `shape:stride`, including nested/hierarchical modes (e.g. `((2,4),(2,4)):((1,8),(2,16))`). Auto-stride: omit the stride to default to column-major.
- **TV Layout** — Map a thread-value layout onto a tile to see how threads and values are distributed across a 2D grid. Supports two input methods: direct `(TV_Layout, Tile)` or derived-from `(Thread_Layout, Value_Layout)` via `make_layout_tv`. Click a thread to isolate its cells.

### Operations

Every operation tab shows the inputs and the result as linked visualizations, not just the algebra.

- **Composition & Complement** — Render `A`, `B`, `B_complement`, and `composition(A, B)`. Toggle the complement view to see how `complement(B, size(A))` fills in the remaining layout. Cells from the first tile in A are edge-highlighted in amber; the complement layout mirrors those highlights so the correspondence is immediate.
- **Complement** (standalone) — Render a layout and its complement against a given cotarget size. Useful for isolating what the complement operation actually produces before involving composition.
- **Logical Divide** — `logical_divide(A, tiler)` with full tile coloring: cells belonging to the same tile share a color across A and the result. Supports single-layout and multi-line (by-mode) tilers. For 2-mode tilers, the two axes use distinct accent colors (red for mode-0, deep blue for mode-1) so the row-axis and column-axis selections are visually separable.
- **Zipped Divide** — `zipped_divide(A, tiler)` with the same coloring as Logical Divide, but with the result rearranged into `((tile-local...), (across-tile...))`. Colors stay consistent so you can see that zipped_divide is purely a rearrangement of the same cells.
- **Logical Product** — `logical_product(A, tiler)` with slide-based coloring: A is the block being reproduced and gets color 0; each "slide" of A across the tiler produces the next tile, colored in the next shade. Supports single-layout and multi-line (by-mode) tilers.
- **Zipped Product** — `zipped_product(A, tiler)` (single-layout tiler). Each column of the result is one copy of A, so column `k` gets color `k` and column 0 matches A.

### Workspace

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
?key=tv-1-(32,4):(1,32)-(8,16):(16,1)
?key=tv-2-(2,3):(3,1)-(2,2):(2,1)
?key=composition-(4,4):(4,1)-(2,2):(1,2)
?key=complement-(2,2):(1,2)-(4,4):(1,4)
?key=logical_divide-(12,32):(32,1)-3:1\n8:1
?key=zipped_divide-(12,32):(32,1)-3:1\n8:1
?key=logical_product-(2,2):(1,2)-(2,2):(1,2)
?key=zipped_product-(2,2):(1,2)-(2,2):(1,2)
```

## Local development

No build step. Just serve the directory:

```
cd layout-visualizer
python3 -m http.server 8000
# open http://localhost:8000
```

See `CLAUDE.md` for architecture notes (file layout, adding a new tab, input conventions, URL scheme).
