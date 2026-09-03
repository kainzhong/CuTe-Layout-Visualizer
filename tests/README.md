# Tests

The JS in this repo is a **port**: `layout.js` ports pycute / `include/cute/layout.hpp`,
and the Copy tabs port `cutlass.cute`'s copy constructors. A port is correct exactly
when it agrees with the thing it was ported from, so the tests are **differential**:
CuTeDSL is the oracle.

```
cases.json        the shared corpus — inputs written as CuTe layout strings
gen_reference.py  runs cases.json through CuTeDSL       -> reference.json
reference.json    committed golden output (do not hand-edit)
harness.js        loads the browser globals into node
run.js            runs cases.json through the JS port and diffs vs reference.json
unit.js           the parts with no CuTeDSL analogue (parsers, checks CuTe skips)
```

## Running

```bash
npm test                      # or: node tests/run.js
node tests/run.js layout_ops  # only sections whose name contains the filter
node tests/run.js --verbose   # print passing assertions too
```

Running the tests needs **only node** — `reference.json` is committed for exactly that
reason. Regenerating it needs the `nvidia-cutlass-dsl` wheel:

```bash
npm run reference          # rewrite reference.json
npm run reference:check    # regenerate and fail if it drifted (CI)
```

No GPU is required. Every layout in the corpus is static, so the whole generator runs
at **trace time** inside one `@cute.jit` function; nothing is launched. (The generator
prints a note when MLIR compilation of that combined trace fails — that is a property
of tracing a hundred unrelated TMA descriptors in one function, not of the cases. It
verifies every case was evaluated before accepting the run, and fails otherwise.)

## What is checked

| Section | Under test | Oracle |
|---|---|---|
| `layout_ops` | `coalesce`, `filter`, `composition`, `complement`, the inverses, the whole divide / product family, `blocked_product`, `raked_product`, `slice_and_offset`, `size`/`cosize`, `crd2idx`/`idx2crd`, `shape_div`, `product_each` | the same call in `cutlass.cute` |
| `basis_ops` | the same ops over **coordinate (scaled-basis) layouts** — only the ones CuTe defines for them (see the table in `CLAUDE.md`) | ditto |
| `make_layout_tv` | `make_layout_tv(thr, val)`, the `make_tiled_copy_tv` tab's derivation | `cute.make_layout_tv` **and** the `TiledCopy` that `cute.make_tiled_copy_tv` builds |
| `make_tiled_copy` | the tab's reading of `(layout_tv, Tiler_MN)` — `mtcParseTiler` and `parseLayout` | what `cute.make_tiled_copy` reports back |
| `copy_atom` | `ui.js`'s `DTYPE_BITS` and the ValLayout it implies | `cute.make_copy_atom` |
| `tma_atom` | `tmaComputeAtom` — inferred box size, `num_bits_per_tma`, the returned TMA coordinate tensor | `cpasync.make_tiled_tma_atom` |
| `tma_partition` | `tpComputePartition` — `tAsA` and `tAgA`, fed the tensors CuTe itself produced | `cpasync.tma_partition` |
| `swizzle` | `applySwizzleOffset` | `cute::Swizzle<B,M,S>` through a `ComposedLayout` |
| `unit` | parsers and the validation CuTe skips (below) | none — expectations encode the C++ preconditions |

For every layout-valued result the suite compares three things: the printed layout, its
`size`/`cosize`, and **its value at every point of the domain**. The pointwise check is
the one that matters — two different strings can describe the same map, but a divergent
value is always a real bug.

## What CuTeDSL cannot be the oracle for

Some of this tool's checks exist precisely because CuTe does *not* perform them: the
configuration compiles, runs, and is silently wrong (see "Validation that CuTe itself
skips" in `CLAUDE.md`). Those live in `unit.js`, with expectations derived from the C++
preconditions they encode — `mtcCoverageCheck`, `mtcVectorizationCheck`,
`mtcRequireCompact`, the TMA host-`assert()` issues, and `tpComputePartition`'s
permutation / divisibility guards.

The string parsers are the other half: CuTe's inputs are C++ types, this tool's are
text, so `parseLayout`, `mtcParseTiler`, `parseSwizzleSpec`, `tmaParseSmemField` and
`tmaSwizzleInfo`'s byte→element conversion have nothing to diff against and are pinned
directly.

## Adding a case

Add it to `cases.json`, run `npm run reference`, and commit both files. Prefer this over
`unit.js` whenever CuTeDSL can express the same call — a case with an oracle stays true
when CUTLASS changes; a hand-written expectation does not.

If a new tab's operation has no `cutlass.cute` entry point, first check whether its
*pieces* do. `tma_partition` had none as a whole until the tab's core was extracted into
`tpComputePartition`; extracting the DOM-free computation is usually the cheaper move.

## Adding a tab

**Every new tab needs tests here.** See "Testing" in `CLAUDE.md` for the checklist.

## Note on the harness

`harness.js` concatenates every source file into one script and copies the top-level
`const`/`let`/`class` names onto `globalThis`, because those never reach the global
object the way function declarations do. It also fails loudly on a duplicate top-level
lexical name across files — in the browser that would silently shadow, which is the
load-order hazard `CLAUDE.md` warns about.
