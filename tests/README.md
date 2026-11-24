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
node tests/run.js --timings   # report elapsed time per section and overall
```

Running the tests needs **only node** — `reference.json` is committed for exactly that
reason. Regenerating it needs the `nvidia-cutlass-dsl` wheel:

```bash
npm run reference          # rewrite reference.json
npm run reference -- --section smem_layout # regenerate only this section
npm run reference:check    # regenerate and fail if it drifted (CI)
```

No GPU is required. Every layout in the corpus is static, so the whole generator runs
at **trace time** inside one `@cute.jit` function; nothing is launched. (The generator
prints a note when MLIR compilation of that combined trace fails — that is a property
of tracing a hundred unrelated TMA descriptors in one function, not of the cases. It
verifies every case was evaluated before accepting the run, and fails otherwise.)

The full Node suite keeps every corpus case, validation and DOM scenario. WGMMA
constructors are checked for all configurations, but identical actual shape/stride
layouts pointing to the same oracle record are evaluated pointwise only once.
This reduces WGMMA point evaluations from 3,588,224 to 483,328 without sampling
the unique maps. Shape, stride, size, cosize and tile coverage remain checked per
configuration.

The math and DOM harnesses share isolated browser globals through Node's VM API.
When available, `vm.constants.DONT_CONTEXTIFY` avoids proxy overhead on recursive
global function calls; older Node versions use the original contextified fallback.

## What is checked

| Section | Under test | Oracle |
|---|---|---|
| `gemm` | dispatch [4]/[5] and actual rank-one rejection, M/N/K coverage, every invocation’s A/B/C/D offsets, and imported TiledMMA traits | real `cute.gemm` trace plus `cute.slice_and_offset` / tensor indexing for every operand slice |
| `smem_layout` | Hopper A/B/epilogue base layouts, swizzles, all physical offsets, stages, major modes, dtypes and epilogue overrides | `cutlass.utils.hopper_helpers.make_smem_layout_a/b/epi` |
| `composed_layout` | Base and final outputs at every coordinate; Swizzle/Layout inner mappings, scalar/tuple offsets, signed and nested modes | `cute.make_composed_layout` (tuple-offset evaluation uses its actual CuTeDSL components; see architecture notes) |
| `domain_offset` | Unchanged layouts, iterator shifts, signed/nested coordinates and every original/shifted domain point | `cute.make_tensor` and `cute.domain_offset` |
| `index` | `idxComputeIndex`: nested and linear indices, wildcard retention, returned layout, every domain point, and base offset | `cute.slice_and_offset` |
| `layout_ops` | `coalesce`, `filter`, `composition`, `complement`, the inverses, the whole divide / product family, `blocked_product`, `raked_product`, `slice_and_offset`, `size`/`cosize`, `crd2idx`/`idx2crd`, `shape_div`, `product_each` | the same call in `cutlass.cute` |
| `basis_ops` | the same ops over **coordinate (scaled-basis) layouts** — only the ones CuTe defines for them (see the table in [`docs/architecture.md`](../docs/architecture.md)) | ditto |
| `make_layout_tv` | `make_layout_tv(thr, val)`, the `make_tiled_copy_tv` tab's derivation | `cute.make_layout_tv` **and** the `TiledCopy` that `cute.make_tiled_copy_tv` builds |
| `make_tiled_copy` | the tab's reading of `(layout_tv, Tiler_MN)` — `mtcParseTiler` and `parseLayout` | what `cute.make_tiled_copy` reports back |
| `make_tiled_copy_ab` | `mcabComputeTiledCopy`, atom traits, tiled SRC/DST and reference layouts, and every displayed grid entry | `cute.make_tiled_copy_A/B` with the same TiledMMA and CopyAtom |
| `copy_atom` | `ui.js`'s `DTYPE_BITS` and the ValLayout it implies | `cute.make_copy_atom` |
| `tma_atom` | `tmaComputeAtom` — inferred box size, `num_bits_per_tma`, the returned TMA coordinate tensor | `cpasync.make_tiled_tma_atom` |
| `tma_partition` | `tpComputePartition` — `tAsA` and `tAgA`, fed the tensors CuTe itself produced | `cpasync.tma_partition` |
| `retile` | `rtComputeRetile`, every register cell, and unchanged scalar storage footprint | `TiledCopy.get_slice(0).retile(input)` |
| `make_fragment_abc` | `mfragCompute` on a directly entered partition result; scalar register-slot layout and source-position grouping, including source slices selected by the partition shortcut | `MmaAtom.make_fragment_A/B/C` on the same already partitioned layout |
| `swizzle` | `applySwizzleOffset` | `cute::Swizzle<B,M,S>` through a `ComposedLayout` |
| `unit` | parsers and the validation CuTe skips (below) | none — expectations encode the C++ preconditions |

For every layout-valued result the suite compares three things: the printed layout, its
`size`/`cosize`, and **its value at every point of the domain**. The pointwise check is
the one that matters — two different strings can describe the same map, but a divergent
value is always a real bug.

## What CuTeDSL cannot be the oracle for

Some of this tool's checks exist precisely because CuTe does *not* perform them: the
configuration compiles, runs, and is silently wrong (see "Validation that CuTe itself
skips" in [`docs/architecture.md`](../docs/architecture.md)). Those live in `unit.js`, with expectations derived from the C++
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

**Every new tab needs tests here.** See "Testing" in [`docs/architecture.md`](../docs/architecture.md) for the checklist.

## Note on the harness

`harness.js` concatenates every source file into one script and copies the top-level
`const`/`let`/`class` names onto `globalThis`, because those never reach the global
object the way function declarations do. It also fails loudly on a duplicate top-level
lexical name across files — in the browser that would silently shadow, which is the
load-order hazard [`docs/architecture.md`](../docs/architecture.md) warns about.

`tests/decode_popup.js` clicks every Decode from Pasteboard button in the real templates with rejected pastes and checks the inline error, the popup naming the correct source tab(s), its Open/Close/Escape behavior, and that controls are unchanged. It runs in `npm test` and with `node tests/run.js decode_popup`.

`tests/variables.js` exercises the named-variable drawer, clipboard and storage fallbacks, rename/delete/preview/visualize actions, TV Layout import, and layout-valued URL extraction against the existing CuTeDSL oracle. It runs in `npm test` and with `node tests/run.js variables`.

`node tests/run.js decoding` runs pure JavaScript cross-API round trips. It uses the production URL encoder and consumer decoders, checks layout maps against the committed CuTeDSL oracle, and rejects any access to the DOM, clipboard, or localStorage. Coverage includes MMAAtom → TiledMMA → partition_A/B/C → make_fragment, CopyAtom + TiledMMA → TiledCopy, nested TiledCopy URLs → partition_S/D, TV Layout → make_tiled_copy_tv, and fragment/Layout → retile. It also covers full URLs, query strings, optional fields, transpose/matrix counts, and invalid or incompatible inputs. Rendering tests remain separate in dom_smoke.

## Hopper WGMMA atoms

`warpgroup_mma_atom` covers all supported N sizes for all three dense families,
both floating-point A sources (INT8 is SMEM-only due to a CuTeDSL 4.8
register-signedness lowering failure), supported major modes, and additional accumulator/input dtype
combinations (including mixed FP8 and INT8). CuTeDSL is traced in a separate
SM90a process; no Hopper GPU is needed to regenerate the layout oracle.
Register maps are compared at every domain point. Descriptor maps check the
broadcast thread mode and every logical operand value separately.

DOM smoke coverage includes mixed warp/WGMMA presets, default and matrix views,
memory badges, highlight/value controls, source-dependent options, mixed input
types, editable-field exports, clipboard and deep-link round trips, and acceptance
of the complete WGMMA atom by every MMA consumer editor.

On a Hopper H100/H200 node, run:

```bash
python3 tests/gpu_check.py --warpgroup
```

This compiles and executes the same WGMMA corpus and compares a nonzero-C MMA
against NumPy. `--case <substring>` narrows the corpus during debugging.
`--compile-only` checks lowering without a GPU; it does **not** pass the execution gate.

Execution validated with CuTeDSL 4.8.0 on NVIDIA H100 PCIe: all 303 offered
WGMMA configurations compiled and executed with exact agreement against NumPy.

## TMA store

```bash
python3 tests/gpu_check.py --tma
```

Requires SM90 or newer. For each `tma_atom` and `tma_partition` case (rank 2 to 5, batched
included; a `tma_partition` case's atom may be smaller than the tile, giving several passes) it builds a
`CopyBulkTensorTileS2GOp` atom, fills the SMEM tile with threads, fences the async proxy, issues
the TMA store and compares the destination GMEM bit-exactly, including that nothing outside the
tile was written. `--case <substring>` narrows the corpus. The `tma_atom` / `tma_partition`
oracle sections mirror every load case as an `-s2g` store case.

Cases marked `hw_invalid` are atoms CuTeDSL builds but the hardware rejects (SMEM destinations
not 128B-aligned, or a box row narrower than its swizzle). `run.js` requires the tab to flag
exactly these as errors; the gate runs each in its own process (a TMA fault poisons the CUDA
context) and requires it to fail.

Execution validated with CuTeDSL 4.8.0 on NVIDIA GB200 (SM100): of 108 cases, 92 executed
bit-exact and all 16 `hw_invalid` cases failed as flagged. `uint128_t` is not a TMA case: CuTeDSL
4.8 cannot compile a 128-bit TMA copy, so the TMA tabs do not offer it.

## TMEM load / store

```bash
python3 tests/gpu_check.py --tmem
```

Requires SM100. For every tcgen05 `Ld*` / `St*` Op at every repeat CuTeDSL accepts, it allocates
TMEM, builds a 128x256 float accumulator from a tcgen05 TiledMMA and round-trips each element's
index through `make_tmem_copy`: a load test writes with `St32x32bOp` (thread t <-> lane t by the
PTX definition) and reads through the Op under test; a store test writes through it and reads with
`Ld32x32bOp`. Every element must return as its own index. Validated with CuTeDSL 4.8.0 on
NVIDIA GB200: 74/74. The atom layouts are the 370 `tmem_atom` oracle cases.

## StMatrix

```bash
python3 tests/gpu_check.py --stmatrix
```

Requires SM90 or newer. For each warp-MMA `make_tiled_copy_C` case with a StMatrix atom, it fills
every thread's accumulator fragment with each element's index, retiles it to the TiledCopy,
stmatrix-stores into SMEM and reads SMEM back; every element must hold its own index (16-bit
types exactly, 8-bit types mod 256). Validated with CuTeDSL 4.8.0 on NVIDIA GB200: 15/15. The
atom layouts themselves are the 90 `stsm-*` cases in the `ldmatrix_atom` oracle section.

## LdMatrix

```bash
python3 tests/gpu_check.py --ldmatrix                                   # execute
CUTE_DSL_ARCH=sm_90a python3 tests/gpu_check.py --ldmatrix --compile-only  # target check
```

For each `make_tiled_copy_A/B` case with an LdMatrix atom, SMEM holds each element's index, the
TiledCopy loads it into the MMA fragment, and every thread writes the fragment back at its logical
coordinates; every element must return as its own index. Each case runs in its own process (a
fault poisons the CUDA context). `--compile-only` with `CUTE_DSL_ARCH` checks which architectures
accept an Op; `--stmatrix --compile-only` does the same for StMatrix.

Results with CuTeDSL 4.8.0 (GB200 execution; sm_90a / sm_80 compile-only): every 16-bit
`LdMatrix8x8x16bOp` case and its FP8 uses execute correctly and compile for sm_90a (the 16-bit
ones also for sm_80). `LdMatrix16x16x8bOp`, `StMatrix16x8x8bOp` and the unpacking forms of `LdMatrix8x16x8bOp`
execute on GB200 but libNVVM rejects them for sm_90a: they are Blackwell-only. (With
`unpack_bits=None`, `LdMatrix8x16x8bOp` is `LdMatrix8x8x16bOp`; its 6 `make_tiled_copy_A/B`
cases execute correctly.) `StMatrix8x8x16bOp` compiles for sm_90a, not
sm_80. Cases that permute K/N, and the FP8 `LdMatrix16x8x8bOp` / B-operand `16x16x8b` cases, need
an SMEM arrangement a plain row- or column-major tile cannot give (CuTe cannot prove 16 B
contiguity), so the harness cannot exercise them.

## WGMMA in every MMA consumer

Every MMA tab offers every MMA Op, so each consumer has a WGMMA oracle section, traced
in the same SM90a worker: `warpgroup_partition_abc`, `warpgroup_partition_shape_abc`,
`warpgroup_make_fragment_abc`, `warpgroup_gemm`, `warpgroup_make_tiled_copy_ab` and
`warpgroup_partition_sd` (a WGMMA make_tiled_copy_A URL). Cases carry the make_mma_atom
URL fields as `atom`; `run.js` runs them through the same checks as the warp sections
plus descriptor-specific ones (descriptor offsets, the `make_fragment_X(shape)` rejection,
V = 1 descriptor GEMM operands, and the broadcast TiledCopy of a descriptor operand,
which the tab refuses). WGMMA TiledCopy side maps are recorded over their full domain,
since 256 threads exceed `EVAL_CAP`. dom_smoke checks that every MMA picker lists
exactly `MMA_ALL_OPS`, and renders, hands off and round-trips WGMMA in each tab.

## WGMMA TiledMMA

`warpgroup_tiled_mma` adds 22 focused CuTeDSL cases for M/N/K replication, group
ordering, repeated tiles, a nontrivial permutation, source/major modes and mixed
dtypes. It reuses the already exhaustive atom traits instead of multiplying the
entire atom corpus by every tiling. RMEM partitions compare every thread/value;
SMEM partitions compare each group leader's complete values plus all broadcast
thread offsets. Node tests also cover all three display modes, real memory badges,
both orientations, atom and tiled decoding, invalid-import preservation, exports,
deep links, named-variable previews and complete constructor round trips through
every downstream MMA consumer.

```bash
python3 tests/gpu_check.py --warpgroup --tiled
```

This executes the representative cases marked `gpu` in the same corpus. K-split
layouts are checked against CuTeDSL partition ownership, while numerical GEMMs use
disjoint output tiles. Fine-grained permutation can require RMEM A: CuTeDSL 4.8
failed to construct a canonical SMEM descriptor for the tested row permutation.

H100 PCIe execution validation: all 10 marked tiled WGMMA cases compiled and ran
with exact NumPy agreement, including the nontrivial permutation with RMEM A.

`node tests/run.js session` exercises tab persistence through real templates and renderers
with fresh VM contexts to simulate hard refreshes. The DOM shim enforces select option
membership and runs real change handlers, covering dependency order, WGMMA constructor
restoration, tab order/names/active scope, closed-tab ID gaps, unrendered drafts, shared
URLs, debounced and pagehide saves, and unavailable/corrupt storage. It runs in `npm test`.

Variable Table tests also exercise native gzip/base64url table sharing: all-type and Unicode round trips, corruption and size limits, atomic imports, conflict renaming, duplicate skipping, persistence, and clipboard fallback.

Run `node tests/run.js domain_offset` for the focused domain-offset differential section, or `npm run reference -- --section domain_offset` to regenerate its oracle.
