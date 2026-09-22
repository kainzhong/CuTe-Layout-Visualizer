// partition_A / partition_B / partition_C tab (MMA scope): the ThrMMA slice.
//
//   ThrMMA thr  = tiled_mma.get_slice(thr_idx);
//   Tensor tCgA = thr.partition_A(gA);      // or partition_B / partition_C
//
// This tab stops at the partition. What a thread then holds in REGISTERS is
// CuTeDSL's `MmaAtom.make_fragment_A(partition_A(t))` — `pabcComputePartition`
// still returns it as `fragment` and `tests/run.js` diffs it against the DSL,
// but nothing draws it. (C++ wraps the pair as `ThrMMA::partition_fragment_A`;
// CuTeDSL has no such wrapper, which is why this tool does not offer one.)
//
// The MMA counterpart of the partition_S / partition_D tab, and deliberately
// the same three-level picture, because the levels mean the same things:
//
//   1. what one thread does inside one tile        (grid 1)
//   2. how that tile covers the operand's plane    (grid 2)
//   3. how that plane covers the whole tensor      (grid 3)
//
// It is a port of `TiledMMA::thrfrg_A/B/C` and `ThrMMA::partition_A/B/C`
// (include/cute/atom/mma_atom.hpp:291, 468). `thrfrg_X` is four lines:
//
//   t_tensor   = logical_divide(atensor, (permM, permK))     the permutation
//   a_tensor   = zipped_divide(t_tensor, (AtomM, AtomK))     the atom
//   tv_tensor  = a_tensor.compose(AtomLayoutA_TV, _)         (M,K) -> (T,V)
//   thr_tensor = zipped_divide(tv_tensor, (_, (ThrM, ThrK))) split off the warps
//
// which is exactly `mtmThrfrg` in tabs/make_tiled_mma.js — that function grew an
// optional tensor argument for this tab, since make_tiled_mma only ever asks
// about one tile. So there is ONE implementation of the derivation and this tab
// is the same code with a real tensor in it.
//
// ─── The unit level 1 draws is NOT the TiledMMA's tile ──────────────────────
//
// `permutation_mnk` folds into the RETURNED Rest modes, so `partition_A`'s Rest
// is not "one entry per tile" the way `partition_S`'s is. A permuted tile can
// hold SEVERAL Rest positions, because the warp pattern repeats inside it:
//
//   B, atom_layout (2,2,1), permutation_mnk (32,32,16), over a (128,32,4)
//     tile 32x16, but partition_B = ((2,2),8,2,4) — Rest is 8x2, so a Rest
//     position covers 16x16, half the tile in N.
//
// Drawing the tile as level 1 therefore made level 2 count 4x2 where the result
// says 8x2 — an actual bug, and the reason this tab derives its level-1 block
// instead of assuming one: it gathers every (thread, value) cell at Rest 0 and
// takes the region they form. That block is always what one Rest position
// covers, so the three grids are exactly modes 0, 1-2 and 3+ of the returned
// layout and their headers concatenate to it.
//
// The permutation is still visible, in the two places it actually acts. A plain
// multiple only means the tile holds several blocks, which grid 2 reports. A
// permutation LAYOUT also interleaves the rows: the block stays a rectangle in
// its own coordinates but occupies non-consecutive tensor rows, and Rest comes
// back NESTED — `((2,32):(32,1),32,16)` on a (128,48) A gives
// `((2,2,2),(2,2),3)`, Rest mode 0 being (repeat inside the tile, tiles across
// the tensor). Both are said out loud in the headers rather than smoothed over.

/** Which two axes each operand spans, and which of tile_size(M,N,K) they are. */
const PABC_OPERAND = {
  A: { fn: 'partition_A', axes: ['M', 'K'], mnk: [0, 2], sum: false },
  B: { fn: 'partition_B', axes: ['N', 'K'], mnk: [1, 2], sum: false },
  C: { fn: 'partition_C', axes: ['M', 'N'], mnk: [0, 1], sum: true },
};

// ═══════════════════════════════════════════════════════
//  The derivation (DOM-free, so tests/run.js can diff it against CuTeDSL)
// ═══════════════════════════════════════════════════════

/** `filter_zeros(stride, shape)` — shape, with every entry whose stride is 0
 *  replaced by 1. A stride-0 mode is read, never stored, so it costs no
 *  register. (Nothing in this tab's Ops actually produces one: an MMA
 *  broadcasts on the THREAD side, not the value side. Ported anyway, because
 *  the omission would be invisible until an Op that does hits it.) */
function pabcFilterZeros(stride, shape) {
  if (is_tuple(shape)) return shape.map((sh, i) => pabcFilterZeros(stride[i], sh));
  return stride_is_zero(stride) ? 1 : shape;
}

/** Re-impose `shape`'s nesting on a flat list of values. */
function pabcRenest(flat, shape, at) {
  at = at || { i: 0 };
  if (!is_tuple(shape)) return flat[at.i++];
  return shape.map(sh => pabcRenest(flat, sh, at));
}

/** `make_ordered_layout(shape, order)`'s strides: compact, assigned smallest
 *  first in the order `order` ranks the modes. Hierarchy is flattened, which is
 *  what makes a nested Rest mode order correctly against a flat one.
 *
 *  A size-1 mode gets stride **0**, not the running product — the same fact
 *  `mtmCompactStride` exists for. Unobservable in the map, but a tensor exactly
 *  one tile wide has Rest `1,1` and CuTeDSL prints `((2,2,2),1,1):((1,2,4),0,0)`
 *  where the running product would say `8,8`. */
function pabcOrderedStrides(shape, order) {
  const fs = flatten(shape), fo = flatten(order);
  const rank = fs.map((_, i) => i).sort((a, b) => (fo[a] - fo[b]) || (a - b));
  const out = new Array(fs.length);
  let acc = 1;
  for (const i of rank) {
    if (fs[i] === 1) { out[i] = 0; continue; }
    out[i] = acc;
    acc *= fs[i];
  }
  return pabcRenest(out, shape);
}

/** `make_fragment_A|B|C` — mma_atom.hpp:130-193, over the layout `partition_X`
 *  returned. The register tensor a thread declares for that operand.
 *
 *  A and B go through `make_fragment_like` (layout.hpp:455): mode 0 — the
 *  atom's own value mode — is forced compact COL-MAJOR whatever the partition
 *  did, and the trailing modes get compact strides in the ORDER the partition's
 *  strides rank them. That ordering is the whole point of these functions
 *  taking an already-partitioned tensor: it makes a linear walk of the fragment
 *  a monotone walk of the source, so the copy into it vectorizes. Flip the
 *  source's majorness and the fragment's mode order flips with it.
 *
 *  C does NOT: `make_fragment_C` is `make_tensor<FrgTypeC>(shape(ctensor))`,
 *  a plain compact layout of the same shape, because an accumulator is never
 *  read from memory in the order it was partitioned ("We'll never base the
 *  accumulator layout on the input tensor layout" — mma_atom.hpp:140). */
function pabcMakeFragment(L, which) {
  // `mtmCompactStride`, not `prefix_product`: CuTe's compact stride gives a
  // size-1 mode 0 rather than the running product.
  if (which === 'C' || L.rank() <= 1) return new Layout(L.shape, mtmCompactStride(L.shape));
  const shape0 = L.shape[0], stride0 = L.stride[0];
  const s0 = mtmCompactStride(pabcFilterZeros(stride0, shape0));
  const stride0New = is_tuple(shape0)
    ? shape0.map((sh, i) => (stride_is_zero(stride0[i]) ? 0 : s0[i]))
    : (stride_is_zero(stride0) ? 0 : s0);
  const size0 = product(shape0);

  const restShape  = L.shape.slice(1);
  const restOrder  = L.stride.slice(1);
  const restStride = pabcOrderedStrides(restShape, restOrder);
  const scale = (x) => is_tuple(x) ? x.map(scale) : (x === 0 ? 0 : x * size0);
  return new Layout([shape0, ...restShape], [stride0New, ...restStride.map(scale)]);
}


/** `tiled_mma.get_slice(thr).partition_X(tensor)` plus everything the three
 *  grids need.
 *
 *  `atom` is a `mmaWarpAtom(...)` result, `atomLayout` a rank-3 Layout, `perm`
 *  a 3-element array of Layout-or-null, `which` one of 'A' / 'B' / 'C'. */
function pabcComputePartition(atom, atomLayout, perm, which, tensor, thrIdx) {
  const spec = PABC_OPERAND[which];
  if (!spec) throw new Error(`Unknown operand "${which}" — expected A, B or C.`);

  const R = tensor.rank();
  // The same two limits the partition_S/D tab has, and for the same reason —
  // they are what the three grids can draw honestly, not anything CuTe asserts.
  // An MMA operand is always over exactly two axes, so the tiler side is not
  // even a choice here; only the tensor's rank is.
  if (R < 2)
    throw new Error(
      `The tensor has rank ${R}. An MMA operand spans two axes (${spec.axes.join(', ')} for ` +
      `${which}), and CuTe asserts rank(tensor) >= 2 (mma_atom.hpp:293).`);
  if (R > 4)
    throw new Error(
      `The tensor has rank ${R}, i.e. ${R - 2} modes past the operand's two axes. This tab ` +
      `draws those as a 2-D grid, so it takes at most 2 — in a GEMM they are the k-loop and ` +
      `the pipeline stage. Group the extras first if you really have more.`);

  // `thrfrg_X` divides by permutation_mnk, so THAT is what has to divide the
  // tensor. It is not the unit the grids draw — see below.
  const overTile = mtmComputeTiledMma(atom, atomLayout, perm);
  const divExtents = spec.mnk.map(i => overTile.tileMNK[i]);
  const tExt = (is_tuple(tensor.shape) ? tensor.shape : [tensor.shape]).map(m => product(m));
  for (let i = 0; i < 2; i++) {
    if (tExt[i] % divExtents[i] !== 0)
      throw new Error(
        `The TiledMMA's ${spec.axes[i]} tile is ${divExtents[i]}, which does not divide the ` +
        `tensor's mode ${i} (${tExt[i]}). zipped_divide has to be exact. Divisors of ` +
        `${tExt[i]}: ${psdDivisors(tExt[i]).join(', ')}.`);
  }

  if (thrIdx !== null && thrIdx >= overTile.threads)
    throw new Error(
      `Thread index ${thrIdx} is out of range — this TiledMMA has ${overTile.threads} threads ` +
      `(${overTile.warps} warp${overTile.warps === 1 ? '' : 's'} of ${MTM_ATOM_THREADS}), ` +
      `so 0..${overTile.threads - 1}.`);

  // The derivation over the real tensor, and the same one over the COMPACT
  // layout of the same shape. The second is only for the pictures, so a
  // transposed or non-injective tensor cannot alias two cells onto one.
  const opT   = mtmComputeTiledMma(atom, atomLayout, perm, { [which]: tensor })[which];
  const opPos = mtmComputeTiledMma(atom, atomLayout, perm,
                                   { [which]: new Layout(tensor.shape) })[which];

  const frg = opT.frgSize;
  const nThr = product(opT.thr.shape);
  const [M0, M1] = [tExt[0], tExt[1]];

  // ── LEVEL 1: the block ONE Rest position covers ───────────────────────────
  // NOT the permuted tile. `permutation_mnk` folds into Rest, so a permuted
  // tile can hold SEVERAL Rest positions — drawing the tile here would make
  // grid 2 count too few of them, which is exactly the bug this replaced:
  // B with permutation_mnk = (32,32,16) has a 32x16 tile but Rest 8x2, and the
  // grid said 4x2. The block is derived, never assumed: gather every
  // (thread, value) cell at Rest 0 and see what region they form.
  const blockCells = new Map();          // compact flat position -> entries
  for (let t = 0; t < nThr; t++) {
    const tOff = opPos.thr.call(t);
    for (let v = 0; v < frg; v++) {
      const pos = tOff + opPos.val.call(v);
      if (!blockCells.has(pos)) blockCells.set(pos, []);
      blockCells.get(pos).push({ t, v, w: Math.floor(t / MTM_ATOM_THREADS) });
    }
  }
  // Compact the rows and columns the block actually occupies. A permutation
  // LAYOUT interleaves them — the block is then 32 rows out of a 48-row span —
  // so consecutive-ness is reported rather than assumed.
  const rowsUsed = [...new Set([...blockCells.keys()].map(p => p % M0))].sort((a, b) => a - b);
  const colsUsed = [...new Set([...blockCells.keys()].map(p => Math.floor(p / M0) % M1))].sort((a, b) => a - b);
  const rowAt = new Map(rowsUsed.map((r, i) => [r, i]));
  const colAt = new Map(colsUsed.map((c, i) => [c, i]));
  const blockExtents = [rowsUsed.length, colsUsed.length];
  const blockGrid = Array.from({ length: blockExtents[0] }, () =>
    Array.from({ length: blockExtents[1] }, () => ({ entries: [], rest: 0 })));
  for (const [pos, entries] of blockCells) {
    blockGrid[rowAt.get(pos % M0)][colAt.get(Math.floor(pos / M0) % M1)].entries = entries;
  }
  const blockConsecutive =
    rowsUsed[rowsUsed.length - 1] - rowsUsed[0] + 1 === rowsUsed.length &&
    colsUsed[colsUsed.length - 1] - colsUsed[0] + 1 === colsUsed.length;

  // ── LEVELS 2 and 3: the returned layout's Rest modes, verbatim ────────────
  const sweepExtents = [product(opT.restX.shape), product(opT.restY.shape)];
  const extraExtents = opT.restExtra.map(l => product(l.shape));
  const sweepSize = sweepExtents[0] * sweepExtents[1];
  // Rest index r sits at value index frg*r, since `val` is (FrgV, Rest...)
  // col-major. That is the offset the k-loop / tile index advances by.
  const restPosAt = (r) => opPos.val.call(frg * r);
  const restOffAt = (r) => opT.val.call(frg * r);

  return {
    which, spec, thrIdx, threads: overTile.threads, warps: overTile.warps,
    atomLayoutMNK: overTile.atomLayoutMNK, tileMNK: overTile.tileMNK,
    thrLayoutVmnk: overTile.thrVmnk, divExtents,
    tensorExtents: tExt,
    partition: opT.partition,
    // What the thread actually declares in registers.
    fragment: pabcMakeFragment(opT.partition, which),
    fragColMajor: formatLayoutStr(pabcMakeFragment(opT.partition, which).shape,
                                  pabcMakeFragment(opT.partition, which).stride) ===
                  formatLayoutStr(opT.partition.shape, mtmCompactStride(opT.partition.shape)),
    // null when no thread is selected. The LAYOUT is thread-independent —
    // `thrfrg_X`'s thread mode is a separate mode, which is why every thread of
    // a TiledMMA gets the same static fragment type — so only this is per-thread.
    baseOffset: thrIdx === null ? null : opT.thr.call(thrIdx),
    // The three grids' own layouts are literally mode 0, modes 1-2 and modes 3+
    // of the returned layout, so the three headers read as its decomposition.
    frgLayout:   opT.partition.mode(0),
    sweepLayout: make_layout(opT.restX, opT.restY),
    extraLayout: opT.restExtra.length ? make_layout(...opT.restExtra) : null,
    blockGrid, blockExtents, blockConsecutive, blockSpan: [
      rowsUsed[rowsUsed.length - 1] - rowsUsed[0] + 1,
      colsUsed[colsUsed.length - 1] - colsUsed[0] + 1,
    ],
    sweepExtents, extraExtents, sweepSize,
    extraSize: extraExtents.reduce((a, b) => a * b, 1),
    restPosAt, restOffAt,
  };
}

// ═══════════════════════════════════════════════════════
//  The tab
// ═══════════════════════════════════════════════════════

/** The four input sections this tab takes, and the operand toggle above them.
 *  Still parameterized by `p` / `fn` rather than inlined: it was shared with a
 *  second tab, and the next MMA tab to take a TiledMMA plus a tensor will want
 *  the same 80 lines of markup rather than a near-copy of them.
 *
 *  `p` is the id prefix ('pabc'); `fn` names the call in the toggle's buttons
 *  ('partition'). */
function pabcInputSections({ id, p, fn, render, setOp, setOperand }) {
  return `
        <div class="form-group">
          <label>Which operand</label>
          <div class="seg-control seg-stacked" id="${id}-${p}-operand-btns">
            <button class="mode-btn active" onclick="${setOperand}('${id}', 'A')">${fn}_A</button>
            <button class="mode-btn" onclick="${setOperand}('${id}', 'B')">${fn}_B</button>
            <button class="mode-btn" onclick="${setOperand}('${id}', 'C')">${fn}_C</button>
          </div>
          <div class="hint-inline" id="${id}-${p}-operand-hint"></div>
        </div>

        <details class="cuo-section" open>
          <summary>1. The MMA Op</summary>
          <div class="cuo-section-body">
            <div class="form-group">
              <label>MmaAtom type<span style="color:#6b7280;font-weight:normal">&nbsp;&mdash; warp-level (cute.nvgpu.warp) only</span></label>
              <select id="${id}-${p}-op-input" onchange="${setOp}('${id}')">
                <option value="f16bf16" selected>warp.MmaF16BF16Op</option>
                <option value="tf32">warp.MmaTF32Op</option>
                <option value="fp8">warp.MmaFP8Op</option>
              </select>
            </div>
            <div class="form-group" id="${id}-${p}-ab-group">
              <label>ab_dtype</label>
              <select id="${id}-${p}-ab-input" onchange="${render}('${id}')"></select>
            </div>
            <div class="form-group" id="${id}-${p}-acc-group">
              <label>acc_dtype</label>
              <select id="${id}-${p}-acc-input" onchange="${render}('${id}')"></select>
            </div>
            <div class="form-group">
              <label>shape_mnk<span style="color:#6b7280;font-weight:normal">&nbsp;&mdash; M and N are fixed at 16x8; only K varies</span></label>
              <select id="${id}-${p}-k-input" onchange="${render}('${id}')"></select>
            </div>
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>2. The TiledMMA &mdash; the only source of a ThrMMA</summary>
          <div class="cuo-section-body">
            ${layoutInputField({
              id: `${id}-${p}-atomlayout-input`, label: 'atom_layout_mnk', value: '(2, 2, 1)',
              placeholder: '(1, 1, 1)',
              hint: 'Rank 3 &mdash; how many warps along M, N and K. Blank = (1, 1, 1), CuTeDSL\'s own default.',
            })}
            ${layoutInputField({
              id: `${id}-${p}-perm-input`, label: 'permutation_mnk', value: '',
              placeholder: '(_, _, _)',
              hint: 'Rank-3 Tiler. A number is a plain extent; <code>(2,16):(16,1)</code> genuinely permutes; <code>_</code> or blank leaves the mode alone.',
            })}
            <div id="${id}-${p}-tiled-result" class="cuo-result"></div>
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>3. get_slice(thr_idx)</summary>
          <div class="cuo-section-body">
            <div class="form-group">
              <label>Thread index</label>
              <input type="text" id="${id}-${p}-thr-input" value="5" placeholder="all threads">
              <div class="hint-inline">
                A ThrMMA is one thread's view. Leave this <b>empty</b> to show every
                thread instead: the layout <code>${fn}_A</code> returns is the same
                for all of them, so a thread id only changes which cells are highlighted
                and where the fragment starts. Threads are numbered across the whole
                TiledMMA &mdash; warp <i>w</i> owns <code>32w .. 32w+31</code>.
              </div>
            </div>
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>4. The tensor being partitioned</summary>
          <div class="cuo-section-body">
            ${layoutInputField({
              id: `${id}-${p}-tensor-input`,
              // Backticks, not concatenation: `${p}` has to interpolate here, and a
              // single-quoted fragment would ship the literal text `${p}` into the id.
              label: `The <span id="${id}-${p}-tensor-axes">(M, K)</span> tensor &mdash; ` +
                     `the argument to <span id="${id}-${p}-tensor-fn">${fn}_A</span>`,
              hint: 'a plain tensor, NOT pre-divided; rank 2, 3 or 4',
              value: '(64, 48):(48, 1)'
            })}
          </div>
        </details>`;
}

function generatePartitionABCTabContent(id) {
  return `
    <!-- partition_A / partition_B / partition_C panel -->
    <div id="${id}-tab-partition_abc" class="panel">
      <div class="controls">
        <h2>partition_A / B / C</h2>

${pabcInputSections({ id, p: 'pabc', fn: 'partition', render: 'renderPartitionABC', setOp: 'setPabcOp', setOperand: 'setPabcOperand' })}

        ${statusDivs(`${id}-pabc`)}
        <button class="btn btn-render" onclick="renderPartitionABC('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-pabc-export" onclick="exportPABC('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setPABC('${id}','A','f16bf16','half_t','float',16,'(2, 2, 1)','','5','(64, 48):(48, 1)')">A over a 2&times;3 tiling &mdash; the canonical 4-warp GEMM</button>
            <button class="preset-btn" onclick="setPABC('${id}','B','f16bf16','half_t','float',16,'(2, 2, 1)','','5','(32, 48):(48, 1)')">B &mdash; (N, K), broadcast across the M-warps</button>
            <button class="preset-btn" onclick="setPABC('${id}','C','f16bf16','half_t','float',16,'(2, 2, 1)','','5','(64, 32):(32, 1)')">C &mdash; (M, N), the accumulator</button>
            <button class="preset-btn" onclick="setPABC('${id}','A','f16bf16','half_t','float',16,'(2, 2, 1)','','5','(64, 48, 4):(48, 1, 3072)')">A with a k-loop &mdash; grid 3 is a strip of 4</button>
            <button class="preset-btn" onclick="setPABC('${id}','A','f16bf16','half_t','float',16,'(2, 2, 1)','(64, 32, 32)','1','(128, 96):(96, 1)')">permutation_mnk (64,32,32) &mdash; the tile is 64&times;32 but a Rest block is 32&times;16</button>
            <button class="preset-btn" onclick="setPABC('${id}','A','f16bf16','half_t','float',16,'(2, 2, 1)','((2,32):(32,1), 32, 16)','2','(128, 48):(48, 1)')">Permuting layout &mdash; Rest mode 0 comes back a TUPLE</button>
            <button class="preset-btn" onclick="setPABC('${id}','C','f16bf16','half_t','float',16,'(2, 2, 2)','','3','(64, 32):(32, 1)')">C with K-warps &mdash; several warps hold partial sums of a cell</button>
            <button class="preset-btn" onclick="setPABC('${id}','A','tf32','na','na',8,'(2, 2, 1)','','0','(64, 32):(32, 1)')">MmaTF32Op, m16n8k8</button>
          </div>
        </div>

        <div class="hint">
          <b>The same three levels as <code>partition_S</code>, and the same
          reason they are drawn separately:</b> <code>thrfrg_X</code>'s thread and
          value modes do not depend on its Rest mode, so every tile is
          partitioned by the identical grid <b>1</b> and every slice in grid
          <b>3</b> holds the identical grid <b>2</b>. Drawing the tensor at
          element resolution would repeat one picture dozens of times.<br><br>
          <b>The tensor is NOT pre-divided.</b> <code>thrfrg_A</code> runs
          <code>logical_divide(atensor, (permM, permK))</code> and then
          <code>zipped_divide</code> by the atom itself
          (<code>mma_atom.hpp:291</code>). Modes past the operand's two axes are
          never tiled &mdash; they ride along and multiply Rest, which is how one
          TiledMMA partitions <code>gA</code> of shape
          <code>(BLK_M, BLK_K, k)</code>.<br><br>
          <b>The three grids are the returned layout, split.</b> Their blue
          headers are mode 0, modes 1&ndash;2 and modes 3+ of
          <code>partition_X</code>'s result and concatenate back to it &mdash;
          <code>(2,2)</code> + <code>(8,2)</code> + <code>(4)</code> is
          <code>((2,2),8,2,4)</code>. So grid <b>2</b> always shows the Rest the
          call actually returns.<br><br>
          <b>Where this differs from <code>partition_S</code>:
          <code>permutation_mnk</code> folds into those Rest modes.</b> A copy's
          <code>Tiler_MN</code> gives one Rest entry per tile; an MMA's
          permutation does not, because the warp pattern can repeat
          <em>inside</em> the permuted tile. So the unit grid <b>1</b> draws is
          <em>not</em> the TiledMMA's tile &mdash; it is the block one Rest
          position covers, gathered from the derivation rather than assumed.
          With <code>permutation_mnk = (32,32,16)</code> a B tile is
          <code>32&times;16</code> while a Rest block is <code>16&times;16</code>,
          and grid 2 correctly shows <code>8&times;2</code>, not
          <code>4&times;2</code>. Grid 2's header says so whenever the two
          differ.<br>
          A permuting <em>layout</em> does one more thing: it interleaves the
          rows. <code>((2,32):(32,1),32,16)</code> on a <code>(128,48)</code> A
          gives <code>((2,2,2),(2,2),3)</code> &mdash; Rest mode 0 comes back a
          <b>tuple</b>, <code>(repeat inside the tile, blocks across the
          tensor)</code> &mdash; and the block's rows are no longer consecutive
          in the tensor, which grid 1's header reports.<br><br>
          <b>Rest arrives at top level, not nested.</b> <code>partition_A</code>
          slices with <code>(thr_vmk, (_, repeat&lt;rank&gt;(_)))</code> and
          CuTe's <code>slice</code> splices a sliced tuple mode into its parent
          &mdash; which is why CUTLASS's GEMMs write <code>tCgA(_,_,_,k)</code>.
          <br><br>
          <b>What the thread holds in REGISTERS is one more step.</b>
          <code>make_fragment_A(partition_A(t))</code> takes this tab's output
          and gives the register array &mdash; a separate question, since the
          fragment's mode order follows the <em>source's</em> majorness. Not
          drawn here; CuTeDSL exposes it as
          <code>MmaAtom.make_fragment_A</code>.<br><br>
          See the <b>make_tiled_mma</b> tab for all six grids of the TiledMMA
          itself; this tab is one operand of it, against a tensor.
        </div>
      </div>

      <div class="comp-results" style="grid-template-columns:1fr">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-pabc-tile-title">1. The TiledMMA over one tile</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-pabc-tile-mode-btns">
                <button class="mode-btn" onclick="setPabcMode('${id}','tile','value')">value</button>
              </span>
              <button class="mode-btn" id="${id}-pabc-tile-svg-zoom" onclick="toggleZoom('${id}-pabc-tile-svg')">Zoom in</button>
              <button class="mode-btn" onclick="downloadSVG('${id}-pabc-tile-svg', 'partition_abc_tile.svg')">Download SVG</button>
            </span>
          </div>
          <div class="cuo-viz-desc">
            <b>Level 1 &mdash; one thread's work inside one Rest block.</b>
            Drawn by <b>make_tiled_mma</b>'s own builder, with the selected
            thread in colour and every other cell greyed but still labelled.
            Several <code>T</code>/<code>V</code> entries in one cell are a
            broadcast (A across the N-warps, B across the M-warps) or, for C,
            warps holding partial sums of the same accumulator. This is the block
            <em>one Rest position covers</em>, gathered from the derivation
            &mdash; with a <code>permutation_mnk</code> that repeats, it is
            smaller than the TiledMMA's tile. The layout in <b>blue</b> is mode 0
            of the returned layout: what one thread holds here.
          </div>
          <div class="viz-box"><div id="${id}-pabc-tile-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-pabc-sweep-title">2. The tile over the operand's plane</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-pabc-sweep-mode-btns">
                <button class="mode-btn" onclick="setPabcMode('${id}','sweep','value')">value</button>
              </span>
              <button class="mode-btn" id="${id}-pabc-sweep-svg-zoom" onclick="toggleZoom('${id}-pabc-sweep-svg')">Zoom in</button>
              <button class="mode-btn" onclick="downloadSVG('${id}-pabc-sweep-svg', 'partition_abc_sweep.svg')">Download SVG</button>
            </span>
          </div>
          <div class="cuo-viz-desc">
            <b>Level 2 &mdash; how that block covers the operand's two axes.</b>
            One cell per Rest position &mdash; modes 1 and 2 of the returned
            layout, verbatim &mdash; labelled with its coordinate and the block's
            origin in the tensor; <code>value</code> swaps the origin for the
            offset. When <code>permutation_mnk</code> makes the warp pattern
            repeat inside the TiledMMA's tile, that repeat is one of these cells,
            and the header says so.
          </div>
          <div class="viz-box"><div id="${id}-pabc-sweep-svg"></div></div>
        </div>
        <div class="comp-viz-item" id="${id}-pabc-extra-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-pabc-extra-title">3. The plane over the whole tensor</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-pabc-extra-mode-btns">
                <button class="mode-btn" onclick="setPabcMode('${id}','extra','value')">value</button>
              </span>
              <button class="mode-btn" id="${id}-pabc-extra-svg-zoom" onclick="toggleZoom('${id}-pabc-extra-svg')">Zoom in</button>
              <button class="mode-btn" onclick="downloadSVG('${id}-pabc-extra-svg', 'partition_abc_extra.svg')">Download SVG</button>
            </span>
          </div>
          <div class="cuo-viz-desc">
            <b>Level 3 &mdash; how that whole plane covers the tensor.</b> One
            cell per tensor mode past the operand's two axes: modes nothing ever
            tiled. In a GEMM these are the <b>k-loop</b> and the <b>pipeline
            stage</b>, so there is one (drawn as a row, since a k-loop reads
            horizontally) or two, and this panel is hidden when the tensor is
            exactly the plane above.
          </div>
          <div class="viz-box"><div id="${id}-pabc-extra-svg"></div></div>
        </div>
      </div>
    </div>`;
}

const pabcState = {};

/** Read the shared input sections. Both tabs take exactly these, so reading
 *  them twice would be two places for a validation to drift.
 *
 *  Neither tab calls `updateRankWarning`, and for two reasons rather than one:
 *  `atom_layout_mnk` and `permutation_mnk` are rank-3 by definition and are not
 *  drawn as grids (make_tiled_mma's reason), and the tensor is drawn in full
 *  across the grids (partition_sd's reason). There is nothing it could
 *  truthfully say, so the warning box carries real diagnostics instead. */
function pabcReadInputs(tabId, p) {
  const opKey = document.getElementById(`${tabId}-${p}-op-input`).value;
  const op = MMA_WARP_OPS[opKey] || MMA_WARP_OPS.f16bf16;
  mmaSyncControls(tabId, opKey, p);          // options BEFORE values are read

  const k = parseInt(document.getElementById(`${tabId}-${p}-k-input`).value, 10);
  const abDtype  = op.ab  ? document.getElementById(`${tabId}-${p}-ab-input`).value  : 'tfloat32_t';
  const accDtype = op.acc ? document.getElementById(`${tabId}-${p}-acc-input`).value : 'float';
  if (op.ab && abDtype === 'bfloat16_t' && accDtype !== 'float')
    throw new Error(
      `${op.label} requires acc_dtype = float when ab_dtype is bfloat16_t ` +
      `(there is no bf16 MMA with an f16 accumulator).`);

  const atom       = mmaWarpAtom(opKey, k);
  const atomLayout = mtmParseAtomLayout(document.getElementById(`${tabId}-${p}-atomlayout-input`).value);
  const perm       = mtmParsePerm(document.getElementById(`${tabId}-${p}-perm-input`).value);

  const tP = parseLayout(document.getElementById(`${tabId}-${p}-tensor-input`).value);
  const tS = stripTrivialTrailing(tP.shape, tP.stride);
  const tensor = new Layout(tS.shape, tS.stride);

  // Blank means "every thread" — see the note on baseOffset in
  // pabcComputePartition for why that is a truthful picture and not a fiction.
  const thrStr = (document.getElementById(`${tabId}-${p}-thr-input`).value || '').trim();
  if (thrStr !== '' && !/^\d+$/.test(thrStr))
    throw new Error(
      `Thread index must be a whole number, or empty for all threads — got "${thrStr}".`);

  return { opKey, op, k, abDtype, accDtype, atom, atomLayout, perm, tensor,
           thrIdx: thrStr === '' ? null : parseInt(thrStr, 10) };
}

function renderPartitionABC(tabId) {
  showErr(`${tabId}-pabc-error`, '');
  showWarn(`${tabId}-pabc-warning`, '');
  const prev = pabcState[tabId] || {};
  const which = prev.which || 'A';
  pabcSyncOperandField(tabId, which);
  try {
    const inp = pabcReadInputs(tabId, 'pabc');
    const r = pabcComputePartition(inp.atom, inp.atomLayout, inp.perm, which, inp.tensor, inp.thrIdx);
    const { abDtype, accDtype, thrIdx } = inp;

    document.getElementById(`${tabId}-pabc-tiled-result`).innerHTML =
      `<div class="cuo-result-line">thr_layout_vmnk = <b>` +
      `${formatLayoutStr(r.thrLayoutVmnk.shape, r.thrLayoutVmnk.stride)}</b> &mdash; ` +
      `${r.warps} warp${r.warps === 1 ? '' : 's'}, ${r.threads} threads</div>` +
      `<div class="cuo-result-line">Tile Shape MNK = <b>(${r.tileMNK.join(', ')})</b>` +
      `<span style="color:#9ca3af"> &rarr; ${which} divides by ` +
      `${r.divExtents.join('&times;')} (${r.spec.axes.join(', ')})</span></div>`;

    const prevState = pabcState[tabId] || {};
    pabcState[tabId] = {
      ...r, which, ...inp,
      dtype: which === 'C' ? accDtype : abDtype,
      tileMode:  (prevState.tileMode  instanceof Set) ? prevState.tileMode  : new Set(),
      sweepMode: (prevState.sweepMode instanceof Set) ? prevState.sweepMode : new Set(),
      extraMode: (prevState.extraMode instanceof Set) ? prevState.extraMode : new Set(),
    };
    pabcRenderTileViz(tabId);
    pabcRenderSweepViz(tabId);
    pabcRenderExtraViz(tabId);
    updateOuterTabLabel(tabId,
      `${r.spec.fn}:${formatLayoutStr(r.partition.shape, r.partition.stride)}`);
  } catch (e) {
    showErr(`${tabId}-pabc-error`, e.message);
    for (const g of ['tile', 'sweep', 'extra']) {
      const el = document.getElementById(`${tabId}-pabc-${g}-svg`);
      if (el) el.innerHTML = '';
    }
  }
}

/** LEVEL 1 — the block one Rest position covers, drawn by make_tiled_mma's own
 *  builder so the two tabs cannot drift. Block-LOCAL coordinates: when a
 *  permutation layout interleaves the rows, the block is still a rectangle in
 *  its own right and the header says which rows it really occupies. */
function pabcRenderTileViz(tabId) {
  const s = pabcState[tabId];
  if (!s) return;
  const modes = s.tileMode instanceof Set ? s.tileMode : new Set();
  const [M, N] = s.blockExtents;
  const holes = s.blockGrid.reduce((n, row) => n + row.filter(c => !c.entries.length).length, 0);

  const svg = mtmBuildSVG(s.blockGrid, M, N, {
    mode: 'tv', focus: s.thrIdx, dimRest: false,
    sum: s.spec.sum, showValue: modes.has('value'),
  });

  document.getElementById(`${tabId}-pabc-tile-svg`).innerHTML =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `<b style="color:#93c5fd">${formatLayoutStr(s.frgLayout.shape, s.frgLayout.stride)}</b>` +
    ` per thread &mdash; block ${M}&times;${N} (${s.spec.axes.join(', ')}) of ${s.dtype}, ` +
    `${s.threads} threads, ` +
    (s.thrIdx === null ? `every thread shown` : `T${s.thrIdx} highlighted`) +
    (s.blockConsecutive ? '' :
      ` &mdash; <b style="color:#fbbf24">interleaved</b>: these ${M} ` +
      `${s.spec.axes[0]} rows are spread over ${s.blockSpan[0]} of the tensor's, because ` +
      `permutation_mnk reorders them`) +
    (holes ? ` &mdash; <b style="color:#f87171">${holes} cell${holes === 1 ? '' : 's'} ` +
             `claimed by nobody</b>` : '') +
    `</div>` + svg;
  applyZoomState(`${tabId}-pabc-tile-svg`);
  updateModeBtns(`${tabId}-pabc-tile-mode-btns`, modes);
  document.getElementById(`${tabId}-pabc-tile-title`).textContent =
    `1. ${s.which} over one block (${M}×${N})`;
}

/** LEVEL 2 — the returned layout's first two Rest modes, one cell each. */
function pabcRenderSweepViz(tabId) {
  const s = pabcState[tabId];
  if (!s) return;
  const modes = s.sweepMode instanceof Set ? s.sweepMode : new Set();
  const [R0, R1] = s.sweepExtents;

  const svg = buildColoredLayoutSVG([R0, R1], [1, R0], modes, (i, j) => {
    const r = i + j * R0;
    const lines = [`(${i},${j})`];
    lines.push(modes.has('value')
      ? `@${s.restOffAt(r)}`
      : `(${psdRestCoord(s.restPosAt(r), s.tensorExtents).slice(0, 2).join(',')})`);
    return { bg: colorHighlight(r), stroke: '#1e3a5f', sw: 1.5, text: lines };
  });

  document.getElementById(`${tabId}-pabc-sweep-svg`).innerHTML =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `<b style="color:#93c5fd">${formatLayoutStr(s.sweepLayout.shape, s.sweepLayout.stride)}</b>` +
    ` &mdash; ${R0}&times;${R1} blocks of ${s.blockExtents.join('&times;')} over the ` +
    `${s.tensorExtents.slice(0, 2).join('&times;')} plane; second line is the block's ` +
    (modes.has('value') ? `offset` : `origin`) +
    (s.divExtents[0] !== s.blockExtents[0] || s.divExtents[1] !== s.blockExtents[1]
      ? `<br>permutation_mnk makes the TiledMMA's tile ${s.divExtents.join('&times;')}, ` +
        `bigger than the ${s.blockExtents.join('&times;')} one Rest position covers &mdash; ` +
        `the warp pattern repeats inside it, and that repeat is part of this grid.`
      : '') +
    (s.extraSize > 1 ? `<br>The other ${s.extraSize - 1} slice` +
                       `${s.extraSize === 2 ? '' : 's'} look identical &mdash; see 3.` : '') +
    `</div>` + svg;
  applyZoomState(`${tabId}-pabc-sweep-svg`);
  updateModeBtns(`${tabId}-pabc-sweep-mode-btns`, modes);
  document.getElementById(`${tabId}-pabc-sweep-title`).textContent =
    `2. The block over the ${s.spec.axes.join(', ')} plane — ${R0 * R1} position${R0 * R1 === 1 ? '' : 's'}`;
}

/** LEVEL 3 — the Rest modes past the operand's two axes. Same shape and same
 *  reasoning as the partition_S/D tab's, including a single mode being a row. */
function pabcRenderExtraViz(tabId) {
  const s = pabcState[tabId];
  if (!s) return;
  const modes = s.extraMode instanceof Set ? s.extraMode : new Set();
  const E0 = s.extraExtents.length > 0 ? s.extraExtents[0] : 1;
  const E1 = s.extraExtents.length > 1 ? s.extraExtents[1] : 1;
  const oneD = s.extraExtents.length <= 1;
  const rows = oneD ? 1 : E0;
  const cols = oneD ? E0 : E1;

  const svg = buildColoredLayoutSVG([rows, cols], [1, rows], modes, (m, n) => {
    const e = oneD ? n : m + n * E0;
    const lines = [oneD ? `(${e})` : `(${m},${n})`];
    if (modes.has('value')) lines.push(`@${s.restOffAt(e * s.sweepSize)}`);
    return { bg: colorHighlight(e), stroke: '#1e3a5f', sw: 1.5, text: lines };
  });

  document.getElementById(`${tabId}-pabc-extra-svg`).innerHTML =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    (s.extraSize > 1
      ? `<b style="color:#93c5fd">${formatLayoutStr(s.extraLayout.shape, s.extraLayout.stride)}</b>` +
        ` &mdash; tensor mode${s.extraExtents.length === 1 ? ' ' : 's '}` +
        `${s.extraExtents.map((_, k) => k + 2).join(', ')}, never tiled; each holds the whole ` +
        `${s.sweepExtents.join('&times;')}-block plane above`
      : `no modes past the operand's two axes &mdash; the tensor is exactly the plane above`) +
    `</div>` + svg;
  applyZoomState(`${tabId}-pabc-extra-svg`);
  updateModeBtns(`${tabId}-pabc-extra-mode-btns`, modes);
  const item = document.getElementById(`${tabId}-pabc-extra-item`);
  if (item) item.style.display = s.extraSize > 1 ? '' : 'none';
  document.getElementById(`${tabId}-pabc-extra-title`).textContent =
    `3. The plane over the whole tensor — ${s.extraSize} slice${s.extraSize === 1 ? '' : 's'}`;
}

/** Relabel the operand controls. One toggle, one tensor box, so the box's label
 *  has to move with it or it would name the wrong argument. */
function pabcSyncOperandField(tabId, which) {
  const spec = PABC_OPERAND[which] || PABC_OPERAND.A;
  const hint = document.getElementById(`${tabId}-pabc-operand-hint`);
  if (hint) {
    hint.innerHTML =
      `<code>${spec.fn}</code> partitions the <b>(${spec.axes.join(', ')})</b> operand using ` +
      `the atom's <code>tv_layout_${which}</code>. Unlike a Copy Atom's src/dst, the three ` +
      `MMA operands are over three <em>different</em> tiles, which is why they are one ` +
      `control rather than three tabs.`;
  }
  const axesEl = document.getElementById(`${tabId}-pabc-tensor-axes`);
  if (axesEl) axesEl.textContent = `(${spec.axes.join(', ')})`;
  const fnEl = document.getElementById(`${tabId}-pabc-tensor-fn`);
  if (fnEl) fnEl.textContent = spec.fn;
  const group = document.getElementById(`${tabId}-pabc-operand-btns`);
  if (group) group.querySelectorAll('.mode-btn').forEach(b =>
    b.classList.toggle('active', b.textContent.trim() === spec.fn));
}

function setPabcOperand(tabId, which) {
  pabcState[tabId] = Object.assign(pabcState[tabId] || {}, { which });
  pabcSyncOperandField(tabId, which);
  renderPartitionABC(tabId);
}

function setPabcMode(tabId, grid, mode) {
  const s = pabcState[tabId];
  if (!s) return;
  const key = { tile: 'tileMode', sweep: 'sweepMode', extra: 'extraMode' }[grid];
  let modes = s[key];
  if (!(modes instanceof Set)) { modes = new Set(); s[key] = modes; }
  if (modes.has(mode)) modes.delete(mode); else modes.add(mode);
  if (grid === 'tile') pabcRenderTileViz(tabId);
  else if (grid === 'sweep') pabcRenderSweepViz(tabId);
  else pabcRenderExtraViz(tabId);
}

function setPabcOp(tabId) {
  mmaSyncControls(tabId, document.getElementById(`${tabId}-pabc-op-input`).value, 'pabc');
  renderPartitionABC(tabId);
}

function setPABC(tabId, which, opKey, ab, acc, k, atomLayout, perm, thr, tensor) {
  pabcState[tabId] = Object.assign(pabcState[tabId] || {}, { which });
  document.getElementById(`${tabId}-pabc-op-input`).value = opKey;
  mmaSyncControls(tabId, opKey, 'pabc');    // options BEFORE values, or the assign is a no-op
  if (ab  && ab  !== 'na') document.getElementById(`${tabId}-pabc-ab-input`).value  = ab;
  if (acc && acc !== 'na') document.getElementById(`${tabId}-pabc-acc-input`).value = acc;
  document.getElementById(`${tabId}-pabc-k-input`).value = String(k);
  document.getElementById(`${tabId}-pabc-atomlayout-input`).value = atomLayout;
  document.getElementById(`${tabId}-pabc-perm-input`).value = perm;
  document.getElementById(`${tabId}-pabc-thr-input`).value = thr;
  document.getElementById(`${tabId}-pabc-tensor-input`).value = tensor;
  renderPartitionABC(tabId);
}

function exportPABC(tabId) {
  // `na` is how exportMTM spells an absent field: the key is split on `-`, so
  // an empty part cannot travel.
  exportURL(`${tabId}-pabc-export`, 'partition_abc',
    (pabcState[tabId] && pabcState[tabId].which) || 'A',
    document.getElementById(`${tabId}-pabc-op-input`).value,
    document.getElementById(`${tabId}-pabc-ab-input`).value || 'na',
    document.getElementById(`${tabId}-pabc-acc-input`).value || 'na',
    document.getElementById(`${tabId}-pabc-k-input`).value,
    document.getElementById(`${tabId}-pabc-atomlayout-input`).value,
    document.getElementById(`${tabId}-pabc-perm-input`).value || 'na',
    document.getElementById(`${tabId}-pabc-thr-input`).value,
    document.getElementById(`${tabId}-pabc-tensor-input`).value);
}
