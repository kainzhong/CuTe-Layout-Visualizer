// make_copy_atom tab (COPY scope): build ONE Copy_Atom and visualize it.
// Functions become globals on `window` (no module system).
//
// Mirrors `cute.make_copy_atom(op, dtype, **kwargs)` — the two-step shape the
// DSL uses (python/CuTeDSL/cutlass/cute/atom.py:979):
//
//   op   = cute.nvgpu.CopyUniversalOp()                    <- step 1, Op + its own params
//   atom = cute.make_copy_atom(op, dtype, num_bits_per_copy=N)   <- step 2
//
// The Op's constructor params and make_copy_atom's arguments are different
// things and the form keeps them apart. CopyUniversalOp is a bare marker with
// no fields; cpasync.CopyG2SOp carries only a defaulted `cache_mode`;
// warp.LdMatrix8x8x16bOp takes `num_matrices` and `transpose` and is the reason
// the params area renders controls rather than a sentence.
//
// Copy_Atom<UniversalCopy<S>, ValType> (include/cute/atom/copy_traits.hpp:65-78):
//   ThrID        = Layout<_1>
//   ValLayoutSrc = ValLayoutDst = (1, N):(0, 1) with N = num_bits / sizeof_bits(ValType)
//
// Copy_Atom<SM75_U32x{1,2,4}_LDSM_{N,T}, ValType>:
//   ThrID        = Layout<_32>     — warp-collective, always all 32 lanes
//   ValLayoutSrc != ValLayoutDst   — the instruction PERMUTES, which is its point
//
// Replicating an atom over a thread tile is a separate concern — see the
// make_tiled_copy / make_tiled_copy_tv tabs, alongside this one in Copy.

// Copy Ops this tab can build. `params` lists the Op's own constructor
// parameters; `kind` selects the derivation and the visualization.
const MCA_OPS = {
  // The six SIMT Ops come from ui.js's SIMT_COPY_OPS, which the make_tiled_copy
  // and make_tiled_copy_tv tabs read too — one table, three pickers. They all
  // produce a byte-identical Atom (ThrID 1:0, ValLayoutSrc == ValLayoutDst),
  // hence the shared `kind: 'simt'` render path.
  ...Object.fromEntries(Object.entries(SIMT_COPY_OPS).map(([k, o]) =>
    [k, { ...o, kind: 'simt', params: [] }])),
  // The LdMatrix family. All three are warp-collective SMEM→RMEM loads whose
  // source and destination layouts differ; MCA_LDSM_SPECS below carries the
  // geometry and the parameter domains, so an entry here is just the label plus
  // the key into that table.
  ldmatrix: {
    label: 'warp.LdMatrix8x8x16bOp',
    ctor: 'cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose, num_matrices)',
    kind: 'ldmatrix',
    ldsm: 'ldsm8x8x16b',
    cpasync: false,
    params: ['transpose', 'num_matrices'],
    note: 'The 8x8 <code>.m8n8</code> form, over 16-bit units. The only member of the ' +
          'family whose <code>transpose</code> is optional.',
  },
  ldmatrix16x8x8b: {
    label: 'warp.LdMatrix16x8x8bOp',
    ctor: 'cute.nvgpu.warp.LdMatrix16x8x8bOp(transpose, num_matrices, unpack_bits=None)',
    kind: 'ldmatrix',
    ldsm: 'ldsm16x8x8b',
    cpasync: false,
    params: ['transpose', 'num_matrices'],
    note: 'No direct PTX form &mdash; it lowers to <code>.m16n16</code> plus address and ' +
          'value permutations chosen to match <code>stmatrix.m16n8.trans</code>, which is ' +
          'what makes its <code>ValLayoutSrc</code> a rank-4 thread mode. Useful for ' +
          'vectorizing against Ampere-style 8x8 thread-value layouts.',
  },
  ldmatrix8x16x8b: {
    label: 'warp.LdMatrix8x16x8bOp',
    ctor: 'cute.nvgpu.warp.LdMatrix8x16x8bOp(transpose=False, num_matrices, unpack_bits)',
    kind: 'ldmatrix',
    ldsm: 'ldsm8x16x8b',
    cpasync: false,
    params: ['num_matrices'],
    note: 'PTX <code>.m8n16</code> with <code>.b8x16.b4x16_p64</code> / ' +
          '<code>.b8x16.b6x16_p32</code> (SM100+): each 16 B SMEM row holds 16 packed ' +
          '4- or 6-bit values plus padding, unpacked into 8-bit register slots &mdash; how ' +
          'FP4/FP6 operands reach registers. Not a TMEM copy. Its ownership layouts equal ' +
          '<code>LdMatrix8x8x16bOp</code>\'s for every dtype, <code>num_matrices</code> and ' +
          '<code>unpack_bits</code> (CuTeDSL builds exactly that atom when ' +
          '<code>unpack_bits=None</code>), so the picture is the same; <code>unpack_bits</code> ' +
          'only selects the packed source format. <code>transpose</code> is rejected.',
  },
  // The StMatrix family: warp-collective RMEM->SMEM stores, ldmatrix's mirror.
  // Same geometry table (MCA_LDSM_SPECS), with `store: true` swapping which
  // side carries the SMEM row addresses.
  stmatrix: {
    label: 'warp.StMatrix8x8x16bOp',
    ctor: 'cute.nvgpu.warp.StMatrix8x8x16bOp(transpose, num_matrices)',
    kind: 'ldmatrix',
    ldsm: 'stsm8x8x16b',
    cpasync: false,
    params: ['transpose', 'num_matrices'],
    note: 'The 8x8 <code>.m8n8</code> store over 16-bit units: the mirror of ' +
          '<code>LdMatrix8x8x16bOp</code>. Its ValLayoutSrc is the ldmatrix ValLayoutDst ' +
          'and vice versa, for every dtype and parameter. This is the RMEM &rarr; SMEM ' +
          'store an SM90 epilogue uses.',
  },
  stmatrix16x8x8b: {
    label: 'warp.StMatrix16x8x8bOp',
    ctor: 'cute.nvgpu.warp.StMatrix16x8x8bOp(transpose, num_matrices)',
    kind: 'ldmatrix',
    ldsm: 'stsm16x8x8b',
    cpasync: false,
    params: ['transpose', 'num_matrices'],
    note: 'PTX <code>stmatrix.m16n8.trans.b8</code>. Its register side matches ' +
          '<code>LdMatrix16x8x8bOp</code>\'s destination, but its addresses are the plain ' +
          '"lane <i>t</i> addresses row <i>t</i>" form: the permutation lives on the ' +
          'ldmatrix side, which is built to match this store.',
  },
  ldmatrix16x16x8b: {
    label: 'warp.LdMatrix16x16x8bOp',
    ctor: 'cute.nvgpu.warp.LdMatrix16x16x8bOp(transpose, num_matrices, unpack_bits=None)',
    kind: 'ldmatrix',
    ldsm: 'ldsm16x16x8b',
    cpasync: false,
    params: ['transpose', 'num_matrices'],
    note: 'PTX <code>.m16n16</code> with the <code>.b8</code> / <code>.b4x16_p64</code> / ' +
          '<code>.b6x16_p32</code> qualifiers. A 256-byte matrix, so it consumes 16 lanes ' +
          'per matrix rather than 8.',
  },
};

// ── tcgen05 TMEM load / store (SM100+) ─────────────────────────────────────
// A warp moves a lanes x columns block between TMEM and registers. The five
// shapes differ in how the 32 threads own that block; `repeat` widens it.
// Layouts are formulas in 32-bit columns (mcaTmemAtom), checked against
// CuTeDSL for every Op x repeat x dtype (section `tmem_atom`).
const MCA_TMEM_SPECS = {
  '32x32b':   { lanes: 32, ptx: '.32x32b',   maxRepeat: 128,
                note: 'One TMEM lane per thread: thread t owns lane t, and repeat adds 32-bit columns. The plain epilogue read of an accumulator.' },
  '16x64b':   { lanes: 16, ptx: '.16x64b',   maxRepeat: 128,
                note: '16 lanes x 64 bits per repetition: two threads share each lane.' },
  '16x128b':  { lanes: 16, ptx: '.16x128b',  maxRepeat: 64,
                note: '16 lanes x 128 bits per repetition: four threads share each lane. CuTeDSL rejects x128.' },
  '16x256b':  { lanes: 16, ptx: '.16x256b',  maxRepeat: 32,
                note: '16 lanes x 256 bits per repetition, in the m16n8 MMA-fragment arrangement: thread (t0, t1) holds lanes t1 and t1+8, columns 2t0 and 2t0+1. CuTeDSL rejects x64 and x128.' },
  '16x32bx2': { lanes: 16, ptx: '.16x32bx2', maxRepeat: 128,
                note: 'Two 16-lane x 32-bit halves: threads 0-15 and 16-31 each take one half. In PTX the second half sits at an immediate column offset; CuTe draws it as the next column.' },
};
const MCA_TMEM_REPEATS = [1, 2, 4, 8, 16, 32, 64, 128];
for (const [shape, spec] of Object.entries(MCA_TMEM_SPECS)) {
  for (const dir of ['ld', 'st']) {
    const name = `${dir === 'ld' ? 'Ld' : 'St'}${shape}Op`;
    MCA_OPS[`tmem_${dir}${shape}`] = {
      label: `tcgen05.${name}`,
      ctor: `cute.nvgpu.tcgen05.${name}(repeat, ${dir === 'ld' ? 'pack' : 'unpack'})`,
      kind: 'tmem', tmem: shape, store: dir === 'st', cpasync: false,
      params: ['repeat'],
      note: `${dir === 'ld' ? 'TMEM &rarr; RMEM' : 'RMEM &rarr; TMEM'}, PTX <code>tcgen05.${dir}.sync.aligned${spec.ptx}.x{repeat}</code> (SM100+). ` +
            spec.note + ` <code>${dir === 'ld' ? 'pack' : 'unpack'}</code> (16-bit packing in 32-bit registers) leaves the ownership unchanged, so it is not a control.`,
    };
  }
}

// Stamp each entry with its own key, so anything holding an Op object can name
// it without the caller having to thread the key through as well.
for (const [k, v] of Object.entries(MCA_OPS)) v.key = k;

// cp.async hardware widths (SM80+ non-bulk cp.async): 4, 8 or 16 bytes.
const MCA_CPASYNC_BITS = [32, 64, 128];

// ── The LdMatrix family ─────────────────────────────────────────────────────
// Every one of these instructions addresses SMEM one 128-bit row per lane; what
// differs is the size of the matrix each row belongs to, the unit the transpose
// operates on, and which parameters the Op will accept.
//
//   matrixBytes  bytes in one matrix. 8x8 of 16 b and 16x8 of 8 b are both
//                128 B; 16x16 of 8 b is 256 B. Since a lane addresses 16 B,
//                matrixBytes/16 is BOTH the rows per matrix and the lanes one
//                matrix consumes — the tile always gets exactly one lane per row.
//   unitBits     the granularity `.trans` moves. 16 for the b16 Op, 8 for the
//                two 8-bit Ops. This is what makes an element wider than the
//                unit come apart under transpose.
const MCA_LDSM_ROW_BITS = 128;        // one lane's address covers 16 B
const MCA_LDSM_MAX_ELEM_BITS = 64;

const MCA_LDSM_SPECS = {
  ldsm8x8x16b: {
    op: 'LdMatrix8x8x16bOp',
    ptx: 'ldmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].shared.b16',
    matrix: '8x8', unitBits: 16, matrixBytes: 128,
    numMatrices: [1, 2, 4],
    transpose: 'optional',
    // __post_init__ raises "Op doesn't support unpacking" for anything but None.
    unpackBits: null,
    kind: 'b16',
  },
  ldsm16x8x8b: {
    op: 'LdMatrix16x8x8bOp',
    // No direct PTX form: the DSL lowers this to .m16n16 plus address and value
    // permutations that make the result match stmatrix.m16n8.trans. That
    // permutation is why its ValLayoutSrc is a rank-4 thread mode rather than
    // the plain (live, broadcast) pair every other Op here has.
    ptx: 'lowers to ldmatrix .m16n16 + address/value permutation (no direct PTX form)',
    matrix: '16x8', unitBits: 8, matrixBytes: 128,
    numMatrices: [2, 4],
    transpose: 'required',
    unpackBits: [0, 4, 6],
    kind: 'b8perm',
  },
  ldsm16x16x8b: {
    op: 'LdMatrix16x16x8bOp',
    ptx: 'ldmatrix.sync.aligned.m16n16.x{1,2}.trans.shared.b8',
    matrix: '16x16', unitBits: 8, matrixBytes: 256,
    numMatrices: [1, 2],
    transpose: 'required',
    unpackBits: [0, 4, 6],
    kind: 'b8',
  },
  // LdMatrix8x16x8bOp: the packed 4/6-bit -> 8-bit unpacking load. CuTeDSL
  // gives it LdMatrix8x8x16bOp's layouts for every dtype x num_matrices x
  // unpack_bits (and builds exactly that atom when unpack_bits=None), so it
  // shares the b16 geometry; transpose is rejected rather than optional.
  ldsm8x16x8b: {
    op: 'LdMatrix8x16x8bOp',
    ptx: 'ldmatrix.sync.aligned.m8n16.x{1,2,4}.shared.b8x16.{b4x16_p64,b6x16_p32}',
    matrix: '8x8', unitBits: 16, matrixBytes: 128,
    numMatrices: [1, 2, 4],
    transpose: 'forbidden',
    unpackBits: [0, 4, 6],
    kind: 'b16',
  },
  // StMatrix. `store: true`: the computed layouts are swapped, so SRC is the
  // register side and DST the SMEM row addresses. Checked against CuTeDSL for
  // every dtype x num_matrices x transpose (section `ldmatrix_atom`).
  stsm8x8x16b: {
    op: 'StMatrix8x8x16bOp',
    ptx: 'stmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].shared.b16',
    matrix: '8x8', unitBits: 16, matrixBytes: 128,
    numMatrices: [1, 2, 4],
    transpose: 'optional',
    unpackBits: null,
    kind: 'b16',
    store: true,
  },
  stsm16x8x8b: {
    op: 'StMatrix16x8x8bOp',
    ptx: 'stmatrix.sync.aligned.m16n8.x{1,2,4}.trans.shared.b8',
    // Register side = LdMatrix16x8x8bOp's destination; address side is plain
    // (kind 'b8'), not the ldmatrix's rank-4 permutation. num_matrices 1 is
    // legal here although the ldmatrix twin rejects it.
    matrix: '16x8', unitBits: 8, matrixBytes: 128,
    numMatrices: [1, 2, 4],
    transpose: 'required',
    unpackBits: null,
    kind: 'b8',
    store: true,
  },
};

// Presets, one row per configuration. Kept as data rather than hand-written
// markup because the list is FILTERED to the selected Op — 16 buttons for three
// unrelated instructions is a wall, and only one Op's presets can ever apply.
// `op` is the MCA_OPS key, and doubles as the filter tag.
const MCA_PRESETS = [
  { op: 'universal', bits: 128, dtype: 'half_t', label: '128b half_t &rarr; 8 elements' },
  { op: 'universal', bits: 64,  dtype: 'half_t', label: '64b half_t &rarr; 4 elements' },
  { op: 'universal', bits: 128, dtype: 'float',  label: '128b float &rarr; 4 elements' },
  { op: 'universal', bits: 8,   dtype: 'int8_t', label: '8b int8_t &rarr; 1 element (scalar)' },
  { op: 'cpasync',   bits: 128, dtype: 'half_t', label: '128b half_t (cp.async.128)' },
  { op: 'cpasync',   bits: 64,  dtype: 'float',  label: '64b float (cp.async.64)' },
  { op: 'g2r', bits: 128, dtype: 'half_t', label: '128b half_t &mdash; GMEM&rarr;RMEM' },
  { op: 'g2r', bits: 32,  dtype: 'float',  label: '32b float &mdash; one scalar load' },
  { op: 'r2g', bits: 128, dtype: 'float',  label: '128b float &mdash; RMEM&rarr;GMEM' },
  { op: 's2r', bits: 128, dtype: 'half_t', label: '128b half_t &mdash; the SIMT SMEM load' },
  { op: 'r2s', bits: 64,  dtype: 'half_t', label: '64b half_t &mdash; RMEM&rarr;SMEM' },
  { op: 'ldmatrix', bits: 128, dtype: 'half_t', nm: 4, tr: 0, label: '.x4 half_t &mdash; all 32 lanes live' },
  { op: 'ldmatrix', bits: 128, dtype: 'half_t', nm: 2, tr: 0, label: '.x2 half_t &mdash; 16 lanes broadcast' },
  { op: 'ldmatrix', bits: 128, dtype: 'half_t', nm: 1, tr: 0, label: '.x1 half_t &mdash; 24 lanes broadcast' },
  { op: 'ldmatrix', bits: 128, dtype: 'half_t', nm: 4, tr: 1, label: '.x4.trans half_t &mdash; the transpose' },
  { op: 'ldmatrix', bits: 128, dtype: 'int8_t', nm: 1, tr: 0, label: '.x1 int8_t &mdash; 16 elements per row' },
  { op: 'ldmatrix', bits: 128, dtype: 'float',  nm: 4, tr: 1, label: '.x4.trans float &mdash; element wider than the unit' },
  { op: 'ldmatrix16x8x8b', bits: 128, dtype: 'int8_t', nm: 2, tr: 1, label: '.x2 int8_t &mdash; permuted addressing, 16 lanes' },
  { op: 'ldmatrix16x8x8b', bits: 128, dtype: 'int8_t', nm: 4, tr: 1, label: '.x4 int8_t &mdash; all 32 lanes' },
  { op: 'ldmatrix16x8x8b', bits: 128, dtype: 'half_t', nm: 4, tr: 1, label: '.x4 half_t &mdash; element wider than the 8b unit' },
  { op: 'ldmatrix16x16x8b', bits: 128, dtype: 'int8_t', nm: 1, tr: 1, label: '.x1 int8_t &mdash; 16 lanes per matrix' },
  { op: 'ldmatrix16x16x8b', bits: 128, dtype: 'int8_t', nm: 2, tr: 1, label: '.x2 int8_t &mdash; all 32 lanes' },
  { op: 'ldmatrix16x16x8b', bits: 128, dtype: 'half_t', nm: 2, tr: 1, label: '.x2 half_t &mdash; 32x8 tile' },
  { op: 'ldmatrix8x16x8b', bits: 128, dtype: 'uint8_t', nm: 4, tr: 0, label: '.x4 &mdash; 16 unpacked 8-bit values per row, all 32 lanes' },
  { op: 'ldmatrix8x16x8b', bits: 128, dtype: 'uint8_t', nm: 1, tr: 0, label: '.x1 &mdash; 8 lanes address' },
  { op: 'stmatrix', bits: 128, dtype: 'half_t', nm: 4, tr: 0, label: '.x4 half_t &mdash; the epilogue store, all 32 lanes address' },
  { op: 'tmem_ld32x32b', bits: 32, dtype: 'float', nm: 1, label: 'x1 float &mdash; thread t reads TMEM lane t' },
  { op: 'tmem_ld32x32b', bits: 32, dtype: 'float', nm: 8, label: 'x8 float &mdash; 8 accumulator columns per thread' },
  { op: 'tmem_ld32x32b', bits: 32, dtype: 'half_t', nm: 4, label: 'x4 half_t &mdash; two 16-bit values per 32-bit column' },
  { op: 'tmem_ld16x64b', bits: 32, dtype: 'float', nm: 1, label: 'x1 float &mdash; two threads per lane' },
  { op: 'tmem_ld16x128b', bits: 32, dtype: 'float', nm: 2, label: 'x2 float &mdash; four threads per lane' },
  { op: 'tmem_ld16x256b', bits: 32, dtype: 'float', nm: 1, label: 'x1 float &mdash; the m16n8 MMA-fragment arrangement' },
  { op: 'tmem_ld16x256b', bits: 32, dtype: 'half_t', nm: 2, label: 'x2 half_t &mdash; a 16-bit epilogue read' },
  { op: 'tmem_ld16x32bx2', bits: 32, dtype: 'float', nm: 4, label: 'x4 float &mdash; two 16-lane halves' },
  { op: 'tmem_st32x32b', bits: 32, dtype: 'float', nm: 4, label: 'x4 float &mdash; thread t writes TMEM lane t' },
  { op: 'tmem_st16x256b', bits: 32, dtype: 'half_t', nm: 1, label: 'x1 half_t &mdash; write an operand into TMEM' },
  { op: 'stmatrix', bits: 128, dtype: 'half_t', nm: 4, tr: 1, label: '.x4.trans half_t &mdash; store into M-major SMEM' },
  { op: 'stmatrix', bits: 128, dtype: 'half_t', nm: 1, tr: 0, label: '.x1 half_t &mdash; 8 lanes address' },
  { op: 'stmatrix16x8x8b', bits: 128, dtype: 'int8_t', nm: 4, tr: 1, label: '.x4 int8_t &mdash; plain addressing' },
  { op: 'stmatrix16x8x8b', bits: 128, dtype: 'uint8_t', nm: 2, tr: 1, label: '.x2 uint8_t &mdash; 16 lanes address' },
  { op: 'stmatrix16x8x8b', bits: 128, dtype: 'int8_t', nm: 1, tr: 1, label: '.x1 int8_t &mdash; 8 lanes address (no ldmatrix twin)' },
];

/** Preset buttons for every Op, each tagged with `data-op` so
 *  `mcaSyncPresets` can show just the selected Op's. */
function mcaPresetButtons(id) {
  return MCA_PRESETS.map(p => {
    const args = [`'${id}'`, `'${p.op}'`, p.bits, `'${p.dtype}'`];
    if (p.nm !== undefined) args.push(p.nm, p.tr);
    return `<button class="preset-btn" data-op="${p.op}" ` +
           `onclick="setMCA(${args.join(',')})">${p.label}</button>`;
  }).join('\n            ');
}

/** Show only the selected Op's presets. */
function mcaSyncPresets(tabId, opKey) {
  const host = document.getElementById(`${tabId}-mca-presets`);
  if (!host) return;
  host.querySelectorAll('.preset-btn').forEach(b => {
    b.style.display = b.getAttribute('data-op') === opKey ? '' : 'none';
  });
}

/** The Copy_Atom produced by `make_copy_atom(<an LdMatrix Op>, dtype)`.
 *  DOM-free on purpose — `renderMakeCopyAtom` reads the form and draws, this
 *  does the arithmetic, and tests/run.js diffs it against CuTeDSL (see
 *  tests/cases.json, section `ldmatrix_atom`).
 *
 *  Everything below is the instruction's FIXED geometry divided through by the
 *  element width, which is what `copy_internal_type` does in the DSL — hence
 *  `Float32` giving an 8x4 tile where `Float16` gives 8x8 for the same
 *  instruction. Verified against CuTeDSL for every numeric type x every
 *  (num_matrices, transpose, unpack_bits) each Op accepts.
 *
 *  `unpack_bits` is deliberately not a parameter: it selects the LdsmSzPattern
 *  the DSL hands to MLIR (u8 / u4x16p64to8 / u6x16p32to8), i.e. the PTX
 *  qualifier, and changes NO layout. Checked across all 156 accepted
 *  combinations of the two 8-bit Ops.
 *
 *  Returns { thrId, src, dst, tile, liveLanes, ... } with src/dst as plain
 *  {shape, stride} pairs in this repo's Layout convention. */
function mcaLdmatrixAtom(opKey, elemBits, numMatrices, transpose) {
  const spec = MCA_LDSM_SPECS[opKey];
  if (!spec) throw new Error(`Unknown LdMatrix/StMatrix Op "${opKey}"`);
  const e = elemBits;
  const k = numMatrices;

  if (!spec.numMatrices.includes(k))
    throw new Error(
      `num_matrices must be one of ${spec.numMatrices.join(', ')} for ` +
      `${spec.op} (got ${k}) — __post_init__ rejects anything else.`);
  if (spec.transpose === 'forbidden' && transpose)
    throw new Error(`${spec.op} does not support transpose — __post_init__ raises ` +
                    `"Op doesn't support transpose".`);
  if (spec.transpose === 'required' && !transpose)
    throw new Error(`${spec.op} only supports transpose — it raises ` +
                    `"Op only supports transpose" when transpose=False.`);
  if (e > MCA_LDSM_MAX_ELEM_BITS)
    throw new Error(
      `${spec.store ? 'stmatrix' : 'ldmatrix'} is undefined for ${e}-bit elements: one lane addresses a ` +
      `${MCA_LDSM_ROW_BITS}-bit row, so an element wider than ` +
      `${MCA_LDSM_MAX_ELEM_BITS} bits leaves fewer than 2 elements per row and ` +
      `the ${spec.matrix} matrix degenerates.`);

  const R = MCA_LDSM_ROW_BITS / e;              // elements per lane-addressed row
  const rowsPerMatrix = spec.matrixBytes / 16;  // 16 B per row
  const M = rowsPerMatrix * k;                  // tile rows
  const N = R;                                  // tile cols (one 16 B row)
  const L = M;                                  // one lane addresses one row
  const s = spec.unitBits / e;                  // elements per transpose unit

  // ── ValLayoutSrc ───────────────────────────────────────────────────────
  // Lane t supplies the address of row t. An Op consumes `rowsPerMatrix * k`
  // of the 32 lanes; the rest still execute the instruction and still hand
  // over an operand, so the layout has to be total over the warp. CuTe writes
  // that as a stride-0 thread mode, aliasing the ignored lanes onto live ones:
  // in-bounds, branchless, and it makes size exceed cosize by the broadcast
  // factor.
  let src;
  if (spec.kind === 'b8perm') {
    // 16x8x8b bakes in the address permutation that matches stmatrix.m16n8.trans,
    // so its thread mode is rank 4 with non-monotonic strides. The stride-0
    // slot is the broadcast one and only appears at num_matrices == 2.
    src = {
      shape:  [[2, 2, 4, 2], R],
      stride: [[R, 8 * R, 2 * R, L === 32 ? 16 * R : 0], 1],
    };
  } else {
    src = (L === 32)
      ? { shape: [32, R],          stride: [R, 1] }
      : { shape: [[L, 32 / L], R], stride: [[R, 0], 1] };
  }

  // ── ValLayoutDst ───────────────────────────────────────────────────────
  let dst;
  if (spec.kind === 'b16' && !transpose) {
    // One 32-bit register per matrix per lane, in mma-fragment order.
    const S = (MCA_LDSM_ROW_BITS * rowsPerMatrix) / e;  // elements per matrix
    const q = Math.max(1, 32 / e);        // elements in that register
    const T = S / q;                      // lanes needed to hold one matrix
    dst = (k === 1)
      ? { shape: [T, q],      stride: [q, 1] }
      : { shape: [T, [q, k]], stride: [q, [1, S]] };
  } else if (spec.kind === 'b16') {
    // `.trans` transposes the 8x8 matrix OF 16-BIT UNITS. In units the result
    // is ((4,8),(1,2,k)):((16,1),(1,8,64)); `s` re-expresses it in elements and
    // is < 1 exactly when an element spans several units, which is why a 32-bit
    // type ends up with a 16-lane thread mode.
    const thrShape  = s >= 1 ? [4, 8]      : [4, 8 * s];
    const thrStride = s >= 1 ? [16 * s, s] : [16 * s, 1];
    const lead = s >= 1 ? s : 1;          // elements packed inside one unit
    dst = (k === 1)
      ? { shape: [thrShape, [lead, 2]],    stride: [thrStride, [1, 8 * s]] }
      : { shape: [thrShape, [lead, 2, k]], stride: [thrStride, [1, 8 * s, 64 * s]] };
  } else {
    // The two 8-bit Ops share one transposed form, differing only in how many
    // units a matrix row holds. In 8-bit units:
    //     thread ((4, 8), (…)) : ((matrixBytes/4, 1), …)
    //     value  (1, matrixBytes/64, 2 [, k]) : (1, 16, 8 [, matrixBytes])
    // `s` scales it into elements. The element can never be SMALLER than the
    // 8-bit unit here, so `s <= 1` and there is no packed leading mode.
    const A = spec.matrixBytes / 64;      // 2 for 16x8, 4 for 16x16
    const thr = { shape: [4, 8 * s], stride: [(spec.matrixBytes * s) / 4, 1] };
    dst = (k === 1)
      ? { shape: [thr.shape, [1, A, 2]],
          stride: [thr.stride, [1, 16 * s, 8 * s]] }
      : { shape: [thr.shape, [1, A, 2, k]],
          stride: [thr.stride, [1, 16 * s, 8 * s, spec.matrixBytes * s]] };
  }

  // stmatrix is the mirror: the SMEM row addresses are its DESTINATION and the
  // register fragment its SOURCE. `addrSide` names the side the lanes address.
  if (spec.store) [src, dst] = [dst, src];
  return {
    opKey, spec, addrSide: spec.store ? 'dst' : 'src', regSide: spec.store ? 'src' : 'dst',
    thrId: { shape: 32, stride: 1 },   // Layout<_32>; CuTeDSL prints 32:1
    src, dst,
    tile: { shape: [M, N], stride: [N, 1] },  // row-major: flat = m*N + n
    liveLanes: L, rowElems: R, rowsPerMatrix, numMatrices: k, transpose,
    srcStr: formatLayoutStr(src.shape, src.stride),
    dstStr: formatLayoutStr(dst.shape, dst.stride),
  };
}

/** The Copy_Atom of `make_copy_atom(tcgen05.<Ld|St><shape>Op(repeat), dtype)`.
 *  DOM-free; tests/run.js diffs it against CuTeDSL (section `tmem_atom`).
 *
 *  The TMEM side is `(32, total):(0, 1)`: the warp addresses the block as a
 *  whole, so every thread names all of it. The register side is a TV map
 *  over the block, row-major: flat = lane * cols + col. Each shape's map is a
 *  formula in 32-bit columns; a narrower element splits every 32-bit cell
 *  into k = 32 / elemBits adjacent elements (the leading value extent times k,
 *  every other stride times k), which is the recast CuTeDSL applies. Elements
 *  wider than a 32-bit cell are not drawn here. */
function mcaTmemAtom(shape, elemBits, repeat, store) {
  const spec = MCA_TMEM_SPECS[shape];
  if (!spec) throw new Error(`Unknown tcgen05 TMEM shape "${shape}"`);
  const r = repeat;
  if (!MCA_TMEM_REPEATS.includes(r) || r > spec.maxRepeat)
    throw new Error(`repeat x${r} is not offered for ${shape}: CuTeDSL accepts x1 to x${spec.maxRepeat}.`);
  if (![8, 16, 32].includes(elemBits))
    throw new Error(`TMEM cells are 32 bits; this view draws 8-, 16- and 32-bit elements, not ${elemBits}-bit ones.`);
  const k = 32 / elemBits;
  // 32-bit-unit register layouts.
  let thr, val, cols32;
  if (shape === '32x32b') {
    thr = { shape: 32, stride: r }; val = { shape: r, stride: 1 }; cols32 = r;
  } else if (shape === '16x64b') {
    thr = { shape: [2, 2, 8], stride: [16 * r, 1, 2 * r] };
    val = r === 1 ? { shape: 1, stride: 1 } : { shape: [1, r], stride: [1, 2] };
    cols32 = 2 * r;
  } else if (shape === '16x128b') {
    thr = { shape: [4, 8], stride: [1, 4 * r] };
    val = r === 1 ? { shape: [1, 2], stride: [1, 32] } : { shape: [1, 2, r], stride: [1, 32 * r, 4] };
    cols32 = 4 * r;
  } else if (shape === '16x256b') {
    thr = { shape: [4, 8], stride: [2, 8 * r] };
    val = r === 1 ? { shape: [2, 2], stride: [1, 64] } : { shape: [2, 2, r], stride: [1, 64 * r, 8] };
    cols32 = 8 * r;
  } else {  // 16x32bx2
    thr = { shape: [16, 2], stride: [2 * r, r] }; val = { shape: r, stride: 1 }; cols32 = 2 * r;
  }
  // Recast to elements: k adjacent elements per 32-bit cell.
  const scale = x => Array.isArray(x) ? x.map(v => v * k) : x * k;
  thr = { shape: thr.shape, stride: scale(thr.stride) };
  val = Array.isArray(val.shape)
    ? { shape: [val.shape[0] * k, ...val.shape.slice(1)], stride: [1, ...val.stride.slice(1).map(v => v * k)] }
    : { shape: val.shape * k, stride: 1 };
  const reg = { shape: [thr.shape, val.shape], stride: [thr.stride, val.stride] };
  const lanes = spec.lanes, cols = cols32 * k, total = lanes * cols;
  const tmem = { shape: [32, total], stride: [0, 1] };
  const [src, dst] = store ? [reg, tmem] : [tmem, reg];
  return {
    shape, spec, store, repeat: r, k, lanes, cols, cols32, total,
    thrId: { shape: 32, stride: 1 }, src, dst,
    regSide: store ? 'src' : 'dst', tmemSide: store ? 'dst' : 'src',
    tile: { shape: [lanes, cols], stride: [cols, 1] },   // row-major: flat = lane * cols + col
    srcStr: formatLayoutStr(src.shape, src.stride),
    dstStr: formatLayoutStr(dst.shape, dst.stride),
  };
}

function generateMakeCopyAtomTabContent(id) {
  return `
    <!-- make_copy_atom panel -->
    <div id="${id}-tab-make_copy_atom" class="panel">
      <div class="controls">
        <h2>make_copy_atom</h2>

        <details class="cuo-section" open>
          <summary>0. Memory movement</summary>
          <div class="cuo-section-body">
${copyMoveField(id, 'mca')}
          </div>
        </details>

        <details class="cuo-section" open><summary>1. CopyAtom</summary><div class="cuo-section-body">
          <button class="btn btn-render btn-decode" id="${id}-mca-decode-atom" onclick="mcaDecodeAtomPasteboard('${id}','mca','renderMakeCopyAtom')">Decode from Pasteboard</button>
          <details class="cuo-section" id="${id}-mca-atom-manual-construction"><summary>Manual construction</summary><div class="cuo-section-body">
            <div class="form-group">
              <label>Op</label>
              <select id="${id}-mca-op-input" onchange="setMcaOp('${id}')">
                ${simtCopyOpOptions('universal')}
                <option value="ldmatrix">warp.LdMatrix8x8x16bOp</option>
                <option value="ldmatrix16x8x8b">warp.LdMatrix16x8x8bOp</option>
                <option value="ldmatrix16x16x8b">warp.LdMatrix16x16x8bOp</option>
                <option value="ldmatrix8x16x8b">warp.LdMatrix8x16x8bOp</option>
                <option value="stmatrix">warp.StMatrix8x8x16bOp</option>
                <option value="stmatrix16x8x8b">warp.StMatrix16x8x8bOp</option>
                ${Object.entries(MCA_OPS).filter(([, o]) => o.kind === 'tmem').map(([key, o]) => `<option value="${key}">${o.label}</option>`).join('')}
              </select>
            </div>
            <div id="${id}-mca-ldsm-params" style="display:none">
              <div class="form-group">
                <label>num_matrices<span style="color:#6b7280;font-weight:normal">&nbsp;&mdash; the .x{1,2,4} qualifier; the domain differs per Op</span></label>
                <!-- options are rebuilt per Op by mcaSyncLdsmControls -->
                <select id="${id}-mca-nm-input" onchange="renderMakeCopyAtom('${id}')"></select>
              </div>
              <div class="form-group">
                <label>transpose<span style="color:#6b7280;font-weight:normal">&nbsp;&mdash; the .trans qualifier</span></label>
                <select id="${id}-mca-trans-input" onchange="renderMakeCopyAtom('${id}')">
                  <option value="0" selected>False</option>
                  <option value="1">True</option>
                </select>
              </div>
            </div>
            <div id="${id}-mca-tmem-params" style="display:none">
              <div class="form-group">
                <label>repeat<span style="color:#6b7280;font-weight:normal">&nbsp;&mdash; the .x{N} qualifier: how many times the block repeats along TMEM columns</span></label>
                <!-- options are rebuilt per Op by mcaSyncTmemControls -->
                <select id="${id}-mca-rep-input" onchange="renderMakeCopyAtom('${id}')"></select>
              </div>
            </div>
            <div class="form-group">
              <label>tensor_dtype</label>
              <select id="${id}-mca-dtype-input" onchange="renderMakeCopyAtom('${id}')">${dtypeOptions('half_t')}</select>
            </div>
            <div class="form-group" id="${id}-mca-bits-group">
              <label>num_bits_per_copy</label>
              <input type="text" id="${id}-mca-bits-input" value="128">
            </div>
          </div></details>
          <div id="${id}-mca-op-params" class="cuo-result"></div>
          <div id="${id}-mca-atom-result" class="cuo-result"></div>
        </div></details>

        ${statusDivs(`${id}-mca`)}
        <button class="btn btn-render" onclick="renderMakeCopyAtom('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-mca-export" onclick="exportMCA('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list" id="${id}-mca-presets">${mcaPresetButtons(id)}</div>
        </div>

        <div class="hint">
          <b>Two steps, two kinds of parameter.</b> An Op is constructed first
          (with whatever fields it carries), then
          <code>make_copy_atom(op, dtype, num_bits_per_copy=N)</code> turns it
          into an Atom. The SIMT Ops take no constructor parameters, so
          section 1 only asks which one; <code>warp.LdMatrix8x8x16bOp</code>
          takes two, and they appear there.<br><br>
          <b>Six Ops, one picture.</b> <code>CopyUniversalOp</code>,
          <code>cpasync.CopyG2SOp</code> and the four directional SIMT Ops
          (<code>CopyG2ROp</code>, <code>CopyR2GOp</code>, <code>CopyS2ROp</code>,
          <code>CopyR2SOp</code>) all produce a byte-identical Atom:
          <code>ThrID = 1:0</code> and
          <code>ValLayoutSrc == ValLayoutDst == (1, N):(0, 1)</code>. What the
          directional Ops add is a set of <em>memory attributes</em> &mdash;
          ordering, scope, cache and prefetch policy &mdash; passed as
          <code>make_copy_atom</code> kwargs. Measured across all 32 settings of
          <code>CopyG2ROp</code>'s: <b>1</b> distinct
          <code>(ThrID, src, dst)</code> triple, <b>26</b> distinct MLIR atom
          types. They are encoded in the Atom's <em>type</em> and decide the PTX
          qualifiers, not which elements move, so the picture is rightly
          unchanged &mdash; which is why this tab states them rather than
          offering controls that could not change a cell. Two things they
          <em>do</em> change: the emitted instruction, and the fact that
          <code>CopyG2RTrait</code> / <code>CopyR2GTrait</code> can carry a
          <em>runtime</em> <code>cache_policy</code> that
          <code>CopyUniversalTrait</code> cannot.<br><br>
          <b>The direction is fixed by the Op</b> &mdash; baked into the MLIR
          type (<code>!cute_nvgpu.atom.g2r&lt;...&gt;</code>), not chosen at
          construction &mdash; so section 0's picker is a single disabled entry
          for each. Use <code>CopyUniversalOp</code> when you just want the move
          (it is also the only one that lets you omit
          <code>num_bits_per_copy</code> and auto-vectorize); reach for a
          directional Op to say something about how memory behaves. Note
          <code>CopyS2ROp</code> shares SMEM&rarr;RMEM with
          <code>ldmatrix</code> but is the single-thread load &mdash; same
          direction, entirely different layouts.<br><br>
          <b>The LdMatrix family: three Ops, three parameter domains.</b>
          <code>num_matrices</code> is the <code>.x{1,2,4}</code> qualifier and
          decides <em>how many lanes' addresses the hardware consumes</em>. It
          does not change how much any one lane addresses: that is always one
          128-bit row. What it costs in lanes depends on the matrix size &mdash;
          a 128 B matrix (<code>8x8x16b</code>, <code>16x8x8b</code>) takes 8
          lanes, a 256 B one (<code>16x16x8b</code>) takes 16 &mdash; so the
          legal values differ per Op and the picker is rebuilt when you switch.
          <code>transpose</code> is <code>.trans</code>; it is optional only on
          <code>LdMatrix8x8x16bOp</code> and <b>mandatory</b> on both 8-bit Ops,
          which is why the control is pinned and greyed there.<br><br>
          <b>LdMatrix16x8x8bOp has no direct PTX form.</b> It lowers to
          <code>.m16n16</code> plus address and value permutations chosen to
          match <code>stmatrix.m16n8.trans</code>, which is what makes its
          <code>ValLayoutSrc</code> a rank-4 thread mode. Lane <i>t</i> does
          <em>not</em> address row <i>t</i> for this Op &mdash; the SRC pane
          shows the permutation directly.<br><br>
          <b>unpack_bits is fixed to None.</b> Packed 4- or 6-bit source
          unpacking is outside this layout visualization. On the supported
          8-bit Ops it changes the instruction's data interpretation, while
          leaving the SRC/DST ownership layouts unchanged. The 16-bit Op
          rejects unpacking.<br><br>
          <b>LdMatrix8x16x8bOp unpacks FP4/FP6.</b> Each 16 B SMEM row holds 16
          packed 4- or 6-bit values (<code>unpack_bits</code>) that land in 8-bit
          register slots. Its ownership layouts are <code>LdMatrix8x8x16bOp</code>'s,
          so it draws the same; <code>transpose</code> is rejected.<br><br>
          <b>StMatrix is ldmatrix in reverse.</b> <code>StMatrix8x8x16bOp</code>
          and <code>StMatrix16x8x8bOp</code> store registers to SMEM, so the
          <em>destination</em> is the addressing pattern and the <em>source</em> is
          the register fragment. For the 8x8 Op the two layouts are exactly
          ldmatrix's, swapped. The 16x8 Op's addresses are the plain form:
          <code>LdMatrix16x8x8bOp</code>'s permutation exists to match this
          store. Its <code>num_matrices</code> may be 1, 2 or 4.<br><br>
          <b>num_bits_per_copy is ignored by ldmatrix.</b> The instruction's
          width is fixed by the Op, so <code>_make_trait</code> never reads the
          argument &mdash; you can pass one and it changes nothing. The field is
          disabled rather than hidden so that is visible rather than
          mysterious.<br><br>
          <b>Src and Dst are different layouts, and not paired by (t, v).</b>
          For the two SIMT Ops <code>ValLayoutSrc == ValLayoutDst</code>, so the
          panes agree. For every LdMatrix Op they do not: src is an <em>addressing</em>
          pattern (which lane points at which 16 B row) and dst is the
          <em>register outcome</em> (which lane ends up holding which element).
          They cover the same cells and nothing more &mdash; at
          <code>.x1</code> a lane even has 8 source slots against 2 destination
          slots, so there is no slot-to-slot correspondence to look for. The
          data crosses lanes inside the instruction.<br><br>
          <b>What an Atom is.</b> One instruction's worth of copy: how many
          threads participate (<code>ThrID</code>) and which values each one
          moves. Nothing about tiles, thread blocks or tensors is decided
          here.<br><br>
          <b>Next step.</b> To replicate this atom across a thread tile, use
          <b>make_tiled_copy</b> or <b>make_tiled_copy_tv</b> &mdash; though for
          ldmatrix the tiling you actually want comes from
          <code>make_tiled_copy_A/B(atom, tiled_mma)</code>, because the
          destination has to match the MMA's operand fragment.
        </div>
      </div>

      <div class="comp-results" style="grid-template-columns:1fr">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-mca-atom-title">Copy_Atom &mdash; single-atom data</span>
            <span style="display:flex;align-items:center;gap:4px">
              ${copyDirButtons(id, 'mca')}
              <button class="mode-btn" id="${id}-mca-val-btn" onclick="toggleMcaValue('${id}')">value</button>
              <button class="mode-btn" id="${id}-mca-src-svg-zoom" onclick="toggleCopyZoom('${id}','mca')">Zoom in</button>
            </span>
          </div>
          <div class="cuo-viz-desc" id="${id}-mca-viz-desc"></div>
${copyPanes(id, 'mca')}
        </div>
      </div>
    </div>`;
}

const mcaState = {};

function renderMakeCopyAtom(tabId) {
  showErr(`${tabId}-mca-error`, '');
  showWarn(`${tabId}-mca-warning`, '');
  try {
    const opKey = document.getElementById(`${tabId}-mca-op-input`).value;
    const op = MCA_OPS[opKey] || MCA_OPS.universal;
    mcaRenderOpParams(tabId, op);
    syncCopyMoves(tabId, 'mca', opKey);

    const dtype = document.getElementById(`${tabId}-mca-dtype-input`).value;
    const elemBits = DTYPE_BITS[dtype];
    if (!elemBits) throw new Error(`Unknown tensor_dtype "${dtype}"`);

    const prev = mcaState[tabId] || {};
    if (op.kind === 'ldmatrix')  mcaRenderLdmatrix(tabId, opKey, op, dtype, elemBits, prev);
    else if (op.kind === 'tmem') mcaRenderTmem(tabId, opKey, op, dtype, elemBits, prev);
    else                         mcaRenderSimt(tabId, opKey, op, dtype, elemBits, prev);
  } catch (e) {
    showErr(`${tabId}-mca-error`, e.message);
    for (const side of ['src', 'dst']) {
      const el = document.getElementById(`${tabId}-mca-${side}-svg`);
      if (el) el.innerHTML = '';
    }
  }
}

// ── CopyUniversalOp / cpasync.CopyG2SOp ─────────────────────────────────────
// One thread, N contiguous elements, ValLayoutSrc == ValLayoutDst.
function mcaRenderSimt(tabId, opKey, op, dtype, elemBits, prev) {
  const bitsStr = document.getElementById(`${tabId}-mca-bits-input`).value;
  const numBits = parseInt(bitsStr, 10);
  if (!Number.isFinite(numBits) || numBits <= 0) {
    throw new Error(`num_bits_per_copy must be a positive integer, got "${bitsStr}"`);
  }
  if (numBits % elemBits !== 0) {
    throw new Error(
      `num_bits_per_copy (${numBits}) must be a multiple of sizeof_bits(${dtype}) = ${elemBits}`);
  }
  const elements = numBits / elemBits;

  // Per-thread value layout of ONE atom. Copy_Traits<UniversalCopy<S,D>>
  // declares `SrcLayout = Layout<Shape<_1, sizeof_bits<S>>>` with no explicit
  // stride, which compacts to col-major `(_1, _1)`; after
  // `recast_layout<uint1_t, ValType>` mode 1 becomes `(elements):(1)`.
  //
  // Mode 0's stride is DEGENERATE and the toolchains disagree on how to print
  // it: it is a size-1 mode, so its stride is only ever multiplied by the
  // coordinate 0 — `(1,N):(0,1)` and `(1,N):(1,1)` are the same function.
  // C++ `upcast` yields 1; CuTeDSL prints 0. Print the DSL's form, since that
  // is what users are matching against on screen.
  const atomSrc   = new Layout([1, elements], [0, 1]);
  const atomThrID = new Layout(1, 0);        // Layout<_1>; CuTeDSL prints this as 1:0
  const atomStr   = formatLayoutStr(atomSrc.shape, atomSrc.stride);

  if (op.cpasync && !MCA_CPASYNC_BITS.includes(numBits)) {
    showWarn(`${tabId}-mca-warning`,
      `cp.async hardware only accepts num_bits_per_copy ∈ {32, 64, 128} — ` +
      `${numBits} would be rejected by cpasync.CopyG2SOp (CopyUniversalOp would take it).`);
  }

  mcaState[tabId] = {
    kind: 'simt', opKey, op, numBits, dtype, elements, elemBits, atomSrc, atomThrID, atomStr,
    atomDstStr: atomStr,   // ValLayoutDst == ValLayoutSrc for both Ops here
    showValue: !!prev.showValue,
  };

  document.getElementById(`${tabId}-mca-atom-result`).innerHTML =
    `<div class="cuo-result-line"><b>Copy_Atom&lt;${op.label}&lt;${numBits}b&gt;, ${dtype}&gt;</b></div>` +
    `<div class="cuo-result-line">elements_to_copy = ${numBits} / ${elemBits} = <b>${elements}</b></div>` +
    `<div class="cuo-result-line">ThrID        = ${formatLayoutStr(atomThrID.shape, atomThrID.stride)}</div>` +
    `<div class="cuo-result-line">ValLayoutSrc = ${atomStr}</div>` +
    `<div class="cuo-result-line">ValLayoutDst = ${atomStr}</div>` +
    `<div class="cuo-result-line" style="color:#9ca3af">The ${elements} value${elements === 1 ? '' : 's'} ` +
    `must be <b>contiguous</b> in both src and dst when this atom is finally applied to a tensor &mdash; ` +
    `that is the constraint <code>upcast&lt;${elements}&gt;</code> enforces at JIT time.</div>`;

  document.getElementById(`${tabId}-mca-viz-desc`).innerHTML =
    `One atom invocation: the values a single instruction moves, in order. ` +
    `<code>V<i>k</i></code> is the value index within the atom, and it is the ` +
    `only thing a cell can say &mdash; <code>ThrID = 1:0</code> means one ` +
    `thread issues the whole instruction, so there is no thread assignment to ` +
    `draw. Threads enter the picture only when the atom is replicated over a ` +
    `tile; that is <b>make_tiled_copy</b>'s job. For these two Ops ` +
    `<code>ValLayoutSrc == ValLayoutDst</code>, so the two panes are identical ` +
    `&mdash; they diverge for atoms that shuffle, such as <code>ldmatrix</code>.`;

  renderMcaAtomViz(tabId);
  updateOuterTabLabel(tabId, `make_copy_atom:${numBits}b/${dtype}`);
}

// ── The LdMatrix family ─────────────────────────────────────────────────────
function mcaRenderLdmatrix(tabId, opKey, op, dtype, elemBits, prev) {
  const spec = MCA_LDSM_SPECS[op.ldsm];
  const nm = parseInt(document.getElementById(`${tabId}-mca-nm-input`).value, 10);
  const transpose = spec.transpose === 'required' ||
                    (spec.transpose !== 'forbidden' && document.getElementById(`${tabId}-mca-trans-input`).value === '1');
  const unpackBits = null; // CuTeDSL unpack_bits=None; packed-source unpacking is not exposed.
  const a = mcaLdmatrixAtom(op.ldsm, elemBits, nm, transpose);

  // Validation CuTe skips. `.trans` moves whole `unitBits` units, so the layout
  // is a faithful element-level description of the instruction only when the
  // element IS the unit. The DSL recasts happily and returns a layout either way.
  // showWarn writes textContent, like every other tab's warnings — plain text
  // only, no markup.
  const notes = [];
  // Address side vs register side: SRC/DST for ldmatrix, DST/SRC for stmatrix.
  const Addr = a.addrSide === 'src' ? 'Src' : 'Dst', Reg = a.regSide === 'src' ? 'Src' : 'Dst';
  const regL = a[a.regSide], addrL = a[a.addrSide];
  if (transpose && elemBits !== spec.unitBits) {
    notes.push(
      `.trans on ${spec.op} is a ${spec.unitBits}-bit transpose, but ${dtype} is ` +
      `${elemBits}-bit. ` +
      (elemBits < spec.unitBits
        ? `${spec.unitBits / elemBits} adjacent ${dtype} elements move as one unit and stay ` +
          `adjacent — that is the leading value mode of ValLayout${Reg}.`
        : `One ${dtype} element spans ${elemBits / spec.unitBits} units, which the transpose ` +
          `separates, so the thread mode covers only ${product(regL.shape[0])} of the 32 ` +
          `lanes. CuTeDSL returns this layout without complaint; whether it is the ` +
          `instruction you meant is another matter.`));
  }
  if (a.liveLanes < 32) {
    notes.push(
      `.x${nm} consumes addresses from ${a.liveLanes} of the 32 lanes. The other ` +
      `${32 - a.liveLanes} still execute the instruction and still hand over an operand — ` +
      `CuTe maps them with a stride-0 thread mode, so size(ValLayout${Addr}) = ` +
      `${product(addrL.shape[0]) * product(addrL.shape[1])} against cosize = ` +
      `${a.tile.shape[0] * a.tile.shape[1]}. They are drawn in transparent grey.`);
  }
  if (notes.length) showWarn(`${tabId}-mca-warning`, notes.join('  '));

  const numValSrc = product(a.src.shape[1]);
  const numValDst = product(a.dst.shape[1]);
  mcaState[tabId] = {
    kind: 'ldmatrix', opKey, op, spec, dtype, elemBits, atom: a,
    numValSrc, numValDst, unpackBits, showValue: !!prev.showValue,
  };

  const q = `.x${nm}${transpose ? '.trans' : ''}`;
  document.getElementById(`${tabId}-mca-atom-result`).innerHTML =
    `<div class="cuo-result-line"><b>Copy_Atom&lt;${spec.op}${q}, ${dtype}&gt;</b></div>` +
    `<div class="cuo-result-line" style="color:#9ca3af">${spec.ptx}</div>` +
    `<div class="cuo-result-line">ThrID        = ${formatLayoutStr(a.thrId.shape, a.thrId.stride)}` +
    `<span style="color:#9ca3af"> &mdash; warp-collective; all 32 lanes execute it</span></div>` +
    `<div class="cuo-result-line">ValLayoutSrc = ${a.srcStr}` +
    `<span style="color:#9ca3af"> &mdash; ${numValSrc} slots/lane</span></div>` +
    `<div class="cuo-result-line">ValLayoutDst = ${a.dstStr}` +
    `<span style="color:#9ca3af"> &mdash; ${numValDst} slots/lane</span></div>` +
    `<div class="cuo-result-line">tile         = ${a.tile.shape[0]} &times; ${a.tile.shape[1]} ${dtype} ` +
    `= ${a.numMatrices} ${spec.matrix} matri${a.numMatrices === 1 ? 'x' : 'ces'} of ` +
    `${a.rowsPerMatrix} rows &times; ${a.rowElems} ` +
    `(${MCA_LDSM_ROW_BITS} bits = 16 B per row, ${spec.matrixBytes} B per matrix)</div>` +
    `<div class="cuo-result-line">lanes supplying addresses = <b>${a.liveLanes}</b> of 32</div>` +
    `<div class="cuo-result-line" style="color:#9ca3af">` +
    `<code>num_bits_per_copy</code> is <b>not</b> read for this Op &mdash; the width is fixed ` +
    `by the instruction, so <code>_make_trait</code> ignores the argument.</div>`;

  const addrDesc = `<b>${Addr.toUpperCase()}</b> is the <em>addressing</em> pattern: which lane hands over the ` +
    `address of which 16 B row`;
  const regDesc = spec.store
    ? `<b>SRC</b> is the <em>register fragment</em>: which lane holds which element before the store`
    : `<b>DST</b> is the <em>register outcome</em>: which lane ends up holding which element`;
  document.getElementById(`${tabId}-mca-viz-desc`).innerHTML =
    `The same ${a.tile.shape[0]}&times;${a.tile.shape[1]} tile, indexed two ways. ` +
    (spec.store ? `${regDesc}. ${addrDesc}. ` : `${addrDesc}. ${regDesc}. `) +
    `They cover the same cells and ` +
    `nothing more &mdash; there is no <code>(t, v)</code>-to-<code>(t, v)</code> ` +
    `correspondence between them, because the data crosses lanes inside the ` +
    `instruction. Cell colour is the thread id.` +
    (a.liveLanes < 32
      ? ` Lanes handling no address are <b>disabled</b> on the ${a.addrSide} side: they hand ` +
        `over an operand that <code>.x${nm}</code> discards, so they are drawn in ` +
        `transparent grey on top of the live lane they alias.`
      : ` Every lane's address is consumed here, so nothing is greyed out.`) +
    (spec.kind === 'b8perm'
      ? ` Note SRC is <b>not</b> "lane <i>t</i> addresses row <i>t</i>" for this Op &mdash; ` +
        `it bakes in an address permutation to match <code>stmatrix.m16n8.trans</code>, ` +
        `which is what its rank-4 thread mode encodes.`
      : '');

  renderMcaAtomViz(tabId);
  updateOuterTabLabel(tabId, `make_copy_atom:${spec.matrix}${q}/${dtype}`);
}

/** Rebuild the LdMatrix controls for `spec`. The three Ops disagree on the
 *  selectable parameters, and an illegal combination must not be reachable:
 *  num_matrices is {1,2,4} / {2,4} / {1,2}, transpose is optional on the b16 Op
 *  and MANDATORY on both 8-bit ones. unpack_bits is fixed to None. Selections are kept
 *  across an Op change when the new Op still permits them. */
function mcaSyncLdsmControls(tabId, spec, p) {
  p = p || 'mca';
  const nmSel = document.getElementById(`${tabId}-${p}-nm-input`);
  if (nmSel) {
    const want = nmSel.value;
    nmSel.innerHTML = spec.numMatrices.map(k => {
      const lanes = (spec.matrixBytes / 16) * k;
      return `<option value="${k}">${k} &mdash; .x${k} (${lanes} lane` +
             `${lanes === 1 ? '' : 's'} supply addresses)</option>`;
    }).join('');
    nmSel.value = spec.numMatrices.map(String).includes(want)
      ? want : String(spec.numMatrices[spec.numMatrices.length - 1]);
  }

  // A required transpose is pinned to True and disabled — offering a False the
  // constructor throws on would be a control that only produces errors.
  const trSel = document.getElementById(`${tabId}-${p}-trans-input`);
  if (trSel) {
    // A forbidden one is pinned to False the same way.
    const required = spec.transpose === 'required', forbidden = spec.transpose === 'forbidden';
    if (required) trSel.value = '1';
    if (forbidden) trSel.value = '0';
    trSel.disabled = required || forbidden;
    trSel.title = required ? `${spec.op} only supports transpose`
                : forbidden ? `${spec.op} does not support transpose` : '';
  }

}

// Section 1's body. The two SIMT Ops are parameterless, so this states that;
// the LdMatrix controls live in `-mca-ldsm-params` above and are toggled here.
function mcaRenderOpParams(tabId, op) {
  const ldsm = document.getElementById(`${tabId}-mca-ldsm-params`);
  if (ldsm) ldsm.style.display = op.kind === 'ldmatrix' ? '' : 'none';
  if (op.kind === 'ldmatrix') mcaSyncLdsmControls(tabId, MCA_LDSM_SPECS[op.ldsm]);
  const tmem = document.getElementById(`${tabId}-mca-tmem-params`);
  if (tmem) tmem.style.display = op.kind === 'tmem' ? '' : 'none';
  if (op.kind === 'tmem') mcaSyncTmemControls(tabId, op);
  mcaSyncPresets(tabId, op.key);
  // num_bits_per_copy is meaningless for ldmatrix — grey it out rather than
  // hide it, so the absence is legible instead of just missing.
  // ... and for the TMEM Ops, whose width is the shape times repeat.
  const fixedWidth = op.kind === 'ldmatrix' || op.kind === 'tmem';
  const bits = document.getElementById(`${tabId}-mca-bits-input`);
  if (bits) bits.disabled = fixedWidth;
  const bitsGroup = document.getElementById(`${tabId}-mca-bits-group`);
  if (bitsGroup) bitsGroup.style.opacity = fixedWidth ? '0.45' : '';

  const host = document.getElementById(`${tabId}-mca-op-params`);
  if (!host) return;
  host.innerHTML =
    `<div class="cuo-result-line"><b>${op.ctor}</b></div>` +
    (op.params.length === 0
      ? `<div class="cuo-result-line" style="color:#9ca3af">No constructor parameters required. ` +
        `${op.note}</div>` +
        // Memory attributes are make_copy_atom kwargs, not Op fields, and they
        // are stated rather than offered: every one of them leaves ThrID and
        // both ValLayouts untouched, so a control for them could not change a
        // single cell of the picture.
        (op.attrs
          ? `<div class="cuo-result-line" style="color:#9ca3af">Accepts as ` +
            `<code>make_copy_atom</code> kwargs: ` +
            op.attrs.map(a => `<code>${a}</code>`).join(', ') + `. None of them changes ` +
            `<code>ThrID</code> or either ValLayout &mdash; they are encoded in the Atom's ` +
            `<em>type</em> and decide the PTX qualifiers, not which elements move. That is why ` +
            `this tab does not offer them as controls.</div>`
          : '')
      : `<div class="cuo-result-line" style="color:#9ca3af">${op.note}</div>` +
        `<div class="cuo-result-line" style="color:#9ca3af">Constructor parameters: ` +
        op.params.map(p => `<code>${p}</code>`).join(', ') + `.` +
        (op.ldsm && !MCA_LDSM_SPECS[op.ldsm].unpackBits
          ? ` <code>unpack_bits</code> is inherited from <code>BaseOp</code> but rejected ` +
            `by this Op &mdash; it belongs to the two 8-bit LdMatrix variants.`
          : (op.ldsm ? ` <code>unpack_bits=None</code> is fixed in this visualizer.` : '')) +
        `</div>` +
        (op.ldsm
          ? `<div class="cuo-result-line" style="color:#9ca3af">PTX: ` +
            `<code>${MCA_LDSM_SPECS[op.ldsm].ptx}</code></div>`
          : ''));
}

// Re-render when the Op changes. Safe before the first Render: it only repaints
// the params box until there is state to draw.
function setMcaOp(tabId) {
  const opKey = document.getElementById(`${tabId}-mca-op-input`).value;
  const op = MCA_OPS[opKey] || MCA_OPS.universal;
  mcaRenderOpParams(tabId, op);
  syncCopyMoves(tabId, 'mca', opKey);
  if (mcaState[tabId]) renderMakeCopyAtom(tabId);
}

/** The `value` toggle: overlay each cell's position in the atom's tile. Off by
 *  default, like every other tab's mode buttons. */
function toggleMcaValue(tabId) {
  const s = mcaState[tabId];
  if (!s) return;
  s.showValue = !s.showValue;
  renderMcaAtomViz(tabId);
}

function renderMcaAtomViz(tabId) {
  const s = mcaState[tabId];
  if (!s) return;
  const btn = document.getElementById(`${tabId}-mca-val-btn`);
  if (btn) {
    // The SIMT pane already prints Vk in every cell, so the overlay has nothing
    // to add there — hide the button rather than ship one that does nothing.
    btn.style.display = s.kind === 'ldmatrix' || s.kind === 'tmem' ? '' : 'none';
    btn.classList.toggle('active', !!s.showValue);
  }
  if (s.kind === 'ldmatrix')  mcaDrawLdmatrix(tabId, s);
  else if (s.kind === 'tmem') mcaDrawTmem(tabId, s);
  else                        mcaDrawSimt(tabId, s);
}

function mcaDrawSimt(tabId, s) {
  // ValLayoutSrc == ValLayoutDst for every Op here, so the two panes draw the
  // same grid. Rendering them separately anyway keeps the code honest for
  // atoms where they differ. `simtAtomPaneHTML` lives in ui.js because the two
  // tiled-copy tabs draw this identical strip above their tile viz.
  for (const side of ['src', 'dst']) {
    const layoutStr = side === 'src' ? s.atomStr : s.atomDstStr;
    document.getElementById(`${tabId}-mca-${side}-svg`).innerHTML =
      simtAtomPaneHTML(side, layoutStr, s.elements, s.dtype);
    applyZoomState(`${tabId}-mca-${side}-svg`);
  }
  document.getElementById(`${tabId}-mca-atom-title`).textContent =
    `${s.op.label} — ${s.numBits}b / ${s.dtype} → ${s.elements} element${s.elements === 1 ? '' : 's'}`;
}

// The TV grid, once per side. This reuses the TV Layout tab's builder rather
// than a bespoke one because the question is identical — "which (thread, value)
// slot owns this cell of the tile" — and reusing it keeps the colour map, the
// collision stroke and the `value` overlay consistent with that tab.
//
// Two things are passed that the TV tab does not need:
//   dimTids   — the lanes `.x1` / `.x2` ignore (src pane only; on the dst side
//               every lane genuinely receives registers).
//   cellIndex — the atom's tile is ROW-major in its own codomain (flat = m*N+n,
//               one 16 B row per 8 rows of matrix), where the TV tab's tile is
//               col-major. Without this the `value` overlay would print a
//               number that is not the layout's output.
/** One ldmatrix/stmatrix atom pane (header line + SVG) for `mcaLdmatrixAtom`'s
 *  result. Shared with the make_tiled_copy / make_tiled_copy_tv atom strip, so
 *  the two pictures of the same atom cannot drift (the simtAtomPaneHTML rule). */
function mcaLdmatrixPaneHTML(a, side, dtype, labelMode) {
  const N = a.tile.shape[1];
  const dimTids = new Set();
  for (let t = a.liveLanes; t < 32; t++) dimTids.add(t);
  const L = side === 'src' ? a.src : a.dst;
  const str = side === 'src' ? a.srcStr : a.dstStr;
  const numV = product(L.shape[1]);
  const name = side === 'src' ? 'ValLayoutSrc' : 'ValLayoutDst';
  // The address side gets the live-lane count and the greyed broadcast lanes;
  // the register side is every lane's fragment.
  const head = side === a.addrSide
    ? `${name} = ${str} — ${a.liveLanes} of 32 lanes supply an address, ` +
      `${numV} ${dtype} per row`
    : `${name} = ${str} — ${numV} ${dtype} per lane` +
      (a.numMatrices > 1 ? `, ${a.numMatrices} matrices` : '');
  return `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">${head}</div>` +
    buildTVSVG(L.shape, L.stride, a.tile.shape, a.tile.stride, false, 'row', null, labelMode,
               { dimTids: side === a.addrSide ? dimTids : null,
                 cellIndex: (m, n) => m * N + n });
}

function mcaDrawLdmatrix(tabId, s) {
  const a = s.atom;
  const [M, N] = a.tile.shape;
  const labelMode = s.showValue ? 'value' : '';
  for (const side of ['src', 'dst']) {
    document.getElementById(`${tabId}-mca-${side}-svg`).innerHTML = mcaLdmatrixPaneHTML(a, side, s.dtype, labelMode);
    applyZoomState(`${tabId}-mca-${side}-svg`);
  }
  const q = `.x${a.numMatrices}${a.transpose ? '.trans' : ''}`;
  const shape = a.spec.matrix === '8x8' ? 'm8n8' : a.spec.matrix === '16x8' ? 'm16n8' : 'm16n16';
  document.getElementById(`${tabId}-mca-atom-title`).textContent =
    `${a.spec.store ? 'stmatrix' : 'ldmatrix'}.${shape}${q} — ${s.dtype} → ${M}×${N} tile, ${a.liveLanes}/32 lanes addressing`;
}


/** Rebuild the repeat picker for a TMEM Op; its domain differs per shape. */
function mcaSyncTmemControls(tabId, op, p = 'mca') {
  const sel = document.getElementById(`${tabId}-${p}-rep-input`);
  if (!sel) return;
  const spec = MCA_TMEM_SPECS[op.tmem], want = sel.value;
  const reps = MCA_TMEM_REPEATS.filter(r => r <= spec.maxRepeat);
  sel.innerHTML = reps.map(r => `<option value="${r}">x${r}</option>`).join('');
  sel.value = reps.map(String).includes(want) ? want : '1';
}

function mcaRenderTmem(tabId, opKey, op, dtype, elemBits, prev) {
  const repeat = parseInt(document.getElementById(`${tabId}-mca-rep-input`).value, 10);
  const a = mcaTmemAtom(op.tmem, elemBits, repeat, op.store);
  mcaState[tabId] = { kind: 'tmem', opKey, op, dtype, elemBits, atom: a, showValue: !!prev.showValue };
  const reg = a[a.regSide], perThread = product(reg.shape[1]);
  const regName = a.regSide === 'src' ? 'ValLayoutSrc' : 'ValLayoutDst';
  const tmemName = a.tmemSide === 'src' ? 'ValLayoutSrc' : 'ValLayoutDst';
  document.getElementById(`${tabId}-mca-atom-result`).innerHTML =
    `<div class="cuo-result-line"><b>Copy_Atom&lt;${op.label.replace('tcgen05.', '')}.x${repeat}, ${dtype}&gt;</b></div>` +
    `<div class="cuo-result-line" style="color:#9ca3af">tcgen05.${op.store ? 'st' : 'ld'}.sync.aligned${a.spec.ptx}.x${repeat} &mdash; SM100+</div>` +
    `<div class="cuo-result-line">ThrID        = 32:1<span style="color:#9ca3af"> &mdash; one warp; warp w reaches TMEM lanes 32&middot;(w mod 4) &hellip; +31</span></div>` +
    `<div class="cuo-result-line">ValLayoutSrc = ${a.srcStr}</div>` +
    `<div class="cuo-result-line">ValLayoutDst = ${a.dstStr}</div>` +
    `<div class="cuo-result-line">TMEM block = ${a.lanes} lanes &times; ${a.cols32} 32-bit columns` +
    (a.k > 1 ? ` = ${a.lanes} &times; ${a.cols} ${dtype}` : '') + `; ${perThread} ${dtype} per thread</div>` +
    `<div class="cuo-result-line" style="color:#9ca3af">${tmemName} has thread stride 0: the warp addresses the TMEM block as a whole, ` +
    `so every thread names all of it. ${regName} says which thread holds which element. ` +
    `<code>num_bits_per_copy</code> is not read: the width is the shape times repeat.</div>`;
  document.getElementById(`${tabId}-mca-viz-desc`).innerHTML =
    `One instruction's TMEM block: ${a.lanes} lanes (rows) &times; ${a.cols} ${dtype} columns. ` +
    `The <b>TMEM</b> pane labels each cell with its lane <code>l</code> and 32-bit column <code>c</code>` +
    (a.k > 1 ? ` (<code>c.e</code>: element e of the ${a.k} packed in that column)` : '') + `. ` +
    `The <b>RMEM</b> pane shows which thread and value slot holds it. ${a.spec.note}`;
  renderMcaAtomViz(tabId);
  updateOuterTabLabel(tabId, `make_copy_atom:tmem${a.spec.ptx}.x${repeat}/${dtype}`);
}

/** TMEM cell colours: make_copy_atom's violet for one warp's block, and
 *  shallower / deeper shades of it when a tiled copy spans four warps
 *  (index = warp mod 4, i.e. the lane quarter). */
const TMEM_WARP_BG = ['#ede9fe', '#ddd6fe', '#c4b5fd', '#a78bfa'];

/** One TMEM-atom pane (header line + SVG) for `mcaTmemAtom`'s result: the
 *  TMEM block labelled lane / 32-bit column, or the register side as a TV
 *  grid. Shared with make_tmem_copy's atom strip so the two cannot drift. */
function mcaTmemPaneHTML(a, side, dtype, showValue) {
  const [L, C] = a.tile.shape;
  const head = (txt) => `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">${txt}</div>`;
  const name = side === 'src' ? 'ValLayoutSrc' : 'ValLayoutDst', str = side === 'src' ? a.srcStr : a.dstStr;
  if (side === a.tmemSide) {
    return head(`${name} = ${str} — TMEM, addressed by the whole warp`) +
      buildColoredLayoutSVG([L, C], [C, 1], 'value', (m, n) => {
        const col = Math.floor(n / a.k), sub = n % a.k;
        const lines = [`l${m}`, a.k > 1 ? `c${col}.${sub}` : `c${col}`];
        if (showValue) lines.push(String(m * C + n));
        return { bg: TMEM_WARP_BG[0], text: lines };
      });
  }
  const reg = a[a.regSide];
  return head(`${name} = ${str} — ${product(reg.shape[1])} ${dtype} per thread`) +
    buildTVSVG(reg.shape, reg.stride, a.tile.shape, a.tile.stride, false, 'row', null, showValue ? 'value' : '',
               { cellIndex: (m, n) => m * C + n });
}

function mcaDrawTmem(tabId, s) {
  const a = s.atom, [L] = a.tile.shape;
  for (const side of ['src', 'dst']) {
    document.getElementById(`${tabId}-mca-${side}-svg`).innerHTML = mcaTmemPaneHTML(a, side, s.dtype, s.showValue);
    applyZoomState(`${tabId}-mca-${side}-svg`);
  }
  document.getElementById(`${tabId}-mca-atom-title`).textContent =
    `tcgen05.${a.store ? 'st' : 'ld'}${a.spec.ptx}.x${a.repeat} — ${s.dtype} → ${L} lanes × ${a.cols32} columns`;
}

function setMCA(tabId, opKey, bits, dtype, nm, tr) {
  document.getElementById(`${tabId}-mca-op-input`).value    = opKey;
  document.getElementById(`${tabId}-mca-bits-input`).value  = bits;
  document.getElementById(`${tabId}-mca-dtype-input`).value = dtype;
  // The num_matrices options are per-Op, so rebuild them BEFORE assigning —
  // otherwise `.value = '2'` on a select that still holds the previous Op's
  // options silently does nothing and the render uses a stale k.
  const op = MCA_OPS[opKey] || MCA_OPS.universal;
  mcaRenderOpParams(tabId, op);
  // For a TMEM Op the fourth field is its repeat.
  if (nm !== undefined) document.getElementById(`${tabId}-mca-${op.kind === 'tmem' ? 'rep' : 'nm'}-input`).value = String(nm);
  if (tr !== undefined && op.kind !== 'tmem') document.getElementById(`${tabId}-mca-trans-input`).value = String(tr);
  renderMakeCopyAtom(tabId);
}

function exportMCA(tabId) {
  const opKey = document.getElementById(`${tabId}-mca-op-input`).value;
  const base = [
    opKey,
    document.getElementById(`${tabId}-mca-bits-input`).value,
    document.getElementById(`${tabId}-mca-dtype-input`).value,
  ];
  // Only the LdMatrix Ops carry the extra parameters, so a CopyUniversalOp link
  // keeps its 3-input form. All LdMatrix links use 5 inputs; unpack_bits is fixed to None.
  const op = MCA_OPS[opKey] || {};
  if (op.kind === 'ldmatrix') {
    base.push(document.getElementById(`${tabId}-mca-nm-input`).value);
    base.push(document.getElementById(`${tabId}-mca-trans-input`).value);
  } else if (op.kind === 'tmem') {
    base.push(document.getElementById(`${tabId}-mca-rep-input`).value);
  }
  exportURL(`${tabId}-mca-export`, 'make_copy_atom', ...base);
}


// Decode one offered CopyAtom without changing a destination tab's controls.
function mcaDecodeAtomURL(text) {
  const parsed = parsePastedKeyParam(text);
  if (!parsed || parsed.feature !== 'make_copy_atom') throw new Error('Paste a make_copy_atom URL or its query string.');
  const a = parsed.inputs, opKey = a[0], numBits = Number(a[1]), dtype = a[2];
  const op = MCA_OPS[opKey], elemBits = DTYPE_BITS[dtype];
  if (!op || !elemBits || !Number.isSafeInteger(numBits) || numBits <= 0) throw new Error('Invalid CopyAtom Op, dtype, or bit count.');
  const nm = Number(a[3] || 1), trans = a[4] === '1';
  if (op.kind === 'ldmatrix') {
    if (a[4] !== undefined && !['0','1'].includes(a[4])) throw new Error('CopyAtom transpose must be 0 or 1.');
    mcaLdmatrixAtom(op.ldsm, elemBits, nm, trans);
  } else if (op.kind === 'tmem') {
    mcaTmemAtom(op.tmem, elemBits, nm, op.store);   // nm carries the repeat
  } else if (numBits % elemBits) throw new Error('CopyAtom bit count must be a multiple of the dtype width.');
  return {opKey, numBits, dtype, nm, trans};
}

function mcaImportAtom(tabId,p,render,text) {
  const a = mcaDecodeAtomURL(text);
  if (p === 'mca') { setMCA(tabId,a.opKey,String(a.numBits),a.dtype,a.nm,a.trans ? 1 : 0); return; }
  // make_tiled_copy(_tv) take the SIMT and the ldmatrix / stmatrix atoms; a
  // TMEM atom needs make_tmem_copy, which knows the warp-to-lane-quarter rule.
  if (!mtcOffersOp(a.opKey))
    throw new Error('This CopyAtom section takes SIMT, ldmatrix and stmatrix CopyAtoms. A tcgen05 TMEM atom ' +
                    'is tiled by make_tmem_copy, which follows the warp-to-TMEM-lane rule.');
  mtcWriteAtom(tabId, p, a.opKey, a.numBits, a.dtype, a.nm, a.trans ? 1 : 0);
  globalThis[render](tabId);
}

function mcaDecodeAtomPasteboard(tabId,p,render,errorPrefix=p) {
  return decodeFromPasteboard({tabId, errorId:`${tabId}-${errorPrefix}-error`, sources:['make_copy_atom'],
    message:'Paste the make_copy_atom URL or its query string:', importFn:text => mcaImportAtom(tabId,p,render,text)});
}
