// make_tiled_tma_atom tab (COPY scope): build ONE TMA Copy_Atom and visualize
// BOTH results the function returns — the atom (as a TMA descriptor + the box
// it moves) and the TMA coordinate tensor.
// Functions become globals on `window` (no module system).
//
// Scope for now: `cpasync.CopyBulkTensorTileG2SOp` only (plain `.tile` mode TMA
// load, GMEM -> SMEM), num_multicast fixed at 1. The DSL itself rejects
// num_multicast != 1 for this Op (cpasync/helpers.py:521-526); the multicast,
// store and im2col Ops are separate entries in the same `if` chain and belong
// in later revisions of this tab.
//
// The DSL entry point (python/CuTeDSL/cutlass/cute/nvgpu/cpasync/helpers.py:419)
// only packs arguments and hands them to MLIR, so the algorithm mirrored here is
// the C++ one — include/cute/atom/copy_traits_sm90_tma.hpp:
//
//   cta_v_map    = make_identity_layout(gmem.shape).compose(cta_tiler)   :1332
//   inv_smem     = right_inverse(get_nonswizzle_portion(slayout))        :761
//   sidx2gmode   = coalesce(composition(cta_v_map, inv_smem))            :765
//                  truncated at the first mode whose basis coef != 1     :775-784
//   tma_gstride  = coalesce_256(gtensor.compose(sidx2gmode))             :795
//   tma_gbasis   = same modes, but remembering WHICH gmem axis each came
//                  from, + size-1 modes for gmem axes the tile never
//                  touched, grouped to rank <= 5                         :802-843
//   descriptor   = fill_tma_gmem_shape_stride + smem_box_shape           :929-1060
//   tma tensor   = make_coord_tensor(make_layout(gmem.shape, g_stride_))  :152
//
// TMA is NOT a per-thread copy: one thread issues the instruction with a logical
// COORDINATE and the TMA unit does address generation, bounds handling and the
// swizzled SMEM write itself. So there is no TV layout to draw here — the
// `make_tiled_copy` thread panel deliberately has no analogue. What replaces it
// is the box: which gmem elements one instruction moves, and where they land.
//
// The BOX IS INFERRED, never given. `cta_tiler` says which region a CTA takes;
// `smem_layout` says in what order (and with which swizzle) it must land; the
// two together determine the descriptor.

const TMA_MAX_BOX_EXTENT = 256;   // 2^8, copy_traits_sm90_tma.hpp:1005-1014
const TMA_MAX_RANK = 5;           // cuTensorMapEncodeTiled's tensorRank limit

// The only Op this tab builds, kept in a table so the multicast / store / im2col
// variants slot in beside it rather than restructuring the form.
const TMA_OPS = {
  tma_g2s: {
    label: 'cpasync.CopyBulkTensorTileG2SOp',
    ctor: 'cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()',
    params: [],
    note: 'Plain <code>.tile</code>-mode bulk tensor load, GMEM &rarr; SMEM. ' +
          'Takes no constructor parameters. The DSL raises for ' +
          '<code>num_multicast != 1</code> on this Op — the multicast form is a ' +
          'different Op (<code>CopyBulkTensorTileG2SMulticastOp</code>).',
  },
};

// ═══════════════════════════════════════════════════════
//  Swizzle: CuTe's TMA smem swizzle is expressed in BYTES
// ═══════════════════════════════════════════════════════

/** `Sw<B,M,S>` on a TMA smem layout is NOT in the same units as the layout it
 *  is composed with. `upcast` on a `ComposedLayout` with an `smem_ptr_flag_bits`
 *  offset deliberately leaves the swizzle alone (pointer_flagged.hpp:73), so
 *  `GMMA::Layout_K_SW128_Atom<half_t>` keeps `Sw<3,4,3>` while its layout
 *  becomes `(8,64):(64,1)` in ELEMENTS. The triple is byte-based:
 *    base  = 2^M bytes   (M == 4 -> the 16B base `get_tma_swizzle_base` wants)
 *    width = 2^B * 2^M bytes  (B == 3, M == 4 -> the 128B swizzle)
 *  which is why `get_tma_swizzle_bits` (copy_traits_sm90_tma_swizzle.hpp:48)
 *  can insist on M == 4 no matter what element type the layout carries.
 *
 *  This repo's `applySwizzleOffset` swizzles an ELEMENT index, so drawing one of
 *  these over an element grid needs M shifted down by log2(bytes per element) —
 *  `Sw<3,4,3>` on half_t draws as `Sw<3,3,3>`. Getting this wrong silently
 *  produces a plausible-looking but wrong picture, so it is converted once here
 *  and reported in the result box. */
function tmaSwizzleInfo(sw, elemBits) {
  if (!sw) {
    return { sw: null, swElem: null, baseBytes: 0, widthBytes: 0,
             enumName: 'CU_TENSOR_MAP_SWIZZLE_NONE', str: 'none' };
  }
  const { B, M, S } = sw;
  let baseBytes, enumBits;
  if (M === 4) {
    if (B < 0 || B > 3) throw new Error(
      `Swizzle<${B},${M},${S}>: expected B = 0, 1, 2 or 3 when M == 4 ` +
      `(copy_traits_sm90_tma_swizzle.hpp:51). B selects the swizzle width: ` +
      `0 = none, 1 = 32B, 2 = 64B, 3 = 128B.`);
    if (S !== 3) throw new Error(
      `Swizzle<${B},${M},${S}>: expected S = 3 when M == 4 ` +
      `(copy_traits_sm90_tma_swizzle.hpp:82). S is fixed by the 16B base: the ` +
      `swizzled row is 2^S = 8 base units = 128B.`);
    baseBytes = 16;
    enumBits = [0, 32, 64, 128][B];
  } else if (M === 5) {
    if (B !== 2) throw new Error(
      `Swizzle<${B},${M},${S}>: expected B = 2 when M == 5 (32B base, SM100 form).`);
    if (S !== 2) throw new Error(
      `Swizzle<${B},${M},${S}>: expected S = 2 when M == 5 (32B base, SM100 form).`);
    baseBytes = 32;
    enumBits = 128;
  } else if (M === 6) {
    if (B !== 2) throw new Error(
      `Swizzle<${B},${M},${S}>: expected B = 2 when M == 6 (64B base, SM100 form).`);
    baseBytes = 64;
    enumBits = 128;
  } else {
    throw new Error(
      `Swizzle<${B},${M},${S}>: expected 128b = 16B = (2^4)B to 512b = 64B = (2^6)B ` +
      `base swizzle, i.e. M in {4, 5, 6} (copy_traits_sm90_tma_swizzle.hpp:95). ` +
      `Remember the TMA swizzle triple is in BYTES, not elements — CuTe prints ` +
      `the 128B swizzle as Sw<3,4,3> even for a half_t layout.`);
  }

  const elemBytes = elemBits / 8;
  if (elemBytes < 1) throw new Error(
    `Sub-byte element types are not supported by this tab (${elemBits}b).`);
  const shift = Math.log2(elemBytes);
  if (!Number.isInteger(shift)) throw new Error(
    `Element size ${elemBytes}B is not a power of two.`);
  const Melem = M - shift;
  if (Melem < 0) throw new Error(
    `Swizzle base 2^${M} = ${baseBytes}B is smaller than one ${elemBytes}B element, ` +
    `so the swizzle cannot be drawn over an element grid.`);

  return {
    sw,
    swElem: { B, M: Melem, S },        // same swizzle, indexed by ELEMENT
    baseBytes,
    widthBytes: enumBits === 0 ? 0 : (1 << B) * baseBytes,
    enumName: enumBits === 0 ? 'CU_TENSOR_MAP_SWIZZLE_NONE'
                             : `CU_TENSOR_MAP_SWIZZLE_${enumBits}B`,
    str: `Sw<${B},${M},${S}>`,
    elemStr: `Sw<${B},${Melem},${S}>`,
  };
}

/** The swizzle is a closed set at the descriptor level — `CUtensorMapSwizzle` has
 *  exactly NONE / 32B / 64B / 128B — so it is a picker, not a free text field.
 *  Each option is labelled with BOTH spellings: the `Sw<B,M,S>` CuTe prints and
 *  the byte width people actually think in. Values are the bare triple so
 *  `parseSwizzleSpec` reads them unchanged.
 *
 *  Omitted: the M == 6 (64B base) form. `get_tma_swizzle_base` accepts it but
 *  constrains only B, and CUTLASS's own assert message for it is a copy-paste of
 *  the M == 5 one, so the right S is not something to guess at in a dropdown. It
 *  still works if pasted inline (see tmaParseSmemField). */
const TMA_SWIZZLE_CHOICES = [
  ['none',  'none'],
  ['1,4,3', 'Sw<1,4,3> — 32B'],
  ['2,4,3', 'Sw<2,4,3> — 64B'],
  ['3,4,3', 'Sw<3,4,3> — 128B'],
  ['2,5,2', 'Sw<2,5,2> — 128B/32B'],
];
const TMA_SWIZZLE_OPTIONS = TMA_SWIZZLE_CHOICES.map(([v, label]) =>
  `<option value="${v}"${v === '3,4,3' ? ' selected' : ''}>${label.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</option>`
).join('');

// ═══════════════════════════════════════════════════════
//  Parsing
// ═══════════════════════════════════════════════════════

/** Split a CuTe ComposedLayout printout on its top-level " o " separators, so
 *  `Sw<3,4,3> o smem_ptr[16b](0) o (8,64):(64,1)` pastes in verbatim. Angle
 *  brackets count as nesting so a swizzle's commas can't be mistaken for a
 *  layout's. (ui.js's `topLevelCompose` finds only the FIRST separator and is
 *  used for the `<origin> o <layout>` coordinate-tensor form.) */
function tmaSplitComposed(str) {
  const parts = [];
  let depth = 0, start = 0;
  for (let i = 0; i < str.length; i++) {
    const c = str[i];
    if (c === '(' || c === '<' || c === '[') depth++;
    else if (c === ')' || c === '>' || c === ']') depth--;
    else if (c === 'o' && depth === 0) {
      const before = i === 0 ? ' ' : str[i - 1];
      const after = i + 1 >= str.length ? ' ' : str[i + 1];
      if (/\s/.test(before) && /\s/.test(after)) { parts.push(str.slice(start, i)); start = i + 1; }
    }
  }
  parts.push(str.slice(start));
  return parts.map(s => s.trim()).filter(Boolean);
}

/** Read the smem_layout field: an optional `Sw<B,M,S>` prefix, an optional
 *  `smem_ptr[Nb](0)` marker (carried by CuTe's printout, no effect here), and
 *  the layout itself. */
function tmaParseSmemField(raw) {
  const parts = tmaSplitComposed(raw || '');
  let swStr = null, layoutStr = null;
  const ignored = [];
  for (const p of parts) {
    if (/^sw/i.test(p)) swStr = p;
    else if (/^smem_ptr/i.test(p)) ignored.push(p);
    else layoutStr = p;
  }
  if (!layoutStr) throw new Error(
    'smem_layout is empty — give it a layout like (64,64):(64,1), optionally ' +
    'prefixed with a swizzle: Sw<3,4,3> o (64,64):(64,1).');
  const sw = swStr ? parseSwizzleSpec(swStr) : null;
  if (swStr && !sw) throw new Error(
    `Cannot read the swizzle "${swStr}" — expected Sw<B,M,S>, e.g. Sw<3,4,3>. ` +
    `You do not need to write one here at all: use the swizzle picker below the ` +
    `layout box. This prefix exists only so a CuTe printout pastes verbatim.`);
  return { sw, layoutStr, ignored };
}

/** Flat col-major offset of domain index `d` under a flattened shape/stride. */
function tmaFlatOffset(d, fshape, fstride) {
  let rem = d, off = 0;
  for (let k = 0; k < fshape.length; k++) {
    off += (rem % fshape[k]) * fstride[k];
    rem = Math.floor(rem / fshape[k]);
  }
  return off;
}

// ═══════════════════════════════════════════════════════
//  The TMA construction itself
// ═══════════════════════════════════════════════════════

/** Recover a basis-strided layout from the FUNCTION `coordOf(i) -> [i0, i1]`.
 *
 *  CuTe computes `coalesce(composition(cta_v_map, inv_smem))` symbolically over
 *  scaled-basis strides. This tool only ever draws 2-D, so the same result is
 *  obtained by observation: read off each mode's stride as the first coordinate
 *  delta, then extend the mode as long as the block repeats. A composition of
 *  layouts IS a layout, so the greedy decomposition is exact — and because
 *  neighbouring modes only merge when their deltas agree, the result is already
 *  coalesced, exactly like CuTe's.
 *
 *  A mode whose delta moves along TWO axes at once means the composition is not
 *  defined (the tile does not divide the smem layout); CuTe would fail to
 *  compile, so we say so. */
function tmaInferBasisModes(coordOf, size) {
  const modes = [];
  const c0 = coordOf(0);
  let pos = 1;
  while (pos < size) {
    const cp = coordOf(pos);
    const d = [cp[0] - c0[0], cp[1] - c0[1]];
    const nz = [];
    for (let a = 0; a < 2; a++) if (d[a] !== 0) nz.push(a);
    if (nz.length !== 1 || d[nz[0]] < 0) {
      throw new Error(
        `smem_layout and cta_tiler are not composable: advancing ${pos} element` +
        `${pos === 1 ? '' : 's'} in SMEM moves the GMEM coordinate by ` +
        `(${d[0]}, ${d[1]}), which is not a single scaled basis element. ` +
        `composition(cta_v_map, right_inverse(smem_layout)) is undefined here — ` +
        `CuTe fails to compile rather than producing this. Usually the smem ` +
        `layout's mode sizes do not divide the tiler's.`);
    }
    const axis = nz[0], coef = d[axis];
    let e = 1;
    while ((e + 1) * pos <= size) {
      let ok = true;
      for (let q = 0; q < pos && ok; q++) {
        const a = coordOf(e * pos + q), b = coordOf(q);
        if (a[0] !== b[0] + e * d[0] || a[1] !== b[1] + e * d[1]) ok = false;
      }
      if (!ok) break;
      e++;
    }
    if (e === 1) {
      throw new Error(
        `smem_layout and cta_tiler are not composable: the SMEM -> GMEM map is ` +
        `not a layout function (no repeating mode found at stride ${coef}@${axis}).`);
    }
    modes.push({ extent: e, axis, coef });
    pos *= e;
  }
  return modes;
}

/** cute::coalesce_256 (copy_traits_sm90_tma.hpp:676-727), merging modes only
 *  while the merged extent stays within the 256 the TMA box allows. Each
 *  resulting mode remembers the flat modes it swallowed, which is what
 *  `tile_gbasis.compose(make_layout(shape(tma_gstride)))` produces symbolically:
 *  one TMA mode can span several gmem axes when the tensor is contiguous. */
function tmaCoalesce256(flat) {
  const out = [];
  for (const m of flat) {
    if (m.extent === 1) continue;
    const back = out[out.length - 1];
    if (!back || back.extent === 1) {
      if (back) { out.pop(); }
      out.push({ extent: m.extent, stride: m.stride, segs: [m] });
    } else if (back.extent * back.stride === m.stride &&
               m.extent * back.extent <= TMA_MAX_BOX_EXTENT) {
      back.extent *= m.extent;
      back.segs.push(m);
    } else {
      out.push({ extent: m.extent, stride: m.stride, segs: [m] });
    }
  }
  if (out.length === 0) out.push({ extent: 1, stride: 0, segs: [] });
  return out;
}

function tmaGcd(a, b) { while (b) { const t = a % b; a = b; b = t; } return a; }

/** Everything `make_tiled_tma_atom` computes, for a rank-2 GMEM tensor.
 *  Returns the atom facts, the descriptor fields, the TMA tensor's strides, and
 *  a per-SMEM-index table for the visualization.
 *
 *  Violations that CuTe catches with a HOST `assert()` (compiled out in release
 *  builds, and invisible from the DSL) are collected in `issues` rather than
 *  thrown: the box is still well-defined, and seeing the wrong picture next to
 *  the reason is more useful than an empty panel. Only genuinely uncomputable
 *  inputs throw. */
function tmaComputeAtom(cfg) {
  const { gShape, gStride, elemBits, swInfo, sFlatShape, sFlatStride, tiler } = cfg;
  const issues = [];
  const elemBytes = elemBits / 8;

  const [T0, T1] = tiler.extents;
  const [ts0, ts1] = tiler.strides;
  const tileSize = T0 * T1;
  let sSize = 1;
  for (const s of sFlatShape) sSize *= s;

  // copy_traits_sm90_tma.hpp:748 — the one shape relationship TMA insists on.
  // Note it is SIZE, not shape: an (8,512) smem layout against a (64,64) tiler
  // is legal, and reshaping like that is how a wider box gets built.
  if (sSize !== tileSize) {
    throw new Error(
      `TMA requires size(smem_layout) == size(cta_tiler), but ` +
      `${sSize} != ${tileSize} (${T0}x${T1}). This is CuTe's static assert ` +
      `"TMA requires CTA_Tile and SLayout top-level size equivalence" ` +
      `(copy_traits_sm90_tma.hpp:748). Only the sizes must match, not the shapes.`);
  }
  if ((T0 - 1) * ts0 >= gShape[0] || (T1 - 1) * ts1 >= gShape[1]) {
    throw new Error(
      `cta_tiler (${T0}${ts0 !== 1 ? ':' + ts0 : ''}, ${T1}${ts1 !== 1 ? ':' + ts1 : ''}) ` +
      `does not fit inside the gmem tensor (${gShape[0]}, ${gShape[1]}) — the tile ` +
      `would index past the end of the tensor.`);
  }

  // right_inverse(get_nonswizzle_portion(slayout)): smem offset -> smem coord.
  // A non-bijective smem layout does not fail here in CuTe: right_inverse simply
  // returns a PARTIAL inverse, which silently shrinks the box.
  const inv = new Array(sSize).fill(-1);
  for (let d = 0; d < sSize; d++) {
    const off = tmaFlatOffset(d, sFlatShape, sFlatStride);
    if (off < 0 || off >= sSize || inv[off] !== -1) {
      throw new Error(
        `smem_layout is not compact: its image is not exactly [0, ${sSize}) ` +
        (off >= sSize || off < 0
          ? `(coordinate ${d} maps to offset ${off}, outside the layout's own size — padding or holes)`
          : `(coordinates ${inv[off]} and ${d} both map to offset ${off})`) +
        `. TMA writes a contiguous box into SMEM, so the layout must be a ` +
        `permutation. CuTe does not check this — right_inverse() returns a ` +
        `partial inverse and the box silently comes out too small.`);
    }
    inv[off] = d;
  }

  // sidx2gmode: smem offset -> gmem coordinate, via the CTA tile.
  // cta_v_map = make_identity_layout(gmem.shape).compose(cta_tiler), so tile
  // element (m, n) sits at gmem coordinate (m*ts0, n*ts1).
  const coordOf = (i) => {
    const d = inv[i];
    const m = d % T0;
    return [m * ts0, ((d - m) / T0) * ts1];
  };
  const full = tmaInferBasisModes(coordOf, sSize);

  // Truncate at the first mode whose basis coefficient isn't a static 1 — the
  // leading run where +1 SMEM element means +1 GMEM element. That run is the
  // TMA vectorization. copy_traits_sm90_tma.hpp:775-784.
  let smemRank = 0;
  while (smemRank < full.length && full[smemRank].coef === 1) smemRank++;
  if (smemRank === 0) {
    throw new Error(
      `Could not find a common tile-gmem vectorization: the fastest-varying SMEM ` +
      `mode advances the GMEM coordinate by ${full[0].coef} along axis ` +
      `${full[0].axis}, not by 1. This is CuTe's static assert "Could not find a ` +
      `common tile-gmem vectorization. Does the Tile select out major GMEM ` +
      `modes?" (copy_traits_sm90_tma.hpp:781). A strided cta_tiler mode on the ` +
      `contiguous axis does exactly this.`);
  }
  const kept = full.slice(0, smemRank);
  const dropped = full.slice(smemRank);

  // tile_gstride: the same modes, in gmem OFFSETS. (recast<TmaInternalType> is a
  // no-op here — internal_type is out of scope for this tab, so the TMA internal
  // type is the tensor's element type.)
  const tileG = kept.map(m => ({ extent: m.extent, axis: m.axis, coef: m.coef,
                                 stride: m.coef * gStride[m.axis] }));
  const box = tmaCoalesce256(tileG);

  // Append size-1 modes for gmem axes the tile never touched, so the descriptor
  // still has a dimension (and therefore a coordinate) for them. :811-840
  const usedAxes = new Set();
  for (const b of box) for (const s of b.segs) usedAxes.add(s.axis);
  for (let a = 0; a < 2; a++) {
    if (gShape[a] === 1 || gStride[a] === 0) continue;   // no contribution
    if (usedAxes.has(a)) continue;
    box.push({ extent: 1, stride: gStride[a], segs: [{ extent: 1, axis: a, coef: 1 }] });
  }
  if (box.length > TMA_MAX_RANK) {
    issues.push({ level: 'error', text:
      `TMA descriptor rank ${box.length} > ${TMA_MAX_RANK}. CuTe groups trailing ` +
      `modes to stay within the limit; a rank-2 tensor cannot reach it.` });
  }

  // fill_tma_gmem_shape_stride (:858-905). Note the shapes are the FULL gmem
  // extents of the contributing axes, not the box extents.
  const probShape = [], probStride = [];
  for (const b of box) {
    if (b.segs.length === 1) {
      probShape.push(gShape[b.segs[0].axis]);
      probStride.push(gStride[b.segs[0].axis]);
    } else {
      let ps = 0, pd = 0;
      for (const seg of b.segs) {
        const shapeJ = gShape[seg.axis], strideJ = gStride[seg.axis];
        const oldStride = pd;
        pd = tmaGcd(pd, strideJ);
        if (pd !== 0) {
          ps = (ps - 1) * (oldStride / pd) + (shapeJ - 1) * (strideJ / pd) + 1;
        } else {
          ps = shapeJ;
        }
      }
      probShape.push(ps);
      probStride.push(pd);
    }
  }

  // The descriptor stores no stride for dim 0 and assumes it is one element.
  // :969 — "Majorness of smem doesn't match majorness of gmem".
  if (probStride[0] !== 1) {
    const smemMajorAxis = box[0].segs.length ? box[0].segs[0].axis : 0;
    const gmemMajorAxis = gStride[0] === 1 ? 0 : (gStride[1] === 1 ? 1 : -1);
    issues.push({ level: 'error', text:
      `gmem_prob_stride[0] = ${probStride[0]}, must be 1 — "Majorness of smem ` +
      `doesn't match majorness of gmem" (copy_traits_sm90_tma.hpp:969). ` +
      `smem_layout runs fastest along gmem axis ${smemMajorAxis}, but the ` +
      `contiguous gmem axis is ` +
      (gmemMajorAxis === -1 ? `neither (no stride-1 mode)` : `axis ${gmemMajorAxis}`) +
      `. Transpose the smem layout, or give the tensor the other majorness.` });
  }
  const probStrideBytes = probStride.map(s => s * elemBytes);
  for (let j = 1; j < probStrideBytes.length; j++) {
    if (probStrideBytes[j] % 16 !== 0) {
      issues.push({ level: 'error', text:
        `gmem_prob_stride[${j}] = ${probStrideBytes[j]}B is not a multiple of 16B ` +
        `(copy_traits_sm90_tma.hpp:981). Every TMA gmem stride except the ` +
        `innermost must be 16B-aligned.` });
    }
  }
  for (let j = 0; j < box.length; j++) {
    if (box[j].extent > TMA_MAX_BOX_EXTENT) {
      issues.push({ level: 'error', text:
        `smem_box_shape[${j}] = ${box[j].extent} > ${TMA_MAX_BOX_EXTENT} ` +
        `(copy_traits_sm90_tma.hpp:1006). Split the mode — a box extent is ` +
        `8 bits wide in the descriptor.` });
    }
  }

  // Driver-level (cuTensorMapEncodeTiled), not something CuTe asserts: the
  // innermost box row is what the swizzle pattern repeats over, so it has to be
  // 16B-aligned and no wider than the swizzle itself.
  const innerBoxBytes = box[0].extent * elemBytes;
  if (innerBoxBytes % 16 !== 0) {
    issues.push({ level: 'warn', text:
      `Innermost box = ${box[0].extent} x ${elemBytes}B = ${innerBoxBytes}B is not a ` +
      `multiple of 16B. cuTensorMapEncodeTiled rejects this at descriptor ` +
      `creation; CuTe does not check it.` });
  }
  if (swInfo.widthBytes > 0 && innerBoxBytes > swInfo.widthBytes) {
    issues.push({ level: 'warn', text:
      `Innermost box = ${innerBoxBytes}B is wider than the ${swInfo.widthBytes}B ` +
      `swizzle. The swizzle pattern repeats every ${swInfo.widthBytes}B, so the box ` +
      `row must fit inside one period — cuTensorMapEncodeTiled rejects the ` +
      `descriptor. Either narrow the box or use a wider swizzle.` });
  }

  let boxSize = 1;
  for (const b of box) boxSize *= b.extent;
  const numBitsPerTma = boxSize * elemBits;
  const instructions = sSize / boxSize;

  // The TMA tensor's strides (:1090-1120): for each gmem axis, which TMA
  // coordinate does moving one step along it change, and by how much.
  const tmaTensorStride = [];
  for (let a = 0; a < 2; a++) {
    if (gShape[a] === 1 || gStride[a] === 0) { tmaTensorStride.push(0); continue; }
    let j = -1;
    for (let k = 0; k < box.length && j === -1; k++) {
      if (box[k].segs.some(s => s.axis === a)) j = k;
    }
    if (j === -1) { tmaTensorStride.push(0); continue; }
    if (j === 0) {
      tmaTensorStride.push(makeBasis(gStride[a], 0));       // recast_ratio == 1
    } else if (box[j].segs.length === 1) {
      tmaTensorStride.push(makeBasis(1, j));
    } else {
      const scale = Math.ceil((gStride[a] * elemBits) /
                              Math.max(probStrideBytes[j], 16) / 8);
      tmaTensorStride.push(makeBasis(scale, j));
    }
  }

  // Per-SMEM-offset table for the viz: everything both panes need, keyed by the
  // one quantity that links them — the SMEM offset, i.e. the TMA fetch order.
  const cells = new Array(sSize);
  for (let i = 0; i < sSize; i++) {
    const c = coordOf(i);
    cells[i] = {
      smemIdx: i,
      tileIdx: inv[i],
      gcrd: c,
      goff: c[0] * gStride[0] + c[1] * gStride[1],
      inst: Math.floor(i / boxSize),
      boxIdx: i % boxSize,
    };
  }

  return {
    issues, full, kept, dropped, smemRank, box, boxSize, instructions,
    probShape, probStride, probStrideBytes, numBitsPerTma, innerBoxBytes,
    tmaTensorStride, cells, inv, sSize, elemBytes,
  };
}

/** Escape a CuTe printout for HTML. `Sw<3,4,3>` is otherwise eaten as a tag. */
function tmaEsc(str) {
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/** `k@i`-style rendering of a basis stride tuple, matching CuTe's printout. */
function tmaFormatBasisStride(stride) {
  return '(' + stride.map(s => {
    if (isBasis(s)) return `${s.k}@${s.axis}`;
    return String(s);
  }).join(',') + ')';
}

/** The flat sidx2gmode modes as CuTe would print them: `(e0,e1):(c0@a0,c1@a1)`. */
function tmaFormatModes(modes) {
  if (modes.length === 0) return '1:0';
  return `(${modes.map(m => m.extent).join(',')}):` +
         `(${modes.map(m => `${m.coef}@${m.axis}`).join(',')})`;
}

// ═══════════════════════════════════════════════════════
//  Tab template
// ═══════════════════════════════════════════════════════

function generateMakeTiledTmaAtomTabContent(id) {
  return `
    <!-- make_tiled_tma_atom panel -->
    <div id="${id}-tab-make_tiled_tma_atom" class="panel">
      <div class="controls">
        <h2>make_tiled_tma_atom</h2>

        <details class="cuo-section" open>
          <summary>0. Memory movement</summary>
          <div class="cuo-section-body">
${copyMoveField(id, 'tma')}
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>1. The TMA Op</summary>
          <div class="cuo-section-body">
            <div class="form-group">
              <label>TMA operation</label>
              <select id="${id}-tma-op-input" disabled>
                <option value="tma_g2s" selected>cpasync.CopyBulkTensorTileG2SOp</option>
              </select>
            </div>
            <div class="form-group">
              <label>num_multicast<span style="color:#6b7280;font-weight:normal">&nbsp;&mdash; fixed at 1: this Op is the non-multicast form</span></label>
              <input type="number" id="${id}-tma-multicast-input" value="1" disabled>
            </div>
            <div id="${id}-tma-op-params" class="cuo-result"></div>
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>2. gmem_tensor</summary>
          <div class="cuo-section-body">
            ${layoutInputField({ id: `${id}-tma-gmem-input`, label: 'GMEM tensor &mdash; shape:stride', value: '(32, 64):(64, 1)', hint: 'flat rank 2' })}
            <div class="form-group">
              <label>element type</label>
              <select id="${id}-tma-dtype-input">${dtypeOptions('float')}</select>
            </div>
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>3. smem_layout</summary>
          <div class="cuo-section-body">
            ${layoutInputField({ id: `${id}-tma-smem-input`, label: 'SMEM layout', value: '(8, 32):(32, 1)', hint: 'the unswizzled layout' })}
            <div class="form-group">
              <label>swizzle<span style="color:#6b7280;font-weight:normal">&nbsp;&mdash; the triple is in BYTES. <code>Sw&lt;3,4,3&gt;</code> (128B) is the usual one; the 32B-base form is SM100-only</span></label>
              <select id="${id}-tma-swizzle-input">${TMA_SWIZZLE_OPTIONS}</select>
            </div>
            <div id="${id}-tma-swizzle-note" class="cuo-result"></div>
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>4. cta_tiler</summary>
          <div class="cuo-section-body">
            ${layoutInputField({ id: `${id}-tma-tiler-input`, label: 'CTA tiler', value: '(8, 32)', hint: 'a Tiler: (8, 32) or (8:2, 32)' })}
            <div id="${id}-tma-result" class="cuo-result"></div>
          </div>
        </details>

        ${statusDivs(`${id}-tma`)}
        <button class="btn btn-render" onclick="renderMakeTiledTmaAtom('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-tma-export" onclick="exportTMA('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setTMA('${id}','uint128_t','(16, 16):(16, 1)','3,4,3','(8, 8):(8, 1)','(8, 8)')">8&times;8, one 16B element per cell &mdash; the 128B swizzle at its smallest</button>
            <button class="preset-btn" onclick="setTMA('${id}','float','(32, 64):(64, 1)','3,4,3','(8, 32):(32, 1)','(8, 32)')">8&times;32 float + 128B swizzle &mdash; the default, 1 KB box</button>
            <button class="preset-btn" onclick="setTMA('${id}','float','(32, 64):(64, 1)','none','(8, 32):(32, 1)','(8, 32)')">Same tile, no swizzle &mdash; identical box, SMEM unpermuted</button>
            <button class="preset-btn" onclick="setTMA('${id}','half_t','(32, 128):(128, 1)','3,4,3','(8, 64):(64, 1)','(8, 64)')">GMMA::Layout_K_SW128_Atom&lt;half_t&gt; &mdash; the real SMEM atom, 8&times;64</button>
            <button class="preset-btn" onclick="setTMA('${id}','half_t','(8, 8):(8, 1)','none','(1, 8):(8, 1)','(1, 8)')">1&times;8 half &mdash; the smallest legal atom, 16 B per instruction</button>
            <button class="preset-btn" onclick="setTMA('${id}','half_t','(256, 128):(128, 1)','3,4,3','(64, 64):(64, 1)','(64, 64)')">64&times;64 half &mdash; a full GEMM stage, 8 KB box (big)</button>
            <button class="preset-btn" onclick="setTMA('${id}','float','(32, 64):(64, 1)','3,4,3','(8, 64):(64, 1)','(8, 64)')">Box row 256 B vs a 128 B swizzle &mdash; rejected by the driver</button>
            <button class="preset-btn" onclick="setTMA('${id}','float','(32, 64):(64, 1)','3,4,3','(8, 32):(1, 8)','(8, 32)')">M-major SMEM over a K-major tensor &mdash; majorness mismatch</button>
            <button class="preset-btn" onclick="setTMA('${id}','float','(32, 64):(64, 1)','3,4,3','(8, 32):(32, 1)','(8:2, 32)')">Strided tiler (8:2, 32) &mdash; box truncated, 8 instructions per tile</button>
          </div>
        </div>

        <div class="hint">
          <b>The box is inferred, never given.</b>
          <code>cta_tiler</code> says <em>which</em> region of the tensor a CTA
          takes; <code>smem_layout</code> says <em>in what order and with which
          swizzle</em> it must land. Between them they fix the descriptor, which
          is why this function's docstring says it "figures out the bulk tensor
          instruction with the maximum TMA vector length". Invert the SMEM
          layout, compose with the tiler, and the leading stride-1 run is the
          vector.<br><br>
          <b>TMA has no TV layout.</b> One thread issues the instruction with a
          logical <em>coordinate</em>; the TMA unit does address generation,
          bounds handling and the swizzled SMEM write. So there is nothing to
          partition across threads and no per-thread panel here &mdash; the
          <code>TiledCopy</code> wrapper's TV layout is degenerate
          (<code>cta_t_map = 1</code> without multicast).<br><br>
          <b>The swizzle triple is in BYTES.</b> CuTe prints the 128B swizzle as
          <code>Sw&lt;3,4,3&gt;</code> whatever the element type, because
          <code>upcast</code> on a <code>ComposedLayout</code> leaves the swizzle
          alone (<code>pointer_flagged.hpp:73</code>): base
          <code>2^M</code> bytes, width <code>2^B &middot; 2^M</code> bytes. The
          <b>Swizzle</b> tab swizzles an <em>element</em> index, so the same
          swizzle is drawn there as <code>Sw&lt;3,3,3&gt;</code> for
          <code>half_t</code>. Both forms are printed below.<br><br>
          <b>Checks CuTe compiles out.</b> Most TMA constraints are host-side
          <code>assert()</code>s inside <code>make_tma_copy_desc</code>, so they
          vanish in a release build and are invisible from the DSL. They are
          reported inline in the descriptor panel instead of erroring out, so you
          can see the wrong picture next to the reason.<br><br>
          <b>Why the small presets use wide element types.</b> A swizzle only
          does something once the SMEM tile is large enough to reach its source
          bits. <code>Sw&lt;3,4,3&gt;</code> XORs the byte offset's bits
          <code>[7,10)</code> into <code>[4,7)</code>, so it needs 8 rows of
          128B &mdash; that is 8&times;64 <code>half_t</code>, but only
          8&times;8 <code>uint128_t</code>. Same swizzle, same 128B rows, a 64th
          of the cells. The <code>swizzled cells</code> the DST pane shows
          moving is the check: if no cell moves, the tile is too small to
          exercise the swizzle you asked for.<br><br>
          <b>Not covered yet.</b> Multicast, TMA store, im2col,
          <code>gather4</code>/<code>scatter4</code>, <code>internal_type</code>
          recasts, rank &gt; 2 tensors, and <code>tma_partition</code>.
        </div>
      </div>

      <div class="comp-results" style="grid-template-columns:1fr">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-tma-box-title">CTA tile &mdash; GMEM to SMEM, coloured by TMA box</span>
            <span style="display:flex;align-items:center;gap:4px">
              ${copyDirButtons(id, 'tma')}
              <span class="mode-btn-group" id="${id}-tma-box-mode-btns">
                <button class="mode-btn" onclick="setTmaMode('${id}','value')">value</button>
              </span>
              <button class="mode-btn" id="${id}-tma-src-svg-zoom" onclick="toggleCopyZoom('${id}','tma')">Zoom in</button>
              <button class="mode-btn" onclick="downloadSVG('${id}-tma-src-svg', 'tma_box.svg')">Download SVG</button>
            </span>
          </div>
          <div class="cuo-viz-desc">
            One CTA tile: <b>SRC</b> is it in GMEM, <b>DST</b> the same elements
            in SMEM. Colour is shared &mdash; same colour = same box row, so
            matching bands are the elements one contiguous TMA fetch moves, and
            when a tile needs several instructions the later ones are darker.<br><br>
            <b>The grid is the tile, not the box.</b> The box is not a different
            region you could outline &mdash; it is the same cells, described the
            way the descriptor addresses them, so its shape lives in the header
            and the colour bands rather than in the picture's outline. A
            <code>(32,8)</code> box is 8 bands of 32; a <code>(64)</code> box over
            the same 64 cells is one band, because the tile happens to be
            contiguous in GMEM and the descriptor collapsed it to a single run.
            When the box is smaller than the tile, one instruction is one
            brightness, and the header says how many there are.<br><br>
            Every cell names the element it holds by its <b>GMEM coordinate</b>
            <code>(i, j)</code> &mdash; the mapping's input, and the thing the TMA
            instruction actually carries. <code>value</code> adds what that
            coordinate maps to in each space: the element's linear offset from the
            tensor base on SRC, and its offset in shared memory on DST, swizzle
            included. Since the same tuple appears in both panes, finding it twice
            is the copy: that is where the element starts and where it lands.<br><br>
            The SMEM value is the <em>composed</em> layout's output, so it already
            has the swizzle in it. To see what the swizzle did, set the swizzle
            picker to <code>none</code> and compare &mdash; the header reports how
            many cells it moves. Offsets are in elements; multiply by the element
            size for a byte address.
          </div>
${copyPanes(id, 'tma')}
        </div>

        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-tma-tensor-title">TMA tensor (GMEM) &mdash; the returned coordinate tensor</span>
            <span style="display:flex;align-items:center;gap:4px">
              <button class="mode-btn" id="${id}-tma-tensor-svg-zoom" onclick="toggleZoom('${id}-tma-tensor-svg')">Zoom in</button>
            </span>
          </div>
          <div class="cuo-viz-desc">
            The second return value, and always the <b>GMEM-side</b> tensor: with
            this load Op that makes it the SRC, but the same
            <code>get_tma_tensor</code> exists on
            <code>Copy_Traits&lt;SM90_TMA_STORE&gt;</code>
            (<code>copy_traits_sm90_tma.hpp:389</code>), where GMEM is the
            destination. SMEM never gets one &mdash; it is an ordinary layout over
            a flat buffer. Its strides are scaled basis elements, so it maps a
            GMEM coordinate to the <em>TMA coordinate</em> the instruction
            consumes, not to an address; partitioning it
            (<code>local_tile</code>, <code>tma_partition</code>) yields those
            coordinates. Cell colour keys off TMA axis 0, so a swap between CuTe
            mode order and descriptor dimension order shows up as bands turning.
          </div>
          <div class="viz-box"><div id="${id}-tma-tensor-svg"></div></div>
        </div>
        <div class="comp-viz-item collapsed">
          <div class="comp-viz-header">
            <span class="comp-viz-label">TMA descriptor &mdash; cuTensorMapEncodeTiled arguments</span>
          </div>
          <div class="cuo-viz-desc">
            The 128-byte <code>CUtensorMap</code> the TMA unit actually reads:
            your instruction carries only a pointer to it plus a coordinate, and
            it does the address arithmetic. Every field is derived &mdash; you
            entered none of them. Folded by default; expand when you need to check
            what the descriptor really says.
          </div>
          <div class="viz-box" style="background:#0b1220;color:#d1d5db;font-family:monospace;font-size:0.82rem;padding:14px 16px">
            <div id="${id}-tma-desc-body"></div>
          </div>
        </div>

      </div>
    </div>`;
}

// ═══════════════════════════════════════════════════════
//  Render
// ═══════════════════════════════════════════════════════

const tmaState = {};

function renderMakeTiledTmaAtom(tabId) {
  showErr(`${tabId}-tma-error`, '');
  showWarn(`${tabId}-tma-warning`, '');
  try {
    const op = TMA_OPS.tma_g2s;
    document.getElementById(`${tabId}-tma-op-params`).innerHTML =
      `<div class="cuo-result-line"><b>${op.ctor}</b></div>` +
      `<div class="cuo-result-line" style="color:#9ca3af">No constructor parameters required. ${op.note}</div>`;
    syncCopyMoves(tabId, 'tma', 'tma_g2s');

    const gmemStr  = document.getElementById(`${tabId}-tma-gmem-input`).value;
    const smemStr  = document.getElementById(`${tabId}-tma-smem-input`).value;
    const tilerStr = document.getElementById(`${tabId}-tma-tiler-input`).value;
    const dtype    = document.getElementById(`${tabId}-tma-dtype-input`).value;
    const elemBits = DTYPE_BITS[dtype];
    if (!elemBits) throw new Error(`Unknown element type "${dtype}"`);

    // --- gmem tensor: flat rank 2, so a basis axis is just 0 or 1 ---
    const gp = parseLayout(gmemStr);
    if (gp.shape.some(Array.isArray) || gp.stride.some(Array.isArray)) {
      throw new Error(
        `gmem_tensor must be a flat rank-2 layout here (e.g. (256,128):(128,1)); ` +
        `nested modes would make the TMA basis hierarchical, which this 2-D ` +
        `grid cannot draw.`);
    }
    const gShape = gp.shape.slice(), gStride = gp.stride.slice();
    if (gShape.some(s => !Number.isFinite(s) || s <= 0)) {
      throw new Error(`gmem_tensor has a non-positive extent.`);
    }

    // --- smem layout + swizzle ---
    // The picker is the normal path. An inline `Sw<..> o ..` prefix is still
    // accepted so a CuTe printout pastes verbatim, and when present it WINS —
    // someone who pasted a layout means the swizzle they pasted with it. Say so,
    // rather than silently ignoring one of the two.
    const sm = tmaParseSmemField(smemStr);
    const pickedRaw = document.getElementById(`${tabId}-tma-swizzle-input`).value;
    const picked = pickedRaw === 'none' ? null : parseSwizzleSpec(pickedRaw);
    const swInfo = tmaSwizzleInfo(sm.sw || picked, elemBits);
    document.getElementById(`${tabId}-tma-swizzle-note`).innerHTML = sm.sw
      ? `<div class="cuo-result-line" style="color:#f59e0b">Using <b>${tmaEsc(swInfo.str)}</b> from the ` +
        `smem_layout text; the picker above is ignored while that prefix is there.</div>`
      : '';
    const sp = parseLayout(sm.layoutStr);
    let sShape = sp.shape, sStride = sp.stride, stageNote = '';
    const tiler = mtcParseTiler(tilerStr);
    const tilerStrides = tiler.strides.map((s, i) => {
      if (s === null) return 1;
      const v = parseInt(s, 10);
      if (!Number.isFinite(v) || v <= 0) throw new Error(
        `cta_tiler mode ${i} has stride "${s}", which this tab cannot read — ` +
        `use a positive integer stride, e.g. (64:1, 64:2).`);
      return v;
    });
    // rank(smem_layout) == rank(cta_tiler) + 1 means a staged layout; the DSL
    // slices the stage mode off (cpasync/helpers.py:482).
    if (sShape.length === 3) {
      stageNote = ` (stage mode ${sShape[2]} sliced off)`;
      sShape = sShape.slice(0, 2);
      sStride = sStride.slice(0, 2);
    } else if (sShape.length !== 2) {
      throw new Error(
        `smem_layout must have rank 2 (or rank 3 when staged), got rank ${sShape.length}.`);
    }
    const sFlatShape = flatten(sShape), sFlatStride = flatten(sStride);
    const [S0, S1] = productEach(sShape);

    const offenders = collectHighRank([['gmem_tensor', gmemStr], ['smem_layout', sm.layoutStr]]);

    const r = tmaComputeAtom({
      gShape, gStride, elemBits, swInfo,
      sFlatShape, sFlatStride,
      tiler: { extents: tiler.extents, strides: tilerStrides },
    });

    // --- result box ---
    const swLine = swInfo.sw
      ? `<div class="cuo-result-line">swizzle = <b>${tmaEsc(swInfo.str)}</b> (bytes: base ${swInfo.baseBytes}B, ` +
        `width ${swInfo.widthBytes}B) &mdash; over ${dtype} elements that is ` +
        `<b>${tmaEsc(swInfo.elemStr)}</b>, which is the form the Swizzle tab takes</div>`
      : `<div class="cuo-result-line">swizzle = <b>none</b></div>`;
    document.getElementById(`${tabId}-tma-result`).innerHTML =
      `<div class="cuo-result-line"><b>Copy_Atom&lt;${op.label}, ${dtype}&gt;</b></div>` +
      `<div class="cuo-result-line">sidx2gmode = ${tmaFormatModes(r.full)}` +
      (r.dropped.length
        ? ` &rarr; truncated to <b>${tmaFormatModes(r.kept)}</b> at the first coef != 1`
        : ` (nothing truncated: every coefficient is 1)`) + `</div>` +
      `<div class="cuo-result-line">TMA box = <b>(${r.box.map(b => b.extent).join(', ')})</b>` +
      `, ${r.boxSize} elements = ${r.numBitsPerTma / 8} B per instruction</div>` +
      `<div class="cuo-result-line">num_bits_per_tma = <b>${r.numBitsPerTma}</b>` +
      (r.instructions > 1
        ? ` &mdash; the ${tiler.extents[0]}x${tiler.extents[1]} tile needs <b>${r.instructions}</b> TMA instructions`
        : ` &mdash; one instruction covers the whole tile`) + `</div>` +
      swLine + stageNoteLine(stageNote) +
      (sm.ignored.length
        ? `<div class="cuo-result-line" style="color:#9ca3af">Ignored: ${sm.ignored.join(', ')} — the smem_ptr marker only carries the element width.</div>`
        : '');

    const warnParts = [];
    if (offenders.length) {
      warnParts.push(`Note: ${offenders.join(', ')} has rank > 2. The math is still ` +
        `correct, but the visualization flattens it to a 2-D grid.`);
    }
    if (r.issues.length) {
      warnParts.push(`${r.issues.length} TMA constraint violation` +
        `${r.issues.length === 1 ? '' : 's'} — see the descriptor panel below. ` +
        `These are host-side asserts inside make_tma_copy_desc, compiled out in ` +
        `release builds.`);
    }
    if (warnParts.length) showWarn(`${tabId}-tma-warning`, warnParts.join(' '));

    const prev = tmaState[tabId] || {};
    tmaState[tabId] = {
      ...r, gShape, gStride, dtype, elemBits, swInfo, sShape, sStride, S0, S1,
      tiler: { extents: tiler.extents, strides: tilerStrides },
      opLabel: op.label,
      boxMode: (prev.boxMode instanceof Set) ? prev.boxMode : new Set(),
    };

    tmaRenderBoxViz(tabId);
    tmaRenderDescriptor(tabId);
    tmaRenderTensorViz(tabId);
    updateOuterTabLabel(tabId,
      `make_tiled_tma_atom:${r.box.map(b => b.extent).join('x')}/${dtype}`);
  } catch (e) {
    showErr(`${tabId}-tma-error`, e.message);
    for (const side of ['src', 'dst']) {
      const el = document.getElementById(`${tabId}-tma-${side}-svg`);
      if (el) el.innerHTML = '';
    }
    const d = document.getElementById(`${tabId}-tma-desc-body`);
    if (d) d.innerHTML = '';
    const t = document.getElementById(`${tabId}-tma-tensor-svg`);
    if (t) t.innerHTML = '';
  }
}

function stageNoteLine(note) {
  return note ? `<div class="cuo-result-line" style="color:#9ca3af">smem_layout${note}</div>` : '';
}

/** Both panes share one colour map, keyed by the SMEM offset: the box row it
 *  belongs to picks the hue, the instruction number darkens it. That makes a
 *  contiguous TMA fetch one band of colour in GMEM and shows where the swizzle
 *  scatters it in SMEM. */
function tmaCellColor(cell, s) {
  const row = Math.floor(cell.boxIdx / s.box[0].extent);
  const base = colorTV(row);
  if (s.instructions <= 1 || cell.inst === 0) return base;
  return darkenRGB(base, (cell.inst / (s.instructions - 1)) * 0.45);
}

function tmaRenderBoxViz(tabId) {
  const s = tmaState[tabId];
  if (!s) return;
  const modes = s.boxMode instanceof Set ? s.boxMode : new Set();
  const showValue = modes.has('value');
  const [T0, T1] = s.tiler.extents;
  const [ts0, ts1] = s.tiler.strides;
  const box0 = s.box[0].extent;

  // tile domain index -> the cell record (which is keyed by SMEM offset)
  const byTile = new Array(s.sSize);
  for (const c of s.cells) byTile[c.tileIdx] = c;

  // ---- SRC: the CTA tile in GMEM coordinates ----
  const bands = s.boxSize / box0;
  const boxDesc =
    `box (${s.box.map(b => b.extent).join(', ')}) = ` +
    (s.instructions > 1
      ? `${s.boxSize} of these ${s.sSize} cells per instruction, ${s.instructions} instructions`
      : `all ${s.sSize} of these cells in one instruction`) +
    `, drawn as ${bands} colour band${bands === 1 ? '' : 's'} of ${box0}` +
    (s.instructions > 1 ? `, darker = later instruction` : '');
  const srcHeader =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `CTA tile ${T0}&times;${T1} at gmem coord (0,0) &mdash; value = offset from the ` +
    `tensor base pointer<br>${boxDesc}` +
    `</div>`;
  const srcSVG = buildColoredLayoutSVG([T0, T1], [1, T0], 'value', (m, n) => {
    const c = byTile[m + n * T0];
    if (!c) return { bg: '#f0f0f0', text: null };
    // The coordinate always; `value` adds what the tensor's layout maps it to.
    const lines = [`(${c.gcrd[0]},${c.gcrd[1]})`];
    if (showValue) lines.push(String(c.goff));
    return { bg: tmaCellColor(c, s), text: lines };
  });

  // ---- DST: the same elements in SMEM, labelled by swizzled offset ----
  // How many cells the swizzle actually moves. Zero means the tile is too small
  // to reach the swizzle's source bits — the picture is then indistinguishable
  // from SWIZZLE_NONE, which is worth saying out loud rather than letting the
  // user conclude the swizzle does nothing.
  let moved = 0;
  if (s.swInfo.swElem) {
    for (let i = 0; i < s.sSize; i++) if (applySwizzleOffset(i, s.swInfo.swElem) !== i) moved++;
  }
  const dstHeader =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `SMEM tile ${s.S0}&times;${s.S1} &mdash; value = offset in shared memory` +
    (s.swInfo.sw
      ? ` &mdash; ${tmaEsc(s.swInfo.str)} (element form ${tmaEsc(s.swInfo.elemStr)}) ` +
        (moved === 0
          ? `moves <b>0 cells</b>: this tile is too small to reach the swizzle's source bits`
          : `moves ${moved} of ${s.sSize} cells`)
      : ' &mdash; no swizzle, so the value is the layout offset unchanged') +
    `</div>`;
  const dstSVG = buildColoredLayoutSVG(s.sShape, s.sStride, 'value', (m, n, offset) => {
    const c = s.cells[offset];
    if (!c) return { bg: '#f0f0f0', text: null };
    // Same coordinate as the SRC cell holding this element, so the two grids can
    // be read against each other; `value` adds the COMPOSED layout's output —
    // swizzle included, because that is what `slayout(coord)` returns.
    const lines = [`(${c.gcrd[0]},${c.gcrd[1]})`];
    if (showValue) lines.push(String(applySwizzleOffset(offset, s.swInfo.swElem)));
    return { bg: tmaCellColor(c, s), text: lines };
  });

  document.getElementById(`${tabId}-tma-src-svg`).innerHTML = srcHeader + srcSVG;
  document.getElementById(`${tabId}-tma-dst-svg`).innerHTML = dstHeader + dstSVG;
  applyZoomState(`${tabId}-tma-src-svg`);
  applyZoomState(`${tabId}-tma-dst-svg`);
  updateModeBtns(`${tabId}-tma-box-mode-btns`, modes);
  document.getElementById(`${tabId}-tma-box-title`).textContent =
    `CTA tile ${T0}×${T1}` +
    (ts0 !== 1 || ts1 !== 1 ? ` strided (${ts0},${ts1})` : '') +
    ` — TMA box (${s.box.map(b => b.extent).join('×')}), ${s.numBitsPerTma / 8} B per instruction` +
    (s.instructions > 1 ? `, ${s.instructions} of them` : '');
}

function tmaRenderDescriptor(tabId) {
  const s = tmaState[tabId];
  const host = document.getElementById(`${tabId}-tma-desc-body`);
  if (!s || !host) return;
  const pad = (v, n) => String(v).padStart(n, ' ');
  const line = (t) => `<div class="cuo-result-line">${t}</div>`;
  const dims = s.box.length;
  const five = (arr, fill) => {
    const out = arr.slice();
    while (out.length < 5) out.push(fill);
    return out;
  };
  const shapes  = five(s.probShape, 1);
  const strides = five(s.probStrideBytes, 0);
  const boxes   = five(s.box.map(b => b.extent), 1);

  let html = '';
  html += line(`tensorRank        = <b>${dims}</b>`);
  html += line(`tensorDataType    = ${s.dtype} (${s.elemBits}b)`);
  html += line(`globalDim   [5]   = { ${shapes.map(v => pad(v, 5)).join(', ')} }   (elements)`);
  html += line(`globalStrides[5]  = { ${strides.map(v => pad(v, 5)).join(', ')} }   (bytes; dim 0 implicit)`);
  html += line(`boxDim      [5]   = { ${boxes.map(v => pad(v, 5)).join(', ')} }   (elements)`);
  html += line(`elementStrides[5] = { ${[1, 1, 1, 1, 1].map(v => pad(v, 5)).join(', ')} }`);
  html += line(`swizzle           = ${s.swInfo.enumName}`);
  html += line(`interleave        = CU_TENSOR_MAP_INTERLEAVE_NONE`);
  html += line(`l2Promotion       = CU_TENSOR_MAP_L2_PROMOTION_L2_128B`);
  html += line(`oobFill           = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`);
  html += `<div class="cuo-result-line" style="color:#9ca3af;margin-top:6px">` +
          `TMA dim <i>j</i> &larr; gmem ${s.box.map((b, j) =>
            `[${j}] ${b.segs.length ? b.segs.map(g => `axis ${g.axis}`).join(' + ') : '—'}`).join(', ')}` +
          `. Innermost box row = ${s.box[0].extent} &times; ${s.elemBytes}B = ` +
          `<b>${s.innerBoxBytes}B</b>` +
          (s.swInfo.widthBytes ? ` against a ${s.swInfo.widthBytes}B swizzle.` : '.') +
          `</div>`;

  if (s.issues.length === 0) {
    html += `<div class="cuo-result-line" style="color:#34d399;margin-top:6px">` +
            `All TMA constraints satisfied.</div>`;
  } else {
    html += `<div class="cuo-result-line" style="margin-top:6px;color:#fca5a5"><b>` +
            `${s.issues.length} constraint violation${s.issues.length === 1 ? '' : 's'}</b> ` +
            `— CuTe checks these with host-side assert(), which a release build ` +
            `removes:</div>`;
    for (const iss of s.issues) {
      const color = iss.level === 'error' ? '#fca5a5' : '#fcd34d';
      html += `<div class="cuo-result-line" style="color:${color}">• ${iss.text}</div>`;
    }
  }
  host.innerHTML = html;
}

// The coordinate tensor is affine, so a corner is fully representative — and a
// full 256x128 grid would be 32k unreadable cells. Draw as much of the corner as
// fits in a cell budget, which shows small tensors whole.
const TMA_TENSOR_FULL_CELLS = 256;

function tmaRenderTensorViz(tabId) {
  const s = tmaState[tabId];
  const host = document.getElementById(`${tabId}-tma-tensor-svg`);
  if (!s || !host) return;
  const strideStr = tmaFormatBasisStride(s.tmaTensorStride);
  const full = `(0,0) o (${s.gShape.join(',')}):${strideStr}`;
  let P0 = s.gShape[0], P1 = s.gShape[1];
  for (let k = 1; k <= Math.max(s.gShape[0], s.gShape[1]); k++) {
    const a = Math.min(s.gShape[0], k), b = Math.min(s.gShape[1], k);
    if (a * b > TMA_TENSOR_FULL_CELLS) break;
    P0 = a; P1 = b;
  }
  const clipped = P0 !== s.gShape[0] || P1 !== s.gShape[1];
  const ndim = Math.max(basisRank(s.tmaTensorStride), 2);
  host.innerHTML =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `tma_tensor = ${full}` +
    (clipped ? ` &mdash; showing the top-left ${P0}&times;${P1} corner` : '') +
    `</div>` +
    buildBasisLayoutSVG([P0, P1], s.tmaTensorStride, ndim, null, new Set(['value']));
  applyZoomState(`${tabId}-tma-tensor-svg`);
  // "GMEM", not "SRC": this is the GMEM-side tensor, which is the source only
  // because the one Op supported so far is a load. TMA store has the same tensor
  // on its destination.
  document.getElementById(`${tabId}-tma-tensor-title`).textContent =
    `TMA tensor (GMEM) — (${s.gShape.join(',')}):${strideStr}`;
}

// Same shape as every other tab's mode toggle: the cell always names its input
// (the GMEM coordinate) and `value` overlays what the mapping returns.
function setTmaMode(tabId, mode) {
  const s = tmaState[tabId];
  if (!s) return;
  let modes = s.boxMode;
  if (!(modes instanceof Set)) { modes = new Set(); s.boxMode = modes; }
  if (modes.has(mode)) modes.delete(mode); else modes.add(mode);
  tmaRenderBoxViz(tabId);
}

function setTMA(tabId, dtype, gmem, sw, smem, tiler) {
  document.getElementById(`${tabId}-tma-dtype-input`).value   = dtype;
  document.getElementById(`${tabId}-tma-gmem-input`).value    = gmem;
  document.getElementById(`${tabId}-tma-swizzle-input`).value = sw;
  document.getElementById(`${tabId}-tma-smem-input`).value    = smem;
  document.getElementById(`${tabId}-tma-tiler-input`).value   = tiler;
  renderMakeTiledTmaAtom(tabId);
}

function exportTMA(tabId) {
  exportURL(`${tabId}-tma-export`, 'make_tiled_tma_atom',
    document.getElementById(`${tabId}-tma-dtype-input`).value,
    document.getElementById(`${tabId}-tma-gmem-input`).value,
    document.getElementById(`${tabId}-tma-swizzle-input`).value,
    document.getElementById(`${tabId}-tma-smem-input`).value,
    document.getElementById(`${tabId}-tma-tiler-input`).value);
}
