// partition_S / partition_D tab (COPY scope): the ThrCopy slice.
//
//   ThrCopy thr = tiled_copy.get_slice(thr_idx);
//   Tensor  tSgS = thr.partition_S(gS);     // or partition_D(gD)
//
// A TiledCopy is the ONLY source of a ThrCopy (`get_slice`), and a ThrCopy is
// the only thing that can partition a tensor -- so this tab takes a whole
// TiledCopy (atom + layout_tv + Tiler_MN, exactly the make_tiled_copy tab's
// inputs), a thread index, and the tensor to partition.
//
// The port is include/cute/atom/copy_atom.hpp:
//
//   partition_S(s)   = tidfrg_S(s.layout())(thr_idx, _, repeat<rank(s)>(_))   :368
//   tidfrg_S(s)      = tile2thrfrg(zipped_divide(s, Tiler_MN),                :221
//                                  right_inverse(AtomLayoutRef).compose(AtomLayoutSrc))
//   tile2thrfrg(t,r) = t.compose(coalesce(zip(                                :257
//                        zipped_divide(TiledLayout_TV, (AtomNumThr, AtomNumVal))
//                          .compose(r, _)), Shape<_1,Shape<_1,_1>>), _)
//                      (make_coord(_,_), _)
//
// Two facts about the SHAPE of the input and the output, because both are easy
// to get backwards:
//
//   * The tensor handed in is a PLAIN tensor, (M, N, ...) -- it is NOT
//     pre-divided by the tiler. `tidfrg_S` runs `zipped_divide(tensor,
//     Tiler_MN)` itself. The only requirement CuTe states is
//     `rank(tensor) >= rank(Tiler_MN)` (:223); modes beyond the tiler's rank
//     are carried through untouched and simply multiply Rest. That is how one
//     TiledCopy partitions `gA` of shape (BLK_M, BLK_K, k) -- the k-loop mode
//     is never tiled, it just comes along.
//   * The result is `((FrgV, FrgX), RestM, RestN, ...)`, rank 1 + rank(tensor).
//     The Rest modes arrive at TOP level, not nested, because CuTe's `slice`
//     splices a sliced tuple mode into its parent (int_tuple.hpp) -- which is
//     why CUTLASS's GEMMs write `tAgA(_,_,_,k)` rather than `tAgA(_,(_,_),k)`.
//
// So the reading of the result is: FrgV is one atom invocation, FrgX is how
// many atoms this thread issues inside ONE Tiler_MN tile, and the Rest modes
// are how many times the whole TiledCopy has to be replayed to cover the
// tensor. Three nested levels of repetition, in that order.
//
// Every Op this tab offers is SIMT, and for all of them
// `ValLayoutSrc == ValLayoutDst == ValLayoutRef`, so `ref2trg` is the identity
// and partition_S and partition_D differ ONLY in which tensor they are given.
// The derivation below still carries ref2trg through, because that is the term
// a shuffling atom (ldmatrix) would use, and it is the whole reason the two
// functions exist separately.

const PSD_TILE_EDGE = '#dc2626';

// ═══════════════════════════════════════════════════════
//  Tiler parsing
// ═══════════════════════════════════════════════════════

/** Parse Tiler_MN into an ARRAY OF LAYOUTS -- a real CuTe Tiler, one entry per
 *  tensor mode. Unlike `mtcParseTiler` (which only needs extents, because that
 *  tab has no tensor) the strides matter here: `(8:1, 16:2)` really does select
 *  every other column of the tensor, and `zipped_divide` needs the layout to
 *  say so.
 *
 *  Grammar, matching what CuTeDSL prints for `tc.tiler_mn`:
 *    (8, 16)        two shape modes
 *    (8:1, 16:2)    two layout modes  -- note the colons are NOT top-level, so
 *                   `parseLayout` on the whole string would reject this
 *    ((2,4), 16)    a nested shape mode */
function psdParseTiler(str) {
  const t = (str || '').trim();
  if (!t) throw new Error('Tiler_MN is empty — give it a shape like (8, 16).');

  // Strip one layer of parens only if it wraps the WHOLE string.
  let body = t, depth = 0;
  if (body[0] === '(') {
    for (let i = 0; i < body.length; i++) {
      if (body[i] === '(') depth++;
      else if (body[i] === ')' && --depth === 0) {
        if (i === body.length - 1) body = body.slice(1, -1);
        break;
      }
    }
  }

  const raw = [];
  depth = 0;
  let start = 0;
  for (let i = 0; i < body.length; i++) {
    const c = body[i];
    if (c === '(') depth++;
    else if (c === ')') depth--;
    else if (c === ',' && depth === 0) { raw.push(body.slice(start, i)); start = i + 1; }
  }
  const tail = body.slice(start).trim();
  if (tail) raw.push(tail);

  const modes = [];
  for (const piece of raw) {
    const m = piece.trim();
    if (!m) continue;
    const ci = topLevelColon(m);
    try {
      modes.push(ci === -1
        ? new Layout(parseValue(m))
        : new Layout(parseValue(m.slice(0, ci)), parseValue(m.slice(ci + 1))));
    } catch (e) {
      throw new Error(
        `Cannot read Tiler_MN mode "${m}": ${e.message}. Each mode is an extent (8), ` +
        `a layout (8:1), or a nested shape ((2,4)) — separated by top-level commas.`);
    }
    if (!(product(modes[modes.length - 1].shape) > 0)) {
      throw new Error(`Tiler_MN mode "${m}" has non-positive extent.`);
    }
  }
  if (modes.length === 0) throw new Error('Tiler_MN has no modes.');
  return modes;
}

// ═══════════════════════════════════════════════════════
//  The derivation (DOM-free, so tests/run.js can diff it against CuTeDSL)
// ═══════════════════════════════════════════════════════

/** Divisors of n, ascending. Used only to spell out the way past a rejected
 *  tiler, so it walks to sqrt(n) rather than allocating an n-element array —
 *  a GEMM-sized tensor mode is routinely in the thousands. */
function psdDivisors(n) {
  const lo = [], hi = [];
  for (let d = 1; d * d <= n; d++) {
    if (n % d) continue;
    lo.push(d);
    if (d !== n / d) hi.push(n / d);
  }
  return lo.concat(hi.reverse());
}

/** CuTe's `zip` on a single layout: ((a,b),(c,d)) -> ((a,c),(b,d)). */
function psdZip(layout) {
  return new Layout(zip_tuple(layout.shape), zip_tuple(layout.stride));
}

/** tile2thrfrg — copy_atom.hpp:257. `tensorZ` is already
 *  `zipped_divide(tensor, Tiler_MN)`, i.e. ((Tile...),(Rest...)).
 *  Returns (Thr, (FrgV, FrgX), (Rest...)). */
function psdTile2ThrFrg(tensorZ, layout_tv, atomNumThr, atomNumVal, ref2trg) {
  // ((atom_tid,atom_val),(rest_tid,rest_val)) -> (m,n)
  const atom_layout_TV = zipped_divide(layout_tv, [atomNumThr, atomNumVal]);
  // ((trg_tid,trg_val),(rest_tid,rest_val)) -> (m,n)
  const trg_layout_TV = composition(atom_layout_TV, [ref2trg, null]);
  // (thr_idx, (FrgV, FrgX)) -> (m,n)  -- zip pairs the atom's thr/val with the
  // tiling's, then the profile coalesces the thread pair into ONE mode while
  // keeping the two value modes apart.
  const thrval2mn = coalesce(psdZip(trg_layout_TV), [1, [1, 1]]);
  // ((thr,(FrgV,FrgX)),(Rest...))
  const tv_tensor = composition(tensorZ, [thrval2mn, null]);
  // `tv_tensor(make_coord(_,_), _)` — unpacks mode 0 by one level.
  const crd = [[null, null], null];
  return { tidfrg: new Layout(slice_(crd, tv_tensor.shape), slice_(crd, tv_tensor.stride)),
           thrval2mn };
}

/** The whole of partition_S / partition_D for one thread.
 *
 *  `tiler` is an array of Layouts (a CuTe Tiler), `tensor` a plain Layout of
 *  rank >= tiler.length, `side` is 'S' or 'D'.
 *
 *  Returns the two layouts CuTe produces plus the three levels the viz draws:
 *  `tileAt` (one tile), `tilePosAt` / `tileOffAt` (where a tile sits), and the
 *  Rest extents split into the tiler's two modes and the untiled remainder.
 *  Positions go through the COMPACT layout of the same shape, so the picture
 *  never depends on the tensor's strides (a non-injective or transposed tensor
 *  would otherwise alias two tiles onto one cell). */
function psdComputePartition(layout_tv, tiler, tensor, atomNumVal, thrIdx, side) {
  const R = tensor.rank();
  // CuTe itself only asserts rank(tensor) >= rank(Tiler_MN) (copy_atom.hpp:223)
  // and puts no ceiling on either. The two limits below are this TAB's, because
  // they are what the three grids can draw honestly:
  //   * a rank-2 tiler, since grid 1 is a 2-D picture of one tile;
  //   * at most 2 modes past the tiler, since grid 3 is a 2-D picture of them.
  // Two is not an arbitrary cut — the modes that survive untiled are the k-loop
  // and the pipeline stage, and a kernel with three of them is not a thing.
  if (tiler.length !== 2) {
    throw new Error(
      `Tiler_MN has rank ${tiler.length}; this tab draws one tile as a 2-D grid, so it ` +
      `takes a rank-2 tiler. CuTe accepts any rank — use the make_tiled_copy tab if you ` +
      `only need the tile itself.`);
  }
  if (R < 2) {
    throw new Error(
      `Rank of tensor to be partitioned too small: the tensor has rank ${R} but ` +
      `Tiler_MN has rank 2. CuTe asserts rank(tensor) >= rank(Tiler_MN) ` +
      `(copy_atom.hpp:223) — extra tensor modes are fine, missing ones are not.`);
  }
  if (R > 4) {
    throw new Error(
      `The tensor has rank ${R}, i.e. ${R - 2} modes past the tiler's rank. This tab draws ` +
      `those as a 2-D grid, so it takes at most 2 — they are the modes the tiler never ` +
      `touches, which in practice are the k-loop and the pipeline stage. Group the extras ` +
      `first if you really have more.`);
  }

  // Every Op this tab offers is SIMT: ThrID = 1:0 and
  // ValLayoutSrc == ValLayoutDst == ValLayoutRef == (1, AtomNumVal):(0, 1).
  const atomNumThr = 1;
  const atomLayout = new Layout([1, atomNumVal], [0, 1]);
  // right_inverse(Ref).compose(Src|Dst): re-orders the atom's (thr, val) slots
  // from the REFERENCE ordering into this side's ordering. Identity for every
  // Op here, and the only thing that would make S and D differ.
  const ref2trg = composition(right_inverse(atomLayout), atomLayout);

  // Each Tiler_MN mode must DIVIDE the matching tensor mode. In C++ this is a
  // constexpr `shape_div` assert, so a non-dividing tiler does not compile; the
  // DSL's dynamic path has no such check and returns a layout that reads past
  // the end of the tensor. Verified against CuTeDSL: a (8, 16) tiler on a
  // (12, 32) tensor produces `((2,2),2,2):((12,1),8,192)` — identical to this
  // port, and covering 388 positions of a 384-element tensor.
  const tExt = (is_tuple(tensor.shape) ? tensor.shape : [tensor.shape]).map(m => product(m));
  for (let i = 0; i < tiler.length; i++) {
    const t = product(tiler[i].shape);
    if (tExt[i] % t !== 0) {
      throw new Error(
        `Tiler_MN mode ${i} has extent ${t}, which does not divide the tensor's mode ${i} ` +
        `(${tExt[i]}). CuTe's C++ asserts this in shape_div, but CuTeDSL does not — it ` +
        `returns a layout that reads past the end of the tensor, so this is refused here ` +
        `rather than drawn. Divisors of ${tExt[i]}: ${psdDivisors(tExt[i]).join(', ')}.`);
    }
  }

  // The atom must be able to MOVE what FrgV says it moves. mtcVectorizationCheck
  // asks whether T0's first AtomNumVal values are a stride-1 run along one tile
  // axis; `none` means there is no tensor majorness that makes them contiguous,
  // so no copy instruction of that width exists for this partition.
  //
  // This is an ERROR here, where make_tiled_copy / make_tiled_copy_tv report the
  // same check inline. The difference is what the picture claims: those tabs draw
  // the tile's COVERAGE, which is true whatever atom runs over it, while grid 1
  // here is by definition "one atom invocation" (FrgV). Drawing a 4-wide FrgV
  // that no instruction can perform would be a fiction, not a caveat.
  const psdVec = mtcVectorizationCheck(atomNumVal, layout_tv, tiler.map(l => product(l.shape)));
  if (psdVec.kind === 'none') {
    const widths = [];
    for (let n = 1; n <= atomNumVal; n++) {
      if (atomNumVal % n !== 0) continue;
      if (mtcVectorizationCheck(n, layout_tv, tiler.map(l => product(l.shape))).kind !== 'none')
        widths.push(n);
    }
    throw new Error(
      `num_bits_per_copy gives AtomNumVal = ${atomNumVal}, but T0's first ${atomNumVal} ` +
      `values are not a stride-1 run along either tile axis — they sit at ` +
      `${psdVec.coords.map(c => `(${c[0]},${c[1]})`).join(' ')}. No copy instruction that ` +
      `wide exists for this partition, so FrgV would name an atom that cannot run.` +
      (widths.length
        ? ` Widths that do vectorize here: ${widths.join(', ')} elements.`
        : ''));
  }

  let tensorZ;
  try {
    tensorZ = zipped_divide(tensor, tiler);
  } catch (e) {
    throw new Error(
      `zipped_divide(tensor, Tiler_MN) failed: ${e.message}. Tensor ` +
      `${formatLayoutStr(tensor.shape, tensor.stride)} against tiler ` +
      `(${tiler.map(l => formatLayoutStr(l.shape, l.stride)).join(', ')}).`);
  }

  const { tidfrg, thrval2mn } = psdTile2ThrFrg(tensorZ, layout_tv, atomNumThr, atomNumVal, ref2trg);
  const thrSize = product(tidfrg.shape[0]);
  if (thrIdx !== null && thrIdx >= thrSize) {
    throw new Error(
      `Thread index ${thrIdx} is out of range — this TiledCopy has ${thrSize} ` +
      `thread${thrSize === 1 ? '' : 's'} (0..${thrSize - 1}).`);
  }

  // thr_tensor(thr_idx, _, repeat<rank(tensor)>(_)). The rest coord is R
  // SEPARATE underscores, and slicing a tuple mode splices it into the parent —
  // which is what puts RestM/RestN at top level instead of nested.
  //
  // `thrIdx` may be null — "show every thread". The LAYOUT does not depend on
  // it: slicing with an integer drops the thread mode whatever its value, which
  // is the same reason CuTe can give every thread the same static type. Only
  // the base offset is per-thread, so that is what becomes null.
  const sliceCrd = [thrIdx === null ? 0 : thrIdx, null, new Array(R).fill(null)];
  const partition = new Layout(slice_(sliceCrd, tidfrg.shape), slice_(sliceCrd, tidfrg.stride));
  const baseOffset = thrIdx === null
    ? null
    : crd2idx([thrIdx, 0, new Array(R).fill(0)], tidfrg.shape, tidfrg.stride);

  // The same divide over the COMPACT layout of the same shape, so grid 2's
  // positions never depend on the tensor's strides — a non-injective or
  // transposed tensor cannot then alias two tiles onto one cell.
  const posZ = zipped_divide(new Layout(tensor.shape), tiler);

  const frgV = product(tidfrg.shape[1][0]);
  const frgX = product(tidfrg.shape[1][1]);
  const restShape = tidfrg.shape[2];
  const restExtents = (is_tuple(restShape) ? restShape : [restShape]).map(m => product(m));
  // Rest is TWO different things wearing one mode, and the tab draws them
  // separately because they mean different things:
  //   restExtents[0..1] — what dividing tensor modes 0 and 1 by the tiler
  //                       produced: how the tile sweeps the (M, N) plane.
  //   restExtents[2..]  — tensor modes the tiler never touched, carried through
  //                       verbatim: the k-loop / pipeline stage.
  const sweepExtents = restExtents.slice(0, 2);
  const extraExtents = restExtents.slice(2);
  const sweepSize = sweepExtents.reduce((a, b) => a * b, 1);

  // The layout each grid actually draws, so a grid can print its own subject
  // rather than leaving you to reconstruct it. The three are a partition of
  // `zipped_divide(tensor, Tiler_MN)` — mode 0, then mode 1 split at the
  // tiler's rank — so together they are the whole tensor and nothing else.
  const restL = tensorZ.mode(1);
  const tileLayout  = tensorZ.mode(0);
  const sweepLayout = make_layout(restL.mode(0), restL.mode(1));
  const extraLayout = extraExtents.length
    ? make_layout(...extraExtents.map((_, k) => restL.mode(2 + k)))
    : null;

  return {
    side, thrIdx, thrSize, tidfrg, partition, thrval2mn, baseOffset,
    frgV, frgX, valSize: frgV * frgX, restExtents, sweepExtents, extraExtents, sweepSize,
    tileLayout, sweepLayout, extraLayout,
    extraSize: extraExtents.reduce((a, b) => a * b, 1),
    restSize: restExtents.reduce((a, b) => a * b, 1),
    tileExtents: tiler.map(l => product(l.shape)),
    tensorExtents: tExt,
    // LEVEL 1 — where thread t's value v lands inside ONE tile. Read off
    // `thrval2mn`, the same map the composition uses, so this is the
    // partition's own tile rather than a parallel derivation of it. It does not
    // depend on the Rest position: `tidfrg`'s thread and value modes are
    // independent of its Rest mode, which is exactly what makes the two-level
    // picture lossless rather than a summary.
    tileAt: (t, v) => thrval2mn.call(t, v),
    // LEVEL 2 — where tile `r` sits. `(0, r)` is the tile's first element:
    // through the compact layout for the grid position, through the real one
    // for the offset a kernel would see.
    tilePosAt: (r) => posZ.call(0, r),
    tileOffAt: (r) => tensorZ.call(0, r),
  };
}

/** Decompose a flat Rest index into its per-mode components, col-major — the
 *  coordinate you would actually type at `tSgS(_, i, j)`. */
function psdRestCoord(r, restExtents) {
  const out = [];
  let rem = r;
  for (const e of restExtents) { out.push(rem % e); rem = Math.floor(rem / e); }
  return out;
}

// ═══════════════════════════════════════════════════════
//  The tab
// ═══════════════════════════════════════════════════════

/** What the S/D toggle means. One control, because the question is always
 *  "which end of the copy am I partitioning" — only the tensor and the atom's
 *  TV layout change with it. */
const PSD_SIDE = {
  S: { fn: 'partition_S', tensor: 'SRC', atomLayout: 'ValLayoutSrc' },
  D: { fn: 'partition_D', tensor: 'DST', atomLayout: 'ValLayoutDst' },
};

function generatePartitionSDTabContent(id) {
  return `
    <!-- partition_S / partition_D panel -->
    <div id="${id}-tab-partition_sd" class="panel">
      <div class="controls">
        <h2>partition_S / partition_D</h2>

        <div class="form-group">
          <label>Which end of the copy</label>
          <div class="seg-control" id="${id}-psd-side-btns">
            <button class="mode-btn active" onclick="setPsdSide('${id}', 'S')">partition_S</button>
            <button class="mode-btn" onclick="setPsdSide('${id}', 'D')">partition_D</button>
          </div>
          <div class="hint-inline" id="${id}-psd-side-hint"></div>
        </div>

${mtcAtomSection(id, 'psd', '1. The Copy_Atom', 32)}

        <details class="cuo-section" open>
          <summary>2. The TiledCopy &mdash; the only source of a ThrCopy</summary>
          <div class="cuo-section-body">
            ${layoutInputField({
              id: `${id}-psd-tv-input`,
              label: 'layout_tv &mdash; (tid, vid) &rarr; flat (m, n) in the tile',
              hint: 'mode 0 = threads, mode 1 = values',
              value: '((8,4),(2,2)):((16,2),(8,1))'
            })}
            ${layoutInputField({
              id: `${id}-psd-tiler-input`,
              label: 'Tiler_MN &mdash; the tile zipped_divide carves out',
              hint: 'a Tiler: (8, 16) or per-mode layouts (8:1, 16:2)',
              value: '(8, 16)'
            })}
            <div id="${id}-psd-tiled-result" class="cuo-result"></div>
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>3. get_slice(thr_idx)</summary>
          <div class="cuo-section-body">
            <div class="form-group">
              <label>Thread index</label>
              <input type="text" id="${id}-psd-thr-input" value="5" placeholder="all threads">
              <div class="hint-inline">
                A ThrCopy is one thread's view. Leave this <b>empty</b> to show every
                thread instead: the layout <code>partition_S</code> returns is the same
                for all of them, so a thread id only changes which cells are highlighted
                and where the fragment starts.
              </div>
            </div>
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>4. The tensor being partitioned</summary>
          <div class="cuo-section-body">
            ${layoutInputField({
              id: `${id}-psd-tensor-input`,
              label: 'The <span id="' + id + '-psd-tensor-side">SRC</span> tensor &mdash; ' +
                     'the argument to <span id="' + id + '-psd-tensor-fn">partition_S</span>',
              hint: 'a plain tensor, NOT pre-divided; rank &ge; rank(Tiler_MN)',
              value: '(16, 32):(1, 16)'
            })}
          </div>
        </details>

        ${statusDivs(`${id}-psd`)}
        <button class="btn btn-render" onclick="renderPartitionSD('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-psd-export" onclick="exportPSD('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setPSD('${id}','S','universal',32,'half_t','((8,4),(2,2)):((16,2),(8,1))','(8, 16)','5','(16, 32):(1, 16)')">One tile per axis &mdash; the tile sweeps 2&times;2</button>
            <button class="preset-btn" onclick="setPSD('${id}','S','universal',32,'half_t','((8,4),(2,2)):((16,2),(8,1))','(8, 16)','5','(8, 16):(1, 8)')">Tensor == tile &mdash; one Rest position, no replay</button>
            <button class="preset-btn" onclick="setPSD('${id}','D','universal',32,'half_t','((8,4),(2,2)):((16,2),(8,1))','(8, 16)','5','(16, 32):(32, 1)')">partition_D on the row-major destination</button>
            <button class="preset-btn" onclick="setPSD('${id}','S','cpasync',128,'half_t','((8,16),8):((128,1),16)','(16, 64)','3','(16, 64):(64, 1)')">sgemm_sm80 TV layout, tensor == one tile</button>
            <button class="preset-btn" onclick="setPSD('${id}','S','universal',32,'half_t','((8,4),(2,2)):((16,2),(8,1))','(8, 16)','5','(16, 32, 3):(1, 16, 512)')">Rank-3 &mdash; a k-loop mode rides along untiled (grid 3 is a strip)</button>
            <button class="preset-btn" onclick="setPSD('${id}','S','universal',32,'half_t','((8,4),(2,2)):((16,2),(8,1))','(8, 16)','5','(16, 32, 4, 3):(1, 16, 512, 2048)')">Rank-4 &mdash; k-loop &times; pipeline stage (grid 3 is a grid)</button>
            <button class="preset-btn" onclick="setPSD('${id}','S','universal',16,'half_t','((4,4),4):((4,16),1)','(4:1, 16:2)','2','(4, 32):(1, 4)')">Strided tiler (4:1, 16:2) &mdash; the tile is every other column</button>
            <button class="preset-btn" onclick="setPSD('${id}','S','cpasync',128,'half_t','((8,16),8):((128,1),16)','(16, 64)','3','(128, 256):(256, 1)')">GEMM-sized &mdash; 8&times;4 tiles the two levels keep drawable</button>
          </div>
        </div>

        <div class="hint">
          <b>The tensor is NOT pre-divided.</b> You hand
          <code>partition_S</code> a plain <code>(M, N, ...)</code> tensor;
          <code>tidfrg_S</code> runs <code>zipped_divide(tensor, Tiler_MN)</code>
          itself (<code>copy_atom.hpp:226</code>). The only thing CuTe asserts is
          <code>rank(tensor) &ge; rank(Tiler_MN)</code> &mdash; so a rank-2
          tensor under a rank-2 tiler is the normal case, and modes <em>past</em>
          the tiler's rank are never tiled at all: they come along and multiply
          Rest. That is how one TiledCopy partitions <code>gA</code> of shape
          <code>(BLK_M, BLK_K, k)</code>.<br><br>
          <b>Rest is two different things wearing one mode.</b>
          <code>((FrgV, FrgX), RestM, RestN, ...)</code> &mdash; the first two
          Rest modes are what dividing tensor modes 0 and 1 by
          <code>Tiler_MN</code> produced, and any further ones are tensor modes
          the tiler <em>never touched</em>, carried through verbatim. They mean
          different things, so the tab draws them as separate grids:
          <b>2</b> is the tile sweeping the (M, N) plane, <b>3</b> is that whole
          plane repeated along the k-loop and the pipeline stage. Inside one tile
          sits the third level, grid <b>1</b>: <b>FrgV</b> is one atom invocation
          (<code>num_bits_per_copy / sizeof_bits(dtype)</code>) and <b>FrgX</b> is
          how many atoms a thread issues to fill the tile
          (<code>TiledNumVal / AtomNumVal</code>).<br><br>
          <b>Splitting the picture this way loses nothing.</b>
          <code>tidfrg</code>'s thread and value modes do not depend on its Rest
          mode, so <em>every</em> tile is partitioned by the identical grid
          <b>1</b>, and every slice of grid <b>3</b> holds an identical grid
          <b>2</b>. Drawing the tensor at element resolution instead would repeat
          one picture hundreds of times and stop being drawable at GEMM sizes
          &mdash; a 128&times;256 tensor is 32768 cells against 1024 + 32
          here.<br><br>
          <b>This tab takes a rank-2 tiler and at most 2 untiled modes</b>, which
          is a limit of the three grids rather than of CuTe. Two is not an
          arbitrary cut: the modes that survive untiled are the k-loop and the
          pipeline stage.<br><br>
          <b>Rest arrives at top level, not nested.</b>
          <code>partition_S</code> slices with
          <code>(thr_idx, _, repeat&lt;rank&gt;(_))</code>, and CuTe's
          <code>slice</code> splices a sliced tuple mode into its parent &mdash;
          which is why CUTLASS's GEMMs write <code>tAgA(_,_,_,k)</code> and not
          <code>tAgA(_,(_,_),k)</code>.<br><br>
          <b>For these Ops, S and D differ only in the tensor.</b> The step that
          could make them differ is
          <code>right_inverse(ValLayoutRef).compose(ValLayoutSrc|Dst)</code>
          (<code>copy_atom.hpp:226, 247</code>), which re-orders the atom's
          <code>(thr, val)</code> slots from the reference ordering into this
          side's. Every Op offered here is SIMT with
          <code>ValLayoutSrc == ValLayoutDst == ValLayoutRef</code>, so that term
          is the identity. It becomes real for a shuffling atom such as
          <code>ldmatrix</code>, where the source layout says which lane holds
          which <em>address</em> and the destination layout says which lane ends
          up with which <em>value</em>.<br><br>
          <b>The tiler's strides matter here</b>, unlike on the
          <b>make_tiled_copy</b> tab. There the tile is drawn on its own and a
          stride has nothing to point into; here there is a tensor, so
          <code>(4:1, 16:2)</code> really does make tile 0 every other column
          &mdash; visible in grid <b>2</b>, where the tile origins stop being a
          regular block lattice.
        </div>
      </div>

      <div class="comp-results" style="grid-template-columns:1fr">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-psd-tile-title">1. The TiledCopy over one tile</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-psd-tile-mode-btns">
                <button class="mode-btn" onclick="setPsdMode('${id}','tile','value')">value</button>
              </span>
              <button class="mode-btn" id="${id}-psd-tile-svg-zoom" onclick="toggleZoom('${id}-psd-tile-svg')">Zoom in</button>
              <button class="mode-btn" onclick="downloadSVG('${id}-psd-tile-svg', 'partition_sd_tile.svg')">Download SVG</button>
            </span>
          </div>
          <div class="cuo-viz-desc">
            <b>Level 1 &mdash; how the TiledCopy covers one <code>Tiler_MN</code>
            tile.</b> Cell <code>(m, n)</code> shows the <code>(t, v)</code> that
            owns it; colour is the thread, brightness the atom invocation
            (<code>FrgX</code>). The selected thread is at full brightness and
            everything else keeps its labels in grey. This picture is the same
            for every tile below &mdash; <code>tidfrg</code>'s thread and value
            modes are independent of its Rest mode.<br>
            The layout in <b>blue</b> above the grid is the one this grid draws:
            <b>mode 0</b> of <code>zipped_divide(tensor, Tiler_MN)</code>, i.e.
            the tile as it sits in the tensor.
          </div>
          <div class="viz-box"><div id="${id}-psd-tile-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-psd-sweep-title">2. The tile over the (M, N) plane</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-psd-sweep-mode-btns">
                <button class="mode-btn" onclick="setPsdMode('${id}','sweep','value')">value</button>
              </span>
              <button class="mode-btn" id="${id}-psd-sweep-svg-zoom" onclick="toggleZoom('${id}-psd-sweep-svg')">Zoom in</button>
              <button class="mode-btn" onclick="downloadSVG('${id}-psd-sweep-svg', 'partition_sd_sweep.svg')">Download SVG</button>
            </span>
          </div>
          <div class="cuo-viz-desc">
            <b>Level 2 &mdash; how that tile covers the plane the tiler divides.</b>
            One cell per tile, i.e. one replay of the whole TiledCopy: these are
            the two Rest modes <code>zipped_divide</code> produced from tensor
            modes 0 and 1. Each is labelled with the coordinate you would type at
            <code>tSgS(_, i, j)</code> and the tile's origin in the tensor;
            <code>value</code> swaps the origin for the offset. Tile origins are
            evaluated, never assumed &mdash; a Tiler mode with a stride
            (<code>16:2</code>) puts tile 1 one column over, not a tile-width
            away.<br>
            The layout in <b>blue</b> is the one this grid draws: <b>Rest modes
            0&ndash;1</b>, mapping a tile index to the offset of that tile.
          </div>
          <div class="viz-box"><div id="${id}-psd-sweep-svg"></div></div>
        </div>
        <div class="comp-viz-item" id="${id}-psd-extra-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-psd-extra-title">3. The plane over the whole tensor</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-psd-extra-mode-btns">
                <button class="mode-btn" onclick="setPsdMode('${id}','extra','value')">value</button>
              </span>
              <button class="mode-btn" id="${id}-psd-extra-svg-zoom" onclick="toggleZoom('${id}-psd-extra-svg')">Zoom in</button>
              <button class="mode-btn" onclick="downloadSVG('${id}-psd-extra-svg', 'partition_sd_extra.svg')">Download SVG</button>
            </span>
          </div>
          <div class="cuo-viz-desc">
            <b>Level 3 &mdash; how that whole plane covers the tensor.</b> One
            cell per tensor mode past the tiler's rank: modes
            <code>zipped_divide</code> never touched, which arrive in Rest
            verbatim. In a GEMM these are the <b>k-loop</b> and the <b>pipeline
            stage</b>, so there is one of them (a strip) or two (a grid), and
            this panel is hidden when the tensor is exactly the plane above.
            <code>value</code> shows the offset each step adds. A single mode is
            drawn as a <b>row</b> rather than a column &mdash; there is no second
            axis to line it up with, and a k-loop reads horizontally.<br>
            The layout in <b>blue</b> is the one this grid draws: <b>Rest modes
            2+</b>. Grid 1's, grid 2's and this one are a partition of
            <code>zipped_divide(tensor, Tiler_MN)</code> &mdash; together they
            are the whole tensor and nothing else.
          </div>
          <div class="viz-box"><div id="${id}-psd-extra-svg"></div></div>
        </div>
      </div>
    </div>`;
}

const psdState = {};

function renderPartitionSD(tabId) {
  showErr(`${tabId}-psd-error`, '');
  showWarn(`${tabId}-psd-warning`, '');
  const prev = psdState[tabId] || {};
  const side = prev.side || 'S';
  psdSyncSideField(tabId, side);
  try {
    const atom = mtcReadAtom(tabId, 'psd');

    const tvStr     = document.getElementById(`${tabId}-psd-tv-input`).value;
    const tilerStr  = document.getElementById(`${tabId}-psd-tiler-input`).value;
    const thrStr    = (document.getElementById(`${tabId}-psd-thr-input`).value || '').trim();
    const tensorStr = document.getElementById(`${tabId}-psd-tensor-input`).value;

    // The TENSOR is deliberately absent from the rank warning. Nothing here
    // flattens it: grid 1 is over Tiler_MN and grid 2 enumerates every Rest
    // mode, so a rank-3 tensor is drawn in full rather than collapsed. The two
    // inputs that ARE drawn as a 2-D grid are the tiler (grid 1's axes) and
    // layout_tv, so those are what the warning covers.
    const rankPairs = [['layout_tv', tvStr], ['Tiler_MN', tilerStr]];
    updateRankWarning(`${tabId}-psd-warning`, rankPairs);
    const offenders = collectHighRank(rankPairs);

    const tvP = parseLayout(tvStr);
    const tvS = stripTrivialTrailing(tvP.shape, tvP.stride);
    const layout_tv = new Layout(tvS.shape, tvS.stride);
    if (layout_tv.rank() !== 2) {
      throw new Error(
        `layout_tv must have rank 2 — mode 0 is the thread mode, mode 1 the value mode. ` +
        `Got rank ${layout_tv.rank()} from ${formatLayoutStr(layout_tv.shape, layout_tv.stride)}.`);
    }
    const thrSize = product(layout_tv.shape[0]);
    const valSize = product(layout_tv.shape[1]);
    mtcRequireAtomDivides(valSize, atom.atomNumVal, atom.numBits, atom.elemBits, atom.dtype);

    const tiler = psdParseTiler(tilerStr);
    // Blank means "every thread". A real ThrCopy is one thread, but the layout
    // it returns is the same for all of them — only the base offset differs —
    // so drawing the unsliced tile is a truthful default rather than a fiction.
    if (thrStr !== '' && !/^\d+$/.test(thrStr)) {
      throw new Error(
        `Thread index must be a whole number from 0 to ${thrSize - 1}, or empty for all ` +
        `threads — got "${thrStr}".`);
    }
    const thrIdx = thrStr === '' ? null : parseInt(thrStr, 10);

    const tP = parseLayout(tensorStr);
    const tS = stripTrivialTrailing(tP.shape, tP.stride);
    const tensor = new Layout(tS.shape, tS.stride);

    const r = psdComputePartition(layout_tv, tiler, tensor, atom.atomNumVal, thrIdx, side);

    const spec = PSD_SIDE[side];
    const [srcSpace, dstSpace] = copyMove(tabId, 'psd');
    const space = side === 'S' ? srcSpace : dstSpace;
    document.getElementById(`${tabId}-psd-tiled-result`).innerHTML =
      `<div class="cuo-result-line">layout_tv = <b>${formatLayoutStr(layout_tv.shape, layout_tv.stride)}</b></div>` +
      `<div class="cuo-result-line">Tiler_MN  = <b>(${tiler.map(l => formatLayoutStr(l.shape, l.stride)).join(', ')})</b></div>` +
      `<div class="cuo-result-line">TiledNumThr = ${thrSize}, TiledNumVal = ${valSize}, ` +
      `FrgV = ${r.frgV}, FrgX = ${r.frgX}</div>`;

    const notes = [
      offenders.length
        ? `${offenders.join(', ')} has rank > 2 — grid 1 draws modes 0 and 1 of the tile.`
        : '',
      atom.cpasyncWarn,
    ].filter(Boolean);
    if (notes.length) showWarn(`${tabId}-psd-warning`, notes.join(' '));

    psdState[tabId] = {
      ...atom, ...r, side, layout_tv, tiler, tensor, thrSize, valSizeTv: valSize,
      tileMode:  (prev.tileMode  instanceof Set) ? prev.tileMode  : new Set(),
      sweepMode: (prev.sweepMode instanceof Set) ? prev.sweepMode : new Set(),
      extraMode: (prev.extraMode instanceof Set) ? prev.extraMode : new Set(),
    };
    psdRenderTileViz(tabId);
    psdRenderSweepViz(tabId);
    psdRenderExtraViz(tabId);
    // The returned layout has no result box any more, so the outer tab label
    // carries it — the same place Divide / LocalTile put theirs.
    updateOuterTabLabel(tabId,
      `${spec.fn}:${formatLayoutStr(r.partition.shape, r.partition.stride)}`);
  } catch (e) {
    showErr(`${tabId}-psd-error`, e.message);
    for (const k of ['tile', 'sweep', 'extra']) {
      const el = document.getElementById(`${tabId}-psd-${k}-svg`);
      if (el) el.innerHTML = '';
    }
  }
}

/** LEVEL 1 — one Tiler_MN tile, in tile-local coordinates. Identical for every
 *  Rest position, which is what lets it be drawn once. */
function psdRenderTileViz(tabId) {
  const s = psdState[tabId];
  if (!s) return;
  const M = s.tileExtents[0], N = s.tileExtents[1] === undefined ? 1 : s.tileExtents[1];
  const modes = s.tileMode instanceof Set ? s.tileMode : new Set();

  // A cell can have several claimants when layout_tv has a stride-0 mode (that
  // is broadcast, not overlap — `make_tiled_copy_A/B` produce it). Keep the
  // lowest-id claimant, EXCEPT that the selected thread always wins: it is the
  // one thing this tab is about, and letting a lower-numbered co-reader own the
  // cell would draw the focused thread as if it were not there.
  const owner = new Map();
  let shared = 0;
  for (let t = 0; t < s.thrSize; t++) {
    for (let v = 0; v < s.valSize; v++) {
      const p = s.tileAt(t, v);
      const cur = owner.get(p);
      if (!cur) owner.set(p, { t, v });
      else if (cur.t !== t && ++shared && t === s.thrIdx) owner.set(p, { t, v });
    }
  }

  const svg = buildColoredLayoutSVG([M, N], [1, M], modes, (m, n) => {
    const e = owner.get(m + n * M);
    if (!e) return { bg: '#f0f0f0', fg: '#bbb', text: ['—'] };
    const lines = [`T${e.t}`, `V${e.v}`];
    if (modes.has('value')) lines.push(String(m + n * M));
    if (s.thrIdx !== null && e.t !== s.thrIdx) return { bg: '#f0f0f0', fg: '#bbb', text: lines };
    return { bg: mtcThreadAtomColor(e.t, Math.floor(e.v / s.atomNumVal), s.frgX), text: lines };
  });

  document.getElementById(`${tabId}-psd-tile-svg`).innerHTML =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `<b style="color:#93c5fd">${formatLayoutStr(s.tileLayout.shape, s.tileLayout.stride)}</b>` +
    ` &mdash; tile ${M}&times;${N}, ${s.thrSize} threads &times; ${s.valSize} values ` +
    `(FrgV=${s.frgV}, FrgX=${s.frgX}), ` +
    (s.thrIdx === null ? `every thread shown` : `T${s.thrIdx} highlighted`) +
    (shared ? ` &mdash; <b>broadcast</b>: ${shared} claim${shared === 1 ? '' : 's'} beyond the ` +
              `first` + (s.thrIdx === null ? '' :
               `, the cell shows T${s.thrIdx} where it is one of them`) : '') +
    `</div>` + svg;
  applyZoomState(`${tabId}-psd-tile-svg`);
  updateModeBtns(`${tabId}-psd-tile-mode-btns`, modes);
  document.getElementById(`${tabId}-psd-tile-title`).textContent =
    `1. The TiledCopy over one tile (${s.tileExtents.join('×')})`;
}

/** LEVEL 2 — one cell per tile: how the tile covers the plane the tiler
 *  divides, i.e. Rest modes 0 and 1. The modes past the tiler's rank are held
 *  at 0 here; grid 3 is about those. */
function psdRenderSweepViz(tabId) {
  const s = psdState[tabId];
  if (!s) return;
  const modes = s.sweepMode instanceof Set ? s.sweepMode : new Set();
  const [R0, R1] = s.sweepExtents;

  const svg = buildColoredLayoutSVG([R0, R1], [1, R0], modes, (i, j) => {
    const r = i + j * R0;                       // extra modes pinned at 0
    const lines = [`(${i},${j})`];
    lines.push(modes.has('value')
      ? `@${s.tileOffAt(r)}`
      : `(${psdRestCoord(s.tilePosAt(r), s.tensorExtents).slice(0, 2).join(',')})`);
    return { bg: colorHighlight(r), stroke: '#1e3a5f', sw: 1.5, text: lines };
  });

  document.getElementById(`${tabId}-psd-sweep-svg`).innerHTML =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `<b style="color:#93c5fd">${formatLayoutStr(s.sweepLayout.shape, s.sweepLayout.stride)}</b>` +
    ` &mdash; ${R0}&times;${R1} tiles of ${s.tileExtents.join('&times;')} over the ` +
    `${s.tensorExtents.slice(0, 2).join('&times;')} plane; second line is the tile's ` +
    (modes.has('value') ? `offset` : `origin (m, n)`) +
    (s.extraSize > 1 ? `; the other ${s.extraSize} slice${s.extraSize === 1 ? '' : 's'} ` +
                       `look identical &mdash; see 3` : '') +
    `</div>` + svg;
  applyZoomState(`${tabId}-psd-sweep-svg`);
  updateModeBtns(`${tabId}-psd-sweep-mode-btns`, modes);
  document.getElementById(`${tabId}-psd-sweep-title`).textContent =
    `2. The tile over the (M, N) plane — ${R0 * R1} tile${R0 * R1 === 1 ? '' : 's'}`;
}

/** LEVEL 3 — the tensor modes the tiler never touched. One or two of them, so
 *  a strip or a grid.
 *
 *  Hidden, not skipped, when the tensor is exactly the plane above: the SVG is
 *  still rendered into the hidden box so this path runs on every render and
 *  cannot rot unnoticed (dom_smoke asserts the pane is non-empty). A lone 1x1
 *  cell is not worth showing, but a render path that only executes for rank-3
 *  inputs is worth testing. */
function psdRenderExtraViz(tabId) {
  const s = psdState[tabId];
  if (!s) return;
  const modes = s.extraMode instanceof Set ? s.extraMode : new Set();
  const E0 = s.extraExtents.length > 0 ? s.extraExtents[0] : 1;
  const E1 = s.extraExtents.length > 1 ? s.extraExtents[1] : 1;

  // A SINGLE untiled mode is drawn as a ROW, not a column. Everywhere else in
  // this tool mode 0 runs down the rows, but a lone mode has no second axis to
  // be consistent with, so the only question left is which reads better — and a
  // k-loop (which is what one untiled mode almost always is) is read
  // horizontally, in the same orientation the MMA tabs draw K. It is also the
  // cheap axis: these panels are stacked, so height is the scarce one. The cost
  // is that mode 2 moves from horizontal to vertical the moment a mode 3
  // appears; the per-cell labels are `(i)` against `(i,j)` and the header names
  // the modes, so which one you are looking at is never in doubt.
  const oneD = s.extraExtents.length <= 1;
  const rows = oneD ? 1 : E0;
  const cols = oneD ? E0 : E1;

  const svg = buildColoredLayoutSVG([rows, cols], [1, rows], modes, (m, n) => {
    const e = oneD ? n : m + n * E0;
    // Tile (0, 0) at this position in the untiled modes — the offset is what
    // the k-loop / stage index actually advances the pointer by.
    const lines = [oneD ? `(${e})` : `(${m},${n})`];
    if (modes.has('value')) lines.push(`@${s.tileOffAt(e * s.sweepSize)}`);
    return { bg: colorHighlight(e), stroke: '#1e3a5f', sw: 1.5, text: lines };
  });

  const host = document.getElementById(`${tabId}-psd-extra-svg`);
  host.innerHTML =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    (s.extraSize > 1
      ? `<b style="color:#93c5fd">${formatLayoutStr(s.extraLayout.shape, s.extraLayout.stride)}</b>` +
        ` &mdash; tensor mode${s.extraExtents.length === 1 ? ' ' : 's '}` +
        `${s.extraExtents.map((_, k) => k + 2).join(', ')}, never tiled; each holds the whole ` +
        `${s.sweepExtents.join('&times;')}-tile plane above`
      : `no modes past the tiler &mdash; the tensor is exactly the plane above`) +
    `</div>` + svg;
  applyZoomState(`${tabId}-psd-extra-svg`);
  updateModeBtns(`${tabId}-psd-extra-mode-btns`, modes);
  const item = document.getElementById(`${tabId}-psd-extra-item`);
  if (item) item.style.display = s.extraSize > 1 ? '' : 'none';
  document.getElementById(`${tabId}-psd-extra-title`).textContent =
    `3. The plane over the whole tensor — ${s.extraSize} slice${s.extraSize === 1 ? '' : 's'}`;
}

/** Relabel the S/D controls. One toggle whose meaning is which end of the copy
 *  is being partitioned, and there is ONE tensor box — so its label has to move
 *  with the toggle or it would name the wrong argument. */
function psdSyncSideField(tabId, side) {
  const spec = PSD_SIDE[side] || PSD_SIDE.S;
  const hint = document.getElementById(`${tabId}-psd-side-hint`);
  if (hint) {
    hint.innerHTML =
      `<code>${spec.fn}</code> partitions the <b>${spec.tensor}</b> tensor using the atom's ` +
      `<code>${spec.atomLayout}</code>. Every Op here has ` +
      `<code>ValLayoutSrc == ValLayoutDst == ValLayoutRef</code>, so the two calls differ ` +
      `only in which tensor they are given.`;
  }
  const sideEl = document.getElementById(`${tabId}-psd-tensor-side`);
  if (sideEl) sideEl.textContent = spec.tensor;
  const fnEl = document.getElementById(`${tabId}-psd-tensor-fn`);
  if (fnEl) fnEl.textContent = spec.fn;
  const group = document.getElementById(`${tabId}-psd-side-btns`);
  if (group) group.querySelectorAll('.mode-btn').forEach(b =>
    b.classList.toggle('active', b.textContent.trim() === spec.fn));
}

function setPsdSide(tabId, side) {
  psdState[tabId] = Object.assign(psdState[tabId] || {}, { side });
  psdSyncSideField(tabId, side);
  renderPartitionSD(tabId);
}

function setPsdMode(tabId, which, mode) {
  const s = psdState[tabId];
  if (!s) return;
  const key = { tile: 'tileMode', sweep: 'sweepMode', extra: 'extraMode' }[which];
  let modes = s[key];
  if (!(modes instanceof Set)) { modes = new Set(); s[key] = modes; }
  if (modes.has(mode)) modes.delete(mode); else modes.add(mode);
  if (which === 'tile') psdRenderTileViz(tabId);
  else if (which === 'sweep') psdRenderSweepViz(tabId);
  else psdRenderExtraViz(tabId);
}

function setPSD(tabId, side, opKey, bits, dtype, tv, tiler, thr, tensor) {
  psdState[tabId] = Object.assign(psdState[tabId] || {}, { side });
  document.getElementById(`${tabId}-psd-op-input`).value     = opKey;
  document.getElementById(`${tabId}-psd-bits-input`).value   = bits;
  document.getElementById(`${tabId}-psd-dtype-input`).value  = dtype;
  document.getElementById(`${tabId}-psd-tv-input`).value     = tv;
  document.getElementById(`${tabId}-psd-tiler-input`).value  = tiler;
  document.getElementById(`${tabId}-psd-thr-input`).value    = thr;
  document.getElementById(`${tabId}-psd-tensor-input`).value = tensor;
  renderPartitionSD(tabId);
}

function exportPSD(tabId) {
  exportURL(`${tabId}-psd-export`, 'partition_sd',
    (psdState[tabId] && psdState[tabId].side) || 'S',
    document.getElementById(`${tabId}-psd-op-input`).value,
    document.getElementById(`${tabId}-psd-bits-input`).value,
    document.getElementById(`${tabId}-psd-dtype-input`).value,
    document.getElementById(`${tabId}-psd-tv-input`).value,
    document.getElementById(`${tabId}-psd-tiler-input`).value,
    document.getElementById(`${tabId}-psd-thr-input`).value,
    document.getElementById(`${tabId}-psd-tensor-input`).value);
}
