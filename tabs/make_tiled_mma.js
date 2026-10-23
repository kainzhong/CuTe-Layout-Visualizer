// make_tiled_mma tab: replicate warp or warpgroup MMA atoms, then optionally
// reshape/permute the tile. The atom supplies its own 32/128-thread ThrID.
//
// Mirrors `cute.make_tiled_mma(op_or_atom, atom_layout_mnk, permutation_mnk)`
// (python/CuTeDSL/cutlass/cute/atom.py:771). This is the MMA counterpart of
// make_tiled_copy: make_mma_atom says what ONE warp's instruction does, and
// this says how many warps run it and over which tile.
//
//   op   = cute.nvgpu.warp.MmaF16BF16Op(Float16, Float32, (16,8,16))
//   tmma = cute.make_tiled_mma(op, (2,2,1), (32,32,16))
//
// The two arguments do genuinely different things, which is why the tab draws
// two rows of grids rather than one:
//
//   atom_layout_mnk   how many ATOMS (warps) there are, laid out over M, N, K.
//                     Each warp covers an atom-sized area of A, B and C, so the
//                     tile grows to (16*aM, 8*aN, 16*aK). Top row.
//   permutation_mnk   a Tiler that says which tile each mode is REALLY over.
//                     Bigger than the warps cover -> the whole warp pattern
//                     repeats; a non-trivial layout -> the rows/columns the
//                     warps land on are permuted. Bottom row.
//
// ── The derivation ────────────────────────────────────────────────────────
// A port of CUTLASS's `TiledMMA::thrfrg_A/B/C` (include/cute/atom/mma_atom.hpp).
// For operand A over its (X, Y) = (M, K) axes:
//
//   T = (TileX, TileY):(1, TileX)                        the tile, col-major
//   T = logical_divide(T, (permX, permY))                apply the permutation
//   T = zipped_divide(T, (AtomX, AtomY))                 ((AtomX,AtomY),(RestX,RestY))
//   T = composition(T, (AtomLayout_TV, _))               ((ThrV,FrgV),(RestX,RestY))
//   T = zipped_divide(T, (_, (ThrX, ThrY)))              split Rest into thread + value
//
// matching `thrfrg_A` at mma_atom.hpp:291 step for step. The pieces are then
// reassembled the way `get_layoutA_TV` (:416) does it: an `atile` inserts the
// VMNK mode this operand does not depend on as a STRIDE-0 entry -- literally
// `make_layout((aM, aN), (1, 0))` for A and `(0, 1)` for B -- which is what
// makes A broadcast across the N warps and B across the M warps. Finally the
// thread mode is composed with `left_inverse(thr_layout_vmnk)` (C++ spells it
// `right_inverse(make_layout(vmnk, complement(vmnk)))`, which is the same
// thing) so it is indexed by THREAD INDEX rather than by the (v,m,n,k)
// coordinate -- which matters as soon as atom_layout_mnk carries a
// non-column-major stride.
//
// Verified character-for-character against CuTeDSL's `tv_layout_A_tiled` /
// `tv_layout_B_tiled` for plain, K-split and genuinely permuted configurations,
// and pointwise against `partition_A/B/C` (tests/cases.json, section
// `tiled_mma`).
//
// ── C is where CUTLASS contradicts itself, and this follows partition_C ───
// `get_layoutA_TV` and `get_layoutB_TV` both insert that stride-0 `atile` /
// `btile` step. `get_layoutC_TV` (mma_atom.hpp:399) has NO equivalent: it goes
// straight from `thrfrg_C` -- whose thread mode is (ThrV,(ThrM,ThrN)), size
// 32*aM*aN -- to `.compose(thridx_2_thrid, _)`, whose domain is the full
// 32*aM*aN*aK. Composing a smaller layout with a larger identity EXTENDS its
// last mode, so the K factor lands on ThrN's stride instead of becoming a
// stride-0 mode of its own. That is the whole bug, and it only shows when both
// aN > 1 and aK > 1: for (2,2,2), `tv_layout_C_tiled` reports
// `((4,8,2,4),...):((64,1,16,256),...)` -- N and K fused into one `4:256` --
// which sends the k=1 warps to a second copy of C outside the 32x16 tile.
//
// `partition_C` on the same object disagrees and is what a kernel actually
// calls: thread 128 (k=1) gets exactly the cells of thread 0, because the K
// warps hold PARTIAL SUMS of one accumulator and must be reduced afterwards.
// So this port gives C the same stride-0 treatment A and B get, and the
// differential test uses partition_C as its oracle.

const MTM_ATOM_THREADS = 32;      // Physical warp width (also a warp atom's ThrID)

/** The stride `cute.make_layout(shape)` implies, for a shape given without one.
 *  Column-major, EXCEPT that a size-1 mode gets stride 0 rather than the running
 *  product — CuTe's compact stride does that, and `(2,2,1)` reaching
 *  `thr_layout_vmnk` as `(32,2,2,1):(1,32,64,0)` instead of `...,128)` is the
 *  visible consequence. Unobservable in the map (a size-1 mode is only ever
 *  entered at 0) but this tab prints the layout next to one a user may have
 *  copied out of CuTeDSL, so it should be the same string. */
function mtmCompactStride(shape) {
  let run = 1;
  const walk = (s) => {
    if (Array.isArray(s)) return s.map(walk);
    if (s === 1) return 0;
    const r = run;
    run *= s;
    return r;
  };
  return walk(shape);
}

/** Parse `atom_layout_mnk`. A rank-3 layout: `(2,2,1)` is the common form, and
 *  an explicit stride (`(2,2,1):(2,1,4)`) reorders which warp is which. Not
 *  parseLayout, which pads a bare shape out to rank 2 and would silently invent
 *  a mode. */
function mtmParseAtomLayout(raw) {
  const s = (raw || '').trim();
  // Blank means the argument was OMITTED, and CuTeDSL's signature defaults it to
  // (1, 1, 1) — `make_tiled_mma(op_or_atom, atom_layout_mnk=(1,1,1), ...)`. That
  // is also the reading `permutation_mnk`'s blank already gets in this tab, and
  // the two boxes are both optional arguments, so they should agree.
  if (!s) return new Layout([1, 1, 1], mtmCompactStride([1, 1, 1]));
  const ci = topLevelColon(s);
  const shape = parseValue(ci === -1 ? s : s.slice(0, ci).trim());
  const stride = ci === -1 ? mtmCompactStride(shape) : parseValue(s.slice(ci + 1).trim());
  if (!Array.isArray(shape) || shape.length !== 3)
    throw new Error(
      `atom_layout_mnk must be rank 3 (one mode each for M, N and K) — got ` +
      `"${s}". CuTeDSL raises "expects rank-3 MNK atom layout" on anything else.`);
  if (!Array.isArray(stride) || stride.length !== 3)
    throw new Error(`atom_layout_mnk's stride must be rank 3 to match its shape — got "${s}".`);
  return new Layout(shape, stride);
}

/** Parse `permutation_mnk`, a rank-3 Tiler. Each mode is either absent
 *  (`_` / `x` / `None` / empty — CuTe's no-op) or a layout: `32` is the plain
 *  extent, `(2,16):(16,1)` is a genuine permutation of 32 rows. */
function mtmParsePerm(raw) {
  let t = (raw || '').trim();
  if (!t) return [null, null, null];
  if (/^Tile\s*</i.test(t))
    throw new Error(
      `"${t}" is CUTLASS C++ template syntax. Write the permutation as values ` +
      `instead — e.g. (32, 32, 16), or (_, _, _) for none.`);
  // Strip one layer of parens only when it wraps the WHOLE string, so "(32,32,16)"
  // unwraps but "(2,16):(16,1), 32, 16" (already a mode list) is left alone.
  if (t[0] === '(') {
    let depth = 0;
    for (let i = 0; i < t.length; i++) {
      if (t[i] === '(') depth++;
      else if (t[i] === ')' && --depth === 0) {
        if (i === t.length - 1) t = t.slice(1, -1);
        break;
      }
    }
  }
  const modes = [];
  let depth = 0, start = 0;
  for (let i = 0; i < t.length; i++) {
    const c = t[i];
    if (c === '(') depth++;
    else if (c === ')') depth--;
    else if (c === ',' && depth === 0) { modes.push(t.slice(start, i)); start = i + 1; }
  }
  modes.push(t.slice(start));
  if (modes.length !== 3)
    throw new Error(
      `permutation_mnk must have exactly 3 modes (M, N, K) — got ${modes.length} ` +
      `in "${raw.trim()}". Use "_" for a mode you don't want to permute.`);
  return modes.map((m, i) => {
    const p = m.trim();
    if (p === '' || p === '_' || p === 'x' || p === 'X' || p === 'None' || p === 'none') return null;
    if (/^_\d+$/.test(p))
      throw new Error(
        `"${p}" is CUTLASS C++'s integral-constant syntax (Int<${p.slice(1)}>), which ` +
        `only exists for template metaprogramming. Write "${p.slice(1)}".`);
    let shape, stride;
    try {
      const ci = topLevelColon(p);
      shape = parseValue(ci === -1 ? p : p.slice(0, ci).trim());
      stride = ci === -1 ? mtmCompactStride(shape) : parseValue(p.slice(ci + 1).trim());
    } catch (e) {
      throw new Error(`Cannot read permutation_mnk mode ${'MNK'[i]} ("${p}"): ${e.message}`);
    }
    return new Layout(shape, stride);
  });
}

/** `thrfrg_X` for ONE operand, over its own two axes.
 *
 *  atomTV     the atom's (thread, value) -> tile-offset layout
 *  atomExt    [AtomX, AtomY], the extents the single atom covers
 *  perm       [permX, permY], each a Layout or null
 *  thrTile    [ThrX, ThrY], how many atoms atom_layout_mnk tiles along each axis
 *  names      ['M','K'] etc., for error messages only
 *  tensorL    OPTIONAL. `thrfrg_X` is defined over a TENSOR (mma_atom.hpp:291);
 *             make_tiled_mma only ever asks about one tile, so the default is
 *             the tile itself, col-major. The partition_ABC tab passes a real
 *             tensor instead and the same four lines then produce the layout
 *             `partition_X` returns — rank >= 2 works throughout, since
 *             logical_divide / zipped_divide / composition all carry trailing
 *             modes untouched.
 *
 *  Returns the six pieces the caller reassembles, plus the tile it is over and
 *  `restExtra`, the rest modes past the operand's own two (empty for a rank-2
 *  tensor, and therefore always empty for make_tiled_mma). */
function mtmThrfrg(atomTV, atomExt, perm, thrTile, names, tensorL) {
  const tile = [0, 1].map(i => (perm[i] ? product(perm[i].shape) : atomExt[i] * thrTile[i]));
  for (const i of [0, 1]) {
    const covered = atomExt[i] * thrTile[i];
    if (tile[i] % covered !== 0)
      throw new Error(
        `permutation_mnk's ${names[i]} mode has size ${tile[i]}, which is not a multiple ` +
        `of the ${covered} the atoms cover there (atom ${atomExt[i]} × ` +
        `${thrTile[i]} atom${thrTile[i] === 1 ? '' : 's'}). ` +
        `zipped_divide needs it to divide exactly — try ${covered} or ${covered * 2}.`);
  }
  let T = tensorL || new Layout([tile[0], tile[1]], [1, tile[0]]);
  T = logical_divide(T, [perm[0], perm[1]]);
  T = zipped_divide(T, [new Layout(atomExt[0]), new Layout(atomExt[1])]);
  T = composition(T, [atomTV, null]);
  T = zipped_divide(T, [null, [new Layout(thrTile[0]), new Layout(thrTile[1])]]);
  const restAll = T.mode(1).mode(1);
  const restExtra = [];
  for (let i = 2; i < restAll.rank(); i++) restExtra.push(restAll.mode(i));
  return {
    tile,
    thrV:  T.mode(1).mode(0).mode(0),
    frgV:  T.mode(1).mode(0).mode(1),
    thrX:  T.mode(0).mode(1).mode(0),
    thrY:  T.mode(0).mode(1).mode(1),
    restX: restAll.mode(0),
    restY: restAll.mode(1),
    restExtra,
  };
}

/** The whole make_tiled_mma derivation, DOM-free so tests/run.js can diff it
 *  against CuTeDSL. `atom` is a `mmaWarpAtom(...)` result; `perm` is a
 *  3-element array of Layout-or-null.
 *
 *  Every operand comes back as `{ tile, thr, val, tv, frgSize, restShape }`
 *  where `thr` is indexed by THREAD INDEX and `val` has shape
 *  (FrgV, (RestX, RestY)) — the same profile CuTeDSL's `tv_layout_X_tiled`
 *  prints and the same order `partition_X` hands back its fragment in. */
function mtmComputeTiledMma(atom, atomLayout, perm, tensors) {
  const [M, N, K] = atom.shapeMNK;
  const atomThreads = product(atom.thrId.shape);
  const thrVmnk = tiled_product(new Layout(atom.thrId.shape, atom.thrId.stride), atomLayout);
  const aM = product(thrVmnk.shape[1]);
  const aN = product(thrVmnk.shape[2]);
  const aK = product(thrVmnk.shape[3]);
  const threads = atomThreads * aM * aN * aK;
  if (!isBijective(thrVmnk))
    throw new Error(
      `atom_layout_mnk = ${formatLayoutStr(atomLayout.shape, atomLayout.stride)} does not give ` +
      `each atom a distinct id — thr_layout_vmnk = ` +
      `${formatLayoutStr(thrVmnk.shape, thrVmnk.stride)} is not a bijection. ` +
      `A stride of 0, or two modes landing on the same index, would make two atoms share an id.`);
  // thread index -> the (v, m, n, k) coordinate of thr_layout_vmnk. The
  // identity for the usual column-major atom_layout_mnk, a real permutation
  // once a stride is given. `left_inverse` IS the C++ expression: mma_atom.hpp
  // builds `right_inverse(make_layout(vmnk, complement(vmnk)))`, which is what
  // layout.js's left_inverse expands to.
  const idx2vmnk = left_inverse(thrVmnk);

  const build = (which) => {
    // Which two axes this operand spans, which atom_layout modes tile it, and
    // which VMNK mode it does NOT depend on (the stride-0 broadcast).
    // `pm` indexes permutation_mnk / atom_layout_mnk (0=M, 1=N, 2=K); `bcast`
    // indexes the assembled VMNK thread tuple, where 0 is V — so A, which does
    // not depend on N, broadcasts at 2.
    const spec = {
      A: { L: atom.A, ext: [M, K], names: ['M', 'K'], pm: [0, 2], bcast: 2, bsize: aN },
      B: { L: atom.B, ext: [N, K], names: ['N', 'K'], pm: [1, 2], bcast: 1, bsize: aM },
      C: { L: atom.C, ext: [M, N], names: ['M', 'N'], pm: [0, 1], bcast: 3, bsize: aK },
    }[which];
    const thrTile = spec.pm.map(i => [aM, aN, aK][i]);
    const atomTV = new Layout(spec.L.shape, spec.L.stride);
    // `tensors` is the partition_ABC tab's hook: the same derivation over a real
    // tensor rather than over one tile. make_tiled_mma passes nothing.
    const r = mtmThrfrg(atomTV, spec.ext, [perm[spec.pm[0]], perm[spec.pm[1]]], thrTile,
                        spec.names, tensors && tensors[which]);

    // Reassemble the thread mode in VMNK order. The operand's own two thread
    // modes go in the positions atom_layout_mnk gave them; the third is the
    // broadcast, and it is a stride-0 mode rather than an absent one because a
    // layout is a total function over every thread of the TiledMMA.
    const vmnk = [r.thrV, null, null, null];
    vmnk[spec.bcast] = new Layout(spec.bsize, 0);
    const rest = [1, 2, 3].filter(i => i !== spec.bcast);
    vmnk[rest[0]] = r.thrX;
    vmnk[rest[1]] = r.thrY;
    const thr = composition(make_layout(vmnk[0], vmnk[1], vmnk[2], vmnk[3]), idx2vmnk);
    const val = make_layout(r.frgV, make_layout(r.restX, r.restY, ...r.restExtra));
    return {
      tile: r.tile, thr, val,
      space: spec.L.space || 'RMEM', major: spec.L.major || 'K', atomThreads,
      tv: make_layout(thr, val),
      frgSize: product(r.frgV.shape),
      restShape: [product(r.restX.shape), product(r.restY.shape)],
      restX: r.restX, restY: r.restY, restExtra: r.restExtra,
      // What `partition_X(tensor)` returns: the value mode with the rest tuple
      // SPLICED to top level, exactly as CuTe's `slice` does — which is why a
      // kernel writes `tCgA(_,_,_,k)` and not `tCgA(_,(_,_),k)`.
      partition: make_layout(r.frgV, r.restX, r.restY, ...r.restExtra),
    };
  };

  return {
    thrVmnk, threads, warps: threads / MTM_ATOM_THREADS, atomThreads,
    atomGroups: aM * aN * aK,
    atomLayoutMNK: [aM, aN, aK],
    A: build('A'), B: build('B'), C: build('C'),
    // tile_size(i), i.e. TiledMma.get_tile_size — the M, N, K of the whole tile
    tileMNK: [
      perm[0] ? product(perm[0].shape) : M * aM,
      perm[1] ? product(perm[1].shape) : N * aN,
      perm[2] ? product(perm[2].shape) : K * aK,
    ],
  };
}

/** Which (thread, value) pairs own each cell of an operand's tile.
 *  `rest` is the cell's coordinate in the (RestX, RestY) repetition — 0 means
 *  the first copy of the warp pattern, which is the region the bottom row
 *  colours. Every entry in one cell shares it, since the rest modes live on the
 *  value side. */
function mtmOperandGrid(op) {
  const [M, N] = op.tile;
  const nT = product(op.thr.shape), nV = product(op.val.shape);
  const grid = Array.from({ length: M }, () => Array.from({ length: N }, () => ({ entries: [], rest: 0 })));
  // A descriptor broadcasts within each atom. One representative per group
  // suffices for coverage/repetition; the SMEM renderer never labels it T/V.
  const atomThreads = op.atomThreads || MTM_ATOM_THREADS;
  for (let t = 0; t < nT; t += op.space === 'SMEM' ? atomThreads : 1) {
    const tOff = op.thr.call(t);
    for (let v = 0; v < nV; v++) {
      const off = tOff + op.val.call(v);
      const m = off % M, n = Math.floor(off / M);
      if (m < 0 || m >= M || n < 0 || n >= N) continue;
      const cell = grid[m][n];
      if (!cell.entries.length) cell.rest = Math.floor(v / op.frgSize);
      cell.entries.push({ t, v, w: Math.floor(t / MTM_ATOM_THREADS) });
    }
  }
  return grid;
}

/** Swap a grid's axes. Used only to DRAW B as K x N in the quadrant view — the
 *  cells and their entries are untouched, so the TV mapping is the same one,
 *  read with the axes the other way round. */
function mtmTransposeGrid(grid) {
  const M = grid.length, N = M ? grid[0].length : 0;
  return Array.from({ length: N }, (_, n) => Array.from({ length: M }, (_, m) => grid[m][n]));
}

/** Compact label for the set of warps that touch a cell.
 *  `sum` is set for C, where the several warps hold PARTIAL SUMS of the same
 *  accumulator (the K reduction) rather than several readers of one value. */
function mtmWarpLabel(warps, sum, prefix = 'W') {
  const w = [...new Set(warps)].sort((a, b) => a - b);
  if (w.length === 1) return `${prefix}${w[0]}`;
  const contiguous = w[w.length - 1] - w[0] === w.length - 1;
  const body = contiguous ? `${prefix}${w[0]}..${prefix}${w[w.length - 1]}`
                          : w.map(x => `${prefix}${x}`).join(sum ? '+' : ',');
  return sum ? `Σ${body}` : body;
}

// Four lines is about as much as a cell can carry before it stops being read
// and starts being decoration. Past that the warp set collapses to
// mtmWarpLabel's single compact line.
const MTM_MAX_WARP_LINES = 4;

/** The label lines a warp-mode cell carries.
 *
 *  With a warp id in the box, a cell that warp touches says just `W{id}` — the
 *  question being asked is "where does W3 land", not "who else is here", and
 *  the caller greys everything it does not touch. Without one, every warp that
 *  touches the cell gets its own line, exactly as TV mode lists its T/V
 *  entries, so a broadcast (A across the N warps) or a reduction (C across the
 *  K warps) is legible as several names stacked in one cell.
 *
 *  Cells the focused warp does not touch are drawn blank by mtmBuildSVG, so the
 *  listing below is only ever reached when the box is empty. */
function mtmWarpLines(warps, sum, focus, prefix = 'W') {
  const w = [...new Set(warps)].sort((a, b) => a - b);
  if (focus !== null && focus !== undefined && w.includes(focus)) return [`${prefix}${focus}`];
  if (w.length > MTM_MAX_WARP_LINES) return [mtmWarpLabel(w, sum, prefix)];
  return w.map(x => `${prefix}${x}`);
}

/** The label lines a TV-mode cell carries — the `mtmWarpLines` counterpart.
 *
 *  With a thread id in the box, a cell that thread touches shows only ITS slot,
 *  so you read which VALUE of that thread lands there rather than hunting for
 *  it among the threads broadcasting onto the same cell. Without one, every
 *  entry is listed, which is make_mma_atom's picture. */
function mtmTvLines(entries, focus) {
  const mine = (focus === null || focus === undefined)
    ? entries : entries.filter(e => e.t === focus);
  const use = mine.length ? mine : entries;
  if (use.length === 1) return [`T${use[0].t}`, `V${use[0].v}`];
  return use.map(e => `T${e.t}/V${e.v}`);
}

function mtmRestOverlay(grid, {cs, margin, M, N}) {
  let body = '';
  const seg = (x1,y1,x2,y2) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#e53e3e" stroke-width="1.8"/>`;
  for (let m=0; m<M; m++) for (let n=0; n<N; n++) {
    const x=margin+n*cs, y=margin+m*cs;
    if (m>0 && grid[m][n].rest !== grid[m-1][n].rest) body += seg(x,y,x+cs,y);
    if (n>0 && grid[m][n].rest !== grid[m][n-1].rest) body += seg(x,y,x,y+cs);
  }
  return body;
}

/** Draw one operand's tile.
 *
 *  opts.mode      'warp' -> whole warp regions, cells labelled W0 / ΣW0..W1;
 *                 'tv' -> the make_mma_atom picture, TxVx per cell.
 *  opts.focus     the id in the box, or null for "show them all" — a WARP id in
 *                 'warp' mode and a THREAD id in 'tv' mode. Either way only the
 *                 region it touches stays coloured; everything else goes flat
 *                 grey AND loses its label, because once you have picked one
 *                 unit the other names are noise.
 *  opts.dimRest   the bottom row: grey out every cell outside the first
 *                 (Rest = 0) copy of the warp pattern and draw red lines on the
 *                 boundaries between copies. With a focus set this SOFTENS
 *                 rather than overrides — the copies the focused unit touches
 *                 keep its hue at reduced opacity, since the pattern really
 *                 does repeat there and greying it would say otherwise.
 *  opts.sum       C, so several warps in a cell read as a reduction.
 *  opts.showValue add the tile's flat offset to each label.
 *  opts.cellIndex override for that offset. The default assumes the tile is
 *                 col-major in its own codomain; a transposed pane (B in the
 *                 quadrant view) must pass its own or the overlay prints a
 *                 number that is not the layout's output. */
function mtmBuildSVG(grid, M, N, opts) {
  opts = opts || {};
  if (M * N > MAX_CELLS) return errSVG(`Tile too large: ${M}×${N} = ${M * N} cells (max ${MAX_CELLS})`);
  if (M === 0 || N === 0) return errSVG(`Empty tile: ${M}×${N}`);
  if (opts.space === 'SMEM') return buildColoredLayoutSVG([M,N], [1,M], [],
    (m,n) => ({bg: '#a7f3d0', text: opts.showValue
      ? [String(opts.cellIndex ? opts.cellIndex(m,n) : m+n*M)] : null}),
    {overlay: opts.dimRest ? geometry => mtmRestOverlay(grid, geometry) : undefined});

  const cs = cellSize(M, N);
  const margin = cs;
  const W = margin + N * cs;
  const H = margin + M * cs;
  const axisFs = Math.max(7, Math.min(12, Math.floor(cs * 0.26)));
  let body = '';

  for (let n = 0; n < N; n++) {
    body += `<text x="${margin + (n + 0.5) * cs}" y="${margin * 0.55}" text-anchor="middle"
      dominant-baseline="middle" fill="#555" font-size="${axisFs}" font-family="monospace">${n}</text>`;
  }
  for (let m = 0; m < M; m++) {
    body += `<text x="${margin * 0.5}" y="${margin + (m + 0.5) * cs}" text-anchor="middle"
      dominant-baseline="middle" fill="#555" font-size="${axisFs}" font-family="monospace">${m}</text>`;
  }

  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      const x = margin + n * cs, y = margin + m * cs;
      const { entries, rest } = grid[m][n];
      const idxLabel = String(opts.cellIndex ? opts.cellIndex(m, n) : (m + n * M));

      if (!entries.length) {
        body += `<rect x="${x}" y="${y}" width="${cs}" height="${cs}" fill="#f0f0f0" stroke="#ccc" stroke-width="0.5"/>`;
        body += cellTextSVG(x + cs / 2, y + cs / 2, ['—'], cs, '#bbb');
        continue;
      }

      const isWarpMode = opts.mode === 'warp' || opts.mode === 'warpgroup';
      const warps = entries.map(e => opts.mode === 'warpgroup' ? Math.floor(e.t / (opts.atomThreads || 128)) : e.w);
      const focused = opts.focus !== null && opts.focus !== undefined;
      const touched = !focused || (isWarpMode ? warps.includes(opts.focus)
                                              : entries.some(e => e.t === opts.focus));
      const isRepeat = !!opts.dimRest && rest !== 0;

      // Four cell states, and the ordering between them is the whole design:
      //   not touched         -> grey and BLANK. Once you have picked a warp,
      //                          every other name on screen is noise.
      //   touched, a repeat   -> the same hue at reduced opacity. The warp
      //                          pattern repeats across the permuted tile, so
      //                          this unit really does land here too, and
      //                          greying it would say the opposite.
      //   a repeat, unfocused -> grey but still labelled: the bottom row's own
      //                          question is "which part is the repetition".
      //   otherwise           -> full colour.
      const blank = focused && !touched;
      const dimmed = !blank && isRepeat && focused;
      const greyed = blank || (isRepeat && !focused);

      const own = focused && touched
        ? opts.focus
        : (isWarpMode ? Math.min(...warps) : entries[0].t);
      const fill = greyed ? '#f0f0f0' : colorTV(own);
      const fg = greyed ? '#bbb' : (dimmed ? '#4b5563' : '#111');

      body += `<rect x="${x}" y="${y}" width="${cs}" height="${cs}" fill="${fill}"
        fill-opacity="${dimmed ? 0.35 : 1}" stroke="#ccc" stroke-width="0.5"/>`;

      if (blank) continue;
      const focusArg = focused ? opts.focus : null;
      const lines = isWarpMode ? mtmWarpLines(warps, opts.sum, focusArg, opts.groupPrefix)
                               : mtmTvLines(entries, focusArg);
      if (opts.showValue) lines.push(idxLabel);
      body += cellTextSVG(x + cs / 2, y + cs / 2, lines, cs, fg);
    }
  }

  // A tile boundary is a LINE, not a property of the cells beside it — faking
  // one with per-cell strokes gives a doubled, fuzzy edge. Same call as
  // tma_partition's red outlines.
  if (opts.dimRest) body += mtmRestOverlay(grid, {cs,margin,M,N});

  return `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="${svgFitStyle(W, H)}">
    <rect width="${W}" height="${H}" fill="white"/>
    ${body}
  </svg>`;
}

function generateMakeTiledMmaTabContent(id) {
  const vizItem = (stage, side, label) => `
        <div class="comp-viz-item" data-q="${side}">
          <div class="comp-viz-header">
            <span class="comp-viz-title"><span class="comp-viz-label" id="${id}-mtm-${stage}-${side}-title">${label}</span>${memorySpaceBadge(`${id}-mtm-${stage}-${side}-space`, 'RMEM')}</span>
            <span style="display:flex;align-items:center;gap:4px">
              <button class="mode-btn" id="${id}-mtm-${stage}-${side}-val-btn" onclick="toggleMtmValue('${id}')">value</button>
              <button class="mode-btn" id="${id}-mtm-${stage}-${side}-svg-zoom" onclick="toggleZoom('${id}-mtm-${stage}-${side}-svg')">Zoom in</button>
            </span>
          </div>
          <div class="cuo-viz-desc" id="${id}-mtm-${stage}-${side}-desc"></div>
          <div class="viz-box"><div id="${id}-mtm-${stage}-${side}-svg"></div></div>
        </div>`;
  return `
    <!-- make_tiled_mma panel -->
    <div id="${id}-tab-make_tiled_mma" class="panel">
      <div class="controls">
        <h2>make_tiled_mma</h2>

        <div class="form-group">
          <button class="view-toggle" id="${id}-mtm-alt-btn" onclick="toggleMtmAltView('${id}')">
            <span class="view-toggle-icon">&#8862;</span>Alternative View
          </button>
        </div>

        <div class="form-group">
          <label>Cell labels ${infoIcon(MTM_MODE_HINT)}</label>
          <div class="seg-control" id="${id}-mtm-mode-btns">
            <button class="mode-btn" onclick="setMtmMode('${id}', 'tv')">Show TVs</button>
            <button class="mode-btn active" onclick="setMtmMode('${id}', 'warp')">Show Warps</button>
            <button class="mode-btn" id="${id}-mtm-group-mode-btn" style="display:none" onclick="setMtmMode('${id}', 'warpgroup')">Show Warp Groups</button>
          </div>
        </div>

        <div class="form-group">
          <!-- The label TEXT is its own span: mtmSyncFocusField rewrites it per
               mode with textContent, which would otherwise eat the icon. -->
          <label><span id="${id}-mtm-focus-label">${MTM_FOCUS.warp.label}</span>
            ${infoIcon(MTM_FOCUS.warp.hint, `${id}-mtm-focus-hint`)}</label>
          <input type="text" id="${id}-mtm-focus-input" value=""
                 oninput="setMtmHighlight('${id}')" placeholder="${MTM_FOCUS.warp.placeholder}">
        </div>

        <details class="cuo-section" open>
          <summary>1. MMAAtom</summary>
          <div class="cuo-section-body">
            ${mmaAtomEditorMarkup({id,p:'mtm',render:'renderMakeTiledMma',setOp:'setMtmOp',warpgroup:true})}
            <div id="${id}-mtm-op-params" class="cuo-result"></div>
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>2. make_tiled_mma(atom, atom_layout_mnk, permutation_mnk)</summary>
          <div class="cuo-section-body">
            <button class="btn btn-render btn-decode" id="${id}-mtm-decode" onclick="mtmDecodePasteboard('${id}')">Decode from Pasteboard</button>
            <details class="cuo-section" id="${id}-mtm-tiled-manual-construction"><summary>Manual construction</summary><div class="cuo-section-body">
            ${layoutInputField({
              id: `${id}-mtm-atomlayout-input`, label: 'atom_layout_mnk', value: '(2, 2, 1)',
              placeholder: '(1, 1, 1)',
              hint: 'Rank 3 — atoms along M, N and K: one warp or one 128-thread warp group per atom. A stride reorders which warp is which: (2,2,1):(2,1,4). Blank = (1, 1, 1), CuTeDSL\'s own default: one atom, which makes this identical to make_mma_atom.',
            })}
            ${layoutInputField({
              id: `${id}-mtm-perm-input`, label: 'permutation_mnk', value: '',
              placeholder: '(_, _, _)',
              hint: 'Rank-3 Tiler, one mode per axis. A number is a plain extent; (2,16):(16,1) genuinely permutes; "_" leaves the mode alone. Blank = none.',
            })}
            </div></details>
            <div id="${id}-mtm-result" class="cuo-result"></div>
          </div>
        </details>

        ${statusDivs(`${id}-mtm`)}
        <button class="btn btn-render" onclick="renderMakeTiledMma('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-mtm-export" onclick="exportMTM('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setMTM('${id}','f16bf16','half_t','float',16,'(2,2,1)','')">Warp MMA — four warps, (2,2,1)</button>
            <button class="preset-btn" onclick="setMTM('${id}','f16bf16','half_t','float',16,'(2,2,1)','(64,32,16)')">Warp MMA — repeated M/N tile</button>
            <button class="preset-btn" onclick="setMTM('${id}','tf32',null,null,8,'(2,2,1)','')">Warp MMA — TF32</button>
            <button class="preset-btn" onclick="setMTM('${id}','wg_f16bf16','half_t','float',16,'(1,1,1)','',64,'SMEM','K','K')">WGMMA — one warpgroup, 64×64×16</button>
            <button class="preset-btn" onclick="setMTM('${id}','wg_f16bf16','half_t','float',16,'(2,1,1)','',64,'SMEM','K','K')">WGMMA — two warpgroups along M, 256 threads</button>
            <button class="preset-btn" onclick="setMTM('${id}','wg_f16bf16','bfloat16_t','float',16,'(1,2,1)','',32,'SMEM','MN','MN')">WGMMA — M-major A / N-major B, two groups along N</button>
            <button class="preset-btn" onclick="setMTM('${id}','wg_f16bf16','half_t','float',16,'(1,1,1)','(128,64,16)',32,'RMEM','K','MN')">WGMMA — register A, repeated tile</button>
            <button class="preset-btn" onclick="setMTM('${id}','wg_fp8','float_e4m3_t','float',32,'(2,1,1)','',32,'RMEM','K','K','float_e5m2_t')">WGMMA — mixed FP8, register A</button>
            <button class="preset-btn" onclick="setMTM('${id}','wg_int8','int8_t','int32_t',32,'(1,2,1)','',24,'SMEM','K','K','uint8_t')">WGMMA — mixed signed/unsigned INT8</button>
            <button class="preset-btn" onclick="setMTM('${id}','wg_f16bf16','half_t','float',16,'(1,1,2)','',16,'SMEM','K','K')">WGMMA — split K, partial accumulators need reduction</button>
          </div>
        </div>

        <div class="hint">
          <b>Replicate instructions.</b> atom_layout_mnk counts atoms along M, N, K.
          A warp atom has 32 threads; a WGMMA atom has 128 threads (one warpgroup).
          For WGMMA, (2,1,1) means two warpgroups: eight warps and 256 threads.<br><br>
          <b>Views.</b> Show TVs displays register ownership. The other mode groups cells by
          warp or warpgroup. SMEM operands remain one-color logical tiles; focus applies to
          register operands only. Default orientation follows M-major A / N-major B where
          selected. Alternative View always shows MK × KN → MN.<br><br>
          <b>Replication and repetition.</b> atom_layout_mnk adds participating groups.
          A larger permutation_mnk repeats work over a larger tile using the same groups.
          Red boundaries mark the repetitions. A broadcasts across N groups; B across M groups.<br><br>
          <b>Split K.</b> Groups differing along K hold partial accumulators. The C view
          shows their overlapping ownership; make_tiled_mma performs no reduction.<br><br>
          <b>WGMMA execution.</b> Issue GEMM uniformly within each warpgroup, with the
          required fence, commit and wait operations. This constructor partitions work;
          it neither stages shared memory nor supplies those synchronization operations.
          Logical permutations still require a compatible physical SMEM descriptor layout.
          Copy, partition, fragment and gemm tabs currently accept warp MMA constructors only.

        </div>
      </div>

      <div class="comp-results" id="${id}-mtm-results">
        <div class="comp-viz-span mtm-group-title" id="${id}-mtm-warp-group-title">
          1. MMA Atom &times; atom_layout_mnk
        </div>
        <div class="mma-group">
          <div class="mma-q-spacer"></div>
${vizItem('warp', 'a', 'A')}
${vizItem('warp', 'b', 'B')}
${vizItem('warp', 'c', 'C')}
        </div>
        <div class="comp-viz-span mtm-group-title" id="${id}-mtm-perm-group-title">
          2. TiledMMA &mdash; after permutation_mnk
        </div>
        <div class="mma-group">
          <div class="mma-q-spacer"></div>
${vizItem('perm', 'a', 'A')}
${vizItem('perm', 'b', 'B')}
${vizItem('perm', 'c', 'C')}
        </div>
      </div>
    </div>`;
}

const mtmState = {};

function renderMakeTiledMma(tabId) {
  showErr(`${tabId}-mtm-error`, '');
  showWarn(`${tabId}-mtm-warning`, '');
  try {
    const opKey = document.getElementById(`${tabId}-mtm-op-input`).value;
    mmaSyncControls(tabId, opKey, 'mtm');
    const inputs = mtmReadConstructorControls(tabId, 'mtm');
    const {atom, atomLayout, perm, tiled: permStage} = mtmDecodeConstructor(inputs, {warpgroup: true});
    const {op, k} = atom;
    const abDtype = inputs[1] === 'na' ? 'tfloat32_t' : inputs[1];
    const accDtype = inputs[2] === 'na' ? 'float' : inputs[2];
    const previous = mtmState[tabId] || {};
    let mode = previous.mode || 'warp';
    if (atom.warpgroup && !previous.atom?.warpgroup) mode = 'warpgroup';
    if (!atom.warpgroup && mode === 'warpgroup') mode = 'warp';

    // The top row is the SAME derivation with the permutation left out — that
    // is precisely what permutation_mnk adds, so showing it against a
    // separately-computed baseline would be a second implementation to keep
    // honest rather than a comparison.
    const warpStage = perm.some(p => p !== null)
      ? mtmComputeTiledMma(atom, atomLayout, [null, null, null]) : permStage;

    const prev = mtmState[tabId] || {};
    mtmState[tabId] = {
      atom, atomLayout, perm, warpStage, permStage, abDtype, accDtype, op, k,
      hasPerm: perm.some(p => p !== null),
      mode, showValue: !!prev.showValue, altView: !!prev.altView,
      ...mtmReadFocus(tabId, mode, warpStage.warps, warpStage.threads, 'mtm', warpStage.atomGroups),
    };

    mtmSyncFocusField(tabId, mode);
    mmaRenderOpParams(tabId, op, k, abDtype, accDtype, 'mtm', atom);
    mtmRenderResult(tabId);
    mtmRenderViz(tabId);
    updateOuterTabLabel(tabId,
      `make_tiled_mma:${warpStage.atomLayoutMNK.join('x')}/${permStage.tileMNK.join('x')}`);
  } catch (e) {
    showErr(`${tabId}-mtm-error`, e.message);
    for (const stage of ['warp', 'perm'])
      for (const s of ['a', 'b', 'c']) {
        const el = document.getElementById(`${tabId}-mtm-${stage}-${s}-svg`);
        if (el) el.innerHTML = '';
      }
  }
}

const MTM_MODE_HINT =
  'Show TVs labels individual register T/V slots. Show Warps groups by physical 32-thread warp. ' +
  'Show Warp Groups groups WGMMA by its 128-thread atoms. SMEM descriptor tiles stay uniform.';

/** What the focus box means in each mode. One control, because the question is
 *  always "which unit am I looking at" — it is only the unit that changes with
 *  the cell labels. */
const MTM_FOCUS = {
  warpgroup: {
    label: 'Warp group id', placeholder: 'all warpgroups', unit: 'warpgroup', letter: 'WG',
    hint: 'Focus one 128-thread warpgroup. Register cells show WG<id>; shared-memory descriptor tiles remain uniform. Blank shows all groups.',
  },
  warp: {
    label: 'Warp id', placeholder: 'all warps', unit: 'warp', letter: 'W',
    hint: 'Applies to register grids. Only that warp\'s region stays coloured — every cell it ' +
          'touches reads W<id> — and every other cell goes grey and unlabelled. Blank lists ' +
          'every warp that touches a cell, one per line.',
  },
  tv: {
    label: 'Thread ID', placeholder: 'all threads', unit: 'thread', letter: 'T',
    hint: 'Applies to register grids. Only that thread\'s region stays coloured, and each of its ' +
          'cells shows the V slot that lands there; every other cell goes grey and unlabelled. ' +
          'Blank is make_mma_atom\'s picture — every thread at full brightness.',
  },
};

/** Read and VALIDATE the focus box. An id that is not a whole number, or one
 *  outside the range this TiledMMA has, is an ERROR: the count is derived from
 *  atom_layout_mnk, so a stale id after changing it is easy to hit and silently
 *  ignoring the value would leave the wrong picture on screen with no
 *  explanation. The grids still draw unfocused rather than blanking — the field
 *  re-reads on every keystroke, and typing "12" passes through "1".
 *
 *  Returns the message rather than writing it, so the error box has exactly one
 *  author (mtmRenderViz); two functions racing for one element is how a message
 *  goes missing. */
function mtmReadFocus(tabId, mode, warps, threads, p, warpgroups = warps) {
  p = p || 'mtm';
  const spec = MTM_FOCUS[mode] || MTM_FOCUS.warp;
  const limit = mode === 'tv' ? threads : mode === 'warpgroup' ? warpgroups : warps;
  const raw = (document.getElementById(`${tabId}-${p}-focus-input`).value || '').trim();
  if (raw === '') return { focus: null, focusErr: '' };
  if (!/^\d+$/.test(raw))
    return {
      focus: null,
      focusErr: `"${raw}" is not a ${spec.unit} id — enter a whole number from 0 to ` +
                `${limit - 1}, or leave the box empty to show every ${spec.unit}.`,
    };
  const n = parseInt(raw, 10);
  if (n >= limit)
    return {
      focus: null,
      focusErr: `${spec.label} ${n} is out of range — this TiledMMA has ${limit} ` +
                `${spec.unit}${limit === 1 ? '' : 's'} ` +
                `(${spec.letter}0–${spec.letter}${limit - 1}). Showing all of them.`,
    };
  return { focus: n, focusErr: '' };
}

/** Relabel the focus box for the active mode. The control is one box whose
 *  UNIT changes, so the label has to move with the toggle or it would name the
 *  wrong thing. */
function mtmSyncFocusField(tabId, mode, p, hintText) {
  p = p || 'mtm';
  const spec = MTM_FOCUS[mode] || MTM_FOCUS.warp;
  const label = document.getElementById(`${tabId}-${p}-focus-label`);
  const input = document.getElementById(`${tabId}-${p}-focus-input`);
  const hint = document.getElementById(`${tabId}-${p}-focus-hint`);
  if (label) label.textContent = spec.label;
  if (input) input.placeholder = spec.placeholder;
  // setAttribute, not innerHTML: the bubble is `content: attr(data-tooltip)`,
  // and going through the DOM means the text needs no escaping of its own.
  if (hint) hint.setAttribute('data-tooltip', hintText || spec.hint);
}

// The three view controls all re-enter renderMakeTiledMma rather than calling
// mtmRenderViz directly. Redrawing six grids dominates either way, and going
// through the full path means a view toggle can never repaint from a state the
// last Render failed to replace.
function setMtmHighlight(tabId) {
  renderMakeTiledMma(tabId);
}

function setMtmMode(tabId, mode) {
  if (mtmState[tabId]) mtmState[tabId].mode = mode;
  else mtmState[tabId] = { mode };
  mtmSyncFocusField(tabId, mode);
  renderMakeTiledMma(tabId);
}

/** The quadrant-layout toggle. Only the CSS class and B's drawn orientation
 *  change; the derivation is untouched, which is the point — B really is the
 *  same fragment, drawn with its axes the other way round. */
function toggleMtmAltView(tabId) {
  if (mtmState[tabId]) mtmState[tabId].altView = !mtmState[tabId].altView;
  renderMakeTiledMma(tabId);
}

function toggleMtmValue(tabId) {
  if (mtmState[tabId]) mtmState[tabId].showValue = !mtmState[tabId].showValue;
  renderMakeTiledMma(tabId);
}

function setMtmOp(tabId) {
  mmaSyncControls(tabId, document.getElementById(`${tabId}-mtm-op-input`).value, 'mtm');
  renderMakeTiledMma(tabId);
}

function mtmRenderResult(tabId) {
  const s = mtmState[tabId];
  const w = s.warpStage, p = s.permStage;
  const [M, N, K] = s.atom.shapeMNK;
  const permStr = s.hasPerm
    ? '(' + s.perm.map(x => (x ? formatLayoutStr(x.shape, x.stride) : '_')).join(', ') + ')'
    : '(_, _, _)';
  const line = (name, opnd, dims) =>
    `<div class="cuo-result-line">tv_layout_${name} = ` +
    `${formatLayoutStr(opnd.tv.shape, opnd.tv.stride)}</div>` +
    `<div class="cuo-result-line" style="color:#9ca3af">&nbsp;&nbsp;over the ` +
    `${opnd.tile[0]}&times;${opnd.tile[1]} ${dims} tile &mdash; ${opnd.frgSize} element` +
    `${opnd.frgSize === 1 ? '' : 's'}${opnd.space === 'SMEM' ? ' in SMEM (descriptor)' : '/thread per atom'}, Rest = (${opnd.restShape.join(', ')})</div>`;

  document.getElementById(`${tabId}-mtm-result`).innerHTML =
    `<div class="cuo-result-line"><b>TiledMMA&lt;${s.op.label.replace('warp.', '')}&gt;</b></div>` +
    `<div class="cuo-result-line">Thr Layout VMNK = ${formatLayoutStr(w.thrVmnk.shape, w.thrVmnk.stride)}` +
    `<span style="color:#9ca3af"> &mdash; ${w.warps} warp${w.warps === 1 ? '' : 's'}, ` +
    `${w.threads} thread${w.threads === 1 ? '' : 's'}${s.atom.warpgroup ? `; ${w.atomGroups} warpgroup${w.atomGroups === 1 ? '' : 's'}` : ''}</span></div>` +
    `<div class="cuo-result-line">Permutation MNK = ${permStr}</div>` +
    `<div class="cuo-result-line">Atom Shape MNK  = (${M}, ${N}, ${K})` +
    `<span style="color:#9ca3af"> &nbsp;&rarr;&nbsp; warps cover (${M * w.atomLayoutMNK[0]}, ` +
    `${N * w.atomLayoutMNK[1]}, ${K * w.atomLayoutMNK[2]})</span></div>` +
    `<div class="cuo-result-line">Tile Shape MNK  = <b>(${p.tileMNK.join(', ')})</b>` +
    `<span style="color:#9ca3af"> &mdash; tile_size(0/1/2)</span></div>` +
    line('A', p.A, 'A (M&times;K)') +
    line('B', p.B, 'B (N&times;K)') +
    line('C', p.C, 'C (M&times;N)');
}

function mtmRenderViz(tabId) {
  const s = mtmState[tabId];
  if (!s) return;
  document.getElementById(`${tabId}-mtm-mode-btns`).querySelectorAll('.mode-btn')
    .forEach(b => b.classList.toggle('active',
      b.textContent.trim() === ({tv: 'Show TVs', warp: 'Show Warps', warpgroup: 'Show Warp Groups'}[s.mode])));

  document.getElementById(`${tabId}-mtm-group-mode-btn`).style.display = s.atom.warpgroup ? '' : 'none';
  const host = document.getElementById(`${tabId}-mtm-results`);
  if (host) host.classList.toggle('mma-alt', !!s.altView);
  const altBtn = document.getElementById(`${tabId}-mtm-alt-btn`);
  if (altBtn) altBtn.classList.toggle('active', !!s.altView);

  const [aM, aN, aK] = s.warpStage.atomLayoutMNK;
  const bcast = {
    a: aN > 1 ? `Broadcast: the ${aN} N-warps read the same cells.` : '',
    b: aM > 1 ? `Broadcast: the ${aM} M-warps read the same cells.` : '',
    c: aK > 1 ? `Reduction: the ${aK} K-warps each hold a partial sum of the same cells.` : '',
  };
  const stages = [
    ['warp', s.warpStage, false,
     'Each warp covers one atom-sized area. ',
     `atom_layout_mnk = ${formatLayoutStr(s.atomLayout.shape, s.atomLayout.stride)}`],
    ['perm', s.permStage, true,
     s.hasPerm
       ? 'Only the first copy of the warp pattern is coloured; red lines mark where the copies meet. '
       : 'No permutation_mnk, so this is the same tile as above. ',
     s.hasPerm
       ? 'permutation_mnk = (' + s.perm.map(x => (x ? formatLayoutStr(x.shape, x.stride) : '_')).join(', ') + ')'
       : 'permutation_mnk = (_, _, _)'],
  ];

  // Every cell of every tile should be claimed by at least one thread — the
  // operands partition (or, across a broadcast mode, cover) their tiles
  // exactly. A hole is therefore an invariant violation, and the one failure
  // this drawing could otherwise show without saying anything: `mtmOperandGrid`
  // drops any offset outside the tile, so an input that pushed the map out of
  // range would come back as a quietly incomplete picture.
  const uncovered = [];

  for (const [stage, res, isPerm, desc, tail] of stages) {
    for (const side of ['a', 'b', 'c']) {
      const opnd = res[side.toUpperCase()];
      const dtype = side === 'c' ? s.accDtype : side === 'b' ? s.atom.bDtype || s.abDtype : s.abDtype;
      setMemorySpaceBadge(`${tabId}-mtm-${stage}-${side}-space`, opnd.space);
      // In the quadrant view B is drawn K x N so its K axis meets A's and its N
      // axis meets C's. Transposing the GRID re-reads the same cells with the
      // axes swapped — nothing about the derivation moves — but the `value`
      // overlay then needs its own cellIndex, since mtmBuildSVG's default
      // assumes the tile is col-major in its own codomain.
      const rot = s.altView ? side === 'b' : s.atom.warpgroup && side !== 'c' && opnd.major === 'MN';
      const dims = side === 'a' ? (rot ? 'K×M' : 'M×K') : side === 'b' ? (rot ? 'K×N' : 'N×K') : 'M×N';
      const btn = document.getElementById(`${tabId}-mtm-${stage}-${side}-val-btn`);
      if (btn) btn.classList.toggle('active', !!s.showValue);
      if (product(opnd.tile) > MAX_CELLS) throw new Error(`Tile too large: ${opnd.tile.join('×')} cells (max ${MAX_CELLS}).`);
      let grid = mtmOperandGrid(opnd);
      const holes = grid.reduce((n, row) => n + row.filter(c => !c.entries.length).length, 0);
      if (holes) uncovered.push(`${side.toUpperCase()} ${holes}`);
      let [rows, cols] = opnd.tile;
      let cellIndex;
      if (rot) {
        grid = mtmTransposeGrid(grid);
        [rows, cols] = [cols, rows];
        cellIndex = (m, n) => n + opnd.tile[0] * m;
      }

      document.getElementById(`${tabId}-mtm-${stage}-${side}-title`).textContent =
        `${side.toUpperCase()} — ${rows}×${cols} ${dtype} (${dims})` +
        (isPerm && opnd.restShape.some(r => r > 1) ? `, Rest ${opnd.restShape.join('×')}` : '');
      document.getElementById(`${tabId}-mtm-${stage}-${side}-desc`).innerHTML =
        `${opnd.space === 'SMEM' ? `Shared-memory descriptor, ${mmaMajorLabel(side.toUpperCase(), opnd.major)}. Uniform logical tile; focus applies to RMEM only. ` : desc.replace(/warp/g, s.atom.warpgroup ? 'warpgroup' : 'warp')}${bcast[side] ? bcast[side].replace(/warp/g, s.atom.warpgroup ? 'warpgroup' : 'warp') + ' ' : ''}` +
        `${rot ? 'Shown transposed. ' : ''}<code>${tail}</code>`;
      document.getElementById(`${tabId}-mtm-${stage}-${side}-svg`).innerHTML =
        mtmBuildSVG(grid, rows, cols, {
          cellIndex, space: opnd.space, atomThreads: s.permStage.atomThreads, groupPrefix: s.mode === 'warpgroup' ? 'WG' : 'W',
          mode: s.mode,
          focus: s.focus,
          dimRest: isPerm,
          sum: side === 'c',
          showValue: s.showValue,
        });
      applyZoomState(`${tabId}-mtm-${stage}-${side}-svg`);
    }
  }

  // Two boxes, one author each. The focus id is the user's mistake to fix, so
  // it is an error; an uncovered cell is the tool's, so it is a warning.
  showErr(`${tabId}-mtm-error`, s.focusErr);
  showWarn(`${tabId}-mtm-warning`, uncovered.length
    ? `Some cells are claimed by no thread and are drawn as "—" (${uncovered.join(', ')}). ` +
      `Every operand should cover its whole tile, so this picture is incomplete — please ` +
      `report the inputs that produced it.`
    : '');
}

function setMTM(tabId, opKey, ab, acc, k, atomLayout, perm, n=64, aSrc='SMEM', aMajor='K', bMajor='K', bDtype=ab) {
  const inputs = [opKey, ab || 'na', acc || 'na', String(k), atomLayout, perm || 'na'];
  if (MMA_WARPGROUP_OPS[opKey]) inputs.push(String(n), aSrc, aMajor, bMajor, bDtype);
  mtmWriteConstructorControls(tabId, 'mtm', inputs);
  renderMakeTiledMma(tabId);
}

function exportMTM(tabId) {
  exportURL(`${tabId}-mtm-export`, 'make_tiled_mma', ...mtmReadConstructorControls(tabId, 'mtm'));
}

// WGMMA is opt-in: existing partition/copy/fragment/GEMM consumers stay warp-only.
function mtmDecodeConstructor(inputs, {warpgroup = false} = {}) {
  const [opKey, ab, acc, k, atomText, permText] = inputs;
  const wg = !!MMA_WARPGROUP_OPS[opKey];
  if (wg && !warpgroup) throw new Error('This consumer supports warp MMA constructors only.');
  if (inputs.length !== (wg ? 11 : 6)) throw new Error('Incomplete or extra TiledMMA constructor parameters.');
  const {atom} = mmaDecodeAtomInputs([...inputs.slice(0,4), ...inputs.slice(6)]);
  const atomLayout = mtmParseAtomLayout(atomText);
  const perm = mtmParsePerm(permText === 'na' ? '' : permText);
  const tiled = mtmComputeTiledMma(atom, atomLayout, perm);
  return {atom, atomLayout, perm, tiled, inputs: inputs.slice()};
}

function mtmImportConstructor(tabId, text) {
  const parsed = parsePastedKeyParam(text);
  if (!parsed || parsed.feature !== 'make_tiled_mma') throw new Error('Paste a make_tiled_mma URL or its query string.');
  mtmWriteConstructorControls(tabId, 'mtm', parsed.inputs);
  renderMakeTiledMma(tabId);
}

async function mtmDecodePasteboard(tabId) {
  const text = await readPasteboardText('Paste the make_tiled_mma URL or its query string:');
  if (text == null) return;
  try { mtmImportConstructor(tabId, text); }
  catch (e) { showErr(`${tabId}-mtm-error`, e.message); }
}

// Shared editable fallback for tabs that primarily import a TiledMMA URL.
function mtmManualConstructorMarkup({id, p, render, setOp, atomLayout = '(2,2,1)', perm = '', k = null, errorPrefix = p}) {
  return `<details class="cuo-section" id="${id}-${p}-manual-construction">
    <summary>Manual construction</summary>
    <div class="cuo-section-body">
      <details class="cuo-section" open><summary>MMAAtom</summary><div class="cuo-section-body">
        ${mmaAtomEditorMarkup({id,p,render,setOp,k,errorPrefix})}
      </div></details>
      ${layoutInputField({id:`${id}-${p}-atomlayout-input`,label:'atom_layout_mnk',value:atomLayout,placeholder:'(1,1,1)',hint:'Warps along M, N, K. Blank uses (1,1,1).'})}
      ${layoutInputField({id:`${id}-${p}-perm-input`,label:'permutation_mnk',value:perm,placeholder:'(_,_,_)',hint:'An extent expands a mode; a layout permutes it. Blank leaves modes unchanged.'})}
    </div>
  </details>`;
}

function mtmWriteConstructorControls(tabId, p, inputs) {
  mtmDecodeConstructor(inputs, {warpgroup: p === 'mtm'}); // Validate before changing fields.
  mmaAssignAtomInputs(tabId, p, [...inputs.slice(0,4), ...inputs.slice(6)]);
  document.getElementById(`${tabId}-${p}-atomlayout-input`).value = inputs[4];
  document.getElementById(`${tabId}-${p}-perm-input`).value = inputs[5] === 'na' ? '' : inputs[5];
}

function mtmReadConstructorControls(tabId, p) {
  const read = key => document.getElementById(`${tabId}-${p}-${key}-input`).value;
  mmaSyncControls(tabId, read('op'), p);
  const atomInputs = mmaReadAtomInputs(tabId, p);
  return [...atomInputs.slice(0,4), read('atomlayout'), read('perm') || 'na', ...atomInputs.slice(4)];
}
