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
  universal: {
    label: 'CopyUniversalOp',
    ctor: 'cute.nvgpu.CopyUniversalOp()',
    kind: 'simt',
    cpasync: false,
    params: [],
    note: 'CuTe\'s generic single-thread load/store. Takes no constructor parameters — ' +
          'the dataclass has no fields at all.',
  },
  cpasync: {
    label: 'cpasync.CopyG2SOp',
    ctor: 'cute.nvgpu.cpasync.CopyG2SOp()',
    kind: 'simt',
    cpasync: true,
    params: [],
    note: 'SM80+ non-bulk cp.async, GMEM→SMEM. Its one field, cache_mode, ' +
          'defaults to ALWAYS and does not change the Atom\'s layouts.',
  },
  ldmatrix: {
    label: 'warp.LdMatrix8x8x16bOp',
    ctor: 'cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose, num_matrices)',
    kind: 'ldmatrix',
    cpasync: false,
    params: ['transpose', 'num_matrices'],
    note: 'PTX <code>ldmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].shared.b16</code>. ' +
          'A warp-collective SMEM→RMEM load whose source and destination layouts differ.',
  },
};

// cp.async hardware widths (SM80+ non-bulk cp.async): 4, 8 or 16 bytes.
const MCA_CPASYNC_BITS = [32, 64, 128];

// ldmatrix's own constants. The instruction is defined over 16-BIT units: an
// 8x8 matrix of them, addressed one 128-bit row per lane. Everything the atom
// reports is that fixed geometry re-expressed in the chosen element type.
const MCA_LDSM_ROW_BITS  = 128;   // one lane's address covers 16 B
const MCA_LDSM_UNIT_BITS = 16;    // .b16 — the granularity `.trans` operates on
const MCA_LDSM_MAX_ELEM_BITS = 64;

/** The Copy_Atom produced by `make_copy_atom(LdMatrix8x8x16bOp(transpose,
 *  num_matrices), dtype)`. DOM-free on purpose — `renderMakeCopyAtom` reads the
 *  form and draws, this does the arithmetic, and tests/run.js diffs it against
 *  CuTeDSL (see tests/cases.json, section `ldmatrix_atom`).
 *
 *  Every quantity below is the fixed 16-bit geometry divided through by the
 *  element width, which is exactly what `copy_internal_type` does in the DSL —
 *  hence `Float32` giving an 8x4 tile where `Float16` gives 8x8, for the same
 *  instruction. Verified against CuTeDSL for all 13 numeric types x {1,2,4} x
 *  {N,T}; see the derivation notes in CLAUDE.md.
 *
 *  Returns { thrId, src, dst, tile, liveLanes, ... } with src/dst as plain
 *  {shape, stride} pairs in this repo's Layout convention. */
function mcaLdmatrixAtom(elemBits, numMatrices, transpose) {
  const e = elemBits;
  const k = numMatrices;
  if (![1, 2, 4].includes(k))
    throw new Error(`num_matrices must be one of 1, 2, 4 (got ${k}) — ` +
                    `LdMatrix8x8x16bOp.__post_init__ rejects anything else.`);
  if (e > MCA_LDSM_MAX_ELEM_BITS)
    throw new Error(
      `ldmatrix is undefined for ${e}-bit elements: one lane addresses a ` +
      `${MCA_LDSM_ROW_BITS}-bit row, so an element wider than ` +
      `${MCA_LDSM_MAX_ELEM_BITS} bits leaves fewer than 2 elements per row and ` +
      `the 8x8 matrix degenerates.`);

  const R = MCA_LDSM_ROW_BITS / e;          // elements per lane-addressed row
  const S = (MCA_LDSM_ROW_BITS * 8) / e;    // elements in one 8x8 matrix
  const L = 8 * k;                          // lanes whose address is CONSUMED
  const M = 8 * k;                          // tile rows  (8 per matrix)
  const N = R;                              // tile cols  (one 16 B row)

  // ── ValLayoutSrc ───────────────────────────────────────────────────────
  // Lane t supplies the address of row t. `.x1` consumes lanes 0-7, `.x2`
  // 0-15, `.x4` all 32 — and the lanes it does NOT consume still execute the
  // instruction and still hand over an operand, so the layout has to be total
  // over the warp. CuTe writes that as a stride-0 second thread mode, which
  // aliases the ignored lanes onto live ones: in-bounds, branchless, and it
  // makes size(layout) exceed cosize by exactly the broadcast factor.
  const src = (L === 32)
    ? { shape: [32, R],          stride: [R, 1] }
    : { shape: [[L, 32 / L], R], stride: [[R, 0], 1] };

  // ── ValLayoutDst ───────────────────────────────────────────────────────
  let dst;
  if (!transpose) {
    // One 32-bit register per matrix per lane, in mma-fragment order.
    const q = Math.max(1, 32 / e);          // elements in that register
    const T = S / q;                        // lanes needed to hold one matrix
    dst = (k === 1)
      ? { shape: [T, q],      stride: [q, 1] }
      : { shape: [T, [q, k]], stride: [q, [1, S]] };
  } else {
    // `.trans` transposes the 8x8 matrix OF 16-BIT UNITS. In units the result
    // is ((4,8),(1,2,k)):((16,1),(1,8,64)); `scale` re-expresses it in elements
    // and is < 1 exactly when an element spans several units, which is why a
    // 32-bit type ends up with a 16-lane thread mode.
    const scale = MCA_LDSM_UNIT_BITS / e;
    const thrShape  = scale >= 1 ? [4, 8]              : [4, 8 * scale];
    const thrStride = scale >= 1 ? [16 * scale, scale] : [16 * scale, 1];
    const lead = scale >= 1 ? scale : 1;   // elements packed inside one unit
    dst = (k === 1)
      ? { shape: [thrShape, [lead, 2]],    stride: [thrStride, [1, 8 * scale]] }
      : { shape: [thrShape, [lead, 2, k]], stride: [thrStride, [1, 8 * scale, 64 * scale]] };
  }

  return {
    thrId: { shape: 32, stride: 1 },   // Layout<_32>; CuTeDSL prints 32:1
    src, dst,
    tile: { shape: [M, N], stride: [N, 1] },  // row-major: flat = m*N + n
    liveLanes: L, rowElems: R, matrixElems: S, numMatrices: k, transpose,
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

        <details class="cuo-section" open>
          <summary>1. The Copy Op</summary>
          <div class="cuo-section-body">
            <div class="form-group">
              <label>CopyAtom type</label>
              <select id="${id}-mca-op-input" onchange="setMcaOp('${id}')">
                <option value="universal" selected>CopyUniversalOp</option>
                <option value="cpasync">cpasync.CopyG2SOp</option>
                <option value="ldmatrix">warp.LdMatrix8x8x16bOp</option>
              </select>
            </div>
            <div id="${id}-mca-ldsm-params" style="display:none">
              <div class="form-group">
                <label>num_matrices<span style="color:#6b7280;font-weight:normal">&nbsp;&mdash; the .x1 / .x2 / .x4 qualifier</span></label>
                <select id="${id}-mca-nm-input" onchange="renderMakeCopyAtom('${id}')">
                  <option value="1">1 &mdash; .x1 (8 lanes supply addresses)</option>
                  <option value="2">2 &mdash; .x2 (16 lanes supply addresses)</option>
                  <option value="4" selected>4 &mdash; .x4 (all 32 lanes supply addresses)</option>
                </select>
              </div>
              <div class="form-group">
                <label>transpose<span style="color:#6b7280;font-weight:normal">&nbsp;&mdash; the .trans qualifier</span></label>
                <select id="${id}-mca-trans-input" onchange="renderMakeCopyAtom('${id}')">
                  <option value="0" selected>False</option>
                  <option value="1">True</option>
                </select>
              </div>
            </div>
            <div id="${id}-mca-op-params" class="cuo-result"></div>
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>2. make_copy_atom(op, dtype, ...)</summary>
          <div class="cuo-section-body">
            <div class="form-group">
              <label>tensor_dtype</label>
              <select id="${id}-mca-dtype-input" onchange="renderMakeCopyAtom('${id}')">${dtypeOptions('half_t')}</select>
            </div>
            <div class="form-group" id="${id}-mca-bits-group">
              <label>num_bits_per_copy</label>
              <input type="number" id="${id}-mca-bits-input" value="128" min="1" step="1">
            </div>
            <div id="${id}-mca-atom-result" class="cuo-result"></div>
          </div>
        </details>

        ${statusDivs(`${id}-mca`)}
        <button class="btn btn-render" onclick="renderMakeCopyAtom('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-mca-export" onclick="exportMCA('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setMCA('${id}','universal',128,'half_t')">CopyUniversalOp &mdash; 128b half_t &rarr; 8 elements</button>
            <button class="preset-btn" onclick="setMCA('${id}','cpasync',128,'half_t')">cpasync.CopyG2SOp &mdash; 128b half_t (cp.async.128)</button>
            <button class="preset-btn" onclick="setMCA('${id}','universal',64,'half_t')">CopyUniversalOp &mdash; 64b half_t &rarr; 4 elements</button>
            <button class="preset-btn" onclick="setMCA('${id}','universal',128,'float')">CopyUniversalOp &mdash; 128b float &rarr; 4 elements</button>
            <button class="preset-btn" onclick="setMCA('${id}','universal',8,'int8_t')">CopyUniversalOp &mdash; 8b int8_t &rarr; 1 element (scalar)</button>
            <button class="preset-btn" onclick="setMCA('${id}','ldmatrix',128,'half_t',4,0)">ldmatrix .x4 &mdash; half_t (all 32 lanes live)</button>
            <button class="preset-btn" onclick="setMCA('${id}','ldmatrix',128,'half_t',2,0)">ldmatrix .x2 &mdash; half_t (16 lanes broadcast)</button>
            <button class="preset-btn" onclick="setMCA('${id}','ldmatrix',128,'half_t',1,0)">ldmatrix .x1 &mdash; half_t (24 lanes broadcast)</button>
            <button class="preset-btn" onclick="setMCA('${id}','ldmatrix',128,'half_t',4,1)">ldmatrix .x4.trans &mdash; half_t (the transpose)</button>
            <button class="preset-btn" onclick="setMCA('${id}','ldmatrix',128,'int8_t',1,0)">ldmatrix .x1 &mdash; int8_t (16 elements per row)</button>
          </div>
        </div>

        <div class="hint">
          <b>Two steps, two kinds of parameter.</b> An Op is constructed first
          (with whatever fields it carries), then
          <code>make_copy_atom(op, dtype, num_bits_per_copy=N)</code> turns it
          into an Atom. <code>CopyUniversalOp</code> and
          <code>cpasync.CopyG2SOp</code> take no constructor parameters, so
          section 1 only asks which one; <code>warp.LdMatrix8x8x16bOp</code>
          takes two, and they appear there.<br><br>
          <b>ldmatrix: what the two parameters do.</b>
          <code>num_matrices</code> is the <code>.x1</code> /
          <code>.x2</code> / <code>.x4</code> qualifier, and it decides
          <em>how many lanes' addresses the hardware consumes</em> &mdash; 8, 16
          or 32. It does not change how much any one lane addresses: that is
          always one 128-bit row. <code>transpose</code> is
          <code>.trans</code>, which transposes each 8x8 matrix on the way to
          the registers, so a K-major SMEM tile can feed an M-major fragment
          with no extra shuffles.<br><br>
          <b>Why there is no unpack_bits here.</b> The field exists on the
          shared <code>BaseOp</code>, but <code>LdMatrix8x8x16bOp</code> rejects
          it outright &mdash; <code>__post_init__</code> raises
          <code>"Op doesn't support unpacking"</code> for anything but
          <code>None</code>. Unpacking belongs to the sub-byte Ops
          (<code>LdMatrix8x16x8bOp</code>, <code>LdMatrix16x8x8bOp</code>,
          <code>LdMatrix16x16x8bOp</code>), where
          <code>unpack_bits &isin; {4, 6}</code> selects the
          <code>.b4x16_p64</code> / <code>.b6x16_p32</code> qualifiers: a
          packed 4- or 6-bit source container widened into 8-bit registers on
          the way out. Those are separate Ops and will be separate entries in
          this dropdown.<br><br>
          <b>num_bits_per_copy is ignored by ldmatrix.</b> The instruction's
          width is fixed by the Op, so <code>_make_trait</code> never reads the
          argument &mdash; you can pass one and it changes nothing. The field is
          disabled rather than hidden so that is visible rather than
          mysterious.<br><br>
          <b>Src and Dst are different layouts, and not paired by (t, v).</b>
          For the two SIMT Ops <code>ValLayoutSrc == ValLayoutDst</code>, so the
          panes agree. For ldmatrix they do not: src is an <em>addressing</em>
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
    if (op.kind === 'ldmatrix') mcaRenderLdmatrix(tabId, opKey, op, dtype, elemBits, prev);
    else                        mcaRenderSimt(tabId, opKey, op, dtype, elemBits, prev);
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

// ── warp.LdMatrix8x8x16bOp ──────────────────────────────────────────────────
function mcaRenderLdmatrix(tabId, opKey, op, dtype, elemBits, prev) {
  const nm = parseInt(document.getElementById(`${tabId}-mca-nm-input`).value, 10);
  const transpose = document.getElementById(`${tabId}-mca-trans-input`).value === '1';
  const a = mcaLdmatrixAtom(elemBits, nm, transpose);

  // Validation CuTe skips. `.trans` on the m8n8 form is a `.b16` transpose, so
  // its granularity is a 16-bit unit whatever `copy_internal_type` says. The
  // DSL recasts happily and returns a layout either way; the layout is only a
  // faithful description of the instruction when the element IS the unit.
  // showWarn writes textContent, like every other tab's warnings — plain text
  // only, no markup.
  const notes = [];
  if (transpose && elemBits !== MCA_LDSM_UNIT_BITS) {
    notes.push(
      `.trans on .m8n8 is a 16-bit transpose (ldmatrix...trans.shared.b16), but ` +
      `${dtype} is ${elemBits}-bit. ` +
      (elemBits < MCA_LDSM_UNIT_BITS
        ? `${MCA_LDSM_UNIT_BITS / elemBits} adjacent ${dtype} elements move as one unit and ` +
          `stay adjacent — that is the leading value mode of ValLayoutDst.`
        : `One ${dtype} element spans ${elemBits / MCA_LDSM_UNIT_BITS} units, which the ` +
          `transpose separates, so the thread mode covers only ` +
          `${product(a.dst.shape[0])} of the 32 lanes. CuTeDSL returns this layout without ` +
          `complaint; whether it is the instruction you meant is another matter.`));
  }
  if (a.liveLanes < 32) {
    notes.push(
      `.x${nm} consumes addresses from lanes 0-${a.liveLanes - 1} only. The other ` +
      `${32 - a.liveLanes} still execute the instruction and still hand over an operand — ` +
      `CuTe maps them with a stride-0 thread mode, so size(ValLayoutSrc) = ` +
      `${product(a.src.shape[0]) * product(a.src.shape[1])} against cosize = ` +
      `${a.tile.shape[0] * a.tile.shape[1]}. They are drawn in transparent grey.`);
  }
  if (notes.length) showWarn(`${tabId}-mca-warning`, notes.join('  '));

  const numValSrc = product(a.src.shape[1]);
  const numValDst = product(a.dst.shape[1]);
  mcaState[tabId] = {
    kind: 'ldmatrix', opKey, op, dtype, elemBits, atom: a,
    numValSrc, numValDst, showValue: !!prev.showValue,
  };

  const q = `.x${nm}${transpose ? '.trans' : ''}`;
  document.getElementById(`${tabId}-mca-atom-result`).innerHTML =
    `<div class="cuo-result-line"><b>Copy_Atom&lt;SM75_U32x${nm}_LDSM_${transpose ? 'T' : 'N'}, ${dtype}&gt;</b></div>` +
    `<div class="cuo-result-line" style="color:#9ca3af">ldmatrix.sync.aligned.m8n8${q}.shared.b16</div>` +
    `<div class="cuo-result-line">ThrID        = ${formatLayoutStr(a.thrId.shape, a.thrId.stride)}` +
    `<span style="color:#9ca3af"> &mdash; warp-collective; all 32 lanes execute it</span></div>` +
    `<div class="cuo-result-line">ValLayoutSrc = ${a.srcStr}` +
    `<span style="color:#9ca3af"> &mdash; ${numValSrc} slots/lane</span></div>` +
    `<div class="cuo-result-line">ValLayoutDst = ${a.dstStr}` +
    `<span style="color:#9ca3af"> &mdash; ${numValDst} slots/lane</span></div>` +
    `<div class="cuo-result-line">tile         = ${a.tile.shape[0]} &times; ${a.tile.shape[1]} ${dtype} ` +
    `= ${a.numMatrices} matri${a.numMatrices === 1 ? 'x' : 'ces'} of 8 rows &times; ` +
    `${a.rowElems} (${MCA_LDSM_ROW_BITS} bits = 16 B per row)</div>` +
    `<div class="cuo-result-line">lanes supplying addresses = <b>${a.liveLanes}</b> of 32</div>` +
    `<div class="cuo-result-line" style="color:#9ca3af">` +
    `<code>num_bits_per_copy</code> is <b>not</b> read for this Op &mdash; the width is fixed ` +
    `by the instruction, so <code>_make_trait</code> ignores the argument.</div>`;

  document.getElementById(`${tabId}-mca-viz-desc`).innerHTML =
    `The same ${a.tile.shape[0]}&times;${a.tile.shape[1]} tile, indexed two ways. ` +
    `<b>SRC</b> is the <em>addressing</em> pattern: which lane hands over the ` +
    `address of which 16 B row. <b>DST</b> is the <em>register outcome</em>: ` +
    `which lane ends up holding which element. They cover the same cells and ` +
    `nothing more &mdash; there is no <code>(t, v)</code>-to-<code>(t, v)</code> ` +
    `correspondence between them, because the data crosses lanes inside the ` +
    `instruction. Cell colour is the thread id.` +
    (a.liveLanes < 32
      ? ` Lanes ${a.liveLanes}&ndash;31 are <b>disabled</b> on the src side: they hand ` +
        `over an operand that <code>.x${nm}</code> discards, so they are drawn in ` +
        `transparent grey on top of the live lane they alias.`
      : ` At <code>.x4</code> every lane's address is consumed, so nothing is greyed out.`);

  renderMcaAtomViz(tabId);
  updateOuterTabLabel(tabId, `make_copy_atom:ldmatrix${q}/${dtype}`);
}

// Section 1's body. The two SIMT Ops are parameterless, so this states that;
// ldmatrix's controls live in `-mca-ldsm-params` above and are toggled here.
function mcaRenderOpParams(tabId, op) {
  const ldsm = document.getElementById(`${tabId}-mca-ldsm-params`);
  if (ldsm) ldsm.style.display = op.kind === 'ldmatrix' ? '' : 'none';
  // num_bits_per_copy is meaningless for ldmatrix — grey it out rather than
  // hide it, so the absence is legible instead of just missing.
  const bits = document.getElementById(`${tabId}-mca-bits-input`);
  if (bits) bits.disabled = op.kind === 'ldmatrix';
  const bitsGroup = document.getElementById(`${tabId}-mca-bits-group`);
  if (bitsGroup) bitsGroup.style.opacity = op.kind === 'ldmatrix' ? '0.45' : '';

  const host = document.getElementById(`${tabId}-mca-op-params`);
  if (!host) return;
  host.innerHTML =
    `<div class="cuo-result-line"><b>${op.ctor}</b></div>` +
    (op.params.length === 0
      ? `<div class="cuo-result-line" style="color:#9ca3af">No constructor parameters required. ` +
        `${op.note}</div>`
      : `<div class="cuo-result-line" style="color:#9ca3af">${op.note}</div>` +
        `<div class="cuo-result-line" style="color:#9ca3af">Constructor parameters: ` +
        op.params.map(p => `<code>${p}</code>`).join(', ') +
        `. <code>unpack_bits</code> is inherited from <code>BaseOp</code> but ` +
        `rejected by this Op &mdash; it belongs to the sub-byte LdMatrix variants.</div>`);
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
    btn.style.display = s.kind === 'ldmatrix' ? '' : 'none';
    btn.classList.toggle('active', !!s.showValue);
  }
  if (s.kind === 'ldmatrix') mcaDrawLdmatrix(tabId, s);
  else                       mcaDrawSimt(tabId, s);
}

function mcaDrawSimt(tabId, s) {
  // Every cell gets the same colour — there is nothing to distinguish, since one
  // instruction moves the whole row. It is the colour the make_tiled_copy tabs
  // give T0's first atom invocation, so the two pictures line up visually.
  const initColor = colorTV(0);
  // ValLayoutSrc == ValLayoutDst for both Ops here, so the two panes draw the
  // same grid. Rendering them separately anyway keeps the code honest for
  // atoms where they differ.
  for (const side of ['src', 'dst']) {
    const layoutStr = side === 'src' ? s.atomStr : s.atomDstStr;
    const lp = parseLayout(layoutStr);
    document.getElementById(`${tabId}-mca-${side}-svg`).innerHTML =
      `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
      `ValLayout${side === 'src' ? 'Src' : 'Dst'} = ${layoutStr} &mdash; ` +
      `${s.elements} contiguous ${s.dtype} element${s.elements === 1 ? '' : 's'}, one instruction` +
      `</div>` +
      buildColoredLayoutSVG(lp.shape, lp.stride, 'value', (m, n, offset) => {
        // One quantity per cell: which value of the atom this slot holds.
        return { bg: initColor, text: [`V${offset}`] };
      });
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
function mcaDrawLdmatrix(tabId, s) {
  const a = s.atom;
  const [M, N] = a.tile.shape;
  const dimTids = new Set();
  for (let t = a.liveLanes; t < 32; t++) dimTids.add(t);
  const labelMode = s.showValue ? 'value' : '';

  for (const side of ['src', 'dst']) {
    const L = side === 'src' ? a.src : a.dst;
    const str = side === 'src' ? a.srcStr : a.dstStr;
    const numV = product(L.shape[1]);
    const head = side === 'src'
      ? `ValLayoutSrc = ${str} — ${a.liveLanes} of 32 lanes supply an address, ` +
        `${numV} ${s.dtype} per row`
      : `ValLayoutDst = ${str} — ${numV} ${s.dtype} per lane` +
        (a.numMatrices > 1 ? `, ${a.numMatrices} matrices` : '');
    document.getElementById(`${tabId}-mca-${side}-svg`).innerHTML =
      `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">${head}</div>` +
      buildTVSVG(L.shape, L.stride, a.tile.shape, a.tile.stride, false, 'row', null, labelMode,
                 { dimTids: side === 'src' ? dimTids : null,
                   cellIndex: (m, n) => m * N + n });
    applyZoomState(`${tabId}-mca-${side}-svg`);
  }
  const q = `.x${a.numMatrices}${a.transpose ? '.trans' : ''}`;
  document.getElementById(`${tabId}-mca-atom-title`).textContent =
    `ldmatrix.m8n8${q} — ${s.dtype} → ${M}×${N} tile, ${a.liveLanes}/32 lanes addressing`;
}


function setMCA(tabId, opKey, bits, dtype, nm, tr) {
  document.getElementById(`${tabId}-mca-op-input`).value    = opKey;
  document.getElementById(`${tabId}-mca-bits-input`).value  = bits;
  document.getElementById(`${tabId}-mca-dtype-input`).value = dtype;
  if (nm !== undefined) document.getElementById(`${tabId}-mca-nm-input`).value = String(nm);
  if (tr !== undefined) document.getElementById(`${tabId}-mca-trans-input`).value = String(tr);
  renderMakeCopyAtom(tabId);
}

function exportMCA(tabId) {
  const opKey = document.getElementById(`${tabId}-mca-op-input`).value;
  const base = [
    opKey,
    document.getElementById(`${tabId}-mca-bits-input`).value,
    document.getElementById(`${tabId}-mca-dtype-input`).value,
  ];
  // Only ldmatrix carries the extra two, so a CopyUniversalOp link keeps the
  // 3-input form it has always had.
  if ((MCA_OPS[opKey] || {}).kind === 'ldmatrix') {
    base.push(document.getElementById(`${tabId}-mca-nm-input`).value);
    base.push(document.getElementById(`${tabId}-mca-trans-input`).value);
  }
  exportURL(`${tabId}-mca-export`, 'make_copy_atom', ...base);
}
