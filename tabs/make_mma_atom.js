// make_mma_atom tab (MMA scope): build ONE warp-level MMA Atom and visualize
// its three operand fragments. Functions become globals on `window`.
//
// Mirrors `cute.make_mma_atom(op)` — the same two-step shape every atom in the
// DSL uses (python/CuTeDSL/cutlass/cute/atom.py:748):
//
//   op   = cute.nvgpu.warp.MmaF16BF16Op(Float16, Float32, (16,8,16))   <- step 1
//   atom = cute.make_mma_atom(op)                                      <- step 2
//
// Note `make_mma_atom` takes ONLY the op — no dtype argument, unlike
// make_copy_atom. An MMA's input and accumulator types are part of the
// instruction, so they live on the Op.
//
// An MMA Atom is the exact analogue of a Copy Atom, with one difference that
// drives this whole tab: a Copy Atom has two TV layouts (src, dst) over ONE
// tile, while an MMA Atom has THREE (A, B, C) over THREE tiles —
//
//   tv_layout_A  over  (M, K)
//   tv_layout_B  over  (N, K)
//   tv_layout_C  over  (M, N)
//
// because the instruction reads two operands and accumulates into a third.
//
// SCOPE: `cute.nvgpu.warp`, and within it the three DENSE `WarpMmaOp`
// subclasses — MmaF16BF16Op, MmaTF32Op, MmaFP8Op. Every configuration offered
// here has been compiled and EXECUTED on a GB200 (tests/gpu_check.py), not just
// traced — tracing alone would not distinguish a usable instruction from one the
// DSL can describe but not lower.
//
// Deliberately excluded:
//   MmaF16BF16SparseOp — the fourth WarpMmaOp subclass. Structured (2:4)
//     sparsity is a niche path, and it does not fit the picture this tab draws:
//     `MmaAtom` exposes no metadata (E) layout, and its A layout is over the
//     LOGICAL (M,K) tile, so the compression that makes it sparse is invisible
//     in the only thing shown. A drawing that cannot show the distinguishing
//     feature is worse than no entry. It traces fine, so re-adding it is a
//     table entry plus a `sparse_metadata_format` control if wanted.
//   MmaMXF4Op / MmaMXF8Op / MmaMXF4NVF4Op / MmaMXF8F6F4Op — block-scaled, they
//     subclass `MmaOp` directly rather than `WarpMmaOp`, are sm_120-only, and
//     could not be traced to build an oracle. Absent rather than untested.

// ── The layouts are a TABLE, not a formula ────────────────────────────────
// CUTLASS defines these as hand-written `MMA_Traits` specializations — one
// per instruction, with the layouts spelled out as literal types. E.g.
// include/cute/atom/mma_traits_sm80.hpp:78 —
//
//   struct MMA_Traits<SM80_16x8x16_F16F16F16F16_TN> {
//     using Shape_MNK = Shape<_16,_8,_16>;
//     using ThrID   = Layout<_32>;
//     using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2,  _2>>,
//                            Stride<Stride<_32,_1>,Stride<_16,_8,_128>>>;
//     using BLayout = ...;  using CLayout = SM80_16x8_Row;
//   };
//
// There is no derivation to port: an MMA atom IS the PTX register signature
// transcribed into layout form. So this table is a faithful port, and the
// differential tests (tests/cases.json, section `mma_atom`) check every entry
// against CuTeDSL rather than against a guess.
//
// Keyed by `<opKey>|<K>` because — verified across all 28 valid parameter
// combinations — the layouts depend ONLY on the Op family and K. Neither
// neither ab_dtype nor acc_dtype changes them: they are
// stated in units of ELEMENTS, and how many elements pack into a 32-bit
// register is fixed by the family.
const MMA_ATOM_TABLE = {
  'f16bf16|8':        { A: '((4,8),(2,2)):((32,1),(16,8))',
                        B: '((4,8),2):((16,1),8)' },
  'f16bf16|16':       { A: '((4,8),(2,2,2)):((32,1),(16,8,128))',
                        B: '((4,8),(2,2)):((16,1),(8,64))' },
  'tf32|4':           { A: '((4,8),2):((16,1),8)',
                        B: '((4,8),1):((8,1),0)' },
  'tf32|8':           { A: '((4,8),(2,2)):((16,1),(8,64))',
                        B: '((4,8),2):((8,1),32)' },
  'fp8|16':           { A: '((4,8),(4,2)):((64,1),(16,8))',
                        B: '((4,8),4):((32,1),8)' },
  'fp8|32':           { A: '((4,8),(4,2,2)):((64,1),(16,8,256))',
                        B: '((4,8),(4,2)):((32,1),(8,128))' },
};

// C is the SAME for every warp MMA here: a 16x8 accumulator, 4 values per
// lane, rows {t/4, t/4+8} x cols {2*(t%4), +1}. It does not vary with K or
// with acc_dtype — an f16 accumulator packs the same 4 values into 2
// registers instead of 4, which changes the register count, not the layout.
const MMA_C_LAYOUT = '((4,8),(2,2)):((32,1),(16,8))';

// The three dense `WarpMmaOp` subclasses. `params` drives which controls
// section 1 shows; `kDomain` / dtype domains are the `__post_init__` checks,
// mirrored so an out-of-domain control can never be selected.
const MMA_WARP_OPS = {
  f16bf16: {
    label: 'warp.MmaF16BF16Op',
    ctor:  'cute.nvgpu.warp.MmaF16BF16Op(ab_dtype, acc_dtype, shape_mnk)',
    params: ['ab_dtype', 'acc_dtype', 'shape_mnk'],
    ab: ['half_t', 'bfloat16_t'],
    acc: ['half_t', 'float'],
    kDomain: [8, 16],
    ptx: 'mma.sync.aligned.m16n8k{K}.row.col.{acc}.f16.f16.{acc}',
    note: 'The classic Ampere tensor-core MMA. bfloat16_t requires a float accumulator.',
  },
  tf32: {
    label: 'warp.MmaTF32Op',
    ctor:  'cute.nvgpu.warp.MmaTF32Op(shape_mnk)',
    params: ['shape_mnk'],
    ab: null, acc: null,               // implied: tf32 in, f32 out
    kDomain: [4, 8],
    ptx: 'mma.sync.aligned.m16n8k{K}.row.col.f32.tf32.tf32.f32',
    note: 'Takes no dtype parameters at all — tf32 in and f32 out are part of the instruction.',
  },
  fp8: {
    label: 'warp.MmaFP8Op',
    ctor:  'cute.nvgpu.warp.MmaFP8Op(ab_dtype, acc_dtype, shape_mnk)',
    params: ['ab_dtype', 'acc_dtype', 'shape_mnk'],
    ab: ['float_e4m3_t', 'float_e5m2_t'],
    acc: ['half_t', 'float'],
    kDomain: [16, 32],
    ptx: 'mma.sync.aligned.m16n8k{K}.row.col.{acc}.e4m3/e5m2...',
    note: 'SM89+ 8-bit tensor cores. Both FP8 encodings give identical layouts — they differ ' +
          'only in how the 8 bits are interpreted.',
  },
};

const MMA_M = 16, MMA_N = 8;   // every warp MMA here is m16n8k*

/** The Atom produced by `make_mma_atom(op)`, DOM-free so tests/run.js can diff
 *  it against CuTeDSL (see tests/cases.json, section `mma_atom`).
 *
 *  Returns the three TV layouts with the tile each is stated over. The tiles
 *  come straight from shape_mnk — A is (M,K), B is (N,K), C is (M,N) — and
 *  every one is COL-major in its own codomain (flat = row + rows*col), so
 *  buildTVSVG's default `value` overlay is already the layout's output and no
 *  cellIndex override is needed. Verified: size == cosize == M*K / N*K / M*N
 *  for all 28 combinations, i.e. every cell is owned exactly once and the
 *  atom has no broadcast. */
function mmaWarpAtom(opKey, k) {
  const op = MMA_WARP_OPS[opKey];
  if (!op) throw new Error(`Unknown MMA Op "${opKey}"`);
  if (!op.kDomain.includes(k))
    throw new Error(
      `${op.label} accepts shape_mnk (16,8,K) with K in {${op.kDomain.join(', ')}} — ` +
      `got K = ${k}. __post_init__ rejects anything else.`);
  const t = MMA_ATOM_TABLE[`${opKey}|${k}`];
  if (!t) throw new Error(`No layout recorded for ${opKey} K=${k}`);

  const mk = (str, rows, cols) => {
    const p = parseLayout(str);
    return {
      shape: p.shape, stride: p.stride, str,
      // (rows, cols) col-major: flat = r + rows*c
      tile: { shape: [rows, cols], stride: [1, rows] },
      numVal: product(p.shape[1]),
    };
  };
  return {
    opKey, op, k,
    shapeMNK: [MMA_M, MMA_N, k],
    thrId: { shape: 32, stride: 1 },        // Layout<_32>; CuTeDSL prints 32:1
    A: mk(t.A, MMA_M, k),
    B: mk(t.B, MMA_N, k),
    C: mk(MMA_C_LAYOUT, MMA_M, MMA_N),
  };
}

function generateMakeMmaAtomTabContent(id) {
  // A and B sit side by side in the two columns `.comp-results` already
  // provides; C spans both, directly underneath, because it is the operand
  // over a different pair of axes (M x N, not M/N x K) and is what the
  // epilogue actually has to deal with.
  const vizItem = (side, label) => `
        <div class="comp-viz-item" data-q="${side}">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-mma-${side}-title">${label}</span>
            <span style="display:flex;align-items:center;gap:4px">
              <button class="mode-btn" id="${id}-mma-${side}-val-btn" onclick="toggleMmaValue('${id}')">value</button>
              <button class="mode-btn" id="${id}-mma-${side}-svg-zoom" onclick="toggleZoom('${id}-mma-${side}-svg')">Zoom in</button>
            </span>
          </div>
          <div class="cuo-viz-desc" id="${id}-mma-${side}-desc"></div>
          <div class="viz-box"><div id="${id}-mma-${side}-svg"></div></div>
        </div>`;
  return `
    <!-- make_mma_atom panel -->
    <div id="${id}-tab-make_mma_atom" class="panel">
      <div class="controls">
        <h2>make_mma_atom</h2>

        <div class="form-group">
          <button class="view-toggle" id="${id}-mma-alt-btn" onclick="toggleMmaAltView('${id}')">
            <span class="view-toggle-icon">&#8862;</span>Alternative View
          </button>
        </div>

        <details class="cuo-section" open>
          <summary>1. The MMA Op</summary>
          <div class="cuo-section-body">
            <div class="form-group">
              <label>MmaAtom type<span style="color:#6b7280;font-weight:normal">&nbsp;&mdash; warp-level (cute.nvgpu.warp) only</span></label>
              <select id="${id}-mma-op-input" onchange="setMmaOp('${id}')">
                <option value="f16bf16" selected>warp.MmaF16BF16Op</option>
                <option value="tf32">warp.MmaTF32Op</option>
                <option value="fp8">warp.MmaFP8Op</option>
              </select>
            </div>
            <div class="form-group" id="${id}-mma-ab-group">
              <label>ab_dtype</label>
              <select id="${id}-mma-ab-input" onchange="renderMakeMmaAtom('${id}')"></select>
            </div>
            <div class="form-group" id="${id}-mma-acc-group">
              <label>acc_dtype</label>
              <select id="${id}-mma-acc-input" onchange="renderMakeMmaAtom('${id}')"></select>
            </div>
            <div class="form-group">
              <label>shape_mnk<span style="color:#6b7280;font-weight:normal">&nbsp;&mdash; M and N are fixed at 16x8; only K varies</span></label>
              <select id="${id}-mma-k-input" onchange="renderMakeMmaAtom('${id}')"></select>
            </div>
            <div id="${id}-mma-op-params" class="cuo-result"></div>
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>2. make_mma_atom(op)</summary>
          <div class="cuo-section-body">
            <div id="${id}-mma-atom-result" class="cuo-result"></div>
          </div>
        </details>

        <div class="form-group">
          <label>Highlight thread (empty = show all threads)</label>
          <input type="text" id="${id}-mma-highlight-tid" value="" placeholder="e.g. 5" oninput="setMmaHighlight('${id}')">
        </div>

        ${statusDivs(`${id}-mma`)}
        <button class="btn btn-render" onclick="renderMakeMmaAtom('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-mma-export" onclick="exportMMA('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setMMA('${id}','f16bf16','half_t','float',16)">m16n8k16 &mdash; f16 x f16 + f32 (the canonical one)</button>
            <button class="preset-btn" onclick="setMMA('${id}','f16bf16','half_t','float',8)">m16n8k8 &mdash; f16 x f16 + f32 (half the K)</button>
            <button class="preset-btn" onclick="setMMA('${id}','f16bf16','half_t','half_t',16)">m16n8k16 &mdash; f16 accumulator (same layouts, half the registers)</button>
            <button class="preset-btn" onclick="setMMA('${id}','f16bf16','bfloat16_t','float',16)">m16n8k16 &mdash; bf16 x bf16 + f32</button>
            <button class="preset-btn" onclick="setMMA('${id}','tf32',null,null,8)">m16n8k8 &mdash; tf32 (no dtype parameters)</button>
            <button class="preset-btn" onclick="setMMA('${id}','tf32',null,null,4)">m16n8k4 &mdash; tf32 (B is one element per lane)</button>
            <button class="preset-btn" onclick="setMMA('${id}','fp8','float_e4m3_t','float',32)">m16n8k32 &mdash; fp8 e4m3 (4 elements per register)</button>
          </div>
        </div>

        <div class="hint">
          <b>An MMA Atom is one instruction's register signature.</b> It says how
          many threads take part (<code>ThrID</code>) and which elements of A, B
          and C each of them holds. That is the whole object &mdash; there is
          nothing here about memory, tiles or how the data arrives.<br><br>
          <b>Three TV layouts over three different tiles.</b> This is the one
          structural difference from a Copy Atom, whose <code>src</code> and
          <code>dst</code> are two views of a <em>single</em> tile. An MMA reads
          two operands and accumulates into a third, so:
          <code>tv_layout_A</code> is over <code>(M, K)</code>,
          <code>tv_layout_B</code> over <code>(N, K)</code>, and
          <code>tv_layout_C</code> over <code>(M, N)</code>. All three are
          bijections &mdash; <code>size == cosize</code> &mdash; so every cell is
          owned by exactly one lane, with no broadcast.<br><br>
          <b>One thread mode, three value modes.</b> Every layout here shares the
          thread mode <code>(4, 8)</code>: <code>t/4</code> picks the row,
          <code>t%4</code> picks the column pair. The value modes just say which
          of the doublings apply &mdash; A doubles in both M and K, B only in K,
          C only in M.<br><br>
          <b>The dtypes pick the instruction, not the layouts.</b> Verified over
          all 16 valid parameter combinations: the three layouts depend only on
          the Op family and K. <code>acc_dtype</code> never changes
          <code>tv_layout_C</code> &mdash; an f16 accumulator packs the same four
          values into two registers instead of four, which changes the register
          count, not the map. Nor does <code>ab_dtype</code> within a family:
          half_t and bfloat16_t agree, and so do e4m3 and e5m2.<br><br>
          <b>Why this tab is a table.</b> CUTLASS defines these as hand-written
          <code>MMA_Traits</code> specializations &mdash; literal
          <code>Layout&lt;Shape&lt;...&gt;, Stride&lt;...&gt;&gt;</code> typedefs,
          one per instruction (<code>include/cute/atom/mma_traits_sm80.hpp:78</code>).
          There is no derivation to port, because an MMA atom is a hardware fact
          rather than a computation. Every entry is diffed against CuTeDSL.<br><br>
          <b>Scope.</b> <code>cute.nvgpu.warp</code>, and within it the three
          <em>dense</em> <code>WarpMmaOp</code> subclasses.
          <code>MmaF16BF16SparseOp</code> is left out: <code>MmaAtom</code>
          exposes no metadata (E) layout and its A layout is over the
          <em>logical</em> (M,&nbsp;K) tile, so the 2:4 compression that makes it
          sparse would be invisible in the only thing this tab draws. The
          block-scaled Ops (<code>MmaMXF4Op</code>, <code>MmaMXF8Op</code>,
          <code>MmaMXF4NVF4Op</code>, <code>MmaMXF8F6F4Op</code>) subclass
          <code>MmaOp</code> directly and are sm_120-only. Hopper
          <code>wgmma</code> and Blackwell <code>tcgen05</code> are a different
          shape of problem: their operands come from SMEM descriptors and TMEM
          respectively, so the "three register fragments" reading does not
          hold.<br><br>
          <b>Next step.</b> <code>make_tiled_mma(op_or_atom, atom_layout_mnk)</code>
          replicates this atom across warps &mdash; the MMA counterpart of
          <b>make_tiled_copy</b>. And <code>tv_layout_A</code> is exactly what
          <code>make_tiled_copy_A</code> hands to a copy as its target, which is
          how <b>make_copy_atom</b>'s ldmatrix ends up producing this fragment.
        </div>
      </div>

      <div class="comp-results" id="${id}-mma-results">
        <div class="mma-group">
          <div class="mma-q-spacer"></div>
${vizItem('a', 'A')}
${vizItem('b', 'B')}
${vizItem('c', 'C')}
        </div>
      </div>
    </div>`;
}

const mmaState = {};

function renderMakeMmaAtom(tabId) {
  showErr(`${tabId}-mma-error`, '');
  showWarn(`${tabId}-mma-warning`, '');
  try {
    const opKey = document.getElementById(`${tabId}-mma-op-input`).value;
    const op = MMA_WARP_OPS[opKey] || MMA_WARP_OPS.f16bf16;
    mmaSyncControls(tabId, opKey);

    const k = parseInt(document.getElementById(`${tabId}-mma-k-input`).value, 10);
    const abDtype  = op.ab  ? document.getElementById(`${tabId}-mma-ab-input`).value  : 'tfloat32_t';
    const accDtype = op.acc ? document.getElementById(`${tabId}-mma-acc-input`).value : 'float';

    // Mirrored from __post_init__: bf16 has no f16-accumulator form.
    if (op.ab && abDtype === 'bfloat16_t' && accDtype !== 'float') {
      throw new Error(
        `${op.label} requires acc_dtype = float when ab_dtype is bfloat16_t ` +
        `(there is no bf16 MMA with an f16 accumulator).`);
    }

    const a = mmaWarpAtom(opKey, k);
    const prev = mmaState[tabId] || {};
    // Same control, same convention as the tv / make_tiled_copy /
    // make_tiled_copy_tv tabs: live on `oninput`, an out-of-range id WARNS and
    // shows everything rather than being ignored, and buildTVSVG does the
    // dimming so a dimmed cell keeps its T/V labels instead of going blank.
    const hl = readHighlightTid(tabId, 'mma', product(a.thrId.shape));
    showWarn(`${tabId}-mma-warning`, hl.warn);
    mmaState[tabId] = { ...a, abDtype, accDtype, showValue: !!prev.showValue,
                        altView: !!prev.altView, highlightTid: hl.tid };

    mmaRenderOpParams(tabId, op, k, abDtype, accDtype);
    mmaRenderResult(tabId);
    mmaRenderViz(tabId);
    updateOuterTabLabel(tabId, `make_mma_atom:${op.label.replace('warp.', '')}/k${k}`);
  } catch (e) {
    showErr(`${tabId}-mma-error`, e.message);
    for (const s of ['a', 'b', 'c']) {
      const el = document.getElementById(`${tabId}-mma-${s}-svg`);
      if (el) el.innerHTML = '';
    }
  }
}

/** Rebuild every per-Op <select> and hide the ones this Op does not take.
 *
 *  A <select> whose options are rebuilt per Op must be REPOPULATED BEFORE its
 *  value is assigned — `sel.value = 'x'` on a select still holding the previous
 *  Op's options is a silent no-op and the render then runs on a stale
 *  parameter. `setMMA` and `applyKeyParam` both call this first, for the same
 *  reason `mcaRenderOpParams` exists on the copy side.
 *
 *  `p` is the id prefix, defaulting to this tab's own `mma`. make_tiled_mma
 *  reuses section 1 verbatim under `mtm`, and one implementation of the
 *  repopulate-then-assign order is easier to keep right than two. */
function mmaSyncControls(tabId, opKey, p) {
  p = p || 'mma';
  const op = MMA_WARP_OPS[opKey] || MMA_WARP_OPS.f16bf16;
  const fill = (sel, values, labelFn) => {
    if (!sel || !values.length) return;
    const want = sel.value;
    sel.innerHTML = values.map(v => `<option value="${v}">${labelFn ? labelFn(v) : v}</option>`).join('');
    // Assign the fallback EXPLICITLY rather than leaning on the browser
    // auto-selecting option 0 after an innerHTML swap. Same shape as
    // syncCopyMoves: keep the old value when the new Op still permits it,
    // otherwise take the first. Without this the very first render reads '' and
    // parseInt gives NaN, since the template ships these selects empty.
    sel.value = values.map(String).includes(want) ? want : String(values[0]);
  };
  const show = (groupId, on) => {
    const g = document.getElementById(groupId);
    if (g) g.style.display = on ? '' : 'none';
  };

  fill(document.getElementById(`${tabId}-${p}-ab-input`), op.ab || []);
  fill(document.getElementById(`${tabId}-${p}-acc-input`), op.acc || []);
  fill(document.getElementById(`${tabId}-${p}-k-input`), op.kDomain,
       v => `(16, 8, ${v})   —   m16n8k${v}`);
  show(`${tabId}-${p}-ab-group`, !!op.ab);
  show(`${tabId}-${p}-acc-group`, !!op.acc);
}

function setMmaOp(tabId) {
  mmaSyncControls(tabId, document.getElementById(`${tabId}-mma-op-input`).value);
  renderMakeMmaAtom(tabId);
}

function mmaRenderOpParams(tabId, op, k, abDtype, accDtype, p) {
  const host = document.getElementById(`${tabId}-${p || 'mma'}-op-params`);
  if (!host) return;
  const args = [];
  if (op.ab)  args.push(abDtype);
  if (op.acc) args.push(accDtype);
  args.push(`(16, 8, ${k})`);
  host.innerHTML =
    `<div class="cuo-result-line"><b>${op.label}(${args.join(', ')})</b></div>` +
    `<div class="cuo-result-line" style="color:#9ca3af">${op.ptx.replace('{K}', k)
        .replace(/\{acc\}/g, accDtype === 'half_t' ? 'f16' : 'f32')}</div>` +
    `<div class="cuo-result-line" style="color:#9ca3af">${op.note}</div>`;
}

function mmaRenderResult(tabId) {
  const s = mmaState[tabId];
  const [M, N, K] = s.shapeMNK;
  const abBits  = DTYPE_BITS[s.abDtype]  || 32;
  const accBits = DTYPE_BITS[s.accDtype] || 32;
  // Every Op here is dense, so elements-per-thread times the element width IS
  // the register count. (This is the line a sparse Op would have to opt out of,
  // its A operand being stored compressed.)
  const regs = (n, bits) =>
    ` = ${(n * bits) / 32} &times; 32-bit register${(n * bits) / 32 === 1 ? '' : 's'}`;
  const line = (name, L, rows, cols, what, bits) =>
    `<div class="cuo-result-line">tv_layout_${name} = ${L.str}</div>` +
    `<div class="cuo-result-line" style="color:#9ca3af">&nbsp;&nbsp;over the ${rows}&times;${cols} ` +
    `${what} tile &mdash; ${L.numVal} element${L.numVal === 1 ? '' : 's'}/lane` +
    `${regs(L.numVal, bits)}</div>`;

  document.getElementById(`${tabId}-mma-atom-result`).innerHTML =
    `<div class="cuo-result-line"><b>MMA_Atom&lt;${s.op.label.replace('warp.', '')}&gt;</b></div>` +
    `<div class="cuo-result-line">ThrID       = ${formatLayoutStr(s.thrId.shape, s.thrId.stride)}` +
    `<span style="color:#9ca3af"> &mdash; warp-collective; mma.sync.aligned needs all 32 lanes converged</span></div>` +
    `<div class="cuo-result-line">Shape MNK   = (${M}, ${N}, ${K})</div>` +
    line('A', s.A, M, K, 'A (M&times;K)', abBits) +
    line('B', s.B, N, K, 'B (N&times;K)', abBits) +
    line('C', s.C, M, N, 'C (M&times;N)', accBits) +
    `<div class="cuo-result-line" style="color:#9ca3af">All three are bijections ` +
    `(<code>size == cosize</code>) &mdash; every cell owned by exactly one lane, no broadcast.</div>`;
}

/** The quadrant-layout toggle. Only the CSS class and B's tile change; the
 *  layouts themselves are untouched, which is the point — B really is the same
 *  fragment, just drawn with its axes the other way round. */
function toggleMmaAltView(tabId) {
  const s = mmaState[tabId];
  if (!s) return;
  s.altView = !s.altView;
  mmaRenderViz(tabId);
}

/** The `value` toggle, shared by all three grids. */
function toggleMmaValue(tabId) {
  const s = mmaState[tabId];
  if (!s) return;
  s.showValue = !s.showValue;
  mmaRenderViz(tabId);
}

// Each operand is drawn with buildTVSVG — the same builder the TV Layout tab
// and make_copy_atom use, since the question is identical ("which (thread,
// value) slot owns this cell"). No `cellIndex` override is needed here: these
// tiles are COL-major in their own codomain, which is buildTVSVG's default.
// No `dimTids` either — an MMA atom has no broadcast (size == cosize).
function mmaRenderViz(tabId) {
  const s = mmaState[tabId];
  if (!s) return;
  const [M, N, K] = s.shapeMNK;
  const labelMode = s.showValue ? 'value' : '';
  const panes = [
    ['a', 'A', s.A, M, K, 'M&times;K', s.abDtype,
     'The A operand. Lane <code>t</code> holds rows {t/4, t/4+8} and a run of k decided by t%4.'],
    ['b', 'B', s.B, N, K, 'N&times;K', s.abDtype,
     'The B operand. Only one row per lane &mdash; B has no M extent, so the row group collapses.'],
    ['c', 'C', s.C, M, N, 'M&times;N', s.accDtype,
     'The accumulator. This is the layout your epilogue has to deal with, and the one ' +
     '<code>make_fragment_C</code> allocates.'],
  ];
  const host = document.getElementById(`${tabId}-mma-results`);
  if (host) host.classList.toggle('mma-alt', !!s.altView);
  const altBtn = document.getElementById(`${tabId}-mma-alt-btn`);
  if (altBtn) altBtn.classList.toggle('active', !!s.altView);

  for (const [side, name, L, rows, cols, dims, dtype, desc] of panes) {
    const btn = document.getElementById(`${tabId}-mma-${side}-val-btn`);
    if (btn) btn.classList.toggle('active', !!s.showValue);
    // In the quadrant view B is drawn K x N so its K axis meets A's and its N
    // axis meets C's. The LAYOUT is untouched: handing buildTVSVG the tile
    // (K, N):(N, 1) instead of (N, K):(1, N) re-reads the same offsets with the
    // axes swapped. The `value` overlay then needs its own cellIndex, since
    // buildTVSVG's default assumes the tile is col-major in its own codomain
    // and would print k + n*K where the layout's output is n + N*k.
    const rot = !!s.altView && side === 'b';
    const tileShape  = rot ? [cols, rows] : L.tile.shape;
    const tileStride = rot ? [rows, 1] : L.tile.stride;
    const opts = rot ? { cellIndex: (m, n) => n + rows * m } : undefined;

    document.getElementById(`${tabId}-mma-${side}-title`).textContent =
      `${name} — ${tileShape[0]}×${tileShape[1]} ${dtype} ` +
      `(${(rot ? 'K&times;N' : dims).replace('&times;', '×')}), ${L.numVal} per lane`;
    document.getElementById(`${tabId}-mma-${side}-desc`).innerHTML =
      `${desc}${rot ? ' Shown transposed, as K&times;N.' : ''} ` +
      `<code>tv_layout_${name} = ${L.str}</code>`;
    document.getElementById(`${tabId}-mma-${side}-svg`).innerHTML =
      buildTVSVG(L.shape, L.stride, tileShape, tileStride, false, 'col',
                 s.highlightTid === undefined ? null : s.highlightTid, labelMode, opts);
    applyZoomState(`${tabId}-mma-${side}-svg`);
  }
}

/** Re-render for the highlight field. Goes through the full render, exactly as
 *  setMtcHighlight does, so the out-of-range warning is recomputed with it. */
function setMmaHighlight(tabId) {
  renderMakeMmaAtom(tabId);
}

function setMMA(tabId, opKey, ab, acc, k) {
  document.getElementById(`${tabId}-mma-op-input`).value = opKey;
  mmaSyncControls(tabId, opKey);        // options BEFORE values, or the assign is a no-op
  if (ab)  document.getElementById(`${tabId}-mma-ab-input`).value  = ab;
  if (acc) document.getElementById(`${tabId}-mma-acc-input`).value = acc;
  document.getElementById(`${tabId}-mma-k-input`).value = String(k);
  renderMakeMmaAtom(tabId);
}

function exportMMA(tabId) {
  exportURL(`${tabId}-mma-export`, 'make_mma_atom',
    document.getElementById(`${tabId}-mma-op-input`).value,
    document.getElementById(`${tabId}-mma-ab-input`).value || 'na',
    document.getElementById(`${tabId}-mma-acc-input`).value || 'na',
    document.getElementById(`${tabId}-mma-k-input`).value);
}
