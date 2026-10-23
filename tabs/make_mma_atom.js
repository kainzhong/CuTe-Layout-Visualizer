// make_mma_atom tab (MMA scope): build ONE warp or warpgroup MMA Atom and visualize
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
// SCOPE: three dense warp Op families and three dense Hopper warpgroup
// families. The warp traits remain shared with warp-only downstream tabs.
// WGMMA uses separate traits and constructor controls, with SMEM operands
// displayed as uniform logical tiles instead of per-thread registers.
// Execution gate: tests/gpu_check.py --warpgroup (Hopper SM90a required).
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

// WGMMA traits from CuTeDSL 4.8 / mma_traits_sm90_gmma.hpp. SMEM TV
// layouts broadcast the whole logical operand across the thread mode; they
// describe a descriptor operand, not 128 independent register fragments.
const MMA_WARPGROUP_OPS = {
  wg_f16bf16: { label: 'warpgroup.MmaF16BF16Op', ab: ['half_t', 'bfloat16_t'], acc: ['half_t', 'float'], kDomain: [16] },
  wg_fp8: { label: 'warpgroup.MmaF8Op', ab: ['float_e4m3_t', 'float_e5m2_t'], acc: ['half_t', 'float'], kDomain: [32] },
  wg_int8: { label: 'warpgroup.MmaI8Op', ab: ['int8_t', 'uint8_t'], acc: ['int32_t'], kDomain: [32] },
};
const MMA_WARPGROUP_A_RMEM = {
  16: '((4,8,4),(2,2,2)):((128,1,16),(64,8,512))',
  32: '((4,8,4),(4,2,2)):((256,1,16),(64,8,1024))',
};

function mmaWarpgroupNs(opKey) {
  return Array.from({length: 32}, (_, i) => (i + 1) * 8)
    .filter(n => opKey !== 'wg_int8' || n === 8 || n === 24 || n % 16 === 0);
}

// CuTe shares the enum value MN; display the actual contiguous operand axis.
function mmaMajorLabel(operand, major) {
  return `${major === 'MN' ? (operand === 'A' ? 'M' : 'N') : 'K'}-major`;
}

function mmaWarpgroupAtom(opKey, ab, acc, k, n, aSrc, aMajor, bMajor, bDtype = ab) {
  const op = MMA_WARPGROUP_OPS[opKey];
  if (!op) throw new Error(`Unknown warpgroup MMA Op "${opKey}"`);
  if (!op.ab.includes(ab) || !op.ab.includes(bDtype) || !op.acc.includes(acc))
    throw new Error('Unsupported warpgroup MMA dtype.');
  if (opKey === 'wg_f16bf16' && (ab !== bDtype || (ab === 'bfloat16_t' && acc !== 'float')))
    throw new Error('F16/BF16 requires matching A/B types; bfloat16_t requires a float accumulator.');
  if (!op.kDomain.includes(k))
    throw new Error('Unsupported warpgroup MMA instruction shape.');
  if (!mmaWarpgroupNs(opKey).includes(n))
    throw new Error(opKey === 'wg_int8'
      ? 'Illegal N: INT8 WGMMA requires 8, 24, or a multiple of 16 from 16 through 256.'
      : 'Illegal N: WGMMA requires a multiple of 8 from 8 through 256.');
  if (!['SMEM', 'RMEM'].includes(aSrc) || !['K', 'MN'].includes(aMajor) || !['K', 'MN'].includes(bMajor))
    throw new Error('Unsupported warpgroup operand source or major mode.');
  // CuTeDSL 4.8 gemm verification rejects signed/unsigned register A as i8.
  if (opKey === 'wg_int8' && aSrc !== 'SMEM')
    throw new Error('INT8 WGMMA currently supports SMEM A only (CuTeDSL register signedness limitation).');
  if ((aSrc === 'RMEM' && aMajor !== 'K') || (opKey !== 'wg_f16bf16' && (aMajor !== 'K' || bMajor !== 'K')))
    throw new Error('This warpgroup MMA requires K-major operands (only SMEM F16/BF16 supports M-major A or N-major B).');
  const mk = (str, rows, cols, space, major) => {
    const p = parseLayout(str);
    return { ...p, str, tile: {shape: [rows, cols], stride: [1, rows]},
      numVal: product(p.shape[1]), space, major };
  };
  const c = n === 8 ? '((4,8,4),(2,2)):((128,1,16),(64,8))'
    : `((4,8,4),(2,2,${n / 8})):((128,1,16),(64,8,512))`;
  return { opKey, op, k, n, aSrc, aMajor, bMajor, bDtype, abDtype: ab, accDtype: acc,
    warpgroup: true, shapeMNK: [64, n, k], thrId: {shape: 128, stride: 1},
    A: mk(aSrc === 'RMEM' ? MMA_WARPGROUP_A_RMEM[k] : `(128,(64,${k})):(0,(1,64))`, 64, k, aSrc, aMajor),
    B: mk(`(128,(${n},${k})):(0,(1,${n}))`, n, k, 'SMEM', bMajor),
    C: mk(c, 64, n, 'RMEM', 'MN') };
}

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
            <span class="comp-viz-title"><span class="comp-viz-label" id="${id}-mma-${side}-title">${label}</span>${memorySpaceBadge(`${id}-mma-${side}-space`, 'RMEM')}</span>
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
          <summary>1. MMAAtom</summary>
          <div class="cuo-section-body">
            ${mmaAtomEditorMarkup({id,p:'mma',render:'renderMakeMmaAtom',setOp:'setMmaOp',warpgroup:true})}
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
            <button class="preset-btn" onclick="setMMA('${id}','f16bf16','half_t','float',16)">Warp MMA — m16n8k16 FP16 / FP32</button>
            <button class="preset-btn" onclick="setMMA('${id}','tf32',null,null,8)">Warp MMA — m16n8k8 TF32</button>
            <button class="preset-btn" onclick="setMMA('${id}','fp8','float_e4m3_t','float',32)">Warp MMA — m16n8k32 FP8</button>
            <button class="preset-btn" onclick="setMMA('${id}','wg_f16bf16','half_t','float',16,64,'SMEM','K','K')">WGMMA — m64n64k16 FP16, SMEM / SMEM</button>
            <button class="preset-btn" onclick="setMMA('${id}','wg_f16bf16','bfloat16_t','float',16,128,'SMEM','MN','MN')">WGMMA — m64n128k16 BF16, M-major A / N-major B</button>
            <button class="preset-btn" onclick="setMMA('${id}','wg_f16bf16','half_t','half_t',16,32,'RMEM','K','MN')">WGMMA — m64n32k16 FP16, register A</button>
            <button class="preset-btn" onclick="setMMA('${id}','wg_fp8','float_e4m3_t','float',32,64,'RMEM','K','K','float_e5m2_t')">WGMMA — m64n64k32 mixed FP8, register A</button>
            <button class="preset-btn" onclick="setMMA('${id}','wg_int8','int8_t','int32_t',32,24,'SMEM','K','K','uint8_t')">WGMMA — m64n24k32 signed / unsigned INT8</button>
          </div>
        </div>

        <div class="hint">
          <b>One MMA instruction.</b> Warp MMA uses 32 threads; Hopper WGMMA uses 128.
          Register operands show thread/value ownership. Shared-memory operands are drawn in
          one color: their TV layouts broadcast the operand across threads and describe SMEM
          descriptors rather than per-thread registers. Highlighting affects register operands only.<br><br>
          <b>Views.</b> The default view follows each operand's major mode. Alternative View
          always shows A as M×K, B as K×N, and C as M×N; it changes the display, not the TV layouts.
          SMEM cells are unlabeled by default; value shows logical offsets, not physical addresses or swizzle.<br><br>
          <b>Execution.</b> Warp MMA is synchronous. WGMMA is asynchronous and requires
          warpgroup-uniform issue plus fence, commit_group, and wait_group synchronization.<br><br>
          Sparse metadata, block-scaled operands and Blackwell tcgen05 remain outside this tab.
          TiledMMA supports both scopes; partition and fragment tabs currently accept warp atoms only.
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
    mmaSyncControls(tabId, opKey);
    const inputs = mmaReadAtomInputs(tabId);
    const {atom: a, op} = mmaDecodeAtomInputs(inputs);
    const k = a.k, abDtype = inputs[1] === 'na' ? 'tfloat32_t' : inputs[1];
    const accDtype = inputs[2] === 'na' ? 'float' : inputs[2];
    const prev = mmaState[tabId] || {};
    // Same control, same convention as the tv / make_tiled_copy /
    // make_tiled_copy_tv tabs: live on `oninput`, an out-of-range id WARNS and
    // shows everything rather than being ignored, and buildTVSVG does the
    // dimming so a dimmed cell keeps its T/V labels instead of going blank.
    const hl = readHighlightTid(tabId, 'mma', product(a.thrId.shape));
    showWarn(`${tabId}-mma-warning`, hl.warn);
    mmaState[tabId] = { ...a, abDtype, accDtype, showValue: !!prev.showValue,
                        altView: !!prev.altView, highlightTid: hl.tid };

    mmaRenderOpParams(tabId, op, k, abDtype, accDtype, 'mma', a);
    mmaRenderResult(tabId);
    mmaRenderViz(tabId);
    updateOuterTabLabel(tabId, `make_mma_atom:${op.label}/${a.shapeMNK.join('x')}`);
  } catch (e) {
    mmaState[tabId] = null;
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
  const wg = ['mma', 'mtm'].includes(p) && MMA_WARPGROUP_OPS[opKey];
  const op = wg || MMA_WARP_OPS[opKey] || MMA_WARP_OPS.f16bf16;
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
       v => wg ? `K = ${v}` : `(16, 8, ${v})   —   m16n8k${v}`);
  if (['mma', 'mtm'].includes(p)) {
    document.getElementById(`${tabId}-${p}-ab-label`).textContent = wg && opKey !== 'wg_f16bf16' ? 'a_dtype' : 'ab_dtype';
    document.getElementById(`${tabId}-${p}-shape-label`).textContent = wg ? 'instruction_shape (K; M = 64, N below)' : 'shape_mnk';
    fill(document.getElementById(`${tabId}-${p}-b-input`), op.ab || []);
    fill(document.getElementById(`${tabId}-${p}-asrc-input`), opKey === 'wg_int8' ? ['SMEM'] : ['SMEM', 'RMEM']);
    const source = document.getElementById(`${tabId}-${p}-asrc-input`).value;
    fill(document.getElementById(`${tabId}-${p}-amajor-input`), wg && opKey === 'wg_f16bf16' && source === 'SMEM' ? ['K', 'MN'] : ['K'], v => mmaMajorLabel('A', v));
    fill(document.getElementById(`${tabId}-${p}-bmajor-input`), wg && opKey === 'wg_f16bf16' ? ['K', 'MN'] : ['K'], v => mmaMajorLabel('B', v));
    for (const key of ['n','asrc','amajor','bmajor']) show(`${tabId}-${p}-${key}-group`, !!wg);
    show(`${tabId}-${p}-b-group`, !!wg && opKey !== 'wg_f16bf16');
  }
  show(`${tabId}-${p}-ab-group`, !!op.ab);
  show(`${tabId}-${p}-acc-group`, !!op.acc);
}

function setMmaOp(tabId) {
  mmaSyncControls(tabId, document.getElementById(`${tabId}-mma-op-input`).value);
  renderMakeMmaAtom(tabId);
}

function mmaRenderOpParams(tabId, op, k, abDtype, accDtype, p, atom) {
  const host = document.getElementById(`${tabId}-${p || 'mma'}-op-params`);
  if (!host) return;
  if (atom && atom.warpgroup) {
    const args = atom.opKey === 'wg_f16bf16' ? [abDtype, accDtype] : [abDtype, atom.bDtype, accDtype];
    host.textContent = `${op.label}(${args.join(', ')}, (${atom.shapeMNK.join(', ')}), ${atom.aSrc}, ${mmaMajorLabel('A', atom.aMajor)}, ${mmaMajorLabel('B', atom.bMajor)})`;
    return;
  }
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
  const line = (name, L, rows, cols, what, bits) => L.space === 'SMEM'
    ? `<div class="cuo-result-line">tv_layout_${name} = ${L.str}</div>` +
      `<div class="cuo-result-line">${rows}×${cols} SMEM operand; descriptor, ${mmaMajorLabel(name, L.major)}; broadcast across threads.</div>`
    : `<div class="cuo-result-line">tv_layout_${name} = ${L.str}</div>` +
    `<div class="cuo-result-line" style="color:#9ca3af">&nbsp;&nbsp;over the ${rows}&times;${cols} ` +
    `${what} tile &mdash; ${L.numVal} element${L.numVal === 1 ? '' : 's'}/${s.warpgroup ? 'thread' : 'lane'}` +
    `${regs(L.numVal, bits)}</div>`;

  document.getElementById(`${tabId}-mma-atom-result`).innerHTML =
    `<div class="cuo-result-line"><b>MMA_Atom&lt;${s.op.label.replace('warp.', '')}&gt;</b></div>` +
    `<div class="cuo-result-line">ThrID       = ${formatLayoutStr(s.thrId.shape, s.thrId.stride)}` +
    `<span style="color:#9ca3af"> &mdash; ${s.warpgroup ? 'warpgroup-collective; wgmma.mma_async needs all 128 threads' : 'warp-collective; mma.sync.aligned needs all 32 lanes converged'}</span></div>` +
    `<div class="cuo-result-line">Shape MNK   = (${M}, ${N}, ${K})</div>` +
    line('A', s.A, M, K, 'A (M&times;K)', abBits) +
    line('B', s.B, N, K, 'B (N&times;K)', DTYPE_BITS[s.bDtype] || abBits) +
    line('C', s.C, M, N, 'C (M&times;N)', accBits) +
    `<div class="cuo-result-line" style="color:#9ca3af">${s.warpgroup
      ? 'RMEM operands show register ownership; SMEM operands are descriptor broadcasts.'
      : 'All three are bijections (size == cosize): every cell owned by exactly one lane.'}</div>`;
}

/** Matrix view fixes A/B/C to MK/KN/MN regardless of operand major modes.
 * The operand TV layouts are untouched. */
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

// RMEM uses the shared TV renderer; descriptor operands use the shared
// colored-layout renderer with one color and optional logical offsets.
function mmaRenderViz(tabId) {
  const s = mmaState[tabId];
  if (!s) return;
  const [M, N, K] = s.shapeMNK;
  const labelMode = s.showValue ? 'value' : '';
  const panes = [
    ['a', 'A', s.A, M, K, 'M&times;K', s.abDtype,
     'The A operand. Lane <code>t</code> holds rows {t/4, t/4+8} and a run of k decided by t%4.'],
    ['b', 'B', s.B, N, K, 'N&times;K', s.bDtype || s.abDtype,
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
    const rot = s.altView ? side === 'b' : (s.warpgroup && side !== 'c' && L.major === 'MN');
    const tileShape  = rot ? [cols, rows] : L.tile.shape;
    const displayDims = side === 'a' ? (rot ? 'K×M' : 'M×K') : side === 'b' ? (rot ? 'K×N' : 'N×K') : 'M×N';
    const space = L.space || 'RMEM';
    setMemorySpaceBadge(`${tabId}-mma-${side}-space`, space);
    const tileStride = rot ? [rows, 1] : L.tile.stride;
    const opts = rot ? { cellIndex: (m, n) => n + rows * m } : undefined;

    document.getElementById(`${tabId}-mma-${side}-title`).textContent =
      `${name} — ${tileShape[0]}×${tileShape[1]} ${dtype} ` +
      `(${displayDims}), ${space === 'SMEM' ? 'descriptor' : `${L.numVal} per ${s.warpgroup ? 'thread' : 'lane'}`}`;
    document.getElementById(`${tabId}-mma-${side}-desc`).innerHTML =
      `${s.warpgroup ? (space === 'SMEM' ? `Shared-memory operand, ${mmaMajorLabel(name, L.major)}. Logical tile; no per-thread register ownership.` : 'Register fragment distributed across 128 threads.') : desc}${rot ? ' Shown transposed.' : ''} ` +
      `<code>tv_layout_${name} = ${L.str}</code>`;
    document.getElementById(`${tabId}-mma-${side}-svg`).innerHTML =
      space === 'SMEM'
      ? buildColoredLayoutSVG(tileShape, tileStride, [], (m, n, offset) => ({bg: '#a7f3d0', text: s.showValue ? [String(offset)] : null}))
      : buildTVSVG(L.shape, L.stride, tileShape, tileStride, false, 'col',
                 s.highlightTid === undefined ? null : s.highlightTid, labelMode, opts);
    applyZoomState(`${tabId}-mma-${side}-svg`);
  }
}

/** Re-render for the highlight field. Goes through the full render, exactly as
 *  setMtcHighlight does, so the out-of-range warning is recomputed with it. */
function setMmaHighlight(tabId) {
  renderMakeMmaAtom(tabId);
}

function setMMA(tabId, opKey, ab, acc, k, n=64, aSrc='SMEM', aMajor='K', bMajor='K', bDtype=ab) {
  const inputs = [opKey, ab || 'na', acc || 'na', String(k)];
  if (MMA_WARPGROUP_OPS[opKey]) inputs.push(String(n), aSrc, aMajor, bMajor, bDtype);
  mmaAssignAtomInputs(tabId, 'mma', inputs);
  renderMakeMmaAtom(tabId);
}

function mmaReadAtomInputs(tabId, p = 'mma') {
  const read = key => document.getElementById(`${tabId}-${p}-${key}-input`).value;
  const op = read('op');
  const inputs = [op, op === 'tf32' ? 'na' : read('ab') || 'na', op === 'tf32' ? 'na' : read('acc') || 'na', read('k')];
  if (MMA_WARPGROUP_OPS[op]) inputs.push(read('n'), read('asrc'), read('amajor'), read('bmajor'),
    op === 'wg_f16bf16' ? read('ab') : read('b'));
  return inputs;
}

function exportMMA(tabId) {
  exportURL(`${tabId}-mma-export`, 'make_mma_atom', ...mmaReadAtomInputs(tabId));
}

function mmaDecodeAtomInputs(inputs) {
  const [opKey, ab, acc, k, n, aSrc, aMajor, bMajor, bDtype] = inputs;
  if (MMA_WARPGROUP_OPS[opKey]) {
    if (inputs.length !== 9) throw new Error('Warpgroup MMAAtom requires N, operand source, major modes and B dtype.');
    return {
      atom: mmaWarpgroupAtom(opKey, ab, acc, Number(k), Number(n), aSrc, aMajor, bMajor, bDtype),
      op: MMA_WARPGROUP_OPS[opKey]
    };
  }
  if (inputs.length !== 4) throw new Error('Unexpected warp MMAAtom parameters.');
  const atom = mmaWarpAtom(opKey, Number(k)), op = MMA_WARP_OPS[opKey];
  if (op.ab && (!op.ab.includes(ab) || !op.acc.includes(acc))) throw new Error('Unsupported MMAAtom dtype.');
  if (op.ab && ab === 'bfloat16_t' && acc !== 'float') throw new Error('bfloat16_t requires a float accumulator.');
  return {atom, op};
}

function mmaAssignAtomInputs(tabId, p, inputs) {
  const {atom} = mmaDecodeAtomInputs(inputs);
  if (!['mma', 'mtm'].includes(p) && atom.warpgroup) throw new Error('This consumer supports warp MMA atoms only.');
  const [op,ab,acc,k,n,aSrc,aMajor,bMajor,bDtype] = inputs;
  const write = (key,value) => { document.getElementById(`${tabId}-${p}-${key}-input`).value = String(value); };
  write('op',op);
  mmaSyncControls(tabId,op,p);
  write('ab',ab === 'na' ? '' : ab);
  write('acc',acc === 'na' ? '' : acc);
  write('k',k);
  if (atom.warpgroup) {
    write('n',n); write('asrc',aSrc);
    mmaSyncControls(tabId,op,p); // Rebuild major modes after the source changes.
    write('amajor',aMajor); write('bmajor',bMajor); write('b',bDtype || ab);
  }
}

function mmaAtomEditorMarkup({id,p,render,setOp,k=null,errorPrefix=p,warpgroup=false}) {
  return `<button class="btn btn-render btn-decode" id="${id}-${p}-decode-atom" onclick="mmaDecodeAtomPasteboard('${id}','${p}','${render}','${errorPrefix}')">Decode from Pasteboard</button>
    <details class="cuo-section" id="${id}-${p}-atom-manual-construction"><summary>Manual construction</summary><div class="cuo-section-body">
      <div class="form-group"><label>Op</label>
        <select id="${id}-${p}-op-input" onchange="${setOp || render}('${id}')">
          ${Object.entries(warpgroup ? {...MMA_WARP_OPS, ...MMA_WARPGROUP_OPS} : MMA_WARP_OPS).map(([key,op]) => `<option value="${key}">${op.label}</option>`).join('')}
        </select>
      </div>
      <div class="form-group" id="${id}-${p}-ab-group"><label id="${id}-${p}-ab-label">ab_dtype</label><select id="${id}-${p}-ab-input" onchange="${render}('${id}')"></select></div>
      <div class="form-group" id="${id}-${p}-acc-group"><label>acc_dtype</label><select id="${id}-${p}-acc-input" onchange="${render}('${id}')"></select></div>
      <div class="form-group"><label id="${id}-${p}-shape-label">shape_mnk</label><select id="${id}-${p}-k-input" onchange="${render}('${id}')">${k === null ? '' : `<option value="${k}">(16, 8, ${k})</option>`}</select></div>
      ${warpgroup ? `<div class="form-group" id="${id}-${p}-n-group"><label>N</label><input type="number" id="${id}-${p}-n-input" value="64" min="8" max="256" step="8" onchange="${render}('${id}')"></div>` : ''}
      ${warpgroup ? [
        ['b','b_dtype',[]], ['asrc','A source',['SMEM','RMEM']],
        ['amajor','A major mode',['K','MN']], ['bmajor','B major mode',['K','MN']]
      ].map(([key,label,values]) => `<div class="form-group" id="${id}-${p}-${key}-group"><label>${label}</label><select id="${id}-${p}-${key}-input" onchange="${render}('${id}')">${values.map(v => `<option value="${v}">${v}</option>`).join('')}</select></div>`).join('') : ''}
    </div></details>`;
}

function mmaImportAtom(tabId, p, render, text) {
  const parsed = parsePastedKeyParam(text);
  if (!parsed || parsed.feature !== 'make_mma_atom') throw new Error('Paste a make_mma_atom URL or its query string.');
  if (p === 'mtm') mtmDecodeConstructor([
    ...parsed.inputs.slice(0,4),
    document.getElementById(`${tabId}-mtm-atomlayout-input`).value,
    document.getElementById(`${tabId}-mtm-perm-input`).value || 'na',
    ...parsed.inputs.slice(4)
  ], {warpgroup: true});
  mmaAssignAtomInputs(tabId, p, parsed.inputs);
  globalThis[render](tabId);
}

async function mmaDecodeAtomPasteboard(tabId,p,render,errorPrefix=p) {
  const text = await readPasteboardText('Paste the make_mma_atom URL or its query string:');
  if (text === null || text === undefined) return;
  try { mmaImportAtom(tabId,p,render,text); }
  catch (e) { showErr(`${tabId}-${errorPrefix}-error`,e.message); }
}
