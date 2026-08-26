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
// no fields; cpasync.CopyG2SOp carries only a defaulted `cache_mode`. Ops that
// DO need constructor params (warp.LdMatrix* takes num_matrices/transpose,
// tcgen05.Ld* takes repeat/pack) slot into the same params area.
//
// Copy_Atom<UniversalCopy<S>, ValType> (include/cute/atom/copy_traits.hpp:65-78):
//   ThrID        = Layout<_1>
//   ValLayoutSrc = ValLayoutDst = (1, N):(0, 1) with N = num_bits / sizeof_bits(ValType)
//
// Replicating this atom over a thread tile is a separate concern — see the
// make_tiled_copy / make_tiled_copy_tv tabs, alongside this one in Copy.

// Copy Ops this tab can build. `params` lists the Op's own constructor
// parameters — empty for both of these, which is why the params area just says
// so rather than rendering controls.
const MCA_OPS = {
  universal: {
    label: 'CopyUniversalOp',
    ctor: 'cute.nvgpu.CopyUniversalOp()',
    cpasync: false,
    params: [],
    note: 'CuTe\'s generic single-thread load/store. Takes no constructor parameters — ' +
          'the dataclass has no fields at all.',
  },
  cpasync: {
    label: 'cpasync.CopyG2SOp',
    ctor: 'cute.nvgpu.cpasync.CopyG2SOp()',
    cpasync: true,
    params: [],
    note: 'SM80+ non-bulk cp.async, GMEM→SMEM. Its one field, cache_mode, ' +
          'defaults to ALWAYS and does not change the Atom\'s layouts.',
  },
};

// cp.async hardware widths (SM80+ non-bulk cp.async): 4, 8 or 16 bytes.
const MCA_CPASYNC_BITS = [32, 64, 128];

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
              </select>
            </div>
            <div id="${id}-mca-op-params" class="cuo-result"></div>
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>2. make_copy_atom(op, dtype, ...)</summary>
          <div class="cuo-section-body">
            <div class="form-group">
              <label>tensor_dtype</label>
              <select id="${id}-mca-dtype-input">${dtypeOptions('half_t')}</select>
            </div>
            <div class="form-group">
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
          </div>
        </div>

        <div class="hint">
          <b>Two steps, two kinds of parameter.</b> An Op is constructed first
          (with whatever fields it carries), then
          <code>make_copy_atom(op, dtype, num_bits_per_copy=N)</code> turns it
          into an Atom. Both Ops offered here take <em>no</em> constructor
          parameters, so section 1 only asks which one. Ops that do take them
          &mdash; <code>warp.LdMatrix8x8x16bOp(num_matrices, transpose)</code>,
          <code>tcgen05.Ld16x64bOp(repeat, pack)</code> &mdash; would fill in
          that area.<br><br>
          <b>Why these two share a visualization.</b>
          <code>CopyUniversalOp</code> and <code>cpasync.CopyG2SOp</code> have
          <em>byte-for-byte identical</em> <code>Copy_Traits</code>:
          <code>ThrID = Layout&lt;_1&gt;</code>,
          <code>SrcLayout = DstLayout = Layout&lt;Shape&lt;_1, num_bits&gt;&gt;</code>
          (<code>include/cute/atom/copy_traits.hpp:65-78</code> and
          <code>copy_traits_sm80.hpp:41-54</code>). The user-visible contract
          difference is that <code>cp.async</code> hardware accepts only
          <code>num_bits_per_copy &isin; {32, 64, 128}</code>, and that the DSL
          <em>requires</em> the argument for <code>CopyG2SOp</code> while
          <code>CopyUniversalOp</code> lets you omit it and auto-vectorize.<br><br>
          <b>What an Atom is.</b> One instruction's worth of copy: how many
          threads participate (<code>ThrID</code>) and which values each one
          moves. For these two that's one thread moving
          <code>N = num_bits / sizeof_bits(tensor_dtype)</code> <em>contiguous</em>
          elements. Nothing about tiles, thread blocks or tensors is decided
          here.<br><br>
          <b>Next step.</b> To replicate this atom across a thread tile, use
          <b>make_tiled_copy</b> or <b>make_tiled_copy_tv</b>. To reason about
          how the resulting access pattern hits real memory, take the TV layout
          over to the <b>TV Layout</b> tab.
        </div>
      </div>

      <div class="comp-results" style="grid-template-columns:1fr">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-mca-atom-title">Copy_Atom &mdash; single-atom data</span>
            <span style="display:flex;align-items:center;gap:4px">
              ${copyDirButtons(id, 'mca')}
              <button class="mode-btn" id="${id}-mca-src-svg-zoom" onclick="toggleCopyZoom('${id}','mca')">Zoom in</button>
            </span>
          </div>
          <div class="cuo-viz-desc">
            One atom invocation: the values a single instruction moves, in order.
            <code>V<i>k</i></code> is the value index within the atom, and it is
            the only thing a cell can say &mdash; <code>ThrID = 1:0</code> means
            one thread issues the whole instruction, so there is no thread
            assignment to draw. Threads enter the picture only when the atom is
            replicated over a tile; that is <b>make_tiled_copy</b>'s job, and its
            grid is the one that carries <code>T</code> labels. For these two Ops
            <code>ValLayoutSrc == ValLayoutDst</code>, so the two panes are
            identical &mdash; they diverge for atoms that shuffle, such as
            <code>ldmatrix</code>.
          </div>
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

    const bitsStr = document.getElementById(`${tabId}-mca-bits-input`).value;
    const dtype   = document.getElementById(`${tabId}-mca-dtype-input`).value;
    const numBits = parseInt(bitsStr, 10);
    if (!Number.isFinite(numBits) || numBits <= 0) {
      throw new Error(`num_bits_per_copy must be a positive integer, got "${bitsStr}"`);
    }
    const elemBits = DTYPE_BITS[dtype];
    if (!elemBits) throw new Error(`Unknown tensor_dtype "${dtype}"`);
    if (numBits % elemBits !== 0) {
      throw new Error(
        `num_bits_per_copy (${numBits}) must be a multiple of sizeof_bits(${dtype}) = ${elemBits}`);
    }
    const elements = numBits / elemBits;

    // Per-thread value layout of ONE atom. Copy_Traits<UniversalCopy<S,D>>
    // declares `SrcLayout = Layout<Shape<_1, sizeof_bits<S>>>` with no explicit
    // stride, which compacts to col-major stride `(_1, _1)`. After
    // `recast_layout<uint1_t, ValType>` (upcast by sizeof_bits(ValType)), mode 0
    // stays `(1):(1)` and mode 1 becomes `(elements):(1)`. Net effect: the
    // canonical per-value atom layout is `(1, elements):(1, 1)` — which is what
    // we produce by letting Layout's constructor auto-fill col-major strides.
    const atomSrc   = new Layout([1, elements]);
    const atomThrID = new Layout(1, 0);        // Layout<_1>; CuTeDSL prints this as 1:0
    const atomStr   = formatLayoutStr(atomSrc.shape, atomSrc.stride);

    if (op.cpasync && !MCA_CPASYNC_BITS.includes(numBits)) {
      showWarn(`${tabId}-mca-warning`,
        `cp.async hardware only accepts num_bits_per_copy ∈ {32, 64, 128} — ` +
        `${numBits} would be rejected by cpasync.CopyG2SOp (CopyUniversalOp would take it).`);
    }

    const prev = mcaState[tabId] || {};
    mcaState[tabId] = {
      opKey, op, numBits, dtype, elements, elemBits, atomSrc, atomThrID, atomStr,
      atomDstStr: atomStr,   // ValLayoutDst == ValLayoutSrc for both Ops here
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

    renderMcaAtomViz(tabId);
    updateOuterTabLabel(tabId, `make_copy_atom:${numBits}b/${dtype}`);
  } catch (e) {
    showErr(`${tabId}-mca-error`, e.message);
    for (const side of ['src', 'dst']) {
      const el = document.getElementById(`${tabId}-mca-${side}-svg`);
      if (el) el.innerHTML = '';
    }
  }
}

// Section 1's body. Both Ops we offer are parameterless, so this states that
// rather than rendering an empty box — the area exists so an Op that DOES take
// constructor params (ldmatrix, tcgen05) has somewhere obvious to put them.
function mcaRenderOpParams(tabId, op) {
  const host = document.getElementById(`${tabId}-mca-op-params`);
  if (!host) return;
  host.innerHTML =
    `<div class="cuo-result-line"><b>${op.ctor}</b></div>` +
    (op.params.length === 0
      ? `<div class="cuo-result-line" style="color:#9ca3af">No constructor parameters required. ` +
        `${op.note}</div>`
      : op.params.map(p => `<div class="cuo-result-line">${p}</div>`).join(''));
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

function renderMcaAtomViz(tabId) {
  const s = mcaState[tabId];
  if (!s) return;
  const p = parseLayout(s.atomStr);
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


function setMCA(tabId, opKey, bits, dtype) {
  document.getElementById(`${tabId}-mca-op-input`).value    = opKey;
  document.getElementById(`${tabId}-mca-bits-input`).value  = bits;
  document.getElementById(`${tabId}-mca-dtype-input`).value = dtype;
  renderMakeCopyAtom(tabId);
}

function exportMCA(tabId) {
  exportURL(`${tabId}-mca-export`, 'make_copy_atom',
    document.getElementById(`${tabId}-mca-op-input`).value,
    document.getElementById(`${tabId}-mca-bits-input`).value,
    document.getElementById(`${tabId}-mca-dtype-input`).value);
}
