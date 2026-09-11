// partition_fragment_A / B / C tab (MMA scope): the registers a thread declares.
//
//   Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));   // (MMA,MMA_M,MMA_K)
//
// That is the call a kernel writes — `sgemm_sm80.cu:160` — and it is defined as
//
//   partition_fragment_A(t) = make_fragment_A(partition_A(t))   mma_atom.hpp:508
//
// so this tab is the **partition_A / B / C** tab plus one step. It takes the
// same inputs (`pabcInputSections` / `pabcReadInputs`) and runs the same
// derivation (`pabcComputePartition`); what it draws is the other half.
// CuTeDSL has no wrapper — there you compose the two calls, which is also how
// tests/gen_reference.py produces this tab's oracle.
//
// ─── Why the fragment is not just "compact, same shape" ─────────────────────
//
// `make_fragment_A|B` (mma_atom.hpp:148) forwards to `make_fragment_like`
// (layout.hpp:455), which orders the trailing modes by the PARTITION's strides.
// The C++ says why in a comment on the function group:
//
//   "we can inspect the layout of the partitioned data and attempt to match it
//    in generated fragment to promote vectorization when copying from partition
//    to fragment"
//
// So a linear walk of the registers is a monotone walk of the source, and the
// copy into them vectorizes. Flip the source between row- and column-major and
// the fragment's mode order flips with it — that is grid 2's whole subject, and
// there is a preset for each.
//
// `make_fragment_C` is the exception: `make_tensor<FrgTypeC>(shape(ctensor))`,
// plain compact, because an accumulator is never read in the order it was
// partitioned. CUTLASS states that twice — the second time by providing a
// STATIC `partition_fragment_C(mma, shapeMN)` that needs only a shape, while
// A and B carry "should not be used in a static context" (:557-585).

const PFAB_MAX_STRIP = 512;      // registers drawn in the load-order strip

/** Registers in index order, each with the source offset it loads from.
 *  The fragment is compact and bijective, so inverting it is a scatter. */
function pfabRegisterOrder(fragment, partition, restExtents) {
  const V = product(fragment.shape[0]);
  const n = product(fragment.shape);
  const src = new Array(n).fill(null);
  const R = restExtents.reduce((a, b) => a * b, 1);
  for (let v = 0; v < V; v++) {
    for (let r = 0; r < R; r++) {
      const rc = psdRestCoord(r, restExtents);
      src[fragment.call(v, ...rc)] = partition.call(v, ...rc);
    }
  }
  // Maximal stretches where consecutive registers hold consecutive source
  // elements — the runs a vector load can cover.
  const runOf = new Array(n).fill(0);
  let runs = [], cur = 1;
  for (let i = 1; i <= n; i++) {
    if (i < n && src[i] - src[i - 1] === 1) { cur++; continue; }
    runs.push(cur); cur = 1;
  }
  let at = 0;
  runs.forEach((len, k) => { for (let i = 0; i < len; i++) runOf[at++] = k; });
  return { src, runOf, runs, maxRun: Math.max(...runs) };
}

function generatePartitionFragmentABCTabContent(id) {
  return `
    <!-- partition_fragment_A / B / C panel -->
    <div id="${id}-tab-partition_fragment_abc" class="panel">
      <div class="controls">
        <h2>partition_fragment_A / B / C</h2>

${pabcInputSections({ id, p: 'pfab', fn: 'partition_fragment', render: 'renderPartitionFragmentABC', setOp: 'setPfabOp', setOperand: 'setPfabOperand' })}

        ${statusDivs(`${id}-pfab`)}
        <button class="btn btn-render" onclick="renderPartitionFragmentABC('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-pfab-export" onclick="exportPFAB('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setPFAB('${id}','A','f16bf16','half_t','float',16,'(2, 2, 1)','','5','(64, 48):(48, 1)')">A from a <b>row-major</b> source</button>
            <button class="preset-btn" onclick="setPFAB('${id}','A','f16bf16','half_t','float',16,'(2, 2, 1)','','5','(64, 48):(1, 64)')">A from a <b>column-major</b> source &mdash; the mode order flips</button>
            <button class="preset-btn" onclick="setPFAB('${id}','B','f16bf16','half_t','float',16,'(2, 2, 1)','','5','(32, 48):(48, 1)')">B &mdash; (N, K)</button>
            <button class="preset-btn" onclick="setPFAB('${id}','C','f16bf16','half_t','float',16,'(2, 2, 1)','','5','(64, 32):(32, 1)')">C &mdash; the accumulator, always plain compact</button>
            <button class="preset-btn" onclick="setPFAB('${id}','C','f16bf16','half_t','float',16,'(2, 2, 1)','','5','(64, 32):(1, 64)')">C from a column-major source &mdash; identical, the order is ignored</button>
            <button class="preset-btn" onclick="setPFAB('${id}','A','f16bf16','half_t','float',16,'(2, 2, 1)','','5','(64, 48, 4):(48, 1, 3072)')">A with a k-loop &mdash; 4&times; the registers</button>
            <button class="preset-btn" onclick="setPFAB('${id}','A','f16bf16','half_t','float',16,'(2, 2, 1)','','5','(32, 16):(16, 1)')">Tensor == one tile &mdash; Rest 1,1 comes back stride <b>0</b></button>
            <button class="preset-btn" onclick="setPFAB('${id}','A','tf32','na','na',8,'(2, 2, 1)','','0','(64, 32):(32, 1)')">MmaTF32Op, m16n8k8</button>
          </div>
        </div>

        <div class="hint">
          <b>This is <code>partition_X</code> plus one step.</b>
          <code>thr_mma.partition_fragment_A(sA)</code> is
          <code>make_fragment_A(partition_A(sA))</code>
          (<code>mma_atom.hpp:508</code>), and it is the form a kernel writes.
          Use the <b>partition_A / B / C</b> tab for the first half &mdash; how
          the tile covers the tensor; this one is about what the thread then
          declares in registers. (CuTeDSL has no wrapper: you compose the two.)
          <br><br>
          <b>The fragment has the partition's shape and compact strides</b>, so
          the register count is fixed the moment you pick the TiledMMA and the
          tensor. What is <em>not</em> fixed is the ORDER, and that is the whole
          reason <code>make_fragment_A</code> takes an already-partitioned tensor
          rather than a shape. The C++ says so directly:<br>
          <i>"we can inspect the layout of the partitioned data and attempt to
          match it in generated fragment to promote vectorization when copying
          from partition to fragment."</i><br>
          Mode 0 &mdash; the atom's own value mode &mdash; is forced compact
          col-major whatever the partition did. The trailing modes get compact
          strides in the order the <em>partition's</em> strides rank them, so a
          linear walk of the registers is a monotone walk of the source. Switch
          the A presets between <code>(48, 1)</code> and <code>(1, 64)</code> and
          the strides go from <code>24, 8</code> to <code>8, 16</code>.<br><br>
          <b><code>make_fragment_C</code> ignores all of that</b> &mdash; it is
          <code>make_tensor&lt;FrgTypeC&gt;(shape(ctensor))</code>, plain compact,
          because an accumulator is never read in the order it was partitioned.
          CUTLASS states this a second time by providing a <em>static</em>
          <code>partition_fragment_C(mma, shapeMN)</code> that needs only a
          shape, while A and B carry a comment that they "often depend on the
          layout of A and B and/or the thread_idx" and "should not be used in a
          static context" (<code>mma_atom.hpp:557-585</code>). The two C presets
          above give the same fragment from opposite-majorness sources.<br><br>
          <b>A size-1 mode's compact stride is 0</b>, not the running product, so
          a tensor exactly one tile wide gives
          <code>((2,2,2),1,1):((1,2,4),0,0)</code>. Same fact as
          <code>make_tiled_mma</code>'s <code>thr_layout_vmnk</code>.
        </div>
      </div>

      <div class="comp-results" style="grid-template-columns:1fr">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-pfab-frag-title">1. The register map</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-pfab-frag-mode-btns">
                <button class="mode-btn" onclick="setPfabMode('${id}','frag','value')">value</button>
              </span>
              <button class="mode-btn" id="${id}-pfab-frag-svg-zoom" onclick="toggleZoom('${id}-pfab-frag-svg')">Zoom in</button>
              <button class="mode-btn" onclick="downloadSVG('${id}-pfab-frag-svg', 'partition_fragment_map.svg')">Download SVG</button>
            </span>
          </div>
          <div class="cuo-viz-desc">
            The fragment over the partition's own domain: one <b>row</b> per slot
            of the atom's value mode, one <b>column</b> per Rest position. Each
            cell is the <b>register index</b> that slot lands in;
            <code>value</code> adds the offset in the source it is loaded from.
            Column hues match the Rest grids on the <b>partition_A / B / C</b>
            tab.
          </div>
          <div class="viz-box"><div id="${id}-pfab-frag-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-pfab-strip-title">2. The load order</span>
            <span style="display:flex;align-items:center;gap:4px">
              <button class="mode-btn" id="${id}-pfab-strip-svg-zoom" onclick="toggleZoom('${id}-pfab-strip-svg')">Zoom in</button>
              <button class="mode-btn" onclick="downloadSVG('${id}-pfab-strip-svg', 'partition_fragment_order.svg')">Download SVG</button>
            </span>
          </div>
          <div class="cuo-viz-desc">
            The same registers laid out in <b>register order</b>,
            <code>r0</code> to <code>rN-1</code>, each labelled with the source
            offset it loads from. A block of one colour is a run of registers
            holding <em>consecutive</em> source elements &mdash; the stretch a
            vector load can cover. This is what
            <code>make_fragment_A</code>'s ordering is optimising, so it is where
            the row-major and column-major presets visibly differ.
          </div>
          <div class="viz-box"><div id="${id}-pfab-strip-svg"></div></div>
        </div>
      </div>
    </div>`;
}

const pfabState = {};

function renderPartitionFragmentABC(tabId) {
  showErr(`${tabId}-pfab-error`, '');
  showWarn(`${tabId}-pfab-warning`, '');
  const prev = pfabState[tabId] || {};
  const which = prev.which || 'A';
  pfabSyncOperandField(tabId, which);
  try {
    const inp = pabcReadInputs(tabId, 'pfab');
    const r = pabcComputePartition(inp.atom, inp.atomLayout, inp.perm, which, inp.tensor, inp.thrIdx);

    const restExtents = [...r.sweepExtents, ...r.extraExtents];
    const order = pfabRegisterOrder(r.fragment, r.partition, restExtents);

    document.getElementById(`${tabId}-pfab-tiled-result`).innerHTML =
      `<div class="cuo-result-line">thr_layout_vmnk = <b>` +
      `${formatLayoutStr(r.thrLayoutVmnk.shape, r.thrLayoutVmnk.stride)}</b> &mdash; ` +
      `${r.warps} warp${r.warps === 1 ? '' : 's'}, ${r.threads} threads</div>` +
      `<div class="cuo-result-line">Tile Shape MNK = <b>(${r.tileMNK.join(', ')})</b>` +
      `<span style="color:#9ca3af"> &rarr; ${which} divides by ` +
      `${r.divExtents.join('&times;')} (${r.spec.axes.join(', ')})</span></div>`;

    pfabState[tabId] = {
      ...r, ...inp, which, order, restExtents,
      dtype: which === 'C' ? inp.accDtype : inp.abDtype,
      fragMode: (prev.fragMode instanceof Set) ? prev.fragMode : new Set(),
    };
    pfabRenderMapViz(tabId);
    pfabRenderStripViz(tabId);
    updateOuterTabLabel(tabId,
      `partition_fragment_${which}:${formatLayoutStr(r.fragment.shape, r.fragment.stride)}`);
  } catch (e) {
    showErr(`${tabId}-pfab-error`, e.message);
    for (const g of ['frag', 'strip']) {
      const el = document.getElementById(`${tabId}-pfab-${g}-svg`);
      if (el) el.innerHTML = '';
    }
  }
}

/** GRID 1 — the fragment over the partition's domain: which register holds
 *  which (value slot, Rest position). */
function pfabRenderMapViz(tabId) {
  const s = pfabState[tabId];
  if (!s) return;
  const modes = s.fragMode instanceof Set ? s.fragMode : new Set();
  const V = product(s.fragment.shape[0]);
  const Rn = s.restExtents.reduce((a, b) => a * b, 1);

  const svg = buildColoredLayoutSVG([V, Rn], [1, V], modes, (v, r) => {
    const rc = psdRestCoord(r, s.restExtents);
    const lines = [`r${s.fragment.call(v, ...rc)}`];
    if (modes.has('value')) lines.push(`@${s.partition.call(v, ...rc)}`);
    return { bg: colorHighlight(r), stroke: '#1e3a5f', sw: 1.5, text: lines };
  });

  document.getElementById(`${tabId}-pfab-frag-svg`).innerHTML =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `<b style="color:#93c5fd">${formatLayoutStr(s.fragment.shape, s.fragment.stride)}</b>` +
    ` &mdash; ${V * Rn} ${s.dtype} register${V * Rn === 1 ? '' : 's'} per thread, from ` +
    `partition_${s.which} = ${formatLayoutStr(s.partition.shape, s.partition.stride)}<br>` +
    (s.which === 'C'
      ? `plain compact &mdash; make_fragment_C ignores the partition's order`
      : s.fragColMajor
        ? `the partition's mode order is already col-major, so this is too`
        : `<b style="color:#fbbf24">ordered by the partition's strides</b>, not col-major ` +
          `&mdash; that is what makes the copy into it vectorize`) +
    `</div>` + svg;
  applyZoomState(`${tabId}-pfab-frag-svg`);
  updateModeBtns(`${tabId}-pfab-frag-mode-btns`, modes);
  document.getElementById(`${tabId}-pfab-frag-title`).textContent =
    `1. The register map — ${V * Rn} registers`;
}

/** GRID 2 — the registers in index order, coloured by contiguous-source run. */
function pfabRenderStripViz(tabId) {
  const s = pfabState[tabId];
  if (!s) return;
  const { src, runOf, runs, maxRun } = s.order;
  const n = src.length;
  if (n > PFAB_MAX_STRIP) {
    document.getElementById(`${tabId}-pfab-strip-svg`).innerHTML =
      errSVG(`${n} registers is too many to lay out in a strip (max ${PFAB_MAX_STRIP})`);
    return;
  }

  const svg = buildColoredLayoutSVG([1, n], [1, 1], new Set(), (m, r) =>
    ({ bg: colorHighlight(runOf[r]), stroke: '#1e3a5f', sw: 1.5,
       text: [`r${r}`, `@${src[r]}`] }));

  document.getElementById(`${tabId}-pfab-strip-svg`).innerHTML =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `${n} registers in ${runs.length} run${runs.length === 1 ? '' : 's'} of consecutive ` +
    `source elements; longest <b>${maxRun}</b> &mdash; ` +
    (maxRun === 1
      ? `no two consecutive registers hold adjacent source elements, so every load is scalar`
      : `${maxRun} adjacent ${s.dtype} land in ${maxRun} adjacent registers, so a ` +
        `${maxRun * (DTYPE_BITS[s.dtype] || 0)}-bit load covers one`) +
    (s.thrIdx === null ? ` &mdash; offsets are relative to the thread's base` : '') +
    `</div>` + svg;
  applyZoomState(`${tabId}-pfab-strip-svg`);
  document.getElementById(`${tabId}-pfab-strip-title`).textContent =
    `2. The load order — longest run ${maxRun}`;
}

/** Relabel the operand controls. Same shape as pabcSyncOperandField; separate
 *  because the two tabs name different calls in the same widget. */
function pfabSyncOperandField(tabId, which) {
  const spec = PABC_OPERAND[which] || PABC_OPERAND.A;
  const fn = `partition_fragment_${which}`;
  const hint = document.getElementById(`${tabId}-pfab-operand-hint`);
  if (hint) {
    hint.innerHTML =
      `<code>${fn}</code> partitions the <b>(${spec.axes.join(', ')})</b> operand and then ` +
      `builds the register array for it. ` +
      (which === 'C'
        ? `C is the exception: its fragment ignores the partition's order entirely.`
        : `Its mode order follows the source's, which is what makes the load vectorize.`);
  }
  const axesEl = document.getElementById(`${tabId}-pfab-tensor-axes`);
  if (axesEl) axesEl.textContent = `(${spec.axes.join(', ')})`;
  const fnEl = document.getElementById(`${tabId}-pfab-tensor-fn`);
  if (fnEl) fnEl.textContent = fn;
  const group = document.getElementById(`${tabId}-pfab-operand-btns`);
  if (group) group.querySelectorAll('.mode-btn').forEach(b =>
    b.classList.toggle('active', b.textContent.trim() === fn));
}

function setPfabOperand(tabId, which) {
  pfabState[tabId] = Object.assign(pfabState[tabId] || {}, { which });
  pfabSyncOperandField(tabId, which);
  renderPartitionFragmentABC(tabId);
}

function setPfabMode(tabId, grid, mode) {
  const s = pfabState[tabId];
  if (!s) return;
  let modes = s.fragMode;
  if (!(modes instanceof Set)) { modes = new Set(); s.fragMode = modes; }
  if (modes.has(mode)) modes.delete(mode); else modes.add(mode);
  pfabRenderMapViz(tabId);
}

function setPfabOp(tabId) {
  mmaSyncControls(tabId, document.getElementById(`${tabId}-pfab-op-input`).value, 'pfab');
  renderPartitionFragmentABC(tabId);
}

function setPFAB(tabId, which, opKey, ab, acc, k, atomLayout, perm, thr, tensor) {
  pfabState[tabId] = Object.assign(pfabState[tabId] || {}, { which });
  document.getElementById(`${tabId}-pfab-op-input`).value = opKey;
  mmaSyncControls(tabId, opKey, 'pfab');    // options BEFORE values, or the assign is a no-op
  if (ab  && ab  !== 'na') document.getElementById(`${tabId}-pfab-ab-input`).value  = ab;
  if (acc && acc !== 'na') document.getElementById(`${tabId}-pfab-acc-input`).value = acc;
  document.getElementById(`${tabId}-pfab-k-input`).value = String(k);
  document.getElementById(`${tabId}-pfab-atomlayout-input`).value = atomLayout;
  document.getElementById(`${tabId}-pfab-perm-input`).value = perm;
  document.getElementById(`${tabId}-pfab-thr-input`).value = thr;
  document.getElementById(`${tabId}-pfab-tensor-input`).value = tensor;
  renderPartitionFragmentABC(tabId);
}

function exportPFAB(tabId) {
  exportURL(`${tabId}-pfab-export`, 'partition_fragment_abc',
    (pfabState[tabId] && pfabState[tabId].which) || 'A',
    document.getElementById(`${tabId}-pfab-op-input`).value,
    document.getElementById(`${tabId}-pfab-ab-input`).value || 'na',
    document.getElementById(`${tabId}-pfab-acc-input`).value || 'na',
    document.getElementById(`${tabId}-pfab-k-input`).value,
    document.getElementById(`${tabId}-pfab-atomlayout-input`).value,
    document.getElementById(`${tabId}-pfab-perm-input`).value || 'na',
    document.getElementById(`${tabId}-pfab-thr-input`).value,
    document.getElementById(`${tabId}-pfab-tensor-input`).value);
}
