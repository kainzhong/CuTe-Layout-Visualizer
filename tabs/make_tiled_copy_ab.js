// make_tiled_copy_A/B/C: match a CopyAtom to a TiledMMA operand.
// Constructor: copy_atom.hpp make_tiled_copy_A/B/C; side maps: tile2thrfrg.
// No tensor is partitioned here. For loads, SRC is the logical addressing
// pattern, not SMEM byte offsets, and DST names register ownership. For the C
// store (the epilogue), SRC is the accumulator register side.
// Offered Ops per operand: A/B are SMEM -> RMEM loads; C is the epilogue store
// RMEM -> SMEM (stmatrix, or SIMT) or a SIMT load of a C source. `s2r` / `r2s`
// are CopyUniversalOp presented in that direction.
const MCAB_OPERAND_COPY_OPS = {
  A: ['ldmatrix', 'ldmatrix16x8x8b', 'ldmatrix16x16x8b', 'ldmatrix8x16x8b', 's2r'],
  B: ['ldmatrix', 'ldmatrix16x8x8b', 'ldmatrix16x16x8b', 'ldmatrix8x16x8b', 's2r'],
  C: ['stmatrix', 'stmatrix16x8x8b', 'r2s', 's2r'],
};
const MCAB_COPY_OPS = ['ldmatrix', 'ldmatrix16x8x8b', 'ldmatrix16x16x8b', 'ldmatrix8x16x8b', 'stmatrix', 'stmatrix16x8x8b', 's2r', 'r2s'];

/** make_tiled_copy_ab URL fields -> the optional copy dtype for operand C. It
 *  follows the Op-specific TiledMMA tail (WGMMA: 5 fields, warp MMA: none), so
 *  URLs without it keep parsing. Empty means "the accumulator dtype". */
function mcabURLCopyDtype(a) {
  const wg = !!MMA_WARPGROUP_OPS[a[1]];
  const v = a[14 + (wg ? 5 : 0)];
  return v && v !== 'na' ? v : '';
}

/** The CopyAtom's value type. A/B copy the operand's own type. C may differ:
 *  an epilogue converts float accumulators to e.g. half_t before stmatrix, and
 *  the atom is built with that output type. */
function mcabCopyDtype(atom, operand, cdtype) {
  return operand === 'C' && cdtype ? cdtype : atom.dtypes[operand];
}
const MCAB_COPY_LABELS = { s2r: 'CopyUniversalOp (SMEM → RMEM)', r2s: 'CopyUniversalOp (RMEM → SMEM)' };

function mcabCopyOps(operand) {
  return MCAB_OPERAND_COPY_OPS[operand] || MCAB_OPERAND_COPY_OPS.A;
}

function mcabCopyLabel(key) {
  return MCAB_COPY_LABELS[key] || MCA_OPS[key].label;
}

function mcabCopyOptions(operand) {
  return mcabCopyOps(operand).map(key => `<option value="${key}">${mcabCopyLabel(key)}</option>`).join('');
}

/** Rebuild an operand-dependent CopyAtom Op picker before assigning a value. */
function mcabSyncCopyOptions(tabId, p, operand) {
  const sel = document.getElementById(`${tabId}-${p}-copy-input`);
  if (!sel) return;
  const ops = mcabCopyOps(operand), want = sel.value;
  const markup = mcabCopyOptions(operand);
  if (sel.innerHTML !== markup) sel.innerHTML = markup;
  sel.value = ops.includes(want) ? want : ops[0];
}
const mcabState = {};

/** Read existing CopyAtom traits without assuming SIMT SRC == DST. */
function mcabCopyAtom(opKey, dtype, nm, transpose, bits) {
  if (!MCAB_COPY_OPS.includes(opKey)) throw new Error(`Unsupported copy Op "${opKey}" for make_tiled_copy_A/B/C.`);
  const elemBits = DTYPE_BITS[dtype] || (/^float_e[45]m[23]_t$/.test(dtype) ? 8 : 0);
  if (!elemBits) throw new Error(`Unknown copy dtype "${dtype}".`);
  if (opKey === 's2r' || opKey === 'r2s') {
    if (!Number.isInteger(bits) || bits <= 0 || bits % elemBits)
      throw new Error(`num_bits_per_copy must be a positive multiple of ${elemBits} for ${dtype}.`);
    const tv = new Layout([1, bits / elemBits], [0, 1]);
    return { src: tv, dst: tv, ref: tv, threads: 1, values: bits / elemBits, liveLanes: 1, elemBits, opKey,
             label: `CopyUniversalOp (${bits} bits)` };
  }
  const op = MCA_OPS[opKey];
  const atom = mcaLdmatrixAtom(op.ldsm, elemBits, nm, transpose);
  const src = new Layout(atom.src.shape, atom.src.stride);
  const dst = new Layout(atom.dst.shape, atom.dst.stride);
  if (product(src.shape[0]) !== 32 || product(dst.shape[0]) !== 32)
    throw new Error(`${dtype} spans multiple transpose units in ${MCA_LDSM_SPECS[op.ldsm].op}; this view requires a full 32-lane element mapping. Choose CopyUniversalOp, or use a matching instruction unit size.`);
  // CuTe's RefLayout is the register side: DST for ldmatrix, SRC for stmatrix.
  const ref = atom.regSide === 'src' ? src : dst;
  return { src, dst, ref, threads: 32, values: product(ref.shape[1]),
           liveLanes: atom.liveLanes, addrSide: atom.addrSide, elemBits, atom, opKey,
           label: `${MCA_LDSM_SPECS[op.ldsm].op}.x${nm}${transpose ? '.trans' : ''}` };
}

/** Exactly CuTe's reference -> SRC/DST transformation, using the shared
 * partition helper on a compact logical tile. Coalescing preserves the copy
 * atom's value mode and the instruction-repetition mode separately. */
function mcabSideLayout(tv, tile, copyAtom, side) {
  const ref2side = composition(right_inverse(copyAtom.ref), copyAtom[side]);
  const tensorZ = new Layout([tile, 1], [[1, tile[0]], 0]);
  const r = psdTile2ThrFrg(tensorZ, tv, copyAtom.threads, copyAtom.values, ref2side);
  return make_layout(r.tidfrg.mode(0), r.tidfrg.mode(1));
}

/** DOM-free constructor + both tiled TV maps; CuTeDSL is the oracle. */
function mcabComputeTiledCopy(mma, operand, copyAtom) {
  if (!['A', 'B', 'C'].includes(operand)) throw new Error('Operand must be A, B or C.');
  if (copyAtom.opKey && !mcabCopyOps(operand).includes(copyAtom.opKey))
    throw new Error(operand === 'C'
      ? 'make_tiled_copy_C offers the accumulator store (stmatrix or SIMT, RMEM → SMEM) and a SIMT C-source load (SMEM → RMEM).'
      : `The RMEM → SMEM store applies to the accumulator; choose operand C, or an SMEM → RMEM load for ${operand}.`);
  const opnd = mma[operand];
  // A WGMMA shared-memory operand is read by the instruction through a
  // descriptor; there is no register fragment for an SMEM -> RMEM copy to fill.
  // (CuTeDSL still builds a degenerate TiledCopy in which every thread of a
  // warpgroup receives the whole tile; this view does not present that.)
  if (opnd.space === 'SMEM')
    throw new Error(`WGMMA reads operand ${operand} directly from shared memory through a descriptor, so ` +
      `make_tiled_copy_${operand} has no register fragment to fill. ` +
      (operand === 'A' ? 'Set the WGMMA A source to RMEM (register A) to load A with a TiledCopy.'
                       : 'WGMMA B is always a shared-memory operand; choose operand A with A source RMEM.'));
  // CuTe's get_layoutC_TV (mma_atom.hpp) lacks the stride-0 K step that its A/B
  // siblings have, so with both N and K replication it contradicts partition_C;
  // make_tiled_copy_C is built from it. The tiled_mma oracle pins the divergence.
  if (operand === 'C' && mma.atomLayoutMNK && mma.atomLayoutMNK[1] > 1 && mma.atomLayoutMNK[2] > 1)
    throw new Error('With atom_layout_mnk replicated along both N and K, CuTe\'s get_layoutC_TV (which ' +
      'make_tiled_copy_C uses) contradicts partition_C, so the TiledCopy would not match the accumulator ' +
      'fragments. Reduce the K-split partial sums first, or use N = 1 or K = 1.');
  const tv = opnd.tv;
  const values = product(tv.shape[1]);
  if (mma.threads % copyAtom.threads)
    throw new Error(`TiledCopy threads (${mma.threads}) must be a multiple of CopyAtom threads (${copyAtom.threads}).`);
  if (values % copyAtom.values)
    throw new Error(`TiledCopy uses too few vals for selected CopyAtom: ${values} values/thread must be a multiple of ${copyAtom.values}. Choose a smaller num_matrices or expand the TiledMMA operand tile.`);
  return { operand, tile: opnd.tile, tv, src: mcabSideLayout(tv, opnd.tile, copyAtom, 'src'),
           dst: mcabSideLayout(tv, opnd.tile, copyAtom, 'dst'),
           threads: mma.threads, warps: mma.warps, values,
           copiesPerThread: values / copyAtom.values, copyAtom, mmaOperand: opnd };
}

/** Adapt a Copy TV map to the shared MMA grid renderer. In SRC, ignored
 * ldmatrix address lanes are omitted: they still execute but supply no row.
 * The complete (including aliased lanes) map stays in r.src and the oracle. */
function mcabSideGrid(r, side) {
  const tv = r[side];
  const grid = mtmOperandGrid({ tile: r.tile, thr: tv.mode(0), val: tv.mode(1),
                              frgSize: product(tv.shape[1]) });
  if (side === (r.copyAtom.addrSide || 'src') && r.copyAtom.threads === 32)
    for (const row of grid) for (const cell of row)
      cell.entries = cell.entries.filter(e => e.t % 32 < r.copyAtom.liveLanes);
  return grid;
}

function generateMakeTiledCopyABTabContent(id) {
  return `
    <div id="${id}-tab-make_tiled_copy_ab" class="panel">
      <div class="controls">
        <h2>make_tiled_copy_A / B / C</h2>
        <div class="form-group">
          <label>Operand</label>
          <select id="${id}-mcab-operand-input" onchange="renderMakeTiledCopyAB('${id}')">
            <option value="A" selected>A (M × K)</option><option value="B">B (N × K)</option><option value="C">C (M × N)</option>
          </select>
        </div>
        <div class="form-group">
          <label>Compare</label>
          <div class="seg-control" id="${id}-mcab-compare-btns">
            <button class="mode-btn active" onclick="setMcabCompare('${id}','src_dst')">SRC/DST</button>
            <button class="mode-btn" onclick="setMcabCompare('${id}','mma_copy')">MMA/COPY</button>
          </div>
        </div>
        <div class="form-group">
          <label>Cell labels ${infoIcon(MTM_MODE_HINT)}</label>
          <div class="seg-control" id="${id}-mcab-mode-btns">
            <button class="mode-btn" onclick="setMcabMode('${id}','tv')">Show TVs</button>
            <button class="mode-btn active" onclick="setMcabMode('${id}','warp')">Show Warps</button>
            <button class="mode-btn" id="${id}-mcab-group-mode-btn" style="display:none" onclick="setMcabMode('${id}','warpgroup')">Show Warp Groups</button>
          </div>
        </div>
        <div class="form-group">
          <label><span id="${id}-mcab-focus-label">Warp id</span>${infoIcon('Focus one warp or thread across both panes. Blank shows all participants.', `${id}-mcab-focus-hint`)}</label>
          <input type="text" id="${id}-mcab-focus-input" value="" placeholder="all warps" oninput="renderMakeTiledCopyAB('${id}')">
        </div>
        <details class="cuo-section" open>
          <summary>1. TiledMMA</summary>
          <div class="cuo-section-body">
            <button class="btn btn-render btn-decode" id="${id}-mcab-decode-mma" onclick="mcabDecodePasteboard('${id}','make_tiled_mma')">Decode from Pasteboard</button>
            ${mtmManualConstructorMarkup({id,p:'mcab',render:'renderMakeTiledCopyAB',atomLayout:'(2, 2, 1)',perm:'(32, 32, 16)',k:16})}
            <div id="${id}-mcab-op-params" class="cuo-result"></div>
          </div>
        </details>
        <details class="cuo-section" open>
          <summary>2. CopyAtom</summary>
          <div class="cuo-section-body">
            <button class="btn btn-render btn-decode" id="${id}-mcab-decode-copy" onclick="mcabDecodePasteboard('${id}','make_copy_atom')">Decode from Pasteboard</button>
            <details class="cuo-section" id="${id}-mcab-copy-manual-construction">
              <summary>Manual construction</summary>
              <div class="cuo-section-body">
                <div class="form-group"><label>Op</label>
                  <select id="${id}-mcab-copy-input" onchange="renderMakeTiledCopyAB('${id}')">${mcabCopyOptions('A')}</select>
                </div>
                <div class="form-group" id="${id}-mcab-nm-group"><label>num_matrices</label><select id="${id}-mcab-nm-input" onchange="renderMakeTiledCopyAB('${id}')"></select></div>
                <div class="form-group" id="${id}-mcab-trans-group"><label>transpose</label><select id="${id}-mcab-trans-input" onchange="renderMakeTiledCopyAB('${id}')"><option value="0" selected>False</option><option value="1">True</option></select></div>
                <div class="form-group" id="${id}-mcab-bits-group"><label>num_bits_per_copy</label><input type="text" id="${id}-mcab-bits-input" value="16"></div>
                <div class="form-group" id="${id}-mcab-cdtype-group" style="display:none"><label>dtype<span style="color:#6b7280;font-weight:normal">&nbsp;&mdash; the CopyAtom's value type. An epilogue converts float accumulators to its output type, e.g. half_t, before stmatrix</span></label>
                  <select id="${id}-mcab-cdtype-input" onchange="renderMakeTiledCopyAB('${id}')"><option value="">same as the accumulator</option>${dtypeOptions('')}</select></div>
              </div>
            </details>
            <div id="${id}-mcab-atom-result" class="cuo-result"></div>
          </div>
        </details>
        <button class="btn btn-render" onclick="renderMakeTiledCopyAB('${id}')">Render</button>
        <button class="btn btn-render" id="${id}-mcab-export" onclick="exportMCAB('${id}')">Export URL</button>
        ${statusDivs(`${id}-mcab`)}
        <div id="${id}-mcab-result" class="cuo-result"></div>
        <div class="presets"><label>Presets</label><div class="preset-buttons">
          <button class="preset-btn" onclick="setMCAB('${id}','A','f16bf16','half_t','float',16,'(2,2,1)','(32,32,16)','ldmatrix',4,0,16)">A — .x4, four warps</button>
          <button class="preset-btn" onclick="setMCAB('${id}','B','f16bf16','half_t','float',16,'(2,2,1)','(32,32,16)','ldmatrix',4,0,16)">B — .x4, N expanded to 32</button>
          <button class="preset-btn" onclick="setMCAB('${id}','A','f16bf16','half_t','float',16,'(2,2,1)','(32,32,16)','ldmatrix',1,0,16)">A — .x1, four loads per warp</button>
          <button class="preset-btn" onclick="setMCAB('${id}','B','f16bf16','half_t','float',16,'(2,2,1)','','ldmatrix',2,1,16)">B — .x2.trans, N contiguous</button>
          <button class="preset-btn" onclick="setMCAB('${id}','A','f16bf16','bfloat16_t','float',16,'(2,2,1):(2,1,4)','((2,16):(16,1),32,16)','ldmatrix',2,0,16)">A — permuted rows and warp order</button>
          <button class="preset-btn" onclick="setMCAB('${id}','B','fp8','float_e4m3_t','float',32,'(2,2,1)','(32,32,32)','ldmatrix16x8x8b',4,1,8)">B — FP8, permuted 8-bit load</button>
          <button class="preset-btn" onclick="setMCAB('${id}','A','fp8','float_e4m3_t','float',32,'(2,2,1)','(32,32,32)','ldmatrix16x16x8b',2,1,8)">A — FP8, 16×16 load</button>
          <button class="preset-btn" onclick="setMCAB('${id}','A','tf32',null,null,8,'(2,2,1)','','s2r',1,0,32)">A — TF32, scalar SIMT load</button>
          <button class="preset-btn" onclick="setMCAB('${id}','C','f16bf16','half_t','float',16,'(2,2,1)','(32,32,16)','stmatrix',4,0,16,'warp','na','mma_copy','half_t')">C — stmatrix .x4 epilogue: float accumulators stored as half_t</button>
          <button class="preset-btn" onclick="setMCAB('${id}','C','f16bf16','half_t','float',16,'(2,2,1)','(32,32,16)','stmatrix',4,1,16,'warp','na','src_dst','half_t')">C — stmatrix .x4.trans, for M-major SMEM</button>
          <button class="preset-btn" onclick="setMCAB('${id}','C','f16bf16','half_t','float',16,'(2,2,1)','(32,32,16)','r2s',1,0,64)">C — SIMT epilogue store, 64-bit pairs</button>
          <button class="preset-btn" onclick="setMCAB('${id}','C','f16bf16','half_t','half_t',16,'(2,2,1)','(32,32,16)','r2s',1,0,32,'tv','na','mma_copy')">C — half_t accumulator, MMA/COPY</button>
          <button class="preset-btn" onclick="setMCAB('${id}','C','f16bf16','half_t','float',16,'(2,2,1)','(32,32,16)','s2r',1,0,32)">C — load a C source, 32-bit</button>
          <button class="preset-btn" onclick="setMCAB('${id}','A','wg_f16bf16','half_t','float',16,'(2,1,1)','','ldmatrix',4,0,16,'warpgroup','na','src_dst',64,'RMEM','K','K','half_t')">WGMMA register A — .x4, two warpgroups</button>
          <button class="preset-btn" onclick="setMCAB('${id}','C','wg_f16bf16','half_t','float',16,'(1,1,1)','','stmatrix',4,0,16,'warpgroup','na','mma_copy',64,'SMEM','K','K','half_t','half_t')">WGMMA accumulator C — the SM90 stmatrix epilogue</button>
          <button class="preset-btn" onclick="setMCAB('${id}','A','wg_fp8','float_e4m3_t','float',32,'(1,1,1)','','ldmatrix',4,0,16,'tv','na','mma_copy',64,'RMEM','K','K','float_e4m3_t')">WGMMA FP8 register A — MMA/COPY</button>
        </div></div>
        <div class="hint">
          For an A/B load, SRC names the lanes supplying shared-memory row addresses; DST names the lanes receiving register values. Values can cross lanes inside ldmatrix. Ignored address lanes are omitted from SRC.<br><br>
          Operand C is the accumulator: its RMEM → SMEM store is the epilogue, and its SMEM → RMEM load reads a C source. SRC is the register side of a store.<br><br>
          Compare SRC/DST to see the copy's lane mapping, or MMA/COPY to compare the MMA operand with the copy's register side (DST for a load, SRC for a store).<br><br>
          Copy dtype follows the MMA operand (the accumulator dtype for C). unpack_bits is fixed to None. These are logical ownership maps; an actual SMEM tensor and its alignment are checked when partitioning and executing a copy.
        </div>
      </div>
      <div class="comp-results">
        <div class="comp-viz-span mtm-group-title">Compare</div>
        <div class="comp-viz-item comp-viz-span">
          <div class="comp-viz-header"><span class="comp-viz-title" id="${id}-mcab-transfer-title">Operand A</span><span><button class="mode-btn" onclick="toggleCopyZoom('${id}','mcab')">Zoom in</button></span></div>
          <div class="cuo-viz-desc" id="${id}-mcab-transfer-desc"></div>
          ${copyPanes(id, 'mcab')}
        </div>
      </div>
    </div>`;
}

// Import constructor inputs without navigating away from this comparison.
function mcabImportConstructor(tabId, feature, text) {
  const parsed = parsePastedKeyParam(text);
  if (!parsed || parsed.feature !== feature)
    throw new Error(`Paste a ${feature} URL or its query string (?key=...).`);
  const inputs = parsed.inputs;
  const write = (key, value) => { document.getElementById(`${tabId}-mcab-${key}-input`).value = String(value); };
  if (feature === 'make_tiled_mma') {
    mtmWriteConstructorControls(tabId, 'mcab', inputs);
  } else if (feature === 'make_copy_atom') {
    // UniversalCopy is presented in the operand's usual direction: an
    // SMEM→RMEM load for A/B, the RMEM→SMEM epilogue store for C.
    const operand = document.getElementById(`${tabId}-mcab-operand-input`).value;
    const copyKey = inputs[0] === 'universal' ? (operand === 'C' ? 'r2s' : 's2r') : inputs[0];
    if (!mcabCopyOps(operand).includes(copyKey))
      throw new Error(`Copy Op "${inputs[0]}" is not supported for make_tiled_copy_${operand}.`);
    const { atom } = mtmDecodeConstructor(mtmReadConstructorControls(tabId, 'mcab'));
    const dtype = atom.dtypes[operand];
    // A and B copy the operand's own type. C's CopyAtom carries the epilogue's
    // output type, which may differ from the accumulator.
    if (operand !== 'C' && inputs[2] !== dtype)
      throw new Error(`CopyAtom dtype ${inputs[2]} does not match MMA operand ${operand} dtype ${dtype}. Decode a matching TiledMMA first.`);
    if (operand === 'C' && !DTYPE_BITS[inputs[2]])
      throw new Error(`CopyAtom dtype ${inputs[2]} is not offered for make_tiled_copy_C.`);
    mcabSyncCopyOptions(tabId, 'mcab', operand);
    if (operand === 'C') write('cdtype', inputs[2] === dtype ? '' : inputs[2]);
    write('copy', copyKey);
    mcabSyncCopyControls(tabId);
    write('bits', inputs[1]);
    if (copyKey !== 's2r') {
      write('nm', inputs[3] || '1');
      write('trans', inputs[4] || '0');
    }
  } else {
    throw new Error(`Unsupported constructor "${feature}".`);
  }
  renderMakeTiledCopyAB(tabId);
}

function mcabDecodePasteboard(tabId, feature) {
  return decodeFromPasteboard({tabId, errorId: `${tabId}-mcab-error`, sources: [feature],
    message: `Paste the ${feature} URL or its query string:`, importFn: text => mcabImportConstructor(tabId, feature, text)});
}

function mcabSyncCopyControls(tabId, p = 'mcab') {
  const key = document.getElementById(`${tabId}-${p}-copy-input`).value;
  const ldsm = key !== 's2r' && key !== 'r2s';
  for (const part of ['nm', 'trans']) document.getElementById(`${tabId}-${p}-${part}-group`).style.display = ldsm ? '' : 'none';
  document.getElementById(`${tabId}-${p}-bits-group`).style.display = ldsm ? 'none' : '';
  if (!ldsm) return;
  const spec = MCA_LDSM_SPECS[(MCA_OPS[key] || MCA_OPS.ldmatrix).ldsm];
  // Reuse the same dynamic option and required-transpose logic as CopyAtom.
  mcaSyncLdsmControls(tabId, spec, p);
}

function renderMakeTiledCopyAB(tabId) {
  showErr(`${tabId}-mcab-error`, '');
  showWarn(`${tabId}-mcab-warning`, '');
  try {
    const read = key => document.getElementById(`${tabId}-mcab-${key}-input`).value;
    const operand = read('operand');
    // The shared constructor reader/decoder: every MMA Op is offered here.
    const { atom, tiled: mma } = mtmDecodeConstructor(mtmReadConstructorControls(tabId, 'mcab'));
    const op = atom.op, k = atom.k;
    mcabSyncCopyOptions(tabId, 'mcab', operand);
    mcabSyncCopyControls(tabId);
    document.getElementById(`${tabId}-mcab-cdtype-group`).style.display = operand === 'C' ? '' : 'none';
    const ab = mcabCopyDtype(atom, operand, read('cdtype')), acc = atom.dtypes.C;
    const copyAtom = mcabCopyAtom(read('copy'), ab, Number(read('nm')), read('trans') === '1', Number(read('bits')));
    const result = mcabComputeTiledCopy(mma, operand, copyAtom);
    if (product(result.tile) > MAX_CELLS) throw new Error(`Operand tile exceeds ${MAX_CELLS} cells.`);
    const prev = mcabState[tabId] || {};
    let mode = prev.mode || 'warp';
    if (mode === 'warpgroup' && !atom.warpgroup) mode = 'warp';
    const focusState = mtmReadFocus(tabId, mode, mma.warps, mma.threads, 'mcab', mma.atomGroups);
    mcabState[tabId] = { mode, compare: prev.compare || 'src_dst', ...focusState, mma, result, ab, acc, copyAtom, atom };
    mtmSyncFocusField(tabId, mode, 'mcab', 'Focus one warp, warpgroup or thread across both panes. Blank shows all participants.');
    mmaRenderOpParams(tabId, op, k, atom.dtypes.A, acc, 'mcab', atom);
    const fmt = L => formatLayoutStr(L.shape, L.stride);
    document.getElementById(`${tabId}-mcab-atom-result`).innerHTML =
      `<div class="cuo-result-line"><b>${copyAtom.label}</b> — ${ab}, unpack_bits=None</div>` +
      `<div class="cuo-result-line">TV Layout Src = ${fmt(copyAtom.src)}</div>` +
      `<div class="cuo-result-line">TV Layout Dst = ${fmt(copyAtom.dst)}</div>`;
    document.getElementById(`${tabId}-mcab-result`).innerHTML = mtcDecodedSummary({
      ...result, tiler: result.tile.map(n => new Layout(n)),
      constructorName: `make_tiled_copy_${operand}(copy_atom, tiled_mma)`,
    });
    mcabRenderViz(tabId);
    showErr(`${tabId}-mcab-error`, focusState.focusErr);
    if (copyAtom.atom && copyAtom.atom.transpose && copyAtom.elemBits !== MCA_LDSM_SPECS[copyAtom.atom.opKey].unitBits)
      showWarn(`${tabId}-mcab-warning`, 'The transpose acts on instruction-sized bit units rather than whole operand elements. This is the CuTeDSL ownership layout; check the intended load before using it in a kernel.');
    updateOuterTabLabel(tabId, `make_tiled_copy_${operand}:${result.tile.join('x')}/${copyAtom.label}`);
  } catch (e) {
    showErr(`${tabId}-mcab-error`, e.message);
    document.getElementById(`${tabId}-mcab-result`).innerHTML = '';
    document.getElementById(`${tabId}-mcab-atom-result`).innerHTML = '';
    for (const key of ['src', 'dst']) document.getElementById(`${tabId}-mcab-${key}-svg`).innerHTML = '';
    delete mcabState[tabId];
  }
}

function mcabRenderViz(tabId) {
  const s = mcabState[tabId];
  const r = s.result;
  const [rows, cols] = r.tile;
  const axes = { A: 'M×K', B: 'N×K', C: 'M×N' }[r.operand];
  document.getElementById(`${tabId}-mcab-mode-btns`).querySelectorAll('.mode-btn').forEach(b =>
    b.classList.toggle('active', b.textContent.trim() ===
      ({tv: 'Show TVs', warp: 'Show Warps', warpgroup: 'Show Warp Groups'}[s.mode])));
  const groupBtn = document.getElementById(`${tabId}-mcab-group-mode-btn`);
  if (groupBtn) groupBtn.style.display = s.atom.warpgroup ? '' : 'none';
  const copyKey = document.getElementById(`${tabId}-mcab-copy-input`).value;
  const [srcSpace, dstSpace] = COPY_OP_MOVES[copyKey][0];
  const compareMma = s.compare === 'mma_copy';
  document.getElementById(`${tabId}-mcab-compare-btns`).querySelectorAll('.mode-btn').forEach(b =>
    b.classList.toggle('active', b.textContent.trim() === (compareMma ? 'MMA/COPY' : 'SRC/DST')));
  // MMA/COPY compares the MMA registers with the copy's register side: DST
  // for a load, SRC for the epilogue store.
  const regSide = srcSpace === 'RMEM' ? 'src' : 'dst';
  const rightSide = compareMma ? regSide : 'dst';
  setMemorySpaceBadge(`${tabId}-mcab-src-space`, compareMma ? 'RMEM' : srcSpace);
  document.getElementById(`${tabId}-mcab-src-label`).textContent = compareMma ? 'TiledMMA' : 'SRC';
  document.getElementById(`${tabId}-mcab-dst-label`).textContent = `TiledCopy (${rightSide.toUpperCase()})`;
  setMemorySpaceBadge(`${tabId}-mcab-dst-space`, rightSide === 'src' ? srcSpace : dstSpace);
  document.getElementById(`${tabId}-mcab-transfer-title`).textContent = `${r.operand} — ${rows}×${cols} ${s.ab} (${axes})`;
  document.getElementById(`${tabId}-mcab-transfer-desc`).textContent =
    compareMma ? `MMA operand register slots compared with the TiledCopy ${rightSide === 'src' ? 'source (register) slots of the store' : 'destination slots'} over the same logical operand coordinates.` :
    (r.copyAtom.threads === 1 ? 'SRC and DST label the thread and value that read and write each element. Both panes use logical operand coordinates.'
      : 'SRC labels the address-supplying lanes; DST labels the receiving lanes. Both panes use logical operand coordinates.') +
    (r.copyAtom.threads === 32 && r.copyAtom.liveLanes < 32 ? ` Only lanes 0–${r.copyAtom.liveLanes - 1} supply addresses for each ldmatrix invocation.` : '');
  const views = {
    src: compareMma ? mtmOperandGrid(r.mmaOperand) : mcabSideGrid(r, 'src'),
    dst: mcabSideGrid(r, rightSide),
  };
  for (const [key, grid] of Object.entries(views)) {
    const svg = `${tabId}-mcab-${key}-svg`;
    document.getElementById(svg).innerHTML = mtmBuildSVG(grid, rows, cols, {
      mode: s.mode, focus: s.focus, atomThreads: s.mma.atomThreads,
      groupPrefix: s.mode === 'warpgroup' ? 'WG' : 'W' });
    applyZoomState(svg);
  }

}

function setMcabCompare(tabId, compare) {
  mcabState[tabId] = { ...(mcabState[tabId] || {}), compare };
  renderMakeTiledCopyAB(tabId);
}

function setMcabMode(tabId, mode) {
  mcabState[tabId] = { ...(mcabState[tabId] || {}), mode };
  renderMakeTiledCopyAB(tabId);
}

/** URL field order; `...tail` is the atom's Op-specific constructor tail
 *  (WGMMA: N, A source, A/B major, B dtype), after the optional fields, then
 *  operand C's optional copy dtype (see mcabURLCopyDtype). */
function setMCAB(tabId, operand, opKey, ab, acc, k, atomLayout, perm, copyKey, nm, transpose, bits, mode, focus, compare, ...tail) {
  const write = (key, value) => { document.getElementById(`${tabId}-mcab-${key}-input`).value = String(value); };
  const cdtype = mcabURLCopyDtype([operand, opKey, ab, acc, k, atomLayout, perm, copyKey, nm, transpose, bits, mode, focus, compare, ...tail]);
  mtmWriteConstructorControls(tabId, 'mcab', mtmJoinConstructor(
    [opKey, ab || 'na', acc || 'na', String(k), atomLayout, perm || 'na'], tail));
  write('operand', operand);
  mcabSyncCopyOptions(tabId, 'mcab', operand);
  write('copy', copyKey);
  mcabSyncCopyControls(tabId);
  write('nm', nm);
  write('trans', transpose);
  write('bits', bits);
  write('focus', focus && focus !== 'na' ? focus : '');
  write('cdtype', operand === 'C' ? cdtype : '');
  mcabState[tabId] = { mode: mode || 'warp', compare: compare === 'mma_copy' ? 'mma_copy' : 'src_dst' };
  renderMakeTiledCopyAB(tabId);
}

function exportMCAB(tabId) {
  const read = key => document.getElementById(`${tabId}-mcab-${key}-input`).value;
  const { head, tail } = mtmSplitConstructor(mtmReadConstructorControls(tabId, 'mcab'));
  head[4] = head[4] || '(1,1,1)';
  exportURL(`${tabId}-mcab-export`, 'make_tiled_copy_ab', read('operand'), ...head,
    read('copy'), read('nm') || '1', read('trans'), read('bits'),
    (mcabState[tabId] || {}).mode || 'warp', read('focus') || 'na', (mcabState[tabId] || {}).compare || 'src_dst', ...tail,
    ...(read('operand') === 'C' && read('cdtype') ? [read('cdtype')] : []));
}

/** make_tiled_copy_ab URL fields -> TiledMMA constructor (tail after compare). */
function mcabConstructorFromURL(a) {
  return mtmJoinConstructor(a.slice(1, 7), a.slice(14));
}
