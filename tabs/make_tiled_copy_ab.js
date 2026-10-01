// make_tiled_copy_A/B: match a CopyAtom to a TiledMMA operand.
// Constructor: copy_atom.hpp make_tiled_copy_A/B; side maps: tile2thrfrg.
// No tensor is partitioned here. SRC is the logical addressing pattern, not
// SMEM byte offsets; DST and the reference TV map name register ownership.
const MCAB_COPY_OPS = ['ldmatrix', 'ldmatrix16x8x8b', 'ldmatrix16x16x8b', 's2r'];
const mcabState = {};

/** Read existing CopyAtom traits without assuming SIMT SRC == DST. */
function mcabCopyAtom(opKey, dtype, nm, transpose, bits) {
  if (!MCAB_COPY_OPS.includes(opKey)) throw new Error(`Unsupported SMEM → RMEM copy Op "${opKey}".`);
  const elemBits = DTYPE_BITS[dtype] || (/^float_e[45]m[23]_t$/.test(dtype) ? 8 : 0);
  if (!elemBits) throw new Error(`Unknown copy dtype "${dtype}".`);
  if (opKey === 's2r') {
    if (!Number.isInteger(bits) || bits <= 0 || bits % elemBits)
      throw new Error(`num_bits_per_copy must be a positive multiple of ${elemBits} for ${dtype}.`);
    const tv = new Layout([1, bits / elemBits], [0, 1]);
    return { src: tv, dst: tv, ref: tv, threads: 1, values: bits / elemBits, liveLanes: 1, elemBits,
             label: `CopyUniversalOp (${bits} bits)` };
  }
  const op = MCA_OPS[opKey];
  const atom = mcaLdmatrixAtom(op.ldsm, elemBits, nm, transpose);
  const src = new Layout(atom.src.shape, atom.src.stride);
  const dst = new Layout(atom.dst.shape, atom.dst.stride);
  if (product(src.shape[0]) !== 32 || product(dst.shape[0]) !== 32)
    throw new Error(`${dtype} spans multiple transpose units in ${MCA_LDSM_SPECS[op.ldsm].op}; this view requires a full 32-lane element mapping. Choose CopyUniversalOp, or use a matching instruction unit size.`);
  return { src, dst, ref: dst, threads: 32, values: product(dst.shape[1]),
           liveLanes: atom.liveLanes, elemBits, atom,
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
  if (operand !== 'A' && operand !== 'B') throw new Error('Operand must be A or B.');
  const opnd = mma[operand];
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
  if (side === 'src' && r.copyAtom.threads === 32)
    for (const row of grid) for (const cell of row)
      cell.entries = cell.entries.filter(e => e.t % 32 < r.copyAtom.liveLanes);
  return grid;
}

function generateMakeTiledCopyABTabContent(id) {
  return `
    <div id="${id}-tab-make_tiled_copy_ab" class="panel">
      <div class="controls">
        <h2>make_tiled_copy_A / B</h2>
        <div class="form-group">
          <label>Operand</label>
          <select id="${id}-mcab-operand-input" onchange="renderMakeTiledCopyAB('${id}')">
            <option value="A" selected>A (M × K)</option><option value="B">B (N × K)</option>
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
          </div>
        </div>
        <div class="form-group">
          <label><span id="${id}-mcab-focus-label">Warp id</span>${infoIcon('Focus one warp or thread across both panes. Blank shows all participants.', `${id}-mcab-focus-hint`)}</label>
          <input type="text" id="${id}-mcab-focus-input" value="" placeholder="all warps" oninput="renderMakeTiledCopyAB('${id}')">
        </div>
        <details class="cuo-section" open>
          <summary>1. TiledMMA</summary>
          <div class="cuo-section-body">
            <button class="btn btn-render" id="${id}-mcab-decode-mma" onclick="mcabDecodePasteboard('${id}','make_tiled_mma')">Decode from Pasteboard</button>
            <div class="form-group"><label>MMA Op</label>
              <select id="${id}-mcab-op-input" onchange="renderMakeTiledCopyAB('${id}')">
                <option value="f16bf16" selected>warp.MmaF16BF16Op</option>
                <option value="tf32">warp.MmaTF32Op</option><option value="fp8">warp.MmaFP8Op</option>
              </select>
            </div>
            <div class="form-group" id="${id}-mcab-ab-group"><label>ab_dtype</label><select id="${id}-mcab-ab-input" onchange="renderMakeTiledCopyAB('${id}')"></select></div>
            <div class="form-group" id="${id}-mcab-acc-group"><label>acc_dtype</label><select id="${id}-mcab-acc-input" onchange="renderMakeTiledCopyAB('${id}')"></select></div>
            <div class="form-group"><label>shape_mnk</label><select id="${id}-mcab-k-input" onchange="renderMakeTiledCopyAB('${id}')"><option value="16" selected>(16, 8, 16)</option></select></div>
            <div id="${id}-mcab-op-params" class="cuo-result"></div>
            ${layoutInputField({ id: `${id}-mcab-atomlayout-input`, label: 'atom_layout_mnk', value: '(2, 2, 1)', hint: 'Rank 3: warps along M, N, K. Blank uses (1,1,1). A stride reorders warps.' })}
            ${layoutInputField({ id: `${id}-mcab-perm-input`, label: 'permutation_mnk', value: '(32, 32, 16)', hint: 'Rank-3 tiler. An extent expands a mode; a layout permutes it; blank leaves the modes unchanged.' })}
          </div>
        </details>
        <details class="cuo-section" open>
          <summary>2. CopyAtom</summary>
          <div class="cuo-section-body">
            <button class="btn btn-render" id="${id}-mcab-decode-copy" onclick="mcabDecodePasteboard('${id}','make_copy_atom')">Decode from Pasteboard</button>
            <div class="form-group"><label>Copy Op</label>
              <select id="${id}-mcab-copy-input" onchange="renderMakeTiledCopyAB('${id}')">${MCAB_COPY_OPS.map(key => `<option value="${key}">${key === 's2r' ? 'CopyUniversalOp (SMEM → RMEM)' : MCA_OPS[key].label}</option>`).join('')}</select>
            </div>
            <div class="form-group" id="${id}-mcab-nm-group"><label>num_matrices</label><select id="${id}-mcab-nm-input" onchange="renderMakeTiledCopyAB('${id}')"></select></div>
            <div class="form-group" id="${id}-mcab-trans-group"><label>transpose</label><select id="${id}-mcab-trans-input" onchange="renderMakeTiledCopyAB('${id}')"><option value="0" selected>False</option><option value="1">True</option></select></div>
            <div class="form-group" id="${id}-mcab-bits-group"><label>num_bits_per_copy</label><input type="text" id="${id}-mcab-bits-input" value="16"></div>
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
        </div></div>
        <div class="hint">
          SRC names the lanes supplying shared-memory row addresses; DST names the lanes receiving register values. Values can cross lanes inside ldmatrix. Ignored address lanes are omitted from SRC.<br><br>
          Compare SRC/DST to see the copy's lane mapping, or MMA/COPY to compare the MMA operand with the copy destination. The right pane always shows the TiledCopy destination.<br><br>
          Copy dtype follows the MMA operand. unpack_bits is fixed to None. These are logical ownership maps; an actual SMEM tensor and its alignment are checked when partitioning and executing a copy.
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
  const pasted = String(text).trim();
  const query = pasted.includes('?') ? pasted.slice(pasted.indexOf('?') + 1) : pasted;
  const parsed = parseKeyParam(query.split('#')[0]);
  if (!parsed || parsed.feature !== feature)
    throw new Error(`Paste a ${feature} URL or its query string (?key=...).`);
  const inputs = parsed.inputs;
  const write = (key, value) => { document.getElementById(`${tabId}-mcab-${key}-input`).value = String(value); };
  if (feature === 'make_tiled_mma') {
    if (!MMA_WARP_OPS[inputs[0]]) throw new Error(`Unsupported MMA Op "${inputs[0]}".`);
    write('op', inputs[0]);
    mmaSyncControls(tabId, inputs[0], 'mcab');
    if (inputs[1] !== 'na') write('ab', inputs[1]);
    if (inputs[2] !== 'na') write('acc', inputs[2]);
    write('k', inputs[3]);
    write('atomlayout', inputs[4]);
    write('perm', inputs[5] === 'na' ? '' : inputs[5]);
  } else if (feature === 'make_copy_atom') {
    // UniversalCopy is presented as an SMEM→RMEM load in this tab.
    const copyKey = inputs[0] === 'universal' ? 's2r' : inputs[0];
    if (!MCAB_COPY_OPS.includes(copyKey))
      throw new Error(`Copy Op "${inputs[0]}" is not supported for this SMEM → RMEM comparison.`);
    const dtype = document.getElementById(`${tabId}-mcab-ab-input`).value;
    if (inputs[2] !== dtype)
      throw new Error(`CopyAtom dtype ${inputs[2]} does not match MMA operand dtype ${dtype}. Decode a matching TiledMMA first.`);
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

async function mcabDecodePasteboard(tabId, feature) {
  let pasted;
  try {
    pasted = await navigator.clipboard.readText();
  } catch (e) {
    // Direct-file browsers may deny clipboard reads; allow an ordinary paste.
    pasted = prompt(`Paste the ${feature} URL or its query string:`);
  }
  if (pasted === null || pasted === undefined) return;
  try {
    mcabImportConstructor(tabId, feature, pasted);
  } catch (e) {
    showErr(`${tabId}-mcab-error`, e.message);
  }
}

function mcabSyncCopyControls(tabId) {
  const key = document.getElementById(`${tabId}-mcab-copy-input`).value;
  const ldsm = key !== 's2r';
  for (const part of ['nm', 'trans']) document.getElementById(`${tabId}-mcab-${part}-group`).style.display = ldsm ? '' : 'none';
  document.getElementById(`${tabId}-mcab-bits-group`).style.display = ldsm ? 'none' : '';
  if (!ldsm) return;
  const spec = MCA_LDSM_SPECS[(MCA_OPS[key] || MCA_OPS.ldmatrix).ldsm];
  // Reuse the same dynamic option and required-transpose logic as CopyAtom.
  mcaSyncLdsmControls(tabId, spec, 'mcab');
}

function renderMakeTiledCopyAB(tabId) {
  showErr(`${tabId}-mcab-error`, '');
  showWarn(`${tabId}-mcab-warning`, '');
  try {
    const read = key => document.getElementById(`${tabId}-mcab-${key}-input`).value;
    const operand = read('operand');
    const opKey = read('op');
    if (!MMA_WARP_OPS[opKey]) throw new Error(`Unknown MMA Op "${opKey}".`);
    const op = MMA_WARP_OPS[opKey];
    mmaSyncControls(tabId, opKey, 'mcab');
    mcabSyncCopyControls(tabId);
    const k = Number(read('k'));
    const ab = op.ab ? read('ab') : 'tfloat32_t';
    const acc = op.acc ? read('acc') : 'float';
    if (ab === 'bfloat16_t' && acc !== 'float') throw new Error('BF16 MMA requires a float accumulator.');
    const atom = mmaWarpAtom(opKey, k);
    const atomLayout = mtmParseAtomLayout(read('atomlayout'));
    const perm = mtmParsePerm(read('perm'));
    const mma = mtmComputeTiledMma(atom, atomLayout, perm);
    const copyAtom = mcabCopyAtom(read('copy'), ab, Number(read('nm')), read('trans') === '1', Number(read('bits')));
    const result = mcabComputeTiledCopy(mma, operand, copyAtom);
    if (product(result.tile) > MAX_CELLS) throw new Error(`Operand tile exceeds ${MAX_CELLS} cells.`);
    const prev = mcabState[tabId] || {};
    const mode = prev.mode || 'warp';
    const focusState = mtmReadFocus(tabId, mode, mma.warps, mma.threads, 'mcab');
    mcabState[tabId] = { mode, compare: prev.compare || 'src_dst', ...focusState, mma, result, ab, acc, copyAtom };
    mtmSyncFocusField(tabId, mode, 'mcab', 'Focus one warp or thread across both panes. Blank shows all participants.');
    mmaRenderOpParams(tabId, op, k, ab, acc, 'mcab');
    const fmt = L => formatLayoutStr(L.shape, L.stride);
    document.getElementById(`${tabId}-mcab-atom-result`).innerHTML =
      `<div class="cuo-result-line"><b>${copyAtom.label}</b> — ${ab}, unpack_bits=None</div>` +
      `<div class="cuo-result-line">TV Layout Src = ${fmt(copyAtom.src)}</div>` +
      `<div class="cuo-result-line">TV Layout Dst = ${fmt(copyAtom.dst)}</div>`;
    document.getElementById(`${tabId}-mcab-result`).innerHTML =
      `<div class="cuo-result-line"><b>make_tiled_copy_${operand}(copy_atom, tiled_mma)</b></div>` +
      `<div class="cuo-result-line">Tiler MN = (${result.tile.map(n => `${n}:1`).join(',')})</div>` +
      `<div class="cuo-result-line">TV Layout tiled = ${fmt(result.tv)}</div>` +
      `<div class="cuo-result-line">TV Layout Src = ${fmt(result.src)}</div>` +
      `<div class="cuo-result-line">TV Layout Dst = ${fmt(result.dst)}</div>` +
      `<div class="cuo-result-line">${mma.warps} warps, ${mma.threads} threads; ${result.values} destination values/thread; ${result.copiesPerThread} copy instruction${result.copiesPerThread === 1 ? '' : 's'} per ${copyAtom.threads === 32 ? 'warp' : 'thread'}.</div>`;
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
  const axes = r.operand === 'A' ? 'M×K' : 'N×K';
  document.getElementById(`${tabId}-mcab-mode-btns`).querySelectorAll('.mode-btn').forEach(b =>
    b.classList.toggle('active', b.textContent.trim() === (s.mode === 'tv' ? 'Show TVs' : 'Show Warps')));
  const copyKey = document.getElementById(`${tabId}-mcab-copy-input`).value;
  const [srcSpace, dstSpace] = COPY_OP_MOVES[copyKey][0];
  const compareMma = s.compare === 'mma_copy';
  document.getElementById(`${tabId}-mcab-compare-btns`).querySelectorAll('.mode-btn').forEach(b =>
    b.classList.toggle('active', b.textContent.trim() === (compareMma ? 'MMA/COPY' : 'SRC/DST')));
  setMemorySpaceBadge(`${tabId}-mcab-src-space`, compareMma ? 'RMEM' : srcSpace);
  document.getElementById(`${tabId}-mcab-src-label`).textContent = compareMma ? 'TiledMMA' : 'SRC';
  document.getElementById(`${tabId}-mcab-dst-label`).textContent = 'TiledCopy (DST)';
  setMemorySpaceBadge(`${tabId}-mcab-dst-space`, dstSpace);
  document.getElementById(`${tabId}-mcab-transfer-title`).textContent = `${r.operand} — ${rows}×${cols} ${s.ab} (${axes})`;
  document.getElementById(`${tabId}-mcab-transfer-desc`).textContent =
    compareMma ? 'MMA operand register slots compared with TiledCopy destination slots over the same logical operand coordinates.' :
    'SRC labels the address-supplying lanes; DST labels the receiving lanes. Both panes use logical operand coordinates.' +
    (r.copyAtom.threads === 32 && r.copyAtom.liveLanes < 32 ? ` Only lanes 0–${r.copyAtom.liveLanes - 1} supply addresses for each ldmatrix invocation.` : '');
  const views = {
    src: compareMma ? mtmOperandGrid(r.mmaOperand) : mcabSideGrid(r, 'src'),
    dst: mcabSideGrid(r, 'dst'),
  };
  for (const [key, grid] of Object.entries(views)) {
    const svg = `${tabId}-mcab-${key}-svg`;
    document.getElementById(svg).innerHTML = mtmBuildSVG(grid, rows, cols, { mode: s.mode, focus: s.focus });
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

function setMCAB(tabId, operand, opKey, ab, acc, k, atomLayout, perm, copyKey, nm, transpose, bits, mode, focus, compare) {
  const write = (key, value) => { document.getElementById(`${tabId}-mcab-${key}-input`).value = String(value); };
  write('operand', operand);
  write('op', opKey);
  mmaSyncControls(tabId, opKey, 'mcab');
  if (ab && ab !== 'na') write('ab', ab);
  if (acc && acc !== 'na') write('acc', acc);
  write('k', k);
  write('atomlayout', atomLayout);
  write('perm', perm === 'na' ? '' : perm);
  write('copy', copyKey);
  mcabSyncCopyControls(tabId);
  write('nm', nm);
  write('trans', transpose);
  write('bits', bits);
  write('focus', focus && focus !== 'na' ? focus : '');
  mcabState[tabId] = { mode: mode || 'warp', compare: compare === 'mma_copy' ? 'mma_copy' : 'src_dst' };
  renderMakeTiledCopyAB(tabId);
}

function exportMCAB(tabId) {
  const read = key => document.getElementById(`${tabId}-mcab-${key}-input`).value;
  exportURL(`${tabId}-mcab-export`, 'make_tiled_copy_ab', read('operand'), read('op'),
    read('ab') || 'na', read('acc') || 'na', read('k'), read('atomlayout') || '(1,1,1)',
    read('perm') || 'na', read('copy'), read('nm') || '1', read('trans'), read('bits'),
    (mcabState[tabId] || {}).mode || 'warp', read('focus') || 'na', (mcabState[tabId] || {}).compare || 'src_dst');
}
