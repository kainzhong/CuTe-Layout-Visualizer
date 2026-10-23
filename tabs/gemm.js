// GEMM invocation coverage over already partitioned RMEM fragments.
const GEMM_DEFAULT_MMA = ['f16bf16', 'half_t', 'half_t', '16', '(2,2,1)', '(32,32,16)'];
const gemmState = {};


// Dispatch according to algorithm.py's rank rules, then enumerate logical
// atom invocations. This is coverage, not the compiler's instruction order.
function gemmCompute(mmaInputs, aText, bText, cText, dText) {
  const mma = mtmDecodeConstructor(mmaInputs);
  const A = parseExactLayoutInput(aText), B = parseExactLayoutInput(bText);
  const C = parseExactLayoutInput(cText), D = parseExactLayoutInput(String(dText || '').trim() || cText);
  const ar = A.rank(), cr = C.rank();
  if (ar !== B.rank()) throw new Error('A and B must have the same rank.');
  if (cr !== D.rank()) throw new Error('C and D must have the same rank.');
  if (ar === 1) throw new Error('CuTeDSL warp gemm requires A/B rank 2 or 3. For one invocation, use (V,1) A/B and (V,1,1) C/D fragments.');
  let dispatch;
  if (ar === 2 && cr === 2) dispatch = 3;
  else if (ar === 2 && cr === 3) dispatch = 4;
  else if (ar === 3 && cr === 3) dispatch = 5;
  else throw new Error('Unsupported gemm ranks. Pass (V,M)/(V,N) or (V,M,K)/(V,N,K) fragments with matching C/D.');
  if (dispatch === 3)
    throw new Error(`Dispatch [${dispatch}] has implicit V=1, which does not match this warp MMA. Include the per-instruction V mode from partition/make_fragment.`);
  const modeSize = (L, i) => product(Array.isArray(L.shape) ? L.shape[i] : L.shape);
  for (const [name, L, which] of [['A', A, 'A'], ['B', B, 'B'], ['C', C, 'C'], ['D', D, 'C']]) {
    if (modeSize(L, 0) !== mma.atom[which].numVal)
      throw new Error(`${name} V mode has ${modeSize(L, 0)} values; this MMA expects ${mma.atom[which].numVal}. Pass a register fragment, not an unpartitioned matrix.`);
  }
  const M = modeSize(A, 1);
  const N = modeSize(B, 1);
  const K = dispatch === 5 ? modeSize(A, 2) : 1;
  if (modeSize(C, 1) !== M || modeSize(C, 2) !== N || modeSize(D, 1) !== M || modeSize(D, 2) !== N)
    throw new Error('A.M, B.N and C/D.(M,N) extents must agree.');
  if (dispatch === 5 && modeSize(B, 2) !== K) throw new Error('A.K and B.K extents must agree.');
  if (M * N * K > MAX_CELLS) throw new Error(`Too many invocations: ${M} × ${N} × ${K}; maximum ${MAX_CELLS}.`);
  const slots = (L, which, m, n, k) => {
    const coord = which === 'A' ? [null, m] : which === 'B' ? [null, n] : [null, m, n];
    if (dispatch === 5 && (which === 'A' || which === 'B')) coord.push(k);
    const [slice, offset] = slice_and_offset(coord, L);
    return Array.from({length: slice.size()}, (_, i) => slice.call(i) + offset);
  };
  const calls = [];
  for (let k = 0; k < K; k++) for (let n = 0; n < N; n++) for (let m = 0; m < M; m++) {
    calls.push({m, n, k, a: slots(A, 'A', m, n, k), b: slots(B, 'B', m, n, k),
      c: slots(C, 'C', m, n, k), d: slots(D, 'D', m, n, k)});
  }
  return {mma, A, B, C, D, dispatch, M, N, K, calls};
}

// Project the same logical repetitions into MK / NK (or KN) / MN panes.
// B's canonical value index remains n + k*N when drawn transposed.
function gemmCoverageGrid(r, which, altView, selectedK) {
  const transposed = which === 'b' && altView;
  const shape = which === 'a' ? [r.M, r.K] : which === 'c' ? [r.M, r.N]
    : transposed ? [r.K, r.N] : [r.N, r.K];
  const stride = transposed ? [r.N, 1] : [1, shape[0]];
  const axes = which === 'a' ? ['M','K'] : which === 'c' ? ['M','N'] : transposed ? ['K','N'] : ['N','K'];
  const cells = [];
  for (let col = 0; col < shape[1]; col++) for (let row = 0; row < shape[0]; row++) {
    const k = which === 'c' ? selectedK : which === 'b' && transposed ? row : col;
    const cell = {k, active: k === selectedK};
    if (which === 'a' || which === 'c') cell.m = row;
    if (which === 'b') cell.n = transposed ? col : row;
    if (which === 'c') cell.n = col;
    cells.push(cell);
  }
  return {shape, stride, axes, cells};
}

// Per-cell dimensions use atom × warp replication, not permutation_mnk's
// potentially larger tiling period. Fragment rest modes already count repeats.
function gemmResultDimensions(r) {
  const tiled = r.mma.atom.shapeMNK.map((x, i) => x * r.mma.tiled.atomLayoutMNK[i]);
  const gemm = [r.M, r.N, r.K];
  return {tiled, gemm, final: tiled.map((x, i) => x * gemm[i])};
}

function gemmFormatResult(r) {
  const dims = gemmResultDimensions(r);
  const matmul = ([m,n,k]) => `(${m}, ${k}) @ (${k}, ${n}) = (${m}, ${n})`;
  return `Dispatch rule applied: [${r.dispatch}]\n\n` +
    `TiledMMA: ${matmul(dims.tiled)}\n\n` +
    `GEMM: ${matmul(dims.gemm)}\nEach element represents a TiledMMA invocation.\n\n` +
    `Final result: ${matmul(dims.final)}\nEach element is a scalar.`;
}

function gemmSyncD(tabId) {
  const d = document.getElementById(`${tabId}-gemm-d-input`);
  d.disabled = true;
  d.value = document.getElementById(`${tabId}-gemm-c-input`).value;
}

function gemmDispatchRulesMarkup(id) {
  const rules = [
    ['(V) × (V) → (V)', '(V,1,1) × (V,1,1) → (V,1,1)'],
    ['(M) × (N) → (M,N)', '(1,M,1) × (1,N,1) → (1,M,N)'],
    ['(M,K) × (N,K) → (M,N)', '(1,M,K) × (1,N,K) → (1,M,N)'],
    ['(V,M) × (V,N) → (V,M,N)', '(V,M,1) × (V,N,1) → (V,M,N)'],
    ['(V,M,K) × (V,N,K) → (V,M,N)', 'Already in (V,M,K) / (V,N,K) form'],
  ];
  return `<div class="comp-viz-span">
    <h3>Dispatch rules</h3>
    <div class="gemm-dispatch-scroll"><table class="gemm-dispatch-table">
      <thead><tr><th>Rule</th><th>A × B → C/D</th><th>Normalized form</th></tr></thead>
      <tbody>${rules.map((rule, i) => {
        const mark = text => i === 0 ? `<s>${text}</s>` : text;
        return `<tr id="${id}-gemm-rule-${i+1}">
        <th scope="row">${mark(`Dispatch [${i+1}]`)} <span class="gemm-dispatch-applied">Applied</span></th>
        <td><code>${mark(rule[0])}</code></td><td><code>${mark(rule[1])}</code></td>
      </tr>`;
      }).join('')}</tbody>
    </table></div>
    <div class="hint">Use [4] for one K slice, or [5] to reduce over an explicit K mode inside gemm. V is the per-thread value mode for one MMA instruction; its size can differ for A, B and C/D. Rule [1] is rejected by the current CuTeDSL warp backend. Rules [2] and [3] assume V=1 and do not match the warp MMA atoms offered here.</div>
  </div>`;
}

function gemmHighlightDispatch(tabId, dispatch) {
  document.getElementById(`${tabId}-gemm-k-control`).hidden = dispatch !== 3 && dispatch !== 5;
  for (let i = 1; i <= 5; i++) {
    const row = document.getElementById(`${tabId}-gemm-rule-${i}`);
    row.classList.toggle('active', i === dispatch);
    if (i === dispatch) row.setAttribute('aria-current', 'true');
    else row.removeAttribute('aria-current');
  }
}

function generateGemmTabContent(id) {
  return `<div id="${id}-tab-gemm" class="panel">
    <div class="controls">
      <h2>gemm</h2>
      <div class="form-group"><button class="view-toggle" id="${id}-gemm-alt-btn" onclick="toggleGemmAltView('${id}')"><span class="view-toggle-icon">&#8862;</span>Alternative View</button></div>
      <details class="cuo-section" id="${id}-gemm-mma-section" open>
        <summary>1. TiledMMA</summary>
        <div class="cuo-section-body">
          <button class="btn btn-render btn-decode" onclick="gemmDecodePasteboard('${id}')">Decode from Pasteboard</button>
          ${mtmManualConstructorMarkup({id,p:'gemm-mma',render:'renderGemm',errorPrefix:'gemm',atomLayout:'(2,2,1)',perm:'(32,32,16)',k:16})}
          <div id="${id}-gemm-mma-summary" class="comp-result-box visible" style="white-space:pre-wrap"></div>
        </div>
      </details>
      <details class="cuo-section" id="${id}-gemm-fragments-section" open>
        <summary>2. Per-thread RMEM fragments</summary>
        <div class="cuo-section-body">
          ${layoutInputField({id:`${id}-gemm-a-input`,label:'A layout',value:'((2,2,2),4):((1,2,4),16)'})}
          ${layoutInputField({id:`${id}-gemm-b-input`,label:'B layout',value:'((2,2),8):((1,2),8)'})}
          ${layoutInputField({id:`${id}-gemm-c-input`,label:'C layout',value:'((2,2),4,8):((1,2),4,16)',oninput:`gemmSyncD('${id}')`})}
          ${layoutInputField({id:`${id}-gemm-d-input`,label:'D layout (same as C)',value:'((2,2),4,8):((1,2),4,16)',disabled:true})}
        </div>
      </details>
      ${statusDivs(`${id}-gemm`)}
      <div id="${id}-gemm-result" class="comp-result-box" style="white-space:pre-wrap"></div>
      <button class="btn btn-render" onclick="renderGemm('${id}')">Render</button>
      <button class="btn btn-render" id="${id}-gemm-export" onclick="exportGemm('${id}')">Export URL</button>
      <div class="presets"><h3>Presets</h3><div class="preset-list">
        <button class="preset-btn" onclick="setGemm('${id}',GEMM_DEFAULT_MMA,'((2,2,2),4):((1,2,4),16)','((2,2),8):((1,2),8)','((2,2),4,8):((1,2),4,16)','',0)">One K-block: 4 × 8</button>
        <button class="preset-btn" onclick="setGemm('${id}',GEMM_DEFAULT_MMA,'((2,2,2),4,2):((1,2,4),16,8)','((2,2),8,2):((1,2),8,4)','((2,2),4,8):((1,2),4,16)','',0)">Full fragments: 4 × 8 × 2</button>
        <button class="preset-btn" onclick="setGemm('${id}',GEMM_DEFAULT_MMA,'(8,1):(1,0)','(4,1):(1,0)','(4,1,1):(1,0,0)','',0)">Single invocation</button>
        <button class="preset-btn" onclick="setGemm('${id}',GEMM_DEFAULT_MMA,'(8,2):(1,8)','(4,3):(1,4)','(4,2,3):(1,4,8)','',0)">Rectangular GEMM: 2 × 3</button>
      </div></div>
      <div class="hint">Each A/B cell is an operand block for one TiledMMA unit; each C/D cell accumulates the products along K.
      Axes are fragment repetition indices, not matrix coordinates. permutation_mnk controls placement within the matrix.
      This shows coverage, not instruction scheduling.
      Final dimensions count scalar elements; permutation_mnk controls their placement. D follows C. All fragments are RMEM.</div>
    </div>
    <div class="comp-results" id="${id}-gemm-results">
      ${gemmDispatchRulesMarkup(id)}
      <div class="comp-viz-span">
        <label class="gemm-k-control" id="${id}-gemm-k-control" hidden>Highlight K <select id="${id}-gemm-k-input" onchange="renderGemm('${id}')"><option value="0">0</option></select></label>
        <div class="cuo-viz-desc" id="${id}-gemm-desc"></div>
      </div>
      <div class="mma-group">
        <div class="mma-q-spacer"></div>
        ${['a','b','c'].map(which => `<div class="comp-viz-item" data-q="${which}">
          <div class="comp-viz-header"><span class="comp-viz-title"><span id="${id}-gemm-${which}-title">${which.toUpperCase()}</span> <span class="memory-space-badge" data-space="RMEM">RMEM</span></span></div>
          <div class="viz-box"><div id="${id}-gemm-${which === 'c' ? 'tile' : which}-svg"></div></div>
        </div>`).join('')}
      </div>
    </div>
  </div>`;
}

function gemmMmaSummary(mma) {
  const p = mma.inputs;
  return `make_tiled_mma(${MMA_WARP_OPS[p[0]].label}, ${p[4]}, ${p[5]})\n` +
    `Atom MNK = (${mma.atom.shapeMNK.join(',')})\n` +
    `A/B dtype = ${p[1] === 'na' ? 'tfloat32_t' : p[1]}; C/D dtype = ${p[2] === 'na' ? 'float' : p[2]}\n` +
    `Thr Layout VMNK = ${formatLayoutStr(mma.tiled.thrVmnk.shape, mma.tiled.thrVmnk.stride)}\n` +
    `Base warp replication MNK = (${mma.atom.shapeMNK.map((x,i) => x * mma.tiled.atomLayoutMNK[i]).join(',')})\n` +
    `Permutation tile MNK = (${mma.tiled.tileMNK.join(',')})\n${mma.tiled.warps} warps, ${mma.tiled.threads} threads`;
}

function renderGemm(tabId) {
  const p = `${tabId}-gemm`;
  const state = gemmState[tabId] || (gemmState[tabId] = {mmaInputs: GEMM_DEFAULT_MMA.slice()});
  showErr(`${p}-error`, '');
  gemmSyncD(tabId);
  gemmHighlightDispatch(tabId, null);
  const resultEl = document.getElementById(`${p}-result`);
  const hosts = {a: `${p}-a-svg`, b: `${p}-b-svg`, c: `${p}-tile-svg`};
  try {
    state.mmaInputs = mtmReadConstructorControls(tabId, 'gemm-mma');
    document.getElementById(`${p}-mma-summary`).textContent = gemmMmaSummary(mtmDecodeConstructor(state.mmaInputs));
    const get = x => document.getElementById(`${p}-${x}-input`).value;
    const r = gemmCompute(state.mmaInputs, get('a'), get('b'), get('c'), get('c'));
    const select = document.getElementById(`${p}-k-input`);
    const requested = Number(select.value);
    select.innerHTML = Array.from({length:r.K}, (_, i) => `<option value="${i}">${i}</option>`).join('');
    const k = Number.isSafeInteger(requested) && requested >= 0 && requested < r.K ? requested : 0;
    select.value = String(k);
    state.result = r;
    gemmHighlightDispatch(tabId, r.dispatch);
    // All M/N/K modes are explicitly drawn. V is one atom's operand block.
    showWarn(`${p}-warning`, '');
    resultEl.textContent = gemmFormatResult(r);
    resultEl.classList.add('visible');
    const altView = !!state.altView;
    document.getElementById(`${p}-results`).classList.toggle('mma-alt', altView);
    document.getElementById(`${p}-alt-btn`).classList.toggle('active', altView);
    document.getElementById(`${p}-desc`).textContent = `M = ${r.M}, N = ${r.N}, K = ${r.K}. ` +
      (r.dispatch === 3 || r.dispatch === 5
        ? `Highlighted K = ${k}: A[:,${k}] × ${altView ? `B[${k},:]` : `B[:,${k}]`} updates every C/D cell. Each C/D cell accumulates ${r.K} products.`
        : 'Each C/D cell receives one A × B product; this dispatch has no explicit K mode.');
    for (const which of ['a','b','c']) {
      const grid = gemmCoverageGrid(r, which, altView, k);
      document.getElementById(`${p}-${which}-title`).textContent = `${which === 'c' ? 'C/D' : which.toUpperCase()} — ${grid.axes.join(' × ')} (${grid.shape.join(' × ')}); rows ${grid.axes[0]}, columns ${grid.axes[1]}`;
      document.getElementById(hosts[which]).innerHTML = buildColoredLayoutSVG(grid.shape, grid.stride, new Set(['value']),
        (row, col) => {
          const cell = grid.cells[row + col * grid.shape[0]];
          return {bg: cell.active ? colorHighlight(k) : '#f3f4f6', text: ['TiledMMA']};
        });
      applyZoomState(hosts[which]);
    }
    updateOuterTabLabel(tabId, `gemm:${r.M}×${r.N}×${r.K}`);
  } catch (e) {
    showErr(`${p}-error`, e.message);
    resultEl.classList.remove('visible');
    for (const hostId of Object.values(hosts)) document.getElementById(hostId).innerHTML = '';
    document.getElementById(`${p}-desc`).textContent = '';
    delete state.result;
  }
}

function gemmImportMma(tabId, text) {
  const parsed = parsePastedKeyParam(text);
  if (!parsed || parsed.feature !== 'make_tiled_mma') throw new Error('Paste a make_tiled_mma URL or its query string.');
  mtmWriteConstructorControls(tabId, 'gemm-mma', parsed.inputs); // Validate before changing controls.
  gemmState[tabId] = {mmaInputs: parsed.inputs.slice(), altView: !!(gemmState[tabId] || {}).altView};
  renderGemm(tabId);
}

async function gemmDecodePasteboard(tabId) {
  const text = await readPasteboardText('Paste the make_tiled_mma URL or its query string:');
  if (text === null || text === undefined) return;
  try { gemmImportMma(tabId, text); }
  catch (e) { showErr(`${tabId}-gemm-error`, e.message); }
}

function setGemm(tabId, mmaInputs, a, b, c, d, k, altView) {
  mtmWriteConstructorControls(tabId, 'gemm-mma', mmaInputs);
  gemmState[tabId] = {mmaInputs: mmaInputs.slice(), altView: !!altView};
  for (const [key, value] of Object.entries({a,b,c})) document.getElementById(`${tabId}-gemm-${key}-input`).value = value || '';
  renderGemm(tabId); // Populate K options before assigning an imported value.
  document.getElementById(`${tabId}-gemm-k-input`).value = String(k || 0);
  renderGemm(tabId);
}

function exportGemm(tabId) {
  const state = gemmState[tabId] || {mmaInputs: GEMM_DEFAULT_MMA};
  exportURL(`${tabId}-gemm-export`, 'gemm', ...mtmReadConstructorControls(tabId, 'gemm-mma'),
    ...['a','b','c'].map(x => document.getElementById(`${tabId}-gemm-${x}-input`).value),
    'same',
    document.getElementById(`${tabId}-gemm-k-input`).value, state.altView ? 'mk_kn_mn' : 'mk_nk_mn');
}


function toggleGemmAltView(tabId) {
  const state = gemmState[tabId] || (gemmState[tabId] = {mmaInputs: GEMM_DEFAULT_MMA.slice()});
  state.altView = !state.altView;
  renderGemm(tabId);
}
