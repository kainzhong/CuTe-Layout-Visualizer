// make_fragment_A/B/C: the fragment made from an ALREADY partitioned operand,
// for every MMA Op the MMA tabs offer. RMEM operands (warp MMA, WGMMA C and
// register A) get a register tensor; WGMMA SMEM operands get a descriptor
// tensor. The layout math lives in mmaMakeFragment / pabcMakeFragment so the
// partition tab and this tab cannot disagree.

const MFRAG_MAX_REGS = 512;
const MFRAG_DEFAULT_PARTITIONS = {
  A: '((2,2,2),2,3):((1,384,8),1536,16)',
  B: '((2,2),2,3):((1,8),768,16)',
  C: '((2,2),2,2):((1,256),1024,16)',
};
const mfragState = {};

/** Register word -> ordered scalar slots -> positions in the input partition.
 * No values are read: a layout tells us addresses, not runtime contents.
 * `atomInputs` are the make_mma_atom URL fields: [op, ab, acc, K, ...tail]. */
function mfragCompute(partition, which, atomInputs) {
  if (!PABC_OPERAND[which]) throw new Error(`Unknown MMA operand ${which}.`);
  const { atom } = mmaDecodeAtomInputs(atomInputs);
  const opKey = atom.opKey, k = atom.k;
  if (partition.rank() < 3)
    throw new Error(`make_fragment_${which} expects an already-partitioned layout of rank at least 3 (value mode plus two Rest modes).`);
  const want = atom[which].numVal;
  if (product(partition.shape[0]) !== want)
    throw new Error(`Partition mode 0 has ${product(partition.shape[0])} values; ${atom.op.label} ` +
      `${atom.shapeMNK.join('x')} expects ${want} for operand ${which}. Pass the result of partition_${which}.`);
  const dtype = atom.dtypes[which];
  if (atom[which].space === 'SMEM') {
    // A descriptor tensor: one 64-bit matrix descriptor per Rest position,
    // offsets in 128-bit units. CuTeDSL requires an SMEM source tensor.
    const fragment = mmaMakeFragment(partition, which, atom);
    const count = product(fragment.shape);
    if (count > MAX_CELLS) throw new Error(`This fragment has ${count} descriptors; the view supports at most ${MAX_CELLS}.`);
    return { partition, fragment, descriptor: true, registers: [], dtype, bits: mmaDtypeBits(dtype),
             which, opKey, k, atom };
  }
  const bits = mmaDtypeBits(dtype);
  if (!bits || bits > 32 || 32 % bits)
    throw new Error(`Cannot show ${dtype} as 32-bit MMA registers.`);
  const pack = 32 / bits;
  const fragment = pabcMakeFragment(partition, which);
  const regCount = Math.ceil(fragment.cosize() / pack);
  if (regCount > MFRAG_MAX_REGS)
    throw new Error(`This fragment needs ${regCount} register words; the view supports at most ${MFRAG_MAX_REGS}.`);
  const total = product(fragment.shape);
  if (total > MFRAG_MAX_REGS * 4)
    throw new Error(`This partition contains ${total} value positions; the view supports at most ${MFRAG_MAX_REGS * 4}.`);
  const registers = Array.from({ length: regCount },
    (_, index) => ({ index, slots: Array(pack).fill(null), group: 0 }));
  for (let i = 0; i < total; i++) {
    const coord = unflatten(i, fragment.shape);
    const dst = crd2idx(coord, fragment.shape, fragment.stride);
    const reg = registers[Math.floor(dst / pack)];
    const slot = dst % pack;
    const rest = flatten(coord.slice(1));
    const restShape = flatten(fragment.shape.slice(1));
    let group = 0, scale = 1;
    for (let j = 0; j < rest.length; j++) { group += rest[j] * scale; scale *= restShape[j]; }
    if (reg.slots[slot] === null) reg.slots[slot] = { source: crd2idx(coord, partition.shape, partition.stride), coord };
    reg.group = group;
  }
  return { partition, fragment, descriptor: false, registers, dtype, bits, pack, which, opKey, k, atom };
}

/** One cell per descriptor: rows are Rest mode 1, columns flatten later modes.
 *  Each cell records the descriptor offset (128-bit units) and the SMEM element
 *  where its atom block starts in the partitioned source. */
function mfragDescriptorGrid(result) {
  const { partition, fragment } = result;
  const rows = product(fragment.shape[1]);
  const tailShape = fragment.shape.slice(2);
  const cols = product(tailShape);
  const cells = [];
  for (let n = 0; n < cols; n++) {
    const tail = unflatten(n, tailShape);
    for (let m = 0; m < rows; m++) {
      const coord = [0, unflatten(m, fragment.shape[1]), ...tail];
      const pcoord = [unflatten(0, partition.shape[0]), unflatten(m, partition.shape[1]), ...tail];
      cells.push({ rest: m + n * rows, desc: crd2idx(coord, fragment.shape, fragment.stride),
                   source: crd2idx(pcoord, partition.shape, partition.stride) });
    }
  }
  return { rows, cols, cells };
}

function mfragBuildDescriptorSVG(result) {
  const grid = mfragDescriptorGrid(result);
  return buildColoredLayoutSVG([grid.rows, grid.cols], [1, grid.rows], '', (m, n) => {
    const cell = grid.cells[m + n * grid.rows];
    return { bg: colorHighlight(cell.rest), stroke: '#1e3a5f', sw: 1.5,
             text: [`(${m},${n})`, `desc +${cell.desc}`, `@${cell.source}`] };
  });
}

/** One grid cell per scalar value in the partition domain. Keep the atom
 * value mode and first Rest mode on rows; later Rest modes run across columns. */
function mfragValueGrid(result) {
  const { partition, fragment, pack } = result;
  const atomValues = product(fragment.shape[0]);
  const rest0 = product(fragment.shape[1]);
  const rest1 = product(fragment.shape[2]);
  const rows = atomValues * rest0;
  const cols = product(fragment.shape.slice(2));
  const cells = [];
  for (let n = 0; n < cols; n++) {
    const tail = unflatten(n, fragment.shape.slice(2));
    for (let m = 0; m < rows; m++) {
      const value = m % atomValues;
      const firstRest = Math.floor(m / atomValues);
      const coord = [unflatten(value, fragment.shape[0]),
        unflatten(firstRest, fragment.shape[1]), ...tail];
      const scalar = crd2idx(coord, fragment.shape, fragment.stride);
      cells.push({ register: Math.floor(scalar / pack),
        source: crd2idx(coord, partition.shape, partition.stride),
        value, rest0: firstRest, rest1: n % rest1,
        rest: firstRest + (n % rest1) * rest0 });
    }
  }
  return { rows, cols, cells, atomValues, rest0, rest1 };
}

/** The same Rest-block boundaries as partition_ABC's level-2 grid. Draw each
 * boundary once across the value cells so neighboring strokes do not double. */
function mfragRestOverlay(grid, { cs, margin }) {
  const line = (x1, y1, x2, y2) =>
    `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#1e3a5f" stroke-width="2"/>`;
  let out = '';
  for (let i = 0; i <= grid.rest0; i++) {
    const y = margin + i * grid.atomValues * cs;
    out += line(margin, y, margin + grid.cols * cs, y);
  }
  for (let j = 0; j <= grid.cols; j++) {
    const x = margin + j * cs;
    out += line(x, margin, x, margin + grid.rows * cs);
  }
  return out;
}

/** Shared register-fragment drawing: both make_fragment and retile label
 * logical scalar slots with exactly the same T/V, R#, and Rest colors. */
function mfragBuildValueSVG(result, thread) {
  const grid = mfragValueGrid(result);
  return buildColoredLayoutSVG([grid.rows, grid.cols], [1, grid.rows], '', (m, n) => {
    const cell = grid.cells[m + n * grid.rows];
    return { bg: colorHighlight(cell.rest), text: [
      `(${cell.rest0},${cell.rest1})`, `T${thread}V${cell.value}`, `R${cell.register}`] };
  }, { overlay: ctx => mfragRestOverlay(grid, ctx) });
}

function generateMakeFragmentABCTabContent(id) {
  return `
    <div id="${id}-tab-make_fragment_abc" class="panel">
      <div class="controls">
        <h2>make_fragment_A / B / C</h2>
        <div class="form-group">
          <label>Operand</label>
          <div class="seg-control" id="${id}-mfrag-operand-btns">
            <button class="mode-btn active" onclick="setMfragOperand('${id}','A')">A</button>
            <button class="mode-btn" onclick="setMfragOperand('${id}','B')">B</button>
            <button class="mode-btn" onclick="setMfragOperand('${id}','C')">C</button>
          </div>
        </div>
        <details class="cuo-section" open>
          <summary>1. MMAAtom</summary>
          <div class="cuo-section-body">
            ${mmaAtomEditorMarkup({id,p:'mfrag',render:'renderMakeFragmentABC',setOp:'setMfragOp',k:16})}
            <div id="${id}-mfrag-op-params" class="cuo-result"></div>
          </div>
        </details>
        <details class="cuo-section" open>
          <summary>2. Already partitioned operand</summary>
          <div class="cuo-section-body">
            ${layoutInputField({ id: `${id}-mfrag-partition-input`, label: 'partition_A/B/C result layout',
              value: MFRAG_DEFAULT_PARTITIONS.A,
              hint: 'The exact layout returned by get_slice(t).partition_A/B/C; mode 0 is the atom value mode.' })}
            <div class="form-group"><label>Thread ID for T/V labels</label>
              <input type="text" id="${id}-mfrag-thr-input" value="0" placeholder="0" oninput="renderMakeFragmentABC('${id}')">
              <div class="hint-inline">The partitioned layout is the same for every thread and does not encode its ID. This field labels the selected thread; partition_A/B/C passes its thread ID here.</div>
            </div>
          </div>
        </details>
        ${statusDivs(`${id}-mfrag`)}
        <button class="btn btn-render" onclick="renderMakeFragmentABC('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-mfrag-export" onclick="exportMfrag('${id}')">Export URL</button>
        <div id="${id}-mfrag-result" class="cuo-result"></div>
        <div class="hint">Each outlined, colored block is one Rest position from partition_A/B/C's second grid. Its smaller cells show that thread's <code>T#V#</code> values and their 32-bit register <code>R#</code>. Repeated register labels mean values share a word. These are logical labels, not runtime numerical values. WGMMA C and register A are register fragments too; a WGMMA shared-memory A/B operand instead becomes a tensor of descriptors, one per Rest position.</div>
        <div class="presets"><h3>Presets</h3><div class="preset-list">
          <button class="preset-btn" onclick="setMfrag('${id}','A','f16bf16','half_t','float',16,'((2,2,2),2,3):((1,384,8),1536,16)')">A — row-major source</button>
          <button class="preset-btn" onclick="setMfrag('${id}','A','f16bf16','half_t','float',16,'((2,2,2),2,3):((64,8,512),32,1024)')">A — column-major source</button>
          <button class="preset-btn" onclick="setMfrag('${id}','B','f16bf16','half_t','float',16,'((2,2),2,3):((1,8),768,16)')">B — row-major source</button>
          <button class="preset-btn" onclick="setMfrag('${id}','C','f16bf16','half_t','float',16,'((2,2),2,2):((1,256),1024,16)')">C — accumulator</button>
          <button class="preset-btn" onclick="setMfrag('${id}','C','wg_f16bf16','half_t','float',16,'((2,2,8),2,2):((128,8,1024),8192,64)',0,64,'SMEM','K','K','half_t')">WGMMA C — 64×64 accumulator, 128 threads</button>
          <button class="preset-btn" onclick="setMfrag('${id}','A','wg_f16bf16','half_t','float',16,'((2,2,2),2,4):((128,8,1024),8192,16)',0,64,'RMEM','K','K','half_t')">WGMMA register A (RS)</button>
          <button class="preset-btn" onclick="setMfrag('${id}','A','wg_f16bf16','half_t','float',16,'((64,16),2,4):((64,1),4096,16)',0,64,'SMEM','K','K','half_t')">WGMMA SMEM A — descriptor tensor</button>
        </div></div>
      </div>
      <div class="comp-results" style="grid-template-columns:1fr">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-title"><span class="comp-viz-label" id="${id}-mfrag-title">Scalar values → registers</span>${memorySpaceBadge(`${id}-mfrag-space`, 'RMEM')}</span>
            <button class="mode-btn" id="${id}-mfrag-svg-zoom" onclick="toggleZoom('${id}-mfrag-svg')">Zoom in</button>
          </div>
          <div class="cuo-viz-desc" id="${id}-mfrag-desc"></div>
          <div class="viz-box"><div id="${id}-mfrag-svg"></div></div>
        </div>
      </div>
    </div>`;
}

function renderMakeFragmentABC(tabId) {
  showErr(`${tabId}-mfrag-error`, '');
  showWarn(`${tabId}-mfrag-warning`, '');
  const host = document.getElementById(`${tabId}-mfrag-svg`);
  if (host) host.innerHTML = '';
  document.getElementById(`${tabId}-mfrag-result`).innerHTML = '';
  try {
    const which = (mfragState[tabId] || {}).which || 'A';
    const opKey = document.getElementById(`${tabId}-mfrag-op-input`).value;
    mmaSyncControls(tabId, opKey, 'mfrag');
    const atomInputs = mmaReadAtomInputs(tabId, 'mfrag');
    const { atom, op } = mmaDecodeAtomInputs(atomInputs);
    mmaRenderOpParams(tabId, op, atom.k, atom.dtypes.A, atom.dtypes.C, 'mfrag', atom);
    const raw = document.getElementById(`${tabId}-mfrag-partition-input`).value;
    const thrRaw = document.getElementById(`${tabId}-mfrag-thr-input`).value.trim();
    if (!/^\d+$/.test(thrRaw)) throw new Error('Thread ID must be a nonnegative whole number.');
    const thread = Number(thrRaw);
    if (!Number.isSafeInteger(thread)) throw new Error('Thread ID is too large.');
    const p = parseLayout(raw);
    const s = mfragCompute(new Layout(p.shape, p.stride), which, atomInputs);
    mfragState[tabId] = { ...(mfragState[tabId] || {}), ...s, which, thread, atomInputs };
    const str = formatLayoutStr(s.fragment.shape, s.fragment.stride);
    const axes = PABC_OPERAND[which].axes;
    setMemorySpaceBadge(`${tabId}-mfrag-space`, s.descriptor ? 'SMEM' : 'RMEM');
    if (s.descriptor) {
      const n = product(s.fragment.shape);
      document.getElementById(`${tabId}-mfrag-result`).innerHTML =
        `<div class="cuo-result-line">make_fragment_${which} = <b>${str}</b></div>` +
        `<div class="cuo-result-line">${n} WGMMA shared-memory descriptor${n === 1 ? '' : 's'}; ` +
        `offsets in 128-bit units (${128 / s.bits} ${s.dtype} elements). The source must be an SMEM tensor.</div>`;
      document.getElementById(`${tabId}-mfrag-title`).textContent =
        `${which} — ${n} descriptor${n === 1 ? '' : 's'} over SMEM`;
      document.getElementById(`${tabId}-mfrag-desc`).innerHTML =
        `<code>${str}</code> &mdash; one descriptor per (${axes[0]}, ${axes[1]}) Rest position, matching ` +
        `partition_${which}'s second grid. Each addresses the whole ${s.atom.shapeMNK.join('x')} atom's ${which} ` +
        `block for all 128 threads; <code>desc +</code> is its offset and <code>@</code> the block's first SMEM element.`;
      host.innerHTML = mfragBuildDescriptorSVG(s);
      applyZoomState(`${tabId}-mfrag-svg`);
      updateOuterTabLabel(tabId, `make_fragment_${which}:${str}`);
      return;
    }
    document.getElementById(`${tabId}-mfrag-result`).innerHTML =
      `<div class="cuo-result-line">make_fragment_${which} = <b>${str}</b></div>` +
      `<div class="cuo-result-line">${s.registers.length} × 32-bit registers; ${s.pack} ${s.dtype} value${s.pack === 1 ? '' : 's'} per register</div>`;
    document.getElementById(`${tabId}-mfrag-title`).textContent =
      `${which} — ${product(s.fragment.shape)} values in ${s.registers.length} registers`;
    document.getElementById(`${tabId}-mfrag-desc`).innerHTML =
      `<code>${str}</code> &mdash; each colored block is one (${axes[0]}, ${axes[1]}) Rest position, matching partition_${which}'s second grid. Smaller cells show its atom values${s.fragment.rank() > 3 ? '; later Rest modes repeat the grid' : ''}.`;
    host.innerHTML = mfragBuildValueSVG(s, thread);
    applyZoomState(`${tabId}-mfrag-svg`);
    updateOuterTabLabel(tabId, `make_fragment_${which}:${str}`);
  } catch (e) {
    showErr(`${tabId}-mfrag-error`, e.message);
  }
}

function setMfragOperand(tabId, which) {
  const prev = mfragState[tabId] || {};
  const saved = { ...(prev.inputs || {}) };
  saved[prev.which || 'A'] = document.getElementById(`${tabId}-mfrag-partition-input`).value;
  mfragState[tabId] = { ...prev, which, inputs: saved };
  document.getElementById(`${tabId}-mfrag-partition-input`).value = saved[which] || MFRAG_DEFAULT_PARTITIONS[which];
  const group = document.getElementById(`${tabId}-mfrag-operand-btns`);
  if (group) group.querySelectorAll('.mode-btn').forEach(b =>
    b.classList.toggle('active', b.textContent.trim() === which));
  renderMakeFragmentABC(tabId);
}

function setMfragOp(tabId) {
  mmaSyncControls(tabId, document.getElementById(`${tabId}-mfrag-op-input`).value, 'mfrag');
  renderMakeFragmentABC(tabId);
}

/** URL field order; `...tail` is the atom's Op-specific tail (WGMMA: N,
 *  A source, A/B major, B dtype), validated before any control changes. */
function setMfrag(tabId, which, opKey, ab, acc, k, partition, thread = 0, ...tail) {
  const atomInputs = [opKey, ab || 'na', acc || 'na', String(k), ...(MMA_WARPGROUP_OPS[opKey] ? tail.slice(0, 5) : [])];
  mmaAssignAtomInputs(tabId, 'mfrag', atomInputs);
  mfragState[tabId] = { ...(mfragState[tabId] || {}), which };
  document.getElementById(`${tabId}-mfrag-partition-input`).value = partition;
  document.getElementById(`${tabId}-mfrag-thr-input`).value = String(thread);
  const group = document.getElementById(`${tabId}-mfrag-operand-btns`);
  if (group) group.querySelectorAll('.mode-btn').forEach(b =>
    b.classList.toggle('active', b.textContent.trim() === which));
  renderMakeFragmentABC(tabId);
}

/** make_fragment_abc URL fields -> make_mma_atom fields. */
function mfragAtomInputsFromURL(a) {
  // Legacy links spelled a fixed accumulator as `na`; it resolves to float.
  const op = MMA_ALL_OPS[a[1]], acc = a[3] === 'na' && op && op.acc ? 'float' : a[3];
  return [a[1], a[2], acc, a[4], ...(MMA_WARPGROUP_OPS[a[1]] ? a.slice(7, 12) : [])];
}

function exportMfrag(tabId) {
  mmaSyncControls(tabId, document.getElementById(`${tabId}-mfrag-op-input`).value, 'mfrag');
  const atomInputs = mmaReadAtomInputs(tabId, 'mfrag');
  exportURL(`${tabId}-mfrag-export`, 'make_fragment_abc',
    (mfragState[tabId] || {}).which || 'A', ...atomInputs.slice(0, 4),
    document.getElementById(`${tabId}-mfrag-partition-input`).value,
    document.getElementById(`${tabId}-mfrag-thr-input`).value, ...atomInputs.slice(4));
}
