// make_tmem_copy tab (Copy scope): `tcgen05.make_tmem_copy(atom, tmem_tensor)`,
// the TiledCopy a Blackwell epilogue uses to move an accumulator between TMEM
// and registers.
//
// CuTeDSL implements it in MLIR; the C++ it mirrors is short
// (cute/atom/copy_traits_sm100.hpp, make_tmem_copy):
//
//   atom_t = (32,4):(0, 32·DP)                 4 warps; a warp's lanes share one
//                                              address, each warp moves 32 lanes
//   atom_v = coalesce(upcast<bits>(ValID))     the Op's (bits per lane × repeat,
//                                              lanes):(1, DP_b) in elements
//   make_cotiled_copy(atom, (atom_t, atom_v), tmem.layout())
//
// DP = 2^21 / bits is one TMEM lane in element units (the lane sits in the
// address's upper 16 bits of 32-bit columns). The (32,4):(0, 32·DP) thread map
// IS the lane-quarter rule: warp w of the four reaches lanes 32w..32w+31.
//
// tmcComputeCopy is the DOM-free port, diffed against CuTeDSL in tests/run.js
// (section `tmem_copy`): the tiler, layout_tv, both side maps, and every
// thread's register-side partition of the identity tensor.

/** Bits one lane carries per repetition, by TMEM shape (the Op's ValID). The
 *  16x32bx2 shape's two 32-bit halves sit in adjacent columns, hence 64. */
const TMC_SHAPE_BITS = { '32x32b': 32, '16x64b': 64, '16x128b': 128, '16x256b': 256, '16x32bx2': 64 };
const TMC_OPS = Object.keys(MCA_OPS).filter(k => MCA_OPS[k].kind === 'tmem');

/** TMEM::DP<T>: one TMEM lane in element units. */
function tmcLaneStride(bits) { return (1 << 21) / bits; }

/** The tcgen05 Copy_Atom, in the form the shared tidfrg derivation takes:
 *  SRC / DST in elements, the TMEM side as CuTe's reference layout (true for
 *  loads and stores alike: a store's RefLayout is its load twin's). */
function tmcAtom(opKey, dtype, repeat) {
  const op = MCA_OPS[opKey];
  if (!op || op.kind !== 'tmem') throw new Error(`"${opKey}" is not a tcgen05 TMEM load or store Op.`);
  const bits = DTYPE_BITS[dtype];
  if (!bits || bits > 32)
    throw new Error(`make_tmem_copy here takes 8-, 16- or 32-bit elements; ${dtype} is ${bits || 'not'} bits.`);
  const a = mcaTmemAtom(op.tmem, bits, Number(repeat), op.store);
  const src = new Layout(a.src.shape, a.src.stride), dst = new Layout(a.dst.shape, a.dst.stride);
  const ref = a.tmemSide === 'src' ? src : dst;
  return { opKey, op, a, bits, dtype, repeat: Number(repeat), store: op.store, src, dst, ref,
           threads: 32, values: product(ref.shape[1]), tmemSide: a.tmemSide, regSide: a.regSide,
           label: `${op.label}(x${repeat})` };
}

/** DOM-free make_tmem_copy over a TMEM tensor layout. Returns the TiledCopy's
 *  layouts plus, for the picture, which register of which thread holds every
 *  tensor element, and which warps are mapped outside their lane quarter. */
function tmcComputeCopy(atom, data) {
  const flat = product_each(data.shape);
  if (flat.length !== 2)
    throw new Error(`The TMEM tensor has rank ${flat.length}; this tab draws a (rows, columns) tensor, so ` +
      `it takes rank 2. Slice the accumulator to one tile first, as a kernel does before make_tmem_copy.`);
  const DP = tmcLaneStride(atom.bits);
  const lanes = MCA_TMEM_SPECS[atom.op.tmem].lanes;
  const perLane = TMC_SHAPE_BITS[atom.op.tmem] * atom.repeat / atom.bits;
  const atom_t = new Layout([32, 4], [0, 32 * DP]);
  const atom_v = coalesce(new Layout([perLane, lanes], [1, DP]));
  const atom_tv = make_layout(atom_t, atom_v);

  // make_cotiled_copy (copy_atom.hpp:528). CuTe appends 1:0 to the inverse so
  // addresses past the data map to stride 0; this port's composition
  // coalesces its left side, which would drop a size-1 mode, so the appended
  // stride-0 mode gets an extent that covers the overflow instead.
  const oob = (L, reach) => make_layout(L, new Layout(Math.ceil(reach / L.size()) + 1, 0));
  const linv = left_inverse(data);
  const tvData = composition(oob(linv, atom_tv.cosize()), atom_tv);        // (tid,vid) -> data index
  const back = coalesce(composition(oob(data, tvData.mode(1).cosize()), tvData.mode(1)));
  const want = coalesce(atom_tv.mode(1));
  if (formatLayoutStr(back.shape, back.stride) !== formatLayoutStr(want.shape, want.stride)) {
    const hint = lanes === 32
      ? ` ${atom.op.label} needs 32 consecutive TMEM lanes in each warp's quarter; an accumulator with ` +
        `fewer rows per quarter (e.g. M = 64: 16 rows in each) needs a 16-lane shape — 16x64b, 16x128b, ` +
        `16x256b or 16x32bx2.`
      : ` Check that each lane quarter holds at least ${lanes} consecutive rows and ${perLane} consecutive ` +
        `${atom.dtype} columns.`;
    throw new Error(`One ${atom.label} moves ${lanes} lanes × ${perLane} ${atom.dtype} per warp, and those ` +
      `TMEM addresses are not all in this tensor — CuTe's assert "The memory pointed to by AtomTVLayout ` +
      `does not exist in the DataLayout" (copy_atom.hpp:543).${hint}`);
  }
  const tiler = flat.map((_, i) =>
    filter(composition(new Layout(flat, flat.map((_, j) => (j === i ? 1 : 0))), tvData)));
  const tile2data = composition(new Layout(flat), tiler);
  const tv = composition(left_inverse(tile2data), tvData);
  const tile = tiler.map(t => t.size());
  const src = mcabSideLayout(tv, tile, atom, 'src');
  const dst = mcabSideLayout(tv, tile, atom, 'dst');

  // partition_S/D over the whole tensor: tidfrg of zipped_divide(tensor, tiler)
  // gives (thread, (FrgV, FrgX), Rest) -> data index, exactly as partition_sd.
  const reg = atom.regSide === 'src' ? atom.src : atom.dst;
  const tf = psdTile2ThrFrg(zipped_divide(new Layout(flat), tiler), tv, atom.threads, atom.values,
                            composition(right_inverse(atom.ref), reg)).tidfrg;
  const frag = make_layout(tf.mode(1), tf.mode(2));
  const fragSize = frag.size(), threads = product(tv.shape[0]), M = flat[0];
  const cells = Array.from({ length: flat[0] * flat[1] }, () => []);
  const perThread = [], outside = new Set();
  let maxLane = 0;
  for (let t = 0; t < threads; t++) {
    const base = tf.mode(0).call(t), coords = [];
    for (let v = 0; v < fragSize; v++) {
      const idx = base + frag.call(v), m = idx % M, n = Math.floor(idx / M);
      const lane = Math.floor(data.call(idx) / DP);
      maxLane = Math.max(maxLane, lane);
      // The hardware: warp w (mod 4) reaches only lanes 32(w%4)..+31.
      const legal = Math.floor(lane / 32) % 4 === Math.floor(t / 32) % 4;
      if (!legal) outside.add(Math.floor(t / 32));
      coords.push([m, n]);
      cells[idx].push({ t, v, legal });
    }
    perThread.push(coords);
  }
  if (maxLane >= 128)
    throw new Error(`The tensor reaches TMEM lane ${maxLane}, but TMEM has 128 lanes. A lane is ` +
      `${DP} ${atom.dtype} address units (2^21 / ${atom.bits}); check the row stride.`);
  return { DP, flat, tiler, tile, tv, src, dst, threads, fragSize, perThread, cells,
           outside: [...outside].sort((a, b) => a - b), lanes, perLane };
}

/** Advice the computation cannot throw on: a row stride that is past one
 *  lane's columns but not a whole lane is almost always the 32-bit stride
 *  pasted onto a narrower type (CuTe accepts it and builds a strange tiler). */
function tmcStrideWarning(data, atom) {
  const DP = tmcLaneStride(atom.bits), cols = 512 * 32 / atom.bits;
  const odd = flatten(data.stride).filter(s => s > cols && s % DP);
  return odd.length
    ? `Stride ${odd[0]} is past the ${cols} ${atom.dtype} columns of one TMEM lane but is not a whole ` +
      `number of lanes. For ${atom.dtype} one lane is ${DP} address units (2^21 / ${atom.bits}), so row r ` +
      `of an accumulator sits at r·${DP}; ${odd[0]} looks like the ${odd[0] === 65536 ? '32-bit ' : ''}stride ` +
      `of a different element type. CuTe accepts it, but the rows are not TMEM lanes.`
    : '';
}

function generateMakeTmemCopyTabContent(id) {
  const opOptions = TMC_OPS.map(k => `<option value="${k}">${MCA_OPS[k].label}</option>`).join('');
  return `
    <!-- make_tmem_copy panel -->
    <div id="${id}-tab-make_tmem_copy" class="panel">
      <div class="controls">
        <h2>make_tmem_copy</h2>
        <details class="cuo-section" open>
          <summary>1. CopyAtom (TMEM)</summary>
          <div class="cuo-section-body">
            <button class="btn btn-render btn-decode" id="${id}-tmc-decode-atom" onclick="tmcDecodeAtomPasteboard('${id}')">Decode from Pasteboard</button>
            <details class="cuo-section" id="${id}-tmc-atom-manual-construction"><summary>Manual construction</summary><div class="cuo-section-body">
              <div class="form-group"><label>Op</label>
                <select id="${id}-tmc-op-input" onchange="tmcSyncControls('${id}')">${opOptions}</select></div>
              <div class="form-group"><label>repeat<span style="color:#6b7280;font-weight:normal">&nbsp;&mdash; the .x{N} qualifier</span></label>
                <select id="${id}-tmc-rep-input">${MCA_TMEM_REPEATS.map(r => `<option value="${r}"${r === 8 ? ' selected' : ''}>x${r}</option>`).join('')}</select></div>
              <div class="form-group"><label>dtype</label>
                <select id="${id}-tmc-dtype-input">${dtypeOptions('float')}</select></div>
            </div></details>
            <div id="${id}-tmc-atom-result" class="cuo-result"></div>
          </div>
        </details>
        <details class="cuo-section" open>
          <summary>2. TMEM tensor</summary>
          <div class="cuo-section-body">
            ${layoutInputField({
              id: `${id}-tmc-tensor-input`,
              label: 'tmem_tensor &mdash; (rows, columns) &rarr; TMEM address',
              hint: 'a row is a lane: stride 2^21 / bits (65536 for 32-bit, 131072 for 16-bit, 262144 for 8-bit)',
              value: '(128,32):(65536,1)' })}
          </div>
        </details>
        <div class="form-group">
          <label>Highlight thread (empty = show all threads)</label>
          <input type="text" id="${id}-tmc-highlight-tid" value="" placeholder="e.g. 33" oninput="renderMakeTmemCopy('${id}')">
        </div>
        ${statusDivs(`${id}-tmc`)}
        <button class="btn btn-render" onclick="renderMakeTmemCopy('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-tmc-export" onclick="exportTMC('${id}')">Export URL</button>
        <div id="${id}-tmc-result" class="cuo-result"></div>
        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setTMC('${id}','tmem_ld32x32b',8,'float','(128,32):(65536,1)')">Ld32x32b.x8 &mdash; one accumulator row per thread</button>
            <button class="preset-btn" onclick="setTMC('${id}','tmem_ld16x256b',1,'float','(128,32):(65536,1)')">Ld16x256b.x1 &mdash; the m16n8 fragment arrangement</button>
            <button class="preset-btn" onclick="setTMC('${id}','tmem_ld16x32bx2',4,'float','(128,32):(65536,1)')">Ld16x32bx2.x4 &mdash; two 16-lane halves</button>
            <button class="preset-btn" onclick="setTMC('${id}','tmem_st32x32b',4,'float','(128,32):(65536,1)')">St32x32b.x4 &mdash; registers back into TMEM</button>
            <button class="preset-btn" onclick="setTMC('${id}','tmem_ld32x32b',8,'float','(128,64):(65536,1)')">128&times;64 accumulator &mdash; 8 column tiles per thread</button>
            <button class="preset-btn" onclick="setTMC('${id}','tmem_ld16x256b',1,'float','((16,4),32):((65536,2097152),1)')">M = 64 accumulator &mdash; 16 rows in each lane quarter</button>
            <button class="preset-btn" onclick="setTMC('${id}','tmem_ld32x32b',4,'half_t','(128,32):(131072,1)')">half_t &mdash; one lane is 131072 element units</button>
            <button class="preset-btn" onclick="setTMC('${id}','tmem_ld32x32b',4,'float','(64,16):(65536,1)')">64 rows &mdash; warps 2&ndash;3 land outside their lanes</button>
          </div>
        </div>
        <div class="hint">
          <b>What it builds.</b> A 128-thread TiledCopy over the TMEM tensor: four warps, and warp
          <i>w</i> (mod 4) can only reach TMEM lanes 32<i>w</i>&hellip;32<i>w</i>+31. CuTe builds that rule
          in: its atom thread map is <code>(32,4):(0, 32&middot;DP)</code>. The Op decides how one warp's
          32 threads split their quarter: <code>32x32b</code> gives each thread one row; the 16-lane
          shapes share rows between threads.<br><br>
          <b>In a kernel</b>, slice it with <code>get_slice(tidx)</code>, then
          <code>partition_S</code> the TMEM tensor and <code>partition_D</code> the destination.
          <code>tidx</code> must agree with the real warp index mod 4: threads 32&ndash;63 are assumed
          to be a warp in lane quarter 1.<br><br>
          <b>Rows are lanes.</b> The tensor is in TMEM address units: a column is one element, a lane
          is 2^21 / bits of them. The accumulator from <code>make_fragment_C</code> is already in this
          form; paste its layout.
        </div>
      </div>
      <div class="comp-results" style="grid-template-columns:1fr">
        ${layoutVisualization({ hostId: `${id}-tmc-tmem-svg`,
          title: `<span id="${id}-tmc-tmem-title">TMEM tensor</span> ${memorySpaceBadge(`${id}-tmc-tmem-space`, 'TMEM')}`,
          onMode: mode => `setTmcMode('${id}','tmemMode','${mode}')` })}
        ${layoutVisualization({ hostId: `${id}-tmc-reg-svg`,
          title: `<span id="${id}-tmc-reg-title">Registers</span> ${memorySpaceBadge(`${id}-tmc-reg-space`, 'RMEM')}`,
          onMode: mode => `setTmcMode('${id}','regMode','${mode}')` })}
        <div class="comp-viz-item" id="${id}-tmc-thread-item" style="display:none">
          <div class="comp-viz-header"><span class="comp-viz-label" id="${id}-tmc-thread-title">Per-thread view</span></div>
          <div class="viz-box" style="background:#0b1220;color:#d1d5db;font-family:monospace;font-size:0.82rem;padding:14px 16px">
            <div id="${id}-tmc-thread-body"></div>
          </div>
        </div>
      </div>
    </div>`;
}

const tmcState = {};

/** Rebuild the per-Op repeat options before anything assigns into them. */
function tmcSyncControls(tabId) {
  mcaSyncTmemControls(tabId, MCA_OPS[document.getElementById(`${tabId}-tmc-op-input`).value], 'tmc');
}

function renderMakeTmemCopy(tabId) {
  showErr(`${tabId}-tmc-error`, '');
  showWarn(`${tabId}-tmc-warning`, '');
  const read = key => document.getElementById(`${tabId}-tmc-${key}-input`).value;
  try {
    tmcSyncControls(tabId);
    const atom = tmcAtom(read('op'), read('dtype'), read('rep'));
    const tensorStr = read('tensor');
    const p = parseLayout(tensorStr);
    const data = new Layout(p.shape, p.stride);
    const r = tmcComputeCopy(atom, data);
    const fmt = L => formatLayoutStr(L.shape, L.stride);
    document.getElementById(`${tabId}-tmc-atom-result`).innerHTML =
      `<div class="cuo-result-line"><b>Copy_Atom&lt;${atom.op.label}.x${atom.repeat}, ${atom.dtype}&gt;</b></div>` +
      `<div class="cuo-result-line">one warp: ${r.lanes} lanes &times; ${r.perLane} ${atom.dtype} ` +
      `(${atom.store ? 'RMEM &rarr; TMEM' : 'TMEM &rarr; RMEM'})</div>` +
      `<div class="cuo-result-line">ValLayoutSrc = ${fmt(atom.src)}</div>` +
      `<div class="cuo-result-line">ValLayoutDst = ${fmt(atom.dst)}</div>`;
    const instr = r.fragSize / atom.values;
    const quarters = [0, 1, 2, 3].map(w => {
      const rows = new Set();
      r.perThread.slice(32 * w, 32 * w + 32).forEach(c => c.forEach(([m]) => rows.add(m)));
      const list = [...rows].sort((a, b) => a - b);
      return `warp ${w}: ${list.length ? tmcRanges(list) : '&mdash;'}` + (r.outside.includes(w) ? ' <b>(outside its lanes)</b>' : '');
    });
    document.getElementById(`${tabId}-tmc-result`).innerHTML =
      `<div class="cuo-result-line"><b>make_tmem_copy(atom, tmem_tensor)</b></div>` +
      `<div class="cuo-result-line">Tiler MN = (${r.tiler.map(fmt).join(',')})</div>` +
      `<div class="cuo-result-line">TV Layout tiled = ${fmt(r.tv)}</div>` +
      `<div class="cuo-result-line">TV Layout Src = ${fmt(r.src)}</div>` +
      `<div class="cuo-result-line">TV Layout Dst = ${fmt(r.dst)}</div>` +
      `<div class="cuo-result-line">4 warps, 128 threads; ${r.fragSize} values per thread in ${instr} ` +
      `${atom.op.label}.x${atom.repeat} per warp; one lane = ${r.DP} address units</div>` +
      `<div class="cuo-result-line">Rows per warp &mdash; ${quarters.join('; ')}</div>`;
    const warns = [tmcStrideWarning(data, atom)];
    if (r.outside.length)
      warns.push(`Warp${r.outside.length > 1 ? 's' : ''} ${r.outside.join(', ')} ${r.outside.length > 1 ? 'are' : 'is'} ` +
        `mapped onto TMEM lanes outside ${r.outside.length > 1 ? 'their' : 'its'} quarter (the tensor has fewer ` +
        `than 128 rows, so CuTe aliases the missing warps onto existing ones). Hardware cannot execute that ` +
        `part: run this copy only on the warps whose quarter holds the rows (CuTe asserts it at run time, ` +
        `copy_traits_sm100.hpp:399).`);
    const hl = readHighlightTid(tabId, 'tmc', r.threads);
    warns.push(hl.warn);
    const warn = warns.filter(Boolean).join(' ');
    if (warn) showWarn(`${tabId}-tmc-warning`, warn);
    const prev = tmcState[tabId] || {};
    tmcState[tabId] = { atom, data, r, highlightTid: hl.tid,
      tmemMode: prev.tmemMode instanceof Set ? prev.tmemMode : new Set(['value']),
      regMode: prev.regMode instanceof Set ? prev.regMode : new Set(['value']) };
    tmcRenderViz(tabId);
    updateOuterTabLabel(tabId, `make_tmem_copy:${atom.op.tmem}.x${atom.repeat}/${atom.dtype}`);
  } catch (e) {
    showErr(`${tabId}-tmc-error`, e.message);
    for (const k of ['tmem', 'reg']) document.getElementById(`${tabId}-tmc-${k}-svg`).innerHTML = '';
    document.getElementById(`${tabId}-tmc-result`).innerHTML = '';
    document.getElementById(`${tabId}-tmc-thread-item`).style.display = 'none';
    delete tmcState[tabId];
  }
}

/** [0,1,2,3,8,9] -> "0–3, 8–9". */
function tmcRanges(list) {
  const out = [];
  for (let i = 0; i < list.length;) {
    let j = i;
    while (j + 1 < list.length && list[j + 1] === list[j] + 1) j++;
    out.push(i === j ? `${list[i]}` : `${list[i]}–${list[j]}`);
    i = j + 1;
  }
  return out.join(', ');
}

function tmcRenderViz(tabId) {
  const s = tmcState[tabId], r = s.r, a = s.atom, [M, N] = r.flat;
  const k = 32 / a.bits, focus = s.highlightTid;
  const warpColor = t => colorTV(Math.floor(t / 32));
  // TMEM pane: the tensor itself, each cell colored by the warp that reads it.
  // Value is the TMEM address, shown as lane / 32-bit column (.element when
  // several elements pack one column) rather than as the raw 2^16-scaled int.
  const quarterOf = (m) => Math.floor(Math.floor(s.data.call(m) / r.DP) / 32);
  const tmemSVG = buildColoredLayoutSVG(s.data.shape, s.data.stride, s.tmemMode, (m, n, addr, flatI) => {
    const e = r.cells[m + n * M];
    const lane = Math.floor(addr / r.DP), col = addr % r.DP;
    const where = `l${lane} c${Math.floor(col / k)}${k > 1 ? '.' + (col % k) : ''}`;
    const owner = e.find(x => x.legal) || e[0];
    const lines = owner ? [`T${owner.t}V${owner.v}`] : ['—'];
    lines.push(...buildCellLines(s.tmemMode, where, flatI, `(${m},${n})`));
    const dim = focus !== null && !e.some(x => x.t === focus);
    const illegal = e.some(x => !x.legal);
    return { bg: !owner || dim ? '#f0f0f0' : warpColor(owner.t), fg: dim ? '#bbb' : undefined, text: lines,
             stroke: illegal ? '#e53e3e' : undefined, sw: illegal ? 1.5 : undefined };
  }, { overlay: ({ cs, margin, W }) => {
    // A line where the lane quarter changes between consecutive rows.
    let out = '';
    for (let m = 1; m < M; m++) if (quarterOf(m) !== quarterOf(m - 1))
      out += `<line x1="${margin}" x2="${W}" y1="${margin + m * cs}" y2="${margin + m * cs}" stroke="#7c3aed" stroke-width="2.5"/>`;
    return out;
  } });
  renderLayoutVisualization({ hostId: `${tabId}-tmc-tmem-svg`, modes: s.tmemMode, svg: tmemSVG });
  // Register pane: thread x value, each cell the tensor coordinate it holds.
  const regSVG = buildColoredLayoutSVG([r.threads, r.fragSize], [1, r.threads], s.regMode, (t, v, _, flatI) => {
    const [m, n] = r.perThread[t][v];
    const dim = focus !== null && t !== focus;
    return { bg: dim ? '#f0f0f0' : warpColor(t), fg: dim ? '#bbb' : undefined,
             text: buildCellLines(s.regMode, `(${m},${n})`, flatI, `T${t}V${v}`) };
  });
  renderLayoutVisualization({ hostId: `${tabId}-tmc-reg-svg`, modes: s.regMode, svg: regSVG });
  document.getElementById(`${tabId}-tmc-tmem-title`).textContent =
    `${a.tmemSide.toUpperCase()} · TMEM tensor ${M}×${N} — colour = warp, purple lines = lane quarters`;
  document.getElementById(`${tabId}-tmc-reg-title`).textContent =
    `${a.regSide.toUpperCase()} · registers: thread × value → tensor coordinate`;
  setMemorySpaceBadge(`${tabId}-tmc-tmem-space`, 'TMEM');
  setMemorySpaceBadge(`${tabId}-tmc-reg-space`, 'RMEM');
  tmcRenderThreadPanel(tabId);
}

function tmcRenderThreadPanel(tabId) {
  const s = tmcState[tabId], item = document.getElementById(`${tabId}-tmc-thread-item`);
  if (!s || s.highlightTid === null) { item.style.display = 'none'; return; }
  const t = s.highlightTid, r = s.r, w = Math.floor(t / 32);
  const lanes = new Set(r.perThread[t].map(([m, n]) => Math.floor(s.data.call(m + n * r.flat[0]) / r.DP)));
  item.style.display = '';
  document.getElementById(`${tabId}-tmc-thread-title`).textContent = `Per-thread view — T${t}`;
  document.getElementById(`${tabId}-tmc-thread-body`).innerHTML =
    `<div class="cuo-result-line">T${t} = warp ${w}, lane ${t % 32}; its warp reaches TMEM lanes ` +
    `${32 * (w % 4)}–${32 * (w % 4) + 31}</div>` +
    `<div class="cuo-result-line">TMEM lanes it reads: ${tmcRanges([...lanes].sort((a, b) => a - b))}</div>` +
    `<div class="cuo-result-line">registers V0..V${r.fragSize - 1} &rarr; ` +
    r.perThread[t].map(([m, n]) => `(${m},${n})`).join(' ') + `</div>`;
}

function setTmcMode(tabId, key, mode) {
  const s = tmcState[tabId];
  if (!s) return;
  toggleLayoutVisualizationMode(s[key], mode);
  tmcRenderViz(tabId);
}

function setTMC(tabId, opKey, repeat, dtype, tensor) {
  document.getElementById(`${tabId}-tmc-op-input`).value = opKey;
  tmcSyncControls(tabId);
  document.getElementById(`${tabId}-tmc-rep-input`).value = String(repeat);
  document.getElementById(`${tabId}-tmc-dtype-input`).value = dtype;
  document.getElementById(`${tabId}-tmc-tensor-input`).value = tensor;
  renderMakeTmemCopy(tabId);
}

function exportTMC(tabId) {
  const read = key => document.getElementById(`${tabId}-tmc-${key}-input`).value;
  exportURL(`${tabId}-tmc-export`, 'make_tmem_copy', read('op'), read('rep'), read('dtype'), read('tensor'));
}

/** The CopyAtom section takes a make_copy_atom URL, but only a tcgen05 one. */
function tmcImportAtom(tabId, text) {
  const a = mcaDecodeAtomURL(text);
  if (!TMC_OPS.includes(a.opKey))
    throw new Error('make_tmem_copy takes a tcgen05 TMEM load or store CopyAtom. For SIMT, ldmatrix and ' +
                    'stmatrix atoms use make_tiled_copy or make_tiled_copy_tv.');
  setTMC(tabId, a.opKey, a.nm, a.dtype, document.getElementById(`${tabId}-tmc-tensor-input`).value);
}

function tmcDecodeAtomPasteboard(tabId) {
  return decodeFromPasteboard({ tabId, errorId: `${tabId}-tmc-error`, sources: ['make_copy_atom'],
    message: 'Paste the make_copy_atom URL (a tcgen05 TMEM Op) or its query string:',
    importFn: text => tmcImportAtom(tabId, text) });
}

/** Plain-text summary for the variable table. */
function tmcSummaryText(inputs) {
  const atom = tmcAtom(inputs[0], inputs[2], inputs[1]);
  const p = parseLayout(inputs[3]);
  const r = tmcComputeCopy(atom, new Layout(p.shape, p.stride));
  const fmt = L => formatLayoutStr(L.shape, L.stride);
  return `make_tmem_copy(${atom.op.label}.x${atom.repeat} ${atom.dtype}, ${inputs[3]})\n` +
    `Tiler MN = (${r.tiler.map(fmt).join(',')})\nTV Layout tiled = ${fmt(r.tv)}\n` +
    `TV Layout Src = ${fmt(r.src)}\nTV Layout Dst = ${fmt(r.dst)}\n128 threads; ${r.fragSize} values per thread`;
}
