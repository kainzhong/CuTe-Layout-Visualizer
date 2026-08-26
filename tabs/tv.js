// TV Layout tab: HTML panel, state, render/mode/preset/export helpers,
// plus computeTVFromThrVal and the TV-specific legend builder.
// Functions become globals on `window` (no module system).

function generateTVTabContent(id) {
  return `
    <!-- TV Layout panel -->
    <div id="${id}-tab-tv" class="panel">
      <div class="controls">
        <h2>TV Layout</h2>
        ${layoutInputField({
          id: `${id}-tv-layout-input`,
          label: 'TV Layout &mdash; (num_threads, num_values):(t_stride, v_stride)',
          hint: 'pre-filled from the thr_layout &times; val_layout below &mdash; edit either side',
          value: '((8,4),(2,2)):((16,2),(8,1))'
        })}
        ${layoutInputField({
          id: `${id}-tv-tile-input`,
          label: 'Tile &mdash; (M, N)',
          hint: 'a tile is just a shape; the TV layout’s output is col-major into it',
          value: '(8, 16)'
        })}

        <div class="form-group">
          <label>Highlight thread (empty = none)</label>
          <input type="text" id="${id}-tv-highlight-tid" value="" placeholder="e.g. 3" oninput="setHighlightTid('${id}')">
        </div>

        <details class="cuo-section">
          <summary>Check Coalesced Read (GMEM)</summary>
          <div class="cuo-section-body">
            ${layoutInputField({
              id: `${id}-tv-gmem-input`,
              label: 'GMEM layout &mdash; logical (m, n) &rarr; physical offset',
              value: '',
              placeholder: 'e.g. (8, 16):(16, 1) for row-major',
              oninput: `setTVDataLayout('${id}')`
            })}
            <div class="form-group" style="display:flex;gap:5px;margin-top:-4px">
              <button class="btn" style="flex:1;font-size:0.75rem;padding:4px" onclick="setTVDataMajor('${id}','gmem','row')">Row-major</button>
              <button class="btn" style="flex:1;font-size:0.75rem;padding:4px" onclick="setTVDataMajor('${id}','gmem','col')">Col-major</button>
            </div>
            <div class="form-group">
              <label>tensor_dtype<span style="color:#6b7280;font-weight:normal">&nbsp;&mdash; sectors are counted in bytes, so element size matters</span></label>
              <select id="${id}-tv-gmem-dtype" onchange="setTVDtype('${id}')">${dtypeOptions('half_t')}</select>
            </div>
            <div class="form-group">
              <button class="btn" id="${id}-tv-coal-check-btn" style="width:100%;font-size:0.75rem;padding:5px;display:flex;align-items:center;justify-content:center;gap:6px" onclick="toggleTVCoalesced('${id}')">
                <span>Show coalesced-read view</span>
                <span class="cuo-info-icon" onclick="event.stopPropagation()" data-tooltip="Each color = one warp-wide memory issue: all threads of one warp accessing the same group of consecutive vids that a single instruction can fetch. The vector width is derived from the value layout and the GMEM layout — the widest run of adjacent addresses each thread owns — so there is nothing to configure. Each cell is labelled with its physical offset and the 32-byte sector it falls in (§N) — a sector is a fixed number of BYTES, so tensor_dtype changes how many cells share one. If the offsets inside one color form a consecutive run, that issue is coalesced (one cache-line transaction). If they are scattered, the warp needs several transactions — the summary line above the grid counts the distinct 32-byte sectors each issue touches.">i</span>
              </button>
            </div>
          </div>
        </details>

        <details class="cuo-section">
          <summary>Check Bank Conflict (SMEM)</summary>
          <div class="cuo-section-body">
            ${layoutInputField({
              id: `${id}-tv-smem-input`,
              label: 'SMEM layout &mdash; logical (m, n) &rarr; physical offset',
              value: '',
              placeholder: 'e.g. (8, 16):(16, 1) for row-major',
              oninput: `setTVDataLayout('${id}')`
            })}
            <div class="form-group" style="display:flex;gap:5px;margin-top:-4px">
              <button class="btn" style="flex:1;font-size:0.75rem;padding:4px" onclick="setTVDataMajor('${id}','smem','row')">Row-major</button>
              <button class="btn" style="flex:1;font-size:0.75rem;padding:4px" onclick="setTVDataMajor('${id}','smem','col')">Col-major</button>
            </div>
            <div class="form-group">
              <label>tensor_dtype<span style="color:#6b7280;font-weight:normal">&nbsp;&mdash; a bank is 4 bytes wide, so element size decides which bank a cell lands in</span></label>
              <select id="${id}-tv-smem-dtype" onchange="setTVDtype('${id}')">${dtypeOptions('half_t')}</select>
            </div>
            <div class="form-group">
              <label>All threads access V#<span style="color:#6b7280;font-weight:normal">&nbsp;&mdash; one wave (every thread reads vid=N); other cells gray</span></label>
              <input type="text" id="${id}-tv-wave-input" value="" placeholder="e.g. 0" oninput="setTVWaveVid('${id}')">
            </div>
            <div class="form-group">
              <label>Bank filter (0..31, empty = off) &mdash; amber edge on matching cells</label>
              <input type="text" id="${id}-tv-bank-input" value="" placeholder="e.g. 7" oninput="setTVBank('${id}')">
            </div>
            <div class="form-group">
              <label>Swizzle &mdash; <code>B, M, S</code> (empty = no swizzle; e.g. <code>3, 3, 3</code>). Applied to the element offset before bank is computed</label>
              <input type="text" id="${id}-tv-swizzle-input" value="" placeholder="e.g. 3, 3, 3" oninput="setTVSwizzle('${id}')">
            </div>
            <div class="form-group">
              <button class="btn" id="${id}-tv-bank-check-btn" style="width:100%;font-size:0.75rem;padding:5px;display:flex;align-items:center;justify-content:center;gap:6px" onclick="toggleTVBankCheck('${id}')">
                <span>Show bank-conflict view</span>
                <span class="cuo-info-icon" onclick="event.stopPropagation()" data-tooltip="Appends #&lt;bank&gt; (the 32-bank SMEM bank id) to each cell's label. Cell color is unchanged — still keyed to thread id — so you read off T/V/bank for every (m, n) and decide for yourself which warp-instruction grouping matters. Use the bank filter to amber-edge all cells in one bank and visually count which threads land on it. Set the Swizzle input above to apply CuTe's Swizzle<B,M,S> transform to the element offset before bank is computed.">i</span>
              </button>
            </div>
          </div>
        </details>

        <div class="form-group" style="border-top:1px solid #374151;padding-top:12px">
          <label style="color:#93c5fd;letter-spacing:0.5px">&mdash; OR compute from thr/val &mdash;</label>
          <div style="color:#6b7280;font-size:0.72rem;line-height:1.4;margin-top:-2px">The TV Layout and
            Tile above are what these two produce through <code>make_layout_tv</code>. Edit them and hit
            <b>Compute</b> to refill both.</div>
        </div>
        ${layoutInputField({ id: `${id}-tv-thr-input`, label: 'thr_layout', value: '(4, 8):(8, 1)', placeholder: 'e.g. (2, 3):(3, 1)' })}
        ${layoutInputField({ id: `${id}-tv-val-input`, label: 'val_layout', value: '(2, 2):(2, 1)', placeholder: 'e.g. (2, 2):(2, 1)' })}
        <button class="btn" style="width:100%;font-size:0.8rem" onclick="computeTVFromThrVal('${id}')">Compute TV + Tile from thr/val</button>
        ${statusDivs(`${id}-tv`)}
        <button class="btn btn-render" onclick="renderTV('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-tv-export" onclick="exportTV('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets &mdash; thr_layout &times; val_layout</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setTVFromThrVal('${id}','(2,3):(3,1)','(2,2):(2,1)')">2x3 threads, 2x2 values (row-maj) &rarr; 4x6 tile</button>
            <button class="preset-btn" onclick="setTVFromThrVal('${id}','(4,4):(1,4)','(2,2):(1,2)')">4x4 threads, 2x2 values (col-maj) &rarr; 8x8 tile</button>
            <button class="preset-btn" onclick="setTVFromThrVal('${id}','(4,8):(8,1)','(2,2):(2,1)')">4x8 threads, 2x2 values (row-maj) &rarr; 8x16 tile</button>
            <button class="preset-btn" onclick="setTVFromThrVal('${id}','(8,4):(1,8)','(2,2):(1,2)')">8x4 threads, 2x2 values (col-maj) &rarr; 16x8 tile</button>
            <button class="preset-btn" onclick="setTVFromThrVal('${id}','(16,2):(1,16)','(2,2):(1,2)')">32 threads (16x2), 2x2 values &rarr; 32x4 tile</button>
          </div>
        </div>

        <div class="hint">
          TV layout maps each <code>(tid, vid)</code> pair to a
          position in the MxN tile.<br><br>
          Cell color = thread ID.<br>
          Cell text = T<i>tid</i> / V<i>vid</i>.<br>
          Empty cells (&mdash;) are not covered by any thread.<br><br>
          <b>Tile is just a shape</b> &mdash; <code>(M, N)</code>. In CuTe,
          <code>make_layout_tv</code> returns <code>tiler_mn</code> as a plain
          shape tuple, and the TV layout's output is a column-major flat index
          into that tile. If you want a row-major visualization, encode that
          in the TV layout's own strides rather than the tile.
        </div>
      </div>

      <div class="visualization">
        <div class="viz-header">
          <span class="viz-title" id="${id}-tv-title">&mdash;</span>
          <span style="display:flex;align-items:center;gap:8px">
            <span class="mode-btn-group" id="${id}-tv-mode-btns">
              <button class="mode-btn" onclick="setTVMode('${id}','value')">value</button>
            </span>
            <button class="btn" id="${id}-tv-svg-host-zoom" onclick="toggleZoom('${id}-tv-svg-host')">Zoom in</button>
            <button class="btn" onclick="downloadSVG('${id}-tv-svg-host', 'tv_layout.svg')">Download SVG</button>
          </span>
        </div>
        <div class="viz-box">
          <div id="${id}-tv-svg-host"></div>
        </div>
        <div class="legend" id="${id}-tv-legend"></div>
      </div>
    </div>`;
}

const tvState = {};

// Widest single load/store any current NVIDIA architecture issues: 16 bytes
// (LDG.128 / STG.128 / cp.async.128). A derived vector width is capped here —
// owning more adjacent elements than this buys more instructions, not a wider one.
const TV_MAX_ACCESS_BITS = 128;


function renderTV(tabId) {
  showErr(`${tabId}-tv-error`, '');
  try {
    const tvInput = document.getElementById(`${tabId}-tv-layout-input`).value;
    const tileInput = document.getElementById(`${tabId}-tv-tile-input`).value;
    const gmemInput = document.getElementById(`${tabId}-tv-gmem-input`).value;
    const smemInput = document.getElementById(`${tabId}-tv-smem-input`).value;
    updateRankWarning(`${tabId}-tv-warning`, [
      ['TV layout', tvInput], ['Tile', tileInput],
      ['GMEM layout', gmemInput], ['SMEM layout', smemInput]
    ]);
    const tvL   = parseLayout(tvInput);
    const tileL = parseLayout(tileInput);

    const numT = product(tvL.shape[0]);
    const numV = product(tvL.shape[1]);
    const M    = product(tileL.shape[0]);
    const N    = product(tileL.shape[1]);

    // Read the highlight-thread input; empty or invalid → null (no filter).
    const highlightRaw = document.getElementById(`${tabId}-tv-highlight-tid`).value.trim();
    const highlightTid = highlightRaw === '' ? null : parseInt(highlightRaw, 10);
    const highlightValid = highlightTid !== null && !isNaN(highlightTid);
    const prev = tvState[tabId] || {};
    // Default to '' (no labels) — user opts in by clicking `value`.
    // Using `!== undefined` preserves an explicitly-cleared mode on re-render.
    const mode = (prev.mode !== undefined) ? prev.mode : '';
    tvState[tabId] = {
      tvL, tileL, mode,
      bankCheck:       !!prev.bankCheck,
      coalCheck:       !!prev.coalCheck,
      bankToHighlight: (prev.bankToHighlight != null) ? prev.bankToHighlight : null,
      swizzle:         prev.swizzle || null,
      waveVid:         (prev.waveVid != null) ? prev.waveVid : null,
    };

    const titleHL = highlightValid ? `  \u2014  highlight T${highlightTid}` : '';
    document.getElementById(`${tabId}-tv-title`).textContent =
      `${numT} threads \u00d7 ${numV} values  \u2014  ${M}\u00d7${N} tile${titleHL}`;

    const host = document.getElementById(`${tabId}-tv-svg-host`);
    if (tvState[tabId].coalCheck) {
      host.innerHTML = renderTVCoalescedSVG(
        tabId, tvL, tileL, numT, numV, M, N,
        highlightValid ? highlightTid : null
      );
    } else if (tvState[tabId].bankCheck) {
      host.innerHTML = renderTVBankSVG(
        tabId, tvL, tileL, numT, numV, M, N,
        highlightValid ? highlightTid : null, mode
      );
    } else {
      host.innerHTML =
        buildTVSVG(tvL.shape, tvL.stride, tileL.shape, tileL.stride, false, 'col',
                   highlightValid ? highlightTid : null, mode);
    }
    applyZoomState(`${tabId}-tv-svg-host`);
    updateModeBtns(`${tabId}-tv-mode-btns`, mode ? new Set([mode]) : new Set());

    buildLegend(tabId, numT);

    const bankBtn = document.getElementById(`${tabId}-tv-bank-check-btn`);
    if (bankBtn) bankBtn.classList.toggle('active', !!tvState[tabId].bankCheck);
    const coalBtn = document.getElementById(`${tabId}-tv-coal-check-btn`);
    if (coalBtn) coalBtn.classList.toggle('active', !!tvState[tabId].coalCheck);

    updateOuterTabLabel(tabId, `TV-Layout:${tvInput.trim()}`);
  } catch (e) {
    showErr(`${tabId}-tv-error`, e.message);
    document.getElementById(`${tabId}-tv-svg-host`).innerHTML = '';
    document.getElementById(`${tabId}-tv-legend`).innerHTML = '';
  }
}

// Bank-check render path: same coloring as buildTVSVG (thread id), but each
// cell's text is extended with `#<bank>`, and cells whose bank matches the
// user's bank filter get an amber 3px edge. Element offset is mapped through
// the user-supplied data layout (or defaults to col-major flat), optionally
// swizzled, then converted to bytes / 4 % 32 for the SMEM bank id.
// Resolve the user's data layout (logical flat (m + n*M) -> physical element
// offset) plus the dtype's byte width. Both memory checks need exactly this,
// and both refuse to run without a data layout: coalescing and bank ids are
// properties of the memory the TV layout is applied to, not of the TV layout.
function tvResolveData(tabId, which, M, N, checkName) {
  const dataStr = document.getElementById(`${tabId}-tv-${which}-input`).value.trim();
  const dtype = document.getElementById(`${tabId}-tv-${which}-dtype`).value;
  const bytesPerElement = (DTYPE_BITS[dtype] || 16) / 8;
  if (dataStr === '') {
    throw new Error(
      `${which.toUpperCase()} layout is required for the ${checkName} check. ` +
      'Use the Row-major / Col-major buttons or enter a layout manually.');
  }
  const dp = parseLayout(dataStr);
  const sp = stripTrivialTrailing(dp.shape, dp.stride);
  const dataL = new Layout(sp.shape, sp.stride);
  if (dataL.size() !== M * N) {
    throw new Error(
      `${which.toUpperCase()} layout size (${dataL.size()}) does not match tile size ` +
      `(${M}\u00d7${N} = ${M * N}).`);
  }
  return { dataL, dtype, bytesPerElement };
}

// (m, n) -> list of { tid, vid } that land there. A well-formed TV layout puts
// exactly one entry in every cell; more than one is a collision and gets a red
// stroke downstream, zero means the cell is uncovered.
function tvBuildGrid(tvL, tileL, numT, numV, M, N) {
  function scalarStride(x) {
    if (typeof x === 'number') return x;
    if (Array.isArray(x)) return scalarStride(x[0]);
    return 1;
  }
  const sm = scalarStride(tileL.stride[0]);
  const sn = scalarStride(tileL.stride[1]);
  const grid = Array.from({length: M}, () => Array.from({length: N}, () => []));
  for (let tid = 0; tid < numT; tid++) {
    for (let vid = 0; vid < numV; vid++) {
      const c0  = unflatten(tid, tvL.shape[0]);
      const c1  = unflatten(vid, tvL.shape[1]);
      const idx = crd2idx(c0, tvL.shape[0], tvL.stride[0]) +
                  crd2idx(c1, tvL.shape[1], tvL.stride[1]);
      const m = Math.floor(idx / sm) % M;
      const n = Math.floor(idx / sn) % N;
      if (m >= 0 && m < M && n >= 0 && n < N) grid[m][n].push({ tid, vid });
    }
  }
  return grid;
}

function renderTVBankSVG(tabId, tvL, tileL, numT, numV, M, N, highlightTid, labelMode) {
  const s = tvState[tabId];
  const showIdx = labelMode === 'value';
  const { dataL, bytesPerElement } = tvResolveData(tabId, 'smem', M, N, 'SMEM bank conflict');
  const grid = tvBuildGrid(tvL, tileL, numT, numV, M, N);

  const bankHL = s.bankToHighlight;
  const sw = s.swizzle;
  const waveVid = s.waveVid;
  // A cell entry "matches" the active filters if it lines up with the
  // highlight thread (when set) AND the wave vid (when set). With neither
  // filter active, every entry matches.
  const matchesFilters = (e) => {
    if (highlightTid !== null && e.tid !== highlightTid) return false;
    if (waveVid !== null && e.vid !== waveVid) return false;
    return true;
  };
  const anyFilter = highlightTid !== null || waveVid !== null;
  return buildColoredLayoutSVG([M, N], [1, M], 'value', (m, n) => {
    const entries = grid[m][n];
    const logical = m + n * M;
    const physical = dataL.call(logical);
    const swizzled = applySwizzleOffset(physical, sw);
    const bank = Math.floor(swizzled * bytesPerElement / 4) % 32;
    const bankLine = `#${bank}`;

    if (entries.length === 0) {
      return { bg: '#f0f0f0', fg: '#bbb', text: ['\u2014', bankLine] };
    }
    const dimmed = anyFilter && !entries.some(matchesFilters);
    if (entries.length === 1) {
      const { tid, vid } = entries[0];
      const bg = dimmed ? '#f0f0f0' : colorTV(tid);
      const fg = dimmed ? '#bbb' : '#111';
      const lines = [`T${tid}`, `V${vid}`];
      if (showIdx) lines.push(String(m + n * M));
      lines.push(bankLine);
      let stroke, swEdge;
      if (bankHL != null && bank === bankHL && !dimmed) { stroke = '#f59e0b'; swEdge = 3; }
      return { bg, fg, text: lines, stroke, sw: swEdge };
    }
    // Multiple (tid, vid) on one cell - match buildTVSVG's red collision stroke.
    const bg = dimmed ? '#f0f0f0' : colorTV(entries[0].tid);
    const fg = dimmed ? '#bbb' : '#111';
    const stroke = dimmed ? '#ccc' : '#e53e3e';
    const swEdge = dimmed ? 0.5 : 1.5;
    const lines = entries.map(e => `T${e.tid}/V${e.vid}`);
    if (showIdx) lines.push(String(m + n * M));
    lines.push(bankLine);
    return { bg, fg, text: lines, stroke, sw: swEdge };
  });
}

// GMEM coalesced-read path. One color = one warp-wide memory issue: every
// thread of a warp accessing the same block of W consecutive vids. Cell label
// is the physical offset from the data layout, so consecutive numbers inside
// one color == coalesced.
//
// W is DERIVED, not asked for: it is the widest vector each thread could
// actually load, i.e. the largest divisor of numV for which every thread's W
// consecutive vids sit at W consecutive addresses (CuTe's `upcast<W>`
// precondition). The value layout and the data layout already determine this
// between them — nobody hands a thread 8 adjacent elements and then loads them
// one at a time — so making the user restate it only invites a wrong answer.
//
// This is the visualization that used to live in the CopyUniversalOp tab's
// partition section; it belongs here because coalescing is a property of the
// (TV layout, data layout) pair, not of the copy atom.
function renderTVCoalescedSVG(tabId, tvL, tileL, numT, numV, M, N, highlightTid) {
  const s = tvState[tabId];
  const { dataL, dtype, bytesPerElement } = tvResolveData(tabId, 'gmem', M, N, 'GMEM coalesced read');
  const grid = tvBuildGrid(tvL, tileL, numT, numV, M, N);

  // Width and dtype are two views of ONE quantity: bytes moved per thread per
  // instruction. CuTe makes this explicit — a 128-bit copy of half_t recasts the
  // tensor to a 128-bit type and the per-thread layout becomes 1 element wide.
  // So the hardware ceiling is a byte ceiling: no single load moves more than
  // 128 bits, and owning 16 adjacent half_t buys two instructions, not one.
  const feasible = tvVectorFeasibility(tvL, grid, dataL, numT, numV, M, N);
  const runW = feasible.maxWidth(numV, numV);           // widest contiguous run
  const capW = Math.max(1, Math.floor(TV_MAX_ACCESS_BITS / (bytesPerElement * 8)));
  const W = feasible.maxWidth(numV, capW);
  const capped = runW > W;
  const WARP = 32;
  const numWarps = Math.max(1, Math.ceil(numT / WARP));
  const vecsPerThread = numV / W;

  // group id = (warp, vector index within the thread), assigned in issue order
  // so the color ramp reads as "first instruction" -> "last instruction".
  const groupId = (tid, vid) => Math.floor(vid / W) * numWarps + Math.floor(tid / WARP);

  // Per-group coalescing summary: how many distinct 32-byte sectors does one
  // warp-wide issue touch, versus the minimum it could touch?
  const sectors = new Map();   // gid -> Set of 32B sector ids
  const bytesIn = new Map();   // gid -> element count
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      for (const e of grid[m][n]) {
        const gid = groupId(e.tid, e.vid);
        const off = dataL.call(m + n * M);
        if (!sectors.has(gid)) { sectors.set(gid, new Set()); bytesIn.set(gid, 0); }
        sectors.get(gid).add(Math.floor(off * bytesPerElement / 32));
        bytesIn.set(gid, bytesIn.get(gid) + 1);
      }
    }
  }
  // Compare each group against ITS OWN minimum. Taking max(got) vs max(min)
  // across groups would be a cross-group comparison: with a partial last warp
  // the groups hold different element counts, so a small wasteful group could
  // hide behind a large efficient one and still report "fully coalesced".
  let worst = 0, best = Infinity, ideal = 0, worstRatio = 1, badGroups = 0;
  for (const [gid, sec] of sectors) {
    const got = sec.size;
    const min = Math.max(1, Math.ceil(bytesIn.get(gid) * bytesPerElement / 32));
    best = Math.min(best, got);
    if (got > worst) { worst = got; ideal = min; }
    if (got > min) { badGroups++; worstRatio = Math.max(worstRatio, got / min); }
  }
  const perfect = badGroups === 0;
  const elemBits = bytesPerElement * 8;
  const widthNote = W > 1
    ? `<b>Vector width ${W}</b> (a ${W * elemBits}-bit access), derived from the value layout ` +
      `and ${dtype}: each thread's ${W} consecutive values are adjacent in memory. `
    : `<b>Vector width 1</b> \u2014 no thread has even two adjacent ${dtype} under this GMEM ` +
      `layout, so every element is its own access. `;
  const capNote = capped
    ? `Each thread actually owns a run of ${runW} adjacent ${dtype}, but no single load moves more ` +
      `than ${TV_MAX_ACCESS_BITS} bits, so it takes ${runW / W} instructions. ` +
      `(Widening tensor_dtype instead of the run is the same thing \u2014 that is what CuTe's ` +
      `<code>recast</code> does.) `
    : '';
  const summary =
    `<div style="font-size:0.78rem;font-family:monospace;margin-bottom:4px;color:#9ca3af">` +
    widthNote + capNote +
    `${numWarps} warp${numWarps === 1 ? '' : 's'} \u00d7 ${vecsPerThread} instruction${vecsPerThread === 1 ? '' : 's'} ` +
    `= ${sectors.size} memory issue${sectors.size === 1 ? '' : 's'}. ` +
    `Colour = one issue; each cell shows its physical offset and its 32B sector ` +
    `(<b>\u00A7</b>, ${32 / bytesPerElement} ${dtype} per sector). ` +
    `Each issue touches ${best === worst ? best : `${best}\u2013${worst}`} distinct 32B sector${worst === 1 ? '' : 's'}. ` +
    (perfect
      ? `<b style="color:#10b981">\u2713 Fully coalesced</b> \u2014 every issue touches only the ` +
        `sectors it actually needs.`
      : `<b style="color:#ef4444">\u2717 Not coalesced</b> \u2014 ${badGroups} of ${sectors.size} ` +
        `issue${badGroups === 1 ? '' : 's'} fetch more sectors than their own payload requires, ` +
        `up to ${worstRatio.toFixed(1)}\u00d7 (worst issue: ${worst} sectors for a ${ideal}-sector payload).`) +
    `</div>`;

  // Each cell shows its physical offset AND the 32-byte sector that offset falls
  // in. The sector line is what makes tensor_dtype visible: colors and offsets
  // are element-indexed and identical for every dtype, but a sector is a fixed
  // 32 BYTES, so a wider element packs fewer cells into each one. Without this
  // the dtype selector silently changed nothing but a number in the summary.
  const sectorOf = (off) => Math.floor(off * bytesPerElement / 32);
  function svgFor() { return buildColoredLayoutSVG([M, N], [1, M], 'value', (m, n) => {
    const entries = grid[m][n];
    const off = dataL.call(m + n * M);
    const lines = [String(off), `\u00A7${sectorOf(off)}`];
    if (entries.length === 0) return { bg: '#f0f0f0', fg: '#bbb', text: ['\u2014', lines[1]] };
    const dimmed = highlightTid !== null && !entries.some(e => e.tid === highlightTid);
    if (dimmed) return { bg: '#f0f0f0', fg: '#bbb', text: lines };
    const gid = groupId(entries[0].tid, entries[0].vid);
    const stroke = entries.length > 1 ? '#e53e3e' : undefined;
    return {
      bg: tvCoalescedColor(gid),
      fg: '#111',
      text: lines,
      stroke,
      sw: entries.length > 1 ? 1.5 : undefined,
    };
  }); }
  return summary + svgFor();
}

/** Is a width-W vector instruction actually issuable? Builds each thread's
 *  vid -> physical-offset table once, then answers for any W: a thread's W
 *  consecutive vids must sit at W consecutive addresses, which is exactly what
 *  CuTe's `upcast<W>` demands of the per-thread source slice.
 *
 *  This is the missing half of the vector-width input. Grouping vids into
 *  "instructions" is a fiction unless the hardware could really fetch them
 *  together, and a sector count computed over a fictional instruction reads as
 *  a mild inefficiency when the truth is that the copy will not compile. */
function tvVectorFeasibility(tvL, grid, dataL, numT, numV, M, N) {
  const off = Array.from({length: numT}, () => new Array(numV).fill(null));
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      const physical = dataL.call(m + n * M);
      for (const e of grid[m][n]) off[e.tid][e.vid] = physical;
    }
  }
  const runOK = (tid, from, W) => {
    const base = off[tid][from];
    if (base === null) return false;
    for (let k = 1; k < W; k++) {
      if (off[tid][from + k] !== base + k) return false;
    }
    return true;
  };
  const ok = (W) => {
    if (W <= 1) return true;
    if (numV % W !== 0) return false;
    for (let tid = 0; tid < numT; tid++) {
      for (let v = 0; v < numV; v += W) if (!runOK(tid, v, W)) return false;
    }
    return true;
  };
  return {
    ok,
    /** First (thread, vector) that breaks, for the error message. */
    example(W) {
      for (let tid = 0; tid < numT; tid++) {
        for (let v = 0; v < numV; v += W) {
          if (!runOK(tid, v, W)) {
            return { tid, from: v, to: v + W - 1, offsets: off[tid].slice(v, v + W) };
          }
        }
      }
      return { tid: 0, from: 0, to: W - 1, offsets: off[0].slice(0, W) };
    },
    /** Largest divisor of numV that is issuable, at most `cap`. */
    maxWidth(numV, cap) {
      for (let W = Math.min(numV, cap); W >= 2; W--) if (numV % W === 0 && ok(W)) return W;
      return 1;
    },
  };
}

// Color for a memory-issue group: cycle through TV_COLORS, darken each lap so
// long instruction sequences stay distinguishable.
function tvCoalescedColor(gid) {
  const base = colorTV(gid % 8);
  const loops = Math.floor(gid / 8);
  return loops === 0 ? base : darkenRGB(base, Math.min(loops * 0.2, 0.6));
}

// Toggle the GMEM coalesced-read overlay. Mutually exclusive with the bank
// check \u2014 they color the same grid by different groupings.
function toggleTVCoalesced(tabId) {
  if (!tvState[tabId]) tvState[tabId] = {};
  const st = tvState[tabId];
  st.coalCheck = !st.coalCheck;
  if (st.coalCheck) st.bankCheck = false;
  const coalBtn = document.getElementById(`${tabId}-tv-coal-check-btn`);
  if (coalBtn) coalBtn.classList.toggle('active', !!st.coalCheck);
  const bankBtn = document.getElementById(`${tabId}-tv-bank-check-btn`);
  if (bankBtn) bankBtn.classList.toggle('active', !!st.bankCheck);
  if (st.tvL) renderTV(tabId);
}

// Toggle the bank-conflict overlay. Empty/invalid state (no tvL yet) is
// silently ignored - the next Render will pick up the new flag.
function toggleTVBankCheck(tabId) {
  if (!tvState[tabId]) tvState[tabId] = {};
  tvState[tabId].bankCheck = !tvState[tabId].bankCheck;
  if (tvState[tabId].bankCheck) tvState[tabId].coalCheck = false;
  const btn = document.getElementById(`${tabId}-tv-bank-check-btn`);
  if (btn) btn.classList.toggle('active', !!tvState[tabId].bankCheck);
  const coalBtn = document.getElementById(`${tabId}-tv-coal-check-btn`);
  if (coalBtn) coalBtn.classList.toggle('active', !!tvState[tabId].coalCheck);
  if (tvState[tabId].tvL) renderTV(tabId);
}

// Live update when the swizzle input changes.
function setTVSwizzle(tabId) {
  if (!tvState[tabId]) tvState[tabId] = {};
  const raw = (document.getElementById(`${tabId}-tv-swizzle-input`).value || '').trim();
  let sw = null;
  if (raw !== '') {
    const mm = raw.match(/^\s*(?:Swizzle\s*<\s*)?(\d+)\s*[,\s]\s*(\d+)\s*[,\s]\s*(\d+)\s*>?\s*$/i);
    if (mm) {
      const B = parseInt(mm[1], 10);
      const M = parseInt(mm[2], 10);
      const S = parseInt(mm[3], 10);
      if (B >= 0 && M >= 0 && S >= 0 && (M + S + B) < 31) sw = { B, M, S };
    }
  }
  tvState[tabId].swizzle = sw;
  if (tvState[tabId].tvL && (tvState[tabId].bankCheck || tvState[tabId].coalCheck)) renderTV(tabId);
}

// Live update when the wave-vid input changes. Selects the (warp-wide) wave
// where every thread is reading values at vid=N — cells whose entry's vid
// doesn't match are dimmed.
function setTVWaveVid(tabId) {
  if (!tvState[tabId]) tvState[tabId] = {};
  const raw = (document.getElementById(`${tabId}-tv-wave-input`).value || '').trim();
  let vid = null;
  if (raw !== '') {
    const n = parseInt(raw, 10);
    if (Number.isFinite(n) && n >= 0) vid = n;
  }
  tvState[tabId].waveVid = vid;
  if (tvState[tabId].tvL && (tvState[tabId].bankCheck || tvState[tabId].coalCheck)) renderTV(tabId);
}

// Live update when the bank-filter input changes.
function setTVBank(tabId) {
  if (!tvState[tabId]) tvState[tabId] = {};
  const raw = (document.getElementById(`${tabId}-tv-bank-input`).value || '').trim();
  let bank = null;
  if (raw !== '') {
    const n = parseInt(raw, 10);
    if (Number.isFinite(n) && n >= 0 && n < 32) bank = n;
  }
  tvState[tabId].bankToHighlight = bank;
  if (tvState[tabId].tvL && (tvState[tabId].bankCheck || tvState[tabId].coalCheck)) renderTV(tabId);
}

// Re-render when dtype changes (bank ids depend on bytes/element).
function setTVDtype(tabId) {
  if (tvState[tabId] && tvState[tabId].tvL && (tvState[tabId].bankCheck || tvState[tabId].coalCheck)) renderTV(tabId);
}

// Live update when the data layout input changes. Parse errors / size
// mismatches surface through renderTV's existing try/catch.
function setTVDataLayout(tabId) {
  if (tvState[tabId] && tvState[tabId].tvL && (tvState[tabId].bankCheck || tvState[tabId].coalCheck)) renderTV(tabId);
}

// Fill the data layout input with row-major or col-major over the current
// tile shape, so the user doesn't have to type `(M, N):(N, 1)` themselves.
// Reads M, N from the Tile input; falls back to a parse error if it's empty/bad.
function setTVDataMajor(tabId, which, major) {
  try {
    const tileStr = document.getElementById(`${tabId}-tv-tile-input`).value;
    const tileL = parseLayout(tileStr);
    const M = product(tileL.shape[0]);
    const N = product(tileL.shape[1]);
    const layout = major === 'row' ? `(${M}, ${N}):(${N}, 1)` : `(${M}, ${N}):(1, ${M})`;
    document.getElementById(`${tabId}-tv-${which}-input`).value = layout;
    if (tvState[tabId] && tvState[tabId].tvL && (tvState[tabId].bankCheck || tvState[tabId].coalCheck)) renderTV(tabId);
  } catch (e) {
    showErr(`${tabId}-tv-error`, `Cannot derive ${major}-major data layout: ${e.message}`);
  }
}

/** Re-render when the highlight-thread input changes (live update). */
function setHighlightTid(tabId) {
  if (tvState[tabId] && tvState[tabId].tvL) renderTV(tabId);
}

/** Toggle the 'value' label (the TV layout's output = col-major flat position).
 *  Empty state is allowed — click the active button to hide labels entirely. */
function setTVMode(tabId, mode) {
  if (!tvState[tabId]) tvState[tabId] = {};
  tvState[tabId].mode = (tvState[tabId].mode === mode) ? '' : mode;
  if (tvState[tabId].tvL) renderTV(tabId);
}

function buildLegend(tabId, numT) {
  const legend = document.getElementById(`${tabId}-tv-legend`);
  const shown = Math.min(numT, 16);
  let html = '';
  for (let tid = 0; tid < shown; tid++) {
    html += `<div class="legend-item">
      <div class="legend-swatch" style="background:${TV_COLORS[tid % 8]}"></div>
      T${tid}
    </div>`;
  }
  if (numT > 16) html += `<div class="legend-item">\u2026 (${numT} threads total)</div>`;
  legend.innerHTML = html;
}

function setTV(tabId, tv, tile) {
  document.getElementById(`${tabId}-tv-layout-input`).value = tv;
  document.getElementById(`${tabId}-tv-tile-input`).value = tile;
  renderTV(tabId);
}

/** Preset helper that populates thr_layout + val_layout and derives TV + Tile
 *  via `make_layout_tv`. Use this for presets that teach the thr/val mental
 *  model rather than the already-combined TV form. */
function setTVFromThrVal(tabId, thr, val) {
  document.getElementById(`${tabId}-tv-thr-input`).value = thr;
  document.getElementById(`${tabId}-tv-val-input`).value = val;
  computeTVFromThrVal(tabId);
}

/** Fill TV Layout + Tile inputs from thr_layout and val_layout via make_layout_tv. */
function computeTVFromThrVal(tabId) {
  showErr(`${tabId}-tv-error`, '');
  try {
    const thrStr = document.getElementById(`${tabId}-tv-thr-input`).value;
    const valStr = document.getElementById(`${tabId}-tv-val-input`).value;
    const thrRaw = parseLayout(thrStr);
    const valRaw = parseLayout(valStr);
    const thrP = stripTrivialTrailing(thrRaw.shape, thrRaw.stride);
    const valP = stripTrivialTrailing(valRaw.shape, valRaw.stride);
    const thr = new Layout(thrP.shape, thrP.stride);
    const val = new Layout(valP.shape, valP.stride);

    const { tiler_mn, layout_tv } = make_layout_tv(thr, val);

    const tvString = formatLayoutStr(layout_tv.shape, layout_tv.stride);
    const [M, N] = tiler_mn;
    // layout_tv outputs col-major flat indices into tiler_mn. parseLayout
    // defaults shape-only inputs to col-major strides (1, M), so emit the shape.
    const tileString = `(${M}, ${N})`;

    document.getElementById(`${tabId}-tv-layout-input`).value = tvString;
    document.getElementById(`${tabId}-tv-tile-input`).value = tileString;
    renderTV(tabId);
  } catch (e) {
    showErr(`${tabId}-tv-error`, 'Failed to compute from thr/val: ' + e.message);
  }
}

function exportTV(tabId) {
  const thr = document.getElementById(`${tabId}-tv-thr-input`).value.trim();
  const val = document.getElementById(`${tabId}-tv-val-input`).value.trim();
  // If both thr_layout and val_layout are provided, prefer method 2
  // (TV+Tile can be derived from them). Otherwise fall back to method 1.
  if (thr && val) {
    exportURL(`${tabId}-tv-export`, 'tv-2', thr, val);
  } else {
    const tv = document.getElementById(`${tabId}-tv-layout-input`).value;
    const tile = document.getElementById(`${tabId}-tv-tile-input`).value;
    exportURL(`${tabId}-tv-export`, 'tv-1', tv, tile);
  }
}
