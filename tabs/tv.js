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
          value: '(32, 4):(1, 32)'
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
          <summary>Check SMEM bank conflict</summary>
          <div class="cuo-section-body">
            ${layoutInputField({
              id: `${id}-tv-data-input`,
              label: 'Data layout &mdash; logical (m, n) &rarr; physical offset',
              value: '',
              placeholder: 'e.g. (8, 16):(16, 1) for row-major',
              oninput: `setTVDataLayout('${id}')`
            })}
            <div class="form-group" style="display:flex;gap:5px;margin-top:-4px">
              <button class="btn" style="flex:1;font-size:0.75rem;padding:4px" onclick="setTVDataMajor('${id}','row')">Row-major</button>
              <button class="btn" style="flex:1;font-size:0.75rem;padding:4px" onclick="setTVDataMajor('${id}','col')">Col-major</button>
            </div>
            <div class="form-group">
              <label>tensor_dtype</label>
              <select id="${id}-tv-dtype-input" onchange="setTVDtype('${id}')">${tvDtypeOptions()}</select>
            </div>
            <div class="form-group">
              <button class="btn" id="${id}-tv-bank-check-btn" style="width:100%;font-size:0.75rem;padding:5px;display:flex;align-items:center;justify-content:center;gap:6px" onclick="toggleTVBankCheck('${id}')">
                <span>Check Bank Conflict (SMEM)</span>
                <span class="cuo-info-icon" onclick="event.stopPropagation()" data-tooltip="Appends #&lt;bank&gt; (the 32-bank SMEM bank id) to each cell's label. Cell color is unchanged — still keyed to thread id — so you read off T/V/bank for every (m, n) and decide for yourself which warp-instruction grouping matters. Use the bank filter to amber-edge all cells in one bank and visually count which threads land on it. Set the Swizzle input below to apply CuTe's Swizzle<B,M,S> transform to the element offset before bank is computed.">i</span>
              </button>
            </div>
            <div class="form-group">
              <label>All threads access V#<span style="color:#6b7280;font-weight:normal">&nbsp;&mdash; one wave (every thread reads vid=N); other cells gray</span></label>
              <input type="text" id="${id}-tv-wave-input" value="" placeholder="e.g. 0" oninput="setTVWaveVid('${id}')">
            </div>
            <div class="form-group">
              <label>Bank filter (0..31, empty = off) &mdash; amber edge on matching cells; only active while <b>Check Bank Conflict</b> is enabled</label>
              <input type="text" id="${id}-tv-bank-input" value="" placeholder="e.g. 7" oninput="setTVBank('${id}')">
            </div>
            <div class="form-group">
              <label>Swizzle &mdash; <code>B, M, S</code> (empty = no swizzle; e.g. <code>3, 3, 3</code>). Applied to the element offset before bank is computed</label>
              <input type="text" id="${id}-tv-swizzle-input" value="" placeholder="e.g. 3, 3, 3" oninput="setTVSwizzle('${id}')">
            </div>
          </div>
        </details>

        <div class="form-group" style="border-top:1px solid #374151;padding-top:12px">
          <label style="color:#93c5fd;letter-spacing:0.5px">&mdash; OR compute from thr/val &mdash;</label>
        </div>
        ${layoutInputField({ id: `${id}-tv-thr-input`, label: 'thr_layout', value: '', placeholder: 'e.g. (2, 3):(3, 1)' })}
        ${layoutInputField({ id: `${id}-tv-val-input`, label: 'val_layout', value: '', placeholder: 'e.g. (2, 2):(2, 1)' })}
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

// Dtype options for the bank-check section. Mirrors CUO_DTYPE_BITS (defined
// in copy_universal_op.js). The function runs at HTML-template-generation
// time inside `addOuterTab`, which fires AFTER all <script> tags have loaded,
// so the global is guaranteed to exist by then.
function tvDtypeOptions() {
  let html = '';
  for (const [name, bits] of Object.entries(CUO_DTYPE_BITS)) {
    const sel = name === 'half_t' ? ' selected' : '';
    html += `<option value="${name}"${sel}>${name} (${bits})</option>`;
  }
  return html;
}

function renderTV(tabId) {
  showErr(`${tabId}-tv-error`, '');
  try {
    const tvInput = document.getElementById(`${tabId}-tv-layout-input`).value;
    const tileInput = document.getElementById(`${tabId}-tv-tile-input`).value;
    const dataInput = document.getElementById(`${tabId}-tv-data-input`).value;
    updateRankWarning(`${tabId}-tv-warning`, [
      ['TV layout', tvInput], ['Tile', tileInput], ['Data layout', dataInput]
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
      bankToHighlight: (prev.bankToHighlight != null) ? prev.bankToHighlight : null,
      swizzle:         prev.swizzle || null,
      waveVid:         (prev.waveVid != null) ? prev.waveVid : null,
    };

    const titleHL = highlightValid ? `  \u2014  highlight T${highlightTid}` : '';
    document.getElementById(`${tabId}-tv-title`).textContent =
      `${numT} threads \u00d7 ${numV} values  \u2014  ${M}\u00d7${N} tile${titleHL}`;

    const host = document.getElementById(`${tabId}-tv-svg-host`);
    if (tvState[tabId].bankCheck) {
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
function renderTVBankSVG(tabId, tvL, tileL, numT, numV, M, N, highlightTid, labelMode) {
  const s = tvState[tabId];
  const showIdx = labelMode === 'value';
  const dataStr = document.getElementById(`${tabId}-tv-data-input`).value.trim();
  const dtype = document.getElementById(`${tabId}-tv-dtype-input`).value;
  const elemBits = CUO_DTYPE_BITS[dtype] || 16;
  const bytesPerElement = elemBits / 8;

  // Data layout maps logical flat (m + n*M) -> physical element offset.
  // Required \u2014 empty input disables the bank viz entirely.
  if (dataStr === '') {
    throw new Error(
      'Data layout is required for SMEM bank conflict check. ' +
      'Use the Row-major / Col-major buttons or enter a layout manually.');
  }
  const dp = parseLayout(dataStr);
  const sp = stripTrivialTrailing(dp.shape, dp.stride);
  const dataL = new Layout(sp.shape, sp.stride);
  if (dataL.size() !== M * N) {
    throw new Error(
      `Data layout size (${dataL.size()}) does not match tile size (${M}\u00d7${N} = ${M * N}).`);
  }

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
    const swizzled = cuoApplySwizzle(physical, sw);
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

// Toggle the bank-conflict overlay. Empty/invalid state (no tvL yet) is
// silently ignored - the next Render will pick up the new flag.
function toggleTVBankCheck(tabId) {
  if (!tvState[tabId]) tvState[tabId] = {};
  tvState[tabId].bankCheck = !tvState[tabId].bankCheck;
  const btn = document.getElementById(`${tabId}-tv-bank-check-btn`);
  if (btn) btn.classList.toggle('active', !!tvState[tabId].bankCheck);
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
  if (tvState[tabId].tvL && tvState[tabId].bankCheck) renderTV(tabId);
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
  if (tvState[tabId].tvL && tvState[tabId].bankCheck) renderTV(tabId);
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
  if (tvState[tabId].tvL && tvState[tabId].bankCheck) renderTV(tabId);
}

// Re-render when dtype changes (bank ids depend on bytes/element).
function setTVDtype(tabId) {
  if (tvState[tabId] && tvState[tabId].tvL && tvState[tabId].bankCheck) renderTV(tabId);
}

// Live update when the data layout input changes. Parse errors / size
// mismatches surface through renderTV's existing try/catch.
function setTVDataLayout(tabId) {
  if (tvState[tabId] && tvState[tabId].tvL && tvState[tabId].bankCheck) renderTV(tabId);
}

// Fill the data layout input with row-major or col-major over the current
// tile shape, so the user doesn't have to type `(M, N):(N, 1)` themselves.
// Reads M, N from the Tile input; falls back to a parse error if it's empty/bad.
function setTVDataMajor(tabId, major) {
  try {
    const tileStr = document.getElementById(`${tabId}-tv-tile-input`).value;
    const tileL = parseLayout(tileStr);
    const M = product(tileL.shape[0]);
    const N = product(tileL.shape[1]);
    const layout = major === 'row' ? `(${M}, ${N}):(${N}, 1)` : `(${M}, ${N}):(1, ${M})`;
    document.getElementById(`${tabId}-tv-data-input`).value = layout;
    if (tvState[tabId] && tvState[tabId].tvL && tvState[tabId].bankCheck) renderTV(tabId);
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
