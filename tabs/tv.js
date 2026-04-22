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

function renderTV(tabId) {
  showErr(`${tabId}-tv-error`, '');
  try {
    const tvInput = document.getElementById(`${tabId}-tv-layout-input`).value;
    const tileInput = document.getElementById(`${tabId}-tv-tile-input`).value;
    updateRankWarning(`${tabId}-tv-warning`, [
      ['TV layout', tvInput], ['Tile', tileInput]
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
    tvState[tabId] = { tvL, tileL, mode };

    const titleHL = highlightValid ? `  \u2014  highlight T${highlightTid}` : '';
    document.getElementById(`${tabId}-tv-title`).textContent =
      `${numT} threads \u00d7 ${numV} values  \u2014  ${M}\u00d7${N} tile${titleHL}`;

    document.getElementById(`${tabId}-tv-svg-host`).innerHTML =
      buildTVSVG(tvL.shape, tvL.stride, tileL.shape, tileL.stride, false, 'col',
                 highlightValid ? highlightTid : null, mode);
    applyZoomState(`${tabId}-tv-svg-host`);
    updateModeBtns(`${tabId}-tv-mode-btns`, mode ? new Set([mode]) : new Set());

    buildLegend(tabId, numT);
    updateOuterTabLabel(tabId, `TV-Layout:${tvInput.trim()}`);
  } catch (e) {
    showErr(`${tabId}-tv-error`, e.message);
    document.getElementById(`${tabId}-tv-svg-host`).innerHTML = '';
    document.getElementById(`${tabId}-tv-legend`).innerHTML = '';
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
