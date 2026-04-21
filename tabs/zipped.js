// Zipped / Tiled / Flat Divide tab: HTML panel, state, render/mode/preset/export helpers.
// All three variants represent the same set of cells at the same positions;
// only the mode grouping of the returned layout differs. We render the shared
// 2D grid once (from zipped_divide's result) and let the user pick which
// *textual* form to display via a dropdown.
// Functions become globals on `window` (no module system).

function generateZippedDivideTabContent(id) {
  return `
    <!-- Zipped / Tiled / Flat Divide panel -->
    <div id="${id}-tab-zipped" class="panel">
      <div class="controls">
        <h2>Zipped / Tiled / Flat Divide</h2>
        ${layoutInputField({ id: `${id}-zd-a-input`, label: 'Layout A &mdash; the target to partition', value: '(12, 32):(32, 1)' })}
        ${layoutInputField({ id: `${id}-zd-tiler-input`, label: 'Tiler &mdash; layout or multi-line tiler (one per line)', value: '3:4\n8:4', textarea: true, rows: 2 })}
        <div class="form-group">
          <label>Variant &mdash; pick how the result layout is grouped</label>
          <select id="${id}-zd-variant" onchange="updateZdVariant('${id}')">
            <option value="zipped">zipped_divide</option>
            <option value="tiled">tiled_divide</option>
            <option value="flat">flat_divide</option>
          </select>
        </div>
        ${statusDivs(`${id}-zd`)}
        <div id="${id}-zd-result" class="comp-result-box"></div>
        <div id="${id}-zd-variant-note" style="font-size:0.78rem;color:#6b7280;margin-top:-8px;margin-bottom:10px">
          All three variants produce the same cells at the same positions &mdash;
          they differ only in how the returned layout's modes are grouped. The
          visualization below is identical for all three.
        </div>
        <button class="btn btn-render" onclick="renderZippedDivide('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-zd-export" onclick="exportZD('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setZD('${id}','12:1','3:4')">1-mode: A=12:1, B=3:4 (every-4th pick)</button>
            <button class="preset-btn" onclick="setZD('${id}','(4,4):(1,4)','(2,2):(1,4)')">1-mode: A(4,4) col-maj, B=(2,2):(1,4)</button>
            <button class="preset-btn" onclick="setZD('${id}','(8,8):(1,8)','4:2\\n4:2')">2-mode: A(8,8) col-maj, &lt;4:2, 4:2&gt;</button>
            <button class="preset-btn" onclick="setZD('${id}','(12,32):(32,1)','3:4\\n8:4')">2-mode: A(12,32) row-maj, &lt;3:4, 8:4&gt;</button>
            <button class="preset-btn" onclick="setZD('${id}','(6,12):(12,1)','3:2\\n4:3')">2-mode: A(6,12) row-maj, &lt;3:2, 4:3&gt; mixed</button>
            <button class="preset-btn" onclick="setZD('${id}','(8,16):(1,8)','(2,2):(1,4)\\n(4,2):(1,8)')">2-mode: A(8,16) col-maj, nested &lt;(2,2):(1,4), (4,2):(1,8)&gt;</button>
          </div>
        </div>

        <div class="hint">
          <code>zipped_divide(A, tiler)</code> is <code>logical_divide</code> followed
          by <code>hier_unzip</code>: it partitions A by the tiler, then gathers the
          tile-local modes into a single outer mode 0 and the across-tile modes into
          a single outer mode 1. Result is always rank 2, with shape
          <code>((B0.shape, B1.shape, ...), (rest0.shape, rest1.shape, ...))</code>.<br><br>
          Compare to the Logical Divide tab: <code>logical_divide</code> keeps each
          axis's split with that axis (rank == rank(A)); <code>zipped_divide</code>
          transposes the nested structure so every "tile-local" part is in mode 0
          and every "across-tile" part is in mode 1.<br><br>
          No coloring applied yet &mdash; this view just renders A, the tiler(s),
          and the divided result as standalone layouts.
        </div>
      </div>

      <div class="comp-results" style="grid-template-columns:1fr">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-zd-a-title">A</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-zd-a-mode-btns">
                <button class="mode-btn active" onclick="setZdMode('${id}','a','value')">value</button>
                <button class="mode-btn" onclick="setZdMode('${id}','a','index')">index</button>
                <button class="mode-btn" onclick="setZdMode('${id}','a','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-zd-a-svg-zoom" onclick="toggleZoom('${id}-zd-a-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-zd-a-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-zd-tiler-title">Tiler</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-zd-tiler-mode-btns">
                <button class="mode-btn active" onclick="setZdMode('${id}','tiler','value')">value</button>
                <button class="mode-btn" onclick="setZdMode('${id}','tiler','index')">index</button>
                <button class="mode-btn" onclick="setZdMode('${id}','tiler','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-zd-tiler-svg-zoom" onclick="toggleZoom('${id}-zd-tiler-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-zd-tiler-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-zd-result-title">Result = zipped_divide(A, Tiler)</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-zd-result-mode-btns">
                <button class="mode-btn active" onclick="setZdMode('${id}','result','value')">value</button>
                <button class="mode-btn" onclick="setZdMode('${id}','result','index')">index</button>
                <button class="mode-btn" onclick="setZdMode('${id}','result','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-zd-result-svg-zoom" onclick="toggleZoom('${id}-zd-result-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-zd-result-svg"></div></div>
        </div>
      </div>
    </div>`;
}

const zdState = {};

const ZD_VARIANT_NAMES = {
  zipped: 'zipped_divide',
  tiled:  'tiled_divide',
  flat:   'flat_divide',
};

// Same accent colors as the Logical Divide tab — cells from the same tile
// get the same color in both layouts, since zipped_divide only rearranges.
const ZD_ROW_COLOR  = '#dc2626';  // mode-0 (B0 / row axis labels / tiler[0] cell text)
const ZD_COL_COLOR  = '#1d4ed8';  // mode-1 (B1 / col axis labels / tiler[1] cell text)
const ZD_EDGE_COLOR = '#dc2626';  // mode-1 single-tiler edge + its cell text (red)

function renderZippedDivide(tabId) {
  showErr(`${tabId}-zd-error`, '');
  try {
    const aStr = document.getElementById(`${tabId}-zd-a-input`).value;
    const tilerRaw = document.getElementById(`${tabId}-zd-tiler-input`).value;

    const tilerLines = tilerRaw.split('\n').map(s => s.trim()).filter(s => s.length > 0);
    if (tilerLines.length === 0) throw new Error('Tiler input is empty');
    const isTiler = tilerLines.length > 1;

    const warnInputs = [['A', aStr]];
    tilerLines.forEach((line, i) =>
      warnInputs.push([tilerLines.length > 1 ? `Tiler[${i}]` : 'Tiler', line]));
    updateRankWarning(`${tabId}-zd-warning`, warnInputs);

    const aParsed = parseLayout(aStr);
    const aStripped = stripTrivialTrailing(aParsed.shape, aParsed.stride);
    const aLayout = new Layout(aStripped.shape, aStripped.stride);

    let tilerArg, tilerLayouts;
    if (isTiler) {
      tilerLayouts = tilerLines.map(line => {
        const p = parseLayout(line);
        const s = stripTrivialTrailing(p.shape, p.stride);
        return new Layout(s.shape, s.stride);
      });
      tilerArg = tilerLayouts;
    } else {
      const p = parseLayout(tilerLines[0]);
      const s = stripTrivialTrailing(p.shape, p.stride);
      const singleTiler = new Layout(s.shape, s.stride);
      tilerLayouts = [singleTiler];
      tilerArg = singleTiler;
    }

    const RZipped = zipped_divide(aLayout, tilerArg);
    const RTiled  = tiled_divide(aLayout, tilerArg);
    const RFlat   = flat_divide(aLayout, tilerArg);

    const resultStrs = {
      zipped: formatLayoutStr(RZipped.shape, RZipped.stride),
      tiled:  formatLayoutStr(RTiled.shape,  RTiled.stride),
      flat:   formatLayoutStr(RFlat.shape,   RFlat.stride),
    };

    // Visualization always uses zipped_divide's (rank-2) form so productEach
    // flattens cleanly into a 2D grid. The three variants re-group the same
    // cells, so the picture would be identical under any of them.
    const rParsed = parseLayout(resultStrs.zipped);

    // Build tile-index functions + selection (mirrors Logical Divide exactly for A,
    // but uses the zipped arrangement for R's mode-0/mode-1 decomposition).
    const [M_A, N_A] = productEach(aParsed.shape);
    let aTileIdxFn, rTileIdxFn, aSelection;
    if (isTiler && tilerLayouts.length === 2) {
      const [B0, B1] = tilerLayouts;
      const sizeA0 = product(aLayout.shape[0]);
      const sizeA1 = product(aLayout.shape[1]);
      const C0 = complement(B0, sizeA0);
      const C1 = complement(B1, sizeA1);
      const sizeB0 = B0.size(), sizeB1 = B1.size();
      const sizeC0 = C0.size();
      const mode0Map = new Map();
      for (let k0 = 0; k0 < sizeC0; k0++) {
        for (let i = 0; i < sizeB0; i++) mode0Map.set(B0.call(i) + C0.call(k0), k0);
      }
      const mode1Map = new Map();
      for (let k1 = 0; k1 < C1.size(); k1++) {
        for (let j = 0; j < sizeB1; j++) mode1Map.set(B1.call(j) + C1.call(k1), k1);
      }
      aTileIdxFn = (m, n) => {
        const k0 = mode0Map.get(m);
        const k1 = mode1Map.get(n);
        return (k0 === undefined || k1 === undefined) ? null : k0 + k1 * sizeC0;
      };
      // Zipped R's shape is ((B0.shape, B1.shape), (C0.shape, C1.shape)).
      // Top-level grid: (sizeB0 * sizeB1) x (sizeC0 * sizeC1). Tile identity
      // is determined entirely by the mode-1 coord (m_r is tile-local):
      //   k0 = n_r % sizeC0,  k1 = floor(n_r / sizeC0)
      rTileIdxFn = (m_r, n_r) => {
        const k0 = n_r % sizeC0;
        const k1 = Math.floor(n_r / sizeC0);
        return k0 + k1 * sizeC0;
      };
      const rows = new Set();
      for (let i = 0; i < sizeB0; i++) rows.add(B0.call(i));
      const cols = new Set();
      for (let j = 0; j < sizeB1; j++) cols.add(B1.call(j));
      aSelection = { rows, cols, rowColor: ZD_ROW_COLOR, colColor: ZD_COL_COLOR };
    } else {
      const B = tilerLayouts[0];
      const sizeA = aLayout.size();
      const C = complement(B, sizeA);
      const sizeB = B.size();
      const valToTile = new Map();
      for (let k = 0; k < C.size(); k++) {
        for (let i = 0; i < sizeB; i++) valToTile.set(B.call(i) + C.call(k), k);
      }
      aTileIdxFn = (m, n) => {
        const r = valToTile.get(m + n * M_A);
        return r === undefined ? null : r;
      };
      // Mode-1 case: zipped_divide == logical_divide (hier_unzip is a no-op for
      // single tilers), so R's shape is (sizeB, sizeC). Tile index == column.
      rTileIdxFn = (m_r, n_r) => n_r;
      const edgeCells = new Set();
      for (let i = 0; i < sizeB; i++) edgeCells.add(B.call(i));
      aSelection = { edgeCells, edgeColor: ZD_EDGE_COLOR };
    }

    const tilerText = isTiler ? `<${tilerLines.join(', ')}>` : tilerLines[0];

    zdState[tabId] = {
      aParsed, aStr,
      tilerLayouts, tilerLines, isTiler, tilerText,
      rParsed, resultStrs,
      aTileIdxFn, rTileIdxFn, aSelection,
      modes: {
        a:      new Set(['value']),
        tiler:  new Set(['value']),
        result: new Set(['value']),
      }
    };

    document.getElementById(`${tabId}-zd-a-title`).textContent = `A: ${aStr.trim()}`;
    document.getElementById(`${tabId}-zd-tiler-title`).textContent =
      isTiler ? `Tiler: ${tilerText}` : `Tiler: ${tilerLines[0]}`;

    renderZdGrid(tabId, 'a');
    renderZdGrid(tabId, 'tiler');
    renderZdGrid(tabId, 'result');
    updateZdVariant(tabId);

    updateOuterTabLabel(tabId, `ZipDivide:${aStr.trim()}`);
  } catch (e) {
    showErr(`${tabId}-zd-error`, e.message);
    const resultEl = document.getElementById(`${tabId}-zd-result`);
    if (resultEl) resultEl.classList.remove('visible');
    ['a', 'tiler', 'result'].forEach(w => {
      const el = document.getElementById(`${tabId}-zd-${w}-svg`);
      if (el) el.innerHTML = '';
    });
  }
}

// Refresh only the text output (result box + result title) to match the
// current dropdown selection. The visualization is unaffected.
function updateZdVariant(tabId) {
  const s = zdState[tabId];
  if (!s) return;
  const variant = document.getElementById(`${tabId}-zd-variant`).value;
  const fnName  = ZD_VARIANT_NAMES[variant];
  const rStr    = s.resultStrs[variant];

  const resultEl = document.getElementById(`${tabId}-zd-result`);
  resultEl.textContent = `${fnName}(${s.aStr.trim()}, ${s.tilerText}) = ${rStr}`;
  resultEl.classList.add('visible');
  document.getElementById(`${tabId}-zd-result-title`).textContent =
    `Result (${fnName}): ${rStr}`;
}

function renderZdGrid(tabId, which) {
  const s = zdState[tabId];
  if (!s) return;
  const modes = s.modes[which];
  const host = document.getElementById(`${tabId}-zd-${which}-svg`);

  if (which === 'tiler' && s.isTiler) {
    // Multi-line tiler: same coloring convention as the Logical Divide tab —
    // Tiler[0] in row-red, Tiler[1] in col-blue, to match A's axis labels.
    let html = '';
    s.tilerLayouts.forEach((tl, i) => {
      const lStr = s.tilerLines[i];
      const tParsed = parseLayout(formatLayoutStr(tl.shape, tl.stride));
      const color = i === 0 ? ZD_ROW_COLOR : ZD_COL_COLOR;
      html += `<div style="font-size:0.78rem;color:${color};font-family:monospace;font-weight:600;margin:${i > 0 ? '10px' : '0'} 0 4px">Tiler[${i}]: ${lStr}</div>`;
      html += buildLayoutSVG(tParsed.shape, tParsed.stride, modes, color);
    });
    host.innerHTML = html;
  } else {
    let svg;
    switch (which) {
      case 'a':
        svg = buildTiledLayoutSVG(s.aParsed.shape, s.aParsed.stride, modes, s.aTileIdxFn, s.aSelection);
        break;
      case 'tiler': {
        const tl = s.tilerLayouts[0];
        const tParsed = parseLayout(formatLayoutStr(tl.shape, tl.stride));
        svg = buildLayoutSVG(tParsed.shape, tParsed.stride, modes, ZD_EDGE_COLOR);
        break;
      }
      case 'result':
        svg = buildTiledLayoutSVG(s.rParsed.shape, s.rParsed.stride, modes, s.rTileIdxFn);
        break;
    }
    host.innerHTML = svg;
  }

  applyZoomState(`${tabId}-zd-${which}-svg`);
  updateModeBtns(`${tabId}-zd-${which}-mode-btns`, modes);
}

function setZdMode(tabId, which, mode) {
  const s = zdState[tabId];
  if (!s) return;
  const modes = s.modes[which];
  if (modes.has(mode)) {
    if (modes.size > 1) modes.delete(mode);
  } else {
    modes.add(mode);
  }
  renderZdGrid(tabId, which);
}

function setZD(tabId, a, tiler) {
  document.getElementById(`${tabId}-zd-a-input`).value = a;
  document.getElementById(`${tabId}-zd-tiler-input`).value = tiler;
  renderZippedDivide(tabId);
}

function exportZD(tabId) {
  const a = document.getElementById(`${tabId}-zd-a-input`).value;
  const tiler = document.getElementById(`${tabId}-zd-tiler-input`).value;
  exportURL(`${tabId}-zd-export`, 'zipped_divide', a, tiler);
}
