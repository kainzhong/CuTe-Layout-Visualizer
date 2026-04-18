// Logical Divide tab: HTML panel, state, render/mode/preset/export helpers.
// Functions become globals on `window` (no module system).

function generateDivideTabContent(id) {
  return `
    <!-- Logical Divide panel -->
    <div id="${id}-tab-divide" class="panel">
      <div class="controls">
        <h2>Logical Divide</h2>
        ${layoutInputField({ id: `${id}-ld-a-input`, label: 'Layout A &mdash; the target to partition', value: '(12, 32):(32, 1)' })}
        ${layoutInputField({ id: `${id}-ld-tiler-input`, label: 'Tiler &mdash; layout or multi-line tiler (one per line)', value: '3:1\n8:1', textarea: true, rows: 2 })}
        ${statusDivs(`${id}-ld`)}
        <div id="${id}-ld-result" class="comp-result-box"></div>
        <button class="btn btn-render" onclick="renderLogicalDivide('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-ld-export" onclick="exportLD('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setLD('${id}','12:1','3:4')">1-mode: A=12:1, B=3:4 (every-4th pick)</button>
            <button class="preset-btn" onclick="setLD('${id}','(4,4):(1,4)','(2,2):(1,4)')">1-mode: A(4,4) col-maj, B=(2,2):(1,4)</button>
            <button class="preset-btn" onclick="setLD('${id}','(8,8):(1,8)','4:2\\n4:2')">2-mode: A(8,8) col-maj, &lt;4:2, 4:2&gt;</button>
            <button class="preset-btn" onclick="setLD('${id}','(12,32):(32,1)','3:4\\n8:4')">2-mode: A(12,32) row-maj, &lt;3:4, 8:4&gt;</button>
            <button class="preset-btn" onclick="setLD('${id}','(6,12):(12,1)','3:2\\n4:3')">2-mode: A(6,12) row-maj, &lt;3:2, 4:3&gt; mixed</button>
            <button class="preset-btn" onclick="setLD('${id}','(8,16):(1,8)','(2,2):(1,4)\\n(4,2):(1,8)')">2-mode: A(8,16) col-maj, nested &lt;(2,2):(1,4), (4,2):(1,8)&gt;</button>
          </div>
        </div>

        <div class="hint">
          <code>logical_divide(A, tiler)</code> partitions A by tiler. For a single-layout
          tiler <code>B</code>, the result has shape <code>(B.shape, rest.shape)</code>
          where <code>rest = complement(B, size(A))</code>. For a multi-line
          (by-mode) tiler, each mode of A is divided by the corresponding tiler line.<br><br>
          <b>Coloring:</b> cells belonging to the same tile share a color in both
          A and the result. A tile = one "slide" of the tiler over A. Mode-1 case:
          tile = set of B's outputs (1D vector across A). Mode-2 case: tile = product
          of B[0]'s mode-0 selection and B[1]'s mode-1 selection (a B0.size x B1.size
          block in the result).
        </div>
      </div>

      <div class="comp-results" style="grid-template-columns:1fr">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-ld-a-title">A</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-ld-a-mode-btns">
                <button class="mode-btn active" onclick="setLdMode('${id}','a','value')">value</button>
                <button class="mode-btn" onclick="setLdMode('${id}','a','index')">index</button>
                <button class="mode-btn" onclick="setLdMode('${id}','a','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-ld-a-svg-zoom" onclick="toggleZoom('${id}-ld-a-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-ld-a-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-ld-tiler-title">Tiler</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-ld-tiler-mode-btns">
                <button class="mode-btn active" onclick="setLdMode('${id}','tiler','value')">value</button>
                <button class="mode-btn" onclick="setLdMode('${id}','tiler','index')">index</button>
                <button class="mode-btn" onclick="setLdMode('${id}','tiler','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-ld-tiler-svg-zoom" onclick="toggleZoom('${id}-ld-tiler-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-ld-tiler-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-ld-result-title">Result = logical_divide(A, Tiler)</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-ld-result-mode-btns">
                <button class="mode-btn active" onclick="setLdMode('${id}','result','value')">value</button>
                <button class="mode-btn" onclick="setLdMode('${id}','result','index')">index</button>
                <button class="mode-btn" onclick="setLdMode('${id}','result','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-ld-result-svg-zoom" onclick="toggleZoom('${id}-ld-result-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-ld-result-svg"></div></div>
        </div>
      </div>
    </div>`;
}

const ldState = {};

// Accent colors for the "first tile" highlights. Red for mode-0, deep blue
// for mode-1 so the two axes/tilers in mode-2 are visually distinguishable.
const LD_ROW_COLOR  = '#dc2626';  // mode-0 (B0 / row axis labels / tiler[0] cell text)
const LD_COL_COLOR  = '#1d4ed8';  // mode-1 (B1 / col axis labels / tiler[1] cell text)
const LD_EDGE_COLOR = '#dc2626';  // mode-1 single-tiler edge + its cell text (red)

function renderLogicalDivide(tabId) {
  showErr(`${tabId}-ld-error`, '');
  try {
    const aStr = document.getElementById(`${tabId}-ld-a-input`).value;
    const tilerRaw = document.getElementById(`${tabId}-ld-tiler-input`).value;

    // Split tiler by newlines — multi-line = by-mode tiler
    const tilerLines = tilerRaw.split('\n').map(s => s.trim()).filter(s => s.length > 0);
    if (tilerLines.length === 0) throw new Error('Tiler input is empty');
    const isTiler = tilerLines.length > 1;

    // Rank-warning for all inputs
    const warnInputs = [['A', aStr]];
    tilerLines.forEach((line, i) =>
      warnInputs.push([tilerLines.length > 1 ? `Tiler[${i}]` : 'Tiler', line]));
    updateRankWarning(`${tabId}-ld-warning`, warnInputs);

    const aParsed = parseLayout(aStr);
    const aStripped = stripTrivialTrailing(aParsed.shape, aParsed.stride);
    const aLayout = new Layout(aStripped.shape, aStripped.stride);

    // Build tiler: either a single Layout or a tuple of Layouts
    let tilerArg, tilerLayouts;
    if (isTiler) {
      tilerLayouts = tilerLines.map(line => {
        const p = parseLayout(line);
        const s = stripTrivialTrailing(p.shape, p.stride);
        return new Layout(s.shape, s.stride);
      });
      tilerArg = tilerLayouts;  // array = tuple for composition/divide
    } else {
      const p = parseLayout(tilerLines[0]);
      const s = stripTrivialTrailing(p.shape, p.stride);
      const singleTiler = new Layout(s.shape, s.stride);
      tilerLayouts = [singleTiler];
      tilerArg = singleTiler;
    }

    // Compute the divide
    const R = logical_divide(aLayout, tilerArg);

    // Normalize to rank-2 for rendering
    const rStr = formatLayoutStr(R.shape, R.stride);
    const rParsed = parseLayout(rStr);

    // Build tile-index functions for coloring A and the result.
    // A tile represents one "slide" of the tiler over A; all cells belonging
    // to the same tile share a color in both A and the result. The tile size
    // is size(B) for mode-1, or size(B0)*size(B1) for mode-2.
    // Also build `aSelection` = the "first tile" picks (edge markers for mode-1,
    // axis-label highlights for mode-2) to show what the tiler literally selects.
    const [M_A, N_A] = productEach(aParsed.shape);
    let aTileIdxFn, rTileIdxFn, aSelection;
    if (isTiler && tilerLayouts.length === 2) {
      // Mode-2 case: tile indexed by (k0, k1), flattened as k0 + k1 * sizeC0
      const [B0, B1] = tilerLayouts;
      const sizeA0 = product(aLayout.shape[0]);
      const sizeA1 = product(aLayout.shape[1]);
      const C0 = complement(B0, sizeA0);
      const C1 = complement(B1, sizeA1);
      const sizeB0 = B0.size(), sizeB1 = B1.size();
      const sizeC0 = C0.size();
      // mode0Map: m_a -> k0 (which B0-tile this mode-0 position belongs to)
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
      // Result shape: ((sizeB0, sizeC0), (sizeB1, sizeC1)). 2D grid size is
      // (sizeB0 * sizeC0) x (sizeB1 * sizeC1). Tile index: (m_r / sizeB0, n_r / sizeB1).
      rTileIdxFn = (m_r, n_r) => {
        const k0 = Math.floor(m_r / sizeB0);
        const k1 = Math.floor(n_r / sizeB1);
        return k0 + k1 * sizeC0;
      };
      // Mode-2 selection: B0's outputs are mode-0 positions (row labels),
      // B1's outputs are mode-1 positions (column labels).
      const rows = new Set();
      for (let i = 0; i < sizeB0; i++) rows.add(B0.call(i));
      const cols = new Set();
      for (let j = 0; j < sizeB1; j++) cols.add(B1.call(j));
      aSelection = { rows, cols, rowColor: LD_ROW_COLOR, colColor: LD_COL_COLOR };
    } else {
      // Mode-1 case: single layout tiler, tile indexed by k alone
      const B = tilerLayouts[0];
      const sizeA = aLayout.size();
      const C = complement(B, sizeA);
      const sizeB = B.size();
      // valToTile: flat-1D-coord-into-A -> tile index k
      const valToTile = new Map();
      for (let k = 0; k < C.size(); k++) {
        for (let i = 0; i < sizeB; i++) valToTile.set(B.call(i) + C.call(k), k);
      }
      aTileIdxFn = (m, n) => {
        const r = valToTile.get(m + n * M_A);
        return r === undefined ? null : r;
      };
      // Result shape is (sizeB, sizeC); tile index is simply the column n_r.
      rTileIdxFn = (m_r, n_r) => n_r;
      // Mode-1 selection: edge-highlight the cells of the first tile (k=0),
      // which are the cells whose flat 1-D coord is one of B's outputs.
      const edgeCells = new Set();
      for (let i = 0; i < sizeB; i++) edgeCells.add(B.call(i));
      aSelection = { edgeCells, edgeColor: LD_EDGE_COLOR };
    }

    ldState[tabId] = {
      aParsed, aStr,
      tilerLayouts, tilerLines, isTiler,
      rParsed, rStr,
      aTileIdxFn, rTileIdxFn, aSelection,
      modes: {
        a:      new Set(['value']),
        tiler:  new Set(['value']),
        result: new Set(['value']),
      }
    };

    // Result text
    const resultEl = document.getElementById(`${tabId}-ld-result`);
    const tilerText = isTiler ? `<${tilerLines.join(', ')}>` : tilerLines[0];
    resultEl.textContent = `logical_divide(${aStr.trim()}, ${tilerText}) = ${rStr}`;
    resultEl.classList.add('visible');

    // Titles
    document.getElementById(`${tabId}-ld-a-title`).textContent = `A: ${aStr.trim()}`;
    document.getElementById(`${tabId}-ld-tiler-title`).textContent =
      isTiler ? `Tiler: ${tilerText}` : `Tiler: ${tilerLines[0]}`;
    document.getElementById(`${tabId}-ld-result-title`).textContent = `Result: ${rStr}`;

    renderLdGrid(tabId, 'a');
    renderLdGrid(tabId, 'tiler');
    renderLdGrid(tabId, 'result');

    updateOuterTabLabel(tabId, `Divide:${aStr.trim()}`);
  } catch (e) {
    showErr(`${tabId}-ld-error`, e.message);
    const resultEl = document.getElementById(`${tabId}-ld-result`);
    if (resultEl) resultEl.classList.remove('visible');
    ['a', 'tiler', 'result'].forEach(w => {
      const el = document.getElementById(`${tabId}-ld-${w}-svg`);
      if (el) el.innerHTML = '';
    });
  }
}

function renderLdGrid(tabId, which) {
  const s = ldState[tabId];
  if (!s) return;
  const modes = s.modes[which];
  const host = document.getElementById(`${tabId}-ld-${which}-svg`);

  if (which === 'tiler' && s.isTiler) {
    // Multi-line tiler: render each element stacked with a label. Tiler[0]'s cells
    // are text-colored in the row-red and Tiler[1]'s in the col-red so they visually
    // match the highlighted axis labels on A.
    let html = '';
    s.tilerLayouts.forEach((tl, i) => {
      const lStr = s.tilerLines[i];
      const tParsed = parseLayout(formatLayoutStr(tl.shape, tl.stride));
      const color = i === 0 ? LD_ROW_COLOR : LD_COL_COLOR;
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
        // Mode-1 single tiler: color cell text in the edge color (matches A's
        // red cell-edge highlights).
        const tl = s.tilerLayouts[0];
        const tParsed = parseLayout(formatLayoutStr(tl.shape, tl.stride));
        svg = buildLayoutSVG(tParsed.shape, tParsed.stride, modes, LD_EDGE_COLOR);
        break;
      }
      case 'result':
        svg = buildTiledLayoutSVG(s.rParsed.shape, s.rParsed.stride, modes, s.rTileIdxFn);
        break;
    }
    host.innerHTML = svg;
  }

  applyZoomState(`${tabId}-ld-${which}-svg`);
  updateModeBtns(`${tabId}-ld-${which}-mode-btns`, modes);
}

function setLdMode(tabId, which, mode) {
  const s = ldState[tabId];
  if (!s) return;
  const modes = s.modes[which];
  if (modes.has(mode)) {
    if (modes.size > 1) modes.delete(mode);
  } else {
    modes.add(mode);
  }
  renderLdGrid(tabId, which);
}

function setLD(tabId, a, tiler) {
  document.getElementById(`${tabId}-ld-a-input`).value = a;
  document.getElementById(`${tabId}-ld-tiler-input`).value = tiler;
  renderLogicalDivide(tabId);
}

function exportLD(tabId) {
  const a = document.getElementById(`${tabId}-ld-a-input`).value;
  const tiler = document.getElementById(`${tabId}-ld-tiler-input`).value;
  exportURL(`${tabId}-ld-export`, 'logical_divide', a, tiler);
}
