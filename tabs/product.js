// Logical Product tab: HTML panel, state, render/mode/preset/export helpers.
// Functions become globals on `window` (no module system).

function generateLogicalProductTabContent(id) {
  return `
    <!-- Logical Product panel -->
    <div id="${id}-tab-product" class="panel">
      <div class="controls">
        <h2>Logical Product</h2>
        ${layoutInputField({ id: `${id}-lp-a-input`, label: 'Layout A &mdash; the block layout to reproduce', value: '(2, 2):(1, 2)' })}
        ${layoutInputField({ id: `${id}-lp-tiler-input`, label: 'Tiler &mdash; layout or multi-line tiler (one per line)', value: '(2, 2):(1, 2)', textarea: true, rows: 2 })}
        ${statusDivs(`${id}-lp`)}
        <div id="${id}-lp-result" class="comp-result-box"></div>
        <button class="btn btn-render" onclick="renderLogicalProduct('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-lp-export" onclick="exportLP('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setLP('${id}','5:1','4:1')">1-mode: A=5:1, B=4:1 (5-block reproduced 4x -> 20)</button>
            <button class="preset-btn" onclick="setLP('${id}','(5,3):(1,5)','2:1')">1-mode: A(5,3) col-maj block, B=2:1 -> 15x2</button>
            <button class="preset-btn" onclick="setLP('${id}','(5,3):(1,5)','(2,2):(1,2)')">1-mode: A(5,3), B(2,2) -> 10x6 block grid</button>
            <button class="preset-btn" onclick="setLP('${id}','(4,3):(3,1)','2:1\\n3:1')">2-mode: A(4,3) row-maj, &lt;2:1, 3:1&gt; -> 8x9</button>
            <button class="preset-btn" onclick="setLP('${id}','(3,5):(1,3)','2:1\\n2:1')">2-mode: A(3,5) col-maj, &lt;2:1, 2:1&gt; -> 6x10</button>
            <button class="preset-btn" onclick="setLP('${id}','(4,5):(5,1)','2:1\\n2:1')">2-mode: A(4,5) row-maj, &lt;2:1, 2:1&gt; -> 8x10</button>
            <button class="preset-btn" onclick="setLP('${id}','(5,4):(1,5)','3:1\\n2:1')">2-mode: A(5,4) col-maj, &lt;3:1, 2:1&gt; -> 15x8</button>
          </div>
        </div>

        <div class="hint">
          <code>logical_product(A, tiler)</code> reproduces block layout <code>A</code>
          over <code>tiler</code>. For a single-layout tiler <code>B</code>, the result is
          <code>make_layout(A, composition(complement(A, size(A) * cosize(B)), B))</code>
          &mdash; A as the inner block, and a complement-of-A composed with B as the
          outer replication pattern. For a multi-line (by-mode) tiler, each mode of A
          is reproduced by the corresponding tiler line.<br><br>
          Think of it as the dual of <code>logical_divide</code>: divide takes a big
          layout and splits it into tiles; product takes a small block and tiles it out.<br><br>
          <b>Coloring:</b> A is one "tile" and gets color 0. The result is A
          slid across the tiler &mdash; the initial tile (at the origin) shares
          A's color 0, and each subsequent slide gets the next color. For a
          1-mode tiler, tiles march along the N axis; for a 2-mode tiler, they
          fill a 2D grid (tile = <code>t0 + t1 * size(B0)</code>).
        </div>
      </div>

      <div class="comp-results" style="grid-template-columns:1fr">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-lp-a-title">A</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-lp-a-mode-btns">
                <button class="mode-btn active" onclick="setLpMode('${id}','a','value')">value</button>
                <button class="mode-btn" onclick="setLpMode('${id}','a','index')">index</button>
                <button class="mode-btn" onclick="setLpMode('${id}','a','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-lp-a-svg-zoom" onclick="toggleZoom('${id}-lp-a-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-lp-a-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-lp-tiler-title">Tiler</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-lp-tiler-mode-btns">
                <button class="mode-btn active" onclick="setLpMode('${id}','tiler','value')">value</button>
                <button class="mode-btn" onclick="setLpMode('${id}','tiler','index')">index</button>
                <button class="mode-btn" onclick="setLpMode('${id}','tiler','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-lp-tiler-svg-zoom" onclick="toggleZoom('${id}-lp-tiler-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-lp-tiler-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-lp-result-title">Result = logical_product(A, Tiler)</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-lp-result-mode-btns">
                <button class="mode-btn active" onclick="setLpMode('${id}','result','value')">value</button>
                <button class="mode-btn" onclick="setLpMode('${id}','result','index')">index</button>
                <button class="mode-btn" onclick="setLpMode('${id}','result','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-lp-result-svg-zoom" onclick="toggleZoom('${id}-lp-result-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-lp-result-svg"></div></div>
        </div>
      </div>
    </div>`;
}

const lpState = {};

function renderLogicalProduct(tabId) {
  showErr(`${tabId}-lp-error`, '');
  try {
    const aStr = document.getElementById(`${tabId}-lp-a-input`).value;
    const tilerRaw = document.getElementById(`${tabId}-lp-tiler-input`).value;

    const tilerLines = tilerRaw.split('\n').map(s => s.trim()).filter(s => s.length > 0);
    if (tilerLines.length === 0) throw new Error('Tiler input is empty');
    const isTiler = tilerLines.length > 1;

    const warnInputs = [['A', aStr]];
    tilerLines.forEach((line, i) =>
      warnInputs.push([tilerLines.length > 1 ? `Tiler[${i}]` : 'Tiler', line]));
    updateRankWarning(`${tabId}-lp-warning`, warnInputs);

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

    let R;
    try {
      R = logical_product(aLayout, tilerArg);
    } catch (e) {
      if (/Assertion/.test(e.message)) {
        throw new Error(
          `logical_product failed: A's shape/stride is not compatible with the tiler. ` +
          `This happens when A's mode sizes/strides don't divide cleanly with the tiler's strides. ` +
          `Try a simpler tiler (e.g. shape-only "k:1" per line) or a more compatible A.`);
      }
      throw e;
    }

    const rStr = formatLayoutStr(R.shape, R.stride);
    const rParsed = parseLayout(rStr);

    // Coloring: A is one block; the tiler slides copies of A. All of A shares
    // color 0, and each tile-position in the result gets colorHighlight(k)
    // where k = 0 is the initial position (matches A), then 1, 2, ... as the
    // tiler slides.
    const aTileIdxFn = () => 0;
    let rTileIdxFn;
    if (isTiler && tilerLayouts.length === 2 && aLayout.rank() >= 2) {
      // 2-mode tiler: R.shape = ((A0, outer0), (A1, outer1)). In the 2D grid,
      // m_r axis spans size(A0)*size(B0) with A0 innermost, and similarly for
      // n_r. Tile index = t0 + t1 * size(B0).
      const sizeA0 = product(aLayout.shape[0]);
      const sizeA1 = product(aLayout.shape[1]);
      const sizeB0 = tilerLayouts[0].size();
      rTileIdxFn = (m_r, n_r) => {
        const t0 = Math.floor(m_r / sizeA0);
        const t1 = Math.floor(n_r / sizeA1);
        return t0 + t1 * sizeB0;
      };
    } else {
      // 1-mode tiler: R.shape = (A.shape, outer.shape). 2D grid is
      // size(A) x size(outer); tile index is simply the column n_r.
      rTileIdxFn = (_m_r, n_r) => n_r;
    }

    lpState[tabId] = {
      aParsed, aStr,
      tilerLayouts, tilerLines, isTiler,
      rParsed, rStr,
      aTileIdxFn, rTileIdxFn,
      modes: {
        a:      new Set(['value']),
        tiler:  new Set(['value']),
        result: new Set(['value']),
      }
    };

    const resultEl = document.getElementById(`${tabId}-lp-result`);
    const tilerText = isTiler ? `<${tilerLines.join(', ')}>` : tilerLines[0];
    resultEl.textContent = `logical_product(${aStr.trim()}, ${tilerText}) = ${rStr}`;
    resultEl.classList.add('visible');

    document.getElementById(`${tabId}-lp-a-title`).textContent = `A: ${aStr.trim()}`;
    document.getElementById(`${tabId}-lp-tiler-title`).textContent =
      isTiler ? `Tiler: ${tilerText}` : `Tiler: ${tilerLines[0]}`;
    document.getElementById(`${tabId}-lp-result-title`).textContent = `Result: ${rStr}`;

    renderLpGrid(tabId, 'a');
    renderLpGrid(tabId, 'tiler');
    renderLpGrid(tabId, 'result');

    updateOuterTabLabel(tabId, `Product:${aStr.trim()}`);
  } catch (e) {
    showErr(`${tabId}-lp-error`, e.message);
    const resultEl = document.getElementById(`${tabId}-lp-result`);
    if (resultEl) resultEl.classList.remove('visible');
    ['a', 'tiler', 'result'].forEach(w => {
      const el = document.getElementById(`${tabId}-lp-${w}-svg`);
      if (el) el.innerHTML = '';
    });
  }
}

function renderLpGrid(tabId, which) {
  const s = lpState[tabId];
  if (!s) return;
  const modes = s.modes[which];
  const host = document.getElementById(`${tabId}-lp-${which}-svg`);

  if (which === 'tiler' && s.isTiler) {
    let html = '';
    s.tilerLayouts.forEach((tl, i) => {
      const lStr = s.tilerLines[i];
      const tParsed = parseLayout(formatLayoutStr(tl.shape, tl.stride));
      html += `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin:${i > 0 ? '10px' : '0'} 0 4px">Tiler[${i}]: ${lStr}</div>`;
      html += buildLayoutSVG(tParsed.shape, tParsed.stride, modes);
    });
    host.innerHTML = html;
  } else {
    let svg;
    switch (which) {
      case 'a':
        svg = buildTiledLayoutSVG(s.aParsed.shape, s.aParsed.stride, modes, s.aTileIdxFn);
        break;
      case 'tiler': {
        const tl = s.tilerLayouts[0];
        const tParsed = parseLayout(formatLayoutStr(tl.shape, tl.stride));
        svg = buildLayoutSVG(tParsed.shape, tParsed.stride, modes);
        break;
      }
      case 'result':
        svg = buildTiledLayoutSVG(s.rParsed.shape, s.rParsed.stride, modes, s.rTileIdxFn);
        break;
    }
    host.innerHTML = svg;
  }

  applyZoomState(`${tabId}-lp-${which}-svg`);
  updateModeBtns(`${tabId}-lp-${which}-mode-btns`, modes);
}

function setLpMode(tabId, which, mode) {
  const s = lpState[tabId];
  if (!s) return;
  const modes = s.modes[which];
  if (modes.has(mode)) {
    if (modes.size > 1) modes.delete(mode);
  } else {
    modes.add(mode);
  }
  renderLpGrid(tabId, which);
}

function setLP(tabId, a, tiler) {
  document.getElementById(`${tabId}-lp-a-input`).value = a;
  document.getElementById(`${tabId}-lp-tiler-input`).value = tiler;
  renderLogicalProduct(tabId);
}

function exportLP(tabId) {
  const a = document.getElementById(`${tabId}-lp-a-input`).value;
  const tiler = document.getElementById(`${tabId}-lp-tiler-input`).value;
  exportURL(`${tabId}-lp-export`, 'logical_product', a, tiler);
}
