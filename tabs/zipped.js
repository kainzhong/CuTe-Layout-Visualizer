// Zipped Divide tab: HTML panel, state, render/mode/preset/export helpers.
// Functions become globals on `window` (no module system).

function generateZippedDivideTabContent(id) {
  return `
    <!-- Zipped Divide panel -->
    <div id="${id}-tab-zipped" class="panel">
      <div class="controls">
        <h2>Zipped Divide</h2>
        ${layoutInputField({ id: `${id}-zd-a-input`, label: 'Layout A &mdash; the target to partition', value: '(12, 32):(32, 1)' })}
        ${layoutInputField({ id: `${id}-zd-tiler-input`, label: 'Tiler &mdash; layout or multi-line tiler (one per line)', value: '3:4\n8:4', textarea: true, rows: 2 })}
        ${statusDivs(`${id}-zd`)}
        <div id="${id}-zd-result" class="comp-result-box"></div>
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

    const R = zipped_divide(aLayout, tilerArg);

    const rStr = formatLayoutStr(R.shape, R.stride);
    const rParsed = parseLayout(rStr);

    zdState[tabId] = {
      aParsed, aStr,
      tilerLayouts, tilerLines, isTiler,
      rParsed, rStr,
      modes: {
        a:      new Set(['value']),
        tiler:  new Set(['value']),
        result: new Set(['value']),
      }
    };

    const resultEl = document.getElementById(`${tabId}-zd-result`);
    const tilerText = isTiler ? `<${tilerLines.join(', ')}>` : tilerLines[0];
    resultEl.textContent = `zipped_divide(${aStr.trim()}, ${tilerText}) = ${rStr}`;
    resultEl.classList.add('visible');

    document.getElementById(`${tabId}-zd-a-title`).textContent = `A: ${aStr.trim()}`;
    document.getElementById(`${tabId}-zd-tiler-title`).textContent =
      isTiler ? `Tiler: ${tilerText}` : `Tiler: ${tilerLines[0]}`;
    document.getElementById(`${tabId}-zd-result-title`).textContent = `Result: ${rStr}`;

    renderZdGrid(tabId, 'a');
    renderZdGrid(tabId, 'tiler');
    renderZdGrid(tabId, 'result');

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

function renderZdGrid(tabId, which) {
  const s = zdState[tabId];
  if (!s) return;
  const modes = s.modes[which];
  const host = document.getElementById(`${tabId}-zd-${which}-svg`);

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
        svg = buildLayoutSVG(s.aParsed.shape, s.aParsed.stride, modes);
        break;
      case 'tiler': {
        const tl = s.tilerLayouts[0];
        const tParsed = parseLayout(formatLayoutStr(tl.shape, tl.stride));
        svg = buildLayoutSVG(tParsed.shape, tParsed.stride, modes);
        break;
      }
      case 'result':
        svg = buildLayoutSVG(s.rParsed.shape, s.rParsed.stride, modes);
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
