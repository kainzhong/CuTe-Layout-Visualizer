// Complement (standalone) tab: HTML panel, state, render/mode/preset/export
// helpers. Functions become globals on `window` (no module system).

function generateComplementTabContent(id) {
  return `
    <!-- Complement (standalone) panel -->
    <div id="${id}-tab-complement" class="panel">
      <div class="controls">
        <h2>Complement</h2>
        ${layoutInputField({ id: `${id}-cpl-layout-input`, label: 'Layout &mdash; the inner layout to tile', value: '(2, 2):(1, 2)' })}
        ${layoutInputField({ id: `${id}-cpl-cotarget-input`, label: 'Cotarget &mdash; layout or shape defining the codomain to cover', value: '(4, 4):(1, 4)' })}
        ${statusDivs(`${id}-cpl`)}
        <div id="${id}-cpl-result" class="comp-result-box"></div>
        <button class="btn btn-render" onclick="renderComplementFeature('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-cpl-export" onclick="exportCpl('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setCpl('${id}','(2,2):(1,2)','(4,4):(1,4)')">Tile (2,2):(1,2) over 4x4</button>
            <button class="preset-btn" onclick="setCpl('${id}','(4):(2)','(8):(1)')">Strided (4):(2) over 8</button>
            <button class="preset-btn" onclick="setCpl('${id}','(3):(4)','(12):(1)')">3:4 over 12</button>
            <button class="preset-btn" onclick="setCpl('${id}','(2,2):(4,1)','(4,4):(4,1)')">(2,2):(4,1) over 4x4 row-major</button>
          </div>
        </div>

        <div class="hint">
          Visualize <code>complement(L, size(cotarget))</code>: the cotarget grid is
          drawn, with L's selected cells highlighted and the tiles shifted by each
          complement offset shown faded. Together the original and shifted copies
          cover the full cotarget space.<br><br>
          Cotarget can be a shape (e.g. <code>(4, 4)</code>) or a full layout
          (e.g. <code>(4, 4):(4, 1)</code>) &mdash; the shape determines the grid
          size and the stride determines how flat offsets map to cells for display.
        </div>
      </div>

      <div class="comp-results" style="grid-template-columns:1fr">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label">Cotarget with L tiled by complement</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-cpl-main-mode-btns">
                <button class="mode-btn active" onclick="setCplMode('${id}','main','value')">value</button>
                <button class="mode-btn" onclick="setCplMode('${id}','main','index')">index</button>
                <button class="mode-btn" onclick="setCplMode('${id}','main','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-cpl-main-svg-zoom" onclick="toggleZoom('${id}-cpl-main-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-cpl-main-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-cpl-layout-title">L</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-cpl-layout-mode-btns">
                <button class="mode-btn active" onclick="setCplMode('${id}','layout','value')">value</button>
                <button class="mode-btn" onclick="setCplMode('${id}','layout','index')">index</button>
                <button class="mode-btn" onclick="setCplMode('${id}','layout','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-cpl-layout-svg-zoom" onclick="toggleZoom('${id}-cpl-layout-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-cpl-layout-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-cpl-complement-title">Complement layout</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-cpl-complement-mode-btns">
                <button class="mode-btn active" onclick="setCplMode('${id}','complement','value')">value</button>
                <button class="mode-btn" onclick="setCplMode('${id}','complement','index')">index</button>
                <button class="mode-btn" onclick="setCplMode('${id}','complement','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-cpl-complement-svg-zoom" onclick="toggleZoom('${id}-cpl-complement-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-cpl-complement-svg"></div></div>
        </div>
      </div>
    </div>`;
}

const cplState = {};

function renderComplementFeature(tabId) {
  showErr(`${tabId}-cpl-error`, '');
  try {
    const lStr = document.getElementById(`${tabId}-cpl-layout-input`).value;
    const ctStr = document.getElementById(`${tabId}-cpl-cotarget-input`).value;
    updateRankWarning(`${tabId}-cpl-warning`, [
      ['Layout', lStr], ['Cotarget', ctStr]
    ]);

    // Parse L and cotarget; cotarget can be shape-only or full layout
    const lParsed = parseLayout(lStr);
    const ctParsed = parseLayout(ctStr);
    const [M_ct, N_ct] = productEach(ctParsed.shape);
    const sizeCt = M_ct * N_ct;

    // Build Layout objects from parsed inputs (strip trailing trivial modes)
    const lStripped = stripTrivialTrailing(lParsed.shape, lParsed.stride);
    const lLayout = new Layout(lStripped.shape, lStripped.stride);

    // Compute the complement
    const C = complement(lLayout, sizeCt);

    // Collect L's output values and complement's output values
    const lOutputs = new Set();
    const lSize = lLayout.size();
    for (let i = 0; i < lSize; i++) {
      const v = lLayout.call(i);
      if (v < 0 || v >= sizeCt) {
        throw new Error(`L(${i}) = ${v} out of range [0, ${sizeCt}) for cotarget`);
      }
      lOutputs.add(v);
    }
    const anchors = new Set();
    const cSize = C.size();
    for (let i = 0; i < cSize; i++) {
      anchors.add(C.call(i));
    }

    // Build shadowMap: for each NON-zero anchor c, for each r in L.outputs, map (c+r) -> r
    const shadowMap = new Map();
    for (const c of anchors) {
      if (c === 0) continue;
      for (const r of lOutputs) {
        shadowMap.set(c + r, r);
      }
    }

    // Build highlightSet of flat positions in cotarget where cotarget(m,n) is in lOutputs.
    // The cotarget's stride tells us how to map flat offset -> (m,n).
    const highlightSet = new Set();
    for (let m = 0; m < M_ct; m++) {
      for (let n = 0; n < N_ct; n++) {
        const v = layoutAt(ctParsed.shape, ctParsed.stride, m, n);
        if (lOutputs.has(v)) highlightSet.add(m + n * M_ct);
      }
    }

    // Normalize complement for rendering
    const cStr = formatLayoutStr(C.shape, C.stride);
    const cForRender = parseLayout(cStr);

    cplState[tabId] = {
      lParsed, ctParsed, lLayout, C, cStr,
      lOutputs, shadowMap, highlightSet,
      complementShape: cForRender.shape, complementStride: cForRender.stride,
      modes: {
        main:       new Set(['value']),
        layout:     new Set(['value']),
        complement: new Set(['value']),
      }
    };

    // Result text
    const resultEl = document.getElementById(`${tabId}-cpl-result`);
    resultEl.textContent = `complement(${lStr.trim()}, ${sizeCt}) = ${cStr}`;
    resultEl.classList.add('visible');

    // Titles for the sub-viz panels
    document.getElementById(`${tabId}-cpl-layout-title`).textContent = `L: ${lStr.trim()}`;
    document.getElementById(`${tabId}-cpl-complement-title`).textContent = `Complement: ${cStr}`;

    renderCplGrid(tabId, 'main');
    renderCplGrid(tabId, 'layout');
    renderCplGrid(tabId, 'complement');

    updateOuterTabLabel(tabId, `Complement:${lStr.trim()}`);
  } catch (e) {
    showErr(`${tabId}-cpl-error`, e.message);
    const resultEl = document.getElementById(`${tabId}-cpl-result`);
    if (resultEl) resultEl.classList.remove('visible');
    ['main', 'layout', 'complement'].forEach(w => {
      const el = document.getElementById(`${tabId}-cpl-${w}-svg`);
      if (el) el.innerHTML = '';
    });
  }
}

function renderCplGrid(tabId, which) {
  const s = cplState[tabId];
  if (!s) return;
  const modes = s.modes[which];
  const host = document.getElementById(`${tabId}-cpl-${which}-svg`);
  let svg;
  switch (which) {
    case 'main':
      // Cotarget with L highlighted + complement shifts faded + anchor edges
      svg = buildHighlightedLayoutSVG(
        s.ctParsed.shape, s.ctParsed.stride,
        s.highlightSet, modes,
        /* edgeSet */ null,
        s.shadowMap);
      break;
    case 'layout':
      svg = buildLayoutSVG(s.lParsed.shape, s.lParsed.stride, modes);
      break;
    case 'complement':
      svg = buildLayoutSVG(s.complementShape, s.complementStride, modes);
      break;
  }
  host.innerHTML = svg;
  applyZoomState(`${tabId}-cpl-${which}-svg`);
  updateModeBtns(`${tabId}-cpl-${which}-mode-btns`, modes);
}

function setCplMode(tabId, which, mode) {
  const s = cplState[tabId];
  if (!s) return;
  const modes = s.modes[which];
  if (modes.has(mode)) {
    if (modes.size > 1) modes.delete(mode);
  } else {
    modes.add(mode);
  }
  renderCplGrid(tabId, which);
}

function setCpl(tabId, l, ct) {
  document.getElementById(`${tabId}-cpl-layout-input`).value = l;
  document.getElementById(`${tabId}-cpl-cotarget-input`).value = ct;
  renderComplementFeature(tabId);
}

function exportCpl(tabId) {
  const l = document.getElementById(`${tabId}-cpl-layout-input`).value;
  const ct = document.getElementById(`${tabId}-cpl-cotarget-input`).value;
  exportURL(`${tabId}-cpl-export`, 'complement', l, ct);
}
