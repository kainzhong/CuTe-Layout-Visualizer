// Layout tab: HTML panel, state, render/mode/preset/export helpers.
// Functions become globals on `window` (no module system).

function generateLayoutTabContent(id) {
  return `
    <!-- Layout panel -->
    <div id="${id}-tab-layout" class="panel active">
      <div class="controls">
        <h2>Layout</h2>
        ${layoutInputField({ id: `${id}-layout-input`, label: 'Shape : Stride', value: '(10, 10):(1, 10)' })}
        ${statusDivs(`${id}-layout`)}
        <button class="btn btn-render" onclick="renderLayout('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827;display:none" id="${id}-layout-inverse-btn" onclick="toggleLayoutInverse('${id}')">Render Inverse</button>
        <div id="${id}-layout-inverse-info" class="comp-result-box"></div>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-layout-export" onclick="exportLayout('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setL('${id}','(10, 10):(1, 10)')">Column-major (10x10)</button>
            <button class="preset-btn" onclick="setL('${id}','(10, 10):(10, 1)')">Row-major (10x10)</button>
            <button class="preset-btn" onclick="setL('${id}','(8, 8):(1, 8)')">Column-major (8x8)</button>
            <button class="preset-btn" onclick="setL('${id}','(8, 8):(8, 1)')">Row-major (8x8)</button>
            <button class="preset-btn" onclick="setL('${id}','(8, 8):(2, 16)')">Strided (8x8, s=2)</button>
            <button class="preset-btn" onclick="setL('${id}','((2,2),(2,2)):((1,4),(2,8))')">Nested ((2,2),(2,2))</button>
            <button class="preset-btn" onclick="setL('${id}','((4,2),(4,2)):((1,8),(2,32))')">Nested ((4,2),(4,2))</button>
            <button class="preset-btn" onclick="setL('${id}','(4, 32):(32, 1)')">Warp row-major (4x32)</button>
          </div>
        </div>

        <div class="hint">
          Format: <code>shape:stride</code><br>
          Examples:<br>
          <code>(M, N):(s0, s1)</code><br>
          <code>((a,b), c):((sa,sb), sc)</code><br>
          Omit <code>:stride</code> for column-major default.
        </div>
      </div>

      <div class="visualization">
        <div class="viz-header">
          <span class="viz-title" id="${id}-layout-title">&mdash;</span>
          <span style="display:flex;align-items:center;gap:8px">
            <span class="mode-btn-group" id="${id}-layout-mode-btns">
              <button class="mode-btn active" onclick="setLayoutMode('${id}','value')">value</button>
              <button class="mode-btn" onclick="setLayoutMode('${id}','index')">index</button>
              <button class="mode-btn" onclick="setLayoutMode('${id}','coord')">coord</button>
            </span>
            <button class="btn" id="${id}-layout-svg-host-zoom" onclick="toggleZoom('${id}-layout-svg-host')">Zoom in</button>
            <button class="btn" onclick="downloadSVG('${id}-layout-svg-host', 'layout.svg')">Download SVG</button>
          </span>
        </div>
        <div class="viz-box">
          <div id="${id}-layout-svg-host"></div>
        </div>
      </div>
    </div>`;
}

const layoutState = {};

function renderLayout(tabId) {
  showErr(`${tabId}-layout-error`, '');
  try {
    const inputVal = document.getElementById(`${tabId}-layout-input`).value;
    updateRankWarning(`${tabId}-layout-warning`, [['Layout', inputVal]]);
    let { shape, stride } = parseLayout(inputVal);
    const [M, N] = productEach(shape);

    // Check bijectivity on the "semantic" layout (strip trivial trailing modes)
    const stripped = stripTrivialTrailing(shape, stride);
    const layoutObj = new Layout(stripped.shape, stripped.stride);
    const bijective = isBijective(layoutObj);

    // Compute inverse layout (only valid when bijective).
    let invShape = null, invStride = null, invStr = null;
    if (bijective) {
      const inv = right_inverse(layoutObj);
      invStr = formatLayoutStr(inv.shape, inv.stride);
      const invParsed = parseLayout(invStr);
      invShape = invParsed.shape;
      invStride = invParsed.stride;
    }

    layoutState[tabId] = {
      shape, stride,
      bijective, invShape, invStride,
      showInverse: false,
      modes: new Set(['value'])
    };

    renderLayoutSVG(tabId);
    updateModeBtns(`${tabId}-layout-mode-btns`, layoutState[tabId].modes);
    updateOuterTabLabel(tabId, `Layout:${inputVal.trim()}`);

    // Show/hide the "Render Inverse" button; reset its label
    const invBtn = document.getElementById(`${tabId}-layout-inverse-btn`);
    if (invBtn) {
      invBtn.style.display = bijective ? '' : 'none';
      invBtn.textContent = 'Render Inverse';
      invBtn.classList.remove('active');
    }
    // Set the inverse text (only shown when the inverse is being rendered)
    const infoEl = document.getElementById(`${tabId}-layout-inverse-info`);
    if (infoEl) {
      if (bijective) {
        infoEl.textContent = `Left & Right Inverse = ${invStr}`;
      } else {
        infoEl.textContent = '';
      }
      infoEl.classList.remove('visible');
    }
  } catch (e) {
    showErr(`${tabId}-layout-error`, e.message);
    document.getElementById(`${tabId}-layout-svg-host`).innerHTML = '';
    const invBtn = document.getElementById(`${tabId}-layout-inverse-btn`);
    if (invBtn) invBtn.style.display = 'none';
    const infoEl = document.getElementById(`${tabId}-layout-inverse-info`);
    if (infoEl) infoEl.classList.remove('visible');
  }
}

/** Render the SVG using either the original layout or its inverse, based on state. */
function renderLayoutSVG(tabId) {
  const s = layoutState[tabId];
  if (!s) return;
  const useInverse = s.showInverse && s.invShape;
  const shape  = useInverse ? s.invShape  : s.shape;
  const stride = useInverse ? s.invStride : s.stride;
  const [M, N] = productEach(shape);
  const shapeStr  = JSON.stringify(shape).replace(/"/g, '');
  const strideStr = JSON.stringify(stride).replace(/"/g, '');
  const title = useInverse
    ? `inverse: shape=${shapeStr}  stride=${strideStr}  \u2014  ${M}\u00d7${N} grid`
    : `shape=${shapeStr}  stride=${strideStr}  \u2014  ${M}\u00d7${N} grid`;
  document.getElementById(`${tabId}-layout-title`).textContent = title;
  document.getElementById(`${tabId}-layout-svg-host`).innerHTML =
    buildLayoutSVG(shape, stride, s.modes);
  applyZoomState(`${tabId}-layout-svg-host`);
}

function toggleLayoutInverse(tabId) {
  const s = layoutState[tabId];
  if (!s || !s.bijective) return;
  s.showInverse = !s.showInverse;
  renderLayoutSVG(tabId);
  const btn = document.getElementById(`${tabId}-layout-inverse-btn`);
  if (btn) {
    btn.textContent = s.showInverse ? 'Hide Inverse' : 'Render Inverse';
    btn.classList.toggle('active', s.showInverse);
  }
  const infoEl = document.getElementById(`${tabId}-layout-inverse-info`);
  if (infoEl) infoEl.classList.toggle('visible', s.showInverse);
}

function setLayoutMode(tabId, mode) {
  const s = layoutState[tabId];
  if (!s) return;
  if (s.modes.has(mode)) {
    if (s.modes.size > 1) s.modes.delete(mode);
  } else {
    s.modes.add(mode);
  }
  renderLayoutSVG(tabId);
  updateModeBtns(`${tabId}-layout-mode-btns`, s.modes);
}

function setL(tabId, val) {
  document.getElementById(`${tabId}-layout-input`).value = val;
  renderLayout(tabId);
}

function exportLayout(tabId) {
  const val = document.getElementById(`${tabId}-layout-input`).value;
  exportURL(`${tabId}-layout-export`, 'layout', val);
}
