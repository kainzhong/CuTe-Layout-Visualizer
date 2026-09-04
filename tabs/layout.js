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
          </div>
          <h3>Presets &mdash; coordinate (TMA) layouts</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setL('${id}','(4, 5):(1@0, 1@1)')">Identity &mdash; make_identity_layout((4,5))</button>
            <button class="preset-btn" onclick="setL('${id}','(3, 4):(1@1, 1@0)')">Transpose &mdash; reversed coordinates</button>
            <button class="preset-btn" onclick="setL('${id}','(8, 8):(4@0, 1@1)')">Scaled &mdash; mode 0 steps 4 on axis 0</button>
            <button class="preset-btn" onclick="setL('${id}','(2, 2) o (4, 4):(1@0, 1@1)')">Sliced tile &mdash; origin (2,2)</button>
            <button class="preset-btn" onclick="setL('${id}','(3, 4):(1@0, 3@0)')">Both modes &rarr; axis 0 (= an ordinary layout)</button>
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
          <span style="display:flex;align-items:center;gap:4px">
            <span class="mode-btn-group" id="${id}-layout-mode-btns">
              <button class="mode-btn active" onclick="setLayoutMode('${id}','value')">value</button>
              <button class="mode-btn" onclick="setLayoutMode('${id}','index')">index</button>
              <button class="mode-btn" onclick="setLayoutMode('${id}','coord')">coord</button>
            </span>
            <button class="mode-btn" id="${id}-layout-svg-host-zoom" onclick="toggleZoom('${id}-layout-svg-host')">Zoom in</button>
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
    // This is the one tab that accepts CuTe's basis strides (`k@i`) and the
    // `<origin> o <layout>` coordinate-tensor printout — see the "Coordinate
    // layouts" note in CLAUDE.md. Everywhere else parseLayout rejects them.
    let { shape, stride, basis, origin, ndim } = parseLayout(inputVal, { basis: true });
    const [M, N] = productEach(shape);

    // A basis-strided layout maps to a coordinate, not to a 1-D offset, so the
    // scalar machinery below (bijectivity, right_inverse) does not apply.
    const stripped = basis ? null : stripTrivialTrailing(shape, stride);
    const layoutObj = basis ? null : new Layout(stripped.shape, stripped.stride);
    const bijective = basis ? false : isBijective(layoutObj);

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
      shape, stride, basis, origin, ndim,
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
      } else if (basis && ndim === 1) {
        infoEl.textContent =
          `Every stride targets axis 0, so this coordinate layout collapses to a single axis — ` +
          `it is an ordinary layout written in basis form. Integer strides are the @0 special case.`;
      } else if (basis) {
        infoEl.textContent =
          `Coordinate layout — each cell maps to a ${ndim}-D coordinate, not a 1-D offset, ` +
          `so there is nothing to invert. Colour keys off output axis 0.`;
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
  const fmtStride = x => isBasis(x) ? `${x.k}@${x.axis}`
    : Array.isArray(x) ? `[${x.map(fmtStride).join(',')}]` : String(x);
  const strideStr = fmtStride(stride);
  const title = useInverse
    ? `inverse: shape=${shapeStr}  stride=${strideStr}  \u2014  ${M}\u00d7${N} grid`
    : `shape=${shapeStr}  stride=${strideStr}  \u2014  ${M}\u00d7${N} grid`;
  document.getElementById(`${tabId}-layout-title`).textContent =
    s.basis && !useInverse
      ? `shape=${shapeStr}  basis strides  \u2014  ${M}\u00d7${N} grid, cells are ${s.ndim}-D coordinates` +
        (s.origin ? `, origin (${s.origin.join(',')})` : '')
      : title;
  document.getElementById(`${tabId}-layout-svg-host`).innerHTML =
    (s.basis && !useInverse)
      ? buildBasisLayoutSVG(shape, stride, s.ndim, s.origin, s.modes)
      : buildLayoutSVG(shape, stride, s.modes);
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
