// Zipped / Tiled / Flat Product tab: HTML panel, state, render/mode/preset/export helpers.
// All three variants represent the same set of cells at the same positions;
// only the mode grouping of the returned layout differs. We render the shared
// 2D grid once (from zipped_product's result) and let the user pick which
// *textual* form to display via a dropdown.
// Functions become globals on `window` (no module system).

function generateZippedProductTabContent(id) {
  return `
    <!-- Zipped / Tiled / Flat Product panel -->
    <div id="${id}-tab-zipped_product" class="panel">
      <div class="controls">
        <h2>Zipped / Tiled / Flat Product</h2>
        ${layoutInputField({ id: `${id}-zp-a-input`, label: 'Layout A &mdash; the block to reproduce', value: '(2, 2):(1, 2)' })}
        ${layoutInputField({ id: `${id}-zp-tiler-input`, label: 'Tiler &mdash; a single layout (no layout-array tiler supported)', value: '(2, 2):(1, 2)' })}
        <div class="form-group">
          <label>Variant &mdash; pick how the result layout is grouped</label>
          <select id="${id}-zp-variant" onchange="updateZpVariant('${id}')">
            <option value="zipped">zipped_product</option>
            <option value="tiled">tiled_product</option>
            <option value="flat">flat_product</option>
          </select>
        </div>
        ${statusDivs(`${id}-zp`)}
        <div id="${id}-zp-result" class="comp-result-box"></div>
        <div id="${id}-zp-variant-note" style="font-size:0.78rem;color:#6b7280;margin-top:-8px;margin-bottom:10px">
          All three variants produce the same cells at the same positions &mdash;
          they differ only in how the returned layout's modes are grouped. The
          visualization below is identical for all three.
        </div>
        <button class="btn btn-render" onclick="renderZippedProduct('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-zp-export" onclick="exportZP('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setZP('${id}','5:1','4:1')">A=5:1, B=4:1 (5-block reproduced 4x -> 20)</button>
            <button class="preset-btn" onclick="setZP('${id}','(5,3):(1,5)','2:1')">A(5,3) col-maj block, B=2:1 -> 15x2</button>
            <button class="preset-btn" onclick="setZP('${id}','(5,3):(1,5)','(2,2):(1,2)')">A(5,3), B=(2,2):(1,2) -> 15x4</button>
            <button class="preset-btn" onclick="setZP('${id}','(4,3):(3,1)','(2,3):(1,2)')">A(4,3) row-maj, B=(2,3):(1,2) -> 12x6</button>
            <button class="preset-btn" onclick="setZP('${id}','(3,5):(1,3)','(2,2):(1,2)')">A(3,5) col-maj, B=(2,2):(1,2) -> 15x4</button>
          </div>
        </div>

        <div class="hint">
          All three variants start from
          <code>logical_product(A, B) = make_layout(A, composition(complement(A, size(A) * cosize(B)), B))</code>,
          then re-group the modes:
          <ul style="margin:6px 0 0 18px;padding:0">
            <li><code>zipped_product</code> keeps it rank-2 as <code>(block, outer)</code> with outer nested.</li>
            <li><code>tiled_product</code> unpacks <code>outer</code>'s top level, giving rank <code>1 + rank(outer)</code>.</li>
            <li><code>flat_product</code> flattens every nested mode to the top level.</li>
          </ul>
          This tab does not accept a layout-array tiler &mdash; use the Logical
          Product tab for that.<br><br>
          <b>Coloring:</b> A is one block and gets color 0. The rendered grid is
          <code>size(A) x size(B)</code> with one copy of A per column, so
          column n is tile n &mdash; column 0 shares A's color 0, column 1 gets
          color 1, and so on.
        </div>
      </div>

      <div class="comp-results" style="grid-template-columns:1fr">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-zp-a-title">A</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-zp-a-mode-btns">
                <button class="mode-btn active" onclick="setZpMode('${id}','a','value')">value</button>
                <button class="mode-btn" onclick="setZpMode('${id}','a','index')">index</button>
                <button class="mode-btn" onclick="setZpMode('${id}','a','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-zp-a-svg-zoom" onclick="toggleZoom('${id}-zp-a-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-zp-a-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-zp-tiler-title">Tiler</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-zp-tiler-mode-btns">
                <button class="mode-btn active" onclick="setZpMode('${id}','tiler','value')">value</button>
                <button class="mode-btn" onclick="setZpMode('${id}','tiler','index')">index</button>
                <button class="mode-btn" onclick="setZpMode('${id}','tiler','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-zp-tiler-svg-zoom" onclick="toggleZoom('${id}-zp-tiler-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-zp-tiler-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-zp-result-title">Result</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-zp-result-mode-btns">
                <button class="mode-btn active" onclick="setZpMode('${id}','result','value')">value</button>
                <button class="mode-btn" onclick="setZpMode('${id}','result','index')">index</button>
                <button class="mode-btn" onclick="setZpMode('${id}','result','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-zp-result-svg-zoom" onclick="toggleZoom('${id}-zp-result-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-zp-result-svg"></div></div>
        </div>
      </div>
    </div>`;
}

const zpState = {};

const ZP_VARIANT_NAMES = {
  zipped: 'zipped_product',
  tiled:  'tiled_product',
  flat:   'flat_product',
};

function renderZippedProduct(tabId) {
  showErr(`${tabId}-zp-error`, '');
  try {
    const aStr = document.getElementById(`${tabId}-zp-a-input`).value;
    const tilerStr = document.getElementById(`${tabId}-zp-tiler-input`).value;

    updateRankWarning(`${tabId}-zp-warning`, [['A', aStr], ['Tiler', tilerStr]]);

    const aParsed = parseLayout(aStr);
    const aStripped = stripTrivialTrailing(aParsed.shape, aParsed.stride);
    const aLayout = new Layout(aStripped.shape, aStripped.stride);

    const tParsedIn = parseLayout(tilerStr);
    const tStripped = stripTrivialTrailing(tParsedIn.shape, tParsedIn.stride);
    const tilerLayout = new Layout(tStripped.shape, tStripped.stride);

    let RZipped, RTiled, RFlat;
    try {
      RZipped = zipped_product(aLayout, tilerLayout);
      RTiled  = tiled_product(aLayout, tilerLayout);
      RFlat   = flat_product(aLayout, tilerLayout);
    } catch (e) {
      if (/Assertion/.test(e.message)) {
        throw new Error(
          `product failed: A's shape/stride is not compatible with the tiler. ` +
          `This happens when A's mode sizes/strides don't divide cleanly with the tiler's strides. ` +
          `Try a simpler tiler (e.g. shape-only "k:1") or a more compatible A.`);
      }
      throw e;
    }

    const resultStrs = {
      zipped: formatLayoutStr(RZipped.shape, RZipped.stride),
      tiled:  formatLayoutStr(RTiled.shape,  RTiled.stride),
      flat:   formatLayoutStr(RFlat.shape,   RFlat.stride),
    };

    // Visualization always uses zipped_product's (rank-2) form so productEach
    // flattens cleanly into a 2D grid. The three variants re-group the same
    // cells, so the picture would be identical under any of them.
    const rParsed = parseLayout(resultStrs.zipped);

    // Coloring: A is one block and gets color 0. The result is
    // size(A) x size(B) with one copy of A per column, so tile index = n_r.
    const aTileIdxFn = () => 0;
    const rTileIdxFn = (_m_r, n_r) => n_r;

    zpState[tabId] = {
      aParsed, aStr,
      tilerLayout, tilerStr,
      rParsed, resultStrs,
      aTileIdxFn, rTileIdxFn,
      modes: {
        a:      new Set(['value']),
        tiler:  new Set(['value']),
        result: new Set(['value']),
      }
    };

    document.getElementById(`${tabId}-zp-a-title`).textContent = `A: ${aStr.trim()}`;
    document.getElementById(`${tabId}-zp-tiler-title`).textContent = `Tiler: ${tilerStr.trim()}`;

    renderZpGrid(tabId, 'a');
    renderZpGrid(tabId, 'tiler');
    renderZpGrid(tabId, 'result');
    updateZpVariant(tabId);

    updateOuterTabLabel(tabId, `ZipProduct:${aStr.trim()}`);
  } catch (e) {
    showErr(`${tabId}-zp-error`, e.message);
    const resultEl = document.getElementById(`${tabId}-zp-result`);
    if (resultEl) resultEl.classList.remove('visible');
    ['a', 'tiler', 'result'].forEach(w => {
      const el = document.getElementById(`${tabId}-zp-${w}-svg`);
      if (el) el.innerHTML = '';
    });
  }
}

// Refresh only the text output (result box + result title) to match the
// current dropdown selection. The visualization is unaffected.
function updateZpVariant(tabId) {
  const s = zpState[tabId];
  if (!s) return;
  const variant = document.getElementById(`${tabId}-zp-variant`).value;
  const fnName  = ZP_VARIANT_NAMES[variant];
  const rStr    = s.resultStrs[variant];

  const resultEl = document.getElementById(`${tabId}-zp-result`);
  resultEl.textContent = `${fnName}(${s.aStr.trim()}, ${s.tilerStr.trim()}) = ${rStr}`;
  resultEl.classList.add('visible');
  document.getElementById(`${tabId}-zp-result-title`).textContent =
    `Result (${fnName}): ${rStr}`;
}

function renderZpGrid(tabId, which) {
  const s = zpState[tabId];
  if (!s) return;
  const modes = s.modes[which];
  const host = document.getElementById(`${tabId}-zp-${which}-svg`);

  let svg;
  switch (which) {
    case 'a':
      svg = buildTiledLayoutSVG(s.aParsed.shape, s.aParsed.stride, modes, s.aTileIdxFn);
      break;
    case 'tiler': {
      const tParsed = parseLayout(formatLayoutStr(s.tilerLayout.shape, s.tilerLayout.stride));
      svg = buildLayoutSVG(tParsed.shape, tParsed.stride, modes);
      break;
    }
    case 'result':
      svg = buildTiledLayoutSVG(s.rParsed.shape, s.rParsed.stride, modes, s.rTileIdxFn);
      break;
  }
  host.innerHTML = svg;

  applyZoomState(`${tabId}-zp-${which}-svg`);
  updateModeBtns(`${tabId}-zp-${which}-mode-btns`, modes);
}

function setZpMode(tabId, which, mode) {
  const s = zpState[tabId];
  if (!s) return;
  const modes = s.modes[which];
  if (modes.has(mode)) {
    if (modes.size > 1) modes.delete(mode);
  } else {
    modes.add(mode);
  }
  renderZpGrid(tabId, which);
}

function setZP(tabId, a, tiler) {
  document.getElementById(`${tabId}-zp-a-input`).value = a;
  document.getElementById(`${tabId}-zp-tiler-input`).value = tiler;
  renderZippedProduct(tabId);
}

function exportZP(tabId) {
  const a = document.getElementById(`${tabId}-zp-a-input`).value;
  const tiler = document.getElementById(`${tabId}-zp-tiler-input`).value;
  exportURL(`${tabId}-zp-export`, 'zipped_product', a, tiler);
}
