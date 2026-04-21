// Blocked Product tab: HTML panel, state, render/mode/preset/export helpers.
// Functions become globals on `window` (no module system).

function generateBlockedProductTabContent(id) {
  return `
    <!-- Blocked Product panel -->
    <div id="${id}-tab-blocked_product" class="panel">
      <div class="controls">
        <h2>Blocked Product</h2>
        ${layoutInputField({ id: `${id}-bp-a-input`, label: 'Layout A &mdash; the block to reproduce', value: '(2, 2):(1, 2)' })}
        ${layoutInputField({ id: `${id}-bp-tiler-input`, label: 'Tiler &mdash; a single layout (block and tiler are padded to common rank)', value: '(3, 3):(1, 3)' })}
        ${statusDivs(`${id}-bp`)}
        <div id="${id}-bp-result" class="comp-result-box"></div>
        <button class="btn btn-render" onclick="renderBlockedProduct('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-bp-export" onclick="exportBP('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setBP('${id}','(2,2):(1,2)','(3,3):(1,3)')">A(2,2), tiler(3,3) -> 6x6 blocked grid</button>
            <button class="preset-btn" onclick="setBP('${id}','(2,3):(1,2)','(3,2):(1,3)')">A(2,3) col-maj, tiler(3,2) -> 6x6</button>
            <button class="preset-btn" onclick="setBP('${id}','(3,2):(2,1)','(2,3):(3,1)')">A(3,2) row-maj, tiler(2,3) row-maj -> 6x6</button>
            <button class="preset-btn" onclick="setBP('${id}','(2,2):(1,2)','(4,2):(1,4)')">A(2,2), tiler(4,2) -> 8x4 wider-in-M</button>
            <button class="preset-btn" onclick="setBP('${id}','4:1','(2,3):(1,2)')">A=4:1 (rank-1), tiler(2,3) -> 8x3</button>
            <button class="preset-btn" onclick="setBP('${id}','(2,2):(2,1)','(2,2):(1,2)')">A(2,2) row-maj, tiler(2,2) col-maj -> 4x4</button>
          </div>
        </div>

        <div class="hint">
          <code>blocked_product(A, tiler)</code> reproduces the block <code>A</code>
          across <code>tiler</code>, zipping each block mode with the corresponding
          tile mode so copies of <code>A</code> are laid down as <em>contiguous
          sub-blocks</em> of a larger matrix. Each output axis has size
          <code>size(A_i) &times; size(tiler_i)</code>.<br><br>
          Compare to <code>logical_product</code>, which keeps A on one axis and
          tile copies on another (grid is <code>size(A) &times; size(tiler)</code>).
          <code>blocked_product</code> merges them per-axis, giving the
          &ldquo;tiled matrix&rdquo; shape you&rsquo;d want when building a tile
          from a per-thread block and a thread layout. <code>tile_to_shape</code>
          in CuTe is built on top of it.<br><br>
          <b>Coloring:</b> A is one block and gets color 0. Each copy of A in the
          result is a tile; tile <code>(t0, t1)</code> (block position in the big
          matrix) gets color <code>t0 + t1 &times; size(tiler[0])</code>, so the
          tile at the top-left origin matches A.
        </div>
      </div>

      <div class="comp-results" style="grid-template-columns:1fr">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-bp-a-title">A</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-bp-a-mode-btns">
                <button class="mode-btn active" onclick="setBpMode('${id}','a','value')">value</button>
                <button class="mode-btn" onclick="setBpMode('${id}','a','index')">index</button>
                <button class="mode-btn" onclick="setBpMode('${id}','a','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-bp-a-svg-zoom" onclick="toggleZoom('${id}-bp-a-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-bp-a-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-bp-tiler-title">Tiler</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-bp-tiler-mode-btns">
                <button class="mode-btn active" onclick="setBpMode('${id}','tiler','value')">value</button>
                <button class="mode-btn" onclick="setBpMode('${id}','tiler','index')">index</button>
                <button class="mode-btn" onclick="setBpMode('${id}','tiler','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-bp-tiler-svg-zoom" onclick="toggleZoom('${id}-bp-tiler-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-bp-tiler-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-bp-result-title">Result = blocked_product(A, Tiler)</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-bp-result-mode-btns">
                <button class="mode-btn active" onclick="setBpMode('${id}','result','value')">value</button>
                <button class="mode-btn" onclick="setBpMode('${id}','result','index')">index</button>
                <button class="mode-btn" onclick="setBpMode('${id}','result','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-bp-result-svg-zoom" onclick="toggleZoom('${id}-bp-result-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-bp-result-svg"></div></div>
        </div>
      </div>
    </div>`;
}

const bpState = {};

function renderBlockedProduct(tabId) {
  showErr(`${tabId}-bp-error`, '');
  try {
    const aStr = document.getElementById(`${tabId}-bp-a-input`).value;
    const tilerStr = document.getElementById(`${tabId}-bp-tiler-input`).value;

    updateRankWarning(`${tabId}-bp-warning`, [['A', aStr], ['Tiler', tilerStr]]);

    const aParsed = parseLayout(aStr);
    const aStripped = stripTrivialTrailing(aParsed.shape, aParsed.stride);
    const aLayout = new Layout(aStripped.shape, aStripped.stride);

    const tParsedIn = parseLayout(tilerStr);
    const tStripped = stripTrivialTrailing(tParsedIn.shape, tParsedIn.stride);
    const tilerLayout = new Layout(tStripped.shape, tStripped.stride);

    let R;
    try {
      R = blocked_product(aLayout, tilerLayout);
    } catch (e) {
      if (/Assertion/.test(e.message)) {
        throw new Error(
          `blocked_product failed: A's shape/stride is not compatible with the tiler. ` +
          `This happens when A's mode sizes/strides don't divide cleanly with the tiler's strides. ` +
          `Try a simpler tiler (e.g. shape-only "(a,b):(1,a)") or a more compatible A.`);
      }
      throw e;
    }

    const rStr = formatLayoutStr(R.shape, R.stride);
    const rParsed = parseLayout(rStr);

    // Coloring: A is one block and gets color 0. In the result, each output
    // axis mode i has shape (b_i, t_i) with b_i innermost (column-major),
    // so a cell at (m_r, n_r) has:
    //   a0 = m_r % size(b0),  t0 = floor(m_r / size(b0))
    //   a1 = n_r % size(b1),  t1 = floor(n_r / size(b1))
    // Tile index (which copy of A) = t0 + t1 * size(tiler_0).
    const m0Shape = R.mode(0).shape;
    const m1Shape = R.mode(1).shape;
    const sizeB0 = product(Array.isArray(m0Shape) ? m0Shape[0] : m0Shape);
    const sizeT0 = product(Array.isArray(m0Shape) ? m0Shape[1] : 1);
    const sizeB1 = product(Array.isArray(m1Shape) ? m1Shape[0] : m1Shape);
    const aTileIdxFn = () => 0;
    const rTileIdxFn = (m_r, n_r) => {
      const t0 = Math.floor(m_r / sizeB0);
      const t1 = Math.floor(n_r / sizeB1);
      return t0 + t1 * sizeT0;
    };

    bpState[tabId] = {
      aParsed, aStr,
      tilerLayout, tilerStr,
      rParsed, rStr,
      aTileIdxFn, rTileIdxFn,
      modes: {
        a:      new Set(['value']),
        tiler:  new Set(['value']),
        result: new Set(['value']),
      }
    };

    const resultEl = document.getElementById(`${tabId}-bp-result`);
    resultEl.textContent = `blocked_product(${aStr.trim()}, ${tilerStr.trim()}) = ${rStr}`;
    resultEl.classList.add('visible');

    document.getElementById(`${tabId}-bp-a-title`).textContent = `A: ${aStr.trim()}`;
    document.getElementById(`${tabId}-bp-tiler-title`).textContent = `Tiler: ${tilerStr.trim()}`;
    document.getElementById(`${tabId}-bp-result-title`).textContent = `Result: ${rStr}`;

    renderBpGrid(tabId, 'a');
    renderBpGrid(tabId, 'tiler');
    renderBpGrid(tabId, 'result');

    updateOuterTabLabel(tabId, `BlockProduct:${aStr.trim()}`);
  } catch (e) {
    showErr(`${tabId}-bp-error`, e.message);
    const resultEl = document.getElementById(`${tabId}-bp-result`);
    if (resultEl) resultEl.classList.remove('visible');
    ['a', 'tiler', 'result'].forEach(w => {
      const el = document.getElementById(`${tabId}-bp-${w}-svg`);
      if (el) el.innerHTML = '';
    });
  }
}

function renderBpGrid(tabId, which) {
  const s = bpState[tabId];
  if (!s) return;
  const modes = s.modes[which];
  const host = document.getElementById(`${tabId}-bp-${which}-svg`);

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

  applyZoomState(`${tabId}-bp-${which}-svg`);
  updateModeBtns(`${tabId}-bp-${which}-mode-btns`, modes);
}

function setBpMode(tabId, which, mode) {
  const s = bpState[tabId];
  if (!s) return;
  const modes = s.modes[which];
  if (modes.has(mode)) {
    if (modes.size > 1) modes.delete(mode);
  } else {
    modes.add(mode);
  }
  renderBpGrid(tabId, which);
}

function setBP(tabId, a, tiler) {
  document.getElementById(`${tabId}-bp-a-input`).value = a;
  document.getElementById(`${tabId}-bp-tiler-input`).value = tiler;
  renderBlockedProduct(tabId);
}

function exportBP(tabId) {
  const a = document.getElementById(`${tabId}-bp-a-input`).value;
  const tiler = document.getElementById(`${tabId}-bp-tiler-input`).value;
  exportURL(`${tabId}-bp-export`, 'blocked_product', a, tiler);
}
