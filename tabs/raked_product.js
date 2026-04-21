// Raked Product tab: HTML panel, state, render/mode/preset/export helpers.
// Functions become globals on `window` (no module system).

function generateRakedProductTabContent(id) {
  return `
    <!-- Raked Product panel -->
    <div id="${id}-tab-raked_product" class="panel">
      <div class="controls">
        <h2>Raked Product</h2>
        ${layoutInputField({ id: `${id}-rp-a-input`, label: 'Layout A &mdash; the block to reproduce (interleaved)', value: '(2, 2):(1, 2)' })}
        ${layoutInputField({ id: `${id}-rp-tiler-input`, label: 'Tiler &mdash; a single layout (block and tiler are padded to common rank)', value: '(3, 3):(1, 3)' })}
        ${statusDivs(`${id}-rp`)}
        <div id="${id}-rp-result" class="comp-result-box"></div>
        <button class="btn btn-render" onclick="renderRakedProduct('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-rp-export" onclick="exportRP('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setRP('${id}','(2,2):(1,2)','(3,3):(1,3)')">A(2,2), tiler(3,3) -> 6x6 raked</button>
            <button class="preset-btn" onclick="setRP('${id}','(2,3):(1,2)','(3,2):(1,3)')">A(2,3), tiler(3,2) -> 6x6</button>
            <button class="preset-btn" onclick="setRP('${id}','(3,2):(2,1)','(2,3):(3,1)')">A(3,2) row-maj, tiler(2,3) row-maj -> 6x6</button>
            <button class="preset-btn" onclick="setRP('${id}','(2,2):(1,2)','(4,2):(1,4)')">A(2,2), tiler(4,2) -> 8x4</button>
            <button class="preset-btn" onclick="setRP('${id}','(2,2):(2,1)','(2,2):(1,2)')">A(2,2) row-maj, tiler(2,2) col-maj -> 4x4</button>
            <button class="preset-btn" onclick="setRP('${id}','4:1','(2,3):(1,2)')">A=4:1 (rank-1), tiler(2,3) -> 8x3</button>
          </div>
        </div>

        <div class="hint">
          <code>raked_product(A, tiler)</code> is the block-interleaved twin of
          <code>blocked_product</code>. It lays down the same set of cells as
          <code>logical_product</code>, but zips <em>tile-mode first, block-mode
          second</em>, so cells of a single copy of A are spread (raked) across
          the output tile instead of clumped into a contiguous sub-block.<br><br>
          This is what <code>make_layout_tv</code> uses to build thread-value
          layouts: each thread "owns" a set of cells distributed across the tile
          rather than a contiguous chunk &mdash; which is what you want for
          coalesced memory access in GPU kernels.<br><br>
          <b>Coloring:</b> A is one block and gets color 0. In the result, cells
          belonging to the <em>same</em> copy of A share a color, and different
          copies get different colors. Because of the rake, cells with the same
          color are scattered across the tile at stride <code>size(tiler_i)</code>
          along each axis. Compare the color pattern here to the Blocked Product
          tab with the same inputs &mdash; same set of colors, same counts, but
          scattered instead of clumped.
        </div>
      </div>

      <div class="comp-results" style="grid-template-columns:1fr">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-rp-a-title">A</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-rp-a-mode-btns">
                <button class="mode-btn active" onclick="setRpMode('${id}','a','value')">value</button>
                <button class="mode-btn" onclick="setRpMode('${id}','a','index')">index</button>
                <button class="mode-btn" onclick="setRpMode('${id}','a','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-rp-a-svg-zoom" onclick="toggleZoom('${id}-rp-a-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-rp-a-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-rp-tiler-title">Tiler</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-rp-tiler-mode-btns">
                <button class="mode-btn active" onclick="setRpMode('${id}','tiler','value')">value</button>
                <button class="mode-btn" onclick="setRpMode('${id}','tiler','index')">index</button>
                <button class="mode-btn" onclick="setRpMode('${id}','tiler','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-rp-tiler-svg-zoom" onclick="toggleZoom('${id}-rp-tiler-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-rp-tiler-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-rp-result-title">Result = raked_product(A, Tiler)</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-rp-result-mode-btns">
                <button class="mode-btn active" onclick="setRpMode('${id}','result','value')">value</button>
                <button class="mode-btn" onclick="setRpMode('${id}','result','index')">index</button>
                <button class="mode-btn" onclick="setRpMode('${id}','result','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-rp-result-svg-zoom" onclick="toggleZoom('${id}-rp-result-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-rp-result-svg"></div></div>
        </div>
      </div>
    </div>`;
}

const rpState = {};

function renderRakedProduct(tabId) {
  showErr(`${tabId}-rp-error`, '');
  try {
    const aStr = document.getElementById(`${tabId}-rp-a-input`).value;
    const tilerStr = document.getElementById(`${tabId}-rp-tiler-input`).value;

    updateRankWarning(`${tabId}-rp-warning`, [['A', aStr], ['Tiler', tilerStr]]);

    const aParsed = parseLayout(aStr);
    const aStripped = stripTrivialTrailing(aParsed.shape, aParsed.stride);
    const aLayout = new Layout(aStripped.shape, aStripped.stride);

    const tParsedIn = parseLayout(tilerStr);
    const tStripped = stripTrivialTrailing(tParsedIn.shape, tParsedIn.stride);
    const tilerLayout = new Layout(tStripped.shape, tStripped.stride);

    // raked_product uses R = max(rank(block), rank(tiler)); pre-pad both to
    // at least rank 2 so rank-1 inputs still produce a 2D-renderable result.
    const R_rank = Math.max(aLayout.rank(), tilerLayout.rank(), 2);
    const aPadded = append_layout(aLayout, R_rank);
    const tPadded = append_layout(tilerLayout, R_rank);

    let R;
    try {
      R = raked_product(aPadded, tPadded);
    } catch (e) {
      if (/Assertion/.test(e.message)) {
        throw new Error(
          `raked_product failed: A's shape/stride is not compatible with the tiler. ` +
          `This happens when A's mode sizes/strides don't divide cleanly with the tiler's strides. ` +
          `Try a simpler tiler (e.g. shape-only "(a,b):(1,a)") or a more compatible A.`);
      }
      throw e;
    }

    const rStr = formatLayoutStr(R.shape, R.stride);
    const rParsed = parseLayout(rStr);

    // Coloring: A is one block and gets color 0. raked_product's output has
    // mode i = (tile_i, block_i) (zip order reversed from blocked_product),
    // so under column-major unflatten tile-index varies *fastest*:
    //   t0 = m_r % size(tile_0),  a0 = floor(m_r / size(tile_0))
    //   t1 = n_r % size(tile_1),  a1 = floor(n_r / size(tile_1))
    // Cells of one A copy share (t0, t1), so tile index = t0 + t1 * size(tiler_0).
    const m0Shape = R.mode(0).shape;
    const m1Shape = R.mode(1).shape;
    const sizeT0 = product(Array.isArray(m0Shape) ? m0Shape[0] : m0Shape);
    const sizeT1 = product(Array.isArray(m1Shape) ? m1Shape[0] : m1Shape);
    const aTileIdxFn = () => 0;
    const rTileIdxFn = (m_r, n_r) => {
      const t0 = m_r % sizeT0;
      const t1 = n_r % sizeT1;
      return t0 + t1 * sizeT0;
    };

    rpState[tabId] = {
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

    const resultEl = document.getElementById(`${tabId}-rp-result`);
    resultEl.textContent = `raked_product(${aStr.trim()}, ${tilerStr.trim()}) = ${rStr}`;
    resultEl.classList.add('visible');

    document.getElementById(`${tabId}-rp-a-title`).textContent = `A: ${aStr.trim()}`;
    document.getElementById(`${tabId}-rp-tiler-title`).textContent = `Tiler: ${tilerStr.trim()}`;
    document.getElementById(`${tabId}-rp-result-title`).textContent = `Result: ${rStr}`;

    renderRpGrid(tabId, 'a');
    renderRpGrid(tabId, 'tiler');
    renderRpGrid(tabId, 'result');

    updateOuterTabLabel(tabId, `RakedProduct:${aStr.trim()}`);
  } catch (e) {
    showErr(`${tabId}-rp-error`, e.message);
    const resultEl = document.getElementById(`${tabId}-rp-result`);
    if (resultEl) resultEl.classList.remove('visible');
    ['a', 'tiler', 'result'].forEach(w => {
      const el = document.getElementById(`${tabId}-rp-${w}-svg`);
      if (el) el.innerHTML = '';
    });
  }
}

function renderRpGrid(tabId, which) {
  const s = rpState[tabId];
  if (!s) return;
  const modes = s.modes[which];
  const host = document.getElementById(`${tabId}-rp-${which}-svg`);

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

  applyZoomState(`${tabId}-rp-${which}-svg`);
  updateModeBtns(`${tabId}-rp-${which}-mode-btns`, modes);
}

function setRpMode(tabId, which, mode) {
  const s = rpState[tabId];
  if (!s) return;
  const modes = s.modes[which];
  if (modes.has(mode)) {
    if (modes.size > 1) modes.delete(mode);
  } else {
    modes.add(mode);
  }
  renderRpGrid(tabId, which);
}

function setRP(tabId, a, tiler) {
  document.getElementById(`${tabId}-rp-a-input`).value = a;
  document.getElementById(`${tabId}-rp-tiler-input`).value = tiler;
  renderRakedProduct(tabId);
}

function exportRP(tabId) {
  const a = document.getElementById(`${tabId}-rp-a-input`).value;
  const tiler = document.getElementById(`${tabId}-rp-tiler-input`).value;
  exportURL(`${tabId}-rp-export`, 'raked_product', a, tiler);
}
