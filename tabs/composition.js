// Composition & Complement (toggle) tab: HTML panel, state, render/mode
// helpers, plus the toggle/show/hide helpers for the optional complement
// section. Functions become globals on `window` (no module system).

function generateCompositionTabContent(id) {
  return `
    <!-- Composition panel -->
    <div id="${id}-tab-composition" class="panel">
      <div class="controls">
        <h2>Composition &amp; Complement</h2>
        ${layoutInputField({ id: `${id}-comp-a-input`, label: 'Layout A &mdash; the outer layout', value: '(4, 4):(4, 1)' })}
        ${layoutInputField({ id: `${id}-comp-b-input`, label: 'Layout B &mdash; layout or tiler (one layout per line for by-mode tiler)', value: '(2, 2):(1, 2)', textarea: true, rows: 2 })}
        ${statusDivs(`${id}-comp`)}
        <div id="${id}-comp-result" class="comp-result-box"></div>
        <button class="btn btn-render" onclick="renderComposition('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-comp-complement-btn" onclick="toggleComplement('${id}')">Render complement</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-comp-export" onclick="exportComp('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setComp('${id}','(4,4):(4,1)','(2,2):(1,2)')">Row-maj A(4x4), Col-maj B(2x2)</button>
            <button class="preset-btn" onclick="setComp('${id}','(8,8):(1,8)','(4,4):(1,4)')">Col-maj A(8x8), Col-maj B(4x4)</button>
            <button class="preset-btn" onclick="setComp('${id}','(8,8):(1,8)','(2,4):(2,8)')">Col-maj A(8x8), Strided B(2x4)</button>
            <button class="preset-btn" onclick="setComp('${id}','(4,8):(8,1)','(2,4):(1,2)')">Row-maj A(4x8), Col-maj B(2x4)</button>
            <button class="preset-btn" onclick="setComp('${id}','(4,4):(1,4)','(4,4):(1,0)')">Broadcast: B stride-0 col</button>
            <button class="preset-btn" onclick="setComp('${id}','(12,(4,8)):(59,(13,1))','(3):(4)\\n(8):(2)')">Tiler &lt;3:4, 8:2&gt; on A(12x32)</button>
            <button class="preset-btn" onclick="setComp('${id}','(12,(4,8)):(59,(13,1))','3\\n8')">Shape-tiler &lt;3:1, 8:1&gt; on A(12x32)</button>
            <button class="preset-btn" onclick="setComp('${id}','(4,8):(8,1)','2\\n4')">Shape-tiler &lt;2:1, 4:1&gt; on A(4x8)</button>
          </div>
        </div>

        <div class="hint">
          B can be a single layout or multiple layouts
          (one per line) for by-mode tiler composition.<br><br>
          <b>Single layout</b>: <code>R(c) = A(B(c))</code> &mdash;
          B's output is a 1-D coordinate into A.<br>
          <b>Tiler (multi-line)</b>: each line applies to one mode
          of A independently. E.g. line 1 = <code>(3):(4)</code>,
          line 2 = <code>(8):(2)</code> gives tiler
          <code>&lt;3:4, 8:2&gt;</code>.<br>
          A shape like <code>3</code> is treated as <code>3:1</code>
          (stride-1).<br><br>
          Non-injective B (e.g. stride 0) gives broadcasting behavior.
        </div>
      </div>

      <div class="comp-results">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-comp-a-title">A</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-comp-a-mode-btns">
                <button class="mode-btn active" onclick="setCompMode('${id}','a','value')">value</button>
                <button class="mode-btn" onclick="setCompMode('${id}','a','index')">index</button>
                <button class="mode-btn" onclick="setCompMode('${id}','a','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-comp-a-svg-zoom" onclick="toggleZoom('${id}-comp-a-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-comp-a-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-comp-b-title">B</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-comp-b-mode-btns">
                <button class="mode-btn active" onclick="setCompMode('${id}','b','value')">value</button>
                <button class="mode-btn" onclick="setCompMode('${id}','b','index')">index</button>
                <button class="mode-btn" onclick="setCompMode('${id}','b','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-comp-b-svg-zoom" onclick="toggleZoom('${id}-comp-b-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-comp-b-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label">A highlighted by B</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-comp-highlight-mode-btns">
                <button class="mode-btn active" onclick="setCompMode('${id}','highlight','value')">value</button>
                <button class="mode-btn" onclick="setCompMode('${id}','highlight','index')">index</button>
                <button class="mode-btn" onclick="setCompMode('${id}','highlight','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-comp-highlight-svg-zoom" onclick="toggleZoom('${id}-comp-highlight-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-comp-highlight-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-comp-r-title">R = A(B)</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-comp-r-mode-btns">
                <button class="mode-btn active" onclick="setCompMode('${id}','r','value')">value</button>
                <button class="mode-btn" onclick="setCompMode('${id}','r','index')">index</button>
                <button class="mode-btn" onclick="setCompMode('${id}','r','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-comp-r-svg-zoom" onclick="toggleZoom('${id}-comp-r-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-comp-r-svg"></div></div>
        </div>
        <div id="${id}-comp-complement-section" class="comp-viz-item comp-viz-complement" style="display:none">
          <div class="comp-viz-header">
            <span class="comp-viz-label">Complement layout &mdash; tiles R over A to cover the full coordinate space</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-comp-complement-mode-btns">
                <button class="mode-btn active" onclick="setCompMode('${id}','complement','value')">value</button>
                <button class="mode-btn" onclick="setCompMode('${id}','complement','index')">index</button>
                <button class="mode-btn" onclick="setCompMode('${id}','complement','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-comp-complement-svg-zoom" onclick="toggleZoom('${id}-comp-complement-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-comp-complement-svg"></div></div>
        </div>
      </div>
    </div>`;
}

const compState = {};

function renderComposition(tabId) {
  showErr(`${tabId}-comp-error`, '');
  const svgIds = ['comp-a-svg', 'comp-b-svg', 'comp-highlight-svg', 'comp-r-svg'];

  try {
    const aInput = document.getElementById(`${tabId}-comp-a-input`).value;
    const bRaw = document.getElementById(`${tabId}-comp-b-input`).value;
    const aL = parseLayout(aInput);
    const [M_A, N_A] = productEach(aL.shape);

    const bLines = bRaw.split('\n').map(s => s.trim()).filter(s => s.length > 0);
    if (bLines.length === 0) throw new Error('B input is empty');

    // Warn if any input has outer rank > 2
    const warnInputs = [['A', aInput]];
    bLines.forEach((line, i) => warnInputs.push([bLines.length > 1 ? `B[${i}]` : 'B', line]));
    updateRankWarning(`${tabId}-comp-warning`, warnInputs);

    const isTiler = bLines.length > 1;
    let rGrid, highlightSet, M_R, N_R, bL, bLayouts;

    if (isTiler) {
      if (bLines.length !== 2) {
        throw new Error(
          `Tiler must have exactly 2 layouts (one per mode of rank-2 A), got ${bLines.length}`);
      }
      bLayouts = bLines.map(line => parseLayout(line));
      bL = null;

      const sizeA0 = product(aL.shape[0]);
      const sizeA1 = product(aL.shape[1]);

      const sizeB0 = product(bLayouts[0].shape[0]) * product(bLayouts[0].shape[1]);
      const r0Vals = [], b0Out = new Set();
      for (let i = 0; i < sizeB0; i++) {
        const v = evalLayoutFlat(bLayouts[0], i);
        if (v < 0 || v >= sizeA0)
          throw new Error(`B[0](${i}) = ${v} out of range [0, ${sizeA0}) for A mode-0`);
        b0Out.add(v);
        r0Vals.push(evalModeAt(aL.shape[0], aL.stride[0], v));
      }

      const sizeB1 = product(bLayouts[1].shape[0]) * product(bLayouts[1].shape[1]);
      const r1Vals = [], b1Out = new Set();
      for (let j = 0; j < sizeB1; j++) {
        const v = evalLayoutFlat(bLayouts[1], j);
        if (v < 0 || v >= sizeA1)
          throw new Error(`B[1](${j}) = ${v} out of range [0, ${sizeA1}) for A mode-1`);
        b1Out.add(v);
        r1Vals.push(evalModeAt(aL.shape[1], aL.stride[1], v));
      }

      M_R = sizeB0;
      N_R = sizeB1;

      rGrid = [];
      for (let i = 0; i < M_R; i++) {
        rGrid[i] = [];
        for (let j = 0; j < N_R; j++)
          rGrid[i][j] = r0Vals[i] + r1Vals[j];
      }

      highlightSet = new Set();
      for (const m of b0Out)
        for (const n of b1Out)
          highlightSet.add(m + n * M_A);

    } else {
      bL = parseLayout(bLines[0]);
      bLayouts = null;

      const [M_B, N_B] = productEach(bL.shape);
      const sizeA = M_A * N_A;

      highlightSet = new Set();
      for (let m = 0; m < M_B; m++) {
        for (let n = 0; n < N_B; n++) {
          const bVal = layoutAt(bL.shape, bL.stride, m, n);
          if (bVal < 0 || bVal >= sizeA)
            throw new Error(`B(${m},${n}) = ${bVal} out of range [0, ${sizeA}) for A (size ${M_A}x${N_A})`);
          highlightSet.add(bVal);
        }
      }

      rGrid = [];
      for (let m = 0; m < M_B; m++) {
        rGrid[m] = [];
        for (let n = 0; n < N_B; n++) {
          const bVal = layoutAt(bL.shape, bL.stride, m, n);
          rGrid[m][n] = layoutAt(aL.shape, aL.stride, bVal % M_A, Math.floor(bVal / M_A));
        }
      }
      M_R = M_B;
      N_R = N_B;
    }

    compState[tabId] = {
      aL, bL, bLayouts, isTiler, bInputLines: bLines,
      rGrid, highlightSet, M_R, N_R,
      modes: {
        a: new Set(['value']), b: new Set(['value']),
        highlight: new Set(['value']), r: new Set(['value']),
        complement: new Set(['value'])
      }
    };

    // Hide complement section on fresh render; user must click "Render complement"
    const compSection = document.getElementById(`${tabId}-comp-complement-section`);
    if (compSection) compSection.style.display = 'none';
    const compBtn = document.getElementById(`${tabId}-comp-complement-btn`);
    if (compBtn) { compBtn.textContent = 'Render complement'; compBtn.classList.remove('active'); }

    document.getElementById(`${tabId}-comp-a-title`).textContent = `A: ${aInput.trim()}`;
    document.getElementById(`${tabId}-comp-b-title`).textContent =
      isTiler ? `Tiler: <${bLines.join(', ')}>` : `B: ${bLines[0]}`;
    document.getElementById(`${tabId}-comp-r-title`).textContent =
      isTiler ? 'R = A(Tiler)' : 'R = A(B)';

    renderCompGrid(tabId, 'a');
    renderCompGrid(tabId, 'b');
    renderCompGrid(tabId, 'highlight');
    renderCompGrid(tabId, 'r');

    // Algebraic composition R = A(B) via layout.js
    const resultEl = document.getElementById(`${tabId}-comp-result`);
    try {
      const aStripped = stripTrivialTrailing(aL.shape, aL.stride);
      const aLayout = new Layout(aStripped.shape, aStripped.stride);
      let R;
      if (isTiler) {
        const bTuple = bLayouts.map(b => {
          const p = stripTrivialTrailing(b.shape, b.stride);
          return new Layout(p.shape, p.stride);
        });
        R = composition(aLayout, bTuple);
      } else {
        const bStripped = stripTrivialTrailing(bL.shape, bL.stride);
        const bLayout = new Layout(bStripped.shape, bStripped.stride);
        R = composition(aLayout, bLayout);
      }
      const label = isTiler ? 'R = A(Tiler)' : 'R = A(B)';
      resultEl.textContent = `${label} = ${formatLayoutStr(R.shape, R.stride)}`;
      resultEl.classList.add('visible');
    } catch (e) {
      resultEl.textContent = `Algebraic composition unavailable: ${e.message}`;
      resultEl.classList.add('visible');
    }

    updateOuterTabLabel(tabId, `Compose:${aInput.trim()}`);
  } catch (e) {
    showErr(`${tabId}-comp-error`, e.message);
    svgIds.forEach(h => {
      const el = document.getElementById(`${tabId}-${h}`);
      if (el) el.innerHTML = '';
    });
    const resultEl = document.getElementById(`${tabId}-comp-result`);
    if (resultEl) resultEl.classList.remove('visible');
  }
}

function renderCompGrid(tabId, which) {
  const s = compState[tabId];
  if (!s) return;
  const modes = s.modes[which];
  const host = document.getElementById(`${tabId}-comp-${which}-svg`);

  if (which === 'b' && s.isTiler) {
    let html = '';
    s.bLayouts.forEach((bl, i) => {
      html += `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin:${i > 0 ? '10px' : '0'} 0 4px">B[${i}]: ${s.bInputLines[i]}</div>`;
      html += buildLayoutSVG(bl.shape, bl.stride, modes);
    });
    host.innerHTML = html;
  } else {
    let svg;
    switch (which) {
      case 'a':
        svg = buildLayoutSVG(s.aL.shape, s.aL.stride, modes);
        break;
      case 'b':
        svg = buildLayoutSVG(s.bL.shape, s.bL.stride, modes);
        break;
      case 'highlight':
        svg = buildHighlightedLayoutSVG(s.aL.shape, s.aL.stride, s.highlightSet, modes, s.complementAnchors, s.shadowMap);
        break;
      case 'r':
        svg = buildGridSVG(s.rGrid, s.M_R, s.N_R, modes);
        break;
      case 'complement':
        if (s.complementShape) {
          // Every cell in the complement viz is an anchor — border them in the
          // same amber as the cell edges on "A highlighted by B" so the
          // correspondence is visually obvious.
          svg = buildLayoutSVG(s.complementShape, s.complementStride, modes, null, '#f59e0b');
        } else {
          svg = '';
        }
        break;
    }
    host.innerHTML = svg;
  }

  applyZoomState(`${tabId}-comp-${which}-svg`);
  updateModeBtns(`${tabId}-comp-${which}-mode-btns`, modes);
}

function setCompMode(tabId, which, mode) {
  const s = compState[tabId];
  if (!s) return;
  const modes = s.modes[which];
  if (modes.has(mode)) {
    if (modes.size > 1) modes.delete(mode);
  } else {
    modes.add(mode);
  }
  renderCompGrid(tabId, which);
}

/** Compute complement(R, size(A)) where R = A(B), and visualize it.
 *  - Appends "complement(R, A) = ..." to the result box
 *  - Shows a new viz below the 4-panel grid with the complement layout
 *  - Adds amber edge markers to "A highlighted by B" at each anchor offset
 *    (cells whose A(m,n) equals one of the complement's output values). */
function toggleComplement(tabId) {
  const s = compState[tabId];
  if (!s) {
    showErr(`${tabId}-comp-error`, 'Click Render first before Render complement.');
    return;
  }
  if (s.shadowMap) hideComplement(tabId);
  else             showComplement(tabId);
}

function hideComplement(tabId) {
  const s = compState[tabId];
  if (!s) return;
  delete s.shadowMap;
  delete s.complementAnchors;
  delete s.complementShape;
  delete s.complementStride;
  // Restore the result text (strip the complement line)
  const resultEl = document.getElementById(`${tabId}-comp-result`);
  if (s.preComplementResultText !== undefined) {
    resultEl.textContent = s.preComplementResultText;
    delete s.preComplementResultText;
  }
  // Hide the complement viz section
  document.getElementById(`${tabId}-comp-complement-section`).style.display = 'none';
  // Re-render the highlighted A to remove edges + shadow tiles
  renderCompGrid(tabId, 'highlight');
  // Update button state
  const btn = document.getElementById(`${tabId}-comp-complement-btn`);
  if (btn) { btn.textContent = 'Render complement'; btn.classList.remove('active'); }
}

function showComplement(tabId) {
  const s = compState[tabId];
  if (!s) return;
  showErr(`${tabId}-comp-error`, '');
  try {
    // Reconstruct R = A(B) via layout.js composition
    const aStripped = stripTrivialTrailing(s.aL.shape, s.aL.stride);
    const aLayout = new Layout(aStripped.shape, aStripped.stride);
    let R;
    if (s.isTiler) {
      const bTuple = s.bLayouts.map(b => {
        const p = stripTrivialTrailing(b.shape, b.stride);
        return new Layout(p.shape, p.stride);
      });
      R = composition(aLayout, bTuple);
    } else {
      const bStripped = stripTrivialTrailing(s.bL.shape, s.bL.stride);
      const bLayout = new Layout(bStripped.shape, bStripped.stride);
      R = composition(aLayout, bLayout);
    }

    const sizeA = aLayout.size();
    const C = complement(R, sizeA);

    // Collect anchor offsets: C(i) for i in [0, size(C))
    const anchors = new Set();
    const cSize = C.size();
    for (let i = 0; i < cSize; i++) {
      anchors.add(C.call(i));
    }

    // R's output VALUES (not B's flat positions). These are A's values at the
    // cells the original R tile selects — what we need to color-match against.
    const rOutputs = new Set();
    for (let m = 0; m < s.M_R; m++) {
      for (let n = 0; n < s.N_R; n++) {
        rOutputs.add(s.rGrid[m][n]);
      }
    }

    // Build shadowMap: for each NON-zero anchor c, for each r in R's outputs,
    // map (c + r) -> r. These cells are the shifted copies of the original R tile;
    // they get colored like their corresponding r, but faded.
    const shadowMap = new Map();
    for (const c of anchors) {
      if (c === 0) continue;
      for (const r of rOutputs) {
        shadowMap.set(c + r, r);
      }
    }

    // Normalize the complement Layout to a rank-2 shape/stride for rendering
    const cStr = formatLayoutStr(C.shape, C.stride);
    const cParsed = parseLayout(cStr);

    s.complementAnchors = anchors;
    s.shadowMap = shadowMap;
    s.complementShape = cParsed.shape;
    s.complementStride = cParsed.stride;

    // Append complement info to the result text box (remember prev to allow restore)
    const resultEl = document.getElementById(`${tabId}-comp-result`);
    s.preComplementResultText = resultEl.textContent;
    resultEl.textContent = `${resultEl.textContent}\ncomplement(R, A) = ${cStr}`;
    resultEl.style.whiteSpace = 'pre-wrap';

    // Show the complement viz section and render it
    document.getElementById(`${tabId}-comp-complement-section`).style.display = 'block';
    renderCompGrid(tabId, 'complement');
    // Re-render the highlighted A so the edge markers and shadow tiles appear
    renderCompGrid(tabId, 'highlight');
    // Update button state
    const btn = document.getElementById(`${tabId}-comp-complement-btn`);
    if (btn) { btn.textContent = 'Hide complement'; btn.classList.add('active'); }
  } catch (e) {
    showErr(`${tabId}-comp-error`, 'Could not compute complement: ' + e.message);
  }
}

function setComp(tabId, a, b) {
  document.getElementById(`${tabId}-comp-a-input`).value = a;
  document.getElementById(`${tabId}-comp-b-input`).value = b;
  renderComposition(tabId);
}

function exportComp(tabId) {
  const a = document.getElementById(`${tabId}-comp-a-input`).value;
  const b = document.getElementById(`${tabId}-comp-b-input`).value;
  exportURL(`${tabId}-comp-export`, 'composition', a, b);
}
