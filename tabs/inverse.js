// left_inverse / right_inverse tab: one layout f, both inverses, and the two
// identities they promise. Functions become globals on `window`.
//
//   left_inverse(f)  = L with L(f(j)) = I(j) = j for every j < size(f)
//   right_inverse(f) = R with f(R(i)) = I(i) = i for every i < size(R)
//
// Neither CuTe function reports a meaningless result: right_inverse silently
// shrinks to the contiguous prefix it can invert, and left_inverse of a
// non-injective f returns a layout that cannot satisfy L(f(j)) = j. This tab
// checks both identities pointwise and warns instead.

const invState = {};

// Rows = mode 0, columns = the remaining modes (colexicographic), so the cell
// index is the layout's own 1-D domain index and every mode is drawn.
function invDisplay(layout) {
  if (layout.rank() > 1) {
    return {shape: [layout.shape[0], layout.shape.slice(1)], stride: [layout.stride[0], layout.stride.slice(1)]};
  }
  return {shape: [layout.shape, 1], stride: [layout.stride, 0]};
}

/** DOM-free: both inverses of `text` and the pointwise checks of their identities. */
function invCompute(text) {
  const f = parseExactLayoutInput(text);
  const fText = formatLayoutStr(f.shape, f.stride);
  const sizeF = f.size();
  if (sizeF > MAX_CELLS) throw new Error(`size(f) = ${sizeF} exceeds ${MAX_CELLS.toLocaleString()} cells. Choose a smaller layout.`);
  const fVals = [];
  for (let j = 0; j < sizeF; j++) fVals.push(f.call(j));

  // First collision, if any: f(j1) = f(j2) with j1 < j2.
  const firstAt = new Map();
  let collision = null;
  fVals.forEach((k, j) => {
    if (collision) return;
    if (firstAt.has(k)) collision = {j1: firstAt.get(k), j2: j, k};
    else firstAt.set(k, j);
  });
  const injective = !collision;
  const image = new Set(fVals);

  const R = right_inverse(f);
  const sizeR = R.size();
  const rightVals = [];
  for (let i = 0; i < sizeR; i++) {
    const j = R.call(i);
    rightVals.push(j >= 0 && j < sizeF ? fVals[j] : null);
  }
  const rightBad = rightVals.map((v, i) => v === i ? null : i).filter(i => i !== null);

  let L = null, lError = '';
  try { L = left_inverse(f); } catch (e) { lError = e.message; }
  let leftVals = null, leftBad = [];
  if (L) {
    const sizeL = L.size();
    leftVals = fVals.map(k => (k >= 0 && k < sizeL ? L.call(k) : null));
    leftBad = leftVals.map((v, j) => v === j ? null : j).filter(j => j !== null);
  }

  const warnings = [];
  const fmtCol = c => `f(${c.j1}) = f(${c.j2}) = ${c.k}`;
  if (lError) {
    warnings.push(`CuTe cannot compute left_inverse(f) as a layout. ${lError} CuTeDSL raises "unable to compute the left inverse" here, so L and L(f(j)) are not drawn.`);
  } else if (!injective) {
    warnings.push(`left_inverse(f) returned ${formatLayoutStr(L.shape, L.stride)}, but f is not injective (${fmtCol(collision)}), ` +
      `so no layout can satisfy L(f(j)) = I(j). It fails at ${leftBad.length} of ${sizeF} points (outlined in red).`);
  } else if (leftBad.length) {
    warnings.push(`left_inverse(f) returned ${formatLayoutStr(L.shape, L.stride)}, but L(f(j)) ≠ I(j) at ${leftBad.length} of ${sizeF} points (outlined in red).`);
  }
  if (sizeR < sizeF) {
    let reason;
    if (!injective) reason = `it is not injective: ${fmtCol(collision)}`;
    else {
      let gap = 0;
      while (image.has(gap)) gap++;
      reason = `offset ${gap} is never produced`;
    }
    warnings.push(`right_inverse(f) is only a partial inverse: f is not a bijection onto [0, ${sizeF}) because ${reason}. ` +
      `R = ${formatLayoutStr(R.shape, R.stride)} covers only i < ${sizeR}, so f(R(i)) = I(i) holds on ${sizeR} of size(f) = ${sizeF} offsets.`);
  }
  if (rightBad.length) {
    warnings.push(`f(R(i)) ≠ I(i) at ${rightBad.length} of ${sizeR} points (outlined in red).`);
  }

  return {f, fText, sizeF, fVals, injective, collision, image,
    R, RText: formatLayoutStr(R.shape, R.stride), sizeR, rightVals, rightBad,
    L, LText: L ? formatLayoutStr(L.shape, L.stride) : null, lError, leftVals, leftBad,
    bijective: injective && sizeR === sizeF, warnings};
}

function invSummary(r) {
  const check = (ok, good, bad) => ok ? `✓ ${good}` : `✗ ${bad}`;
  const left = r.L
    ? `L = left_inverse(f)  = ${r.LText}\n    ${check(!r.leftBad.length,
        `L(f(j)) = I(j) for every j < ${r.sizeF}`,
        `L(f(j)) ≠ I(j) at ${r.leftBad.length} of ${r.sizeF} points`)}`
    : `L = left_inverse(f)  — undefined (CuTeDSL raises)`;
  const right = `R = right_inverse(f) = ${r.RText}\n    ${check(!r.rightBad.length && r.sizeR === r.sizeF,
      `f(R(i)) = I(i) for every i < ${r.sizeR}`,
      r.rightBad.length ? `f(R(i)) ≠ I(i) at ${r.rightBad.length} of ${r.sizeR} points`
        : `f(R(i)) = I(i) only for i < ${r.sizeR}, not all ${r.sizeF}`)}`;
  const kind = r.bijective ? 'bijection onto [0, size(f)), so L = R = f⁻¹'
    : r.injective ? 'injective, not onto [0, size(f))' : 'not injective';
  return `f = ${r.fText}    size ${r.sizeF} · ${kind}\n${left}\n${right}`;
}

function generateInverseTabContent(id) {
  const pane = (key, title) => layoutVisualization({hostId: `${id}-inv-${key}-svg`,
    modeGroupId: `${id}-inv-${key}-mode-btns`, title, titleId: `${id}-inv-${key}-title`,
    onMode: mode => `toggleInvMode('${id}','${key}','${mode}')`});
  return `
    <div id="${id}-tab-inverse" class="panel">
      <div class="controls">
        <h2>left_inverse / right_inverse</h2>
        <div class="hint" style="margin:0 0 12px">
          <b>left_inverse</b>: returns a layout <code>L</code> such that <code>L(f(j)) = I(j)</code>
          for every <code>j &lt; size(f)</code>, i.e. <code>L ∘ f = I</code>.<br>
          <b>right_inverse</b>: returns a layout <code>R</code> such that <code>f(R(i)) = I(i)</code>
          for every <code>i &lt; size(R)</code>, i.e. <code>f ∘ R = I</code>.<br>
          where <code>I</code> is a layout with identity mapping: <code>I(i) = i</code>.
        </div>
        ${layoutInputField({id: `${id}-inv-layout-input`, label: 'Layout f', value: '(4,2):(2,1)'})}
        ${statusDivs(`${id}-inv`)}
        <div id="${id}-inv-result" class="comp-result-box" style="white-space:pre-wrap"></div>
        <button class="btn btn-render" onclick="renderInverse('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-inv-export" onclick="exportInverse('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setInv('${id}','(4,2):(2,1)')">Bijection (4,2):(2,1) — L = R</button>
            <button class="preset-btn" onclick="setInv('${id}','((2,2),2):((4,1),2)')">Bijection with nested modes</button>
            <button class="preset-btn" onclick="setInv('${id}','4:2')">Gaps 4:2 — L exists, R is partial</button>
            <button class="preset-btn" onclick="setInv('${id}','(2,2):(3,1)')">Non-dividing strides (2,2):(3,1)</button>
            <button class="preset-btn" onclick="setInv('${id}','(2,4):(0,1)')">Broadcast (2,4):(0,1) — not injective</button>
            <button class="preset-btn" onclick="setInv('${id}','(4,2):(1,2)')">Overlap (4,2):(1,2) — not injective</button>
            <button class="preset-btn" onclick="setInv('${id}','(2,2):(2,3)')">No left inverse (2,2):(2,3)</button>
          </div>
        </div>

        <div class="hint">
          Colors follow f's domain index <code>j</code>: cell <code>j</code> of f, cell
          <code>f(j)</code> of L and the cell of R holding <code>j</code> share a color.<br><br>
          Gray cells in L are offsets f never produces. L's values there are
          unconstrained, and CuTe fills them without meaning.<br><br>
          The two bottom grids evaluate the identities: each cell should show its own
          index. A red outline marks a point where the identity fails.<br><br>
          When f is a bijection onto <code>[0, size(f))</code>, L = R = f⁻¹. Otherwise
          right_inverse inverts only the contiguous run of offsets starting at 0, and
          left_inverse needs f to be injective.
        </div>
      </div>

      <div class="comp-results">
        <div class="comp-viz-span">${pane('f', 'f')}</div>
        ${pane('left', 'L = left_inverse(f)')}
        ${pane('right', 'R = right_inverse(f)')}
        ${pane('leftchk', 'L(f(j)) — should equal I(j) = j')}
        ${pane('rightchk', 'f(R(i)) — should equal I(i) = i')}
      </div>
    </div>`;
}

const INV_PANES = ['f', 'left', 'right', 'leftchk', 'rightchk'];
const INV_BAD = {bg: '#fee2e2', edgeColor: '#dc2626'};

function invModes(id) {
  return invState[id] ||= {modes: Object.fromEntries(INV_PANES.map(k => [k, new Set(['value'])]))};
}

function renderInverse(id) {
  const p = `${id}-inv`, state = invModes(id);
  const result = document.getElementById(`${p}-result`);
  showErr(`${p}-error`, '');
  try {
    const raw = document.getElementById(`${p}-layout-input`).value;
    const r = invCompute(raw);
    state.r = r;
    updateRankWarning(`${p}-warning`, [['Layout f', r.fText]]);
    const rank = document.getElementById(`${p}-warning`).textContent;
    showWarn(`${p}-warning`, [...r.warnings, rank].filter(Boolean).join(' '));
    result.textContent = invSummary(r);
    result.classList.add('visible');
    document.getElementById(`${p}-f-title`).textContent = `f = ${r.fText}`;
    document.getElementById(`${p}-left-title`).textContent = r.L ? `L = left_inverse(f) = ${r.LText}` : 'L = left_inverse(f) — undefined';
    document.getElementById(`${p}-right-title`).textContent = `R = right_inverse(f) = ${r.RText}`;
    INV_PANES.forEach(k => invRenderPane(id, k));
    updateOuterTabLabel(id, `inverse:${r.fText}`);
  } catch (e) {
    state.r = null;
    showErr(`${p}-error`, e.message);
    result.classList.remove('visible');
    INV_PANES.forEach(k => { document.getElementById(`${p}-${k}-svg`).innerHTML = ''; });
  }
}

function invRenderPane(id, key) {
  const state = invModes(id), r = state.r;
  if (!r) return;
  const modes = state.modes[key], hostId = `${id}-inv-${key}-svg`;
  const coordOf = (m, n) => `(${m},${n})`;
  let svg;
  if (key === 'f') {
    const d = invDisplay(r.f);
    svg = buildColoredLayoutSVG(d.shape, d.stride, modes, (m, n, value, index) => ({bg: colorBW(index)}));
  } else if (key === 'right') {
    const d = invDisplay(r.R);
    svg = buildColoredLayoutSVG(d.shape, d.stride, modes, (m, n, value) => ({bg: colorBW(value)}));
  } else if (key === 'left') {
    if (!r.L) svg = errSVG('left_inverse(f) is undefined — see the warning');
    else {
      const d = invDisplay(r.L);
      svg = buildColoredLayoutSVG(d.shape, d.stride, modes, (m, n, value, index) =>
        r.image.has(index) ? {bg: colorBW(value)} : {bg: '#f3f4f6', fg: '#9ca3af'});
    }
  } else {
    // An identity check is drawn over the domain of its inner layout: f's for
    // L(f(j)), R's for f(R(i)). Each cell shows the composed value.
    const left = key === 'leftchk';
    if (left && !r.L) svg = errSVG('left_inverse(f) is undefined — nothing to check');
    else {
      const d = invDisplay(left ? r.f : r.R);
      const vals = left ? r.leftVals : r.rightVals;
      const bad = new Set(left ? r.leftBad : r.rightBad);
      svg = buildColoredLayoutSVG(d.shape, d.stride, modes, (m, n, value, index) => {
        const v = vals[index];
        return {bg: bad.has(index) ? INV_BAD.bg : colorBW(v),
          text: buildCellLines(modes, v === null ? '—' : v, index, coordOf(m, n))};
      }, {selection: {edgeCells: bad, edgeColor: INV_BAD.edgeColor}});
    }
  }
  renderLayoutVisualization({hostId, modeGroupId: `${id}-inv-${key}-mode-btns`, modes, svg});
}

function toggleInvMode(id, key, mode) {
  toggleLayoutVisualizationMode(invModes(id).modes[key], mode);
  invRenderPane(id, key);
}

function setInv(id, layout) {
  invCompute(layout);  // a failed import (bad link) must leave the controls alone
  document.getElementById(`${id}-inv-layout-input`).value = layout;
  renderInverse(id);
}

function exportInverse(id) {
  exportURL(`${id}-inv-export`, 'inverse', document.getElementById(`${id}-inv-layout-input`).value);
}
