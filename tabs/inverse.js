// left_inverse / right_inverse tab: one layout f and the inverse picked with
// the seg-control, drawn top (f) / bottom (the inverse). Functions become
// globals on `window`.
//
//   left_inverse(f)  = L with L(f(j)) = I(j) = j for every j < size(f)
//   right_inverse(f) = R with f(R(i)) = I(i) = i for every i < size(R)
//
// Neither CuTe function reports a meaningless result: right_inverse silently
// shrinks to the contiguous prefix it can invert, and left_inverse of a
// non-injective f returns a layout that cannot satisfy L(f(j)) = j. This tab
// checks both identities pointwise and warns instead.

const invState = {};
const INV_SIDES = ['left', 'right'];
const INV_PANES = ['f', 'out'];
const INV_BAD_EDGE = '#dc2626';
const INV_UNUSED = {bg: '#f3f4f6', fg: '#9ca3af'};

// Rows = mode 0, columns = the remaining modes (colexicographic), so the cell
// index is the layout's own 1-D domain index and every mode is drawn.
function invDisplay(layout) {
  if (layout.rank() > 1) {
    return {shape: [layout.shape[0], layout.shape.slice(1)], stride: [layout.stride[0], layout.stride.slice(1)]};
  }
  return {shape: [layout.shape, 1], stride: [layout.stride, 0]};
}

/** DOM-free: both inverses of `text`, the pointwise checks of their identities,
 *  and one warning list per inverse. */
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
  const rVals = [];
  for (let i = 0; i < sizeR; i++) rVals.push(R.call(i));
  const rightBad = rVals.map((j, i) => (j >= 0 && j < sizeF && fVals[j] === i) ? null : i).filter(i => i !== null);
  const rightReached = new Set(rVals);

  let L = null, lError = '';
  try { L = left_inverse(f); } catch (e) { lError = e.message; }
  let leftBad = [];
  if (L) {
    const sizeL = L.size();
    leftBad = fVals.map((k, j) => (k >= 0 && k < sizeL && L.call(k) === j) ? null : j).filter(j => j !== null);
  }

  const fmtCol = c => `f(${c.j1}) = f(${c.j2}) = ${c.k}`;
  const leftWarnings = [];
  if (lError) {
    leftWarnings.push(`CuTe cannot compute left_inverse(f) as a layout. ${lError} CuTeDSL raises "unable to compute the left inverse" here.`);
  } else if (!injective) {
    leftWarnings.push(`left_inverse(f) returned ${formatLayoutStr(L.shape, L.stride)}, but f is not injective (${fmtCol(collision)}), ` +
      `so no layout can satisfy L(f(j)) = I(j). It fails at ${leftBad.length} of ${sizeF} points of f (outlined in red).`);
  } else if (leftBad.length) {
    leftWarnings.push(`left_inverse(f) returned ${formatLayoutStr(L.shape, L.stride)}, but L(f(j)) ≠ I(j) at ${leftBad.length} of ${sizeF} points of f (outlined in red).`);
  }
  const rightWarnings = [];
  if (sizeR < sizeF) {
    let reason;
    if (!injective) reason = `it is not injective: ${fmtCol(collision)}`;
    else {
      let gap = 0;
      while (image.has(gap)) gap++;
      reason = `offset ${gap} is never produced`;
    }
    rightWarnings.push(`right_inverse(f) is only a partial inverse: f is not a bijection onto [0, ${sizeF}) because ${reason}. ` +
      `R = ${formatLayoutStr(R.shape, R.stride)} covers only i < ${sizeR}, so f(R(i)) = I(i) holds on ${sizeR} of size(f) = ${sizeF} offsets. ` +
      `Gray cells of f are never returned by R.`);
  }
  if (rightBad.length) rightWarnings.push(`f(R(i)) ≠ I(i) at ${rightBad.length} of ${sizeR} points.`);

  return {f, fText, sizeF, fVals, injective, collision, image,
    R, RText: formatLayoutStr(R.shape, R.stride), sizeR, rightBad, rightReached, rightWarnings,
    L, LText: L ? formatLayoutStr(L.shape, L.stride) : null, lError, leftBad, leftWarnings,
    bijective: injective && sizeR === sizeF};
}

function invSummary(r, side) {
  const kind = r.bijective ? 'bijection onto [0, size(f)), so L = R = f⁻¹'
    : r.injective ? 'injective, not onto [0, size(f))' : 'not injective';
  const head = `f = ${r.fText}    size ${r.sizeF} · ${kind}`;
  if (side === 'left') {
    if (!r.L) return `${head}\nL = left_inverse(f) — undefined (CuTeDSL raises)`;
    return `${head}\nL = left_inverse(f) = ${r.LText}\n    ` + (r.leftBad.length
      ? `✗ L(f(j)) ≠ I(j) at ${r.leftBad.length} of ${r.sizeF} points`
      : `✓ L(f(j)) = I(j) for every j < ${r.sizeF}`);
  }
  return `${head}\nR = right_inverse(f) = ${r.RText}\n    ` + (r.rightBad.length
    ? `✗ f(R(i)) ≠ I(i) at ${r.rightBad.length} of ${r.sizeR} points`
    : r.sizeR < r.sizeF ? `✗ f(R(i)) = I(i) only for i < ${r.sizeR}, not all ${r.sizeF}`
      : `✓ f(R(i)) = I(i) for every i < ${r.sizeR}`);
}

function generateInverseTabContent(id) {
  const pane = (key, title) => layoutVisualization({hostId: `${id}-inv-${key}-svg`,
    modeGroupId: `${id}-inv-${key}-mode-btns`, title, titleId: `${id}-inv-${key}-title`,
    onMode: mode => `toggleInvMode('${id}','${key}','${mode}')`});
  return `
    <div id="${id}-tab-inverse" class="panel">
      <div class="controls">
        <h2>left_inverse / right_inverse</h2>
        <div class="form-group">
          <label>Inverse</label>
          <div class="seg-control" id="${id}-inv-side-btns">
            <button class="mode-btn active" onclick="setInvSide('${id}', 'left')">left_inverse</button>
            <button class="mode-btn" onclick="setInvSide('${id}', 'right')">right_inverse</button>
          </div>
          <div class="hint" id="${id}-inv-side-hint" style="margin:8px 0 0"></div>
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
            <button class="preset-btn" onclick="setInv('${id}','4:2')">Gaps 4:2 — not onto</button>
            <button class="preset-btn" onclick="setInv('${id}','(2,2):(3,1)')">Non-dividing extents (2,2):(3,1)</button>
            <button class="preset-btn" onclick="setInv('${id}','(2,4):(0,1)')">Broadcast (2,4):(0,1) — not injective</button>
            <button class="preset-btn" onclick="setInv('${id}','(4,2):(1,2)')">Overlap (4,2):(1,2) — not injective</button>
            <button class="preset-btn" onclick="setInv('${id}','(2,2):(2,3)')">Non-dividing strides (2,2):(2,3) — no left inverse</button>
          </div>
        </div>

        <div class="hint">
          Colors follow f's domain index <code>j</code>: cell <code>j</code> of f and every
          cell of the inverse whose value is <code>j</code> share a color.<br><br>
          Gray cells are the part the identity does not constrain: offsets of L that f never
          produces, or cells of f that R never returns. A red outline on f marks a point
          where <code>L(f(j)) ≠ j</code>.<br><br>
          When f is a bijection onto <code>[0, size(f))</code>, L = R = f⁻¹. Otherwise
          right_inverse inverts only the contiguous run of offsets starting at 0, and
          left_inverse needs f to be injective.
        </div>
      </div>

      <div class="comp-results" style="grid-template-columns:1fr">
        ${pane('f', 'f')}
        ${pane('out', 'L = left_inverse(f)')}
      </div>
    </div>`;
}

function invModes(id) {
  const s = invState[id] ||= {};
  // Session restore may create the store with only `inverseSide` in it.
  s.modes ||= Object.fromEntries(INV_PANES.map(k => [k, new Set(['value'])]));
  if (!INV_SIDES.includes(s.inverseSide)) s.inverseSide = 'left';
  return s;
}

const INV_SIDE_HINT = {
  left: 'left_inverse: returns a layout L such that L(f(j)) = I(j) for every j < size(f), i.e. L ∘ f = I, where I is a layout with identity mapping: I(i) = i.',
  right: 'right_inverse: returns a layout R such that f(R(i)) = I(i) for every i < size(R), i.e. f ∘ R = I, where I is a layout with identity mapping: I(i) = i.',
};

function invSyncSide(id) {
  const side = invModes(id).inverseSide;
  const group = document.getElementById(`${id}-inv-side-btns`);
  if (group) group.querySelectorAll('.mode-btn').forEach(b =>
    b.classList.toggle('active', b.textContent.trim() === `${side}_inverse`));
  document.getElementById(`${id}-inv-side-hint`).textContent = INV_SIDE_HINT[side];
}

function renderInverse(id) {
  const p = `${id}-inv`, state = invModes(id), side = state.inverseSide;
  const result = document.getElementById(`${p}-result`);
  showErr(`${p}-error`, '');
  invSyncSide(id);
  try {
    const raw = document.getElementById(`${p}-layout-input`).value;
    const r = invCompute(raw);
    state.r = r;
    updateRankWarning(`${p}-warning`, [['Layout f', r.fText]]);
    const rank = document.getElementById(`${p}-warning`).textContent;
    const warnings = side === 'left' ? r.leftWarnings : r.rightWarnings;
    showWarn(`${p}-warning`, [...warnings, rank].filter(Boolean).join(' '));
    result.textContent = invSummary(r, side);
    result.classList.add('visible');
    document.getElementById(`${p}-f-title`).textContent = `f = ${r.fText}`;
    document.getElementById(`${p}-out-title`).textContent = side === 'left'
      ? (r.L ? `L = left_inverse(f) = ${r.LText}` : 'L = left_inverse(f) — undefined')
      : `R = right_inverse(f) = ${r.RText}`;
    INV_PANES.forEach(k => invRenderPane(id, k));
    updateOuterTabLabel(id, `${side}_inverse:${r.fText}`);
  } catch (e) {
    state.r = null;
    showErr(`${p}-error`, e.message);
    result.classList.remove('visible');
    INV_PANES.forEach(k => { document.getElementById(`${p}-${k}-svg`).innerHTML = ''; });
  }
}

function invRenderPane(id, key) {
  const state = invModes(id), r = state.r, side = state.inverseSide;
  if (!r) return;
  const modes = state.modes[key], hostId = `${id}-inv-${key}-svg`;
  let svg;
  if (key === 'f') {
    // f's cells are colored by their own domain index j. With R selected, the
    // cells R never returns are grayed; with L, a failing j is outlined.
    const d = invDisplay(r.f);
    const bad = new Set(side === 'left' ? r.leftBad : []);
    svg = buildColoredLayoutSVG(d.shape, d.stride, modes, (m, n, value, index) =>
      side === 'right' && !r.rightReached.has(index) ? INV_UNUSED : {bg: colorBW(index)},
      {selection: {edgeCells: bad, edgeColor: INV_BAD_EDGE}});
  } else if (side === 'right') {
    const d = invDisplay(r.R);
    svg = buildColoredLayoutSVG(d.shape, d.stride, modes, (m, n, value) => ({bg: colorBW(value)}));
  } else if (!r.L) {
    svg = errSVG('left_inverse(f) is undefined — see the warning');
  } else {
    // An offset f never produces is unconstrained by L(f(j)) = j.
    const d = invDisplay(r.L);
    svg = buildColoredLayoutSVG(d.shape, d.stride, modes, (m, n, value, index) =>
      r.image.has(index) ? {bg: colorBW(value)} : INV_UNUSED);
  }
  renderLayoutVisualization({hostId, modeGroupId: `${id}-inv-${key}-mode-btns`, modes, svg});
}

function toggleInvMode(id, key, mode) {
  toggleLayoutVisualizationMode(invModes(id).modes[key], mode);
  invRenderPane(id, key);
}

function setInvSide(id, side) {
  if (!INV_SIDES.includes(side)) throw new Error(`Unknown inverse "${side}"; expected left or right.`);
  invModes(id).inverseSide = side;
  renderInverse(id);
}

/** Preset / URL entry point. `side` is optional: presets keep the current pick. */
function setInv(id, layout, side) {
  invCompute(layout);  // a failed import (bad link) must leave the controls alone
  if (side !== undefined && !INV_SIDES.includes(side)) throw new Error(`Unknown inverse "${side}"; expected left or right.`);
  document.getElementById(`${id}-inv-layout-input`).value = layout;
  if (side !== undefined) invModes(id).inverseSide = side;
  renderInverse(id);
}

function exportInverse(id) {
  exportURL(`${id}-inv-export`, 'inverse', document.getElementById(`${id}-inv-layout-input`).value, invModes(id).inverseSide);
}
