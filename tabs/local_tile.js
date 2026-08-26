// local_tile tab (OPERATIONS scope): `local_tile(A, tiler, coord)` — cut A into
// tiles and slice one (or a row, or all) out of the "rest" mode.
// Functions become globals on `window` (no module system).
//
// The whole function is two lines (include/cute/tensor_impl.hpp:1037 ->
// inner_partition:984):
//
//   auto tensor_tiled = zipped_divide(tensor, tiler);   // ((BLK...), (rest...))
//   return tensor_tiled(repeat<R0>(_), append<R1>(coord, _));
//
// So it is `zipped_divide` plus a slice, and everything interesting is in the
// slice — which is why this tab greys out what the coordinate did NOT pick
// rather than colouring the whole tensor. The Zipped / Tiled / Flat Divide tab
// already draws the un-sliced picture.
//
// Two details of that second line are worth seeing, because both bite people:
//
//   * `repeat<R0>(_)` UNPACKS mode 0 by ONE LEVEL — it does not flatten. R0 is
//     mode 0's top-level rank, so ((2,2),(4,2)) comes back as two modes that
//     are each still tuples, while a plain (128,64) tile comes back as two
//     scalars and merely looks flattened. Either way mode 0 stops being a
//     single mode, which is why TMA code calls `group_modes(x, 0, 2)` before
//     `tma_partition` — it folds back what local_tile just unfolded.
//   * `append<R1>(coord, _)` pads a short coord with underscores, and every
//     underscore KEEPS that rest mode instead of selecting from it. The DSL's
//     `local_tile(t, tile, (None, None))` picks no tile at all: it exposes the
//     tile indices as modes, to be sliced later at the copy.
//
// `slice_and_offset(crd, layout)` in layout.js is the same operation CuTe uses,
// so the result layout and its base offset come straight from the port.

function generateLocalTileTabContent(id) {
  return `
    <!-- local_tile panel -->
    <div id="${id}-tab-local_tile" class="panel">
      <div class="controls">
        <h2>Local Tile</h2>
        ${layoutInputField({ id: `${id}-lt-a-input`, label: 'Layout A &mdash; the tensor being tiled', value: '(16, 16):(16, 1)' })}
        ${layoutInputField({ id: `${id}-lt-tiler-input`, label: 'Tiler', value: '(4, 4)', hint: '<code>(4, 4)</code> = one tiler per mode; <code>(2,2):(1,4)</code> = one layout tiler; or one per line', textarea: true, rows: 2 })}
        ${layoutInputField({ id: `${id}-lt-coord-input`, label: 'Coord &mdash; index into the "rest" mode', value: '(1, 2)', hint: '<code>_</code> keeps that mode; blank keeps all' })}
        ${statusDivs(`${id}-lt`)}
        <div id="${id}-lt-result" class="comp-result-box"></div>
        <button class="btn btn-render" onclick="renderLocalTile('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-lt-export" onclick="exportLT('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setLT('${id}','(32, 64):(64, 1)','(8, 32)','(1, 0)')">Pick one tile &mdash; (8,32) tiles, coord (1,0)</button>
            <button class="preset-btn" onclick="setLT('${id}','(32, 64):(64, 1)','(8, 32)','(_, 0)')">Keep a mode &mdash; coord (_, 0) selects a whole column of tiles</button>
            <button class="preset-btn" onclick="setLT('${id}','(8, 8):(1, 8)','(4, 4)','(0, 1)')">Small: 8x8 col-major, (4,4) tiles, coord (0,1)</button>
            <button class="preset-btn" onclick="setLT('${id}','(12, 32):(32, 1)','3:4\n8:4','(1, 2)')">Strided tiler &lt;3:4, 8:4&gt; &mdash; the tile is scattered</button>
            <button class="preset-btn" onclick="setLT('${id}','(8, 16):(1, 8)','(2,2):(1,4)\n(4,2):(1,8)','(1, 1)')">Nested tile modes &mdash; result is ((2,2),(4,2)), NOT flattened</button>
            <button class="preset-btn" onclick="setLT('${id}','(16, 16):(1@0, 1@1)','(4, 4)','(1, 2)')">Coordinate tensor &mdash; <code>(1@0, 1@1)</code>, tile lands at origin (4,8)</button>
            <button class="preset-btn" onclick="setLT('${id}','(16, 16):(1@0, 1@1)','(4, 4)','(_, _)')">Coordinate tensor, coord (_, _) &mdash; what the TMA flow feeds tma_partition</button>
            <button class="preset-btn" onclick="setLT('${id}','(8, 16):(1@1, 1@0)','(4, 4)','(1, 1)')">Transposed coordinate tensor &mdash; <code>(1@1, 1@0)</code> swaps the axes</button>
          </div>
        </div>

        <div class="hint">
          <b>It is <code>zipped_divide</code> plus a slice.</b>
          <code>zipped_divide(A, tiler)</code> gives
          <code>((BLK...), (rest...))</code>; <code>local_tile</code> then slices
          the second mode with your coord and keeps the first whole. Nothing else
          happens &mdash; see the <b>Zipped / Tiled / Flat Divide</b> tab for the
          un-sliced picture.<br><br>
          <b><code>_</code> keeps a mode, it does not pick.</b> A coord shorter
          than the rest rank is padded with underscores
          (<code>append&lt;R1&gt;(coord, _)</code>), and each underscore leaves
          that tile index as a <em>mode of the result</em> to be sliced later.
          <code>local_tile(t, tile, (None, None))</code> selects no tile at all
          &mdash; it just exposes the indices. That is what the TMA flow does,
          then picks the tile at <code>copy</code> time with
          <code>tAgA[(None, bidx, bidy)]</code>.<br><br>
          <b>Mode 0 is unpacked one level &mdash; not flattened.</b> The slice
          is <code>repeat&lt;R0&gt;(_)</code>, R0 <em>separate</em> underscores,
          where R0 is mode 0's <em>top-level</em> rank. So
          <code>((2,2),(4,2))</code> comes back as two modes that are each still
          tuples; only a tile whose sub-modes are already scalars looks
          flattened. CuTe does not flatten here on purpose: the sub-structure
          records which tiler mode produced which part, and
          <code>composition</code> / <code>partition</code> downstream match on
          that profile. If you want it genuinely flat, that is
          <code>flatten()</code>, or <code>flat_divide</code> from the start
          &mdash; the third variant in the <b>Zipped / Tiled / Flat Divide</b>
          tab. Either way mode 0 stops being one mode, which is why TMA code
          calls <code>group_modes(x, 0, 2)</code> before
          <code>tma_partition</code>: it folds back what
          <code>local_tile</code> just unfolded.<br><br>
          <b>Coordinate (TMA) tensors work here.</b> A stride like
          <code>1@0</code> is a scaled basis element, so the layout maps a
          coordinate to a <em>coordinate</em> rather than to an offset &mdash;
          that is what <code>make_tiled_tma_atom</code> hands back.
          <code>local_tile</code> is one of the operations CuTe defines for them,
          because it is <code>composition</code> plus
          <code>complement(<em>tiler</em>)</code> and never touches A's strides
          except to scale them. The tile's constant then comes back as an
          <em>origin coordinate</em> rather than an offset. Complement, the
          inverses and every product are <em>not</em> defined for these &mdash;
          they need strides that can be ordered and divided.<br><br>
          <b>The offset is not in the layout.</b> Slicing produces a layout
          <em>and</em> a base offset (<code>slice_and_offset</code>); a layout
          structurally cannot carry a constant, so the tile's position in A lives
          in the tensor's iterator, not its layout. The result box prints both.
        </div>
      </div>

      <div class="comp-results" style="grid-template-columns:1fr">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-lt-a-title">A &mdash; selected tile(s) highlighted</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-lt-a-mode-btns">
                <button class="mode-btn active" onclick="setLtMode('${id}','a','value')">value</button>
                <button class="mode-btn" onclick="setLtMode('${id}','a','index')">index</button>
                <button class="mode-btn" onclick="setLtMode('${id}','a','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-lt-a-svg-zoom" onclick="toggleZoom('${id}-lt-a-svg')">Zoom in</button>
              <button class="mode-btn" onclick="downloadSVG('${id}-lt-a-svg', 'local_tile.svg')">Download SVG</button>
            </span>
          </div>
          <div class="cuo-viz-desc">
            All of A. Only what the coord selected is coloured &mdash; everything
            else is greyed, because that is what <code>local_tile</code> throws
            away. With an <code>_</code> in the coord several tiles survive, and
            each keeps its own colour so you can see that the tile index became a
            <em>mode</em> rather than a choice.
          </div>
          <div class="viz-box"><div id="${id}-lt-a-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-lt-tile-title">The tile</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-lt-tile-mode-btns">
                <button class="mode-btn active" onclick="setLtMode('${id}','tile','value')">value</button>
                <button class="mode-btn" onclick="setLtMode('${id}','tile','index')">index</button>
                <button class="mode-btn" onclick="setLtMode('${id}','tile','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-lt-tile-svg-zoom" onclick="toggleZoom('${id}-lt-tile-svg')">Zoom in</button>
            </span>
          </div>
          <div class="cuo-viz-desc">
            One tile on its own, in tile-local coordinates. <code>value</code> is
            its offset <em>into A</em>, base offset included &mdash; so these
            numbers match the highlighted cells above.
          </div>
          <div class="viz-box"><div id="${id}-lt-tile-svg"></div></div>
        </div>
      </div>
    </div>`;
}

const ltState = {};

/** Parse the coord field: `(1, 0)`, `(_, 0)`, `_`, `1`, or blank.
 *  `_` (and a blank field) become null, which is this port's underscore. */
function ltParseCoord(raw) {
  const t = (raw || '').trim();
  if (t === '') return [];                    // blank -> all underscores
  let body = t;
  if (body[0] === '(' && body[body.length - 1] === ')') body = body.slice(1, -1);
  return body.split(',').map(piece => {
    const p = piece.trim();
    if (p === '' || p === '_' || p === 'None' || p === 'none') return null;
    const v = parseInt(p, 10);
    if (!Number.isFinite(v) || v < 0) {
      throw new Error(
        `Cannot read coord component "${p}" — use a non-negative integer, or ` +
        `"_" to keep that mode.`);
    }
    return v;
  });
}

/** Evaluate a basis-strided rank-2 layout at (tile-local t, tile-index r).
 *  `idx2crd` makes each flat index congruent with its mode's shape first, so
 *  nested tile modes recurse correctly. */
function ltBasisAt(L, t, r, ndim) {
  const out = new Array(ndim).fill(0);
  crd2basis([idx2crd(t, L.shape[0]), idx2crd(r, L.shape[1])], L.shape, L.stride, out);
  return out;
}

function renderLocalTile(tabId) {
  showErr(`${tabId}-lt-error`, '');
  try {
    const aStr     = document.getElementById(`${tabId}-lt-a-input`).value;
    const tilerRaw = document.getElementById(`${tabId}-lt-tiler-input`).value;
    const coordStr = document.getElementById(`${tabId}-lt-coord-input`).value;

    const tilerLines = tilerRaw.split('\n').map(s => s.trim()).filter(s => s.length > 0);
    if (tilerLines.length === 0) throw new Error('Tiler input is empty');

    const warnInputs = [['A', aStr]];
    tilerLines.forEach((line, i) =>
      warnInputs.push([tilerLines.length > 1 ? `Tiler[${i}]` : 'Tiler', line]));
    updateRankWarning(`${tabId}-lt-warning`, warnInputs);

    // `{ basis: true }`: local_tile is one of the ops CuTe defines for coordinate
    // (TMA) tensors, because it is composition + complement of the TILER — never
    // of A. See the stride arithmetic note in layout.js.
    const aParsed = parseLayout(aStr, { basis: true });
    const aStripped = stripTrivialTrailing(aParsed.shape, aParsed.stride);
    const aLayout = new Layout(aStripped.shape, aStripped.stride);

    // Tiler conventions, in CuTe's terms rather than the Divide tabs':
    //   several lines        -> one tiler per mode (a Tiler tuple)
    //   one line WITH  ":"   -> a single layout tiler, e.g. (2,2):(1,4)
    //   one line WITHOUT ":" -> a Shape tiler, e.g. Shape<_8,_32>, whose modes
    //                           each apply to the matching mode of A
    // The last is the case `local_tile(data, Shape<_32,_64>{}, coord)` uses and
    // the one people type. Reading `(8, 32)` as a single layout instead (which
    // is what the Divide tabs do) collapses the rest mode to rank 1 and makes a
    // 2-D coord impossible — worth keeping straight.
    let tilerArg;
    if (tilerLines.length > 1) {
      tilerArg = tilerLines.map(line => {
        const p = parseLayout(line);
        const st = stripTrivialTrailing(p.shape, p.stride);
        return new Layout(st.shape, st.stride);
      });
    } else if (topLevelColon(tilerLines[0]) !== -1) {
      const p = parseLayout(tilerLines[0]);
      const st = stripTrivialTrailing(p.shape, p.stride);
      tilerArg = new Layout(st.shape, st.stride);
    } else {
      // parseValue, not parseLayout: the latter pads a bare `8` out to (8,1),
      // which would silently add a second tiler mode.
      const shape = parseValue(tilerLines[0]);
      tilerArg = Array.isArray(shape) ? shape.map(x => new Layout(x)) : new Layout(shape);
    }

    // zipped_divide(A, tiler) = ((BLK...), (rest...)) — everything local_tile
    // does after this is one slice.
    const Z = zipped_divide(aLayout, tilerArg);
    const tileShape = Z.shape[0], restShape = Z.shape[1];
    const R0 = is_tuple(tileShape) ? tileShape.length : 1;
    const R1 = is_tuple(restShape) ? restShape.length : 1;
    const restArr = is_tuple(restShape) ? restShape : [restShape];

    // append<R1>(coord, _): pad short coords with underscores.
    const coord = ltParseCoord(coordStr);
    if (coord.length > R1) {
      throw new Error(
        `Coord has ${coord.length} components but the "rest" mode is rank ${R1} ` +
        `(${restArr.join(', ')} tiles). Give at most ${R1}, or use "_" to keep a mode.`);
    }
    const restCrd = coord.slice();
    while (restCrd.length < R1) restCrd.push(null);
    restCrd.forEach((c, i) => {
      if (c !== null && c >= restArr[i]) {
        throw new Error(
          `Coord component ${i} is ${c}, but A only holds ${restArr[i]} tile` +
          `${restArr[i] === 1 ? '' : 's'} along that axis (0..${restArr[i] - 1}).`);
      }
    });

    // repeat<R0>(_) on the tile mode, the coord on the rest mode. This is
    // literally inner_partition's one line.
    const tileCrd = R0 === 1 ? null : new Array(R0).fill(null);
    const restCrdArg = R1 === 1 ? restCrd[0] : restCrd;
    // `slice_and_offset` in one call would compute the offset with crd2idx, which
    // is meaningless for basis strides (it would multiply a stride OBJECT). Take
    // the structural half from slice_ either way, and get the constant the way
    // each kind of layout can express it: an integer offset, or — for a
    // coordinate tensor — the coordinate of the tile's first element.
    const sliceCrd = [tileCrd, restCrdArg];
    const resultLayout = new Layout(slice_(sliceCrd, Z.shape), slice_(sliceCrd, Z.stride));
    const baseOffset = aParsed.basis ? null : crd2idx(sliceCrd, Z.shape, Z.stride);

    // Which tiles survived the slice?
    const restSize = product(restShape);
    const kept = [];
    for (let r = 0; r < restSize; r++) {
      const rc = idx2crd(r, restShape);
      const rcArr = is_tuple(rc) ? rc : [rc];
      if (rcArr.every((c, i) => restCrd[i] === null || restCrd[i] === c)) kept.push({ r, rcArr });
    }

    // Which GRID POSITION does each (tile-local, tile-index) pair name? Dividing
    // the identity layout by the same tiler answers that directly, so the
    // highlight never depends on A's strides — which is what lets a coordinate
    // (TMA) tensor work here at all, and also removes the collision risk the old
    // offset round-trip had on non-injective layouts.
    const [M_A, N_A] = productEach(aParsed.shape);
    const Zpos = zipped_divide(new Layout(aStripped.shape), tilerArg);

    const tileSize = product(tileShape);
    const posTile = new Map();       // grid position -> which kept tile owns it
    kept.forEach((k, ki) => {
      for (let t = 0; t < tileSize; t++) posTile.set(Zpos.call(t, k.r), ki);
    });

    // Origin of the first selected tile. Evaluated at a real point rather than
    // through the sliced coord, so nested tile modes can't trip the recursion.
    const ndim = aParsed.ndim || 2;
    const originCrd = aParsed.basis ? ltBasisAt(Z, 0, kept[0].r, ndim) : null;

    const keptStr = kept.map(k => `(${k.rcArr.join(',')})`).join(' ');
    const resultStr = formatLayoutStr(resultLayout.shape, resultLayout.stride);
    document.getElementById(`${tabId}-lt-result`).innerHTML =
      `<div class="cuo-result-line">zipped_divide(A, tiler) = <b>${formatLayoutStr(Z.shape, Z.stride)}</b> ` +
      `&mdash; ${product(tileShape)}-element tiles, ${restSize} of them</div>` +
      `<div class="cuo-result-line">slice = (repeat&lt;${R0}&gt;(_), ` +
      `(${restCrd.map(c => c === null ? '_' : c).join(',')}))</div>` +
      `<div class="cuo-result-line">local_tile(...) = <b>${resultStr}</b>` +
      (aParsed.basis
        ? ` &nbsp;at origin <b>(${originCrd.join(',')})</b>`
        : ` &nbsp;+ offset <b>${baseOffset || 0}</b>`) + `</div>` +
      `<div class="cuo-result-line" style="color:#9ca3af">Result rank ${resultLayout.rank()} = ` +
      `${R0} tile mode${R0 === 1 ? '' : 's'} (mode 0 unpacked one level, sub-modes keep their nesting)` +
      (restCrd.filter(c => c === null).length
        ? ` + ${restCrd.filter(c => c === null).length} kept tile-index mode` +
          `${restCrd.filter(c => c === null).length === 1 ? '' : 's'}`
        : ` (coord was fully concrete, so no tile-index modes survive)`) +
      `. Selected ${kept.length} of ${restSize} tiles: ${keptStr}</div>`;

    ltState[tabId] = {
      aParsed, Z, resultLayout, baseOffset, originCrd, kept, posTile, M_A, N_A,
      tileShape, R0, restCrd, tileSize,
      aMode: (ltState[tabId] && ltState[tabId].aMode) || 'value',
      tileMode: (ltState[tabId] && ltState[tabId].tileMode) || 'value',
    };
    ltRenderA(tabId);
    ltRenderTile(tabId);
    updateOuterTabLabel(tabId, `LocalTile:${resultStr}`);
  } catch (e) {
    showErr(`${tabId}-lt-error`, e.message);
    for (const k of ['a', 'tile']) {
      const el = document.getElementById(`${tabId}-lt-${k}-svg`);
      if (el) el.innerHTML = '';
    }
  }
}

/** A, with everything the coord did NOT select greyed out. Each surviving tile
 *  gets its own colour, so a kept (`_`) mode reads as several tiles rather than
 *  one big selection. */
function ltRenderA(tabId) {
  const s = ltState[tabId];
  if (!s) return;
  const modes = new Set([s.aMode]);
  const label = (m, n) =>
      s.aMode === 'value' ? [layoutValueLabel(s.aParsed, m, n)]
    : s.aMode === 'index' ? [String(m + n * s.M_A)]
    :                       [`(${m},${n})`];
  const svg = buildColoredLayoutSVG(s.aParsed.shape, s.aParsed.stride, modes, (m, n) => {
    const ki = s.posTile.get(m + n * s.M_A);
    if (ki === undefined) return { bg: '#e8e8e8', fg: '#bbb', stroke: '#ddd', text: label(m, n) };
    return { bg: colorHighlight(ki), stroke: '#1e3a5f', sw: 2, text: label(m, n) };
  });
  document.getElementById(`${tabId}-lt-a-svg`).innerHTML =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `A ${s.M_A}&times;${s.N_A} &mdash; ${s.kept.length} of ${product(s.Z.shape[1])} tiles selected` +
    (s.kept.length > 1 ? `, one colour each` : '') +
    `</div>` + svg;
  applyZoomState(`${tabId}-lt-a-svg`);
  updateModeBtns(`${tabId}-lt-a-mode-btns`, modes);
  document.getElementById(`${tabId}-lt-a-title`).textContent =
    `A — ${s.kept.length} of ${product(s.Z.shape[1])} tiles selected`;
}

/** The first selected tile on its own, in tile-local coordinates, with values
 *  offset into A so they match the highlighted cells above. */
function ltRenderTile(tabId) {
  const s = ltState[tabId];
  if (!s) return;
  const modes = new Set([s.tileMode]);
  const k = s.kept[0];
  const tileL = new Layout(s.tileShape, s.Z.stride[0]);
  const [Mt, Nt] = productEach(s.tileShape);
  // A coordinate tensor has no 1-D offset — its "offset" is a coordinate, which
  // slice_and_offset returns as the origin. Show that instead of adding it.
  const tileParsed = { shape: s.tileShape, stride: s.Z.stride[0], basis: s.aParsed.basis,
                       ndim: s.aParsed.ndim || 2, origin: s.originCrd };
  const off = s.aParsed.basis ? null : s.Z.call(0, k.r);
  const svg = buildColoredLayoutSVG(s.tileShape, s.Z.stride[0], modes, (m, n) => ({
    bg: colorHighlight(0), stroke: '#1e3a5f', sw: 2,
    text: s.tileMode === 'value'
            ? [s.aParsed.basis ? layoutValueLabel(tileParsed, m, n)
                               : String(off + layoutAt(s.tileShape, s.Z.stride[0], m, n))]
        : s.tileMode === 'index' ? [String(m + n * Mt)]
        : [`(${m},${n})`],
  }));
  document.getElementById(`${tabId}-lt-tile-svg`).innerHTML =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `tile (${k.rcArr.join(',')}) &mdash; ${formatLayoutStr(tileL.shape, tileL.stride)}` +
    (off === null ? ` at origin (${(s.originCrd || []).join(',')})` : ` at offset ${off}`) +
    (s.kept.length > 1 ? ` &mdash; first of ${s.kept.length} selected` : '') +
    `</div>` + svg;
  applyZoomState(`${tabId}-lt-tile-svg`);
  updateModeBtns(`${tabId}-lt-tile-mode-btns`, modes);
  document.getElementById(`${tabId}-lt-tile-title`).textContent =
    `The tile — ${Mt}×${Nt}` + (off === null ? '' : ` at offset ${off}`);
}

function setLtMode(tabId, which, mode) {
  const s = ltState[tabId];
  if (!s) return;
  if (which === 'a') { s.aMode = mode; ltRenderA(tabId); }
  else { s.tileMode = mode; ltRenderTile(tabId); }
}

function setLT(tabId, a, tiler, coord) {
  document.getElementById(`${tabId}-lt-a-input`).value     = a;
  document.getElementById(`${tabId}-lt-tiler-input`).value = tiler;
  document.getElementById(`${tabId}-lt-coord-input`).value = coord;
  renderLocalTile(tabId);
}

function exportLT(tabId) {
  exportURL(`${tabId}-lt-export`, 'local_tile',
    document.getElementById(`${tabId}-lt-a-input`).value,
    document.getElementById(`${tabId}-lt-tiler-input`).value,
    document.getElementById(`${tabId}-lt-coord-input`).value);
}
