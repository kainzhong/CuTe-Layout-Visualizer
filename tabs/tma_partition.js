// tma_partition tab (COPY scope): CuTe's "VectorCopy Partitioner" for TMA.
// Functions become globals on `window` (no module system).
//
// include/cute/atom/copy_traits_sm90_tma.hpp:1409 —
//
//   inv_smem_layout = right_inverse(get_nonswizzle_portion(layout<0>(stensor)))
//   layout_v        = tile_to_shape(make_layout(inv_smem_layout), size<0>(stensor))
//   tma_layout_v    = Layout<Copy_Atom::NumValSrc>            // one instruction
//   layout_V        = make_tile(logical_divide(layout_v, tma_layout_v))
//   for each tensor: coalesce(tensor.compose(append<R>(layout_V, _)))
//                      -> ((TMA, TMA_Iter), Rest...)
//   then domain_offset(multicast_coord, ...)
//
// Despite the name it partitions VECTORS, not threads — TMA has no threads to
// partition. What it splits is mode 0 of every tensor you hand it, into
// "one instruction's worth x how many instructions".
//
// Three things this tab exists to make concrete:
//
//   * The atom contributes exactly ONE NUMBER, `NumValSrc`. `tma_layout_v` is a
//     rank-1 layout of that size; the atom's value-layout structure is never
//     consulted. For TMA that layout is `Layout<Shape<_1, NumBitsPerTMA>>`, so
//     there is no structure to consult anyway.
//   * The ORDER comes from SMEM alone. `layout_V` is built from
//     `layout<0>(stensor)`; the GMEM tensor contributes nothing to it. The
//     operation says "GMEM, follow SMEM's order" — the same principle that
//     decides the box in make_tiled_tma_atom.
//   * The SAME partition is applied to every tensor passed
//     (`cute::transform(make_tuple(gtensors..., stensor), ...)`), which is why
//     it takes both and returns both. Its one static assert is
//     `size<0>(stensor) == size<0>(tensor)` — mode 0 must already BE the tile on
//     both sides, which is what the caller's `group_modes(x, 0, 2)` is for.
//
// Multicast is out of scope here (cta_layout is 1), so `domain_offset` is always
// zero. It lands with the multicast Op, alongside the same gap in the
// make_tiled_tma_atom tab.

function generateTmaPartitionTabContent(id) {
  return `
    <!-- tma_partition panel -->
    <div id="${id}-tab-tma_partition" class="panel">
      <div class="controls">
        <h2>tma_partition</h2>

        <details class="cuo-section" open>
          <summary>1. The atom &mdash; only its element count is used</summary>
          <div class="cuo-section-body">
            <div class="form-group">
              <label>TMA operation</label>
              <select id="${id}-tp-op-input" disabled>
                <option value="tma_g2s" selected>cpasync.CopyBulkTensorTileG2SOp</option>
              </select>
            </div>
            <div class="form-group">
              <label>ThrID<span style="color:#6b7280;font-weight:normal">&nbsp;&mdash; always one thread for TMA</span></label>
              <input type="text" id="${id}-tp-thrid-input" value="1:0" disabled>
            </div>
            <div class="form-group">
              <label>TV Layout Src = Dst &mdash; (1, N):(0,1)<span style="color:#6b7280;font-weight:normal">&nbsp;&mdash; N = values one instruction moves</span></label>
              <input type="number" id="${id}-tp-vals-input" value="16" min="1" step="1">
            </div>
            <div class="form-group">
              <label>Value type</label>
              <select id="${id}-tp-dtype-input">${dtypeOptions('float')}</select>
            </div>
            <div id="${id}-tp-atom-result" class="cuo-result"></div>
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>2. cta_coord, cta_layout</summary>
          <div class="cuo-section-body">
            <div class="form-group">
              <label>cta_coord / cta_layout<span style="color:#6b7280;font-weight:normal">&nbsp;&mdash; the CLUSTER coordinate, not the tile's. Fixed while multicast is out of scope</span></label>
              <input type="text" id="${id}-tp-cta-input" value="0 / (1)" disabled>
            </div>
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>3. smem_tensor</summary>
          <div class="cuo-section-body">
            ${layoutInputField({ id: `${id}-tp-smem-input`, label: 'smem_tensor &mdash; (tile, Rest...)', value: '((4, 16), 2):((16, 1), 64)', hint: 'mode 0 IS the tile; it alone decides the ORDER' })}
            <div class="form-group">
              <label>swizzle<span style="color:#6b7280;font-weight:normal">&nbsp;&mdash; stripped by get_nonswizzle_portion, so it changes nothing here</span></label>
              <select id="${id}-tp-swizzle-input">${TMA_SWIZZLE_OPTIONS}</select>
            </div>
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>4. gmem_tensor</summary>
          <div class="cuo-section-body">
            ${layoutInputField({ id: `${id}-tp-gmem-input`, label: 'gmem_tensor &mdash; (tile, Rest...)', value: '((4, 16), (2, 2)):((1@0, 1@1), (4@0, 16@1))', hint: 'mode 0 IS the tile; anything after is Rest' })}
            <div id="${id}-tp-result" class="cuo-result"></div>
          </div>
        </details>

        <div class="form-group">
          <label>Highlight instruction (empty = show all)</label>
          <input type="text" id="${id}-tp-highlight" value="" placeholder="e.g. 0" oninput="renderTmaPartition('${id}')">
        </div>

        ${statusDivs(`${id}-tp`)}
        <button class="btn btn-render" onclick="renderTmaPartition('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-tp-export" onclick="exportTP('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setTP('${id}',16,'float','3,4,3','((4, 16), 2):((16, 1), 64)','((4, 16), (2, 2)):((1@0, 1@1), (4@0, 16@1))')">The usual call &mdash; 4 passes of 16, 2 SMEM stages, 2x2 tiles</button>
            <button class="preset-btn" onclick="setTP('${id}',64,'float','3,4,3','(4, 16):(16, 1)','((4, 16), (2, 2)):((1@0, 1@1), (4@0, 16@1))')">One pass per tile &mdash; tAsA is flat, no SMEM stages</button>
            <button class="preset-btn" onclick="setTP('${id}',8,'float','3,4,3','(4, 16):(16, 1)','((4, 16), (2, 2)):((1@0, 1@1), (4@0, 16@1))')">8 passes of 8 &mdash; the finest ramp</button>
            <button class="preset-btn" onclick="setTP('${id}',32,'bfloat16_t','3,4,3','((4, 32), 2):((32, 1), 128)','((4, 32), (2, 2)):((1@0, 1@1), (4@0, 32@1))')">bf16, 4 passes of 32</button>
            <button class="preset-btn" onclick="setTP('${id}',16,'float','none','(4, 16):(1, 4)','((4, 16), (2, 2)):((1@0, 1@1), (4@0, 16@1))')">M-major SMEM &mdash; same shape, the order runs down the columns</button>
            <button class="preset-btn" onclick="setTP('${id}',16,'float','3,4,3','(4, 16):(16, 1)','(4, 16):(1@0, 1@1)')">No Rest &mdash; a single tile, so tAgA is rank 1 like tAsA</button>
            <button class="preset-btn" onclick="setTP('${id}',4096,'bfloat16_t','3,4,3','((32,128),2):((128,1),4096)','((32,128),(12,2)):((1@1,1@0),(32@1,128@0))')">Production scale &mdash; 4096-element tile, 2 stages, 24 tiles (clipped to fit)</button>
          </div>
        </div>

        <div class="hint">
          <b>It partitions vectors, not threads.</b> CuTe calls it the
          "VectorCopy Partitioner" and TMA has no threads to partition &mdash; one
          thread issues the whole instruction. What it splits is mode 0 of every
          tensor you pass, into <code>(TMA, TMA_Iter)</code>: one instruction's
          worth, and how many instructions the tile needs.<br><br>
          <b>The atom contributes one number.</b>
          <code>tma_layout_v = Layout&lt;NumValSrc&gt;</code> is rank 1; the
          atom's value layout is never inspected. A TMA atom is always
          <code>ThrID = 1:0</code> with
          <code>TV Layout Src = Dst = (1, N):(0,1)</code> &mdash; one thread, N
          values &mdash; and <code>NumValSrc</code> is that N. Src and Dst are
          identical for TMA load, multicast <em>and</em> store
          (<code>copy_traits_sm90_tma.hpp:103, 261, 363</code>); they diverge
          only for shuffling atoms like <code>ldmatrix</code>.<br><br>
          <b>You do not give a TMA atom a bit count.</b>
          <code>num_bits_per_tma</code> is <em>derived</em> by
          <code>make_tiled_tma_atom</code> from the box it inferred, so this tab
          asks for the atom's value count and reports the bit equivalent, not the
          other way round.<br><br>
          <b>SMEM decides the order; GMEM follows.</b> <code>layout_V</code> is
          built from <code>layout&lt;0&gt;(stensor)</code> alone, then applied
          <em>identically</em> to every tensor. That is why the function takes
          both and returns both, and why the swizzle is stripped first &mdash;
          <code>get_nonswizzle_portion</code>, exactly as when the box was
          derived. Changing the swizzle picker above changes nothing here, which
          is the point.<br><br>
          <b>Rest passes through.</b> <code>append&lt;R&gt;(layout_V, _)</code>
          puts an underscore on every remaining mode, so the tile indices survive
          untouched &mdash; you slice them later, at the <code>copy</code>. The
          one thing it asserts is
          <code>size&lt;0&gt;(stensor) == size&lt;0&gt;(tensor)</code>, which is
          what the caller's <code>group_modes(x, 0, 2)</code> is for.<br><br>
          <b>Not covered.</b> Multicast: <code>domain_offset</code> shifts into
          this CTA's slice of the instruction, and with
          <code>cta_layout = (1)</code> that offset is always zero.
        </div>
      </div>

      <div class="comp-results" style="grid-template-columns:1fr">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-tp-s-title">tAsA</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-tp-mode-btns">
                <button class="mode-btn" onclick="setTpMode('${id}','value')">value</button>
              </span>
              <button class="mode-btn" id="${id}-tp-s-svg-zoom" onclick="toggleZoom('${id}-tp-s-svg')">Zoom in</button>
              <button class="mode-btn" onclick="downloadSVG('${id}-tp-s-svg', 'tAsA.svg')">Download SVG</button>
            </span>
          </div>
          <div class="cuo-viz-desc">
            One SMEM tile, in tile coordinates &mdash; so <b>one hue</b>.
            Brightness says which atom pass covers each part of it: flat when the
            tile is exactly one atom, ramped light-to-dark when it takes several.
            Each band is consecutive in SMEM, which is the whole reason the split
            was taken in SMEM order.
          </div>
          <div class="viz-box"><div id="${id}-tp-s-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-tp-g-title">tAgA</span>
            <span style="display:flex;align-items:center;gap:4px">
              <button class="mode-btn" id="${id}-tp-g-svg-zoom" onclick="toggleZoom('${id}-tp-g-svg')">Zoom in</button>
              <button class="mode-btn" onclick="downloadSVG('${id}-tp-g-svg', 'tAgA.svg')">Download SVG</button>
            </span>
          </div>
          <div class="cuo-viz-desc">
            <b>Every</b> tile of <code>Rest</code>. <b>Hue is the tile</b>;
            brightness is the same pass-ramp as above, repeated verbatim inside
            each one &mdash; which is exactly the claim
            <code>tma_partition</code> makes, since it touches mode 0 only and
            Rest passes through untouched. Nothing is sliced here; you do that
            later at the copy with <code>tAgA[(None, k0, k1)]</code>. Red lines
            are the tile boundaries; cells name the GMEM coordinate. You slice Rest later, at the copy, with
            <code>tAgA[(None, k0, k1)]</code>.
          </div>
          <div class="viz-box"><div id="${id}-tp-g-svg"></div></div>
        </div>
      </div>
    </div>`;
}

// Same red the Divide tabs use for mode-0 accents.
const TP_TILE_EDGE = '#dc2626';
// Cell budget for the tAgA panel; real Rest counts blow straight past MAX_CELLS.
const TP_MAX_CELLS = 8192;

/** Two independent things, two independent channels: HUE says which tile,
 *  BRIGHTNESS says which instruction covers that part of it. A tile that takes
 *  one atom pass is flat; a tile needing several ramps light-to-dark across
 *  them. Every tile carries the identical ramp, which is the whole claim
 *  tma_partition makes — Rest passes through, so the split repeats verbatim. */
function tpShade(base, inst, iters) {
  if (iters <= 1) return base;
  const t = inst / (iters - 1);                        // 0 .. 1
  return t < 0.5 ? lightenRGB(base, (0.5 - t) * 0.7)   // early: lighter
                 : darkenRGB(base,  (t - 0.5) * 0.9);  // late:  darker
}

const tpState = {};

function renderTmaPartition(tabId) {
  showErr(`${tabId}-tp-error`, '');
  showWarn(`${tabId}-tp-warning`, '');
  try {
    const valsStr = document.getElementById(`${tabId}-tp-vals-input`).value;
    const dtype   = document.getElementById(`${tabId}-tp-dtype-input`).value;
    const smemStr = document.getElementById(`${tabId}-tp-smem-input`).value;
    const gmemStr = document.getElementById(`${tabId}-tp-gmem-input`).value;
    const swRaw   = document.getElementById(`${tabId}-tp-swizzle-input`).value;

    const elemBits = DTYPE_BITS[dtype];
    if (!elemBits) throw new Error(`Unknown element type "${dtype}"`);
    // NumValSrc = size<1>(ValLayoutSrc) — read straight off the atom. You do NOT
    // hand a TMA atom a bit count: make_tiled_tma_atom DERIVES num_bits_per_tma
    // from the box it inferred, and the atom then presents as (1, N) values.
    const numValSrc = parseInt(valsStr, 10);
    if (!Number.isFinite(numValSrc) || numValSrc <= 0) {
      throw new Error(`The value count N must be a positive integer, got "${valsStr}"`);
    }
    const numBits = numValSrc * elemBits;

    const sm = tmaParseSmemField(smemStr);
    const swInfo = tmaSwizzleInfo(sm.sw || (swRaw === 'none' ? null : parseSwizzleSpec(swRaw)), elemBits);
    const sp = parseLayout(sm.layoutStr);
    // The GMEM side is a coordinate tensor in the real flow, so basis strides
    // are expected here.
    const gp = parseLayout(gmemStr, { basis: true });

    // tma_partition receives tensors ALREADY grouped: mode 0 IS the tile and
    // everything after it is Rest. Do NOT re-group here — folding a stage mode
    // or the tile indices into the tile is exactly what made `((32,128),2)` look
    // like an 8192-element tile and invent a mismatch.
    //
    // One ambiguity: parseValue unwraps single-element parens, so a no-Rest
    // tensor written `((4,16))` is indistinguishable from `(4,16)`. Rule: a
    // TUPLE mode 0 means "tile, then Rest"; a scalar mode 0 means the whole
    // layout is the tile.
    const tileOf = (L) => is_tuple(L.shape[0])
      ? { tile: L.shape[0], stride: L.stride[0],
          rest: L.shape.slice(1), restStride: L.stride.slice(1) }
      // Scalar mode 0: the whole layout is the tile, so there is no Rest — the
      // stride list has to be emptied alongside the shape list, or the printed
      // result comes out rank-mismatched, e.g. `((64,1)):((1,0),1)`.
      : { tile: L.shape, stride: L.stride, rest: [], restStride: [] };
    const sPart = tileOf(sp), gPart = tileOf(gp);
    const sTile = sPart.tile, sTileStride = sPart.stride;
    const gTile = gPart.tile;
    const restShape = gPart.rest;
    const [S0, S1] = productEach(sTile);
    const sFlatShape = flatten(sTile), sFlatStride = flatten(sTileStride);
    let tileSize = 1;
    for (const x of sFlatShape) tileSize *= x;
    let gTileSize = 1;
    for (const x of flatten(gTile)) gTileSize *= x;

    // The one thing tma_partition asserts, per tensor:
    //   CUTE_STATIC_ASSERT_V(size<0>(stensor) == size<0>(tensor))
    if (tileSize !== gTileSize) {
      throw new Error(
        `size<0>(stensor) = ${tileSize} but size<0>(gmem tensor) = ${gTileSize}. ` +
        `This is tma_partition's only static assert — mode 0 must already BE the ` +
        `tile on both sides, which is what group_modes(x, 0, 2) is for.`);
    }


    // right_inverse(get_nonswizzle_portion(...)) — a partial inverse if the
    // layout is not a permutation, which silently shortens layout_v.
    const inv = new Array(tileSize).fill(-1);
    for (let d = 0; d < tileSize; d++) {
      const off = tmaFlatOffset(d, sFlatShape, sFlatStride);
      if (off < 0 || off >= tileSize || inv[off] !== -1) {
        throw new Error(
          `smem layout is not a permutation of [0, ${tileSize}) — ` +
          `right_inverse() would return a PARTIAL inverse, so layout_v would not ` +
          `cover the tile and the split would silently be wrong. CuTe does not ` +
          `check this.`);
      }
      inv[off] = d;
    }

    // logical_divide(layout_v, Layout<NumValSrc>) needs the split to be exact.
    if (tileSize % numValSrc !== 0) {
      const ok = [];
      for (let n = 1; n <= tileSize; n++) if (tileSize % n === 0) ok.push(n);
      throw new Error(
        `logical_divide(layout_v, Layout<${numValSrc}>) is not exact: the tile holds ` +
        `${tileSize} elements, which ${numValSrc} does not divide. One instruction ` +
        `has to tile the mode evenly. Value counts of ` +
        `{${ok.slice(0, 12).join(', ')}${ok.length > 12 ? ', …' : ''}} would divide it.`);
    }
    const iters = tileSize / numValSrc;

    // NB: not productEach — that always returns exactly two entries
    // (`[product(shape[0]), product(shape[1])]`), so an empty Rest would come
    // back as [undefined, undefined] and print as "((32,8), , )".
    const extentsOf = (modes) => modes.map(x => product(x));
    const restExtents = extentsOf(restShape);
    let restSize = 1;
    for (const e of restExtents) restSize *= e;
    const sRestExtents = extentsOf(sPart.rest);

    // Highlight picker
    const hlRaw = (document.getElementById(`${tabId}-tp-highlight`).value || '').trim();
    let highlight = null;
    if (hlRaw !== '') {
      const v = parseInt(hlRaw, 10);
      if (Number.isFinite(v) && v >= 0 && v < iters) highlight = v;
      else showWarn(`${tabId}-tp-warning`,
        `Instruction "${hlRaw}" is out of range [0, ${iters}) — showing all.`);
    }

    // Compute the results rather than describing them: this is literally CuTe's
    //   layout_V = logical_divide(right_inverse(smem tile), Layout<NumValSrc>)
    //   coalesce(tensor.compose(layout_V), Shape<Shape<_1,_1>>)
    // and it reproduces CuTeDSL's printout exactly — including the cases where
    // mode 0 does NOT coalesce to a flat (N, Iter), because its sub-modes sit on
    // different basis axes. Synthesizing `(${numValSrc}, ${iters})` got that
    // wrong for every transposed GMEM tile.
    const sTileL = new Layout(sTile, sTileStride);
    const gTileL = new Layout(gTile, gPart.stride);
    const layoutV = logical_divide(right_inverse(sTileL), new Layout(numValSrc));
    const COALESCE_PROFILE = [1, 1];                 // Shape<Shape<_1,_1>>
    const sMode0 = coalesce(composition(sTileL, layoutV), COALESCE_PROFILE);
    const gMode0 = coalesce(composition(gTileL, layoutV), COALESCE_PROFILE);

    const joinLayout = (mode0, restSh, restSt) => formatLayoutStr(
      [mode0.shape].concat(restSh), [mode0.stride].concat(restSt));
    const sPartStr = joinLayout(sMode0, sPart.rest, sPart.restStride);
    const gPartStr = joinLayout(gMode0, gPart.rest, gPart.restStride);
    const partStr = gPartStr;

    // Print it the way CuTe prints a Copy_Atom, so it can be matched by eye.
    document.getElementById(`${tabId}-tp-atom-result`).innerHTML =
      `<div class="cuo-result-line"><b>Copy Atom</b></div>` +
      `<div class="cuo-result-line">&nbsp;&nbsp;ThrID:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1:0</div>` +
      `<div class="cuo-result-line">&nbsp;&nbsp;TV Layout Src: (1,${numValSrc}):(0,1)</div>` +
      `<div class="cuo-result-line">&nbsp;&nbsp;TV Layout Dst: (1,${numValSrc}):(0,1)</div>` +
      `<div class="cuo-result-line">&nbsp;&nbsp;Value type:&nbsp;&nbsp;&nbsp;&nbsp;${dtype}</div>` +
      `<div class="cuo-result-line" style="color:#9ca3af">NumValSrc = size&lt;1&gt;(ValLayoutSrc) = ` +
      `<b>${numValSrc}</b>. tma_layout_v = ${numValSrc}:1, rank 1 &mdash; only this count is used. ` +
      `Equivalent num_bits_per_tma = ${numValSrc} &times; ${elemBits} = ${numBits} ` +
      `(${numBits / 8} B), which make_tiled_tma_atom <em>derives</em> rather than accepts.</div>`;

    // Just the two results. The derivation (inv_smem_layout, layout_v, layout_V)
    // is in the hint for anyone who wants it — nobody reads it inline.
    document.getElementById(`${tabId}-tp-result`).innerHTML =
      `<div class="cuo-result-line">tAsA = <b>${tmaEsc(sPartStr)}</b></div>` +
      `<div class="cuo-result-line">tAgA = <b>${tmaEsc(gPartStr)}</b></div>`;

    const prev = tpState[tabId] || {};
    tpState[tabId] = {
      sp: { shape: sTile, stride: sTileStride }, S0, S1, tileSize, numValSrc, iters,
      highlight, restExtents, partStr, gPartStr, sPartStr, dtype,
      inv, gTile, gTileStride: gPart.stride, gFull: gp, gRestBase: gPart.rest.length,
      T0: product(sTile[0]), T1: product(sTile[1]),
      gBasis: gp.basis, gNd: gp.basis ? Math.max(basisRank(gp.stride), gp.ndim || 2) : 0,
      numBits, mode: (prev.mode instanceof Set) ? prev.mode : new Set(),
    };
    tpRenderViz(tabId);
    updateOuterTabLabel(tabId, `tma_partition:${numValSrc}x${iters}`);
  } catch (e) {
    showErr(`${tabId}-tp-error`, e.message);
    for (const side of ['s', 'g']) {
      const el = document.getElementById(`${tabId}-tp-${side}-svg`);
      if (el) el.innerHTML = '';
    }
  }
}

function tpRenderViz(tabId) {
  const s = tpState[tabId];
  if (!s) return;
  const modes = s.mode instanceof Set ? s.mode : new Set();
  const { T0, T1 } = s;

  // Both panels are drawn in TILE coordinates and share one colour map: which
  // INSTRUCTION owns each element. A tile that is exactly one atom comes out a
  // single colour; a tile holding several instructions comes out banded.
  const instAt = (m, n) => Math.floor(layoutAt(s.sp.shape, s.sp.stride, m, n) / s.numValSrc);
  const lit = (i) => s.highlight === null || i === s.highlight;

  // ---- tAsA: the SMEM tile ----
  // One tile, so one hue; the ramp across it is the instruction count.
  const sSVG = buildColoredLayoutSVG(s.sp.shape, s.sp.stride, modes, (m, n, offset) => {
    const i = instAt(m, n);
    if (!lit(i)) return { bg: '#e8e8e8', stroke: '#ddd', text: null };
    return { bg: tpShade(colorTV(0), i, s.iters), text: [String(offset)] };
  });

  // ---- tAgA: every tile of Rest, each partitioned identically ----
  // Tiles are laid out in canonical blocks — tile (r0, r1) at rows r0*T0, cols
  // r1*T1 — rather than at their true stride positions, so the picture stays a
  // grid whatever the Rest strides are. The truth is in the cell labels, which
  // evaluate the actual GMEM layout at (m, n, r...).
  const rest = s.restExtents;
  let R0 = rest.length > 0 ? rest[0] : 1;
  let R1 = rest.length > 1 ? rest[1] : 1;
  const restBeyond = rest.slice(2).reduce((a, b) => a * b, 1);
  // Real tile counts get large fast — 24 tiles of a 4096-element tile is ~100k
  // cells. Draw as many whole tiles as fit a budget and say how many were left
  // out, rather than emitting an unreadable and slow grid.
  const fullR0 = R0, fullR1 = R1;
  while (R0 * R1 > 1 && T0 * R0 * T1 * R1 > TP_MAX_CELLS) {
    if (R0 >= R1 && R0 > 1) R0--; else if (R1 > 1) R1--; else break;
  }
  const clipped = R0 !== fullR0 || R1 !== fullR1;
  const gShape = [T0 * R0, T1 * R1];
  // The GMEM tensor is (tile, Rest...) with a possibly nested tile, so build a
  // congruent coordinate: idx2crd over the tile, then one index per Rest mode.
  const gLabel = (m, n, r0, r1) => {
    const flatTile = m + n * product(s.gTile[0]);
    const shape  = s.gRestBase ? s.gFull.shape  : s.gTile;
    const stride = s.gRestBase ? s.gFull.stride : s.gTileStride;
    const crd = s.gRestBase
      ? [idx2crd(flatTile, s.gTile)].concat(
          s.gFull.shape.slice(1).map((sh, k) => idx2crd(k === 0 ? r0 : (k === 1 ? r1 : 0), sh)))
      : idx2crd(flatTile, s.gTile);
    if (s.gNd) {
      const out = new Array(s.gNd).fill(0);
      crd2basis(crd, shape, stride, out);
      return `(${out.join(',')})`;
    }
    return String(crd2idx(crd, shape, stride));
  };
  const gSVG = buildColoredLayoutSVG(gShape, [1, gShape[0]], modes, (M, N) => {
    const m = M % T0, r0 = Math.floor(M / T0);
    const n = N % T1, r1 = Math.floor(N / T1);
    const i = instAt(m, n);
    if (!lit(i)) return { bg: '#e8e8e8', stroke: '#ddd', text: null };
    // Hue per tile, the same brightness ramp inside each.
    return { bg: tpShade(colorTV(r0 + r1 * R0), i, s.iters), text: [gLabel(m, n, r0, r1)] };
  }, {
    // Tile boundaries are lines ACROSS cells, so they go in the overlay rather
    // than being faked with a stroke on the edge cells.
    overlay: ({ cs, margin, W, H }) => {
      const sw = Math.max(1.5, cs * 0.06);
      let out = '';
      for (let r = 0; r <= R0; r++) {
        const y = margin + r * T0 * cs;
        out += `<line x1="${margin}" y1="${y}" x2="${W}" y2="${y}" ` +
               `stroke="${TP_TILE_EDGE}" stroke-width="${sw}"/>`;
      }
      for (let c = 0; c <= R1; c++) {
        const x = margin + c * T1 * cs;
        out += `<line x1="${x}" y1="${margin}" x2="${x}" y2="${H}" ` +
               `stroke="${TP_TILE_EDGE}" stroke-width="${sw}"/>`;
      }
      return out;
    },
  });

  const ramp = s.iters === 1
    ? `one atom pass covers the tile, so it is flat`
    : `brightness = which of the ${s.iters} atom passes covers it, light to dark`;
  const legend = (extra) =>
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `${extra} &mdash; ${ramp}` +
    (s.highlight !== null ? ` &mdash; <b>pass ${s.highlight} only</b>` : '') +
    `</div>`;
  document.getElementById(`${tabId}-tp-s-svg`).innerHTML =
    legend(`one hue: this is one tile; cell = SMEM offset`) + sSVG;
  document.getElementById(`${tabId}-tp-g-svg`).innerHTML =
    legend(`hue = which tile (${clipped ? `${R0}&times;${R1} of ${fullR0}&times;${fullR1} drawn` : `${R0}&times;${R1} of them`}, ` +
           `<span style="color:${TP_TILE_EDGE}">outlined in red</span>` +
           (restBeyond > 1 ? `, further Rest modes at 0` : '') +
           `); cell = ${s.gBasis ? 'GMEM coordinate' : 'GMEM offset'}`) + gSVG;
  applyZoomState(`${tabId}-tp-s-svg`);
  applyZoomState(`${tabId}-tp-g-svg`);
  updateModeBtns(`${tabId}-tp-mode-btns`, modes);
  document.getElementById(`${tabId}-tp-s-title`).textContent =
    `tAsA = ${s.sPartStr} — one tile, ${s.iters} instruction${s.iters === 1 ? '' : 's'}`;
  const totalTiles = fullR0 * fullR1 * restBeyond;
  document.getElementById(`${tabId}-tp-g-title`).textContent =
    `tAgA = ${s.gPartStr}` +
    (rest.length
      ? (clipped ? ` — ${R0 * R1} of ${totalTiles} tiles drawn` : ` — all ${totalTiles} tiles`)
      : '');
}

function setTpMode(tabId, mode) {
  const s = tpState[tabId];
  if (!s) return;
  let modes = s.mode;
  if (!(modes instanceof Set)) { modes = new Set(); s.mode = modes; }
  if (modes.has(mode)) modes.delete(mode); else modes.add(mode);
  tpRenderViz(tabId);
}

function setTP(tabId, vals, dtype, sw, smem, gmem) {
  document.getElementById(`${tabId}-tp-vals-input`).value    = vals;
  document.getElementById(`${tabId}-tp-dtype-input`).value   = dtype;
  document.getElementById(`${tabId}-tp-swizzle-input`).value = sw;
  document.getElementById(`${tabId}-tp-smem-input`).value    = smem;
  document.getElementById(`${tabId}-tp-gmem-input`).value    = gmem;
  document.getElementById(`${tabId}-tp-highlight`).value     = '';
  renderTmaPartition(tabId);
}

function exportTP(tabId) {
  exportURL(`${tabId}-tp-export`, 'tma_partition',
    document.getElementById(`${tabId}-tp-vals-input`).value,
    document.getElementById(`${tabId}-tp-dtype-input`).value,
    document.getElementById(`${tabId}-tp-swizzle-input`).value,
    document.getElementById(`${tabId}-tp-smem-input`).value,
    document.getElementById(`${tabId}-tp-gmem-input`).value);
}
