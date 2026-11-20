// partition_shape_A / partition_shape_B / partition_shape_C tab (MMA scope).
//
//   auto acc_shape = tiled_mma.partition_shape_C(CTA_TILER[:2]);
//   Tensor tCtAcc  = tiled_mma.make_fragment_C(acc_shape);
//
// The thread-free, stride-free sibling of partition_A/B/C. It answers "what
// shape is one thread's share of this tile?" from the tile's SHAPE alone, so a
// kernel can size a buffer (registers, TMEM, a staged SMEM layout) before any
// tensor or pointer exists. Every thread's partition has the same shape, which
// is why no thread index is involved.
//
// C++ CuTe builds it as "partition a dummy compact layout, keep the shape"
// (mma_atom.hpp:563-610). CuTeDSL — this project's oracle — differs in the two
// places a picture would otherwise lie:
//
//   * the Rest modes come back FLAT. With a permuting layout, partition_A's
//     Rest is nested — ((2,32):(32,1),32,16) on a (128,48) A gives
//     ((2,2,2),(2,2),3) — while partition_shape_A gives ((2,2,2),4,3);
//   * an extent SMALLER than one warp tile rounds UP to 1 (shape_div), rather
//     than being rejected the way partition_A's zipped_divide rejects it.
//
// So the rule this file implements, pinned by tests/cases.json, is
//
//   mode 0     the atom's per-thread value shape            (make_mma_atom)
//   modes 1-2  shape_div(size(shape[i]), atom[i] * warps[i]) (permutation ignored)
//   modes 3+   shape[3:], verbatim — never tiled
//
// An extent that is neither a multiple nor a divisor of its warp tile is the
// error CuTeDSL raises ("failed to partition ... with tiled_mma").
//
// A shape has no strides. They are chosen by whatever consumes it, and the
// usual consumer is make_fragment_X, which this tab reports and hands off to.

/** `fmt` for an int tuple, the CuTe way: `((2,2,2),4,3)`. */
function pshpFormatShape(s) {
  return is_tuple(s) ? `(${s.map(pshpFormatShape).join(',')})` : String(s);
}

/** Parse the shape field. A full layout is accepted — `shape(layout)` is what
 *  C++'s partition_shape takes of its argument anyway — and the caller is told
 *  that the strides were dropped. */
function pshpParseShape(raw) {
  const text = String(raw || '').trim();
  if (!text) throw new Error('Enter a shape such as (64, 48).');
  const hadStride = topLevelColon(text) !== -1;
  const p = parseLayout(text);
  if (p.basis) throw new Error('A shape has no strides; scaled-basis strides do not apply here.');
  // parseLayout pads a scalar to rank 2 with a trailing 1; keep what was typed.
  const typed = hadStride ? p.shape : parseValue(text, false);
  return { shape: typed, hadStride };
}

/** `tiled_mma.partition_shape_X(shape)` and `make_fragment_X` of the result.
 *  DOM-free, so tests/run.js can diff it against CuTeDSL. */
function pshpComputeShape(atom, atomLayout, perm, which, shapeIn) {
  const spec = PABC_OPERAND[which];
  if (!spec) throw new Error(`Unknown operand "${which}" — expected A, B or C.`);
  const modes = is_tuple(shapeIn) ? shapeIn : [shapeIn];
  if (modes.length < 2)
    throw new Error(
      `partition_shape_${which} takes the (${spec.axes.join(', ')}) shape: rank 2 or more, ` +
      `got rank ${modes.length}.`);

  // The same TiledMMA construction every MMA tab uses, so a bad atom_layout_mnk
  // is reported identically. Mode 0 of any partition is the atom's value mode.
  const tiled = mtmComputeTiledMma(atom, atomLayout, perm);
  const frgShape = tiled[which].partition.shape[0];

  const atomExt = spec.mnk.map(i => atom.shapeMNK[i]);
  const warps   = spec.mnk.map(i => tiled.atomLayoutMNK[i]);
  const cover   = atomExt.map((e, i) => e * warps[i]);
  const extents = [product(modes[0]), product(modes[1])];
  const roundedUp = [false, false];
  const rest = extents.map((e, i) => {
    if (e % cover[i] === 0) return e / cover[i];
    if (cover[i] % e === 0) { roundedUp[i] = true; return 1; }
    throw new Error(
      `The ${spec.axes[i]} extent ${e} is neither a multiple nor a divisor of the ` +
      `${cover[i]} ${spec.axes[i]} one Rest position covers (atom ${atomExt[i]} × ` +
      `${warps[i]} ${atom.warpgroup ? 'warpgroup' : 'warp'}${warps[i] === 1 ? '' : 's'} along ${spec.axes[i]}). CuTeDSL rejects ` +
      `it: "failed to partition ... with tiled_mma".`);
  });
  const trailing = modes.slice(2);
  const shape = [frgShape, rest[0], rest[1], ...trailing];

  // Strides come from the consumer. make_fragment_X(shape) reads the shape as
  // CuTe's make_layout(shape) does — compact column-major — and then applies
  // its own rule, which is pabcMakeFragment's, shared with partition_A/B/C.
  // A WGMMA SMEM operand has no register fragment: its make_fragment_X builds a
  // descriptor from an SMEM tensor, and CuTeDSL rejects a bare shape there.
  const descriptor = (atom[which] || {}).space === 'SMEM';
  const fragment = descriptor ? null : pabcMakeFragment(new Layout(shape, autoStride(shape)), which);

  // The input as CuTe reads a bare shape — make_layout(shape), compact
  // column-major (a size-1 mode gets stride 0, as CuTe prints it). This is the
  // dummy layout C++'s partition_shape partitions; the tab prints it.
  const inputLayout = new Layout(modes, mtmCompactStride(modes));

  // The tile partition_shape steps across the shape: the TiledMMA operand
  // WITHOUT permutation_mnk (which this op ignores), i.e. exactly the block
  // one Rest position covers. Grid 1 draws it with its T/V ownership, the same
  // region partition_A/B/C's level-1 grid shows when there is no permutation.
  const tile = (perm.some(Boolean) ? mtmComputeTiledMma(atom, atomLayout, [null, null, null]) : tiled)[which];

  return {
    which, spec, shape, shapeStr: pshpFormatShape(shape),
    input: modes, inputStr: pshpFormatShape(modes), inputLayout, tile,
    frgShape, frgSize: product(frgShape),
    atomExt, warps, cover, extents, rest, roundedUp, trailing,
    fragment, descriptor, threads: tiled.threads, thrLayoutVmnk: tiled.thrVmnk, tileMNK: tiled.tileMNK,
    unit: atom.warpgroup ? 'warpgroup' : 'warp',
    // permutation_mnk changes partition_X but not this — say so when present.
    permIgnored: perm.some(Boolean),
  };
}

// ═══════════════════════════════════════════════════════
//  The tab
// ═══════════════════════════════════════════════════════

function generatePartitionShapeABCTabContent(id) {
  return `
    <!-- partition_shape_A / B / C panel -->
    <div id="${id}-tab-partition_shape_abc" class="panel">
      <div class="controls">
        <h2>partition_shape_A / B / C</h2>

        <div class="form-group">
          <label>Which operand</label>
          <div class="seg-control seg-stacked" id="${id}-pshp-operand-btns">
            <button class="mode-btn active" onclick="setPshpOperand('${id}', 'A')">partition_shape_A</button>
            <button class="mode-btn" onclick="setPshpOperand('${id}', 'B')">partition_shape_B</button>
            <button class="mode-btn" onclick="setPshpOperand('${id}', 'C')">partition_shape_C</button>
          </div>
        </div>

        <details class="cuo-section" open>
          <summary>1. TiledMMA</summary>
          <div class="cuo-section-body">
            <button class="btn btn-render btn-decode" id="${id}-pshp-decode-mma" onclick="pshpDecodePasteboard('${id}')">Decode from Pasteboard</button>
            ${mtmManualConstructorMarkup({id, p: 'pshp', render: 'renderPartitionShapeABC', setOp: 'setPshpOp', atomLayout: '(2, 2, 1)'})}
            <div id="${id}-pshp-tiled-result" class="cuo-result"></div>
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>2. The shape being partitioned</summary>
          <div class="cuo-section-body">
            ${layoutInputField({
              id: `${id}-pshp-shape-input`,
              label: `The <span id="${id}-pshp-shape-axes">(M, K)</span> shape &mdash; ` +
                     `the argument to <span id="${id}-pshp-shape-fn">partition_shape_A</span>`,
              hint: 'a shape, rank 2 or more; extra modes pass through',
              value: '(64, 48)'
            })}
          </div>
        </details>

        ${statusDivs(`${id}-pshp`)}
        <button class="btn btn-render" onclick="renderPartitionShapeABC('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-pshp-export" onclick="exportPshp('${id}')">Export URL</button>
        <button class="btn btn-render" style="margin-top:6px" id="${id}-pshp-make-fragment" onclick="pshpOpenFragment('${id}')">make_fragment</button>
        <div id="${id}-pshp-result" class="cuo-result"></div>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setPshp('${id}','A','f16bf16','half_t','float',16,'(2, 2, 1)','','(64, 48)')">A over (64, 48) &mdash; the canonical 4-warp GEMM</button>
            <button class="preset-btn" onclick="setPshp('${id}','B','f16bf16','half_t','float',16,'(2, 2, 1)','','(32, 48)')">B &mdash; (N, K)</button>
            <button class="preset-btn" onclick="setPshp('${id}','C','f16bf16','half_t','float',16,'(2, 2, 1)','','(64, 32)')">C &mdash; sizing the accumulator</button>
            <button class="preset-btn" onclick="setPshp('${id}','A','f16bf16','half_t','float',16,'(2, 2, 1)','','(64, 48, 4)')">A with a k-loop mode &mdash; passed through</button>
            <button class="preset-btn" onclick="setPshp('${id}','A','f16bf16','half_t','float',16,'(2, 2, 1)','((2,32):(32,1), 32, 16)','(128, 48)')">Permuting layout &mdash; Rest stays flat, unlike partition_A</button>
            <button class="preset-btn" onclick="setPshp('${id}','A','f16bf16','half_t','float',16,'(2, 2, 1)','','(16, 64)')">M smaller than one warp tile &mdash; rounds up to 1</button>
            <button class="preset-btn" onclick="setPshp('${id}','A','tf32','na','na',8,'(2, 2, 1)','','(64, 32)')">MmaTF32Op, m16n8k8</button>
            <button class="preset-btn" onclick="setPshp('${id}','C','wg_f16bf16','half_t','float',16,'(2, 1, 1)','','(256, 128)',64,'SMEM','K','K','half_t')">WGMMA C &mdash; sizing a 256&times;128 accumulator</button>
            <button class="preset-btn" onclick="setPshp('${id}','A','wg_f16bf16','half_t','float',16,'(2, 1, 1)','','(128, 64, 3)',64,'SMEM','K','K','half_t')">WGMMA SMEM A &mdash; a shape, but no register fragment</button>
          </div>
        </div>

        <div class="hint">
          <b>partition_A with no thread and no tensor.</b>
          <code>partition_shape_X(shape)</code> is the shape every thread's
          <code>partition_X</code> would have over a tile of that shape. A kernel
          uses it to size a buffer before any tensor or pointer exists &mdash;
          the TMEM accumulator, a register fragment, or the SMEM operand layout
          that <code>make_smem_layout_a/b</code> builds.<br><br>
          <b>It returns a shape, not a layout.</b> The strides come from what
          consumes it: <code>make_fragment_X(shape)</code> reads the shape as
          compact column-major, like <code>make_layout(shape)</code>. The result
          box shows that fragment; <b>make_fragment</b> opens it in the register
          view.<br><br>
          <b>Modes 1 and 2 are counted in TiledMMA tiles (grid 1), and permutation_mnk does
          not enter.</b> Each is <code>shape_div(extent, atom &times; warps)</code>:
          an extent smaller than one tile rounds up to 1, and one that is
          neither a multiple nor a divisor is rejected. With a permuting layout
          <code>partition_A</code> returns a nested Rest such as
          <code>(2,2)</code>; <code>partition_shape_A</code> returns its flat
          size, <code>4</code>.<br><br>
          See <b>partition_A / B / C</b> for one thread's view of a real tensor.
        </div>
      </div>

      <div class="comp-results" style="grid-template-columns:1fr">
        ${layoutVisualization({
          hostId: `${id}-pshp-tile-svg`,
          title: 'One TiledMMA tile',
          titleId: `${id}-pshp-tile-title`,
          onMode: mode => `setPshpMode('${id}','tile','${mode}')`,
          beforeGrid: `<div class="cuo-viz-desc" id="${id}-pshp-tile-desc">
            <b>The unit partition_shape steps across the shape:</b> one TiledMMA
            tile without <code>permutation_mnk</code> &mdash; the atom times the
            warps (or warpgroups) along each axis &mdash; with the
            <code>T</code>/<code>V</code> that owns each cell, as in
            partition_A/B/C's first grid. Mode 0 of the result is one thread's
            share of this tile. A WGMMA shared-memory operand is one uniform
            descriptor block. Grid 2 shows how many of these tiles cover the shape.
          </div>`,
        })}
        ${layoutVisualization({
          hostId: `${id}-pshp-sweep-svg`,
          title: 'Rest modes over the shape',
          titleId: `${id}-pshp-sweep-title`,
          onMode: mode => `setPshpMode('${id}','sweep','${mode}')`,
          beforeGrid: `<div class="cuo-viz-desc" id="${id}-pshp-sweep-desc">
            <b>Modes 1 and 2 of the returned shape, one cell per Rest position.</b>
            Each cell is one copy of grid 1's TiledMMA tile, and the grid is that Rest shape read
            column-major: <code>value</code> is the Rest index a fragment walks,
            <code>coord</code> its (i, j). The last line is the element where the
            tile starts. Mode 0 is listed above the grid, not drawn: it is what
            every cell holds per thread.
          </div>`,
        })}
      </div>
    </div>`;
}

const pshpState = {};

function pshpReadInputs(tabId) {
  const inputs = mtmReadConstructorControls(tabId, 'pshp');
  const mma = mtmDecodeConstructor(inputs);
  const parsed = pshpParseShape(document.getElementById(`${tabId}-pshp-shape-input`).value);
  return { ...mma, constructor: inputs, ...parsed };
}

function renderPartitionShapeABC(tabId) {
  showErr(`${tabId}-pshp-error`, '');
  showWarn(`${tabId}-pshp-warning`, '');
  document.getElementById(`${tabId}-pshp-result`).innerHTML = '';
  const prev = pshpState[tabId] || {};
  const which = prev.which || 'A';
  pshpSyncOperandField(tabId, which);
  try {
    const inp = pshpReadInputs(tabId);
    const r = pshpComputeShape(inp.atom, inp.atomLayout, inp.perm, which, inp.shape);
    document.getElementById(`${tabId}-pshp-result`).innerHTML =
      `<div class="cuo-result-line">partition_shape_${which}(${r.inputStr}) = <b>${r.shapeStr}</b></div>` +
      (r.descriptor
        ? `<div class="cuo-result-line">make_fragment_${which}(shape): <b>not available</b>` +
          `<span style="color:#9ca3af"> &mdash; ${which} is a WGMMA SMEM descriptor operand; its fragment is ` +
          `built from a partitioned SMEM tensor (partition_${which} &rarr; make_fragment), and CuTeDSL rejects a bare shape.</span></div>`
        : `<div class="cuo-result-line">make_fragment_${which}(shape) = ` +
          `<b>${formatLayoutStr(r.fragment.shape, r.fragment.stride)}</b>` +
          `<span style="color:#9ca3af"> &mdash; the strides a consumer chooses</span></div>`);
    const makeFragmentBtn = document.getElementById(`${tabId}-pshp-make-fragment`);
    if (makeFragmentBtn) makeFragmentBtn.disabled = r.descriptor;

    document.getElementById(`${tabId}-pshp-tiled-result`).innerHTML =
      `<div class="cuo-result-line">thr_layout_vmnk = <b>` +
      `${formatLayoutStr(r.thrLayoutVmnk.shape, r.thrLayoutVmnk.stride)}</b> &mdash; ` +
      `${r.threads} threads</div>` +
      `<div class="cuo-result-line">One Rest position = <b>${r.cover.join('&times;')}</b> ` +
      `(${r.spec.axes.join(', ')})<span style="color:#9ca3af"> &mdash; atom ` +
      `${r.atomExt.join('&times;')} &times; ${r.unit}s ${r.warps.join('&times;')}</span></div>`;

    const notes = [];
    if (inp.hadStride) notes.push('The strides you entered are ignored: partition_shape takes only the shape.');
    r.roundedUp.forEach((up, i) => {
      if (up) notes.push(
        `${r.spec.axes[i]} = ${r.extents[i]} is smaller than the ${r.cover[i]} one Rest position covers, ` +
        `so mode ${i + 1} rounds up to 1 and that tile is only partly inside the shape.`);
    });
    if (notes.length) showWarn(`${tabId}-pshp-warning`, notes.join(' '));

    pshpState[tabId] = {
      ...r, which, constructor: inp.constructor,
      mode: prev.mode instanceof Set ? prev.mode : new Set(['coord']),
      tileMode: prev.tileMode instanceof Set ? prev.tileMode : new Set(['value']),
    };
    pshpRenderTileViz(tabId);
    pshpRenderSweepViz(tabId);
    updateOuterTabLabel(tabId, `partition_shape_${which}:${r.shapeStr}`);
  } catch (e) {
    showErr(`${tabId}-pshp-error`, e.message);
    document.getElementById(`${tabId}-pshp-result`).innerHTML = '';
    for (const g of ['tile', 'sweep']) {
      const el = document.getElementById(`${tabId}-pshp-${g}-svg`);
      if (el) el.innerHTML = '';
    }
    if (pshpState[tabId]) {
      const { which: w, mode, tileMode } = pshpState[tabId];
      pshpState[tabId] = { which: w, mode, tileMode };
    }
  }
}

/** One TiledMMA tile (no permutation_mnk): the block a single Rest position
 *  covers, with the T/V that owns each cell — the ownership comes from the
 *  shared mtmOperandGrid / mtmTvLines, so it matches make_tiled_mma and
 *  partition_A/B/C. The shared value/index/coord labels read the tile as a
 *  column-major layout of its own. A WGMMA descriptor operand is uniform. */
function pshpRenderTileViz(tabId) {
  const s = pshpState[tabId];
  if (!s || !s.tile) return;
  const modes = s.tileMode instanceof Set ? s.tileMode : new Set(['value']);
  const [M, N] = s.tile.tile;
  const smem = s.tile.space === 'SMEM';
  const grid = mtmOperandGrid(s.tile);
  const svg = buildColoredLayoutSVG([M, N], [1, M], modes, (m, n, value, index) => {
    const labels = buildCellLines(modes, value, index, `(${m},${n})`);
    const entries = grid[m][n].entries;
    if (smem) return { bg: '#a7f3d0', text: labels };
    if (!entries.length) return { bg: '#f0f0f0', text: ['—'] };
    return { bg: colorTV(entries[0].t), text: [...mtmTvLines(entries, null), ...labels] };
  });
  const axes = s.spec.axes;
  const full = formatLayoutStr(s.inputLayout.shape, s.inputLayout.stride);
  const head =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `one TiledMMA tile = <b style="color:#93c5fd">${M}&times;${N}</b> (${axes.join(', ')}): atom ` +
    `${s.atomExt.join('&times;')} &times; ${s.unit}s ${s.warps.join('&times;')}` +
    (smem ? ` &mdash; a shared-memory descriptor block, the same for every thread of a warpgroup`
          : ` &mdash; each thread holds <b>${pshpFormatShape(s.frgShape)}</b> of it (mode 0)`) +
    `<br>partition_shape_${s.which} steps it ${s.rest.join('&times;')} times over ` +
    `make_layout(${s.inputStr}) = ${full} (grid 2)` +
    s.roundedUp.map((up, i) => up
      ? `<br><b style="color:#fbbf24">${axes[i]} = ${s.extents[i]}</b> is smaller than this tile's ${s.cover[i]}, ` +
        `so only part of it lies inside the shape` : '').join('') +
    (s.permIgnored ? '<br>permutation_mnk is ignored by partition_shape; this is the tile without it.' : '') +
    `</div>`;
  renderLayoutVisualization({ hostId: `${tabId}-pshp-tile-svg`, modes, svg: head + svg });
  document.getElementById(`${tabId}-pshp-tile-title`).textContent =
    `${s.which}: one TiledMMA tile — ${M}×${N} (${axes.join(', ')})`;
}

/** Modes 1-2 as a grid, one cell per Rest position. The grid IS a layout —
 *  make_layout of the Rest shape, column-major — so it gets the shared
 *  value/index/coord labels, plus the element where each warp tile starts. */
function pshpRenderSweepViz(tabId) {
  const s = pshpState[tabId];
  if (!s || !s.shape) return;
  const modes = s.mode instanceof Set ? s.mode : new Set(['coord']);
  const [R0, R1] = s.rest;
  const hostId = `${tabId}-pshp-sweep-svg`;
  const svg = buildColoredLayoutSVG([R0, R1], [1, R0], modes, (i, j, value, index) => {
    const lines = buildCellLines(modes, value, index, `(${i},${j})`);
    lines.push(`@(${i * s.cover[0]},${j * s.cover[1]})`);
    return { bg: colorHighlight(value), stroke: '#1e3a5f', sw: 1.5, text: lines };
  });
  const head =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `<b style="color:#93c5fd">${s.shapeStr}</b> = mode 0 <b>${pshpFormatShape(s.frgShape)}</b> ` +
    `(${s.frgSize} values per thread per tile) + this <b>${R0}&times;${R1}</b> grid of ` +
    `${s.cover.join('&times;')} TiledMMA tiles (grid 1) over the ${s.extents.join('&times;')} ` +
    `(${s.spec.axes.join(', ')}) shape` +
    (s.trailing.length
      ? ` + <b>${s.trailing.map(pshpFormatShape).join(', ')}</b> passed through unchanged`
      : '') +
    (s.permIgnored
      ? `<br>permutation_mnk does not change this shape; partition_${s.which} may nest these modes.`
      : '') +
    `</div>`;
  renderLayoutVisualization({ hostId, modes, svg: head + svg });
  document.getElementById(`${tabId}-pshp-sweep-title`).textContent =
    `${s.which}: Rest modes over the (${s.spec.axes.join(', ')}) shape — ${R0}×${R1}`;
}

function pshpSyncOperandField(tabId, which) {
  const spec = PABC_OPERAND[which] || PABC_OPERAND.A;
  const axesEl = document.getElementById(`${tabId}-pshp-shape-axes`);
  if (axesEl) axesEl.textContent = `(${spec.axes.join(', ')})`;
  const fnEl = document.getElementById(`${tabId}-pshp-shape-fn`);
  if (fnEl) fnEl.textContent = `partition_shape_${which}`;
  const group = document.getElementById(`${tabId}-pshp-operand-btns`);
  if (group) group.querySelectorAll('.mode-btn').forEach(b =>
    b.classList.toggle('active', b.textContent.trim() === `partition_shape_${which}`));
}

function setPshpOperand(tabId, which) {
  pshpState[tabId] = Object.assign(pshpState[tabId] || {}, { which });
  pshpSyncOperandField(tabId, which);
  renderPartitionShapeABC(tabId);
}

function setPshpMode(tabId, grid, mode) {
  const s = pshpState[tabId];
  if (!s) return;
  const key = grid === 'tile' ? 'tileMode' : 'mode';
  if (!(s[key] instanceof Set)) s[key] = new Set([grid === 'tile' ? 'value' : 'coord']);
  toggleLayoutVisualizationMode(s[key], mode);
  if (grid === 'tile') pshpRenderTileViz(tabId);
  else pshpRenderSweepViz(tabId);
}

function setPshpOp(tabId) {
  mmaSyncControls(tabId, document.getElementById(`${tabId}-pshp-op-input`).value, 'pshp');
  renderPartitionShapeABC(tabId);
}

/** Presets use the URL field order; `...tail` is the atom's Op-specific
 *  constructor tail (WGMMA: N, A source, A/B major, B dtype). */
function setPshp(tabId, which, opKey, ab, acc, k, atomLayout, perm, shape, ...tail) {
  pshpApplyKey(tabId, [which, opKey, ab || 'na', acc || 'na', String(k), atomLayout, perm || 'na', shape, ...tail]);
}

/** Deep link: operand, the six TiledMMA constructor fields, shape, then the
 *  atom's Op-specific tail. The constructor is validated before any control
 *  changes. */
function pshpApplyKey(tabId, inputs) {
  if (!PABC_OPERAND[inputs[0]]) throw new Error(`Unknown operand "${inputs[0]}".`);
  mtmWriteConstructorControls(tabId, 'pshp', mtmJoinConstructor(inputs.slice(1, 7), inputs.slice(8)));
  pshpState[tabId] = Object.assign(pshpState[tabId] || {}, { which: inputs[0] });
  document.getElementById(`${tabId}-pshp-shape-input`).value = inputs[7];
  renderPartitionShapeABC(tabId);
}

// Decode only the constructor; operand and shape stay as entered.
function pshpImportTiledMma(tabId, text) {
  const parsed = parsePastedKeyParam(text);
  if (!parsed || parsed.feature !== 'make_tiled_mma')
    throw new Error('Paste a make_tiled_mma URL or its query string (?key=...).');
  mtmWriteConstructorControls(tabId, 'pshp', parsed.inputs);
  renderPartitionShapeABC(tabId);
}

function pshpDecodePasteboard(tabId) {
  return decodeFromPasteboard({tabId, errorId: `${tabId}-pshp-error`, sources: ['make_tiled_mma'],
    message: 'Paste the make_tiled_mma URL or its query string:', importFn: text => pshpImportTiledMma(tabId, text)});
}

function exportPshp(tabId) {
  const { head, tail } = mtmSplitConstructor(mtmReadConstructorControls(tabId, 'pshp'));
  exportURL(`${tabId}-pshp-export`, 'partition_shape_abc',
    (pshpState[tabId] && pshpState[tabId].which) || 'A', ...head,
    document.getElementById(`${tabId}-pshp-shape-input`).value, ...tail);
}

/** Hand the returned SHAPE to make_fragment_A/B/C, which reads it column-major
 *  exactly as CuTe's make_fragment_X(shape) does. Re-renders first so an edited
 *  field never hands off a stale result. */
function pshpOpenFragment(tabId) {
  renderPartitionShapeABC(tabId);
  const s = pshpState[tabId];
  if (!s || !s.shape) return;
  if (s.descriptor) {
    showErr(`${tabId}-pshp-error`, `make_fragment_${s.which}(shape) is not defined for a WGMMA SMEM descriptor operand. ` +
      `Partition an SMEM tensor in partition_${s.which} and use its make_fragment button instead.`);
    return;
  }
  const c = s.constructor;
  setMfrag(tabId, s.which, ...c.slice(0, 4), s.shapeStr, 0, ...mtmSplitConstructor(c).tail);
  switchInnerTab(tabId, 'make_fragment_abc');
}
