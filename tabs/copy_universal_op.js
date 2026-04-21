// CopyUniversalOp tab: three collapsible parts for constructing and applying
// a CopyUniversalOp-based TiledCopy, with one visualization per part.
// Functions become globals on `window` (no module system).
//
// Pipeline (derived from include/cute/atom/copy_atom.hpp and copy_traits.hpp):
//   1. Copy_Atom<UniversalCopy<S>, ValType>:
//      - ThrID        = Layout<_1>
//      - ValLayoutSrc = ValLayoutDst = (1, N):(0, 1) with N = num_bits / sizeof_bits(ValType)
//        (Copy_Traits<UniversalCopy<S,D>> declares SrcLayout = Layout<Shape<_1,num_bits>>,
//         then Copy_Atom rescales it via recast_layout<uint1_t, ValType>.)
//   2. make_tiled_copy(atom, thr_layout, val_layout):
//      - layout_mn = raked_product(thr_layout, val_layout)    // (M,N) -> (thr,val)
//      - layout_tv = right_inverse(layout_mn).with_shape(...) // (tid,vid) -> (m,n)
//      - Tiler_MN  = product_each(shape(layout_mn))
//   3. partition_S/partition_D(tensor):
//      - zipped_divide(tensor, Tiler_MN) then tile2thrfrg(...)
//      - result per-thread shape ~ (FrgV, FrgX, RestM, RestN, ...)
//      - FrgV = AtomNumVal, FrgX = TiledNumVal/AtomNumVal, Rest* = tensor/tiler along each mode

// Element types we expose in the dropdown, mapped to their bit width.
const CUO_ELEMENT_BITS = {
  'int8_t': 8, 'uint8_t': 8,
  'half_t': 16, 'bfloat16_t': 16, 'int16_t': 16, 'uint16_t': 16,
  'float': 32, 'int32_t': 32, 'uint32_t': 32, 'tfloat32_t': 32,
  'double': 64, 'int64_t': 64, 'uint64_t': 64,
  'uint128_t': 128,
};

function generateCopyUniversalOpTabContent(id) {
  return `
    <!-- CopyUniversalOp panel -->
    <div id="${id}-tab-copy_universal_op" class="panel">
      <div class="controls">
        <h2>CopyUniversalOp</h2>

        <details class="cuo-section" open>
          <summary>1. Construct Copy_Atom &mdash; how one atom moves data</summary>
          <div class="cuo-section-body">
            <div class="form-group">
              <label>num_bits_per_copy</label>
              <input type="number" id="${id}-cuo-bits-input" value="128" min="1" step="1">
            </div>
            <div class="form-group">
              <label>element_type</label>
              <select id="${id}-cuo-etype-input">
                <option value="int8_t">int8_t (8)</option>
                <option value="uint8_t">uint8_t (8)</option>
                <option value="half_t" selected>half_t (16)</option>
                <option value="bfloat16_t">bfloat16_t (16)</option>
                <option value="int16_t">int16_t (16)</option>
                <option value="uint16_t">uint16_t (16)</option>
                <option value="float">float (32)</option>
                <option value="tfloat32_t">tfloat32_t (32)</option>
                <option value="int32_t">int32_t (32)</option>
                <option value="uint32_t">uint32_t (32)</option>
                <option value="double">double (64)</option>
                <option value="int64_t">int64_t (64)</option>
                <option value="uint64_t">uint64_t (64)</option>
                <option value="uint128_t">uint128_t (128)</option>
              </select>
            </div>
            <div id="${id}-cuo-atom-result" class="cuo-result"></div>
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>2. Construct TiledCopy &mdash; replicate the atom over a tile</summary>
          <div class="cuo-section-body">
            ${layoutInputField({ id: `${id}-cuo-thr-input`, label: 'thr_layout', value: '(4, 8):(8, 1)' })}
            ${layoutInputField({ id: `${id}-cuo-val-input`, label: 'val_layout', value: '(2, 8):(8, 1)' })}
            <div id="${id}-cuo-tile-result" class="cuo-result"></div>
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>3. Partition &mdash; apply the tile to a tensor</summary>
          <div class="cuo-section-body">
            <div class="form-group">
              <label>Direction</label>
              <div style="display:flex;gap:5px;margin-top:5px">
                <button class="btn active" id="${id}-cuo-dir-src" style="flex:1;font-size:0.75rem;padding:4px" onclick="setCuoDirection('${id}','src')">Src</button>
                <button class="btn" id="${id}-cuo-dir-dst" style="flex:1;font-size:0.75rem;padding:4px" onclick="setCuoDirection('${id}','dst')">Dst</button>
              </div>
            </div>
            ${layoutInputField({ id: `${id}-cuo-tensor-input`, label: 'Tensor layout (gmem / smem)', value: '(16, 64):(64, 1)' })}
            <div id="${id}-cuo-partition-result" class="cuo-result"></div>
          </div>
        </details>

        ${statusDivs(`${id}-cuo`)}
        <button class="btn btn-render" onclick="renderCopyUniversalOp('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-cuo-export" onclick="exportCUO('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setCUO('${id}',32,'half_t','(4,8):(8,1)','(2,2):(2,1)','src','(16,64):(64,1)')">32b half, 4x8 thr, 2x2 val, 16x64 tensor</button>
            <button class="preset-btn" onclick="setCUO('${id}',128,'half_t','(4,8):(8,1)','(2,8):(8,1)','src','(16,64):(64,1)')">128b half (ldg.128), 4x8 thr, 2x8 val</button>
            <button class="preset-btn" onclick="setCUO('${id}',32,'float','(2,2):(2,1)','(2,2):(2,1)','src','(8,8):(8,1)')">32b float, 2x2 thr, 2x2 val, 8x8 tensor</button>
            <button class="preset-btn" onclick="setCUO('${id}',64,'float','(2,4):(4,1)','(2,2):(2,1)','src','(8,16):(16,1)')">64b float (pair load), 2x4 thr, 2x2 val</button>
          </div>
        </div>

        <div class="hint">
          <code>CopyUniversalOp</code> is CuTe's most generic copy atom &mdash;
          a single thread issues a load/store of <code>num_bits_per_copy</code>
          bits in one go. After rescaling by <code>element_type</code> the atom's
          per-value layout is <code>(1, N):(0, 1)</code> where <code>N =
          num_bits / sizeof_bits(element_type)</code>.<br><br>
          <code>make_tiled_copy(atom, thr_layout, val_layout)</code> replicates
          the atom across a thread tile. The three sections above mirror the
          three steps of that pipeline; the three visualizations on the right
          show the artifact produced by each step.<br><br>
          <b>Pipeline (matches CuTe)</b>:
          <code>layout_mn = raked_product(thr, val)</code>,
          <code>Tiler_MN = product_each(shape(layout_mn))</code>,
          <code>layout_tv = right_inverse(layout_mn).with_shape(thr_size, val_size)</code>.
          Partition: <code>zipped_divide(tensor, Tiler_MN)</code> then compose
          the tile-local mode with <code>layout_tv</code>. (For
          <code>UniversalCopy</code>, <code>AtomLayoutRef == AtomLayoutSrc</code>,
          so the <code>right_inverse(Ref).compose(Src)</code> change-of-basis
          collapses to identity and is skipped.)<br><br>
          No coloring yet &mdash; just the raw layouts and the computed
          partition.
        </div>
      </div>

      <div class="comp-results" style="grid-template-columns:1fr">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-cuo-atom-title">1. Copy_Atom &mdash; single-atom data</span>
            <button class="mode-btn" id="${id}-cuo-atom-svg-zoom" onclick="toggleZoom('${id}-cuo-atom-svg')">Zoom in</button>
          </div>
          <div class="viz-box"><div id="${id}-cuo-atom-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-cuo-tile-title">2. TiledCopy TV layout</span>
            <button class="mode-btn" id="${id}-cuo-tile-svg-zoom" onclick="toggleZoom('${id}-cuo-tile-svg')">Zoom in</button>
          </div>
          <div class="viz-box"><div id="${id}-cuo-tile-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-cuo-tensor-title">3. Tensor layout</span>
            <button class="mode-btn" id="${id}-cuo-tensor-svg-zoom" onclick="toggleZoom('${id}-cuo-tensor-svg')">Zoom in</button>
          </div>
          <div class="viz-box"><div id="${id}-cuo-tensor-svg"></div></div>
        </div>
      </div>
    </div>`;
}

const cuoState = {};

function renderCopyUniversalOp(tabId) {
  showErr(`${tabId}-cuo-error`, '');
  try {
    // ─── Part 1: Copy_Atom ────────────────────────────────────────
    const bitsStr = document.getElementById(`${tabId}-cuo-bits-input`).value;
    const etype   = document.getElementById(`${tabId}-cuo-etype-input`).value;
    const numBits = parseInt(bitsStr, 10);
    if (!Number.isFinite(numBits) || numBits <= 0) {
      throw new Error(`num_bits_per_copy must be a positive integer, got "${bitsStr}"`);
    }
    const elemBits = CUO_ELEMENT_BITS[etype];
    if (!elemBits) throw new Error(`Unknown element_type "${etype}"`);
    if (numBits % elemBits !== 0) {
      throw new Error(
        `num_bits_per_copy (${numBits}) must be a multiple of sizeof_bits(${etype}) = ${elemBits}`);
    }
    const elements = numBits / elemBits;

    // Per-thread value layout of ONE atom. CuTe's Copy_Traits<UniversalCopy<S,D>>
    // declares `SrcLayout = Layout<Shape<_1, sizeof_bits<S>>>` with no explicit
    // stride, which compacts to col-major stride `(_1, _1)`. After
    // `recast_layout<uint1_t, ValType>` (upcast by sizeof_bits(ValType)), mode 0
    // stays `(1):(1)` and mode 1 becomes `(elements):(1)`. Net effect: the canonical
    // per-value atom layout is `(1, elements):(1, 1)` — which is what we produce
    // here by letting Layout's constructor auto-fill col-major strides.
    const atomSrc = new Layout([1, elements]);
    const atomDst = new Layout([1, elements]);
    const atomThrID = new Layout(1);           // Layout<_1>
    const atomStr = formatLayoutStr(atomSrc.shape, atomSrc.stride);

    // ─── Part 2: TiledCopy ────────────────────────────────────────
    const thrStr = document.getElementById(`${tabId}-cuo-thr-input`).value;
    const valStr = document.getElementById(`${tabId}-cuo-val-input`).value;
    updateRankWarning(`${tabId}-cuo-warning`, [
      ['thr_layout', thrStr], ['val_layout', valStr],
      ['Tensor layout', document.getElementById(`${tabId}-cuo-tensor-input`).value],
    ]);
    const thrP = parseLayout(thrStr);
    const valP = parseLayout(valStr);
    const thrSP = stripTrivialTrailing(thrP.shape, thrP.stride);
    const valSP = stripTrivialTrailing(valP.shape, valP.stride);
    const thrL = new Layout(thrSP.shape, thrSP.stride);
    const valL = new Layout(valSP.shape, valSP.stride);

    const atomNumVal = elements;
    const tiledNumVal = valL.size();
    if (tiledNumVal % atomNumVal !== 0) {
      throw new Error(
        `val_layout.size() (${tiledNumVal}) must be a multiple of the atom's ` +
        `num_val (${atomNumVal} = ${numBits}/${elemBits}). ` +
        `Each thread performs val_layout.size() / atom_num_val atom invocations.`);
    }
    const frgX = tiledNumVal / atomNumVal;  // atom invocations per thread per tile

    // layout_mn: (M, N) -> (thr_idx, val_idx)
    const layout_mn = raked_product(thrL, valL);
    const tiler_mn = product_each(layout_mn.shape);
    // layout_tv: (tid, vid) -> flat (m, n) in tiler_mn, reshaped to (thr_size, val_size)
    const tmp = new Layout([thrL.size(), valL.size()]);
    const layout_tv = composition(right_inverse(layout_mn), tmp);

    const layoutMNStr = formatLayoutStr(layout_mn.shape, layout_mn.stride);
    const layoutTVStr = formatLayoutStr(layout_tv.shape, layout_tv.stride);
    const tilerMNStr  = `(${tiler_mn.join(', ')})`;

    // ─── Part 3: Partition ────────────────────────────────────────
    const tensorStr = document.getElementById(`${tabId}-cuo-tensor-input`).value;
    const prev = cuoState[tabId] || {};
    const direction = prev.direction || 'src';
    const tensorP = parseLayout(tensorStr);
    const tensorSP = stripTrivialTrailing(tensorP.shape, tensorP.stride);
    const tensorL = new Layout(tensorSP.shape, tensorSP.stride);

    const [M_tile, N_tile] = tiler_mn;
    const [M_tensor, N_tensor] = productEach(tensorL.shape);
    if (M_tensor % M_tile !== 0 || N_tensor % N_tile !== 0) {
      throw new Error(
        `Tensor ${M_tensor}x${N_tensor} is not tileable by Tiler_MN ${M_tile}x${N_tile}. ` +
        `Each tensor dimension must be divisible by the corresponding tile dimension.`);
    }
    const restM = M_tensor / M_tile;
    const restN = N_tensor / N_tile;
    const perThreadShape = `(FrgV=${atomNumVal}, FrgX=${frgX}, RestM=${restM}, RestN=${restN})`;
    const perThreadCount = atomNumVal * frgX * restM * restN;

    // Exact partition_S/D per CuTe (copy_atom.hpp:221-226):
    //   zipped_divide(tensor, Tiler_MN)   -> ((M_tile, N_tile), (RestM, RestN))
    //   compose tile-local with layout_tv -> (Thr, Val) -> tensor offset
    //   combine with the rest mode        -> ((Thr, Val), (RestM, RestN))
    // For UniversalCopy, AtomLayoutRef == AtomLayoutSrc, so the `ref2trg`
    // change-of-basis `right_inverse(AtomLayoutRef).compose(AtomLayoutSrc)` is
    // the identity — we skip it rather than computing an equivalent no-op.
    let partitionFull = null, partitionPerThread = null, partitionErr = null;
    try {
      const tilerTuple = [new Layout(tiler_mn[0]), new Layout(tiler_mn[1])];
      const divided = zipped_divide(tensorL, tilerTuple);   // ((M_tile, N_tile), (RestM, RestN))
      const tileLocal = divided.mode(0);                    // (M_tile, N_tile) -> tensor offset
      const rest      = divided.mode(1);                    // (RestM, RestN)   -> tensor offset
      const tvTensor  = composition(tileLocal, layout_tv);  // (Thr, Val)       -> tensor offset
      partitionFull      = make_layout(tvTensor, rest);               // ((Thr, Val), (RestM, RestN))
      partitionPerThread = make_layout(tvTensor.mode(1), rest);       // (Val, (RestM, RestN))
    } catch (e) {
      partitionErr = e.message;
    }
    const partitionFullStr      = partitionFull      ? formatLayoutStr(partitionFull.shape, partitionFull.stride)           : null;
    const partitionPerThreadStr = partitionPerThread ? formatLayoutStr(partitionPerThread.shape, partitionPerThread.stride) : null;

    // ─── Stash state & render ─────────────────────────────────────
    cuoState[tabId] = {
      direction,
      numBits, etype, elements, elemBits,
      atomSrc, atomDst, atomThrID, atomStr,
      thrL, valL, layout_mn, layout_tv, tiler_mn,
      tensorL, tensorStr, restM, restN,
      atomNumVal, frgX, perThreadShape, perThreadCount,
    };

    document.getElementById(`${tabId}-cuo-atom-result`).innerHTML =
      `<div class="cuo-result-line"><b>Copy_Atom&lt;UniversalCopy&lt;${numBits}b&gt;, ${etype}&gt;</b></div>` +
      `<div class="cuo-result-line">elements_to_copy = ${numBits} / ${elemBits} = <b>${elements}</b></div>` +
      `<div class="cuo-result-line">ThrID        = ${formatLayoutStr(atomThrID.shape, atomThrID.stride)}</div>` +
      `<div class="cuo-result-line">ValLayoutSrc = ${atomStr}</div>` +
      `<div class="cuo-result-line">ValLayoutDst = ${atomStr}</div>`;

    document.getElementById(`${tabId}-cuo-tile-result`).innerHTML =
      `<div class="cuo-result-line">layout_mn = raked_product(thr, val) = <b>${layoutMNStr}</b></div>` +
      `<div class="cuo-result-line">Tiler_MN  = product_each(shape) = <b>${tilerMNStr}</b></div>` +
      `<div class="cuo-result-line">layout_tv = right_inverse(layout_mn) = <b>${layoutTVStr}</b></div>` +
      `<div class="cuo-result-line">TiledNumThr = ${thrL.size()}, TiledNumVal = ${tiledNumVal}, FrgX = ${frgX}</div>`;

    document.getElementById(`${tabId}-cuo-partition-result`).innerHTML =
      `<div class="cuo-result-line">Direction: <b>${direction.toUpperCase()}</b></div>` +
      `<div class="cuo-result-line">Tensor ${M_tensor}x${N_tensor} / Tiler ${M_tile}x${N_tile} = <b>${restM}x${restN}</b> tiles</div>` +
      `<div class="cuo-result-line">Per-thread shape: <b>${perThreadShape}</b></div>` +
      `<div class="cuo-result-line">Elements per thread: <b>${perThreadCount}</b> (= ${atomNumVal}*${frgX}*${restM}*${restN})</div>` +
      (partitionFull
        ? `<div class="cuo-result-line">Full partition ((Thr, Val), (RestM, RestN)) = <b>${partitionFullStr}</b></div>` +
          `<div class="cuo-result-line">Per-thread view (Val, (RestM, RestN)) = <b>${partitionPerThreadStr}</b></div>`
        : `<div class="cuo-result-line" style="color:#ef4444">Partition compute error: ${partitionErr}</div>`);

    renderCuoAtomViz(tabId);
    renderCuoTileViz(tabId);
    renderCuoTensorViz(tabId);

    // Sync direction buttons
    document.getElementById(`${tabId}-cuo-dir-src`).classList.toggle('active', direction === 'src');
    document.getElementById(`${tabId}-cuo-dir-dst`).classList.toggle('active', direction === 'dst');

    updateOuterTabLabel(tabId, `CopyUniversalOp:${numBits}b/${etype}`);
  } catch (e) {
    showErr(`${tabId}-cuo-error`, e.message);
    ['atom', 'tile', 'tensor'].forEach(w => {
      const el = document.getElementById(`${tabId}-cuo-${w}-svg`);
      if (el) el.innerHTML = '';
    });
  }
}

function renderCuoAtomViz(tabId) {
  const s = cuoState[tabId];
  if (!s) return;
  const host = document.getElementById(`${tabId}-cuo-atom-svg`);
  // For CopyUniversalOp, Src == Dst, so we only need one strip.
  const fmt = formatLayoutStr(s.atomSrc.shape, s.atomSrc.stride);
  const p = parseLayout(fmt);
  host.innerHTML =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `Val layout (src = dst) = ${fmt} &mdash; 1 thread, ${s.elements} ${s.etype} element${s.elements === 1 ? '' : 's'}` +
    `</div>` +
    buildLayoutSVG(p.shape, p.stride);
  applyZoomState(`${tabId}-cuo-atom-svg`);
  document.getElementById(`${tabId}-cuo-atom-title`).textContent =
    `1. Copy_Atom — ${s.numBits}b / ${s.etype} → ${s.elements} element${s.elements === 1 ? '' : 's'}`;
}

function renderCuoTileViz(tabId) {
  const s = cuoState[tabId];
  if (!s) return;
  const host = document.getElementById(`${tabId}-cuo-tile-svg`);
  const fmt = formatLayoutStr(s.layout_tv.shape, s.layout_tv.stride);
  const p = parseLayout(fmt);
  host.innerHTML =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `(tid, vid) → flat (m, n) in Tiler_MN = (${s.tiler_mn.join(',')}). Layout = ${fmt}` +
    `</div>` +
    buildLayoutSVG(p.shape, p.stride);
  applyZoomState(`${tabId}-cuo-tile-svg`);
  document.getElementById(`${tabId}-cuo-tile-title`).textContent =
    `2. TiledCopy TV layout — ${s.thrL.size()} threads × ${s.valL.size()} values`;
}

function renderCuoTensorViz(tabId) {
  const s = cuoState[tabId];
  if (!s) return;
  const host = document.getElementById(`${tabId}-cuo-tensor-svg`);
  const fmt = formatLayoutStr(s.tensorL.shape, s.tensorL.stride);
  const p = parseLayout(fmt);
  host.innerHTML =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `${s.direction.toUpperCase()} tensor = ${fmt} &mdash; ${s.restM}×${s.restN} tiles of size (${s.tiler_mn.join(',')})` +
    `</div>` +
    buildLayoutSVG(p.shape, p.stride);
  applyZoomState(`${tabId}-cuo-tensor-svg`);
  document.getElementById(`${tabId}-cuo-tensor-title`).textContent =
    `3. ${s.direction.toUpperCase()} tensor layout`;
}

function setCuoDirection(tabId, direction) {
  if (!cuoState[tabId]) cuoState[tabId] = {};
  cuoState[tabId].direction = direction;
  // If we haven't rendered yet, just flip the button state; otherwise re-render.
  if (cuoState[tabId].tensorL) {
    renderCopyUniversalOp(tabId);
  } else {
    document.getElementById(`${tabId}-cuo-dir-src`).classList.toggle('active', direction === 'src');
    document.getElementById(`${tabId}-cuo-dir-dst`).classList.toggle('active', direction === 'dst');
  }
}

function setCUO(tabId, bits, etype, thr, val, dir, tensor) {
  document.getElementById(`${tabId}-cuo-bits-input`).value = bits;
  document.getElementById(`${tabId}-cuo-etype-input`).value = etype;
  document.getElementById(`${tabId}-cuo-thr-input`).value = thr;
  document.getElementById(`${tabId}-cuo-val-input`).value = val;
  document.getElementById(`${tabId}-cuo-tensor-input`).value = tensor;
  if (!cuoState[tabId]) cuoState[tabId] = {};
  cuoState[tabId].direction = dir;
  renderCopyUniversalOp(tabId);
}

function exportCUO(tabId) {
  const bits = document.getElementById(`${tabId}-cuo-bits-input`).value;
  const etype = document.getElementById(`${tabId}-cuo-etype-input`).value;
  const thr = document.getElementById(`${tabId}-cuo-thr-input`).value;
  const val = document.getElementById(`${tabId}-cuo-val-input`).value;
  const tensor = document.getElementById(`${tabId}-cuo-tensor-input`).value;
  const dir = (cuoState[tabId] && cuoState[tabId].direction) || 'src';
  exportURL(`${tabId}-cuo-export`, 'copy_universal_op', bits, etype, thr, val, dir, tensor);
}
