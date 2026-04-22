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
const CUO_DTYPE_BITS = {
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
        <h2>CopyUniversalOp / cpasync.CopyG2SOp</h2>

        <details class="cuo-section" open>
          <summary>1. Construct Copy_Atom &mdash; how one atom moves data</summary>
          <div class="cuo-section-body">
            <div class="form-group">
              <label>num_bits_per_copy</label>
              <input type="number" id="${id}-cuo-bits-input" value="128" min="1" step="1">
            </div>
            <div class="form-group">
              <label>tensor_dtype</label>
              <select id="${id}-cuo-dtype-input">
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
            ${layoutInputField({ id: `${id}-cuo-tensor-input`, label: 'Tensor layout (gmem / smem) &mdash; optional', hint: 'leave blank to skip the partition viz and just see atom + tile', value: '(16, 64):(64, 1)' })}
            <div id="${id}-cuo-partition-result" class="cuo-result"></div>
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>4. Highlight thread &mdash; optional</summary>
          <div class="cuo-section-body">
            <div class="form-group">
              <label>Thread ID (empty = show all threads)</label>
              <input type="text" id="${id}-cuo-highlight-tid" value="" placeholder="e.g. 5" oninput="setCuoHighlight('${id}')">
            </div>
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
          <b>One tab, two atoms.</b> <code>CopyUniversalOp</code> (CuTe's
          generic single-thread load/store) and <code>cpasync.CopyG2SOp</code>
          (SM80+ non-bulk <code>cp.async</code> GMEM→SMEM) have <em>byte-for-byte
          identical</em> <code>Copy_Traits</code>:
          <code>ThrID = Layout&lt;_1&gt;</code>,
          <code>SrcLayout = DstLayout = Layout&lt;Shape&lt;_1, num_bits&gt;&gt;</code>
          (see <code>include/cute/atom/copy_traits.hpp:65-78</code> and
          <code>copy_traits_sm80.hpp:41-54</code>). So the atom / tiled-copy /
          partition visualizations are the same for both &mdash; we drive them
          from a single tab. The only user-visible contract difference is
          that <code>cp.async</code> hardware only accepts
          <code>num_bits_per_copy &isin; {32, 64, 128}</code>; the asynchronous
          commit/wait semantics are runtime concerns invisible to the layout
          algebra.<br><br>
          After rescaling by <code>tensor_dtype</code> the atom's per-value
          layout is <code>(1, N):(1, 1)</code> where
          <code>N = num_bits / sizeof_bits(tensor_dtype)</code>.
          <code>make_tiled_copy(atom, thr_layout, val_layout)</code> then
          replicates that atom across a thread tile. The three sections above
          mirror the three steps of that pipeline.<br><br>
          <b>Pipeline (matches CuTe)</b>:
          <code>layout_mn = raked_product(thr, val)</code>,
          <code>Tiler_MN = product_each(shape(layout_mn))</code>,
          <code>layout_tv = right_inverse(layout_mn)<br>.with_shape(thr_size, val_size)</code>.
          Partition: <code>zipped_divide(tensor, Tiler_MN)</code> then compose
          the tile-local mode with <code>layout_tv</code>. (For both atoms,
          <code>AtomLayoutRef == AtomLayoutSrc</code>, so the
          <code>right_inverse(Ref).compose(Src)</code> change-of-basis
          collapses to identity and is skipped.)
        </div>
      </div>

      <div class="comp-results" style="grid-template-columns:1fr">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-cuo-atom-title">1. Copy_Atom &mdash; single-atom data</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-cuo-atom-mode-btns">
                <button class="mode-btn" onclick="setCuoMode('${id}','atom','value')">value</button>
              </span>
              <button class="mode-btn" id="${id}-cuo-atom-svg-zoom" onclick="toggleZoom('${id}-cuo-atom-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-cuo-atom-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-cuo-tile-title">2. TiledCopy TV layout</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-cuo-tile-mode-btns">
                <button class="mode-btn" onclick="setCuoMode('${id}','tile','value')">value</button>
              </span>
              <button class="mode-btn" id="${id}-cuo-tile-svg-zoom" onclick="toggleZoom('${id}-cuo-tile-svg')">Zoom in</button>
            </span>
          </div>
          <div class="cuo-viz-desc">
            The TV layout maps <code>(t, v)</code> to a <b>logical offset</b>
            (col-major flat position inside the tile).
          </div>
          <div class="viz-box"><div id="${id}-cuo-tile-svg"></div></div>
        </div>
        <div class="comp-viz-item" id="${id}-cuo-tensor-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-cuo-tensor-title">3. Tensor layout</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-cuo-tensor-mode-btns">
                <button class="mode-btn" onclick="setCuoMode('${id}','tensor','value')">value</button>
                <button class="mode-btn" onclick="setCuoMode('${id}','tensor','index')">index</button>
              </span>
              <button class="mode-btn" id="${id}-cuo-tensor-svg-zoom" onclick="toggleZoom('${id}-cuo-tensor-svg')">Zoom in</button>
            </span>
          </div>
          <div class="cuo-viz-desc">
            The src/dst tensor layout maps a <b>logical offset</b> to a
            <b>physical memory offset</b> &mdash; that's what the tensor's
            strides encode for any kernel written the standard way (no weird
            address aliasing or hand-crafted indexing tricks).
          </div>
          <div class="viz-box"><div id="${id}-cuo-tensor-svg"></div></div>
        </div>
        <div class="comp-viz-item" id="${id}-cuo-thread-partition-item" style="display:none">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-cuo-thread-partition-title">4. Per-thread partition</span>
          </div>
          <div class="viz-box" style="background:#0b1220;color:#d1d5db;font-family:monospace;font-size:0.82rem;padding:14px 16px">
            <div id="${id}-cuo-thread-partition-body"></div>
          </div>
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
    const dtype   = document.getElementById(`${tabId}-cuo-dtype-input`).value;
    const numBits = parseInt(bitsStr, 10);
    if (!Number.isFinite(numBits) || numBits <= 0) {
      throw new Error(`num_bits_per_copy must be a positive integer, got "${bitsStr}"`);
    }
    const elemBits = CUO_DTYPE_BITS[dtype];
    if (!elemBits) throw new Error(`Unknown tensor_dtype "${dtype}"`);
    if (numBits % elemBits !== 0) {
      throw new Error(
        `num_bits_per_copy (${numBits}) must be a multiple of sizeof_bits(${dtype}) = ${elemBits}`);
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

    // ─── Part 3: Partition (optional — blank tensor skips this section) ──
    const tensorStr = document.getElementById(`${tabId}-cuo-tensor-input`).value;
    const prev = cuoState[tabId] || {};
    const direction = prev.direction || 'src';
    const hasTensor = tensorStr.trim().length > 0;

    const [M_tile, N_tile] = tiler_mn;
    let tensorL = null;
    let M_tensor = 0, N_tensor = 0, restM = 0, restN = 0;
    let perThreadShape = null, perThreadCount = 0;
    let partitionFull = null, partitionPerThread = null, partitionErr = null;
    let partitionFullStr = null, partitionPerThreadStr = null;
    let tvTensor = null;

    if (hasTensor) {
      const tensorP = parseLayout(tensorStr);
      const tensorSP = stripTrivialTrailing(tensorP.shape, tensorP.stride);
      tensorL = new Layout(tensorSP.shape, tensorSP.stride);
      [M_tensor, N_tensor] = productEach(tensorL.shape);
      if (M_tensor % M_tile !== 0 || N_tensor % N_tile !== 0) {
        throw new Error(
          `Tensor ${M_tensor}x${N_tensor} is not tileable by Tiler_MN ${M_tile}x${N_tile}. ` +
          `Each tensor dimension must be divisible by the corresponding tile dimension.`);
      }
      restM = M_tensor / M_tile;
      restN = N_tensor / N_tile;
      perThreadShape = `(FrgV=${atomNumVal}, FrgX=${frgX}, RestM=${restM}, RestN=${restN})`;
      perThreadCount = atomNumVal * frgX * restM * restN;

      // Exact partition_S/D per CuTe (copy_atom.hpp:221-226):
      //   zipped_divide(tensor, Tiler_MN)   -> ((M_tile, N_tile), (RestM, RestN))
      //   compose tile-local with layout_tv -> (Thr, Val) -> tensor offset
      //   combine with the rest mode        -> ((Thr, Val), (RestM, RestN))
      // For UniversalCopy, AtomLayoutRef == AtomLayoutSrc, so the `ref2trg`
      // change-of-basis `right_inverse(AtomLayoutRef).compose(AtomLayoutSrc)` is
      // the identity — we skip it rather than computing an equivalent no-op.
      try {
        const tilerTuple = [new Layout(tiler_mn[0]), new Layout(tiler_mn[1])];
        const divided = zipped_divide(tensorL, tilerTuple);
        const tileLocal = divided.mode(0);
        const rest      = divided.mode(1);
        tvTensor        = composition(tileLocal, layout_tv);
        partitionFull      = make_layout(tvTensor, rest);
        partitionPerThread = make_layout(tvTensor.mode(1), rest);
      } catch (e) {
        partitionErr = e.message;
      }
      partitionFullStr      = partitionFull      ? formatLayoutStr(partitionFull.shape, partitionFull.stride)           : null;
      partitionPerThreadStr = partitionPerThread ? formatLayoutStr(partitionPerThread.shape, partitionPerThread.stride) : null;
    }

    // ─── Read highlight thread id (part 4) ────────────────────────
    const highlightRaw = (document.getElementById(`${tabId}-cuo-highlight-tid`).value || '').trim();
    let highlightTid = null;
    if (highlightRaw !== '') {
      const parsed = parseInt(highlightRaw, 10);
      if (Number.isFinite(parsed) && parsed >= 0 && parsed < thrL.size()) {
        highlightTid = parsed;
      } else {
        showWarn(`${tabId}-cuo-warning`,
          `Highlight thread id "${highlightRaw}" is out of range [0, ${thrL.size()}) — showing all threads.`);
      }
    }

    // ─── Stash state & render ─────────────────────────────────────
    cuoState[tabId] = {
      direction,
      numBits, dtype, elements, elemBits,
      atomSrc, atomDst, atomThrID, atomStr,
      thrL, valL, layout_mn, layout_tv, tiler_mn,
      hasTensor, tensorL, tensorStr, restM, restN,
      atomNumVal, frgX, perThreadShape, perThreadCount,
      tvTensor, partitionFull, partitionPerThread, partitionFullStr, partitionPerThreadStr,
      highlightTid,
      // Picker state as Sets so zero or more modes can be active per viz.
      // Default is empty (unselected) — the viz shows T/V labels out of the
      // box, and the user opts in to the numeric overlays.
      atomMode:   (prev.atomMode   instanceof Set) ? prev.atomMode   : new Set(),
      tileMode:   (prev.tileMode   instanceof Set) ? prev.tileMode   : new Set(),
      tensorMode: (prev.tensorMode instanceof Set) ? prev.tensorMode : new Set(),
    };

    document.getElementById(`${tabId}-cuo-atom-result`).innerHTML =
      `<div class="cuo-result-line"><b>Copy_Atom&lt;UniversalCopy&lt;${numBits}b&gt;, ${dtype}&gt;</b></div>` +
      `<div class="cuo-result-line">elements_to_copy = ${numBits} / ${elemBits} = <b>${elements}</b></div>` +
      `<div class="cuo-result-line">ThrID        = ${formatLayoutStr(atomThrID.shape, atomThrID.stride)}</div>` +
      `<div class="cuo-result-line">ValLayoutSrc = ${atomStr}</div>` +
      `<div class="cuo-result-line">ValLayoutDst = ${atomStr}</div>`;

    document.getElementById(`${tabId}-cuo-tile-result`).innerHTML =
      `<div class="cuo-result-line">layout_mn = raked_product(thr, val) = <b>${layoutMNStr}</b></div>` +
      `<div class="cuo-result-line">Tiler_MN  = product_each(shape) = <b>${tilerMNStr}</b></div>` +
      `<div class="cuo-result-line">layout_tv = right_inverse(layout_mn) = <b>${layoutTVStr}</b></div>` +
      `<div class="cuo-result-line">TiledNumThr = ${thrL.size()}, TiledNumVal = ${tiledNumVal}, FrgX = ${frgX}</div>`;

    const tensorItem = document.getElementById(`${tabId}-cuo-tensor-item`);
    if (hasTensor) {
      document.getElementById(`${tabId}-cuo-partition-result`).innerHTML =
        `<div class="cuo-result-line">Direction: <b>${direction.toUpperCase()}</b></div>` +
        `<div class="cuo-result-line">Tensor ${M_tensor}x${N_tensor} / Tiler ${M_tile}x${N_tile} = <b>${restM}x${restN}</b> tiles</div>` +
        `<div class="cuo-result-line">Per-thread shape: <b>${perThreadShape}</b></div>` +
        `<div class="cuo-result-line">Elements per thread: <b>${perThreadCount}</b> (= ${atomNumVal}*${frgX}*${restM}*${restN})</div>` +
        (partitionFull
          ? `<div class="cuo-result-line">Full partition ((Thr, Val), (RestM, RestN)) = <b>${partitionFullStr}</b></div>` +
            `<div class="cuo-result-line">Per-thread view (Val, (RestM, RestN)) = <b>${partitionPerThreadStr}</b></div>`
          : `<div class="cuo-result-line" style="color:#ef4444">Partition compute error: ${partitionErr}</div>`);
      if (tensorItem) tensorItem.style.display = '';
    } else {
      document.getElementById(`${tabId}-cuo-partition-result`).innerHTML =
        `<div class="cuo-result-line" style="color:#9ca3af">Tensor layout is empty &mdash; partition not computed. Enter a layout above to see the tensor viz and the full partition.</div>`;
      if (tensorItem) tensorItem.style.display = 'none';
      const svg = document.getElementById(`${tabId}-cuo-tensor-svg`);
      if (svg) svg.innerHTML = '';
    }

    renderCuoAtomViz(tabId);
    renderCuoTileViz(tabId);
    if (hasTensor) renderCuoTensorViz(tabId);
    renderCuoThreadPartition(tabId);

    // Sync direction buttons
    document.getElementById(`${tabId}-cuo-dir-src`).classList.toggle('active', direction === 'src');
    document.getElementById(`${tabId}-cuo-dir-dst`).classList.toggle('active', direction === 'dst');

    updateOuterTabLabel(tabId, `CopyUniversalOp:${numBits}b/${dtype}`);
  } catch (e) {
    showErr(`${tabId}-cuo-error`, e.message);
    ['atom', 'tile', 'tensor'].forEach(w => {
      const el = document.getElementById(`${tabId}-cuo-${w}-svg`);
      if (el) el.innerHTML = '';
    });
    const tpItem = document.getElementById(`${tabId}-cuo-thread-partition-item`);
    if (tpItem) tpItem.style.display = 'none';
  }
}

// ─── Color helpers shared across the three vizzes ───────────────────────
// Thread 0 gets the "initial" color; other threads cycle through TV_COLORS.
// Within a thread, successive atom invocations darken the base color so the
// first atom is the lightest and later ones are progressively more saturated.
function cuoThreadAtomColor(tid, atomIdx, totalAtoms) {
  const base = colorTV(tid);
  if (totalAtoms <= 1 || atomIdx === 0) return base;
  const factor = (atomIdx / (totalAtoms - 1)) * 0.45;  // 0 → 0.45
  return darkenRGB(base, factor);
}

// Distinct gray shades per tile index. Handles tileIdx = 0 (used when we want
// to gray out the first tile too, e.g. the non-highlighted cells of tile 0
// when filtering by thread). Lightest at tile 0, darker as tileIdx grows.
function cuoGrayForTileIdx(tileIdx, totalTiles) {
  if (totalTiles <= 1) return grayRGB(220);
  const frac = tileIdx / (totalTiles - 1);   // 0 → 1
  return grayRGB(Math.round(230 - frac * 110));  // 230 → 120
}

// Build a lookup table: flat (m, n) in the tile  →  { tid, vid }, using
// layout_tv's (tid, vid) → flat-mn mapping. Each cell in the tile is reached
// exactly once for a bijective UniversalCopy TV layout.
function cuoBuildTileLookup(s) {
  const [M_tile] = s.tiler_mn;
  const thr_size = s.thrL.size();
  const val_size = s.valL.size();
  const lookup = new Array(s.tiler_mn[0] * s.tiler_mn[1]).fill(null);
  for (let tid = 0; tid < thr_size; tid++) {
    for (let vid = 0; vid < val_size; vid++) {
      const flat = s.layout_tv.call(tid, vid);
      if (flat >= 0 && flat < lookup.length) lookup[flat] = { tid, vid };
    }
  }
  return { lookup, M_tile };
}

function renderCuoAtomViz(tabId) {
  const s = cuoState[tabId];
  if (!s) return;
  const host = document.getElementById(`${tabId}-cuo-atom-svg`);
  const fmt = formatLayoutStr(s.atomSrc.shape, s.atomSrc.stride);
  const p = parseLayout(fmt);
  // Single atom, single thread: every cell gets the "initial" color
  // (= thread 0's base color, reused by the first atom of thread 0 in viz 2).
  const initColor = colorTV(0);
  const modes = s.atomMode instanceof Set ? s.atomMode : new Set();
  host.innerHTML =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `Val layout (src = dst) = ${fmt} &mdash; 1 thread, ${s.elements} ${s.dtype} element${s.elements === 1 ? '' : 's'}` +
    `</div>` +
    buildColoredLayoutSVG(p.shape, p.stride, 'value', (m, n, offset) => {
      // T/V always visible; 'value' picker adds the atom layout's output (= v).
      const lines = [`T0`, `V${offset}`];
      if (modes.has('value')) lines.push(String(offset));
      return { bg: initColor, text: lines };
    });
  applyZoomState(`${tabId}-cuo-atom-svg`);
  updateModeBtns(`${tabId}-cuo-atom-mode-btns`, modes);
  document.getElementById(`${tabId}-cuo-atom-title`).textContent =
    `1. Copy_Atom — ${s.numBits}b / ${s.dtype} → ${s.elements} element${s.elements === 1 ? '' : 's'}`;
}

function renderCuoTileViz(tabId) {
  const s = cuoState[tabId];
  if (!s) return;
  const host = document.getElementById(`${tabId}-cuo-tile-svg`);
  const { lookup, M_tile } = cuoBuildTileLookup(s);
  const tileShape = s.tiler_mn.slice();
  const tileStride = [1, M_tile];  // col-major over Tiler_MN
  const filterTid = s.highlightTid;  // may be null
  const modes = s.tileMode instanceof Set ? s.tileMode : new Set();
  host.innerHTML =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `Tile ${s.tiler_mn[0]}×${s.tiler_mn[1]} &mdash; same color = same thread; ` +
    `brightness = atom invocation (frgX=${s.frgX}${s.frgX > 1 ? ', darker for later atoms' : ''})` +
    (filterTid !== null ? ` &mdash; <b>filtered to T${filterTid}</b>` : '') +
    `</div>` +
    buildColoredLayoutSVG(tileShape, tileStride, 'value', (m, n, offset) => {
      const flat = m + n * M_tile;
      const e = lookup[flat];
      if (!e) return { bg: '#f0f0f0', text: null };
      const dimmed = filterTid !== null && e.tid !== filterTid;
      if (dimmed) {
        return { bg: '#e8e8e8', stroke: '#d1d5db', text: null };
      }
      const atomIdx = Math.floor(e.vid / s.atomNumVal);
      // T/V always visible; 'value' picker adds the tile layout's output at
      // (m, n) = col-major flat = the TV layout's output at this cell.
      const lines = [`T${e.tid}`, `V${e.vid}`];
      if (modes.has('value')) lines.push(String(offset));
      return {
        bg: cuoThreadAtomColor(e.tid, atomIdx, s.frgX),
        text: lines,
      };
    });
  applyZoomState(`${tabId}-cuo-tile-svg`);
  updateModeBtns(`${tabId}-cuo-tile-mode-btns`, modes);
  document.getElementById(`${tabId}-cuo-tile-title`).textContent =
    `2. TiledCopy tile (${s.tiler_mn[0]}×${s.tiler_mn[1]}) — ${s.thrL.size()} threads × ${s.valL.size()} values (FrgV=${s.atomNumVal}, FrgX=${s.frgX})` +
    (filterTid !== null ? ` — filtered to T${filterTid}` : '');
}

function renderCuoTensorViz(tabId) {
  const s = cuoState[tabId];
  if (!s) return;
  const host = document.getElementById(`${tabId}-cuo-tensor-svg`);
  const { lookup, M_tile } = cuoBuildTileLookup(s);
  const [, N_tile] = s.tiler_mn;
  const totalTiles = s.restM * s.restN;
  const filterTid = s.highlightTid;  // may be null
  const modes = s.tensorMode instanceof Set ? s.tensorMode : new Set();

  const fmt = formatLayoutStr(s.tensorL.shape, s.tensorL.stride);
  const p = parseLayout(fmt);
  const headerSuffix = filterTid !== null
    ? `T${filterTid}'s cells colored in every tile; non-T${filterTid} cells gray (brightness = tile index)`
    : `tile 0 (top-left) colored as in viz 2; other tiles gray (brightness = tile index)`;
  host.innerHTML =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `${s.direction.toUpperCase()} tensor &mdash; ${headerSuffix}` +
    `</div>` +
    buildColoredLayoutSVG(p.shape, p.stride, 'value', (m, n, offset, flatI) => {
      const tm = Math.floor(m / M_tile);
      const tn = Math.floor(n / N_tile);
      const tileIdx = tm + tn * s.restM;  // col-major over tiles
      const flat = (m % M_tile) + (n % N_tile) * M_tile;
      const e = lookup[flat];
      let bg;
      if (filterTid === null) {
        if (tileIdx === 0 && e) {
          const atomIdx = Math.floor(e.vid / s.atomNumVal);
          bg = cuoThreadAtomColor(e.tid, atomIdx, s.frgX);
        } else {
          bg = cuoGrayForTileIdx(tileIdx, totalTiles);
        }
      } else if (e && e.tid === filterTid) {
        const atomIdx = Math.floor(e.vid / s.atomNumVal);
        bg = cuoThreadAtomColor(e.tid, atomIdx, s.frgX);
      } else {
        bg = cuoGrayForTileIdx(tileIdx, totalTiles);
      }
      // T/V always visible for every cell with a mapping (bijective TV → every
      // tensor cell has one). 'value' = tensor layout output (physical offset),
      // 'index' = col-major input (m + n*M_tensor). Prefix when both are on.
      const lines = e ? [`T${e.tid}`, `V${e.vid}`] : [];
      if (modes.has('value') && modes.has('index')) {
        lines.push(`val=${offset}`);
        lines.push(`idx=${flatI}`);
      } else if (modes.has('value')) {
        lines.push(String(offset));
      } else if (modes.has('index')) {
        lines.push(String(flatI));
      }
      return { bg, text: lines.length ? lines : null };
    });
  applyZoomState(`${tabId}-cuo-tensor-svg`);
  updateModeBtns(`${tabId}-cuo-tensor-mode-btns`, modes);
  document.getElementById(`${tabId}-cuo-tensor-title`).textContent =
    `3. ${s.direction.toUpperCase()} tensor — ${s.restM}×${s.restN} tiles` +
    (filterTid !== null ? ` (T${filterTid} highlighted)` : ` (tile 0 highlighted)`);
}

// Panel 4 (below viz 3): per-thread partition text block, shown only when a
// valid highlight thread id is set AND we have a tensor. Derives the thread's
// base offset at tile 0 from `tvTensor.call(tid, 0)` (= the tensor offset
// where that thread's vid=0 value lands in the first tile).
function renderCuoThreadPartition(tabId) {
  const s = cuoState[tabId];
  const item = document.getElementById(`${tabId}-cuo-thread-partition-item`);
  const body = document.getElementById(`${tabId}-cuo-thread-partition-body`);
  if (!item || !body) return;
  if (!s || s.highlightTid === null || !s.hasTensor || !s.partitionPerThread || !s.tvTensor) {
    item.style.display = 'none';
    body.innerHTML = '';
    return;
  }
  const tid = s.highlightTid;
  let baseOffset = 'n/a';
  try {
    baseOffset = String(s.tvTensor.call(tid, 0));
  } catch (_) { /* keep 'n/a' */ }
  item.style.display = '';
  document.getElementById(`${tabId}-cuo-thread-partition-title`).textContent =
    `4. Per-thread partition — T${tid}`;
  body.innerHTML =
    `<div class="cuo-result-line">Thread: <b>T${tid}</b> of ${s.thrL.size()}</div>` +
    `<div class="cuo-result-line">Per-thread shape: <b>(FrgV=${s.atomNumVal}, FrgX=${s.frgX}, RestM=${s.restM}, RestN=${s.restN})</b></div>` +
    `<div class="cuo-result-line">Layout (Val, (RestM, RestN)) = <b>${s.partitionPerThreadStr}</b></div>` +
    `<div class="cuo-result-line">Base offset at tile 0 (<code>tvTensor(${tid}, 0)</code>): <b>${baseOffset}</b></div>` +
    `<div class="cuo-result-line">Total elements this thread moves: <b>${s.atomNumVal * s.frgX * s.restM * s.restN}</b></div>`;
}

function setCuoDirection(tabId, direction) {
  if (!cuoState[tabId]) cuoState[tabId] = {};
  cuoState[tabId].direction = direction;
  document.getElementById(`${tabId}-cuo-dir-src`).classList.toggle('active', direction === 'src');
  document.getElementById(`${tabId}-cuo-dir-dst`).classList.toggle('active', direction === 'dst');
  // Only re-render if we have state to paint; without tensor the direction is
  // mostly a label, nothing else changes visually.
  if (cuoState[tabId].tensorL) {
    renderCopyUniversalOp(tabId);
  }
}

// Live re-render when the highlight-thread input changes. Without an existing
// render (no `layout_tv`) there's nothing to filter yet — skip and wait for
// the user to click Render.
function setCuoHighlight(tabId) {
  const s = cuoState[tabId];
  if (!s || !s.layout_tv) return;
  renderCopyUniversalOp(tabId);
}

// Independent toggle per mode per viz. Sets can be empty (user may turn all
// labels off). For the atom and tile vizzes the only option is 'value' (the
// layout's output, == col-major flat position of the cell). For the tensor
// viz both 'value' (tensor layout's physical offset) and 'index' (col-major
// logical offset = input to the tensor layout) are independently toggleable.
function setCuoMode(tabId, which, mode) {
  const s = cuoState[tabId];
  if (!s) return;
  const key =
    which === 'atom'   ? 'atomMode'   :
    which === 'tile'   ? 'tileMode'   :
    which === 'tensor' ? 'tensorMode' : null;
  if (!key) return;
  let modes = s[key];
  if (!(modes instanceof Set)) { modes = new Set(); s[key] = modes; }
  if (modes.has(mode)) modes.delete(mode); else modes.add(mode);
  if (which === 'atom')                       renderCuoAtomViz(tabId);
  else if (which === 'tile')                  renderCuoTileViz(tabId);
  else if (which === 'tensor' && s.hasTensor) renderCuoTensorViz(tabId);
}

function setCUO(tabId, bits, dtype, thr, val, dir, tensor) {
  document.getElementById(`${tabId}-cuo-bits-input`).value = bits;
  document.getElementById(`${tabId}-cuo-dtype-input`).value = dtype;
  document.getElementById(`${tabId}-cuo-thr-input`).value = thr;
  document.getElementById(`${tabId}-cuo-val-input`).value = val;
  document.getElementById(`${tabId}-cuo-tensor-input`).value = tensor;
  if (!cuoState[tabId]) cuoState[tabId] = {};
  cuoState[tabId].direction = dir;
  renderCopyUniversalOp(tabId);
}

function exportCUO(tabId) {
  const bits = document.getElementById(`${tabId}-cuo-bits-input`).value;
  const dtype = document.getElementById(`${tabId}-cuo-dtype-input`).value;
  const thr = document.getElementById(`${tabId}-cuo-thr-input`).value;
  const val = document.getElementById(`${tabId}-cuo-val-input`).value;
  const tensor = document.getElementById(`${tabId}-cuo-tensor-input`).value;
  const dir = (cuoState[tabId] && cuoState[tabId].direction) || 'src';
  exportURL(`${tabId}-cuo-export`, 'copy_universal_op', bits, dtype, thr, val, dir, tensor);
}
