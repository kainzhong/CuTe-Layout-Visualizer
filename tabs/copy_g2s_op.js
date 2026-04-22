// CopyG2SOp tab: same pipeline as CopyUniversalOp (their Copy_Traits are
// byte-for-byte identical — see include/cute/atom/copy_traits_sm80.hpp:41-54
// vs include/cute/atom/copy_traits.hpp:65-78). The only user-facing difference
// is cp.async's alignment constraint: num_bits_per_copy must be 32, 64, or 128.
// We reuse CopyUniversalOp's helpers (cuoThreadAtomColor, cuoGrayForTileIdx,
// cuoCellLabel, cuoBuildTileLookup) since they're pure functions of layout state.

// cp.async only accepts 4 B, 8 B, or 16 B loads — the dropdown enforces this.
const CG_BITS_OPTIONS = [32, 64, 128];

function generateCopyG2SOpTabContent(id) {
  return `
    <!-- CopyG2SOp panel -->
    <div id="${id}-tab-copy_g2s_op" class="panel">
      <div class="controls">
        <h2>cpasync.CopyG2SOp</h2>

        <details class="cuo-section" open>
          <summary>1. Construct Copy_Atom &mdash; one cp.async issue</summary>
          <div class="cuo-section-body">
            <div class="form-group">
              <label>num_bits_per_copy &mdash; cp.async only accepts 32/64/128</label>
              <select id="${id}-cg-bits-input">
                <option value="32">32 bits (4 B)</option>
                <option value="64">64 bits (8 B)</option>
                <option value="128" selected>128 bits (16 B)</option>
              </select>
            </div>
            <div class="form-group">
              <label>tensor_dtype</label>
              <select id="${id}-cg-dtype-input">
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
            <div id="${id}-cg-atom-result" class="cuo-result"></div>
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>2. Construct TiledCopy &mdash; replicate the atom over a tile</summary>
          <div class="cuo-section-body">
            ${layoutInputField({ id: `${id}-cg-thr-input`, label: 'thr_layout', value: '(4, 8):(8, 1)' })}
            ${layoutInputField({ id: `${id}-cg-val-input`, label: 'val_layout', value: '(2, 8):(8, 1)' })}
            <div id="${id}-cg-tile-result" class="cuo-result"></div>
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>3. Partition &mdash; apply the tile to a tensor</summary>
          <div class="cuo-section-body">
            <div class="form-group">
              <label>Direction</label>
              <div style="display:flex;gap:5px;margin-top:5px">
                <button class="btn active" id="${id}-cg-dir-src" style="flex:1;font-size:0.75rem;padding:4px" onclick="setCgDirection('${id}','src')">Src (gmem)</button>
                <button class="btn" id="${id}-cg-dir-dst" style="flex:1;font-size:0.75rem;padding:4px" onclick="setCgDirection('${id}','dst')">Dst (smem)</button>
              </div>
            </div>
            ${layoutInputField({ id: `${id}-cg-tensor-input`, label: 'Tensor layout (gmem or smem) &mdash; optional', hint: 'leave blank to skip the partition viz and just see atom + tile', value: '(16, 64):(64, 1)' })}
            <div id="${id}-cg-partition-result" class="cuo-result"></div>
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>4. Highlight thread &mdash; optional</summary>
          <div class="cuo-section-body">
            <div class="form-group">
              <label>Thread ID (empty = show all threads)</label>
              <input type="text" id="${id}-cg-highlight-tid" value="" placeholder="e.g. 5" oninput="setCgHighlight('${id}')">
            </div>
          </div>
        </details>

        ${statusDivs(`${id}-cg`)}
        <button class="btn btn-render" onclick="renderCopyG2SOp('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-cg-export" onclick="exportCG('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setCG('${id}',128,'half_t','(4,8):(8,1)','(2,8):(8,1)','src','(16,64):(64,1)')">128b half (ldg.128 async), 4x8 thr, 2x8 val</button>
            <button class="preset-btn" onclick="setCG('${id}',128,'float','(4,8):(8,1)','(2,4):(4,1)','src','(16,32):(32,1)')">128b float, 4x8 thr, 2x4 val</button>
            <button class="preset-btn" onclick="setCG('${id}',64,'half_t','(8,4):(4,1)','(2,4):(4,1)','src','(16,16):(16,1)')">64b half (ldg.64 async), 8x4 thr, 2x4 val</button>
            <button class="preset-btn" onclick="setCG('${id}',32,'float','(4,8):(8,1)','(2,2):(2,1)','src','(8,16):(16,1)')">32b float (1 elt/atom), 4x8 thr, 2x2 val</button>
            <button class="preset-btn" onclick="setCG('${id}',32,'half_t','(4,8):(8,1)','(2,2):(2,1)','src','(8,16):(16,1)')">32b half (2 elts/atom), 4x8 thr, 2x2 val</button>
          </div>
        </div>

        <div class="hint">
          <code>cpasync.CopyG2SOp</code> wraps the SM80+ <code>cp.async</code>
          PTX instruction, an asynchronous GMEM&nbsp;→&nbsp;SMEM copy. At the
          layout-algebra level its trait struct
          (<code>Copy_Traits&lt;SM80_CP_ASYNC_CACHEALWAYS&lt;S,D&gt;&gt;</code>)
          is <em>identical</em> to
          <code>Copy_Traits&lt;UniversalCopy&lt;S,D&gt;&gt;</code>:
          <code>ThrID = Layout&lt;_1&gt;</code>, <code>SrcLayout = DstLayout =
          Layout&lt;Shape&lt;_1, num_bits&gt;&gt;</code>. So the three
          visualizations below are computed by exactly the same pipeline as in
          the CopyUniversalOp tab.<br><br>
          What's different is the hardware contract: <code>cp.async</code> only
          issues 4 B, 8 B, or 16 B per thread, so <code>num_bits_per_copy</code>
          is locked to 32, 64, or 128. The copy is also <em>asynchronous</em>
          &mdash; the thread issues, moves on, and later synchronizes with
          <code>cp.commit_group</code> / <code>cp.wait_group</code>; none of
          that affects the layouts.
        </div>
      </div>

      <div class="comp-results" style="grid-template-columns:1fr">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-cg-atom-title">1. Copy_Atom &mdash; single-atom data</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-cg-atom-mode-btns">
                <button class="mode-btn active" onclick="setCgMode('${id}','atom','value')">value</button>
                <button class="mode-btn" onclick="setCgMode('${id}','atom','index')">index</button>
                <button class="mode-btn" onclick="setCgMode('${id}','atom','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-cg-atom-svg-zoom" onclick="toggleZoom('${id}-cg-atom-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-cg-atom-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-cg-tile-title">2. TiledCopy TV layout</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-cg-tile-mode-btns">
                <button class="mode-btn active" onclick="setCgMode('${id}','tile','value')">value</button>
                <button class="mode-btn" onclick="setCgMode('${id}','tile','index')">index</button>
                <button class="mode-btn" onclick="setCgMode('${id}','tile','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-cg-tile-svg-zoom" onclick="toggleZoom('${id}-cg-tile-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-cg-tile-svg"></div></div>
        </div>
        <div class="comp-viz-item" id="${id}-cg-tensor-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-cg-tensor-title">3. Tensor layout</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-cg-tensor-mode-btns">
                <button class="mode-btn active" onclick="setCgMode('${id}','tensor','value')">value</button>
                <button class="mode-btn" onclick="setCgMode('${id}','tensor','index')">index</button>
                <button class="mode-btn" onclick="setCgMode('${id}','tensor','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-cg-tensor-svg-zoom" onclick="toggleZoom('${id}-cg-tensor-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-cg-tensor-svg"></div></div>
        </div>
        <div class="comp-viz-item" id="${id}-cg-thread-partition-item" style="display:none">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-cg-thread-partition-title">4. Per-thread partition</span>
          </div>
          <div class="viz-box" style="background:#0b1220;color:#d1d5db;font-family:monospace;font-size:0.82rem;padding:14px 16px">
            <div id="${id}-cg-thread-partition-body"></div>
          </div>
        </div>
      </div>
    </div>`;
}

const cgState = {};

function renderCopyG2SOp(tabId) {
  showErr(`${tabId}-cg-error`, '');
  try {
    // ─── Part 1: Copy_Atom ────────────────────────────────────────
    const bitsStr = document.getElementById(`${tabId}-cg-bits-input`).value;
    const dtype   = document.getElementById(`${tabId}-cg-dtype-input`).value;
    const numBits = parseInt(bitsStr, 10);
    if (!CG_BITS_OPTIONS.includes(numBits)) {
      throw new Error(`num_bits_per_copy must be one of ${CG_BITS_OPTIONS.join('/')} for cp.async, got "${bitsStr}"`);
    }
    const elemBits = CUO_DTYPE_BITS[dtype];
    if (!elemBits) throw new Error(`Unknown tensor_dtype "${dtype}"`);
    if (numBits % elemBits !== 0) {
      throw new Error(
        `num_bits_per_copy (${numBits}) must be a multiple of sizeof_bits(${dtype}) = ${elemBits}`);
    }
    const elements = numBits / elemBits;

    // Same per-thread val layout as UniversalCopy's recast output:
    //   (1, elements):(1, 1) — one thread copies `elements` values.
    const atomSrc = new Layout([1, elements]);
    const atomDst = new Layout([1, elements]);
    const atomThrID = new Layout(1);
    const atomStr = formatLayoutStr(atomSrc.shape, atomSrc.stride);

    // ─── Part 2: TiledCopy ────────────────────────────────────────
    const thrStr = document.getElementById(`${tabId}-cg-thr-input`).value;
    const valStr = document.getElementById(`${tabId}-cg-val-input`).value;
    updateRankWarning(`${tabId}-cg-warning`, [
      ['thr_layout', thrStr], ['val_layout', valStr],
      ['Tensor layout', document.getElementById(`${tabId}-cg-tensor-input`).value],
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
    const frgX = tiledNumVal / atomNumVal;

    const layout_mn = raked_product(thrL, valL);
    const tiler_mn = product_each(layout_mn.shape);
    const tmp = new Layout([thrL.size(), valL.size()]);
    const layout_tv = composition(right_inverse(layout_mn), tmp);

    const layoutMNStr = formatLayoutStr(layout_mn.shape, layout_mn.stride);
    const layoutTVStr = formatLayoutStr(layout_tv.shape, layout_tv.stride);
    const tilerMNStr  = `(${tiler_mn.join(', ')})`;

    // ─── Part 3: Partition (optional) ─────────────────────────────
    const tensorStr = document.getElementById(`${tabId}-cg-tensor-input`).value;
    const prev = cgState[tabId] || {};
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

    // ─── Part 4: Highlight TID ────────────────────────────────────
    const highlightRaw = (document.getElementById(`${tabId}-cg-highlight-tid`).value || '').trim();
    let highlightTid = null;
    if (highlightRaw !== '') {
      const parsed = parseInt(highlightRaw, 10);
      if (Number.isFinite(parsed) && parsed >= 0 && parsed < thrL.size()) {
        highlightTid = parsed;
      } else {
        showWarn(`${tabId}-cg-warning`,
          `Highlight thread id "${highlightRaw}" is out of range [0, ${thrL.size()}) — showing all threads.`);
      }
    }

    cgState[tabId] = {
      direction,
      numBits, dtype, elements, elemBits,
      atomSrc, atomDst, atomThrID, atomStr,
      thrL, valL, layout_mn, layout_tv, tiler_mn,
      hasTensor, tensorL, tensorStr, restM, restN,
      atomNumVal, frgX, perThreadShape, perThreadCount,
      tvTensor, partitionFull, partitionPerThread, partitionFullStr, partitionPerThreadStr,
      highlightTid,
      atomMode:   (prev.atomMode   instanceof Set) ? prev.atomMode   : new Set(['value']),
      tileMode:   (prev.tileMode   instanceof Set) ? prev.tileMode   : new Set(['value']),
      tensorMode: (prev.tensorMode instanceof Set) ? prev.tensorMode : new Set(['value']),
    };

    document.getElementById(`${tabId}-cg-atom-result`).innerHTML =
      `<div class="cuo-result-line"><b>Copy_Atom&lt;SM80_CP_ASYNC_CACHEALWAYS&lt;${numBits}b&gt;, ${dtype}&gt;</b></div>` +
      `<div class="cuo-result-line">elements_to_copy = ${numBits} / ${elemBits} = <b>${elements}</b></div>` +
      `<div class="cuo-result-line">ThrID        = ${formatLayoutStr(atomThrID.shape, atomThrID.stride)}</div>` +
      `<div class="cuo-result-line">ValLayoutSrc = ${atomStr}</div>` +
      `<div class="cuo-result-line">ValLayoutDst = ${atomStr}</div>`;

    document.getElementById(`${tabId}-cg-tile-result`).innerHTML =
      `<div class="cuo-result-line">layout_mn = raked_product(thr, val) = <b>${layoutMNStr}</b></div>` +
      `<div class="cuo-result-line">Tiler_MN  = product_each(shape) = <b>${tilerMNStr}</b></div>` +
      `<div class="cuo-result-line">layout_tv = right_inverse(layout_mn) = <b>${layoutTVStr}</b></div>` +
      `<div class="cuo-result-line">TiledNumThr = ${thrL.size()}, TiledNumVal = ${tiledNumVal}, FrgX = ${frgX}</div>`;

    const tensorItem = document.getElementById(`${tabId}-cg-tensor-item`);
    if (hasTensor) {
      document.getElementById(`${tabId}-cg-partition-result`).innerHTML =
        `<div class="cuo-result-line">Direction: <b>${direction.toUpperCase()}</b> (${direction === 'src' ? 'gmem' : 'smem'})</div>` +
        `<div class="cuo-result-line">Tensor ${M_tensor}x${N_tensor} / Tiler ${M_tile}x${N_tile} = <b>${restM}x${restN}</b> tiles</div>` +
        `<div class="cuo-result-line">Per-thread shape: <b>${perThreadShape}</b></div>` +
        `<div class="cuo-result-line">Elements per thread: <b>${perThreadCount}</b> (= ${atomNumVal}*${frgX}*${restM}*${restN})</div>` +
        (partitionFull
          ? `<div class="cuo-result-line">Full partition ((Thr, Val), (RestM, RestN)) = <b>${partitionFullStr}</b></div>` +
            `<div class="cuo-result-line">Per-thread view (Val, (RestM, RestN)) = <b>${partitionPerThreadStr}</b></div>`
          : `<div class="cuo-result-line" style="color:#ef4444">Partition compute error: ${partitionErr}</div>`);
      if (tensorItem) tensorItem.style.display = '';
    } else {
      document.getElementById(`${tabId}-cg-partition-result`).innerHTML =
        `<div class="cuo-result-line" style="color:#9ca3af">Tensor layout is empty &mdash; partition not computed. Enter a layout above to see the tensor viz and the full partition.</div>`;
      if (tensorItem) tensorItem.style.display = 'none';
      const svg = document.getElementById(`${tabId}-cg-tensor-svg`);
      if (svg) svg.innerHTML = '';
    }

    renderCgAtomViz(tabId);
    renderCgTileViz(tabId);
    if (hasTensor) renderCgTensorViz(tabId);
    renderCgThreadPartition(tabId);

    document.getElementById(`${tabId}-cg-dir-src`).classList.toggle('active', direction === 'src');
    document.getElementById(`${tabId}-cg-dir-dst`).classList.toggle('active', direction === 'dst');

    updateOuterTabLabel(tabId, `CopyG2SOp:${numBits}b/${dtype}`);
  } catch (e) {
    showErr(`${tabId}-cg-error`, e.message);
    ['atom', 'tile', 'tensor'].forEach(w => {
      const el = document.getElementById(`${tabId}-cg-${w}-svg`);
      if (el) el.innerHTML = '';
    });
    const tpItem = document.getElementById(`${tabId}-cg-thread-partition-item`);
    if (tpItem) tpItem.style.display = 'none';
  }
}

function renderCgAtomViz(tabId) {
  const s = cgState[tabId];
  if (!s) return;
  const host = document.getElementById(`${tabId}-cg-atom-svg`);
  const fmt = formatLayoutStr(s.atomSrc.shape, s.atomSrc.stride);
  const p = parseLayout(fmt);
  const initColor = colorTV(0);
  const modes = s.atomMode;
  host.innerHTML =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `Val layout (src = dst) = ${fmt} &mdash; 1 thread, ${s.elements} ${s.dtype} element${s.elements === 1 ? '' : 's'} per cp.async issue` +
    `</div>` +
    buildColoredLayoutSVG(p.shape, p.stride, modes, (m, n, offset, flatI) => ({
      bg: initColor,
      text: cuoCellLabel(modes, [`T0`, `V${offset}`], flatI, m, n),
    }));
  applyZoomState(`${tabId}-cg-atom-svg`);
  updateModeBtns(`${tabId}-cg-atom-mode-btns`, modes);
  document.getElementById(`${tabId}-cg-atom-title`).textContent =
    `1. Copy_Atom — ${s.numBits}b cp.async / ${s.dtype} → ${s.elements} element${s.elements === 1 ? '' : 's'}`;
}

function renderCgTileViz(tabId) {
  const s = cgState[tabId];
  if (!s) return;
  const host = document.getElementById(`${tabId}-cg-tile-svg`);
  const { lookup, M_tile } = cuoBuildTileLookup(s);
  const tileShape = s.tiler_mn.slice();
  const tileStride = [1, M_tile];
  const modes = s.tileMode;
  const filterTid = s.highlightTid;
  host.innerHTML =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `Tile ${s.tiler_mn[0]}×${s.tiler_mn[1]} &mdash; same color = same thread; ` +
    `brightness = atom invocation (frgX=${s.frgX}${s.frgX > 1 ? ', darker for later atoms' : ''})` +
    (filterTid !== null ? ` &mdash; <b>filtered to T${filterTid}</b>` : '') +
    `</div>` +
    buildColoredLayoutSVG(tileShape, tileStride, modes, (m, n, _offset, flatI) => {
      const flat = m + n * M_tile;
      const e = lookup[flat];
      if (!e) return { bg: '#f0f0f0', text: null };
      const dimmed = filterTid !== null && e.tid !== filterTid;
      if (dimmed) {
        return { bg: '#e8e8e8', stroke: '#d1d5db', text: null };
      }
      const atomIdx = Math.floor(e.vid / s.atomNumVal);
      return {
        bg: cuoThreadAtomColor(e.tid, atomIdx, s.frgX),
        text: cuoCellLabel(modes, [`T${e.tid}`, `V${e.vid}`], flatI, m, n),
      };
    });
  applyZoomState(`${tabId}-cg-tile-svg`);
  updateModeBtns(`${tabId}-cg-tile-mode-btns`, modes);
  document.getElementById(`${tabId}-cg-tile-title`).textContent =
    `2. TiledCopy tile (${s.tiler_mn[0]}×${s.tiler_mn[1]}) — ${s.thrL.size()} threads × ${s.valL.size()} values (FrgV=${s.atomNumVal}, FrgX=${s.frgX})` +
    (filterTid !== null ? ` — filtered to T${filterTid}` : '');
}

function renderCgTensorViz(tabId) {
  const s = cgState[tabId];
  if (!s) return;
  const host = document.getElementById(`${tabId}-cg-tensor-svg`);
  const { lookup, M_tile } = cuoBuildTileLookup(s);
  const [, N_tile] = s.tiler_mn;
  const totalTiles = s.restM * s.restN;
  const filterTid = s.highlightTid;
  const modes = s.tensorMode;

  const fmt = formatLayoutStr(s.tensorL.shape, s.tensorL.stride);
  const p = parseLayout(fmt);
  const dirLabel = s.direction === 'src' ? 'GMEM (Src)' : 'SMEM (Dst)';
  const headerSuffix = filterTid !== null
    ? `T${filterTid}'s cells colored in every tile; non-T${filterTid} cells gray (brightness = tile index)`
    : `tile 0 (top-left) colored as in viz 2; other tiles gray (brightness = tile index)`;
  host.innerHTML =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `${dirLabel} tensor &mdash; ${headerSuffix}` +
    `</div>` +
    buildColoredLayoutSVG(p.shape, p.stride, modes, (m, n) => {
      const tm = Math.floor(m / M_tile);
      const tn = Math.floor(n / N_tile);
      const tileIdx = tm + tn * s.restM;
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
      return { bg };
    });
  applyZoomState(`${tabId}-cg-tensor-svg`);
  updateModeBtns(`${tabId}-cg-tensor-mode-btns`, modes);
  document.getElementById(`${tabId}-cg-tensor-title`).textContent =
    `3. ${dirLabel} tensor — ${s.restM}×${s.restN} tiles` +
    (filterTid !== null ? ` (T${filterTid} highlighted)` : ` (tile 0 highlighted)`);
}

function renderCgThreadPartition(tabId) {
  const s = cgState[tabId];
  const item = document.getElementById(`${tabId}-cg-thread-partition-item`);
  const body = document.getElementById(`${tabId}-cg-thread-partition-body`);
  if (!item || !body) return;
  if (!s || s.highlightTid === null || !s.hasTensor || !s.partitionPerThread || !s.tvTensor) {
    item.style.display = 'none';
    body.innerHTML = '';
    return;
  }
  const tid = s.highlightTid;
  let baseOffset = 'n/a';
  try { baseOffset = String(s.tvTensor.call(tid, 0)); } catch (_) { /* keep */ }
  item.style.display = '';
  document.getElementById(`${tabId}-cg-thread-partition-title`).textContent =
    `4. Per-thread partition — T${tid}`;
  body.innerHTML =
    `<div class="cuo-result-line">Thread: <b>T${tid}</b> of ${s.thrL.size()}</div>` +
    `<div class="cuo-result-line">Per-thread shape: <b>(FrgV=${s.atomNumVal}, FrgX=${s.frgX}, RestM=${s.restM}, RestN=${s.restN})</b></div>` +
    `<div class="cuo-result-line">Layout (Val, (RestM, RestN)) = <b>${s.partitionPerThreadStr}</b></div>` +
    `<div class="cuo-result-line">Base offset at tile 0 (<code>tvTensor(${tid}, 0)</code>): <b>${baseOffset}</b></div>` +
    `<div class="cuo-result-line">Total elements this thread moves: <b>${s.atomNumVal * s.frgX * s.restM * s.restN}</b></div>`;
}

function setCgDirection(tabId, direction) {
  if (!cgState[tabId]) cgState[tabId] = {};
  cgState[tabId].direction = direction;
  document.getElementById(`${tabId}-cg-dir-src`).classList.toggle('active', direction === 'src');
  document.getElementById(`${tabId}-cg-dir-dst`).classList.toggle('active', direction === 'dst');
  if (cgState[tabId].tensorL) renderCopyG2SOp(tabId);
}

function setCgMode(tabId, which, mode) {
  const s = cgState[tabId];
  if (!s) return;
  const key =
    which === 'atom'   ? 'atomMode'   :
    which === 'tile'   ? 'tileMode'   :
    which === 'tensor' ? 'tensorMode' : null;
  if (!key) return;
  let modes = s[key];
  if (!(modes instanceof Set)) { modes = new Set(['value']); s[key] = modes; }
  if (modes.has(mode)) {
    if (modes.size > 1) modes.delete(mode);
  } else {
    modes.add(mode);
  }
  if (which === 'atom')                        renderCgAtomViz(tabId);
  else if (which === 'tile')                   renderCgTileViz(tabId);
  else if (which === 'tensor' && s.hasTensor)  renderCgTensorViz(tabId);
}

function setCgHighlight(tabId) {
  const s = cgState[tabId];
  if (!s || !s.layout_tv) return;
  renderCopyG2SOp(tabId);
}

function setCG(tabId, bits, dtype, thr, val, dir, tensor) {
  document.getElementById(`${tabId}-cg-bits-input`).value = bits;
  document.getElementById(`${tabId}-cg-dtype-input`).value = dtype;
  document.getElementById(`${tabId}-cg-thr-input`).value = thr;
  document.getElementById(`${tabId}-cg-val-input`).value = val;
  document.getElementById(`${tabId}-cg-tensor-input`).value = tensor;
  if (!cgState[tabId]) cgState[tabId] = {};
  cgState[tabId].direction = dir;
  renderCopyG2SOp(tabId);
}

function exportCG(tabId) {
  const bits = document.getElementById(`${tabId}-cg-bits-input`).value;
  const dtype = document.getElementById(`${tabId}-cg-dtype-input`).value;
  const thr = document.getElementById(`${tabId}-cg-thr-input`).value;
  const val = document.getElementById(`${tabId}-cg-val-input`).value;
  const tensor = document.getElementById(`${tabId}-cg-tensor-input`).value;
  const dir = (cgState[tabId] && cgState[tabId].direction) || 'src';
  exportURL(`${tabId}-cg-export`, 'copy_g2s_op', bits, dtype, thr, val, dir, tensor);
}
