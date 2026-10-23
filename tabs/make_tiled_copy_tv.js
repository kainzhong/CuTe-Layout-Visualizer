// make_tiled_copy_tv tab (Copy scope): the derived constructor —
// `make_tiled_copy_tv(atom, thr_layout, val_layout)`. It computes
// (layout_tv, tiler_mn) from a thr/val pair and hands them to
// `make_tiled_copy`, which is the other tab.
//
// Shared machinery (atom section, tile lookup, coloring, coverage and
// vectorization checks, viz renderers) lives in tabs/make_tiled_copy.js.
// Functions become globals on `window` (no module system).
//
// Pipeline — python/CuTeDSL/cutlass/cute/core.py :: make_layout_tv, a
// line-for-line port of the C++ `make_tiled_copy` helper:
//   layout_mn = raked_product(thr_layout, val_layout)          // (M,N) -> (thr,val)
//   Tiler_MN  = product_each(shape(layout_mn))
//   layout_tv = right_inverse(layout_mn)
//                 .with_shape(size(thr_layout), size(val_layout))  // (tid,vid) -> (m,n)
//
// The docstring says "The thread and value layouts must be compact" but nothing
// enforces it: `right_inverse` of a non-injective layout silently returns a
// PARTIAL inverse, so a non-compact thr/val pair yields a TiledCopy that
// overlaps itself and only blows up much later, at `cute.copy`.

function generateMakeTiledCopyTvTabContent(id) {
  return `
    <!-- make_tiled_copy_tv panel -->
    <div id="${id}-tab-make_tiled_copy_tv" class="panel">
      <div class="controls">
        <h2>make_tiled_copy_tv</h2>
${mtcAtomSection(id, 'mtv', '1. CopyAtom')}

        <details class="cuo-section" open>
          <summary>2. Derive layout_tv and Tiler_MN from thr/val</summary>
          <div class="cuo-section-body">
            <button class="btn btn-render btn-decode" id="${id}-mtv-decode-tv" onclick="mtvDecodeTVPasteboard('${id}')">Decode from Pasteboard</button>
            <details class="cuo-section" id="${id}-mtv-tv-manual-construction"><summary>Manual construction</summary><div class="cuo-section-body">
            ${layoutInputField({ id: `${id}-mtv-thr-input`, label: 'thr_layout &mdash; (M, N) &rarr; thread id', value: '(16, 8):(8, 1)' })}
            ${layoutInputField({ id: `${id}-mtv-val-input`, label: 'val_layout &mdash; (M, N) &rarr; value id, per thread', value: '(1, 8):(1, 1)' })}
            </div></details>
            <div id="${id}-mtv-tile-result" class="cuo-result"></div>
          </div>
        </details>

        <div class="form-group">
          <label>Highlight thread (empty = show all threads)</label>
          <input type="text" id="${id}-mtv-highlight-tid" value="" placeholder="e.g. 5" oninput="setMtvHighlight('${id}')">
        </div>

        ${statusDivs(`${id}-mtv`)}
        <button class="btn btn-render" onclick="renderMakeTiledCopyTv('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-mtv-export" onclick="exportMTV('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setMTV('${id}','cpasync',128,'half_t','(16,8):(8,1)','(1,8):(1,1)')">sgemm_sm80.cu &mdash; 128b half, thr 16x8 k-major, val 1x8</button>
            <button class="preset-btn" onclick="setMTV('${id}','universal',128,'float','(32,8):(1,32)','(4,1):(1,1)')">sgemm_2.cu &mdash; 128b float, thr 32x8 m-major, val 4x1</button>
            <button class="preset-btn" onclick="setMTV('${id}','universal',16,'half_t','(4,8):(8,1)','(2,4):(4,1)')">scalar atom &mdash; 16b half, val 2x4 (FrgX=8)</button>
            <button class="preset-btn" onclick="setMTV('${id}','universal',128,'half_t','(4,8):(8,1)','(2,4):(4,1)')">128b half, val 2x4 &mdash; shows why a 2-D val_layout can't vectorize</button>
            <button class="preset-btn" onclick="setMTV('${id}','g2r',128,'half_t','(16,8):(8,1)','(1,8):(1,1)')">CopyG2ROp &mdash; same derivation, GMEM&rarr;RMEM</button>
            <button class="preset-btn" onclick="setMTV('${id}','s2r',128,'half_t','(16,8):(8,1)','(1,8):(1,1)')">CopyS2ROp &mdash; the SIMT SMEM load (not ldmatrix)</button>
          </div>
        </div>

        <div class="hint">
          <b>What this tab computes.</b>
          <code>layout_mn = raked_product(thr, val)</code>,
          <code>Tiler_MN = product_each(shape(layout_mn))</code>,
          <code>layout_tv = right_inverse(layout_mn).with_shape(thr_size, val_size)</code>,
          then <code>make_tiled_copy(atom, layout_tv, Tiler_MN)</code>. The two
          derived values are printed below &mdash; paste them into the
          <b>make_tiled_copy</b> tab to poke at them directly.<br><br>
          <b>The val_layout convention.</b> Production code gives
          <code>val_layout</code> exactly one non-trivial mode, sized
          <code>num_bits / sizeof_bits(dtype)</code>, on the contiguous axis
          &mdash; <code>(1, N)</code> for a K-major tensor,
          <code>(N, 1)</code> for an M-major one (see
          <code>examples/cute/tutorial/sgemm_sm80.cu:368</code> and
          <code>tensorop_gemm.py:_make_gmem_tiled_copy_AB</code>). A genuine 2-D
          <code>val_layout</code> is only valid for a <em>scalar</em> atom, where
          each thread issues <code>FrgX</code> separate instructions.<br><br>
          <b>Strides encode ordering, not gaps.</b> Both layouts must be
          <em>compact</em> (<code>cosize == size</code>). CuTe does not check
          this &mdash; a non-compact layout makes
          <code>right_inverse(layout_mn)</code> a partial inverse, so threads
          silently overlap and the copy is wrong. This tab checks it for you.<br><br>
          <b>Not covered here.</b> Whether the resulting access pattern is
          coalesced in GMEM or bank-conflict-free in SMEM depends on the tensor
          you apply it to, not on the TiledCopy. Take the
          <code>layout_tv</code> and <code>Tiler_MN</code> printed below over to
          the <b>TV Layout</b> tab and supply a data layout there.
        </div>
      </div>
${mtcVizSection(id, 'mtv')}
    </div>`;
}

const mtvState = {};

function renderMakeTiledCopyTv(tabId) {
  showErr(`${tabId}-mtv-error`, '');
  showWarn(`${tabId}-mtv-warning`, '');
  try {
    const atom = mtcReadAtom(tabId, 'mtv');

    const thrStr = document.getElementById(`${tabId}-mtv-thr-input`).value;
    const valStr = document.getElementById(`${tabId}-mtv-val-input`).value;
    updateRankWarning(`${tabId}-mtv-warning`, [['thr_layout', thrStr], ['val_layout', valStr]]);

    const thrP = parseLayout(thrStr), valP = parseLayout(valStr);
    const thrSP = stripTrivialTrailing(thrP.shape, thrP.stride);
    const valSP = stripTrivialTrailing(valP.shape, valP.stride);
    const thrL = new Layout(thrSP.shape, thrSP.stride);
    const valL = new Layout(valSP.shape, valSP.stride);

    // make_layout_tv's stated precondition, which neither CuTe nor the DSL
    // actually enforces.
    mtcRequireCompact('thr_layout', thrL);
    mtcRequireCompact('val_layout', valL);

    const thrSize = thrL.size(), valSize = valL.size();
    mtcRequireAtomDivides(valSize, atom.atomNumVal, atom.numBits, atom.elemBits, atom.dtype);
    const frgX = valSize / atom.atomNumVal;

    const layout_mn = raked_product(thrL, valL);
    const tiler_mn  = product_each(layout_mn.shape);
    const layout_tv = composition(right_inverse(layout_mn), new Layout([thrSize, valSize]));

    const layoutTVStr = formatLayoutStr(layout_tv.shape, layout_tv.stride);
    const tilerMNStr  = `(${tiler_mn.join(', ')})`;

    const look = mtcBuildTileLookup(layout_tv, tiler_mn, thrSize, valSize);
    mtcCoverageCheck(look, tiler_mn, thrSize, valSize, true, mtcHasBroadcast(layout_tv));

    const vecCheck = mtcVectorizationCheck(atom.atomNumVal, layout_tv, tiler_mn);
    document.getElementById(`${tabId}-mtv-tile-result`).innerHTML =
      `<div class="cuo-result-line">layout_mn = raked_product(thr, val) = <b>${formatLayoutStr(layout_mn.shape, layout_mn.stride)}</b></div>` +
      `<div class="cuo-result-line">Tiler_MN  = product_each(shape) = <b>${tilerMNStr}</b></div>` +
      `<div class="cuo-result-line">layout_tv = right_inverse(layout_mn) = <b>${layoutTVStr}</b></div>` +
      `<div class="cuo-result-line">TiledNumThr = ${thrSize}, TiledNumVal = ${valSize}, ` +
      `FrgV = ${atom.atomNumVal}, FrgX = ${frgX}</div>` +
      mtcFormatVecCheck(vecCheck, atom.atomNumVal) +
      `<div class="cuo-result-line" style="color:#9ca3af">Feed these two to the make_tiled_copy tab: ` +
      `layout_tv <b>${layoutTVStr}</b>, Tiler_MN <b>${tilerMNStr}</b></div>`;

    const hl = readHighlightTid(tabId, 'mtv', thrSize);
    const warn = [atom.cpasyncWarn, hl.warn].filter(Boolean).join(' ');
    if (warn) showWarn(`${tabId}-mtv-warning`, warn);

    const prev = mtvState[tabId] || {};
    mtvState[tabId] = {
      ...atom, thrL, valL, layout_mn, layout_tv, tiler_mn, thrSize, valSize, frgX,
      lookup: look.lookup, highlightTid: hl.tid, vecCheck, layoutTVStr, tilerMNStr,
      tileMode: (prev.tileMode instanceof Set) ? prev.tileMode : new Set(),
    };

    mtcRenderAtomViz(tabId, 'mtv', mtvState[tabId]);
    mtcRenderTileViz(tabId, 'mtv', mtvState[tabId]);
    mtcRenderThreadPanel(tabId, 'mtv', mtvState[tabId]);
    updateOuterTabLabel(tabId, `make_tiled_copy_tv:${atom.numBits}b/${atom.dtype}`);
  } catch (e) {
    showErr(`${tabId}-mtv-error`, e.message);
    mtcClearPanes(tabId, 'mtv');
    const item = document.getElementById(`${tabId}-mtv-thread-item`);
    if (item) item.style.display = 'none';
  }
}

function setMtvHighlight(tabId) {
  const s = mtvState[tabId];
  if (!s || !s.layout_tv) return;
  renderMakeTiledCopyTv(tabId);
}

function setMTV(tabId, opKey, bits, dtype, thr, val) {
  document.getElementById(`${tabId}-mtv-op-input`).value    = opKey;
  document.getElementById(`${tabId}-mtv-bits-input`).value  = bits;
  document.getElementById(`${tabId}-mtv-dtype-input`).value = dtype;
  document.getElementById(`${tabId}-mtv-thr-input`).value   = thr;
  document.getElementById(`${tabId}-mtv-val-input`).value   = val;
  renderMakeTiledCopyTv(tabId);
}

function exportMTV(tabId) {
  exportURL(`${tabId}-mtv-export`, 'make_tiled_copy_tv',
    document.getElementById(`${tabId}-mtv-op-input`).value,
    document.getElementById(`${tabId}-mtv-bits-input`).value,
    document.getElementById(`${tabId}-mtv-dtype-input`).value,
    document.getElementById(`${tabId}-mtv-thr-input`).value,
    document.getElementById(`${tabId}-mtv-val-input`).value);
}

// Decode the access-pattern tab without confusing its TV map with an ordinary
// tensor layout. Preserve source thr/val layouts whenever the URL carries them.
function mtvDecodeTVLayout(text) {
  const p=parsePastedKeyParam(text);
  if (!p || p.feature !== 'tv') throw new Error('Paste a TV Layout URL.');
  if (p.method === '2') {
    const tp=parseLayout(p.inputs[0]), vp=parseLayout(p.inputs[1]);
    const thr=new Layout(tp.shape,tp.stride), val=new Layout(vp.shape,vp.stride);
    const r=make_layout_tv(thr,val);
    return {tv:r.layout_tv,tile:r.tiler_mn,thr:p.inputs[0],val:p.inputs[1]};
  }
  const raw=parseLayout(p.inputs[0]), tv=new Layout(raw.shape,raw.stride), tile=parseValue(p.inputs[1]);
  if (!Array.isArray(tile) || tile.length!==2 || tile.some(x=>!Number.isSafeInteger(x)||x<=0) || tv.shape.length!==2)
    throw new Error('TV Layout needs thread/value modes and a positive (M,N) tile.');
  return {tv,tile};
}

function mtvDecodeThrVal(text) {
  const r=mtvDecodeTVLayout(text);
  if (r.thr) return r;
  const T=product(r.tv.shape[0]), V=product(r.tv.shape[1]), [M,N]=r.tile;
  // A primitive TV map need not be a raked product. Only accept a factorization
  // after checking every thread/value point, never infer it from shape alone.
  if (T*V===M*N && T*V<=1000000) {
    for(let tm=1;tm<=Math.min(T,M);tm++) {
      if(T%tm || M%tm) continue;
      const tn=T/tm;
      if(N%tn) continue;
      const vm=M/tm, vn=N/tn;
      for(const ts of [[1,tm],[tn,1]]) for(const vs of [[1,vm],[vn,1]]) {
        const thr=new Layout([tm,tn],ts), val=new Layout([vm,vn],vs);
        const candidate=make_layout_tv(thr,val).layout_tv;
        let match=true;
        for(let t=0;t<T&&match;t++) for(let v=0;v<V;v++) {
          if(candidate.call([t,v])!==r.tv.call([t,v])) {match=false;break;}
        }
        if(match) return {...r,thr:formatLayoutStr(thr.shape,thr.stride),val:formatLayoutStr(val.shape,val.stride)};
      }
    }
  }
  throw new Error('This TV map cannot be factored into supported compact thr/val layouts. Use make_tiled_copy with its TV layout and tile, or export TV Layout from thr_layout/val_layout.');
}

function mtvImportTVLayout(tabId,text) {
  const r=mtvDecodeThrVal(text); // Validate everything before changing controls.
  document.getElementById(`${tabId}-mtv-thr-input`).value=r.thr;
  document.getElementById(`${tabId}-mtv-val-input`).value=r.val;
  renderMakeTiledCopyTv(tabId);
}

async function mtvDecodeTVPasteboard(tabId) {
  try {
    const text=await readPasteboardText('Paste a TV Layout URL:');
    if(text===null || text===undefined) return;
    mtvImportTVLayout(tabId,text);
  } catch(e) {showErr(`${tabId}-mtv-error`,e.message);}
}
