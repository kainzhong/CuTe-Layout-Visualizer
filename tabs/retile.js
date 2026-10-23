// TiledCopy.retile: copy_atom.hpp:263-288. Pure layout transformation;
// source and destination views refer to the same per-thread RMEM storage.
const RT_DEFAULT_TV = '((4,8,2,2),((2,2,2),(1,1))):((64,1,16,0),((32,8,256),(0,0)))';
const rtState = {};

function rtParseLayout(raw) {
  const p = parseLayout(raw);
  return new Layout(p.shape, p.stride);
}

// Static-layout upcast<N>, layout.hpp:1710. Preserve nested modes, zero
// strides, and singleton strides exactly as the CuTe implementation does.
function rtUpcast(layout, factor) {
  const visit = (shape, stride) => {
    if (Array.isArray(shape)) {
      const parts = shape.map((s, i) => visit(s, stride[i]));
      return { shape: parts.map(p => p.shape), stride: parts.map(p => p.stride) };
    }
    if (stride === 0) return { shape, stride };
    const d = Math.abs(stride);
    if (d % factor && factor % d) throw new Error('Retile upcast requires divisible static strides.');
    return { shape: Math.ceil(shape / Math.ceil(factor / d)),
      stride: Math.sign(stride) * Math.ceil(d / factor) };
  };
  const r = visit(layout.shape, layout.stride);
  return new Layout(r.shape, r.stride);
}

function rtComputeRetile(tv, tile, atomValues, input) {
  if (tv.rank() !== 2) throw new Error('TiledCopy TV layout must have two modes: (threads, values).');
  if (!Array.isArray(tile) || tile.length !== 2 || tile.some(n => !Number.isSafeInteger(n) || n < 1))
    throw new Error('This view requires a two-dimensional Tiler_MN.');
  if (input.rank() < 3) throw new Error('Input must have a value mode and at least two Rest modes (rank at least 3).');
  if (!Number.isSafeInteger(atomValues) || atomValues < 1 || product(tv.shape[1]) % atomValues)
    throw new Error('Atom values/thread must be a positive integer dividing the TiledCopy value count.');
  if (input.size() > MFRAG_MAX_REGS * 4) throw new Error('Input fragment is too large for this view.');
  const V = product(input.shape[0]);
  const inverse = composition(right_inverse(tv), new Layout(tile));
  const frgMN = rtUpcast(inverse, product(tv.shape[0]) * V);
  const frgV = zipped_divide(logical_product(new Layout(V), right_inverse(frgMN)), new Layout(atomValues));
  const split = zipped_divide(input, [V, ...product_each(frgMN.shape)]);
  const transformed = composition(split, [frgV, null]);
  const [layout] = slice_and_offset([null, [0, ...Array(input.rank() - 1).fill(null)]], transformed);
  if (layout.size() !== input.size()) throw new Error('Input fragment does not evenly cover the copy retile groups.');
  const footprint = L => Array.from({ length: L.size() }, (_, i) => L.call(i)).sort((a, b) => a - b);
  if (JSON.stringify(footprint(input)) !== JSON.stringify(footprint(layout)))
    throw new Error('This input is not compatible with the TiledCopy reference-value ordering.');
  return { layout, frgMN, frgV, split };
}

function generateRetileTabContent(id) {
  const pane = (side, title) => `<div class="comp-viz-item">
    <div class="comp-viz-header"><span class="comp-viz-title"><span id="${id}-rt-${side}-title">${title}</span>${memorySpaceBadge(`${id}-rt-${side}-space`, 'RMEM')}</span>
      <button class="mode-btn" onclick="toggleZoom('${id}-rt-${side}-svg')">Zoom in</button></div>
    <div class="cuo-viz-desc" id="${id}-rt-${side}-desc"></div>
    <div class="viz-box"><div id="${id}-rt-${side}-svg"></div></div></div>`;
  return `<div id="${id}-tab-retile" class="panel"><div class="controls">
    <h2>TiledCopy.retile</h2>
    <details class="cuo-section" open><summary>1. TiledCopy</summary><div class="cuo-section-body">
      <button class="btn btn-render btn-decode" id="${id}-rt-decode-copy" onclick="rtDecodePasteboard('${id}','copy')">Decode from Pasteboard</button>
      <details class="cuo-section" id="${id}-rt-copy-manual-construction"><summary>Manual construction</summary><div class="cuo-section-body">
      ${layoutInputField({id:`${id}-rt-tv-input`,label:'TV Layout tiled',value:RT_DEFAULT_TV,hint:'Two modes: (threads, values). Decode a make_tiled_copy, make_tiled_copy_tv, or make_tiled_copy_A/B URL.'})}
      ${layoutInputField({id:`${id}-rt-tile-input`,label:'Tiler_MN',value:'(32,16)',hint:'Two independent tiler modes, as printed by TiledCopy.'})}
      <div class="form-group"><label>CopyAtom values per thread</label><input id="${id}-rt-values-input" type="text" value="8"></div>
      </div></details>
    </div></details>
    <details class="cuo-section" open><summary>2. RMEM fragment from make_fragment</summary><div class="cuo-section-body">
      <button class="btn btn-render btn-decode" id="${id}-rt-decode-input" onclick="rtDecodePasteboard('${id}','input')">Decode from Pasteboard</button>
      <details class="cuo-section" id="${id}-rt-input-manual-construction"><summary>Manual construction</summary><div class="cuo-section-body">
      ${layoutInputField({id:`${id}-rt-input-input`,label:'RMEM fragment layout',value:'((2,2,2),4,2):((1,2,4),16,8)',hint:'Copy the Export URL from make_fragment_A/B/C, then click Decode from Pasteboard to fill this fragment layout.'})}
      <div class="form-group"><label>Value dtype</label><select id="${id}-rt-dtype-input">${['half_t','bfloat16_t','float','tfloat32_t','float_e4m3_t','float_e5m2_t','int8_t','uint8_t','int16_t','int32_t'].map(d=>`<option value="${d}">${d}</option>`).join('')}</select></div>
      <div class="form-group"><label>Thread ID for T/V labels</label><input type="text" id="${id}-rt-thread-input" value="0"></div>
      </div></details>
    </div></details>
    ${statusDivs(`${id}-rt`)}
    <button class="btn btn-render" onclick="renderRetile('${id}')">Render</button>
    <button class="btn btn-render" id="${id}-rt-export" onclick="exportRT('${id}')">Export URL</button>
    <div id="${id}-rt-result" class="cuo-result"></div>
    <div class="hint">Retile changes the indexing of existing registers. It does not allocate or move values. Each pane colors its own Rest groups, so regrouping can change a value’s color. R# labels identify the same logical storage across both views; R# names logical packed words, not compiler-assigned physical registers.<br><br>
      cute.copy pairs copy-sized value groups and their Rest iterations. A raw MMA fragment can be used only when its grouping already satisfies the copy operation. Retile prepares that grouping, including gathering values from other modes when needed. In CuTeDSL, a rank-3 V-mode such as (2,2,2) also exceeds cute.copy's current V-mode rank limit; retile produces its supported (atom values, remaining values) profile.<br><br>
      Only the tiled TV map, tiler shape and atom value count enter this operation. No SRC/DST transfer, physical SMEM addressing, or instruction legality is inferred here.</div>
    <div class="presets"><h3>Presets</h3><div class="preset-list">
      <button class="preset-btn" onclick="setRT('${id}',RT_DEFAULT_TV,'(32,16)',8,'half_t','((2,2,2),4,2):((1,2,4),16,8)',0)">A — ldmatrix.x4, eight values stay together</button>
      <button class="preset-btn" onclick="setRT('${id}',RT_DEFAULT_TV,'(32,16)',2,'half_t','((2,2,2),4,2):((1,2,4),16,8)',0)">A — ldmatrix.x1, split into smaller loads</button>
      <button class="preset-btn" onclick="setRT('${id}','((4,8,2,2),((2,2),(2,1))):((64,1,0,8),((32,256),(16,0)))','(32,16)',8,'half_t','((2,2),8,2):((1,2),8,4)',0)">B — ldmatrix.x4 gathers across MMA positions</button>
      <button class="preset-btn" onclick="setRT('${id}','(4,4):(1,4)','(4,4)',2,'float','(2,2,2):(1,2,4)',0)">SIMT — two values per copy</button>
    </div></div>
  </div><div class="comp-results">${pane('src','Input RMEM fragment')}${pane('dst','Retiled register view')}</div></div>`;
}

function renderRetile(tabId) {
  const read = key => document.getElementById(`${tabId}-rt-${key}-input`).value;
  showErr(`${tabId}-rt-error`, '');
  showWarn(`${tabId}-rt-warning`, '');
  document.getElementById(`${tabId}-rt-result`).innerHTML = '';
  for (const side of ['src','dst']) document.getElementById(`${tabId}-rt-${side}-svg`).innerHTML = '';
  try {
    const tv = rtParseLayout(read('tv')), input = rtParseLayout(read('input'));
    const tile = mtcParseTiler(read('tile')).extents;
    const dtype = read('dtype'), bits = DTYPE_BITS[dtype] || (/^float_e[45]m[23]_t$/.test(dtype) ? 8 : 0);
    if (!bits || bits > 32 || 32 % bits) throw new Error('This view supports scalar widths of 8, 16, or 32 bits.');
    const threadRaw = read('thread').trim();
    if (!/^\d+$/.test(threadRaw) || !Number.isSafeInteger(Number(threadRaw))) throw new Error('Thread ID must be a nonnegative whole number.');
    const pack = 32 / bits;
    if (input.cosize() > MFRAG_MAX_REGS * pack || flatten(input.stride).some(d=>d<0)) throw new Error('Input exceeds the supported nonnegative logical register footprint.');
    const result = rtComputeRetile(tv, tile, Number(read('values')), input);
    const thread = Number(threadRaw);
    rtState[tabId] = { tv, tile, input, dtype, thread, pack, ...result };
    const fmt = L => formatLayoutStr(L.shape,L.stride);
    document.getElementById(`${tabId}-rt-result`).innerHTML =
      `<div class="cuo-result-line">retile(input) = <b>${fmt(result.layout)}</b></div>` +
      `<div class="cuo-result-line">${input.size()} scalar values; the same ${Math.ceil(input.cosize()/pack)} logical 32-bit word slots</div>` +
      `<div class="cuo-result-line">Copy grouping = ${fmt(result.frgV)}; values gathered from operand modes = ${fmt(result.frgMN)}</div>`;
    for (const [side, layout] of [['src',input],['dst',result.layout]]) {
      setMemorySpaceBadge(`${tabId}-rt-${side}-space`,'RMEM');
      document.getElementById(`${tabId}-rt-${side}-desc`).textContent = fmt(layout);
      document.getElementById(`${tabId}-rt-${side}-svg`).innerHTML = mfragBuildValueSVG({partition:layout,fragment:layout,pack},thread);
      applyZoomState(`${tabId}-rt-${side}-svg`);
    }
    updateOuterTabLabel(tabId,`retile:${fmt(result.layout)}`);
  } catch (e) { showErr(`${tabId}-rt-error`,e.message); delete rtState[tabId]; }
}

function setRT(tabId,tv,tile,values,dtype,input,thread=0) {
  for (const [key,value] of Object.entries({tv,tile,values,dtype,input,thread}))
    document.getElementById(`${tabId}-rt-${key}-input`).value = String(value);
  renderRetile(tabId);
}

function rtImportURL(tabId,kind,text) {
  const parsed = parsePastedKeyParam(text);
  if (!parsed) throw new Error('Paste an exported URL or its ?key=... query string.');
  const write = (key,value) => { document.getElementById(`${tabId}-rt-${key}-input`).value=String(value); };
  if (kind === 'copy') {
    const r = mtcDecodeURL(parsed);
    write('tv',formatLayoutStr(r.tv.shape,r.tv.stride)); write('tile',`(${r.tile.join(',')})`);
    write('values',r.atomValues); write('dtype',r.dtype);
  } else {
    if (parsed.feature !== 'make_fragment_abc') throw new Error('Paste a make_fragment_A/B/C URL.');
    const a=parsed.inputs;
    const r=mfragCompute(rtParseLayout(a[5]),a[0],a[1],Number(a[4]),
      a[2] === 'na' ? 'tfloat32_t' : a[2], a[3] === 'na' ? 'float' : a[3]);
    write('input',formatLayoutStr(r.fragment.shape,r.fragment.stride)); write('dtype',r.dtype); write('thread',a[6]||0);
  }
  renderRetile(tabId);
}

async function rtDecodePasteboard(tabId,kind) {
  const text=await readPasteboardText(`Paste the ${kind==='copy'?'TiledCopy':'make_fragment'} URL or its query string:`);
  if (text===null || text===undefined) return;
  try { rtImportURL(tabId,kind,text); } catch(e) { showErr(`${tabId}-rt-error`,e.message); }
}

function exportRT(tabId) {
  const read=key=>document.getElementById(`${tabId}-rt-${key}-input`).value;
  exportURL(`${tabId}-rt-export`,'retile',...['tv','tile','values','dtype','input','thread'].map(read));
}
