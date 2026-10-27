// Hopper utility helpers. Computation mirrors hopper_helpers and tile_to_shape;
// the swizzle is expressed in element offsets, exactly as returned by CuTeDSL.
const SML_DTYPES = {...DTYPE_BITS, float_e4m3_t: 8, float_e5m2_t: 8};
const smlState = {a: {}, b: {}, epi: {}};

function smlTuple(text, label) {
  try {
    if (!/^[\d\s(),]+$/.test(String(text).trim())) throw new Error();
    return JSON.parse(String(text).replace(/\(/g,'[').replace(/\)/g,']').replace(/,\s*]/g,']'));
  } catch (_) { throw new Error(`${label} must be a static integer tuple.`); }
}
function smlShape(text, label) {
  const shape = smlTuple(text,label);
  if (!Array.isArray(shape) || shape.length < 2 || flatten(shape).some(x => !Number.isSafeInteger(x) || x <= 0))
    throw new Error(`${label} must contain positive integer dimensions.`);
  return shape;
}

function smlCompute(kind, arch, layoutEnum, tilerText, dtype, stagesText, targetText='', orderText='') {
  if (!['a','b','epi'].includes(kind)) throw new Error('Unknown SMEM layout helper.');
  if (arch !== 'hopper') throw new Error('Blackwell is unavailable until validated on SM100 hardware.');
  if (!['ROW_MAJOR','COL_MAJOR'].includes(layoutEnum)) throw new Error('Choose a LayoutEnum.');
  const bits = SML_DTYPES[dtype];
  if (!bits) throw new Error('Unsupported dtype.');
  const stages = Number(stagesText);
  if (!String(stagesText).trim() || !Number.isSafeInteger(stages) || stages < 1) throw new Error('Stages must be a positive integer.');
  const tile = smlShape(tilerText, kind === 'epi' ? 'epi_tile' : 'mma_tiler_mnk');
  if (tile.length !== (kind === 'epi' ? 2 : 3)) throw new Error(kind === 'epi' ? 'epi_tile must have two modes (M, N).' : 'mma_tiler_mnk must have three modes (M, N, K).');
  if (kind === 'epi' && tile.some(Array.isArray)) throw new Error('epi_tile requires scalar M and N extents in the Hopper helper.');
  const kMajor = layoutEnum === 'ROW_MAJOR';
  const plane = kind === 'epi' ? tile : [tile[kind === 'a' ? 0 : 1], tile[2]];
  const majorSize = product(plane[kMajor ? 1 : 0]);
  if (majorSize % 8) throw new Error('The major dimension must be divisible by 8 (CuTeDSL helper requirement).');
  const majorBits = majorSize * bits;
  const swBytes = majorBits % 1024 === 0 ? 128 : majorBits % 512 === 0 ? 64 : majorBits % 256 === 0 ? 32 : 16;
  const sw = {B: Math.log2(swBytes / 16), M: 4, S: 3};
  const contiguous = swBytes * 8 / bits;
  const atom = kMajor ? new Layout([8,contiguous],[contiguous,1]) : new Layout([contiguous,8],[1,contiguous]);
  const target = kind === 'epi' && targetText.trim() ? smlShape(targetText,'smem_trg_shape') : [...plane,stages];
  const rank = target.length;
  const order = kind === 'epi' && orderText.trim() ? smlTuple(orderText,'smem_order') : (kMajor ? [0,1,2] : [1,0,2]);
  if (!Array.isArray(order) || order.length !== rank || order.some(x => !Number.isSafeInteger(x) || x < 0 || x >= rank) || new Set(order).size !== rank)
    throw new Error('smem_order must be a permutation of the target dimension indices.');
  const padded = append_layout(atom,rank);
  const extents = target.map(product), atomExtents = padded.shape.map(product);
  if (!Number.isSafeInteger(product(extents)) || product(extents) > 1048576) throw new Error('Layout exceeds the visualizer limit of 1,048,576 elements.');
  const repeats = extents.map((x,i)=>Math.ceil(x/atomExtents[i])), stride = repeats.map(()=>0);
  let running = 1;
  for (const i of [...order.keys()].sort((a,b)=>order[a]-order[b])) { stride[i] = repeats[i] === 1 ? 0 : running; running *= repeats[i]; }
  const layout = blocked_product(padded,new Layout(repeats,stride));
  const layoutText = formatLayoutStr(layout.shape,layout.stride);
  const composedText = `Sw<${sw.B},${sw.M},${sw.S}> o ${layoutText}`;
  const major = kind === 'epi' ? (kMajor ? 'N' : 'M') : kMajor ? 'K' : kind === 'a' ? 'M' : 'N';
  return {layout,sw,layoutText,composedText,swBytes:sw.B ? swBytes : 0,major,extents:layout.shape.map(product),requestedExtents:extents};
}

function generateSmemLayoutTabContent(id, kind) {
  const p = `sml-${kind}`, epi = kind === 'epi';
  return `<div id="${id}-tab-make_smem_layout_${kind}" class="panel"><div class="controls">
    <details class="cuo-section" open><summary>1. make_smem_layout_${kind}</summary><div class="cuo-section-body">
      <div class="form-group"><label>Architecture</label><select id="${id}-${p}-arch"><option value="hopper">Hopper (SM90)</option><option value="blackwell" disabled>Blackwell (SM100) — awaiting hardware validation</option></select></div>
      <div class="form-group"><label>${epi ? 'epi_layout' : kind+'_layout'} — LayoutEnum</label><select id="${id}-${p}-enum"><option value="ROW_MAJOR">LayoutEnum.ROW_MAJOR</option><option value="COL_MAJOR">LayoutEnum.COL_MAJOR</option></select></div>
      ${layoutInputField({id:`${id}-${p}-tiler-input`,label:epi ? 'epi_tile — (M, N)' : 'mma_tiler_mnk — (M, N, K)',value:epi ? '(16,16)' : '(16,16,16)'})}
      <div class="form-group"><label>${epi ? 'epi_dtype' : kind+'_dtype'}</label><select id="${id}-${p}-dtype">${dtypeOptions('half_t')}<option value="float_e4m3_t">float_e4m3_t (8)</option><option value="float_e5m2_t">float_e5m2_t (8)</option></select></div>
      <div class="form-group"><label>${epi ? 'epi_stage' : 'num_stages'}</label><input id="${id}-${p}-stages" type="number" min="1" value="1"></div>
      ${epi ? `<details class="cuo-section"><summary>Optional arguments</summary><div class="cuo-section-body">
        ${layoutInputField({id:`${id}-${p}-target-input`,label:'smem_trg_shape (optional)',value:''})}
        ${layoutInputField({id:`${id}-${p}-order-input`,label:'smem_order (optional)',value:''})}
      </div></details>` : ''}
    </div></details>
    ${statusDivs(`${id}-${p}`)}
    <div id="${id}-${p}-result" class="comp-result-box" style="white-space:pre-wrap"></div>
    <button class="btn btn-render" onclick="renderSmemLayout('${id}','${kind}')">Render</button>
    <button class="btn btn-render" id="${id}-${p}-export" onclick="exportSmemLayout('${id}','${kind}')">Export URL</button>
    <div class="presets"><h3>Presets</h3><div class="preset-list">
      <button class="preset-btn" onclick="setSml('${id}','${kind}','hopper','ROW_MAJOR','${epi ? '(16,64)' : '(16,16,64)'}','half_t','2','','')">FP16 · 128B swizzle · 2 stages</button>
      <button class="preset-btn" onclick="setSml('${id}','${kind}','hopper','COL_MAJOR','${epi ? '(32,16)' : '(32,32,16)'}','half_t','1','','')">${kind === 'b' ? 'N' : 'M'}-major · 64B swizzle</button>
      <button class="preset-btn" onclick="setSml('${id}','${kind}','hopper','ROW_MAJOR','${epi ? '(16,8)' : '(16,16,8)'}','half_t','1','','')">FP16 · no swizzle</button>
    </div></div>
    <div class="hint">Returns the Hopper helper's composed layout in element offsets. Use Export URL to register the result in the Variable Table as a Swizzled Layout. This helper describes storage; it does not allocate or move data.</div>
  </div><div class="comp-results" style="grid-template-columns:1fr">
    ${layoutVisualization({hostId:`${id}-${p}-dst-svg`,modeGroupId:`${id}-${p}-mode-btns`,
      title:'SMEM layout — physical element offsets',onMode:mode=>`toggleSmlMode('${id}','${kind}','${mode}')`,
      beforeGrid:`<div class="form-group"><label>Stage / flattened remaining-mode index</label><input id="${id}-${p}-stage-view" type="number" min="0" value="0" onchange="renderSmemLayout('${id}','${kind}')"></div>`})}
  </div></div>`;
}
function generateSmemLayoutATabContent(id) { return generateSmemLayoutTabContent(id,'a'); }
function generateSmemLayoutBTabContent(id) { return generateSmemLayoutTabContent(id,'b'); }
function generateSmemLayoutEpiTabContent(id) { return generateSmemLayoutTabContent(id,'epi'); }
function renderSmemLayoutA(id) { renderSmemLayout(id,'a'); }
function renderSmemLayoutB(id) { renderSmemLayout(id,'b'); }
function renderSmemLayoutEpi(id) { renderSmemLayout(id,'epi'); }
function smlRead(id,kind) {
  return ['arch','enum','tiler-input','dtype','stages',...(kind==='epi'?['target-input','order-input']:[])].map(k=>document.getElementById(`${id}-sml-${kind}-${k}`).value);
}
function renderSmemLayout(id,kind) {
  const p=`${id}-sml-${kind}`, host=document.getElementById(`${p}-dst-svg`), result=document.getElementById(`${p}-result`);
  const state=smlState[kind][id] ||= {mode: new Set(['value'])};
  showErr(`${p}-error`,'');
  showErr(`${p}-warning`,'');
  try {
    const r=smlCompute(kind,...smlRead(id,kind));
    const stageEl=document.getElementById(`${p}-stage-view`), stage=Number(stageEl.value), n=product(r.extents.slice(2));
    stageEl.max=String(n-1);
    if (!String(stageEl.value).trim() || !Number.isSafeInteger(stage) || stage<0 || stage>=n) throw new Error(`Stage / remaining-mode index must be in [0, ${n}).`);
    const coord=[null,null,...idx2crd(stage,r.layout.shape.slice(2))];
    const [plane,offset]=slice_and_offset(coord,r.layout);
    if (plane.size()>MAX_CELLS) throw new Error(`The displayed stage exceeds ${MAX_CELLS.toLocaleString()} cells. Choose a smaller tile.`);
    updateRankWarning(`${p}-warning`,[['Displayed stage',formatLayoutStr(plane.shape,plane.stride)]]);
    if (r.extents.some((x,i)=>x!==r.requestedExtents[i])) showErr(`${p}-warning`,'CuTeDSL rounds the target up to whole SMEM layout atoms. The returned layout includes padding.');
    result.textContent=`Returned layout = ${r.composedText}\n${r.major}-major · ${r.layout.size()} elements`;
    result.classList.add('visible');
    renderLayoutVisualization({hostId:`${p}-dst-svg`,modeGroupId:`${p}-mode-btns`,
      shape:plane.shape,stride:plane.stride,modes:state.mode,offset,swizzle:r.sw});
    updateOuterTabLabel(id,`make_smem_layout_${kind}`);
  } catch(e) { showErr(`${p}-error`,e.message); result.classList.remove('visible'); host.innerHTML=''; }
}
function toggleSmlMode(id,kind,mode) {
  if (!['value','index','coord'].includes(mode)) return;
  const state=smlState[kind][id] ||= {mode: new Set(['value'])};
  toggleLayoutVisualizationMode(state.mode,mode);
  renderSmemLayout(id,kind);
}
function setSml(id,kind,...inputs) {
  smlCompute(kind,...inputs); // Validate before altering controls, including URL imports.
  const fields=['arch','enum','tiler-input','dtype','stages',...(kind==='epi'?['target-input','order-input']:[])];
  fields.forEach((k,i)=>document.getElementById(`${id}-sml-${kind}-${k}`).value=inputs[i] || '');
  document.getElementById(`${id}-sml-${kind}-stage-view`).value='0';
  renderSmemLayout(id,kind);
}
function exportSmemLayout(id,kind) { exportURL(`${id}-sml-${kind}-export`,`make_smem_layout_${kind}`,...smlRead(id,kind)); }
