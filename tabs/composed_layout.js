// A ComposedLayout retains the chain; it need not have ordinary strides.
const composedLayoutState = {};

function clFormat(value) {
  return Array.isArray(value) ? `(${value.map(clFormat).join(',')})` : String(value);
}

// DOM-free constructor/evaluator. All point arrays use CuTe's domain order.
function clCompute(baseText, offsetText, kind, innerText) {
  const base = parseExactLayoutInput(baseText, {basis:true});
  if (base.size() > MAX_CELLS) throw new Error(`Grid exceeds ${MAX_CELLS} cells. Choose a smaller base layout.`);
  const basis = has_basis_stride(base.stride);
  if (basis && flatten(base.stride).some(x => !isBasis(x) && x !== 0))
    throw new Error('Use either integer strides or scaled-basis strides in the base layout.');
  if (!['swizzle','layout'].includes(kind)) throw new Error('Choose Swizzle or Layout as the inner transformation.');
  const offset = domainOffsetTuple(offsetText, 'Offset');
  const ndim = basis ? basisRank(base.stride) : 0;
  if (basis ? !Array.isArray(offset) || offset.length !== ndim || offset.some(Array.isArray) : Array.isArray(offset))
    throw new Error(basis ? `Use a flat offset tuple with ${ndim} entries for this coordinate layout.` : 'Use a scalar integer offset for an integer-strided base layout.');
  const sw = kind === 'swizzle' ? parseSwizzleSpec(innerText) : null;
  if (kind === 'swizzle' && (!sw || basis)) throw new Error('Swizzle requires scalar base offsets and a valid B, M, S specification.');
  const inner = kind === 'layout' ? parseExactLayoutInput(innerText) : null;
  const plain = [], shifted = [], values = [];
  for (let i = 0; i < base.size(); i++) {
    let x;
    if (basis) {
      x = new Array(ndim).fill(0);
      crd2basis(idx2crd(i,base.shape),base.shape,base.stride,x);
    } else x = base.call(i);
    const y = basis ? x.map((v,j) => v + offset[j]) : x + offset;
    if (flatten(y).some(v => !Number.isSafeInteger(v))) throw new Error('Intermediate offset exceeds the safe integer range.');
    // The shared swizzle uses integer bit operations; avoid silent truncation.
    if (sw && (y < 0 || y >= 2**31)) throw new Error('Swizzle input offsets must be in [0, 2^31).');
    const z = sw ? applySwizzleOffset(y,sw)
      : crd2idx(domainOffsetCoordinate(y,inner.shape),inner.shape,inner.stride);
    if (!Number.isSafeInteger(z)) throw new Error('Result exceeds the safe integer range.');
    plain.push(x); shifted.push(y); values.push(z);
  }
  const rows = product(Array.isArray(base.shape) ? base.shape[0] : base.shape);
  const baseLayoutText = formatLayoutStr(base.shape,base.stride);
  const innerLayoutText = inner ? formatLayoutStr(inner.shape,inner.stride) : `Sw<${sw.B},${sw.M},${sw.S}>`;
  return {base,inner,sw,offset,plain,shifted,values,rows,cols:base.size()/rows,
    text:`${innerLayoutText} o ${clFormat(offset)} o ${baseLayoutText}`};
}

function generateComposedLayoutTabContent(id) {
  return `<div id="${id}-tab-composed_layout" class="panel"><div class="controls">
    <h2>Composed Layout</h2>
    <details class="cuo-section" open><summary>1. Base layout (outer)</summary><div class="cuo-section-body">
      ${layoutInputField({id:`${id}-cl-base-input`,label:'Base layout — applied first',value:'(4,4):(4,1)'})}
    </div></details>
    <details class="cuo-section" open><summary>2. Apply Offset</summary><div class="cuo-section-body">
      ${layoutInputField({id:`${id}-cl-offset-input`,label:'Add before the inner transformation',value:'0'})}
    </div></details>
    <details class="cuo-section" open><summary>3. Apply Inner Transformation</summary><div class="cuo-section-body">
      <label>Transformation ${infoIcon('make_composed_layout(inner, offset, outer) evaluates inner(offset + outer(coord)). A ComposedLayout has the base layout’s logical shape; it need not have ordinary strides.')}</label>
      <select id="${id}-cl-kind" onchange="clSyncKind('${id}')"><option value="swizzle">Swizzle</option><option value="layout">Layout</option></select>
      <div id="${id}-cl-swizzle-field">${layoutInputField({id:`${id}-cl-swizzle-input`,label:'Swizzle B, M, S',value:'2,0,2'})}</div>
      <div id="${id}-cl-inner-field" style="display:none">${layoutInputField({id:`${id}-cl-inner-input`,label:'Inner layout — applied last',value:'32:2'})}</div>
    </div></details>
    ${statusDivs(`${id}-cl`)}
    <div id="${id}-cl-result" class="comp-result-box" style="white-space:pre-wrap"></div>
    <button class="btn btn-render" onclick="renderComposedLayout('${id}')">Render</button>
    <button class="btn btn-render" id="${id}-cl-export" onclick="exportCL('${id}')">Export URL</button>
    <div class="presets"><h3>Presets</h3><div class="preset-list">
      <button class="preset-btn" onclick="setCL('${id}','(4,4):(4,1)','0','swizzle','2,0,2')">4×4 XOR swizzle</button>
      <button class="preset-btn" onclick="setCL('${id}','(4,4):(4,1)','1','swizzle','2,0,2')">Offset before swizzle</button>
      <button class="preset-btn" onclick="setCL('${id}','(4,4):(4,1)','0','layout','32:2')">Layout → layout: double offsets</button>
      <button class="preset-btn" onclick="setCL('${id}','(3,4):(1@0,1@1)','(1,2)','layout','(8,8):(8,1)')">Coordinate layout → shifted row-major</button>
      <button class="preset-btn" onclick="setCL('${id}','((2,2),4):((1,8),2)','0','swizzle','2,0,2')">Nested base modes</button>
    </div></div>
    <div class="hint">Each grid keeps the base layout’s logical coordinates. Value shows the output at that stage, not stored data. The offset is added <b>before</b> the inner transformation. A Swizzle accepts scalar offsets; a Layout also accepts matching coordinate tuples from a scaled-basis base. This constructs a mapping without moving data or checking allocation bounds.</div>
  </div><div class="comp-results">
    ${['src','extra','dst'].map((stage,i)=>layoutVisualization({hostId:`${id}-cl-${stage}-svg`,title:['1. Base output','2. Apply Offset','3. Apply Inner Transformation'][i],onMode:mode=>`toggleCLMode('${id}','${stage}','${mode}')`})).join('')}
  </div></div>`;
}

function clSyncKind(id) {
  const sw = document.getElementById(`${id}-cl-kind`).value === 'swizzle';
  document.getElementById(`${id}-cl-swizzle-field`).style.display = sw ? '' : 'none';
  document.getElementById(`${id}-cl-inner-field`).style.display = sw ? 'none' : '';
}

function clReadInputs(id) {
  const read = suffix => document.getElementById(`${id}-cl-${suffix}`).value;
  const kind = read('kind');
  return [read('base-input'),read('offset-input'),kind,read(kind === 'swizzle' ? 'swizzle-input' : 'inner-input')];
}

function renderComposedLayout(id) {
  const p = `${id}-cl`, state = composedLayoutState[id] ||= {};
  const result = document.getElementById(`${p}-result`);
  showErr(`${p}-error`,''); showWarn(`${p}-warning`,'');
  clSyncKind(id);
  try {
    const inputs = clReadInputs(id), r = clCompute(...inputs);
    updateRankWarning(`${p}-warning`,[['Base layout',inputs[0]]]);
    result.textContent = `ComposedLayout = ${r.text}\nC(coord) = inner(offset + outer(coord))\nLogical shape = ${clFormat(r.base.shape)} · ${r.base.size()} elements`;
    result.classList.add('visible');
    const colors = new Map();
    for (const points of [r.plain,r.shifted,r.values]) for (const value of points) {
      const key = JSON.stringify(value);
      if (!colors.has(key)) colors.set(key,colorHighlight(colors.size));
    }
    for (const [stage,points] of [['src',r.plain],['extra',r.shifted],['dst',r.values]]) {
      const modes = state[`${stage}Mode`] ||= new Set(['value']);
      const svg = buildColoredLayoutSVG([r.rows,r.cols],[1,r.rows],modes,(m,n,value,index)=>({
        bg:colors.get(JSON.stringify(points[index])),
        text:buildCellLines(modes,clFormat(points[index]),index,`(${m},${n})`)
      }));
      renderLayoutVisualization({hostId:`${p}-${stage}-svg`,modes,svg});
    }
    updateOuterTabLabel(id,'Composed Layout');
  } catch (e) {
    showErr(`${p}-error`,e.message);
    result.textContent = ''; result.classList.remove('visible');
    for (const stage of ['src','extra','dst']) document.getElementById(`${p}-${stage}-svg`).innerHTML = '';
  }
}

function toggleCLMode(id,stage,mode) {
  const state = composedLayoutState[id] ||= {};
  toggleLayoutVisualizationMode(state[`${stage}Mode`] ||= new Set(['value']),mode);
  renderComposedLayout(id);
}

function setCL(id,base,offset,kind,inner) {
  document.getElementById(`${id}-cl-base-input`).value = base;
  document.getElementById(`${id}-cl-offset-input`).value = offset;
  document.getElementById(`${id}-cl-kind`).value = kind;
  document.getElementById(`${id}-cl-${kind === 'swizzle' ? 'swizzle' : 'inner'}-input`).value = inner;
  renderComposedLayout(id);
}

function exportCL(id) {
  exportURL(`${id}-cl-export`,'composed_layout',...clReadInputs(id));
}
