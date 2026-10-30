// domain_offset rebases the iterator; it does not slice or change the layout.
const domainOffsetState = {};

function domainOffsetTuple(raw, label) {
  const text=String(raw).trim();
  try {
    if (!/^[\d\s(),-]+$/.test(text)) throw new Error();
    const value=JSON.parse(text.replace(/\(/g,'[').replace(/\)/g,']').replace(/,\s*]/g,']'));
    if (flatten(value).some(x=>!Number.isSafeInteger(x))) throw new Error();
    return value;
  } catch (_) {throw new Error(`${label} must contain whole numbers, for example (1, 2). _ / None denotes slicing and is not accepted here.`);}
}

function domainOffsetCoordinate(coord, shape) {
  if (Array.isArray(shape)) {
    if (Array.isArray(coord)) {
      if (coord.length!==shape.length) throw new Error('Coordinate tuple must match the layout mode structure.');
      return shape.map((s,i)=>domainOffsetCoordinate(coord[i],s));
    }
    // CuTe permits flat coordinates into nested modes. Signed division truncates
    // toward zero; the final mode is unbounded (unlike idx2crd's domain wrap).
    let rest=coord;
    return shape.map((s,i)=>{
      const part=i===shape.length-1?rest:rest%product(s);
      if (i<shape.length-1) rest=Math.trunc(rest/product(s));
      return domainOffsetCoordinate(part,s);
    });
  }
  if (Array.isArray(coord)) {
    if (coord.length!==1) throw new Error('Coordinate tuple must match the layout mode structure.');
    return domainOffsetCoordinate(coord[0],shape);
  }
  return coord;
}

function domainOffsetCompute(tensorText, coordText) {
  let text=String(tensorText).trim(), originText=null;
  if (/^sw(?:izzle)?\s*</i.test(text)) throw new Error('Use an affine or scaled-basis layout here; swizzled layouts are not yet supported.');
  const split=topLevelCompose(text);
  if (split>=0) {originText=text.slice(0,split).trim().replace(/^ArithTuple\s*/,'');text=text.slice(split+1).trim();}
  const layout=parseExactLayoutInput(text,{basis:true}), basis=has_basis_stride(layout.stride);
  if (basis && flatten(layout.stride).some(x=>!isBasis(x) && x!==0)) throw new Error('CuTeDSL domain_offset cannot mix nonzero integer strides with scaled-basis strides.');
  const coord=domainOffsetCoordinate(domainOffsetTuple(coordText,'Coordinate'),layout.shape);
  const origin=originText===null?null:domainOffsetTuple(originText,'Initial iterator');
  let initial, delta;
  if (basis) {
    if (origin!==null && (!Array.isArray(origin) || origin.some(Array.isArray))) throw new Error('A coordinate tensor needs a flat tuple iterator such as (0, 0).');
    const ndim=Math.max(1,basisRank(layout.stride),origin?origin.length:0);
    if (origin && origin.length!==ndim) throw new Error('The initial iterator needs one entry per output coordinate axis.');
    initial=origin || new Array(ndim).fill(0);
    delta=new Array(ndim).fill(0);
    crd2basis(coord,layout.shape,layout.stride,delta);
  } else {
    if (Array.isArray(origin)) throw new Error('An ordinary tensor needs a scalar initial iterator, for example 10 o (3,4):(4,1).');
    initial=origin===null?0:origin;
    // Explicit normalized coordinates avoid JavaScript floor-division for
    // negative flat coordinates into hierarchical modes.
    delta=crd2idx(coord,layout.shape,layout.stride);
  }
  const shifted=basis?initial.map((x,i)=>x+delta[i]):initial+delta;
  if ([...flatten(delta),...flatten(shifted)].some(x=>!Number.isSafeInteger(x))) throw new Error('Iterator offset exceeds the safe integer range.');
  const format=x=>Array.isArray(x)?`(${x.map(format).join(',')})`:String(x);
  const layoutText=formatLayoutStr(layout.shape,layout.stride);
  const display=layout.rank()>1?{shape:[layout.shape[0],layout.shape.slice(1)],stride:[layout.stride[0],layout.stride.slice(1)]}
    :{shape:[layout.shape,1],stride:[layout.stride,0]};
  const inDomain=(c,s)=>Array.isArray(s)?s.every((x,i)=>inDomain(c[i],x)):c>=0 && c<s;
  const rowCoord=layout.rank()>1?coord[0]:coord, rowShape=layout.rank()>1?layout.shape[0]:layout.shape;
  const colCoord=layout.rank()>1?coord.slice(1):0, colShape=layout.rank()>1?layout.shape.slice(1):1;
  const selectedGrid=[inDomain(rowCoord,rowShape)?crd2idx(rowCoord,rowShape):null,
    inDomain(colCoord,colShape)?crd2idx(colCoord,colShape):null];
  return {layout,layoutText,coord,basis,initial,delta,shifted,display,selectedGrid,
    initialText:format(initial),deltaText:format(delta),shiftedText:format(shifted),
    tensorText:`${format(shifted)} o ${layoutText}`};
}

// Assign one palette color per addressed output element. Full coordinate tuples
// are keys for coordinate tensors; aliases intentionally share their color.
function domainOffsetColorMap(result) {
  const colors=new Map();
  for (let i=0;i<result.layout.size();i++) {
    let value;
    if (result.basis) {
      const delta=new Array(result.shifted.length).fill(0);
      crd2basis(idx2crd(i,result.layout.shape),result.layout.shape,result.layout.stride,delta);
      value=result.shifted.map((x,j)=>x+delta[j]);
    } else value=result.shifted+result.layout.call(i);
    const key=JSON.stringify(value);
    if (!colors.has(key)) colors.set(key,colorHighlight(colors.size));
  }
  return colors;
}

function generateDomainOffsetTabContent(id) {
  return `<div id="${id}-tab-domain_offset" class="panel"><div class="controls">
    <details class="cuo-section" open><summary>1. Coord</summary><div class="cuo-section-body">
      ${layoutInputField({id:`${id}-dof-coord-input`,label:'Coordinate offset',value:'(1,2)'})}
    </div></details>
    <details class="cuo-section" open><summary>2. Tensor</summary><div class="cuo-section-body">
      ${layoutInputField({id:`${id}-dof-layout-input`,label:'Layout / tensor — optional initial iterator o layout',value:'(3,4):(4,1)'})}
    </div></details>
    ${statusDivs(`${id}-dof`)}
    <div id="${id}-dof-result" class="comp-result-box" style="white-space:pre-wrap"></div>
    <button class="btn btn-render" onclick="renderDomainOffset('${id}')">Render</button>
    <button class="btn btn-render" id="${id}-dof-export" onclick="exportDomainOffset('${id}')">Export URL</button>
    <div class="presets"><h3>Presets</h3><div class="preset-list">
      <button class="preset-btn" onclick="setDomainOffset('${id}','(3,4):(4,1)','(1,2)')">Row-major — shift base by 6 elements</button>
      <button class="preset-btn" onclick="setDomainOffset('${id}','(3,4):(1,3)','(1,2)')">Column-major — shift base by 7 elements</button>
      <button class="preset-btn" onclick="setDomainOffset('${id}','10 o (3,4):(4,1)','(1,2)')">Already shifted — base 10 → 16</button>
      <button class="preset-btn" onclick="setDomainOffset('${id}','((2,2),4):((1,8),2)','((1,0),2)')">Nested modes</button>
      <button class="preset-btn" onclick="setDomainOffset('${id}','(3,4):(1@0,1@1)','(1,2)')">Identity tensor — shift output coordinates</button>
      <button class="preset-btn" onclick="setDomainOffset('${id}','(3,4):(4,1)','(-1,0)')">Negative offset — move the base backward</button>
    </div></div>
    <div class="hint">Matching colors identify the same addressed element in both views. Gray input cells are not addressed by the shifted view. Red row and blue column indices mark the selected coordinate. The outlined cell becomes (0,0) in the shifted view.<br><br>For a row-major <code>(3,4):(4,1)</code> tensor, <code>domain_offset((1,2), t)</code> moves the base by <code>1×4 + 2×1 = 6</code> elements. The returned view is still <b>3×4</b>. It performs no copy, slicing, cropping, or allocation-bounds check. Indexing chooses elements or removes modes; domain_offset keeps every mode. Cells show element offsets relative to the original base (or output coordinates for coordinate tensors), not stored data values.</div>
  </div><div class="comp-results">
    ${layoutVisualization({hostId:`${id}-dof-src-svg`,modeGroupId:`${id}-dof-src-mode-btns`,title:'Original tensor',onMode:mode=>`toggleDomainOffsetMode('${id}','src','${mode}')`})}
    ${layoutVisualization({hostId:`${id}-dof-dst-svg`,modeGroupId:`${id}-dof-dst-mode-btns`,title:'After domain_offset — same shape and strides',onMode:mode=>`toggleDomainOffsetMode('${id}','dst','${mode}')`})}
  </div></div>`;
}

function renderDomainOffset(id) {
  const p=`${id}-dof`, state=domainOffsetState[id] ||= {srcMode:new Set(['value']),dstMode:new Set(['value'])};
  const result=document.getElementById(`${p}-result`);
  showErr(`${p}-error`,'');
  try {
    const raw=document.getElementById(`${p}-layout-input`).value;
    const r=domainOffsetCompute(raw,document.getElementById(`${p}-coord-input`).value);
    if (r.layout.size()>MAX_CELLS) throw new Error(`Grid exceeds ${MAX_CELLS.toLocaleString()} cells. Choose a smaller layout.`);
    updateRankWarning(`${p}-warning`,[['Tensor layout',r.layoutText]]);
    if (r.selectedGrid.includes(null)) {
      const warning=document.getElementById(`${p}-warning`);
      showWarn(`${p}-warning`,[warning.textContent,'Selected coordinate lies outside the original grid; there is no source cell to outline.'].filter(Boolean).join(' '));
    }
    result.textContent=`Iterator: ${r.initialText} + ${r.deltaText} = ${r.shiftedText}\nReturned tensor = ${r.tensorText}\nLayout unchanged · ${r.layout.size()} elements · no cropping or bounds check`;
    result.classList.add('visible');
    const colors=domainOffsetColorMap(r);
    const colorFn=value=>colors.get(JSON.stringify(value)) || '#e5e7eb';
    for (const side of ['src','dst']) {
      const iterator=side==='src'?r.initial:r.shifted;
      const [row,col]=side==='src'?r.selectedGrid:[0,0];
      const selection={rows:new Set(row===null?[]:[row]),cols:new Set(col===null?[]:[col]),
        rowColor:LAYOUT_ROW_COLOR,colColor:LAYOUT_COL_COLOR,
        edgeCells:new Set(row===null || col===null?[]:[row+col*product(r.display.shape[0])])};
      renderLayoutVisualization({hostId:`${p}-${side}-svg`,modeGroupId:`${p}-${side}-mode-btns`,
        ...r.display,modes:side==='src'?state.srcMode:state.dstMode,basis:r.basis,
        ndim:r.basis?iterator.length:0,origin:r.basis?iterator:null,offset:r.basis?0:iterator,selection,colorFn});
    }
    updateOuterTabLabel(id,`domain_offset:${r.shiftedText}`);
  } catch(e) {
    showErr(`${p}-error`,e.message);result.classList.remove('visible');
    for (const side of ['src','dst']) document.getElementById(`${p}-${side}-svg`).innerHTML='';
  }
}
function toggleDomainOffsetMode(id,side,mode) {
  const state=domainOffsetState[id] ||= {srcMode:new Set(['value']),dstMode:new Set(['value'])};
  toggleLayoutVisualizationMode(side==='src'?state.srcMode:state.dstMode,mode);
  renderDomainOffset(id);
}
function setDomainOffset(id,layout,coord) {
  domainOffsetCompute(layout,coord);
  document.getElementById(`${id}-dof-layout-input`).value=layout;
  document.getElementById(`${id}-dof-coord-input`).value=coord;
  renderDomainOffset(id);
}
function exportDomainOffset(id) {
  exportURL(`${id}-dof-export`,'domain_offset',document.getElementById(`${id}-dof-layout-input`).value,document.getElementById(`${id}-dof-coord-input`).value);
}
