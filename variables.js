// Global named snapshots shared by every visualization tab in this browser.
const VARIABLES_STORAGE_KEY = 'cute-layout-visualizer.variables.v1';
const variablesState = {records: [], pending: [], editingId: null, previewId: null, initialized: false, storageWarning: ''};

function variablesParseLayout(raw) {
  let text = String(raw).trim();
  const origin = topLevelCompose(text);
  if (origin !== -1) text = text.slice(origin + 1).trim(); // A tensor origin is not part of its layout.
  if (!/^[\d\s(),:+@-]+$/.test(text)) throw new Error('Paste a layout or an exported visualization URL.');
  parseLayout(text, {basis:true});
  const colon = topLevelColon(text);
  const shape = parseValue(colon < 0 ? text : text.slice(0,colon));
  const stride = colon < 0 ? prefix_product(shape) : parseValue(text.slice(colon+1),true);
  const check = (s,d) => {
    if (Array.isArray(s)) {
      if (!Array.isArray(d) || s.length !== d.length) throw new Error('Layout shape and stride modes must match.');
      s.forEach((v,i) => check(v,d[i]));
    } else if (!Number.isSafeInteger(s) || s <= 0 || !(Number.isSafeInteger(d) || isBasis(d)))
      throw new Error('Invalid layout shape or stride.');
  };
  check(shape,stride);
  return new Layout(shape,stride);
}

function variablesLayoutChoice(layout, label='Layout') {
  return {type:'Layout', value:formatLayoutStr(layout.shape,layout.stride), label};
}

function variablesRawLayoutChoice(raw, label='Layout') {
  let text=String(raw).trim();
  const origin=topLevelCompose(text);
  if (origin !== -1) text=text.slice(origin+1).trim();
  const parsed=variablesParseLayout(text);
  // Preserve explicit tuple nesting exactly, including rank-one nested modes.
  return {type:'Layout',value:topLevelColon(text) < 0 ? formatLayoutStr(parsed.shape,parsed.stride) : text.replace(/\s+/g,''),label};
}

function variablesSwizzledChoice(layoutText, sw, label) {
  const choice = variablesRawLayoutChoice(layoutText,label);
  if (sw && sw.B) choice.value = `Sw<${sw.B},${sw.M},${sw.S}> o ${choice.value}`;
  return choice;
}

// DOM-free: classify constructors and resolve layout-valued results through the
// same pure helpers as their tabs. Layout values never contain an encoding URL.
function variablesClipboardChoices(raw) {
  const text = String(raw || '').trim();
  if (!text) throw new Error('The pasteboard is empty.');
  const parsed = parsePastedKeyParam(text);
  if (!parsed) {
    if (/^sw(?:izzle)?\s*</i.test(text)) {
      const composed = tmaParseSmemField(text);
      return [variablesSwizzledChoice(composed.layoutStr,composed.sw,'Layout')];
    }
    return [variablesRawLayoutChoice(text)];
  }
  const {feature,inputs:a,method} = parsed;
  const base = String(location.href || `${location.origin}${location.pathname}`).split(/[?#]/)[0];
  const url = /^(?:https?|file):\/\//i.test(text) ? text.split('#')[0] : `${base}?${new URLSearchParams({key:[feature,...(method ? [method] : []),...a].join('-')})}`;
  const object = type => [{type,value:url,label:type}];
  const layout = variablesParseLayout;
  const tiler = str => {
    const lines = str.split('\n').map(x=>x.trim()).filter(Boolean).map(layout);
    if (!lines.length) throw new Error('Tiler input is empty.');
    return lines.length === 1 ? lines[0] : lines;
  };
  if (feature === 'make_mma_atom') { mmaDecodeAtomInputs(a); return object('MMAAtom'); }
  if (feature === 'make_copy_atom') { mcaDecodeAtomURL(text); return object('CopyAtom'); }
  if (feature === 'make_tiled_mma') { mtmDecodeConstructor(a); return object('TiledMMA'); }
  if (['make_tiled_copy','make_tiled_copy_tv','make_tiled_copy_ab'].includes(feature)) {
    mtcDecodedSummary(mtcDecodeURL(parsed)); return object('TiledCopy');
  }
  if (feature === 'layout') return [variablesRawLayoutChoice(a[0])];
  if (feature === 'index') return [variablesLayoutChoice(idxComputeIndex(a[0],a[1]).layout)];
  if (feature === 'composition') return [variablesLayoutChoice(composition(layout(a[0]),tiler(a[1])))];
  if (feature === 'complement') return [variablesLayoutChoice(complement(layout(a[0]),layout(a[1]).size()))];
  const ops = {logical_divide, logical_product, blocked_product, raked_product};
  if (ops[feature]) return [variablesLayoutChoice(ops[feature](layout(a[0]),tiler(a[1])))];
  if (feature === 'zipped_divide' || feature === 'zipped_product') {
    const fns = feature === 'zipped_divide' ? {zipped_divide,tiled_divide,flat_divide} : {zipped_product,tiled_product,flat_product};
    return Object.entries(fns).map(([name,fn]) => variablesLayoutChoice(fn(layout(a[0]),tiler(a[1])),name));
  }
  if (feature === 'local_tile') return [variablesLayoutChoice(ltComputeLocalTile(a[0],a[1],a[2],a[3] || '').resultLayout)];
  if (feature === 'tv') { mtvDecodeTVLayout(text); return object('TV Layout'); }
  if (feature === 'swizzle') {
    const sw = swParse(a[1]);
    if (!sw) throw new Error('Invalid swizzle.');
    layout(a[0]); return object('Swizzle');
  }
  if (feature === 'make_fragment_abc') return [variablesLayoutChoice(mfragCompute(layout(a[5]),a[0],a[1],Number(a[4]),a[2] === 'na' ? 'tfloat32_t' : a[2],a[3] === 'na' ? 'float' : a[3]).fragment)];
  if (feature === 'retile') return [variablesLayoutChoice(rtComputeRetile(layout(a[0]),mtcParseTiler(a[1]).extents,Number(a[2]),layout(a[4])).layout)];
  if (feature === 'partition_abc') {
    const mma = mtmDecodeConstructor(a.slice(1,7));
    return [variablesLayoutChoice(pabcComputePartition(mma.atom,mma.atomLayout,mma.perm,a[0],layout(a[8]),a[7] === '' ? null : Number(a[7])).partition)];
  }
  if (feature === 'partition_sd') {
    const side=a[0], copy = method === 'url'
      ? mtcDecodeURL(parsePastedKeyParam(decodeURIComponent(a[1])))
      : mtcDecodeURL({feature:'make_tiled_copy',inputs:a.slice(1,6)});
    const tensor=layout(method === 'url' ? a[3] : a[7]), tid=method === 'url' ? a[2] : a[6];
    return [variablesLayoutChoice(psdComputePartition(copy.tv,copy.tiler,tensor,copy.copyAtom,tid === '' ? null : Number(tid),side).partition)];
  }
  if (feature === 'gemm') return [variablesLayoutChoice(gemmCompute(a.slice(0,6),a[6],a[7],a[8],a[8]).C)];
  if (feature === 'tma_partition') {
    const sm = tmaParseSmemField(a[3]);
    const r = tpComputePartition(parseLayout(sm.layoutStr),parseLayout(a[4],{basis:true}),Number(a[0]));
    return [variablesSwizzledChoice(r.sPartStr,sm.sw || (a[2] === 'none' ? null : parseSwizzleSpec(a[2])),'SMEM partition'),
      variablesLayoutChoice(layout(r.gPartStr),'GMEM partition')];
  }
  if (feature === 'make_tiled_tma_atom') {
    const {gp,r}=variablesTmaResult(a);
    return [...object('CopyAtom (TMA)'),variablesLayoutChoice(new Layout(gp.shape,r.tmaTensorStride),'GMEM coordinate tensor')];
  }
  throw new Error('This URL does not contain a reusable object. Paste its layout directly.');
}

function variablesTmaResult(a) {
    const gp=parseLayout(a[1]), sm=tmaParseSmemField(a[3]), sp=parseLayout(sm.layoutStr), bits=DTYPE_BITS[a[0]];
    if (!bits) throw new Error('Invalid TMA dtype.');
    const sw=tmaSwizzleInfo(sm.sw || (a[2] === 'none' ? null : parseSwizzleSpec(a[2])),bits);
    const t=mtcParseTiler(a[4]); t.strides=t.strides.map(x=>x === null ? 1 : x);
    if (gp.shape.length !== 2 || gp.shape.some(Array.isArray) || ![2,3].includes(sp.shape.length)) throw new Error('TMA requires flat rank-2 GMEM and rank-2 or staged rank-3 SMEM.');
    const r=tmaComputeAtom({gShape:gp.shape,gStride:gp.stride,elemBits:bits,swInfo:sw,sFlatShape:flatten(sp.shape.slice(0,2)),sFlatStride:flatten(sp.stride.slice(0,2)),tiler:t});
    return {gp,r};
}

function variablesValidateName(name, exceptId=null) {
  const value=String(name || '').trim();
  if (!value || value.length > 128) throw new Error('Enter a variable name (up to 128 characters).');
  if (variablesState.records.some(row => row.id !== exceptId && row.name === value)) throw new Error('That name is already tracked. Rename or delete it first.');
  return value;
}

function variablesPersist() {
  try {
    localStorage.setItem(VARIABLES_STORAGE_KEY,JSON.stringify({version:1,records:variablesState.records}));
    variablesState.storageWarning='';
  } catch (_) { variablesState.storageWarning='Browser storage is unavailable. Variables are kept for this session.'; }
}

function variablesRestore() {
  variablesState.records=[];
  try {
    const saved=JSON.parse(localStorage.getItem(VARIABLES_STORAGE_KEY) || 'null');
    if (!saved) return;
    if (saved.version !== 1 || !Array.isArray(saved.records)) throw new Error('Invalid saved variables');
    const ids=new Set(), names=new Set();
    for (const row of saved.records) {
      if (!Number.isSafeInteger(row.id) || row.id < 1 || ids.has(row.id) || typeof row.name !== 'string' || !row.name.trim() || row.name.length > 128 || names.has(row.name) || typeof row.value !== 'string') throw new Error('Invalid saved variable');
      // Re-infer types instead of trusting stored HTML or type labels.
      const valid=variablesClipboardChoices(row.value).find(x=>x.type === row.type && x.value === row.value);
      if (!valid) throw new Error('Invalid saved object');
      ids.add(row.id); names.add(row.name);
      variablesState.records.push({id:row.id,name:row.name,type:valid.type,value:valid.value});
    }
  } catch (_) { variablesState.storageWarning='Saved variables could not be read. You can still track variables for this session.'; }
}

function variablesMessage(message, error=false) {
  const host=document.getElementById('variables-status');
  host.textContent=message || variablesState.storageWarning;
  host.classList.toggle('is-error',error);
}

function variablesAdd(name, choice) {
  const label=variablesValidateName(name);
  if (!choice) throw new Error('Choose an object to register.');
  const id=variablesState.records.reduce((n,row)=>Math.max(n,row.id),0)+1;
  variablesState.records.push({id,name:label,type:choice.type,value:choice.value});
  variablesPersist(); variablesRender();
  variablesMessage(variablesState.storageWarning || `Added ${label}.`);
  document.getElementById('variables-name').value='';
  variablesState.pending=[];
  document.getElementById('variables-output-group').hidden=true;
}

async function variablesLoadPasteboard() {
  const button=document.getElementById('variables-load');
  if (button.disabled) return;
  button.disabled=true;
  try {
    const name=variablesValidateName(document.getElementById('variables-name').value);
    variablesState.pending=[]; document.getElementById('variables-output-group').hidden=true;
    const text=await readPasteboardText('Paste a layout or an exported visualization URL:');
    if (text === null || text === undefined) return;
    const choices=variablesClipboardChoices(text);
    if (choices.length === 1) variablesAdd(name,choices[0]);
    else {
      variablesState.pending=choices;
      const select=document.getElementById('variables-output');
      select.replaceChildren();
      choices.forEach((choice,i)=>{ const option=document.createElement('option'); option.value=String(i); option.textContent=`${choice.label} — ${choice.type}`; select.appendChild(option); });
      select.value='0';
      document.getElementById('variables-output-group').hidden=false;
      variablesMessage('Choose which result to register, then click Add variable.');
    }
  } catch (e) { variablesMessage(e.message,true); }
  finally { button.disabled=false; }
}

function variablesAddSelected() {
  try { variablesAdd(document.getElementById('variables-name').value,variablesState.pending[Number(document.getElementById('variables-output').value)]); }
  catch (e) { variablesMessage(e.message,true); }
}

async function variablesCopy(id) {
  const row=variablesState.records.find(x=>x.id === id);
  if (!row) return;
  try { await navigator.clipboard.writeText(row.value); variablesMessage(`Copied ${row.name}${row.type === 'Layout' ? ' layout' : ''}.`); }
  catch (_) { prompt(row.type === 'Layout' ? 'Copy this layout:' : 'Copy this encoding URL:',row.value); }
}

function variablesVisualization(row) {
  if (row.type !== 'Layout') return parsePastedKeyParam(row.value);
  if (/^Sw</.test(row.value)) {
    const sm=tmaParseSmemField(row.value);
    return {feature:'swizzle',inputs:[sm.layoutStr,`${sm.sw.B},${sm.sw.M},${sm.sw.S}`]};
  }
  return {feature:'layout',inputs:[row.value]};
}

function variablesVisualize(id) {
  const row=variablesState.records.find(x=>x.id === id);
  if (!row) return;
  try {
    const tab=addOuterTab();
    applyKeyParam(tab,variablesVisualization(row));
    setOuterTabName(tab,row.name);
    toggleVariablesPanel(false);
  } catch(e) { variablesMessage(e.message,true); }
}

// Summaries use the same constructor computations and result names as the tabs.
// All text is assigned with textContent, including saved names and layout strings.
function variablesPreviewText(row) {
  if (row.type === 'Layout') return row.value;
  const parsed=parsePastedKeyParam(row.value), a=parsed.inputs;
  const fmt=L=>formatLayoutStr(L.shape,L.stride);
  if (row.type === 'TiledMMA') return gemmMmaSummary(mtmDecodeConstructor(a));
  if (row.type === 'TiledCopy') return mtcDecodedSummary(mtcDecodeURL(parsed))
    .replace(/<\/div>/g,'\n').replace(/<[^>]*>/g,'').replace(/&gt;/g,'>').replace(/&lt;/g,'<').replace(/&amp;/g,'&').trim();
  if (row.type === 'MMAAtom') {
    const m=mmaDecodeAtomInputs(a).atom;
    return `MMA_Atom<${m.op.label}>\nThrID = ${fmt(m.thrId)}\nShape MNK = (${m.shapeMNK})\n`+
      ['A','B','C'].map(k=>`tv_layout_${k} = ${m[k].str}`).join('\n');
  }
  if (row.type === 'CopyAtom') {
    const c=mcaDecodeAtomURL(row.value), op=MCA_OPS[c.opKey];
    if (op.ldsm) {
      const m=mcaLdmatrixAtom(op.ldsm,DTYPE_BITS[c.dtype],c.nm,c.trans);
      return `Copy_Atom<${op.label}.x${c.nm}${c.trans?'.trans':''}, ${c.dtype}>\nThrID = ${fmt(m.thrId)}\nValLayoutSrc = ${m.srcStr}\nValLayoutDst = ${m.dstStr}\ntile = (${m.tile.shape})`;
    }
    const n=c.numBits/DTYPE_BITS[c.dtype];
    return `Copy_Atom<${op.label}<${c.numBits}b>, ${c.dtype}>\nelements_to_copy = ${n}\nThrID = 1:0\nValLayoutSrc = (1,${n}):(0,1)\nValLayoutDst = (1,${n}):(0,1)`;
  }
  if (row.type === 'TV Layout') {
    const t=mtvDecodeTVLayout(row.value);
    return `TV Layout = ${fmt(t.tv)}\nTiler MN = (${t.tile})`;
  }
  if (row.type === 'Swizzle') {
    const sw=swParse(a[1]);
    return `Swizzle<${sw.B}, ${sw.M}, ${sw.S}> applied to ${a[0]}`;
  }
  // TMA's constructor also returns its coordinate tensor.
  const {r}=variablesTmaResult(a);
  return `CopyAtom (TMA)\nTMA box = (${r.box.map(b=>b.extent).join(', ')})`+
    `, ${r.boxSize} elements = ${r.numBitsPerTma/8} B per instruction\nnum_bits_per_tma = ${r.numBitsPerTma}`+
    `\n${r.instructions} TMA instruction${r.instructions===1?'':'s'} per tile\nGMEM layout = ${a[1]}\nSMEM layout = ${a[3]}\nCTA tiler = ${a[4]}`;
}

function variablesRename(id, name) {
  const row=variablesState.records.find(x=>x.id === id);
  if (!row) return;
  row.name=variablesValidateName(name,id);
  variablesState.editingId=null;
  variablesPersist(); variablesRender(); variablesMessage();
}

function variablesDelete(id) {
  variablesState.records=variablesState.records.filter(row=>row.id !== id);
  if (variablesState.editingId === id) variablesState.editingId=null;
  variablesPersist(); variablesRender(); variablesMessage();
}

function variablesRender() {
  const body=document.getElementById('variables-rows'); body.replaceChildren();
  document.getElementById('variables-empty').hidden=variablesState.records.length > 0;
  document.getElementById('variables-count').textContent=String(variablesState.records.length);
  const button=(text,action)=>{const el=document.createElement('button');el.type='button';el.className='btn';el.textContent=text;el.addEventListener('click',action);return el;};
  let editingInput=null;
  for (const row of variablesState.records) {
    const tr=document.createElement('tr'), name=document.createElement('td'), type=document.createElement('td'), actions=document.createElement('td');
    if (variablesState.editingId === row.id) {
      const input=document.createElement('input'); input.type='text'; input.value=row.name; input.maxLength=128; input.setAttribute('aria-label','Variable name');
      name.appendChild(input); editingInput=input;
      actions.appendChild(button('Save',()=>{try{variablesRename(row.id,input.value);}catch(e){variablesMessage(e.message,true);}}));
      actions.appendChild(button('Cancel',()=>{variablesState.editingId=null;variablesRender();}));
      input.addEventListener('keydown',e=>{if(e.key === 'Enter'){e.preventDefault();try{variablesRename(row.id,input.value);}catch(err){variablesMessage(err.message,true);}}});
    } else {
      name.textContent=row.name; name.title=row.name;
      actions.appendChild(button('Copy to Pasteboard',()=>variablesCopy(row.id)));
      actions.appendChild(button('Rename',()=>{variablesState.editingId=row.id;variablesRender();}));
      actions.appendChild(button('Delete',()=>variablesDelete(row.id)));
      actions.appendChild(button('Visualize',()=>variablesVisualize(row.id)));
      const preview=document.createElement('span'); preview.className='variables-preview-anchor';
      const trigger=button('Preview',()=>{variablesState.previewId=variablesState.previewId===row.id?null:row.id;variablesRender();});
      trigger.setAttribute('aria-expanded',String(variablesState.previewId===row.id));
      preview.appendChild(trigger);
      if (variablesState.previewId===row.id) {
        const popup=document.createElement('div'); popup.className='variables-preview'; popup.setAttribute('role','region'); popup.setAttribute('aria-label',`${row.name} preview`);
        const text=document.createElement('pre');
        try { text.textContent=variablesPreviewText(row); } catch(e) {text.textContent=e.message;}
        popup.appendChild(text);
        popup.appendChild(button('Close preview',()=>{variablesState.previewId=null;variablesRender();}));
        preview.appendChild(popup);
      }
      actions.appendChild(preview);
    }
    type.textContent=row.type; tr.appendChild(name);tr.appendChild(type);tr.appendChild(actions);body.appendChild(tr);
  }
  if (editingInput) { editingInput.focus(); editingInput.select(); }
}

function toggleVariablesPanel(open) {
  const panel=document.getElementById('variables-panel');
  const show=typeof open === 'boolean' ? open : panel.hidden;
  panel.hidden=!show;
  document.getElementById('variables-toggle').setAttribute('aria-expanded',String(show));
  if (show) document.getElementById('variables-name').focus();
  else document.getElementById('variables-toggle').focus();
}

function initializeVariables() {
  if (variablesState.initialized) return;
  variablesState.initialized=true;
  variablesRestore(); variablesRender(); variablesMessage();
  document.getElementById('variables-toggle').addEventListener('click',()=>toggleVariablesPanel());
  document.getElementById('variables-close').addEventListener('click',()=>toggleVariablesPanel(false));
  document.getElementById('variables-form').addEventListener('submit',e=>{e.preventDefault();variablesLoadPasteboard();});
  document.getElementById('variables-add-output').addEventListener('click',variablesAddSelected);
  document.addEventListener('keydown',e=>{if(e.key === 'Escape' && !document.getElementById('variables-panel').hidden) toggleVariablesPanel(false);});
}
