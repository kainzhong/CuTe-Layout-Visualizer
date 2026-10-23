// Global Variables drawer: real controller, in-memory browser APIs, and the
// existing CuTeDSL oracle for result extraction. No external DOM dependency.
'use strict';
const fs=require('fs');
const path=require('path');
const {loadVisualizer,ROOT}=require('./harness');
const V=loadVisualizer();
const cases=JSON.parse(fs.readFileSync(path.join(__dirname,'cases.json'),'utf8'));
const refs=JSON.parse(fs.readFileSync(path.join(__dirname,'reference.json'),'utf8'));
const results={pass:0,fail:0,failures:[]};
function check(label,condition) { if(condition) results.pass++; else {results.fail++;results.failures.push(label);} }
async function test(label,fn) {try{await fn();}catch(e){results.fail++;results.failures.push(`${label}: ${e.message}`);} }
const key=(feature,...inputs)=>`?key=${encodeURIComponent([feature,...inputs].join('-'))}`;
const same=(label,actual,expected)=>check(label,String(actual).replace(/\s+/g,'')===String(expected).replace(/\s+/g,''));

async function main() {
  await test('raw layouts and compound types',()=>{
    for (const text of ['(4,8):(1,4)','((2,2,2)):((1,2,4))','(4,5):(1@0,1@1)','():()']) {
      const [r]=V.variablesClipboardChoices(text);
      check('raw Layout type',r.type==='Layout'); same('raw layout copied without a URL',r.value,text);
    }
    for (const [url,type] of [
      [key('tv','2','(4,8):(8,1)','(2,2):(1,2)'),'TV Layout'],
      [key('swizzle','(8,8):(1,8)','3,3,3'),'Swizzle'],
      [key('make_mma_atom','f16bf16','half_t','float',16),'MMAAtom'],
      [key('make_copy_atom','ldmatrix',128,'half_t',4,0),'CopyAtom'],
      [key('make_tiled_mma','f16bf16','half_t','float',16,'(2,2,1)','na'),'TiledMMA'],
      [key('make_tiled_copy','universal',32,'half_t','(4,2):(1,4)','(4,2)'),'TiledCopy'],
      [key('make_tiled_copy_tv','universal',32,'half_t','(4,1):(1,0)','(1,2):(0,1)'),'TiledCopy'],
    ]) {
      const [r]=V.variablesClipboardChoices(url);
      check(`type ${type}`,r.type===type);check('compound payload is a usable encoding URL',V.parsePastedKeyParam(r.value).feature===V.parsePastedKeyParam(url).feature);
    }
  });
  await test('layout result extraction vs committed CuTeDSL oracle',()=>{
    for(const c of cases.index) same(`Index/${c.id}`,V.variablesClipboardChoices(key('index',c.a,c.index))[0].value,refs.index[c.id].layout.str);
    for(const c of cases.make_fragment_abc.filter(c=>!c.slice)) {
      const value=V.variablesClipboardChoices(key('make_fragment_abc',c.which,c.op,c.ab||'na',c.acc||'na',c.k,c.partition))[0];
      check('fragment is Layout',value.type==='Layout');same(`fragment/${c.id}`,value.value,refs.make_fragment_abc[c.id].str);
    }
    for(const c of cases.retile) {
      const ref=refs.retile[c.id];
      same(`retile/${c.id}`,V.variablesClipboardChoices(key('retile',c.tv,`(${c.tile})`,ref.atom_values,c.dtype,c.input,0))[0].value,ref.str);
    }
    for(const c of cases.partition_abc) {
      const choice=V.variablesClipboardChoices(key('partition_abc',c.which,c.op,c.ab,c.acc,c.k,c.atom_layout,c.perm||'na',0,c.tensor))[0];
      same(`partition_abc/${c.id}`,choice.value,refs.partition_abc[c.id].layout.str);
    }
    for(const c of cases.partition_sd.filter(c=>c.copy_url)) {
      const nested=encodeURIComponent(c.copy_url).replace(/-/g,'%2D');
      same(`partition_sd/${c.id}`,V.variablesClipboardChoices(key('partition_sd','url',c.side,nested,0,c.tensor))[0].value,refs.partition_sd[c.id].layout.str);
    }
    for(const c of cases.local_tile) {
      same(`local_tile/${c.id}`,V.variablesClipboardChoices(key('local_tile',c.a,c.tiler,c.coord,c.proj||''))[0].value,(refs.local_tile[c.id].layout || refs.local_tile[c.id]).str);
    }
  });
  await test('multiple results have explicit choices',()=>{
    const z=V.variablesClipboardChoices(key('zipped_divide','(8,8):(1,8)','(2,2)'));
    check('zipped variants are separate Layout choices',z.length===3&&z.every(x=>x.type==='Layout'));
    const c=cases.tma_atom[0];
    const tma=V.variablesClipboardChoices(key('make_tiled_tma_atom',c.dtype,c.gmem,c.swizzle,c.smem,c.tiler));
    check('TMA atom and coordinate layout can be registered separately',tma.length===2&&tma[0].type==='CopyAtom (TMA)'&&tma[1].type==='Layout');
    for (const c of cases.tma_partition.slice(0,3)) {
      const r=refs.tma_partition[c.id];
      const out=V.variablesClipboardChoices(key('tma_partition',r.num_val,c.dtype,c.swizzle,r.smem_grouped.replace(/^S</,'Sw<').replace(/o/g,' o '),r.gmem_grouped));
      check('both TMA partition outputs are Layout',out.length===2&&out.every(x=>x.type==='Layout'));
      same('TMA GMEM partition matches CuTeDSL',out[1].value,r.tAgA);
    }
  });

  await test('gemm has no returned variable layout',()=>{
    const c=cases.gemm[0];
    const url=key('gemm',...c.mma,c.a,c.b,c.c,'same',0);
    const rows=V.variablesClipboardChoices(url), row=rows[0];
    check('gemm type is None',rows.length===1&&row.type==='None');
    check('gemm preserves its invocation URL instead of extracting C',V.parsePastedKeyParam(row.value).feature==='gemm');
    check('gemm preview explains in-place output',V.variablesPreviewText(row).startsWith('Return type: None (gemm writes into D).'));
    check('gemm can still reopen the invocation',V.variablesVisualization(row).feature==='gemm');
  });
  await test('previews and TV import preserve exact mappings',()=>{
    const urls=[key('make_mma_atom','f16bf16','half_t','float',16),key('make_copy_atom','ldmatrix',128,'half_t',4,0),key('make_copy_atom','universal',32,'half_t',1,0),key('make_tiled_mma','f16bf16','half_t','float',16,'(2,2,1)','na'),key('make_tiled_copy','universal',32,'half_t','(4,2):(1,4)','(4,2)'),key('tv','2','(4,8):(8,1)','(2,2):(1,2)'),key('swizzle','(8,8):(1,8)','3,3,3')];
    for(const url of urls) {
      const row=V.variablesClipboardChoices(url)[0];
      const preview=V.variablesPreviewText(row);
      check(`${row.type} preview is text`,preview.length>20&&!preview.includes('<div'));
      check(`${row.type} visualization round trip`,V.variablesVisualization(row).feature===V.parsePastedKeyParam(url).feature);
    }
    const original=key('tv','2','(4,8):(8,1)','(2,2):(1,2)');
    const source=V.mtvDecodeThrVal(original);
    check('TV method 2 preserves original thr/val',source.thr==='(4,8):(8,1)'&&source.val==='(2,2):(1,2)');
    const direct=key('tv','1',V.formatLayoutStr(source.tv.shape,source.tv.stride),'(8,16)');
    const decoded=V.mtvDecodeThrVal(direct), rebuilt=V.make_layout_tv(V.variablesParseLayout(decoded.thr),V.variablesParseLayout(decoded.val));
    for(let t=0;t<32;t++)for(let v=0;v<4;v++)check('primitive TV factorization is pointwise exact',source.tv.call([t,v])===rebuilt.layout_tv.call([t,v]));
    let rejected=false;try{V.mtvDecodeThrVal(key('tv','1','(4,2):(0,1)','(4,2)'));}catch(_){rejected=true;}
    check('unrepresentable TV rejects instead of substituting a mapping',rejected);
    const sw=V.variablesVisualization({type:'Layout',value:'Sw<3,3,3> o (8,8):(1,8)'});
    check('composed Layout routes to a valid Swizzle visualization',!!V.swParse(sw.inputs[1]));
  });

  const html=fs.readFileSync(path.join(ROOT,'index.html'),'utf8');
  const nodes=new Map(), storage=new Map(), listeners={};
  let clipboard='', copied='', denied=false, quota=false, promptValue;
  function el(tag='div') {
    const classes=new Set();
    return {tagName:tag.toUpperCase(),children:[],listeners:{},dataset:{},value:'',textContent:'',hidden:false,disabled:false,
      classList:{toggle(c,on){if(on===undefined)on=!classes.has(c);on?classes.add(c):classes.delete(c);},contains:c=>classes.has(c)},
      appendChild(child){this.children.push(child);return child;},replaceChildren(...children){this.children=children;},
      addEventListener(type,fn){(this.listeners[type]||=([])).push(fn);},
      setAttribute(k,v){this[k]=v;},getAttribute(k){return this[k];},focus(){doc.activeElement=this;},select(){},
      fire(type,event={}){for(const fn of this.listeners[type]||[])fn({preventDefault(){},...event});}};
  }
  for(const m of html.matchAll(/<[^>]+\bid="([^"]+)"[^>]*>/g)) {const node=el();node.hidden=/\bhidden\b/.test(m[0]);nodes.set(m[1],node);}
  const doc={getElementById(id){if(!nodes.has(id))throw new Error(`undeclared ID ${id}`);return nodes.get(id);},createElement:el,
    addEventListener(type,fn){(listeners[type]||=([])).push(fn);},activeElement:null};
  V.document=doc;
  V.localStorage={getItem:k=>storage.get(k)||null,setItem(k,v){if(quota)throw new Error('quota');storage.set(k,v);}};
  V.navigator.clipboard={async readText(){return clipboard;},async writeText(text){if(denied)throw new Error('denied');copied=text;}};
  V.prompt=(message,value)=>{promptValue=value;return null;};
  const node=id=>doc.getElementById(id);
  const load=async(name,value)=>{node('variables-name').value=name;clipboard=value;await V.variablesLoadPasteboard();};
  const tick=()=>new Promise(resolve=>setImmediate(resolve));
  await test('initialization, open and close',()=>{
    V.initializeVariables();V.initializeVariables();
    check('panel starts closed',node('variables-panel').hidden);
    node('variables-toggle').fire('click');
    check('header button opens the right drawer',!node('variables-panel').hidden&&node('variables-toggle')['aria-expanded']==='true');
    check('name receives focus',doc.activeElement===node('variables-name'));
    for(const fn of listeners.keydown||[])fn({key:'Escape'});
    check('Escape closes drawer',node('variables-panel').hidden&&doc.activeElement===node('variables-toggle'));
    node('variables-toggle').fire('click');node('variables-close').fire('click');check('Close button works',node('variables-panel').hidden);
  });
  await test('load, copy and row actions',async()=>{
    await load('tCrA','((2,2,2),4,2):((1,2,4),16,8)');
    check('new row has Name / Layout / actions',node('variables-rows').children[0].children[0].textContent==='tCrA'&&node('variables-rows').children[0].children[1].textContent==='Layout');
    node('variables-rows').children[0].children[2].children[0].fire('click');await tick();
    same('Layout copy is plain shape:stride',copied,'((2,2,2),4,2):((1,2,4),16,8)');
    await load('mma',key('make_tiled_mma',...V.GEMM_DEFAULT_MMA));
    const row=V.variablesState.records.find(x=>x.name==='mma');await V.variablesCopy(row.id);
    check('TiledMMA copy decodes through existing API',V.mtmDecodeConstructor(V.parsePastedKeyParam(copied).inputs).tiled.warps===4);
    await load('mma','(2,2):(1,2)');check('duplicate names do not overwrite',V.variablesState.records.length===2&&node('variables-status').classList.contains('is-error'));
    const first=node('variables-rows').children[0];first.children[2].children[1].fire('click');
    const edit=node('variables-rows').children[0];edit.children[0].children[0].value='rA';edit.children[2].children[0].fire('click');
    check('Rename updates the named variable',V.variablesState.records[0].name==='rA');
    let rejected=false;try{V.variablesRename(row.id,'rA');}catch(_){rejected=true;}check('Rename rejects duplicate names',rejected);
    node('variables-rows').children[0].children[2].children[2].fire('click');check('Delete removes row and updates count',V.variablesState.records.length===1&&node('variables-count').textContent==='1');
  });
  await test('Visualize and Preview buttons',()=>{
    const row=V.variablesState.records[0];
    let applied, title;
    V.addOuterTab=()=> 'newtab';V.applyKeyParam=(id,p)=>{applied={id,p};};V.setOuterTabName=(id,name)=>{title=name;};
    const actions=node('variables-rows').children[0].children[2];
    actions.children[3].fire('click');
    check('Visualize opens a new app tab and populates it',applied.id==='newtab'&&applied.p.feature===V.variablesVisualization(row).feature&&title===row.name);
    check('Visualize closes the drawer',node('variables-panel').hidden);
    actions.children[4].children[0].fire('click');
    let anchor=node('variables-rows').children[0].children[2].children[4];
    check('Preview appears under its button with result text',anchor.children[1].children[0].textContent===V.variablesPreviewText(row));
    anchor.children[1].children[1].fire('click');
    anchor=node('variables-rows').children[0].children[2].children[4];
    check('Close preview removes popup',anchor.children.length===1);
  });
  await test('None type persists and copies the invocation URL',async()=>{
    const c=cases.gemm[0], before=V.variablesState.records.slice();
    await load('gemm_call',key('gemm',...c.mma,c.a,c.b,c.c,'same',0));
    const row=V.variablesState.records.at(-1);
    check('gemm row shows None',node('variables-rows').children.at(-1).children[1].textContent==='None');
    await V.variablesCopy(row.id);
    check('None copies the invocation URL',copied===row.value);
    V.variablesRestore();
    check('None type survives reload',V.variablesState.records.at(-1).type==='None');
    V.variablesState.records=before;V.variablesPersist();V.variablesRender();
  });
  await test('multiple outputs, malformed input and cancellation',async()=>{
    await load('parts',key('zipped_divide','(8,8):(1,8)','(2,2)'));
    check('ambiguous output needs a choice',!node('variables-output-group').hidden&&V.variablesState.records.length===1);
    node('variables-output').value='1';node('variables-add-output').fire('click');
    check('chosen layout registers as Layout',V.variablesState.records[1].type==='Layout'&&node('variables-output-group').hidden);
    const count=V.variablesState.records.length;
    await load('bad','not a layout');check('malformed paste adds no row',V.variablesState.records.length===count&&node('variables-status').classList.contains('is-error'));
    const realRead=V.readPasteboardText;V.readPasteboardText=async()=>null;
    await V.variablesLoadPasteboard();V.readPasteboardText=realRead;
    check('cancelled paste adds nothing and resets button',V.variablesState.records.length===count&&!node('variables-load').disabled);
  });
  await test('persistence, safe text, blocked browser APIs',async()=>{
    await load('<img src=x onerror=alert(1)>','((2,2,2)):((1,2,4))');
    const last=node('variables-rows').children.at(-1).children[0];
    check('names are plain text, not HTML',last.textContent==='<img src=x onerror=alert(1)>'&&last.children.length===0);
    const before=JSON.stringify(V.variablesState.records);V.variablesState.records=[];V.variablesRestore();
    check('reload restores names, types and payloads',JSON.stringify(V.variablesState.records)===before);
    denied=true;await V.variablesCopy(V.variablesState.records[0].id);denied=false;
    check('clipboard denial has a copyable fallback',promptValue===V.variablesState.records[0].value);
    quota=true;await load('session_only','(4,4):(1,4)');quota=false;
    check('storage denial keeps the variable in memory',V.variablesState.records.some(x=>x.name==='session_only')&&node('variables-status').textContent.includes('session'));
    storage.set(V.VARIABLES_STORAGE_KEY,'{broken');V.variablesRestore();check('corrupt storage does not crash',V.variablesState.storageWarning.length>0);
  });
  check('header says Show Variable Table',html.includes('>Show Variable Table <span'));
  check('header integration initializes library before creating tabs',html.includes('src="variables.js"')&&html.indexOf('initializeVariables();')<html.indexOf('const firstTabId = addOuterTab();'));
  const css=fs.readFileSync(path.join(ROOT,'style.css'),'utf8');check('right drawer respects hidden and mobile sizing',css.includes('.variables-panel[hidden]')&&css.includes('width: min(560px, 100vw)'));
  return results;
}
module.exports = {runVariablesTests:main};
if (require.main === module) main().then(r => {
  for (const failure of r.failures) console.log(`FAIL ${failure}`);
  console.log(`${r.fail ? 'FAIL' : 'PASS'} ${r.pass}/${r.pass+r.fail} Variables checks`);
  process.exitCode = r.fail ? 1 : 0;
}).catch(e => { console.error(e); process.exitCode=1; });
