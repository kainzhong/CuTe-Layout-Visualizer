// Session lifecycle through real templates/renderers, strict select values,
// fresh VM contexts for reloads, and an in-memory localStorage. No dependencies.
'use strict';
const vm = require('vm');
const {makeContext} = require('./dom_smoke');

function browser(storage, {search='', navigation='reload', blocked=false} = {}) {
  const {ctx: V} = makeContext();
  const nodes = new Map(), docEvents = {}, windowEvents = {};
  const decode = value => String(value).replace(/&(amp|lt|gt|quot|#39|apos|nbsp);/g,(_,v)=>({amp:'&',lt:'<',gt:'>',quot:'"','#39':"'",apos:"'",nbsp:' '})[v]);
  const attrs = text => Object.fromEntries([...text.matchAll(/([\w-]+)="([^"]*)"/g)].map(m=>[m[1],decode(m[2])]));
  function el(tag='div', attributes={}) {
    const classes = new Set(), attributesMap = {}, listeners = {};
    let id='', value='', html='', className='';
    const node = {tagName:tag.toUpperCase(),style:{},dataset:{},children:[],options:[],textContent:'',checked:false,open:false,
      type:tag==='input'?'text':'',disabled:false,
      classList:{add:c=>classes.add(c),remove:c=>classes.delete(c),contains:c=>classes.has(c),
        toggle(c,on){if(on===undefined)on=!classes.has(c);on?classes.add(c):classes.delete(c);}},
      setAttribute(k,v){attributesMap[k]=String(v);if(k==='class')this.className=v;else if(k==='id')this.id=v;
        else if(k.startsWith('data-'))this.dataset[k.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=String(v);},
      getAttribute(k){return k==='class'?this.className:k==='id'?this.id:attributesMap[k]??null;},
      hasAttribute:k=>Object.hasOwn(attributesMap,k),removeAttribute:k=>delete attributesMap[k],
      appendChild(child){this.children.push(child);child.parentNode=this;return child;},
      remove(){if(this.parentNode)this.parentNode.children=this.parentNode.children.filter(c=>c!==this);
        for(const [key] of nodes)if(key===this.id || (this.id.endsWith('-panel')&&key.startsWith(this.id.slice(0,-5))))nodes.delete(key);},
      addEventListener(type,fn){(listeners[type]||=[]).push(fn);},
      dispatchEvent(event){event.target=this;const handler=attributesMap['on'+event.type];
        if(handler)vm.runInContext(`(function(event){${handler}})`,V).call(this,event);
        for(const fn of listeners[event.type]||[])fn(event);
        if(event.bubbles)for(const fn of docEvents[event.type]||[])fn(event);return true;},
      click(){if(this.onclick)this.onclick();this.dispatchEvent({type:'click',bubbles:true,stopPropagation(){}});},
      querySelectorAll(selector){
        if(this.id==='outer-tab-buttons')return selector.includes('outer-tab')?this.children:[];
        if(this.id==='outer-panels')return selector.includes('outer-panel')?this.children:[];
        const root=this.id.endsWith('-panel')?this:null;
        if(root){
          const prefix=this.id.slice(0,-5);
          const owned=[...nodes.values()].filter(n=>n.id.startsWith(prefix));
          if(selector==='input[id],select[id],textarea[id]')return owned.filter(n=>['INPUT','SELECT','TEXTAREA'].includes(n.tagName));
          if(selector==='details')return this.details;
          if(selector==='.tab-bar .tab')return this.tabs;
          if(selector==='.panel')return owned.filter(n=>n.id.startsWith(prefix+'tab-'));
          if(selector==='.tab-scope-btn')return this.scopes;
        }
        if(this.isTabBar)return selector==='.tab.active'?this.root.tabs.filter(n=>n.classList.contains('active')):[];
        return [];
      },
      querySelector(selector){
        if(this.id.endsWith('-panel')){
          if(selector==='.tab.active')return this.tabs.find(n=>n.classList.contains('active'))||null;
          if(selector==='.tab-nav')return this.nav;
          if(selector==='.tab-bar')return this.bar;
          if(selector==='.panel.active .error-msg')return nodes.get(this.id.slice(0,-5)+'layout-error');
        }
        if(this.isTabBar&&selector.startsWith('.tab'))return this.root.tabs.find(n=>selector.includes('active')?n.classList.contains('active'):n.getAttribute('data-scope')===/data-scope="([^"]+)"/.exec(selector)?.[1])||null;
        return null;
      },
    };
    Object.defineProperties(node,{
      id:{get:()=>id,set(v){if(id)nodes.delete(id);id=String(v);if(id)nodes.set(id,node);}},
      className:{get:()=>[...classes].join(' '),set(v){className=String(v);classes.clear();for(const c of className.split(/\s+/))if(c)classes.add(c);}},
      value:{get:()=>value,set(v){v=String(v);value=node.tagName==='SELECT'&&!node.options.some(o=>o.value===v)?'':v;}},
      innerHTML:{get:()=>html,set(v){html=String(v);
        if(node.tagName==='SELECT'){
          node.options=[...html.matchAll(/<option\b([^>]*)>([\s\S]*?)<\/option>/g)].map(m=>({value:attrs(m[1]).value??decode(m[2]),selected:/\bselected\b/.test(m[1])}));
          value=(node.options.find(o=>o.selected)||node.options[0])?.value||'';return;
        }
        if(node.id.endsWith('-svg')||node.id.endsWith('-svg-host'))return;
        for(const m of html.matchAll(/<([a-z][\w-]*)\b([^>]*)>/g)){
          const a=attrs(m[2]);if(!a.id)continue;
          const child=nodes.get(a.id)||el(m[1],a);
          if(m[1]==='textarea')child.value=decode(html.slice(m.index+m[0].length).split('</textarea>')[0]).trim();
          if(m[1]==='select')child.innerHTML=html.slice(m.index+m[0].length).split('</select>')[0];
          if(m[1]==='details')child.open=/\bopen\b/.test(m[2]);
        }
        if(node.id.endsWith('-panel')){
          node.details=[...html.matchAll(/<details\b([^>]*)>/g)].map(m=>{const a=attrs(m[1]);const d=a.id?nodes.get(a.id):el('details');d.open=/\bopen\b/.test(m[1]);return d;});
          node.tabs=[...html.matchAll(/<div\b([^>]*data-tab="[^>]+)>/g)].map(m=>el('div',attrs(m[1])));
          node.scopes=[...html.matchAll(/<button\b([^>]*class="tab-scope-btn[^>]+)>/g)].map(m=>el('button',attrs(m[1])));
          node.nav=el();node.bar=el();node.bar.isTabBar=true;node.bar.root=node;
        }
      }},
    });
    for(const [k,v]of Object.entries(attributes)){node.setAttribute(k,v);if(k==='type')node.type=v;}
    if(Object.hasOwn(attributes,'value'))node.value=attributes.value;
    return node;
  }
  el('div',{id:'outer-tab-buttons'});el('div',{id:'outer-panels'});
  V.document={getElementById:id=>nodes.get(id)||null,createElement:tag=>el(tag),
    querySelectorAll(selector){if(selector==='.outer-tab'||selector==='#outer-tab-buttons .outer-tab')return nodes.get('outer-tab-buttons').children;
      if(selector==='.outer-panel')return nodes.get('outer-panels').children;return[];},
    addEventListener(type,fn){(docEvents[type]||=[]).push(fn);},removeEventListener(){},visibilityState:'visible'};
  V.addEventListener=(type,fn)=>{(windowEvents[type]||=[]).push(fn);};
  V.Event=class{constructor(type,options={}){this.type=type;Object.assign(this,options);}};
  V.clearTimeout=clearTimeout;V.setTimeout=setTimeout;
  V.performance={getEntriesByType:()=>[{type:navigation}]};
  V.location.search=search;
  V.localStorage={getItem(k){if(blocked)throw Error('blocked');return storage.get(k)||null;},setItem(k,v){if(blocked)throw Error('blocked');storage.set(k,v);}};
  return {V,nodes,fire(type){for(const fn of windowEvents[type]||[])fn();},input(id,value){const n=nodes.get(id);n.value=value;n.dispatchEvent(new V.Event('input',{bubbles:true}));}};
}

async function runSessionTests() {
  const results={pass:0,fail:0,failures:[]};
  const check=(label,condition)=>{if(condition)results.pass++;else{results.fail++;results.failures.push(label);}};
  const test=async(label,fn)=>{try{await fn();}catch(e){results.fail++;results.failures.push(label+': '+e.stack);}};
  const storage=new Map();
  let initial;
  await test('fresh workspace and two real tabs',()=>{
    initial=browser(storage);const {V,nodes}=initial;
    const first=V.initializeTabSession();
    V.setMTM(first,'wg_fp8','float_e4m3_t','float',32,'(2,1,1)','',24,'RMEM','K','K','float_e5m2_t');
    V.switchInnerTab(first,'make_tiled_mma');V.setMtmMode(first,'warp');V.setOuterTabName(first,'My WGMMA');
    nodes.get(first+'-mtm-atom-manual-construction').open=true;
    const discarded=V.addOuterTab(), second=V.addOuterTab();V.closeOuterTab(discarded);
    V.switchInnerTab(second,'layout');V.setL(second,'(4,8):(1,4)');
    V.setOuterTabName(second,'<img src=x onerror=alert(1)>');
    initial.input(second+'-layout-input','unfinished (');
    V.switchOuterTab(first);initial.fire('pagehide');
    const saved=JSON.parse(storage.get(V.TAB_SESSION_STORAGE_KEY));
    check('both tabs persisted in order',saved.tabs.length===2&&saved.tabs[0].name==='My WGMMA');
    check('active tab persisted',saved.active===0);
    check('unrendered invalid edit persisted',saved.tabs[1].controls.find(c=>c.id==='layout-input').value==='unfinished (');
    check('snapshot excludes SVGs',!storage.get(V.TAB_SESSION_STORAGE_KEY).includes('<svg'));
    check('WGMMA constructor was valid before reload',!nodes.get(first+'-mtm-error').textContent);
  });
  await test('hard reload restores constructors, active tab, names and drafts',()=>{
    const previous=JSON.parse(storage.get('cute-layout-visualizer.tabs.v1'));
    const {V,nodes}=browser(storage);V.initializeTabSession();
    check('restored two tabs',nodes.get('outer-tab-buttons').children.length===2);
    check('active outer tab restored',vm.runInContext('activeOuterTabId',V)==='ot1');
    check('active feature and scope restored',nodes.get('ot1-panel').querySelector('.tab.active').getAttribute('data-tab')==='make_tiled_mma'&&nodes.get('ot1-panel').nav.getAttribute('data-scope')==='mma');
    check('dependent WGMMA options restored',V.mtmState.ot1.atom.n===24&&V.mtmState.ot1.atom.aSrc==='RMEM'&&V.mtmState.ot1.atom.bDtype==='float_e5m2_t');
    check('warp view retained',V.mtmState.ot1.mode==='warp');
    check('manual section stays open',nodes.get('ot1-mtm-atom-manual-construction').open);
    check('WGMMA renders after reload',nodes.get('ot1-mtm-perm-c-svg').innerHTML.includes('<svg')&&!nodes.get('ot1-mtm-error').textContent);
    check('draft remains editable',nodes.get('ot2-layout-input').value==='unfinished (');
    check('custom names restored as text',nodes.get('ot1-label').textContent==='My WGMMA'&&nodes.get('ot2-label').textContent==='<img src=x onerror=alert(1)>');
    const current=V.captureTabSession();
    for(let i=0;i<previous.tabs.length;i++) {
      const values=new Map(current.tabs[i].controls.map(field=>[field.id,field]));
      const changed=previous.tabs[i].controls.filter(field=>JSON.stringify(values.get(field.id))!==JSON.stringify(field));
      check('all controls restored for tab '+i+': '+changed.map(field=>field.id).join(','),changed.length===0);
    }
    V.closeOuterTab('ot2');V.saveTabSession();
    check('closed tabs disappear from persistence',JSON.parse(storage.get(V.TAB_SESSION_STORAGE_KEY)).tabs.length===1);
  });
  await test('new shared URL adds a tab and refresh preserves subsequent edits',()=>{
    const search='?key=layout-(16,8):(8,1)';const b=browser(storage,{search,navigation:'navigate'});b.V.initializeTabSession();
    check('shared URL keeps existing workspace',b.nodes.get('outer-tab-buttons').children.length===2);
    const id=vm.runInContext('activeOuterTabId',b.V);b.input(id+'-layout-input','(32,4):(4,1)');b.fire('pagehide');
    const reload=browser(storage,{search});reload.V.initializeTabSession();
    check('reload does not duplicate shared tab',reload.nodes.get('outer-tab-buttons').children.length===2);
    check('reload does not overwrite edits with launch URL',reload.nodes.get('ot2-layout-input').value==='(32,4):(4,1)');
  });
  await test('UTILITY collection and stage survive hard reload',()=>{
    const memory=new Map(), b=browser(memory), id=b.V.initializeTabSession();
    b.V.setSml(id,'epi','hopper','COL_MAJOR','(32,16)','half_t','2','(32,16,3)','(2,1,0)');
    b.nodes.get(id+'-sml-epi-stage-view').value='2';b.V.renderSmemLayout(id,'epi');
    b.V.toggleSmlMode(id,'epi','coord');b.V.toggleSmlMode(id,'epi','value');
    b.V.switchInnerTab(id,'make_smem_layout_epi');b.fire('pagehide');
    const restored=browser(memory);restored.V.initializeTabSession();
    check('UTILITY scope restored',restored.nodes.get('ot1-panel').nav.getAttribute('data-scope')==='utility');
    check('utility overrides and stage restored',restored.nodes.get('ot1-sml-epi-target-input').value==='(32,16,3)'&&restored.nodes.get('ot1-sml-epi-stage-view').value==='2');
    check('utility renders after reload',restored.nodes.get('ot1-sml-epi-dst-svg').innerHTML.includes('<svg')&&!restored.nodes.get('ot1-sml-epi-error').textContent);
    check('utility label toggles restored',restored.V.smlState.epi.ot1.mode.size===1&&restored.V.smlState.epi.ot1.mode.has('coord'));
  });
  await test('domain_offset inputs, scope and label modes survive reload',()=>{
    const memory=new Map(), b=browser(memory), id=b.V.initializeTabSession();
    b.V.setDomainOffset(id,'20 o (3,4):(4,1)','(-1,2)');
    b.V.toggleDomainOffsetMode(id,'dst','coord');
    b.V.switchInnerTab(id,'domain_offset');b.fire('pagehide');
    const restored=browser(memory);restored.V.initializeTabSession();
    check('domain_offset scope restored',restored.nodes.get('ot1-panel').nav.getAttribute('data-scope')==='operations');
    check('signed input survives reload',restored.nodes.get('ot1-dof-coord-input').value==='(-1,2)');
    check('domain_offset labels survive reload',restored.V.domainOffsetState.ot1.dstMode.has('coord'));
    check('domain_offset renders after reload',restored.nodes.get('ot1-dof-dst-svg').innerHTML.includes('<svg')&&!restored.nodes.get('ot1-dof-error').textContent);
  });
  await test('blocked and corrupt storage remain usable',()=>{
    const bad=new Map([['cute-layout-visualizer.tabs.v1','{broken']]);const broken=browser(bad);broken.V.initializeTabSession();
    check('corrupt storage starts with one tab',broken.nodes.get('outer-tab-buttons').children.length===1);
    const blocked=browser(new Map(),{blocked:true});blocked.V.initializeTabSession();blocked.V.addOuterTab();blocked.fire('pagehide');
    check('blocked storage does not prevent adding tabs',blocked.nodes.get('outer-tab-buttons').children.length===2);
  });
  await test('debounced edits save without waiting for unload',async()=>{
    const memory=new Map([['cute-layout-visualizer.variables.v1','existing variables']]);
    const b=browser(memory);const id=b.V.initializeTabSession();b.input(id+'-layout-input','(7,8):(1,7)');
    await new Promise(resolve=>setTimeout(resolve,200));
    const saved=JSON.parse(memory.get(b.V.TAB_SESSION_STORAGE_KEY));check('latest typed value is saved',saved.tabs[0].controls.find(c=>c.id==='layout-input').value==='(7,8):(1,7)');
    check('variables storage is untouched',memory.get('cute-layout-visualizer.variables.v1')==='existing variables');
    b.fire('pagehide');
  });
  return results;
}
module.exports={runSessionTests};
if(require.main===module)runSessionTests().then(r=>{for(const f of r.failures)console.log('FAIL '+f);console.log(`PASS ${r.pass}/${r.pass+r.fail} session checks`);process.exitCode=r.fail?1:0;});
