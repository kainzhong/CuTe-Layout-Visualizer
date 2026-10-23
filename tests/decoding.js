// Pure JS integration tests: the production URL encoder -> consumer decoders
// -> API computations. No controls, clipboard, rendering, DOM shim, or GPU.
'use strict';
const {parseExact, evalAll, fmt}=require('./harness');
const cases=require('./cases.json');
const refs=require('./reference.json');

function runDecodingTests(V, T) {
  const {check,guard}=T;
  const json=x=>JSON.stringify(x);
  const layout=s=>parseExact(V,s);
  const exported=(feature,...args)=>'https://example.test/index.html'+V.encodeKeyParam(feature,...args);
  const decoded=url=>V.parsePastedKeyParam(url);
  const equalLayout=(id,actual,expected)=>{
    check(id,'layout',fmt(V,actual),expected.str);
    check(id,'size',actual.size(),expected.size);
    check(id,'cosize',actual.cosize(),expected.cosize);
    check(id,'every domain point',json(evalAll(V,actual,actual.size())),json(expected.eval));
  };
  const rejects=(id,fn)=>guard(id,()=>{
    let threw=false;try{fn();}catch(_){threw=true;}
    check(id,'rejects invalid input',threw,true);
  });

  // Enforce the test boundary: accidentally calling an importer that renders,
  // reads controls, or touches browser APIs must fail, even under the harness.
  const browserKeys=['document','navigator','localStorage'];
  const saved=browserKeys.map(k=>[k,Object.getOwnPropertyDescriptor(V,k)]);
  try {
    for(const k of browserKeys) Object.defineProperty(V,k,{configurable:true,get(){throw new Error(`${k} used by pure decoding test`);}});

    guard('export URL envelopes',()=>{
      const args=['f16bf16','half_t','float','16','(2, 2, 1)','(32, 32, 16)'];
      const query=V.encodeKeyParam('make_tiled_mma',...args);
      for(const text of [query,query.slice(1),'https://example.test/index.html'+query,'file:///Users/test/CuTe Layout/index.html'+query,'  '+query+'#ignored  ']) {
        const p=decoded(text);
        check('export URL envelopes','feature',p.feature,'make_tiled_mma');
        check('export URL envelopes','exact input strings',json(p.inputs),json(args));
      }
      const a='(8, 16):(1, 8)', b='2:1\n4:1';
      check('multiline input','lossless export/decode',json(decoded(exported('logical_divide',a,b)).inputs),json([a,b]));
    });

    // The atom exported by one API becomes a tiled constructor, whose exported
    // object is consumed by partition_A/B/C and make_fragment_A/B/C.
    for(const c of cases.partition_abc) guard(`MMAAtom -> TiledMMA -> partition_${c.which} -> fragment/${c.id}`,()=>{
      const atomURL=exported('make_mma_atom',c.op,c.ab||'na',c.acc||'na',c.k);
      const ap=decoded(atomURL), atom=V.mmaDecodeAtomInputs(ap.inputs).atom;
      const mmaURL=exported('make_tiled_mma',...ap.inputs,c.atom_layout,c.perm||'na');
      const mma=V.mtmDecodeConstructor(decoded(mmaURL).inputs);
      check(c.id,'atom shape survives constructor handoff',json(mma.atom.shapeMNK),json(atom.shapeMNK));
      const partition=V.pabcComputePartition(mma.atom,mma.atomLayout,mma.perm,c.which,layout(c.tensor),0).partition;
      equalLayout(c.id,partition,refs.partition_abc[c.id].layout);
      const partURL=exported('partition_abc',c.which,...decoded(mmaURL).inputs,0,c.tensor);
      const raw=V.variablesClipboardChoices(partURL)[0];
      check(c.id,'partition clipboard type',raw.type,'Layout');
      equalLayout(c.id+'/raw partition consumer',layout(raw.value),refs.partition_abc[c.id].layout);
      const fragment=V.mfragCompute(layout(raw.value),c.which,c.op,c.k,c.ab||'tfloat32_t',c.acc||'float').fragment;
      equalLayout(c.id+'/fragment',fragment,refs.partition_abc[c.id].fragment);
      const fragURL=exported('make_fragment_abc',c.which,c.op,c.ab||'na',c.acc||'na',c.k,raw.value,0);
      equalLayout(c.id+'/fragment URL consumer',layout(V.variablesClipboardChoices(fragURL)[0].value),refs.partition_abc[c.id].fragment);
    });

    // Both typed constructors feed make_tiled_copy_A/B. Check the distinct SRC
    // and DST maps, so losing ldmatrix transpose/x1/x2/x4 cannot pass unnoticed.
    // FP8 entries in the math corpus have no standalone CopyAtom picker dtype;
    // only test atom exports the UI can produce. FP8 TiledCopy decoding remains
    // covered below by the partition corpus.
    for(const c of cases.make_tiled_copy_ab.filter(c=>V.DTYPE_BITS[c.ab_dtype])) guard(`CopyAtom + TiledMMA -> TiledCopy/${c.id}`,()=>{
      const mmaURL=exported('make_tiled_mma',c.op,c.ab_dtype,c.acc_dtype,c.k,c.atom_layout,c.perm||'na');
      const cpURL=exported('make_copy_atom',c.copy_op,c.bits,c.ab_dtype,c.num_matrices,c.transpose?1:0);
      const cp=V.mcaDecodeAtomURL(cpURL), mp=decoded(mmaURL);
      const url=exported('make_tiled_copy_ab',c.operand,...mp.inputs,cp.opKey,cp.nm,cp.trans?1:0,cp.numBits,'warp','na','src_dst');
      const copy=V.mtcDecodeURL(decoded(url));
      const r=refs.make_tiled_copy_ab[c.id];
      equalLayout(c.id+'/tiled TV',copy.tv,r.tv);
      equalLayout(c.id+'/SRC',V.mcabSideLayout(copy.tv,copy.tile,copy.copyAtom,'src'),r.src);
      equalLayout(c.id+'/DST',V.mcabSideLayout(copy.tv,copy.tile,copy.copyAtom,'dst'),r.dst);
      check(c.id,'dtype survives handoff',copy.dtype,c.ab_dtype);
      check(c.id,'num_matrices survives handoff',cp.nm,c.num_matrices);
      check(c.id,'transpose survives handoff',cp.trans,c.transpose);
    });

    // Decode TiledCopy URLs with optional tails and nested URI encoding, then
    // partition both memory sides using the saved tensor/thread arguments.
    for(const c of cases.partition_sd.filter(c=>c.copy_url)) guard(`TiledCopy -> partition_${c.side}/${c.id}`,()=>{
      const original=decoded(c.copy_url);
      const copyURL=exported(original.feature,...original.inputs);
      const url=exported('partition_sd','url',c.side,V.encodeNestedKeyParam(copyURL),0,c.tensor);
      const p=decoded(url);
      check(c.id,'nested URL method',p.method,'url');
      const cp=V.mtcDecodeURL(decoded(decodeURIComponent(p.inputs[1])));
      const part=V.psdComputePartition(cp.tv,cp.tiler,layout(p.inputs[3]),cp.copyAtom,Number(p.inputs[2]),p.inputs[0]).partition;
      equalLayout(c.id,part,refs.partition_sd[c.id].layout);
      equalLayout(c.id+'/saved Layout',layout(V.variablesClipboardChoices(url)[0].value),refs.partition_sd[c.id].layout);
    });

    // TV Layout is a typed object, not an ordinary tensor layout. Its exported
    // thr/val inputs can be decoded and reused to construct a TiledCopy.
    for(const c of cases.make_layout_tv) guard(`TV Layout -> make_tiled_copy_tv/${c.id}`,()=>{
      const tvURL=exported('tv','2',c.thr,c.val);
      const tv=V.mtvDecodeThrVal(tvURL);
      check(c.id,'source thr exact',tv.thr,c.thr);
      check(c.id,'source val exact',tv.val,c.val);
      const copyURL=exported('make_tiled_copy_tv','universal',c.bits,c.dtype,tv.thr,tv.val);
      const copy=V.mtcDecodeURL(decoded(copyURL));
      equalLayout(c.id,copy.tv,refs.make_layout_tv[c.id].layout_tv);
      check(c.id,'tile survives handoff',json(copy.tile),json(tv.tile));
      const directURL=exported('make_tiled_copy','universal',c.bits,c.dtype,fmt(V,copy.tv),`(${copy.tile.join(',')})`);
      equalLayout(c.id+'/primitive constructor',V.mtcDecodeURL(decoded(directURL)).tv,refs.make_layout_tv[c.id].layout_tv);
    });

    // Layout results copy plain text, preserving hierarchy and zero strides
    // when used as another API's input (here, retile).
    for(const c of cases.retile) guard(`fragment Layout -> retile/${c.id}`,()=>{
      const input=layout(c.input);
      // Compact A/B inputs are also valid partitions for make_fragment. Feed
      // its actual exported result into retile, exactly like the clipboard UI.
      const sourceURL=/^[AB]_plain/.test(c.id)
        ? exported('make_fragment_abc',c.id[0],'f16bf16','half_t','float',16,c.input,0)
        : exported('layout',c.input);
      const raw=V.variablesClipboardChoices(sourceURL)[0];
      const n=refs.retile[c.id].atom_values;
      const url=exported('retile',c.tv,`(${c.tile.join(',')})`,n,c.dtype,raw.value,0);
      const p=decoded(url).inputs;
      const result=V.rtComputeRetile(layout(p[0]),V.parseValue(p[1]),Number(p[2]),layout(p[4])).layout;
      equalLayout(c.id,result,refs.retile[c.id]);
      check(c.id,'input layout unchanged',raw.value,fmt(V,input));
    });

    guard('Swizzle -> composed Layout input',()=>{
      const url=exported('swizzle','(8,64):(64,1)','3,3,3');
      const sw=V.variablesClipboardChoices(url)[0];
      check('Swizzle','typed clipboard payload',sw.type,'Swizzle');
      const p=decoded(sw.value), spec=V.swParse(p.inputs[1]);
      const raw=V.variablesSwizzledChoice(p.inputs[0],spec,'Layout').value;
      const consumer=V.tmaParseSmemField(raw);
      check('Swizzle','descriptor consumer swizzle',json(consumer.sw),json(spec));
      check('Swizzle','descriptor consumer layout',fmt(V,layout(consumer.layoutStr)),fmt(V,layout(p.inputs[0])));
    });

    rejects('CopyAtom rejects MMAAtom URL',()=>V.mcaDecodeAtomURL(exported('make_mma_atom','f16bf16','half_t','float',16)));
    rejects('TV Layout rejects TiledCopy URL',()=>V.mtvDecodeThrVal(exported('make_tiled_copy','universal',32,'half_t','(4,2):(1,4)','(4,2)')));
    rejects('TiledCopy rejects Layout URL',()=>V.mtcDecodeURL(decoded(exported('layout','(4,8):(1,4)'))));
    rejects('MMAAtom rejects incompatible dtype',()=>V.mmaDecodeAtomInputs(decoded(exported('make_mma_atom','f16bf16','bfloat16_t','half_t',16)).inputs));
    for(const text of ['?key=unknown-a-b','?key=make_tiled_mma-f16bf16-half_t','?key=tv-2-only-one-input-too-many']) guard(`malformed/${text}`,()=>check(text,'rejected before constructor decode',decoded(text),null));
  } finally {
    for(const [k,descriptor] of saved) {if(descriptor) Object.defineProperty(V,k,descriptor);else delete V[k];}
  }
}
module.exports={runDecodingTests};
