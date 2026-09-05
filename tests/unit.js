// ═══════════════════════════════════════════════════════
//  JS-only unit tests — the parts with no CuTeDSL analogue
//
//  Two kinds live here:
//   1. INPUT HANDLING. CuTe's inputs are C++ types; the tool's are strings. The
//      parsers (parseLayout, mtcParseTiler, parseSwizzleSpec, tmaParseSmemField)
//      have nothing to diff against, so they are pinned here.
//   2. VALIDATION CUTE SKIPS. mtcCoverageCheck / mtcVectorizationCheck /
//      mtcRequireCompact catch configurations that CuTe compiles and runs while
//      being silently wrong (see "Validation that CuTe itself skips" in
//      CLAUDE.md). By definition the DSL cannot be the oracle for these — the
//      expectations are derived from the C++ preconditions they encode.
//
//  Everything that CAN be diffed against CuTeDSL belongs in cases.json instead.
// ═══════════════════════════════════════════════════════

'use strict';

const { parseExact, fmt } = require('./harness');

function runUnitTests(V, T) {
  const { check, guard, setSection } = T;

  // ── parseLayout: the tab-facing parser, which normalizes to rank 2 ──────────
  setSection('unit/parseLayout');
  const P = (s, o) => {
    const r = V.parseLayout(s, o);
    return V.formatLayoutStr(r.shape, r.stride);
  };
  guard('shape-only-auto-stride', () => {
    check('shape-only-auto-stride', '(4,8)', P('(4,8)'), '(4,8):(1,4)');
  });
  guard('rank1-padded-to-2', () => {
    // parseLayout always returns rank 2 so the grid builders have an (M, N).
    // An EXPLICIT stride pads with 0 (the added mode contributes nothing);
    // shape-only input pads the SHAPE first and then computes the auto stride
    // over the padded shape, so `(8)` becomes (8,1):(1,8). Both are rank-2 maps
    // over the same 8 elements -- the difference only shows in the printout,
    // which is why stripTrivialTrailing exists.
    check('rank1-padded-to-2', '8:1', P('8:1'), '(8,1):(1,0)');
    check('rank1-padded-to-2', '(8)', P('(8)'), '(8,1):(1,8)');
  });
  guard('nested', () => {
    check('nested', '((2,2),4)', P('((2,2),4):((1,8),2)'), '((2,2),4):((1,8),2)');
  });
  guard('unwrap-single-paren', () => {
    // parseValue unwraps (10) -> 10; tma_partition depends on this behaviour.
    check('unwrap-single-paren', '(10):(1)', P('(10):(1)'), '(10,1):(1,0)');
  });
  guard('reject-inner-colon', () => {
    let threw = false;
    try { V.parseLayout('(3:4)'); } catch (e) { threw = /inside parentheses/.test(e.message); }
    check('reject-inner-colon', 'throws on (3:4)', threw, true);
  });
  guard('reject-rank-mismatch', () => {
    let threw = false;
    try { V.parseLayout('(2,3,4):(1,2)'); } catch (e) { threw = /does not match stride rank/.test(e.message); }
    check('reject-rank-mismatch', 'throws on rank mismatch', threw, true);
  });
  guard('basis-opt-in', () => {
    // A `k@i` stride is rejected unless the caller opted in, so every tab that
    // does not support coordinate layouts fails loudly instead of making NaN.
    let threw = false;
    try { V.parseLayout('(4,5):(1@0,1@1)'); } catch (e) { threw = /not accepted by this tab/.test(e.message); }
    check('basis-opt-in', 'rejects k@i without {basis:true}', threw, true);
    check('basis-opt-in', 'accepts with the flag', P('(4,5):(1@0,1@1)', { basis: true }),
          '(4,5):(1@0,1@1)');
  });
  guard('basis-nested-rejected', () => {
    let threw = false;
    try { V.parseLayout('(4,5):(1@0@1,1@1)', { basis: true }); }
    catch (e) { threw = /hierarchical coordinate/.test(e.message); }
    check('basis-nested-rejected', 'rejects k@i@j', threw, true);
  });
  guard('origin-prefix', () => {
    const r = V.parseLayout('(0,0) o (4,4):(1@0,1@1)', { basis: true });
    check('origin-prefix', 'origin', JSON.stringify(r.origin), '[0,0]');
    check('origin-prefix', 'layout', V.formatLayoutStr(r.shape, r.stride), '(4,4):(1@0,1@1)');
  });
  guard('origin-needs-basis', () => {
    let threw = false;
    try { V.parseLayout('(0,0) o (4,4):(1,4)'); } catch (e) { threw = /only supported by the Layout tab/.test(e.message); }
    check('origin-needs-basis', 'rejects origin form elsewhere', threw, true);
  });

  // ── mtcParseTiler: a Tiler is NOT a Layout ─────────────────────────────────
  setSection('unit/mtcParseTiler');
  const T2 = (s) => {
    const t = V.mtcParseTiler(s);
    return t.extents.join(',') + ' | ' + t.strides.map(x => (x === null ? '-' : x)).join(',');
  };
  guard('plain-shape', () => check('plain-shape', '(16, 64)', T2('(16, 64)'), '16,64 | -,-'));
  guard('per-mode-layouts', () => {
    // parseLayout would REFUSE this (a colon inside parens); a tiler mode is an
    // independent layout, which is exactly how CuTeDSL prints tc.tiler_mn.
    check('per-mode-layouts', '(8:1, 16:2)', T2('(8:1, 16:2)'), '8,16 | 1,2');
  });
  guard('nested-mode', () => check('nested-mode', '((2,4), 16)', T2('((2,4), 16)'), '8,16 | -,-'));
  guard('rank1-padded', () => check('rank1-padded', '128', T2('128'), '128,1 | -,-'));
  guard('trailing-comma', () => check('trailing-comma', '(128,)', T2('(128,)'), '128,1 | -,-'));
  guard('rank3-rejected', () => {
    let threw = false;
    try { V.mtcParseTiler('(2,4,8)'); } catch (e) { threw = /rank 3/.test(e.message); }
    check('rank3-rejected', 'throws on rank 3', threw, true);
  });

  // ── parseSwizzleSpec ───────────────────────────────────────────────────────
  setSection('unit/parseSwizzleSpec');
  const SW = (s) => { const r = V.parseSwizzleSpec(s); return r ? `${r.B},${r.M},${r.S}` : 'null'; };
  guard('spellings', () => {
    for (const s of ['3,4,3', 'Sw<3,4,3>', 'Swizzle<3,4,3>', '3 4 3', ' 3, 4, 3 ']) {
      check('spellings', s, SW(s), '3,4,3');
    }
    check('spellings', 'empty', SW(''), 'null');
    check('spellings', 'garbage', SW('nope'), 'null');
  });

  // ── tmaParseSmemField: an inline swizzle prefix overrides the picker ────────
  setSection('unit/tmaParseSmemField');
  guard('inline-prefix', () => {
    const r = V.tmaParseSmemField('Sw<3,4,3> o (8,64):(64,1)');
    check('inline-prefix', 'swizzle', `${r.sw.B},${r.sw.M},${r.sw.S}`, '3,4,3');
    check('inline-prefix', 'layout', r.layoutStr.trim(), '(8,64):(64,1)');
  });
  guard('no-prefix', () => {
    const r = V.tmaParseSmemField('(8,64):(64,1)');
    check('no-prefix', 'swizzle', String(r.sw), 'null');
  });

  // ── tmaSwizzleInfo: the TMA triple is in BYTES, the grid is in ELEMENTS ─────
  setSection('unit/tmaSwizzleInfo');
  guard('byte-vs-element', () => {
    // Sw<3,4,3> on half_t draws as Sw<3,3,3>: M is a byte exponent, and one
    // element is 2 B. Feeding the byte triple to applySwizzleOffset is the bug
    // this conversion exists to prevent.
    const i16 = V.tmaSwizzleInfo({ B: 3, M: 4, S: 3 }, 16);
    check('byte-vs-element', 'half_t elemStr', i16.elemStr, 'Sw<3,3,3>');
    check('byte-vs-element', 'half_t swElem.M', i16.swElem.M, 3);
    check('byte-vs-element', 'width bytes', i16.widthBytes, 128);
    check('byte-vs-element', 'enum', i16.enumName, 'CU_TENSOR_MAP_SWIZZLE_128B');
    const i128 = V.tmaSwizzleInfo({ B: 3, M: 4, S: 3 }, 128);
    check('byte-vs-element', 'uint128_t elemStr', i128.elemStr, 'Sw<3,0,3>');
  });
  guard('rejects-bad-triples', () => {
    for (const [sw, bits, why] of [
      [{ B: 4, M: 4, S: 3 }, 16, 'B out of range at M == 4'],
      [{ B: 3, M: 4, S: 2 }, 16, 'S must be 3 at M == 4'],
      [{ B: 3, M: 7, S: 3 }, 16, 'M outside {4,5,6}'],
    ]) {
      let threw = false;
      try { V.tmaSwizzleInfo(sw, bits); } catch (e) { threw = true; }
      check('rejects-bad-triples', why, threw, true);
    }
  });

  // ── applySwizzleOffset is an involution ────────────────────────────────────
  setSection('unit/applySwizzleOffset');
  guard('involution', () => {
    const sw = { B: 3, M: 3, S: 3 };
    let ok = true;
    for (let x = 0; x < 512; x++) {
      if (V.applySwizzleOffset(V.applySwizzleOffset(x, sw), sw) !== x) { ok = false; break; }
    }
    check('involution', 'swizzle(swizzle(x)) == x', ok, true);
  });
  guard('permutation', () => {
    // A swizzle must permute each period, or SMEM cells would collide.
    const sw = { B: 3, M: 3, S: 3 };
    const seen = new Set();
    for (let x = 0; x < 512; x++) seen.add(V.applySwizzleOffset(x, sw));
    check('permutation', 'image size over [0,512)', seen.size, 512);
  });
  guard('null-is-identity', () => {
    check('null-is-identity', 'applySwizzleOffset(7, null)', V.applySwizzleOffset(7, null), 7);
  });

  // ── Validation CuTe skips: make_tiled_copy ─────────────────────────────────
  // These configurations all COMPILE and RUN under CuTeDSL while being silently
  // wrong, so the DSL cannot be the oracle. The expectations encode the C++
  // preconditions each check exists to enforce.
  setSection('unit/mtc-validation');

  // mtcCoverageCheck reports by THROWING (the render path shows the message in
  // the error box), so the test asks whether it threw and on what.
  const cov = (tvStr, tilerStr) => {
    const tv = V.parseLayout(tvStr);
    const L = new V.Layout(tv.shape, tv.stride);
    const tiler = V.mtcParseTiler(tilerStr);
    const thrSize = V.product(L.shape[0]);
    const valSize = V.product(L.shape[1]);
    const broadcast = V.mtcHasBroadcast(L);
    const look = V.mtcBuildTileLookup(L, tiler.extents, thrSize, valSize);
    try {
      V.mtcCoverageCheck(look, tiler.extents, thrSize, valSize, null, broadcast);
      return 'ok';
    } catch (e) {
      return e.message;
    }
  };

  guard('coverage-exact', () => {
    check('coverage-exact', 'accepted', cov('((8,16),8):((128,1),16)', '(16,64)'), 'ok');
  });
  guard('coverage-tiler-too-big', () => {
    // Uncovered elements are never copied — CuTe asserts nothing here.
    const m = cov('((8,16),8):((128,1),16)', '(16,128)');
    check('coverage-tiler-too-big', 'reports partial coverage', /covers only/.test(m), true);
  });
  guard('coverage-tiler-too-small', () => {
    // layout_tv indexes past the end of the tile, into the next one.
    const m = cov('((8,16),8):((128,1),16)', '(16,32)');
    check('coverage-tiler-too-small', 'reports out-of-tile reads', /outside its tile/.test(m), true);
  });
  guard('coverage-broadcast-ok', () => {
    // A stride-0 mode is INTENDED replication (make_tiled_copy_A does this), so
    // uniform duplicate claims must pass rather than be reported as collisions.
    check('coverage-broadcast-ok', 'accepted', cov('((4,2),4):((4,0),1)', '(4,4)'), 'ok');
  });
  guard('coverage-accidental-overlap', () => {
    // No stride is 0, yet two (tid, vid) pairs land on the same cell.
    const m = cov('((2,2),4):((4,4),1)', '(4,4)');
    check('coverage-accidental-overlap', 'reports non-injectivity', /not injective/.test(m), true);
  });

  // mtcVectorizationCheck returns { kind }: 'm' / 'n' name the tile axis T0's
  // first AtomNumVal values run along (and therefore the majorness the tensor
  // needs); 'none' means no tensor layout can make upcast<N> accept the copy.
  const vec = (atomNumVal, tvStr, tilerStr) => {
    const tv = V.parseLayout(tvStr);
    const L = new V.Layout(tv.shape, tv.stride);
    return V.mtcVectorizationCheck(atomNumVal, L, V.mtcParseTiler(tilerStr).extents).kind;
  };
  guard('vectorization-along-n', () => {
    // T0's 8 values sit at tile indices 0,16,32,... = (0,0)..(0,7): a stride-1
    // run across N, so the tensor must be row-major over the tile.
    check('vectorization-along-n', 'kind', vec(8, '((8,16),8):((128,1),16)', '(16,64)'), 'n');
  });
  guard('vectorization-along-m', () => {
    check('vectorization-along-m', 'kind', vec(8, '((8,16),8):((128,1),1)', '(16,64)'), 'm');
  });
  guard('vectorization-scattered', () => {
    // The vids step by 17 in a 16-row tile: contiguous along neither axis, so
    // CuTe's upcast<8> rejects it at JIT time.
    check('vectorization-scattered', 'kind', vec(8, '((8,16),8):((128,1),17)', '(16,64)'), 'none');
  });
  guard('vectorization-trivial', () => {
    check('vectorization-trivial', 'kind', vec(1, '((8,16),8):((128,1),16)', '(16,64)'), 'trivial');
  });

  guard('require-compact', () => {
    // make_layout_tv's docstring states cosize == size as a precondition and
    // enforces nothing; right_inverse of a non-injective layout is PARTIAL.
    let holes = false, collide = false, fine = true;
    try { V.mtcRequireCompact('thr_layout', new V.Layout([4, 4], [1, 8])); } catch (e) { holes = /cosize/.test(e.message); }
    try { V.mtcRequireCompact('thr_layout', new V.Layout([4, 4], [1, 1])); } catch (e) { collide = /cosize/.test(e.message); }
    try { V.mtcRequireCompact('thr_layout', new V.Layout([4, 4], [1, 4])); } catch (e) { fine = false; }
    check('require-compact', 'rejects holes (cosize > size)', holes, true);
    check('require-compact', 'rejects collisions (cosize < size)', collide, true);
    check('require-compact', 'accepts a compact layout', fine, true);
  });

  guard('require-atom-divides', () => {
    // The one thing CuTe DOES assert: TiledNumVal % AtomNumVal == 0.
    let threw = false;
    try { V.mtcRequireAtomDivides(12, 8, 128, 16, 'half_t'); } catch (e) { threw = true; }
    check('require-atom-divides', 'rejects 12 % 8', threw, true);
    let ok = true;
    try { V.mtcRequireAtomDivides(16, 8, 128, 16, 'half_t'); } catch (e) { ok = false; }
    check('require-atom-divides', 'accepts 16 % 8', ok, true);
  });

  // ── Validation CuTe skips: TMA ─────────────────────────────────────────────
  setSection('unit/tma-validation');

  const tmaCfgOf = ({ dtype, gmem, smem, tiler, swizzle }) => {
    const elemBits = V.DTYPE_BITS[dtype];
    const gp = V.parseLayout(gmem);
    const sp = V.parseLayout(smem);
    const t = V.mtcParseTiler(tiler);
    return {
      gShape: gp.shape.slice(), gStride: gp.stride.slice(), elemBits,
      swInfo: V.tmaSwizzleInfo(swizzle ? V.parseSwizzleSpec(swizzle) : null, elemBits),
      sFlatShape: V.flatten(sp.shape), sFlatStride: V.flatten(sp.stride),
      tiler: { extents: t.extents, strides: t.strides.map(s => (s === null ? 1 : parseInt(s, 10))) },
    };
  };
  const tmaThrows = (cfg, re) => {
    try { V.tmaComputeAtom(tmaCfgOf(cfg)); return false; }
    catch (e) { return re.test(e.message); }
  };

  guard('tma-size-mismatch', () => {
    // CuTe's only static assert here: size(smem_layout) == size(cta_tiler).
    check('tma-size-mismatch', 'throws', tmaThrows(
      { dtype: 'half_t', gmem: '(256,128):(128,1)', smem: '(8,64):(64,1)', tiler: '(8,32)' },
      /top-level size equivalence|!=/), true);
  });
  guard('tma-non-permutation-smem', () => {
    // right_inverse of a non-permutation is a PARTIAL inverse in CuTe, which
    // silently shrinks the box rather than failing.
    check('tma-non-permutation-smem', 'throws', tmaThrows(
      { dtype: 'half_t', gmem: '(256,128):(128,1)', smem: '(8,64):(128,1)', tiler: '(8,64)' },
      /not compact/), true);
  });
  guard('tma-tile-past-end', () => {
    check('tma-tile-past-end', 'throws', tmaThrows(
      { dtype: 'half_t', gmem: '(8,8):(8,1)', smem: '(16,16):(16,1)', tiler: '(16,16)' },
      /does not fit inside|size equivalence|!=/), true);
  });
  guard('tma-majorness-mismatch', () => {
    // A host assert() in CuTe, so it vanishes in release builds: reported as an
    // issue, with the box still drawn.
    const a = V.tmaComputeAtom(tmaCfgOf(
      { dtype: 'half_t', gmem: '(256,128):(128,1)', smem: '(64,64):(1,64)', tiler: '(64,64)' }));
    const msgs = a.issues.filter(i => i.level === 'error').map(i => i.text).join(' ');
    check('tma-majorness-mismatch', 'reports majorness', /Majorness/.test(msgs), true);
    check('tma-majorness-mismatch', 'still produced a box', a.boxSize > 0, true);
  });
  guard('tma-inner-box-alignment', () => {
    // 1 x 4 half_t = 8 B: below the 16 B floor cuTensorMapEncodeTiled demands.
    const a = V.tmaComputeAtom(tmaCfgOf(
      { dtype: 'half_t', gmem: '(256,128):(128,1)', smem: '(4,4):(4,1)', tiler: '(4,4)' }));
    const warns = a.issues.map(i => i.text).join(' ');
    check('tma-inner-box-alignment', 'reports the 16 B floor', /16B/.test(warns), true);
  });

  // ── tpComputePartition: preconditions ──────────────────────────────────────
  setSection('unit/tp-validation');
  guard('tp-size-assert', () => {
    let threw = false;
    try {
      V.tpComputePartition(V.parseLayout('((8,64)):((64,1))'),
                           V.parseLayout('((8,32),(2)):((1@1,1@0),(8@1))', { basis: true }), 512);
    } catch (e) { threw = /only static assert/.test(e.message); }
    check('tp-size-assert', 'throws on size<0> mismatch', threw, true);
  });
  guard('tp-inexact-divide', () => {
    let threw = false;
    try {
      V.tpComputePartition(V.parseLayout('((8,64)):((64,1))'),
                           V.parseLayout('((8,64)):((1@1,1@0))', { basis: true }), 384);
    } catch (e) { threw = /is not exact/.test(e.message); }
    check('tp-inexact-divide', 'throws when NumValSrc does not divide the tile', threw, true);
  });
  guard('tp-non-permutation-smem', () => {
    let threw = false;
    try {
      V.tpComputePartition(V.parseLayout('((8,64)):((128,1))'),
                           V.parseLayout('((8,64)):((1@1,1@0))', { basis: true }), 512);
    } catch (e) { threw = /not a permutation/.test(e.message); }
    check('tp-non-permutation-smem', 'throws', threw, true);
  });

  // ── local_tile: proj, and the coord reading CuTeDSL cannot type ────────────
  // Two things here have no oracle. The nested-rest-mode case is one CuTeDSL
  // cannot evaluate AT ALL -- its MLIR crd2idx reports "failed to infer result
  // type" for a scalar coord against a tuple rest mode -- so the expectation
  // encodes C++ CuTe's rule instead: crd2idx runs idx2crd on a scalar first, so
  // a scalar coord into a tuple mode is a FLAT index into it. The proj parser is
  // the other: proj is a C++ type (`Step<_1, X, _1>`), and this field takes
  // VALUES, so both that spelling and the bare `_1` are refused by name.
  setSection('unit/local_tile');
  guard('lt-nested-rest-mode', () => {
    // A single layout tiler makes the rest mode a tuple: zipped_divide gives
    // ((2,2),(2,(2,16))). Coord (1,2) must select ONE tile -- comparing against
    // idx2crd's nested coordinate matched nothing and left kept[0] undefined.
    const r = V.ltComputeLocalTile('(16,16):(16,1)', '(2,2):(1,4)', '(1,2)', '');
    check('lt-nested-rest-mode', 'rest mode is a tuple',
          JSON.stringify(r.restShape), '[2,[2,16]]');
    check('lt-nested-rest-mode', 'selects exactly one tile', r.kept.length, 1);
    check('lt-nested-rest-mode', 'at the flat index the coord names',
          JSON.stringify(r.kept[0].rcArr), '[1,2]');
    check('lt-nested-rest-mode', 'offset', r.baseOffset, 33);
  });
  guard('lt-parse-proj', () => {
    // A number selects the dimension; x / _ / None masks it out.
    for (const spelling of ['(1, x, 1)', '(1, X, 1)', '(1, None, 1)', '1,_,1']) {
      check('lt-parse-proj', spelling,
            JSON.stringify(V.ltParseProj(spelling)), '[true,false,true]');
    }
    check('lt-parse-proj', 'blank is no projection', V.ltParseProj('   '), null);
    // Any number selects -- `dice` never reads the value, so 2 and 7 are not
    // special-cased into meaning something other than 1.
    check('lt-parse-proj', 'any number selects',
          JSON.stringify(V.ltParseProj('(2, _, 7)')), '[true,false,true]');
  });
  guard('lt-normalize', () => {
    const n = (c, p) => {
      const r = V.ltComputeLocalTile('(32,64):(64,1)', '(8,16,4)', c, p);
      return `${r.projNorm} ${r.coordNorm}`;
    };
    // proj: any number -> 1, any mask spelling -> _. dice never reads the value,
    // so (2, X, 7) and (1, _, 1) are the same projection.
    check('lt-normalize', '(2, X, 7)', n('(1, 2, _)', '(2, X, 7)'), '(1, _, 1) (1, _, _)');
    check('lt-normalize', '(1, None, 1)', n('(1, 2, _)', '(1, None, 1)'), '(1, _, 1) (1, _, _)');
    // coord: only the MASKED slot is rewritten -- 2 is discarded unread, while
    // the surviving 1 and 3 are left exactly as typed.
    check('lt-normalize', 'masked coord slot only',
          n('(1, 2, 3)', '(1, _, 1)'), '(1, _, 1) (1, _, 3)');
    check('lt-normalize', 'mask in the last mode',
          n('(1, 2, 3)', '(1, 1, _)'), '(1, 1, _) (1, 2, _)');
    // A blank coord is implicitly all-underscores; writing that back makes the
    // expansion visible rather than leaving the field looking unrelated.
    check('lt-normalize', 'blank coord expands', n('', '(1, _, 1)'), '(1, _, 1) (_, _, _)');
  });
  guard('lt-normalize-noop-without-proj', () => {
    // Nothing is masked without a proj, so neither field is touched -- in
    // particular a blank coord must stay blank rather than become (_, _).
    const r = V.ltComputeLocalTile('(32,64):(64,1)', '(8,32)', '(1, 0)', '');
    check('lt-normalize-noop-without-proj', 'projNorm', r.projNorm, null);
    check('lt-normalize-noop-without-proj', 'coordNorm', r.coordNorm, null);
  });
  guard('lt-proj-rejects-cpp-syntax', () => {
    // `_1` is Int<1>, a template parameter. It is refused BY NAME rather than
    // as an unreadable token, because `Step<_1, X, _1>` is exactly what someone
    // pastes out of a GEMM kernel and a generic parse error would not help.
    const msg = (t) => { try { V.ltParseProj(t); return ''; } catch (e) { return e.message; } };
    check('lt-proj-rejects-cpp-syntax', 'Step<...> named',
          /CUTLASS C\+\+ template syntax/.test(msg('Step<_1, X, _1>')), true);
    check('lt-proj-rejects-cpp-syntax', '_1 named',
          /integral-constant syntax \(Int<1>\)/.test(msg('(_1, X, _1)')), true);
    check('lt-proj-rejects-cpp-syntax', '_1 suggests the value',
          /Write "1"/.test(msg('(_1, X, _1)')), true);
  });
  guard('lt-proj-rejects', () => {
    const bad = (proj, re) => {
      let msg = '';
      try { V.ltComputeLocalTile('(16,16):(16,1)', '(4,4)', '(1,2)', proj); }
      catch (e) { msg = e.message; }
      check('lt-proj-rejects', `${proj} -> ${re}`, re.test(msg), true);
    };
    bad('(1, _, 1)', /tiler is rank 2/);        // proj rank must match the tiler's
    bad('(_, _)', /masks out every tiler mode/);
    bad('(1, q)', /Cannot read proj component/);
  });
  guard('lt-proj-coord-rank', () => {
    // With a proj the coord is given over the FULL tiler, since dice() pairs the
    // two mode by mode. A short one is a mistake, not a padded coord.
    let msg = '';
    try { V.ltComputeLocalTile('(32,64):(64,1)', '(8,16,4)', '(1,2)', '(1,_,1)'); }
    catch (e) { msg = e.message; }
    check('lt-proj-coord-rank', 'rejects a coord shorter than proj',
          /Coord has 2 components? but proj has 3/.test(msg), true);
  });

  // ── filter's target profile ────────────────────────────────────────────────
  // layout.js mirrors the C++ `filter(layout, profile)` overload, but CuTeDSL's
  // cute.filter takes no target_profile, so there is nothing to diff against.
  // (cute.coalesce DOES take one -- those cases live in cases.json.)
  setSection('unit/filter-profile');
  guard('filter-by-mode', () => {
    const L = new V.Layout([[2, 3], [1, 4]], [[4, 1], [0, 8]]);
    const f = (p) => { const r = V.filter(L, p); return V.formatLayoutStr(r.shape, r.stride); };
    // No profile: filter the whole layout flat -- the stride-0 mode goes and the
    // rest coalesce only where they are actually contiguous (filter does not
    // reorder, so 2:4 and 3:1 stay separate).
    check('filter-by-mode', 'no profile', f(undefined), '(2,3,4):(4,1,8)');
    // A profile keeps the mode structure and filters INSIDE each named mode.
    check('filter-by-mode', 'profile [1]', f([1]), '((2,3),(1,4)):((4,1),(0,8))');
    check('filter-by-mode', 'profile [1,1]', f([1, 1]), '((2,3),4):((4,1),8)');
  });

  // ── isBijective / formatLayoutStr / stripTrivialTrailing ───────────────────
  setSection('unit/helpers');
  guard('isBijective', () => {
    check('isBijective', '(4,4):(4,1)', V.isBijective(new V.Layout([4, 4], [4, 1])), true);
    check('isBijective', '(4,4):(1,8) has a gap', V.isBijective(new V.Layout([4, 4], [1, 8])), false);
    check('isBijective', '(4,4):(1,1) collides', V.isBijective(new V.Layout([4, 4], [1, 1])), false);
  });
  guard('formatLayoutStr-basis', () => {
    // Used to throw "x.map is not a function" on a basis stride; every tab that
    // prints a layout string goes through here.
    check('formatLayoutStr-basis', 'k@i',
          V.formatLayoutStr([4, 5], [V.makeBasis(1, 0), V.makeBasis(1, 1)]), '(4,5):(1@0,1@1)');
  });
  guard('stripTrivialTrailing', () => {
    const r = V.stripTrivialTrailing([8, 1], [1, 0]);
    check('stripTrivialTrailing', 'recovers rank 1', `${r.shape}:${r.stride}`, '8:1');
    const k = V.stripTrivialTrailing([8, 4], [1, 8]);
    check('stripTrivialTrailing', 'leaves rank 2 alone', `${k.shape}:${k.stride}`, '8,4:1,8');
  });

  // ── downloadSVG's watermark ────────────────────────────────────────────────
  // Pure string work on markup the builders produced, so it is pinned here
  // rather than in cases.json — CuTeDSL has nothing to say about an export.
  setSection('unit/watermark');
  guard('watermark-structure', () => {
    const src = V.buildLayoutSVG([4, 4], [1, 4], new Set(['value']));
    const out = V.watermarkSVGMarkup(src);
    check('watermark-structure', 'is an svg', /^<svg\b/.test(out), true);
    check('watermark-structure', 'closes', out.trimEnd().endsWith('</svg>'), true);
    check('watermark-structure', 'carries the URL', out.includes(V.SVG_WATERMARK), true);
    // The original content must survive untouched, pushed down by the band.
    const body = src.slice(src.indexOf('>') + 1, src.lastIndexOf('</svg>'));
    check('watermark-structure', 'original content preserved', out.includes(body), true);
    check('watermark-structure', 'content is translated, not redrawn',
          /<g transform="translate\(0, [\d.]+\)">/.test(out), true);
    // A standalone file needs explicit dimensions; the on-screen `style`
    // (max-height:70vh and friends) must NOT come along.
    check('watermark-structure', 'no inline style', /<svg[^>]*style=/.test(out), false);
    check('watermark-structure', 'has width/height', /<svg[^>]*width="[\d.]+"[^>]*height="[\d.]+"/.test(out), true);
  });
  guard('watermark-geometry', () => {
    // The band grows the box vertically, and the text always FITS horizontally:
    // at the 9px legibility floor a 57-character URL needs ~330px, which is
    // wider than a small grid's entire canvas, so the export widens instead of
    // clipping the text or shrinking it below readability.
    const adv = 0.62, padF = 0.6;
    for (const [M, N] of [[4, 4], [2, 2], [8, 16], [32, 64], [1, 8]]) {
      const src = V.buildLayoutSVG([M, N], [1, M], new Set());
      const inVB = /viewBox="([^"]*)"/.exec(src)[1].split(' ').map(Number);
      const out = V.watermarkSVGMarkup(src);
      const outVB = /viewBox="([^"]*)"/.exec(out)[1].split(' ').map(Number);
      const fs = Number(/font-size="([\d.]+)"/.exec(out)[1]);
      const id = `${M}x${N}`;
      check('watermark-geometry', `${id} taller`, outVB[3] > inVB[3], true);
      check('watermark-geometry', `${id} never narrower`, outVB[2] >= inVB[2], true);
      check('watermark-geometry', `${id} font legible`, fs >= 9 && fs <= 22, true);
      check('watermark-geometry', `${id} text fits`,
            V.SVG_WATERMARK.length * fs * adv + 2 * fs * padF <= outVB[2] + 0.01, true);
      check('watermark-geometry', `${id} no float noise`,
            /viewBox="[-\d. ]*"/.test(out) && !/\d\.\d{3,}/.test(outVB.join(' ')), true);
    }
  });
  guard('watermark-passthrough', () => {
    // Anything we cannot anchor to comes back untouched rather than mangled.
    for (const junk of ['', 'not svg at all', '<svg></svg>',
                        '<svg viewBox="bad">x</svg>', '<svg viewBox="0 0 0 10"></svg>']) {
      check('watermark-passthrough', JSON.stringify(junk), V.watermarkSVGMarkup(junk), junk);
    }
  });
  guard('viz-filename', () => {
    // The overlay's download button has no hand-written name to use, so it
    // derives one; the tab id must not leak into the file name.
    check('viz-filename', 'layout host', V.vizFilename('ot1-layout-svg-host'), 'cute-layout.svg');
    check('viz-filename', 'tv host', V.vizFilename('ot12-tv-svg-host'), 'cute-tv.svg');
    check('viz-filename', 'copy pane', V.vizFilename('ot1-mtc-src-svg'), 'cute-mtc-src.svg');
    check('viz-filename', 'local_tile pane', V.vizFilename('ot1-lt-a-svg'), 'cute-lt-a.svg');
  });

  // ── make_tiled_mma: text inputs and the guards CuTe does not raise ─────────
  //  cases.json diffs the derivation against CuTeDSL. It cannot reach any of
  //  this: the DSL takes a `cute.Layout` and a `Tiler`, never a string, and it
  //  never sees a permutation whose size does not divide.
  setSection('unit/make_tiled_mma');
  const AL = (s) => { const L = V.mtmParseAtomLayout(s); return fmt(V, L); };
  guard('atom-layout-parse', () => {
    // A bare shape gets cute.make_layout's COMPACT stride, which differs from a
    // plain prefix product in one place: a size-1 mode gets 0. That is exactly
    // what makes (2,2,1) print as thr_layout_vmnk (32,2,2,1):(1,32,64,0).
    check('atom-layout-parse', 'shape only', AL('(2,2,1)'), '(2,2,1):(1,2,0)');
    check('atom-layout-parse', 'size-1 first', AL('(1,1,2)'), '(1,1,2):(0,0,1)');
    check('atom-layout-parse', 'explicit stride kept', AL('(2,2,1):(2,1,4)'), '(2,2,1):(2,1,4)');
    check('atom-layout-parse', 'whitespace', AL(' (2, 2, 1) '), '(2,2,1):(1,2,0)');
  });
  guard('atom-layout-refusals', () => {
    const why = (s) => { try { V.mtmParseAtomLayout(s); return null; } catch (e) { return e.message; } };
    // parseLayout would pad "(2,2)" out to rank 2 and silently invent a mode;
    // rank 3 is what CuTeDSL requires ("expects rank-3 MNK atom layout").
    check('atom-layout-refusals', 'rank 2 rejected', /must be rank 3/.test(why('(2,2)')), true);
    check('atom-layout-refusals', 'rank 4 rejected', /must be rank 3/.test(why('(2,2,1,1)')), true);
    check('atom-layout-refusals', 'empty rejected', /empty/.test(why('  ')), true);
    check('atom-layout-refusals', 'stride rank mismatch',
          /stride must be rank 3/.test(why('(2,2,1):(1,2)')), true);
  });
  guard('perm-parse', () => {
    const PM = (s) => V.mtmParsePerm(s).map(x => (x === null ? '_' : fmt(V, x))).join('|');
    check('perm-parse', 'blank is all-None', PM(''), '_|_|_');
    check('perm-parse', 'plain extents', PM('(32, 32, 16)'), '32:1|32:1|16:1');
    check('perm-parse', 'underscores', PM('(_, x, None)'), '_|_|_');
    // A mode may itself be a layout, and its commas must not be read as mode
    // separators — this is why the split is depth-aware.
    check('perm-parse', 'nested layout mode',
          PM('((2,16):(16,1), 32, 16)'), '(2,16):(16,1)|32:1|16:1');
    check('perm-parse', 'unwrapped mode list', PM('32, _, 16'), '32:1|_|16:1');
  });
  guard('perm-refusals', () => {
    const why = (s) => { try { V.mtmParsePerm(s); return null; } catch (e) { return e.message; } };
    check('perm-refusals', 'wrong arity', /exactly 3 modes/.test(why('(32, 32)')), true);
    // Int<32> and Tile<...> are the two things that get pasted straight out of a
    // CUTLASS kernel; naming them beats "cannot read mode".
    check('perm-refusals', 'Int<> syntax named', /integral-constant/.test(why('(_32, 32, 16)')), true);
    check('perm-refusals', 'Tile<> syntax named',
          /C\+\+ template syntax/.test(why('Tile<_32,_32,_16>')), true);
  });
  guard('perm-must-divide', () => {
    // CuTe's zipped_divide needs the permutation to divide what the warps
    // cover. It is a static_assert in C++ and invisible from the DSL, so an
    // indivisible size would otherwise come back as a silently wrong grid.
    const atom = V.mmaWarpAtom('f16bf16', 16);
    const run = (perm) => {
      try {
        V.mtmComputeTiledMma(atom, V.mtmParseAtomLayout('(2,2,1)'), V.mtmParsePerm(perm));
        return null;
      } catch (e) { return e.message; }
    };
    check('perm-must-divide', 'M too small', /M mode has size 16/.test(run('(16, 32, 16)')), true);
    check('perm-must-divide', 'M indivisible', /not a multiple/.test(run('(48, 32, 16)')), true);
    check('perm-must-divide', 'K indivisible', /K mode has size 24/.test(run('(32, 32, 24)')), true);
    check('perm-must-divide', 'exact multiple accepted', run('(64, 32, 32)'), null);
  });
  guard('warp-must-be-distinct', () => {
    // A stride-0 mode in atom_layout_mnk would make two warps the same warp.
    // CuTeDSL accepts the layout; right_inverse would then be a PARTIAL inverse
    // and half the threads would silently vanish from the picture.
    const atom = V.mmaWarpAtom('f16bf16', 16);
    let msg = null;
    try { V.mtmComputeTiledMma(atom, V.mtmParseAtomLayout('(2,2,1):(1,0,0)'), [null, null, null]); }
    catch (e) { msg = e.message; }
    check('warp-must-be-distinct', 'stride-0 atom layout rejected', /not a bijection/.test(msg || ''), true);
  });
  guard('warp-lines', () => {
    // What a warp-mode cell actually says. With a warp id it names only that
    // warp; without one it stacks every warp that touches the cell, one per
    // line like TV mode — until there are more than four, which is where a cell
    // stops being readable and it collapses to the compact single line.
    const L = (warps, sum, focus) => V.mtmWarpLines(warps, sum, focus).join('|');
    check('warp-lines', 'single warp', L([3], false, null), 'W3');
    check('warp-lines', 'A broadcast lists both', L([0, 2], false, null), 'W0|W2');
    check('warp-lines', 'C reduction lists both', L([0, 4], true, null), 'W0|W4');
    check('warp-lines', 'four is still four lines',
          L([0, 1, 2, 3], true, null), 'W0|W1|W2|W3');
    check('warp-lines', 'five collapses', L([0, 1, 2, 3, 4], true, null), '\u03a3W0..W4');
    // A and B are never accumulated into, so the collapsed form must NOT carry
    // the sum sign — several warps on an A cell are readers of one value.
    check('warp-lines', 'five collapses (broadcast, no sigma)',
          L([0, 2, 4, 6, 8], false, null), 'W0,W2,W4,W6,W8');
    check('warp-lines', 'contiguous broadcast, no sigma',
          L([0, 1, 2, 3, 4, 5, 6, 7], false, null), 'W0..W7');
    check('warp-lines', 'sigma is C-only',
          [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [0, 1, 2, 3, 4, 5, 6, 7]]
            .some(w => L(w, false, null).includes('\u03a3')), false);
    check('warp-lines', 'focused cell names only that warp', L([0, 2], false, 2), 'W2');
    check('warp-lines', 'unsorted, deduped', L([2, 0, 2], false, null), 'W0|W2');
  });
  guard('tv-lines', () => {
    // The TV-mode counterpart. A focused thread's cell shows only ITS slot, so
    // you read which value of that thread landed there rather than picking it
    // out of the threads broadcasting onto the same cell.
    const e = (...ts) => ts.map(([t, v]) => ({ t, v, w: Math.floor(t / 32) }));
    const L = (entries, focus) => V.mtmTvLines(entries, focus).join('|');
    check('tv-lines', 'one owner', L(e([5, 3]), null), 'T5|V3');
    check('tv-lines', 'broadcast lists both', L(e([0, 1], [64, 1]), null), 'T0/V1|T64/V1');
    check('tv-lines', 'focused shows only its slot', L(e([0, 1], [64, 1]), 64), 'T64|V1');
  });
  /** Cell census off the rendered markup — the thing the eye actually reads. */
  const mtmCensus = (V_, opnd, opts) => {
    const svg = V_.mtmBuildSVG(V_.mtmOperandGrid(opnd), opnd.tile[0], opnd.tile[1], opts);
    const rects = [...svg.matchAll(/<rect[^>]*fill="([^"]*)"\s*\n?\s*fill-opacity="([^"]*)"/g)];
    return {
      grey: rects.filter(m => m[1] === '#f0f0f0').length,
      full: rects.filter(m => m[1] !== '#f0f0f0' && m[2] === '1').length,
      dim: rects.filter(m => m[1] !== '#f0f0f0' && m[2] !== '1').length,
      total: rects.length,
      // One <text> per LINE, axis rulers included: an M x N grid contributes
      // M + N of them before any cell speaks.
      labelLines: (svg.match(/font-family="monospace"/g) || []).length,
    };
  };
  guard('focus-colours-one-region', () => {
    // The load-bearing claim of the focus box: ONLY the picked unit's cells stay
    // coloured, and the others lose their labels as well as their colour.
    const atom = V.mmaWarpAtom('f16bf16', 16);
    const r = V.mtmComputeTiledMma(atom, V.mtmParseAtomLayout('(2,2,2)'), [null, null, null]);
    // A is 32x32; 8 warps, and A does not depend on N, so one warp owns an
    // atom-sized 16x16 = 256 of the 1024 cells.
    const w = mtmCensus(V, r.A, { mode: 'warp', focus: 2, sum: false });
    check('focus-colours-one-region', 'one warp coloured', w.full, 256);
    check('focus-colours-one-region', 'the rest grey', w.grey, 1024 - 256);
    // 32 + 32 axis labels, then one line per focused cell.
    check('focus-colours-one-region', 'only focused cells are labelled',
          w.labelLines, (32 + 32) + 256 * 1);
    const all = mtmCensus(V, r.A, { mode: 'warp', focus: null, sum: false });
    check('focus-colours-one-region', 'no id -> nothing grey', all.grey, 0);
    check('focus-colours-one-region', 'no id -> all coloured', all.full, 1024);
    // TV mode filters identically, just by thread. Thread 34 is warp 1, and one
    // thread of an m16n8k16 A fragment owns 8 elements.
    const t = mtmCensus(V, r.A, { mode: 'tv', focus: 34, sum: false });
    check('focus-colours-one-region', 'tv mode greys too', t.grey, 1024 - 8);
    check('focus-colours-one-region', 'one thread coloured', t.full, 8);
    check('focus-colours-one-region', 'tv focused cells carry T and V',
          t.labelLines, (32 + 32) + 8 * 2);
  });
  guard('focus-dims-the-repetitions', () => {
    // The bottom row: the warp pattern REPEATS across the permuted tile, so the
    // copies the focused unit also lands in are drawn in its hue at reduced
    // opacity. Greying them would say it does not go there, which is false.
    const atom = V.mmaWarpAtom('f16bf16', 16);
    const r = V.mtmComputeTiledMma(atom, V.mtmParseAtomLayout('(2,2,1)'),
                                   V.mtmParsePerm('(64, 32, 16)'));
    // A is 64x16 = 1024 with Rest (2,1); warp 1 owns 16x16 in each copy.
    const f = mtmCensus(V, r.A, { mode: 'warp', focus: 1, sum: false, dimRest: true });
    check('focus-dims-the-repetitions', 'first copy at full colour', f.full, 256);
    check('focus-dims-the-repetitions', 'the other copy dimmed, not greyed', f.dim, 256);
    check('focus-dims-the-repetitions', 'everything else grey', f.grey, 1024 - 512);
    check('focus-dims-the-repetitions', 'both copies keep labels',
          f.labelLines, (64 + 16) + 512 * 1);
    // Without a focus the bottom row is unchanged: Rest 0 coloured, the copies
    // flat grey but still labelled.
    const n = mtmCensus(V, r.A, { mode: 'warp', focus: null, sum: false, dimRest: true });
    check('focus-dims-the-repetitions', 'no focus -> nothing dimmed', n.dim, 0);
    check('focus-dims-the-repetitions', 'no focus -> half grey', n.grey, 512);
    // Unfocused, every cell names BOTH N-warps that read it, so two lines each.
    check('focus-dims-the-repetitions', 'no focus -> every cell labelled',
          n.labelLines, (64 + 16) + 1024 * 2);
  });
  guard('warp-label', () => {
    // A contiguous run collapses; a strided set does not. C is a REDUCTION, so
    // it gets the sum sign and a "+"; A and B are broadcast reads, so a comma.
    check('warp-label', 'single', V.mtmWarpLabel([3], false), 'W3');
    check('warp-label', 'contiguous read', V.mtmWarpLabel([0, 1], false), 'W0..W1');
    check('warp-label', 'strided read', V.mtmWarpLabel([0, 2], false), 'W0,W2');
    check('warp-label', 'contiguous sum', V.mtmWarpLabel([0, 1], true), '\u03a3W0..W1');
    check('warp-label', 'strided sum', V.mtmWarpLabel([0, 4], true), '\u03a3W0+W4');
    check('warp-label', 'unsorted, deduped', V.mtmWarpLabel([2, 0, 2, 1], false), 'W0..W2');
  });
  guard('rest-region-marks-the-repetition', () => {
    // The bottom row colours Rest == 0 and greys the copies. Without a
    // permutation nothing repeats, so every cell must be Rest 0 -- a stray
    // non-zero there would grey out half a picture that has no repetition.
    const atom = V.mmaWarpAtom('f16bf16', 16);
    const plain = V.mtmComputeTiledMma(atom, V.mtmParseAtomLayout('(2,2,1)'), [null, null, null]);
    const gp = V.mtmOperandGrid(plain.A);
    check('rest-region-marks-the-repetition', 'no perm -> Rest (1,1)',
          plain.A.restShape.join(','), '1,1');
    check('rest-region-marks-the-repetition', 'no perm -> every cell Rest 0',
          gp.every(row => row.every(c => c.rest === 0 && c.entries.length)), true);
    // Doubling M doubles the tile and puts exactly half the cells in Rest 1.
    const wide = V.mtmComputeTiledMma(atom, V.mtmParseAtomLayout('(2,2,1)'),
                                      V.mtmParsePerm('(64, 32, 16)'));
    const gw = V.mtmOperandGrid(wide.A);
    const flat = gw.flat();
    check('rest-region-marks-the-repetition', 'perm -> Rest (2,1)', wide.A.restShape.join(','), '2,1');
    check('rest-region-marks-the-repetition', 'half the cells are the copy',
          flat.filter(c => c.rest === 0).length, flat.length / 2);
    check('rest-region-marks-the-repetition', 'every cell still owned',
          flat.every(c => c.entries.length > 0), true);
  });

  // ── The MMA "Alternative View" rotates B and NOTHING else ─────────────────
  //  The quadrant layout draws B as K x N so its K axis meets A's and its N
  //  axis meets C's. The whole claim is that this is a change of ORIENTATION,
  //  not of mapping — so it is checked by comparing the two drawings cell for
  //  cell rather than by inspecting the code that produces them.
  setSection('unit/mma-alt-view');
  /** Cell labels out of a buildTVSVG grid, row-major, one array per cell.
   *  The axis rulers come first (N column labels, then M row labels). */
  const tvCells = (svg, M, N, perCell) => {
    const txt = [...svg.matchAll(/font-family="monospace">([^<]*)</g)].map(m => m[1]);
    const body = txt.slice(N + M);
    const out = [];
    for (let m = 0; m < M; m++) {
      out.push([]);
      for (let n = 0; n < N; n++) out[m].push(body.slice((m * N + n) * perCell, (m * N + n + 1) * perCell));
    }
    return out;
  };
  guard('rotated-B-is-the-same-map', () => {
    const a = V.mmaWarpAtom('f16bf16', 16);
    const B = a.B, N = 8, K = 16;
    // Every cell of an MMA atom has exactly one owner, so 3 labels each: T, V,
    // and the `value` overlay.
    const normal = tvCells(
      V.buildTVSVG(B.shape, B.stride, B.tile.shape, B.tile.stride, false, 'col', null, 'value'),
      N, K, 3);
    const rotated = tvCells(
      V.buildTVSVG(B.shape, B.stride, [K, N], [N, 1], false, 'col', null, 'value',
                   { cellIndex: (m, n) => n + N * m }),
      K, N, 3);
    let mismatched = 0;
    for (let n = 0; n < N; n++)
      for (let k = 0; k < K; k++)
        if (normal[n][k].join('|') !== rotated[k][n].join('|')) mismatched++;
    check('rotated-B-is-the-same-map', 'cells that disagree under transpose', mismatched, 0);
    // ... and the `value` overlay must still be the LAYOUT's output, not the
    // rotated grid's col-major position. Without the cellIndex override
    // buildTVSVG would print k + n*K here.
    let wrongValue = 0;
    for (let k = 0; k < K; k++)
      for (let n = 0; n < N; n++)
        if (rotated[k][n][2] !== String(n + N * k)) wrongValue++;
    check('rotated-B-is-the-same-map', 'value overlay stays the layout output', wrongValue, 0);
    check('rotated-B-is-the-same-map', 'rotated grid is K rows', rotated.length, K);
    check('rotated-B-is-the-same-map', 'rotated grid is N cols', rotated[0].length, N);
  });
  guard('transpose-grid-is-exact', () => {
    // make_tiled_mma rotates by transposing the GRID, so the cell objects must
    // survive identically — same entries, same Rest index, just re-indexed.
    const atom = V.mmaWarpAtom('f16bf16', 16);
    const r = V.mtmComputeTiledMma(atom, V.mtmParseAtomLayout('(2,2,1)'), [null, null, null]);
    const g = V.mtmOperandGrid(r.B);
    const t = V.mtmTransposeGrid(g);
    check('transpose-grid-is-exact', 'rows become cols', `${t.length}x${t[0].length}`,
          `${g[0].length}x${g.length}`);
    let moved = 0;
    for (let m = 0; m < g.length; m++)
      for (let n = 0; n < g[0].length; n++) if (g[m][n] !== t[n][m]) moved++;
    check('transpose-grid-is-exact', 'every cell is the same object', moved, 0);
    // The value overlay again: mtmBuildSVG's default would be col-major over
    // the TRANSPOSED shape, which is not the layout's output.
    const Nb = r.B.tile[0];
    const svg = V.mtmBuildSVG(t, t.length, t[0].length,
                              { mode: 'warp', focus: null, sum: false, showValue: true,
                                cellIndex: (m, n) => n + Nb * m });
    const nT = V.product(r.B.thr.shape), nV = V.product(r.B.val.shape);
    const real = new Set();
    for (let th = 0; th < nT; th++) {
      const base = r.B.thr.call(th);
      for (let v = 0; v < nV; v++) real.add(base + r.B.val.call(v));
    }
    let offGrid = 0;
    for (let k = 0; k < t.length; k++)
      for (let n = 0; n < t[0].length; n++) if (!real.has(n + Nb * k)) offGrid++;
    check('transpose-grid-is-exact', 'every drawn cell is a real offset', offGrid, 0);
    check('transpose-grid-is-exact', 'it rendered', /<svg/.test(svg), true);
  });

  // ── The load-order hazard CLAUDE.md warns about ────────────────────────────
  setSection('unit/load-order');
  guard('layout-js-wins', () => {
    // layout.js loads after cute.js, so on a name clash layout.js wins silently.
    // These two are the known clashes; the assertion pins the intended winner so
    // a new global in cute.js cannot quietly change which one every tab calls.
    check('layout-js-wins', 'product((2,3)) is layout.js\'s', V.product([2, 3]), 6);
    check('layout-js-wins', 'crd2idx takes (crd, shape, stride)',
          V.crd2idx([1, 2], [4, 4], [4, 1]), 6);
  });
}

module.exports = { runUnitTests };
