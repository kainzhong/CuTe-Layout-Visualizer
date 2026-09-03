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

  // ── unpack_bits vs tensor_dtype ────────────────────────────────────────────
  // Both are arguments to the same make_copy_atom call and they describe
  // OPPOSITE SIDES of the copy, but CuTeDSL relates them not at all: verified
  // byte-for-byte identical layouts across unpack_bits in {None, 4, 6} for
  // every dtype, on all three 8-bit Ops. So a disagreeing pair is accepted in
  // silence and the DSL cannot be the oracle for it.
  setSection('unit/mcaUnpackBitsIssue');
  guard('coherent-pairs-pass', () => {
    // 8-bit dtype is the container these Ops unpack INTO -- the only coherent
    // choice, and the only one that must stay silent.
    check('mcaUnpackBitsIssue', 'int8 + unpack 4', V.mcaUnpackBitsIssue(8, 4), '');
    check('mcaUnpackBitsIssue', 'int8 + unpack 6', V.mcaUnpackBitsIssue(8, 6), '');
    // No unpacking requested: dtype is unconstrained by this check.
    check('mcaUnpackBitsIssue', 'no unpack, half_t', V.mcaUnpackBitsIssue(16, 0), '');
    check('mcaUnpackBitsIssue', 'no unpack, double', V.mcaUnpackBitsIssue(64, 0), '');
  });
  guard('mismatched-pairs-report', () => {
    for (const [e, ub] of [[16, 4], [16, 6], [32, 4], [64, 6]]) {
      const msg = V.mcaUnpackBitsIssue(e, ub);
      check('mcaUnpackBitsIssue', `${e}-bit dtype + unpack ${ub} reports`, msg !== '', true);
      check('mcaUnpackBitsIssue', `${e}-bit dtype + unpack ${ub} names both`,
            msg.includes('unpack_bits') && msg.includes('tensor_dtype'), true);
    }
  });
  guard('padding-arithmetic', () => {
    // The reason no layout changes: 16 packed elements plus their padding
    // occupy exactly the 128-bit row that 16 bytes do. If this stopped holding,
    // the "unpack_bits changes nothing" claim would stop holding with it.
    for (const [ub, pad] of [[4, 64], [6, 32], [8, 0]]) {
      check('unpack padding', `16 x ${ub}b + ${pad}b pad`, 16 * ub + pad, 128);
    }
  });
  guard('sub-byte-dtypes-absent', () => {
    // A sub-byte tensor_dtype would model a DENSELY packed source (32 elements
    // per row at 4 bits), which is not what .b4x16_p64 reads, and a 6-bit one
    // is not even representable -- CuTeDSL returns a 22-element row, i.e. 132
    // bits in a 128-bit row, without complaint. The dropdown offers no width
    // that fails to divide the row, which is what keeps that unreachable here.
    for (const [name, bits] of Object.entries(V.DTYPE_BITS)) {
      check('DTYPE_BITS', `${name} (${bits}b) divides the 128b row`, 128 % bits, 0);
    }
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
