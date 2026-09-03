#!/usr/bin/env node
// ═══════════════════════════════════════════════════════
//  Differential tests: the JS port vs CuTeDSL
//
//  Every layout op in layout.js / cute.js and every copy construction in
//  tabs/*.js is a port of something in `cutlass.cute`. This runs the shared
//  corpus (tests/cases.json) through the JS port and diffs the result against
//  tests/reference.json, which tests/gen_reference.py produced by running the
//  SAME corpus through CuTeDSL.
//
//      node tests/run.js                # run everything
//      node tests/run.js layout_ops     # run only sections matching a filter
//      node tests/run.js --verbose      # print every passing case too
//
//  Regenerating the reference needs CuTeDSL installed; running these tests does
//  not, which is why reference.json is committed.
// ═══════════════════════════════════════════════════════

'use strict';

const fs = require('fs');
const path = require('path');
const { loadVisualizer, parseExact, parseTiler, fmt, evalAll } = require('./harness');
const { runUnitTests } = require('./unit');

const V = loadVisualizer();
const CASES = JSON.parse(fs.readFileSync(path.join(__dirname, 'cases.json'), 'utf8'));
const REF = JSON.parse(fs.readFileSync(path.join(__dirname, 'reference.json'), 'utf8'));

const argv = process.argv.slice(2);
const VERBOSE = argv.includes('--verbose') || argv.includes('-v');
const FILTERS = argv.filter(a => !a.startsWith('-'));

// ═══════════════════════════════════════════════════════
//  Tiny test framework
// ═══════════════════════════════════════════════════════

const C = process.stdout.isTTY
  ? { red: '\x1b[31m', green: '\x1b[32m', dim: '\x1b[2m', bold: '\x1b[1m', off: '\x1b[0m' }
  : { red: '', green: '', dim: '', bold: '', off: '' };

const results = { pass: 0, fail: 0, skip: 0, failures: [] };
let currentSection = '';

function section(name) {
  currentSection = name;
  if (!FILTERS.length || FILTERS.some(f => name.includes(f))) {
    process.stdout.write(`\n${C.bold}${name}${C.off}\n`);
    return true;
  }
  return false;
}

/** One assertion. `what` names the quantity so a failure says which field drifted. */
function check(id, what, actual, expected) {
  const a = String(actual), e = String(expected);
  if (a === e) {
    results.pass++;
    if (VERBOSE) console.log(`  ${C.green}ok${C.off} ${id} ${C.dim}${what} = ${a}${C.off}`);
    return true;
  }
  results.fail++;
  const msg = `${currentSection}/${id}: ${what}\n      js  = ${a}\n      cute= ${e}`;
  results.failures.push(msg);
  console.log(`  ${C.red}FAIL${C.off} ${id} ${C.dim}${what}${C.off}\n      js  = ${a}\n      cute= ${e}`);
  return false;
}

/** Run `fn`, reporting a throw as a failure rather than aborting the suite. */
function guard(id, fn) {
  try {
    fn();
  } catch (e) {
    results.fail++;
    const msg = `${currentSection}/${id}: threw ${e.message}`;
    results.failures.push(msg);
    console.log(`  ${C.red}THREW${C.off} ${id}: ${e.message}`);
  }
}

function refFor(sectionName, id) {
  const r = (REF[sectionName] || {})[id];
  if (!r) {
    results.skip++;
    console.log(`  ${C.dim}skip${C.off} ${id} (no reference — rerun tests/gen_reference.py)`);
  }
  return r;
}

// ═══════════════════════════════════════════════════════
//  Shared comparison for a layout-valued result
// ═══════════════════════════════════════════════════════

/** Compare a produced Layout against the reference record: the printed layout,
 *  its size/cosize, and its value at every point of the domain. The pointwise
 *  check is the one that matters — two different strings can describe the same
 *  map, but a divergent value is always a real bug. */
function checkLayout(id, L, ref) {
  check(id, 'layout', fmt(V, L), ref.str);
  if (ref.size !== undefined) check(id, 'size', L.size(), ref.size);
  if (ref.cosize !== undefined) check(id, 'cosize', L.cosize(), ref.cosize);
  if (ref.eval) {
    const got = evalAll(V, L, ref.eval.length);
    const bad = got.findIndex((x, i) => x !== ref.eval[i]);
    if (bad === -1 && got.length === ref.eval.length) {
      results.pass++;
      if (VERBOSE) console.log(`  ${C.green}ok${C.off} ${id} ${C.dim}eval[${got.length}]${C.off}`);
    } else if (got.length !== ref.eval.length) {
      check(id, 'eval length', got.length, ref.eval.length);
    } else {
      check(id, `eval[${bad}]`, got[bad], ref.eval[bad]);
    }
  }
}

// ═══════════════════════════════════════════════════════
//  1. Layout algebra  (layout.js, the pycute port)
// ═══════════════════════════════════════════════════════

const UNARY = ['coalesce', 'filter', 'right_inverse', 'left_inverse'];
const BINARY = ['composition', 'logical_divide', 'zipped_divide', 'tiled_divide',
                'flat_divide', 'logical_product', 'zipped_product', 'tiled_product',
                'flat_product', 'blocked_product', 'raked_product'];

/** JSON coord -> JS coord. null is the slice marker, matching layout.js's `slice_`. */
function toCrd(x) { return Array.isArray(x) ? x.map(toCrd) : x; }

function runLayoutOps(sectionName, basis) {
  if (!section(sectionName)) return;
  for (const c of CASES[sectionName]) {
    const ref = refFor(sectionName, c.id);
    if (!ref) continue;
    guard(c.id, () => {
      const opts = { basis };
      if (UNARY.includes(c.op)) {
        // coalesce / filter take an optional target profile, which decides how
        // deep the merge goes (`[1,1]` == CuTe's Shape<Shape<_1,_1>>).
        checkLayout(c.id, V[c.op](parseExact(V, c.a, opts), c.profile), ref);
      } else if (BINARY.includes(c.op)) {
        checkLayout(c.id, V[c.op](parseExact(V, c.a, opts), parseTiler(V, c.b)), ref);
      } else if (c.op === 'complement') {
        checkLayout(c.id, V.complement(parseExact(V, c.a), c.n), ref);
      } else if (c.op === 'size') {
        check(c.id, 'size', V.size(parseExact(V, c.a)), ref.value);
      } else if (c.op === 'cosize') {
        check(c.id, 'cosize', V.cosize(parseExact(V, c.a)), ref.value);
      } else if (c.op === 'slice_and_offset') {
        const [L, off] = V.slice_and_offset(toCrd(c.crd), parseExact(V, c.a));
        check(c.id, 'sliced layout', fmt(V, L), ref.str);
        check(c.id, 'offset', off, ref.offset);
      } else if (c.op === 'product_each') {
        check(c.id, 'product_each', fmt(V, V.product_each(V.parseValue(c.shape))), ref.value);
      } else if (c.op === 'shape_div') {
        check(c.id, 'shape_div',
              fmt(V, V.shape_div(V.parseValue(c.shape), V.parseValue(c.shape_b))), ref.value);
      } else if (c.op === 'idx2crd') {
        check(c.id, 'idx2crd', fmt(V, V.idx2crd(c.idx, V.parseValue(c.shape))), ref.value);
      } else if (c.op === 'crd2idx') {
        const L = parseExact(V, c.a);
        check(c.id, 'crd2idx', fmt(V, V.crd2idx(toCrd(c.crd), L.shape, L.stride)), ref.value);
      } else {
        throw new Error(`run.js does not know op "${c.op}"`);
      }
    });
  }
}

runLayoutOps('layout_ops', false);
runLayoutOps('basis_ops', true);

// ═══════════════════════════════════════════════════════
//  2. make_layout_tv  (the make_tiled_copy_tv tab's derivation)
// ═══════════════════════════════════════════════════════

if (section('make_layout_tv')) {
  for (const c of CASES.make_layout_tv) {
    const ref = refFor('make_layout_tv', c.id);
    if (!ref) continue;
    guard(c.id, () => {
      const thr = parseExact(V, c.thr), val = parseExact(V, c.val);
      const { tiler_mn, layout_tv } = V.make_layout_tv(thr, val);
      check(c.id, 'tiler_mn', fmt(V, tiler_mn), ref.tiler_mn);
      checkLayout(c.id, layout_tv, ref.layout_tv);
      // The tab shows what make_tiled_copy_tv would build; CuTe's TiledCopy must
      // agree with the raw derivation, and the tiler is printed as a Tiler
      // (`(16:1, 64:1)`) rather than a shape.
      check(c.id, 'TiledCopy layout_tv', fmt(V, layout_tv), ref.tiled_copy_layout_tv);
      const tilerAsTiler = '(' + tiler_mn.map(e => `${e}:1`).join(',') + ')';
      check(c.id, 'TiledCopy tiler_mn', tilerAsTiler, ref.tiled_copy_tiler_mn);
    });
  }
}

// ═══════════════════════════════════════════════════════
//  3. make_tiled_copy  (the primitive constructor)
//
//  CuTe stores (layout_tv, Tiler_MN) verbatim, so what is under test here is
//  the tab's READING of them: mtcParseTiler must recover the same tiler CuTe
//  reports, and parseLayout the same layout_tv.
// ═══════════════════════════════════════════════════════

if (section('make_tiled_copy')) {
  for (const c of CASES.make_tiled_copy) {
    const ref = refFor('make_tiled_copy', c.id);
    if (!ref) continue;
    guard(c.id, () => {
      const tv = parseExact(V, c.tv);
      check(c.id, 'layout_tv', fmt(V, tv), ref.layout_tv);
      const tiler = V.mtcParseTiler(c.tiler);
      const asTiler = '(' + tiler.extents.map((e, i) =>
        `${e}:${tiler.strides[i] === null ? 1 : tiler.strides[i]}`).join(',') + ')';
      check(c.id, 'tiler_mn', asTiler, ref.tiler_mn);
      check(c.id, 'atom num_val', c.bits / V.DTYPE_BITS[c.dtype], ref.atom_num_val);
      // layout_tv(tid, vid) -> flat index into the tile is what the whole
      // visualization is built on, so check it pointwise.
      const got = evalAll(V, tv, ref.eval.length);
      const bad = got.findIndex((x, i) => x !== ref.eval[i]);
      if (bad === -1) results.pass++;
      else check(c.id, `layout_tv eval[${bad}]`, got[bad], ref.eval[bad]);
    });
  }
}

// ═══════════════════════════════════════════════════════
//  4. make_copy_atom  (DTYPE_BITS and the ValLayout it implies)
// ═══════════════════════════════════════════════════════

if (section('copy_atom')) {
  for (const c of CASES.copy_atom) {
    const ref = refFor('copy_atom', c.id);
    if (!ref) continue;
    guard(c.id, () => {
      const elemBits = V.DTYPE_BITS[c.dtype];
      if (!elemBits) throw new Error(`DTYPE_BITS has no entry for "${c.dtype}"`);
      const numVal = c.bits / elemBits;
      check(c.id, 'num_val', numVal, ref.num_val);
      // Both current Ops have ValLayoutSrc == ValLayoutDst == (1, N):(0, 1).
      check(c.id, 'layout_src_tv', `(1,${numVal}):(0,1)`, ref.layout_src_tv);
      check(c.id, 'layout_dst_tv', `(1,${numVal}):(0,1)`, ref.layout_dst_tv);
    });
  }
}

// ═══════════════════════════════════════════════════════
//  4b. make_copy_atom(warp.LdMatrix8x8x16bOp(...))  (mcaLdmatrixAtom)
// ═══════════════════════════════════════════════════════

if (section('ldmatrix_atom')) {
  for (const c of CASES.ldmatrix_atom) {
    const ref = refFor('ldmatrix_atom', c.id);
    if (!ref) continue;
    guard(c.id, () => {
      const elemBits = V.DTYPE_BITS[c.dtype];
      if (!elemBits) throw new Error(`DTYPE_BITS has no entry for "${c.dtype}"`);
      const opKey = c.op || 'ldsm8x8x16b';
      const a = V.mcaLdmatrixAtom(opKey, elemBits, c.num_matrices, !!c.transpose);
      check(c.id, 'thr_id', fmt(V, new V.Layout(a.thrId.shape, a.thrId.stride)), ref.thr_id);
      checkLayout(`${c.id}/src`, new V.Layout(a.src.shape, a.src.stride), ref.src);
      checkLayout(`${c.id}/dst`, new V.Layout(a.dst.shape, a.dst.stride), ref.dst);

      // The grid the tab draws must be exactly the atom's codomain -- both
      // panes are placed into `tile`, so a tile that is bigger or smaller than
      // cosize would draw empty cells or silently wrap.
      const [M, N] = a.tile.shape;
      check(c.id, 'tile_cells', M * N, ref.src.cosize);
      check(c.id, 'tile_cells_dst', M * N, ref.dst.cosize);

      // The stride-0 broadcast: an Op consumes (matrixBytes/16)*num_matrices of
      // the 32 lanes, so size exceeds cosize by exactly 32/liveLanes. liveLanes
      // is also the tile's row count -- one lane addresses one 16 B row.
      const spec = V.MCA_LDSM_SPECS[opKey];
      check(c.id, 'live_lanes', a.liveLanes,
            (spec.matrixBytes / 16) * c.num_matrices);
      check(c.id, 'live_lanes_eq_rows', a.liveLanes, M);
      check(c.id, 'src_broadcast', ref.src.size / ref.src.cosize, 32 / a.liveLanes);

      // The Op's own parameter domains, as __post_init__ enforces them. A tab
      // that offered an out-of-domain num_matrices would only produce errors.
      check(c.id, 'num_matrices_legal', spec.numMatrices.includes(c.num_matrices), true);
      if (spec.transpose === 'required') check(c.id, 'transpose_forced', c.transpose, true);
      if (c.unpack_bits !== undefined)
        check(c.id, 'unpack_bits_legal', !!spec.unpackBits, true);
    });
  }
}

// ═══════════════════════════════════════════════════════
//  5. make_tiled_tma_atom  (tmaComputeAtom)
// ═══════════════════════════════════════════════════════

/** Build tmaComputeAtom's cfg exactly as renderMakeTiledTmaAtom does. */
function tmaCfg(c) {
  const elemBits = V.DTYPE_BITS[c.dtype];
  const gp = V.parseLayout(c.gmem);
  const sm = V.tmaParseSmemField(c.smem);
  const sw = c.swizzle === 'none' ? null : V.parseSwizzleSpec(c.swizzle);
  const swInfo = V.tmaSwizzleInfo(sm.sw || sw, elemBits);
  const sp = V.parseLayout(sm.layoutStr);
  const tiler = V.mtcParseTiler(c.tiler);
  return {
    cfg: {
      gShape: gp.shape.slice(), gStride: gp.stride.slice(), elemBits, swInfo,
      sFlatShape: V.flatten(sp.shape), sFlatStride: V.flatten(sp.stride),
      tiler: { extents: tiler.extents,
               strides: tiler.strides.map(s => (s === null ? 1 : parseInt(s, 10))) },
    },
    elemBits, gShape: gp.shape.slice(),
  };
}

if (section('tma_atom')) {
  for (const c of CASES.tma_atom) {
    const ref = refFor('tma_atom', c.id);
    if (!ref) continue;
    guard(c.id, () => {
      const { cfg, gShape } = tmaCfg(c);
      const a = V.tmaComputeAtom(cfg);
      // The atom presents as (1, boxSize) values: the box CuTe inferred is
      // exactly the number of elements one TMA instruction moves.
      check(c.id, 'box size (atom num_val)', a.boxSize, ref.num_val);
      check(c.id, 'num_bits_per_tma', a.numBitsPerTma, ref.num_bits);
      check(c.id, 'layout_src_tv', `(1,${a.boxSize}):(0,1)`, ref.layout_src_tv);
      // The returned TMA coordinate tensor -- basis strides and all.
      check(c.id, 'tma tensor',
            V.formatLayoutStr(gShape, a.tmaTensorStride).replace(/\s+/g, ''),
            ref.tma_tensor);
      // Any host-assert violation the tab reports is a claim about CuTe's own
      // preconditions; a case CuTe accepted must not be flagged as an error.
      const errs = a.issues.filter(i => i.level === 'error');
      check(c.id, 'error-level issues on a case CuTe accepted',
            errs.map(i => i.text).join(' | ') || 'none', 'none');
    });
  }
}

// ═══════════════════════════════════════════════════════
//  6. tma_partition  (tpComputePartition)
// ═══════════════════════════════════════════════════════

/** CuTeDSL prints an SMEM tensor's composed layout with its swizzle prefix
 *  (`S<3,4,3> o 0 o (...)`). tpComputePartition works on the non-swizzled
 *  portion -- which is what CuTe does too, via get_nonswizzle_portion -- so the
 *  prefix is stripped before comparing. */
function stripSwizzlePrefix(s) {
  return s.replace(/^S<\d+,\d+,\d+>o\d+o/, '');
}

if (section('tma_partition')) {
  for (const c of CASES.tma_partition) {
    const ref = refFor('tma_partition', c.id);
    if (!ref) continue;
    guard(c.id, () => {
      // Feed the JS side the tensors CuTe itself produced, in the pre-grouped
      // form the tab asks the user for.
      const sp = V.parseLayout(stripSwizzlePrefix(ref.smem_grouped));
      const gp = V.parseLayout(ref.gmem_grouped, { basis: true });
      const P = V.tpComputePartition(sp, gp, ref.num_val);
      check(c.id, 'tAsA', P.sPartStr.replace(/\s+/g, ''), stripSwizzlePrefix(ref.tAsA));
      check(c.id, 'tAgA', P.gPartStr.replace(/\s+/g, ''), ref.tAgA);
      check(c.id, 'iterations', P.iters, P.tileSize / ref.num_val);
    });
  }
}

// ═══════════════════════════════════════════════════════
//  7. Swizzle  (applySwizzleOffset vs cute::Swizzle<B,M,S>)
// ═══════════════════════════════════════════════════════

if (section('swizzle')) {
  for (const c of CASES.swizzle) {
    const ref = refFor('swizzle', c.id);
    if (!ref) continue;
    guard(c.id, () => {
      const sw = V.parseSwizzleSpec(c.bms);
      if (!sw) throw new Error(`parseSwizzleSpec rejected "${c.bms}"`);
      const L = parseExact(V, c.layout);
      const got = [];
      for (let i = 0; i < ref.plain.length; i++) got.push(L.call(i));
      const badPlain = got.findIndex((x, i) => x !== ref.plain[i]);
      if (badPlain !== -1) {
        check(c.id, `base layout[${badPlain}]`, got[badPlain], ref.plain[badPlain]);
        return;
      }
      const sz = got.map(x => V.applySwizzleOffset(x, sw));
      const bad = sz.findIndex((x, i) => x !== ref.swizzled[i]);
      if (bad === -1) { results.pass++; return; }
      check(c.id, `swizzled[${bad}] (offset ${got[bad]})`, sz[bad], ref.swizzled[bad]);
    });
  }
}

// ═══════════════════════════════════════════════════════
//  8. JS-only unit tests (no CuTeDSL analogue)
// ═══════════════════════════════════════════════════════

if (!FILTERS.length || FILTERS.some(f => 'unit'.includes(f))) {
  currentSection = 'unit';
  process.stdout.write(`\n${C.bold}unit${C.off}\n`);
  runUnitTests(V, { check, guard, C, results, setSection: (n) => { currentSection = n; } });
}

// ═══════════════════════════════════════════════════════

const total = results.pass + results.fail;
console.log(
  `\n${results.fail === 0 ? C.green + 'PASS' : C.red + 'FAIL'}${C.off} ` +
  `${results.pass}/${total} assertions` +
  (results.skip ? `, ${results.skip} cases skipped` : ''));
if (results.fail) {
  console.log(`\n${C.red}${results.fail} failure(s):${C.off}`);
  for (const f of results.failures) console.log('  - ' + f.split('\n')[0]);
}
process.exit(results.fail === 0 ? 0 : 1);
