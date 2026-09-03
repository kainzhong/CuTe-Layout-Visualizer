// ═══════════════════════════════════════════════════════
//  Test harness — load the browser globals into node
//
//  The visualizer has no module system: every file is a plain <script> whose
//  top-level declarations land on `window`. Function declarations do end up on
//  the global object, but `const` / `let` / `class` (Layout, DTYPE_BITS,
//  MCA_OPS, ...) only reach the script's *lexical* scope, which is invisible
//  from outside. So we cannot load the files one-by-one into a vm context and
//  read them off it.
//
//  Instead we CONCATENATE every source into a single script and append an
//  epilogue that copies the lexical names onto globalThis. Concatenation is
//  semantically equivalent to the browser's load order for this codebase:
//   - function redeclarations (cute.js and layout.js both define `product`,
//     `crd2idx`) hoist so the LAST one wins, which is exactly what happens in
//     the browser when the second script overwrites the first's global.
//   - there are no duplicate lexical (const/let/class) names across the files;
//     `assertNoDuplicateLexicals` below fails loudly if one is ever introduced,
//     since that would be a SyntaxError here and a silent shadow in the browser.
// ═══════════════════════════════════════════════════════

'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.resolve(__dirname, '..');

/** Load order, mirroring the <script> tags at the bottom of index.html. */
function sourceFiles() {
  const tabs = fs.readdirSync(path.join(ROOT, 'tabs'))
    .filter(f => f.endsWith('.js'))
    .sort()
    .map(f => path.join('tabs', f));
  return ['cute.js', 'layout.js', 'ui.js', ...tabs];
}

/** A DOM stub big enough for the sources to LOAD. The tests only ever call
 *  DOM-free functions; anything that touches an element gets a null back and
 *  would throw, which is the behaviour we want (a loud failure, not a silent
 *  pass on a stubbed-out render). */
function makeDomStub() {
  const noopEl = {
    style: {}, dataset: {}, value: '', textContent: '', innerHTML: '',
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    setAttribute() {}, removeAttribute() {}, getAttribute() { return null; },
    appendChild() {}, addEventListener() {}, querySelector() { return null; },
    querySelectorAll() { return []; },
  };
  return {
    addEventListener() {},
    removeEventListener() {},
    getElementById() { return null; },
    querySelector() { return null; },
    querySelectorAll() { return []; },
    createElement() { return Object.assign({}, noopEl); },
    body: noopEl,
    documentElement: noopEl,
  };
}

function assertNoDuplicateLexicals(files) {
  const seen = new Map();
  const dups = [];
  for (const f of files) {
    const src = fs.readFileSync(path.join(ROOT, f), 'utf8');
    for (const m of src.matchAll(/^(?:const|let|class)\s+([A-Za-z_$][\w$]*)/gm)) {
      if (seen.has(m[1])) dups.push(`${m[1]} (${seen.get(m[1])} and ${f})`);
      else seen.set(m[1], f);
    }
  }
  if (dups.length) {
    throw new Error(
      'Duplicate top-level const/let/class across source files:\n  ' +
      dups.join('\n  ') +
      '\nIn the browser the second declaration silently overwrites the first ' +
      '(see the load-order warning in CLAUDE.md). Rename one of them.');
  }
  return [...seen.keys()];
}

let CACHED = null;

/** Returns the sandbox holding every global the visualizer defines. */
function loadVisualizer() {
  if (CACHED) return CACHED;

  const files = sourceFiles();
  const lexicalNames = assertNoDuplicateLexicals(files);

  const parts = files.map(f =>
    `\n//#region ${f}\n` + fs.readFileSync(path.join(ROOT, f), 'utf8') + `\n//#endregion ${f}\n`);
  // Copy the lexical bindings onto the global object so the caller can see them.
  parts.push(`\n;Object.assign(globalThis, { ${lexicalNames.join(', ')} });\n`);

  const ctx = {
    console,
    document: makeDomStub(),
    navigator: { platform: 'x86_64', userAgent: 'node' },
    location: { search: '', href: 'http://localhost/' },
    history: { replaceState() {} },
    setTimeout, clearTimeout, requestAnimationFrame: (fn) => fn(),
    Blob: class {}, URL: { createObjectURL() { return ''; }, revokeObjectURL() {} },
  };
  ctx.window = ctx;
  ctx.globalThis = ctx;
  vm.createContext(ctx);
  vm.runInContext(parts.join(''), ctx, { filename: 'visualizer-bundle.js' });

  CACHED = ctx;
  return ctx;
}

// ═══════════════════════════════════════════════════════
//  Layout helpers used by the tests
// ═══════════════════════════════════════════════════════

/** Parse a CuTe layout string into a `Layout` WITHOUT parseLayout's rank-2
 *  padding, so `8:1` stays rank 1 and `((2,2),4):((1,8),2)` keeps its nesting.
 *  Built from the same primitives the tabs use (`parseValue`, `topLevelColon`,
 *  `prefix_product`), so the parser under test is still the real one.
 *  `basis` opts in to scaled-basis (`k@i`) strides. */
function parseExact(V, str, opts) {
  const s = String(str).trim();
  const ci = V.topLevelColon(s);
  if (ci === -1) {
    const shape = V.parseValue(s, false);
    return new V.Layout(shape, V.prefix_product(shape));
  }
  const shape = V.parseValue(s.slice(0, ci).trim(), false);
  const stride = V.parseValue(s.slice(ci + 1).trim(), !!(opts && opts.basis));
  return new V.Layout(shape, stride);
}

/** A case's tiler argument: a string is a single layout tiler, an array is a
 *  by-mode tiler (a tuple of layouts, CuTe's `Tiler`). */
function parseTiler(V, spec) {
  if (Array.isArray(spec)) return spec.map(s => parseExact(V, s));
  return parseExact(V, spec);
}

/** Canonical, whitespace-free layout string — the comparison key against
 *  CuTeDSL's printout. */
function fmt(V, x) {
  if (x === null || x === undefined) return String(x);
  if (typeof x === 'number') return String(x);
  if (Array.isArray(x)) return '(' + x.map(e => fmt(V, e)).join(',') + ')';
  if (x instanceof V.Layout) return V.formatLayoutStr(x.shape, x.stride).replace(/\s+/g, '');
  if (x && x.basis === true) return `${x.k}@${x.axis}`;
  return String(x);
}

/** Evaluate a layout at every point of its domain (capped). Returns strings so
 *  ordinary offsets and basis coordinates compare uniformly.
 *
 *  A coordinate (basis-strided) layout maps to a COORDINATE, so it goes through
 *  cute.js's `crd2basis` -- which is what the tabs use for these -- rather than
 *  layout.js's integer `crd2idx`, which would produce NaN. The result is emitted
 *  as "a|b", matching gen_reference.py. */
function evalAll(V, L, cap) {
  const n = Math.min(L.size(), cap === undefined ? 4096 : cap);
  const out = [];
  if (V.has_basis_stride(L.stride)) {
    const nd = Math.max(1, V.basisRank(L.stride));
    for (let i = 0; i < n; i++) {
      const acc = new Array(nd).fill(0);
      V.crd2basis(V.idx2crd(i, L.shape), L.shape, L.stride, acc);
      out.push(acc.join('|'));
    }
    return out;
  }
  for (let i = 0; i < n; i++) out.push(fmt(V, L.call(i)));
  return out;
}

module.exports = { loadVisualizer, parseExact, parseTiler, fmt, evalAll, sourceFiles, ROOT };
