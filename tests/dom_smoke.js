// ═══════════════════════════════════════════════════════
//  DOM smoke test: run every tab's REAL render path
//
//  tests/run.js diffs the math against CuTeDSL. It cannot catch the other half
//  of a tab: a mistyped element id, a preset that throws, a <select> whose
//  options are built after its value is assigned, a pane that never gets an
//  SVG. Those are invisible until someone opens the page — and, for
//  applyKeyParam, until someone opens a shared LINK, which is worse.
//
//  So this builds a DOM shim rich enough to run the actual render functions,
//  then drives:
//    - initCopyPanes + renderAllTabs, i.e. exactly what addOuterTab does
//    - every preset button found in the generated markup
//    - every ?key= URL round-trip through parseKeyParam
//  and fails on a thrown error, a populated error box, an element id the
//  templates never declared, or an empty visualization pane.
//
//  Node-only, no GPU, no browser. Runs from `npm test`.
// ═══════════════════════════════════════════════════════

'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { sourceFiles, ROOT } = require('./harness');

const TAB = 'tab1';

/** A DOM element stub with just enough behaviour for the render paths. */
function makeEl(id) {
  const classes = new Set();
  return {
    id, style: {}, dataset: {}, value: '', textContent: '', innerHTML: '', disabled: false,
    classList: {
      add: (c) => classes.add(c),
      remove: (c) => classes.delete(c),
      toggle: (c, on) => { if (on === undefined) { classes.has(c) ? classes.delete(c) : classes.add(c); }
                           else if (on) classes.add(c); else classes.delete(c); },
      contains: (c) => classes.has(c),
    },
    setAttribute(k, v) { this.dataset[k] = v; },
    removeAttribute() {},
    getAttribute(k) { return this.dataset[k] !== undefined ? this.dataset[k] : null; },
    appendChild() {}, addEventListener() {}, remove() {},
    querySelector() { return null; }, querySelectorAll() { return []; }, closest() { return null; },
  };
}

function makeContext() {
  // Concatenate the sources exactly as the browser loads them, then copy the
  // top-level lexical names onto globalThis — same trick as harness.js, which
  // cannot be reused directly because it installs a null-returning DOM.
  let src = sourceFiles().map(f => fs.readFileSync(path.join(ROOT, f), 'utf8')).join('\n');
  const lexical = [...new Set(
    [...src.matchAll(/^(?:const|let|class)\s+([A-Za-z_$][\w$]*)/gm)].map(m => m[1]))];
  src += `\n;Object.assign(globalThis, { ${lexical.join(', ')} });\n`;

  const missing = new Set();
  const els = new Map();
  const document = {
    addEventListener() {}, removeEventListener() {},
    getElementById(id) {
      // An id the templates never declared is the bug this exists to catch, so
      // record it rather than silently vending a fresh stub.
      if (!els.has(id)) { missing.add(id); els.set(id, makeEl(id)); }
      return els.get(id);
    },
    querySelector() { return null; }, querySelectorAll() { return []; },
    createElement() { return makeEl('created'); },
    body: makeEl('body'), documentElement: makeEl('html'),
  };
  const ctx = {
    console, document,
    navigator: { platform: 'x86_64', userAgent: 'node' },
    location: { search: '', origin: 'http://localhost', pathname: '/' },
    URLSearchParams, setTimeout,
  };
  ctx.window = ctx;
  ctx.globalThis = ctx;
  vm.createContext(ctx);
  vm.runInContext(src, ctx);
  return { ctx, els, missing };
}

/** A browser decodes character references while parsing an attribute, so an
 *  onclick argument written `'Step&lt;_1, X, _1&gt;'` reaches the handler as
 *  `Step<_1, X, _1>`. Reading the raw markup does not, and the difference is
 *  exactly the kind of thing this file exists to catch — in the wrong
 *  direction, since it would fail a preset that works in the browser. */
function decodeEntities(str) {
  return str.replace(/&(lt|gt|amp|quot|apos|#39|nbsp|mdash|hellip|times|rarr);/g, (_, e) => ({
    lt: '<', gt: '>', amp: '&', quot: '"', apos: "'", '#39': "'",
    nbsp: '\u00a0', mdash: '\u2014', hellip: '\u2026', times: '\u00d7', rarr: '\u2192',
  }[e]));
}

/** The arguments an inline handler would actually pass, obtained by EVALUATING
 *  it with its target stubbed — which is what a browser does.
 *
 *  Picking the arguments apart with a regex instead cannot work, because the
 *  markup is two decodings away from the values: the browser first resolves HTML
 *  character references in the attribute, then the JS parser resolves backslash
 *  escapes in the string literals. A multi-line tiler preset is written `\\n` in
 *  the template literal precisely so the handler's own string carries `\n`, and
 *  a regex hands back the two characters where the browser hands back a newline.
 *  Evaluating gets both layers right for free, and forever. */
function handlerArgs(ctx, src, fnName) {
  const real = ctx[fnName];
  let captured = null;
  ctx[fnName] = (...args) => { captured = args; };
  try { vm.runInContext(src, ctx); } finally { ctx[fnName] = real; }
  if (!captured) throw new Error(`handler never called ${fnName}()`);
  return captured.slice(1);           // drop the tab id
}

function runDomSmoke({ verbose = false, log = console.log } = {}) {
  const { ctx, els, missing } = makeContext();
  const results = { pass: 0, fail: 0, failures: [] };

  const html = ctx.generateTabContent(TAB);
  for (const m of html.matchAll(/\bid="([^"]+)"/g)) els.set(m[1], makeEl(m[1]));
  els.set(`${TAB}-label`, makeEl(`${TAB}-label`));   // owned by addOuterTab, not a tab template

  // Every status div, so a render's error output is readable rather than a
  // freshly-minted stub.
  const errorIds = [...html.matchAll(/id="([^"]*-(?:error|warning))"/g)].map(m => m[1]);
  const paneIds = [...html.matchAll(/id="([^"]*-(?:src|dst|[abc])-svg)"/g)].map(m => m[1]);

  // Seed every control with the default the markup declares. A real browser does
  // this for free; without it the tabs render from empty strings and every
  // failure below is the shim's, not the tab's.
  //   <input ... value="X">              -> X
  //   <textarea ...>X</textarea>         -> X (the tiler / layout boxes)
  //   <select ...><option value="X" selected>  -> X, else the FIRST option
  for (const m of html.matchAll(/<input\b[^>]*\bid="([^"]+)"[^>]*>/g)) {
    const v = /\bvalue="([^"]*)"/.exec(m[0]);
    if (v && els.has(m[1])) els.get(m[1]).value = v[1];
  }
  for (const m of html.matchAll(/<textarea\b[^>]*\bid="([^"]+)"[^>]*>([\s\S]*?)<\/textarea>/g)) {
    if (els.has(m[1])) els.get(m[1]).value = m[2].trim();
  }
  for (const m of html.matchAll(/<select\b[^>]*\bid="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
    const opts = [...m[2].matchAll(/<option\s+value="([^"]*)"([^>]*)>/g)];
    if (!opts.length || !els.has(m[1])) continue;
    const chosen = opts.find(o => /\bselected\b/.test(o[2])) || opts[0];
    els.get(m[1]).value = chosen[1];
  }

  function step(label, fn, wantSvgs) {
    missing.clear();
    for (const id of errorIds) if (els.has(id)) els.get(id).textContent = '';
    let threw = null;
    try { fn(); } catch (e) { threw = e.message; }
    const errs = errorIds.filter(id => id.endsWith('-error'))
                         .map(id => (els.get(id) || {}).textContent || '').filter(Boolean);
    const undeclared = [...missing].filter(i => i.startsWith(TAB));
    const empty = (wantSvgs || []).filter(id => !((els.get(id) || {}).innerHTML || '').includes('<svg'));
    const ok = !threw && !errs.length && !undeclared.length && !empty.length;
    if (ok) {
      results.pass++;
      if (verbose) log(`  ok    ${label}`);
      return;
    }
    results.fail++;
    const why = [
      threw && `threw: ${threw}`,
      errs.length && `error box: ${errs[0].slice(0, 140)}`,
      undeclared.length && `undeclared ids: ${undeclared.join(', ')}`,
      empty.length && `empty pane(s): ${empty.join(', ')}`,
    ].filter(Boolean).join(' | ');
    results.failures.push(`${label}: ${why}`);
    log(`  FAIL  ${label}\n        ${why}`);
  }

  // ── what addOuterTab does on panel creation ──────────────────────────────
  step('initCopyPanes', () => ctx.initCopyPanes(TAB));
  step('renderAllTabs (every tab, shipped defaults)', () => ctx.renderAllTabs(TAB, 'layout'), paneIds);

  // ── every inline handler must at least PARSE ─────────────────────────────
  // A handler that is not valid JS is dead on the page: the browser reports a
  // SyntaxError and the button silently does nothing. Two local_tile presets
  // shipped that way for four commits, because a `\n` in the template literal
  // put a real newline inside the onclick's single-quoted string. Cheap to
  // check, and it covers every handler rather than only the ones driven below.
  const handlers = [...html.matchAll(/\son(?:click|input|change)="([\s\S]*?)"/g)]
    .map(m => decodeEntities(m[1]));
  step(`every inline handler parses as JS (${handlers.length})`, () => {
    const bad = handlers.filter(h => { try { new Function(h); return false; } catch (e) { return true; } });
    if (bad.length) throw new Error(`${bad.length} unparseable: ${bad[0].slice(0, 90)}`);
  });

  // ── every preset button in the generated markup ──────────────────────────
  const presetSrc = new Set(
    [...html.matchAll(/class="preset-btn"[^>]*onclick="([\s\S]*?)"/g)].map(m => decodeEntities(m[1])));
  const ran = new Set();
  for (const src of handlers) {
    const hit = /^\s*(set[A-Z][A-Za-z]*)\('tab1'\s*,/.exec(src);
    if (!hit || typeof ctx[hit[1]] !== 'function') continue;
    const fnName = hit[1];
    const pfx = { setMCA: 'mca', setMMA: 'mma', setMTC: 'mtc', setMTV: 'mtv' }[fnName];
    const svgs = pfx ? paneIds.filter(id => id.startsWith(`${TAB}-${pfx}-`)) : [];
    let args;
    try { args = handlerArgs(ctx, src, fnName); } catch (e) { args = null; }
    if (!args) { step(`${fnName} (arg capture)`, () => { throw new Error(`could not evaluate: ${src.slice(0, 90)}`); }); continue; }
    ran.add(src);
    const shown = args.slice(0, 3).map(a => String(a).replace(/\n/g, '\\n'));
    step(`${fnName}(${shown.join(', ')}${args.length > 3 ? ', …' : ''})`,
         () => ctx[fnName](TAB, ...args), svgs);
  }

  // Silently skipping a preset is how the two broken ones survived: the old
  // matcher's `.` does not cross a newline, so it never saw them at all. Every
  // preset button must be accounted for, not merely the ones a regex liked.
  step(`every preset button ran (${presetSrc.size})`, () => {
    const missed = [...presetSrc].filter(src => !ran.has(src));
    if (missed.length) throw new Error(`${missed.length} preset(s) never ran: ${missed[0].slice(0, 90)}`);
  });

  // ── the highlight-thread controls ────────────────────────────────────────
  // Live `oninput` handlers that re-render, so nothing else here reaches them:
  // every control is seeded from its markup default and the field ships empty.
  // Both an in-range id and an out-of-range one are driven, since the latter
  // takes the warn-and-show-everything path rather than the filter path.
  for (const [pfx, setter] of [['tv', 'setHighlightTid'], ['mtc', 'setMtcHighlight'],
                               ['mtv', 'setMtvHighlight']]) {
    const field = els.get(`${TAB}-${pfx}-highlight-tid`);
    if (!field || typeof ctx[setter] !== 'function') continue;
    const svgs = paneIds.filter(id => id.startsWith(`${TAB}-${pfx}-`));
    for (const tid of ['0', '3', '9999', '']) {
      step(`${setter}(${tid === '' ? 'cleared' : `T${tid}`})`, () => {
        field.value = tid;
        ctx[setter](TAB);
      }, svgs);
    }
  }

  // ── local_tile normalizes its proj / coord fields in place ───────────────
  // The write-back is the half a pure test cannot see: ltComputeLocalTile can
  // return the right strings while renderLocalTile puts them in the wrong box.
  {
    const proj = els.get(`${TAB}-lt-proj-input`);
    const coord = els.get(`${TAB}-lt-coord-input`);
    const tiler = els.get(`${TAB}-lt-tiler-input`);
    const a = els.get(`${TAB}-lt-a-input`);
    step('renderLocalTile normalizes proj/coord in place', () => {
      a.value = '(32, 64):(64, 1)';
      tiler.value = '(8, 16, 4)';
      coord.value = '(1, 2, 3)';
      proj.value = '(2, None, 7)';
      ctx.renderLocalTile(TAB);
      if (proj.value !== '(1, _, 1)') throw new Error(`proj field = ${proj.value}`);
      if (coord.value !== '(1, _, 3)') throw new Error(`coord field = ${coord.value}`);
    }, [`${TAB}-lt-a-svg`]);
    step('renderLocalTile leaves both fields alone without a proj', () => {
      a.value = '(16, 16):(16, 1)';
      tiler.value = '(4, 4)';
      coord.value = '(1, 2)';
      proj.value = '';
      ctx.renderLocalTile(TAB);
      if (proj.value !== '') throw new Error(`proj field = ${proj.value}`);
      if (coord.value !== '(1, 2)') throw new Error(`coord field = ${coord.value}`);
    }, [`${TAB}-lt-a-svg`]);
    step('a rejected proj is left on screen to be fixed', () => {
      proj.value = '(1, q, 1)';
      ctx.renderLocalTile(TAB);
      if (proj.value !== '(1, q, 1)') throw new Error(`proj field = ${proj.value}`);
      proj.value = '';                       // leave the tab renderable for later steps
      ctx.renderLocalTile(TAB);
    });
  }

  // ── URL round-trip: every ?key= form must parse ──────────────────────────
  for (const key of [
    'layout-(10,10):(1,10)',
    'make_copy_atom-universal-128-half_t',
    'make_copy_atom-ldmatrix-128-half_t-4-1',
    'make_copy_atom-g2r-128-half_t',
    'make_mma_atom-f16bf16-half_t-float-16',
    'make_mma_atom-tf32-na-na-8',
    'local_tile-(16, 16):(16, 1)-(4, 4)-(1, 2)',
    'local_tile-(32, 64):(64, 1)-(8, 16, 4)-(1, 2, _)-(1, X, 1)',
  ]) {
    step(`?key=${key}`, () => {
      ctx.location.search = `?key=${key}`;
      const parsed = ctx.parseKeyParam();
      if (!parsed) throw new Error('parseKeyParam returned null — FEATURE_SPEC arity mismatch?');
    });
  }

  return results;
}

module.exports = { runDomSmoke };

if (require.main === module) {
  const r = runDomSmoke({ verbose: process.argv.includes('-v') });
  console.log(`\n${r.fail ? 'FAIL' : 'PASS'} ${r.pass}/${r.pass + r.fail} dom-smoke checks`);
  process.exit(r.fail ? 1 : 0);
}
