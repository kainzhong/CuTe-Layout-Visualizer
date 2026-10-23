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
const { sourceFiles, ROOT, createBrowserContext } = require('./harness');

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
  const ctx = createBrowserContext({
    console, document,
    navigator: { platform: 'x86_64', userAgent: 'node' },
    location: { search: '', origin: 'http://localhost', pathname: '/' },
    URLSearchParams, setTimeout,
  });
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
  // Every viz host a render is expected to fill: the Copy tabs' src/dst panes,
  // the MMA tabs' a/b/c grids, and partition_sd's two stacked results.
  const paneIds = [...html.matchAll(/id="([^"]*-(?:src|dst|tile|sweep|extra|frag|strip|mma|copy|[abc])-svg)"/g)].map(m => m[1]);

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

  //  `opts.expectError` inverts the error check for the handful of steps whose
  //  whole point is that a bad input is REPORTED: an illegal warp / thread id
  //  must land in the error box, and a step that quietly passed would mean the
  //  validation had stopped firing.
  function step(label, fn, wantSvgs, opts) {
    opts = opts || {};
    missing.clear();
    for (const id of errorIds) if (els.has(id)) els.get(id).textContent = '';
    let threw = null;
    try { fn(); } catch (e) { threw = e.message; }
    const errs = errorIds.filter(id => id.endsWith('-error'))
                         .map(id => (els.get(id) || {}).textContent || '').filter(Boolean);
    const errProblem = opts.expectError
      ? (errs.some(e => opts.expectError.test(e))
          ? '' : `expected an error matching ${opts.expectError}, got: ${errs[0] || '(none)'}`)
      : (errs.length ? `error box: ${errs[0].slice(0, 140)}` : '');
    const undeclared = [...missing].filter(i => i.startsWith(TAB));
    const empty = (wantSvgs || []).filter(id => !((els.get(id) || {}).innerHTML || '').includes('<svg'));
    const ok = !threw && !errProblem && !undeclared.length && !empty.length;
    if (ok) {
      results.pass++;
      if (verbose) log(`  ok    ${label}`);
      return;
    }
    results.fail++;
    const why = [
      threw && `threw: ${threw}`,
      errProblem,
      undeclared.length && `undeclared ids: ${undeclared.join(', ')}`,
      empty.length && `empty pane(s): ${empty.join(', ')}`,
    ].filter(Boolean).join(' | ');
    results.failures.push(`${label}: ${why}`);
    log(`  FAIL  ${label}\n        ${why}`);
  }

  // ── what addOuterTab does on panel creation ──────────────────────────────
  step('initCopyPanes', () => ctx.initCopyPanes(TAB));
  step('renderAllTabs (every tab, shipped defaults)', () => ctx.renderAllTabs(TAB, 'layout'), paneIds);

  // Space badges describe where the displayed operand lives; their colors are
  // CSS keyed by data-space, while SVG cell colors still identify threads.
  const badge = id => els.get(`${TAB}-${id}-space`);
  const expectBadge = (id, space) => {
    const el = badge(id);
    if (!el || el.textContent !== space || el.getAttribute('data-space') !== space)
      throw new Error(`${id} badge = ${el && el.textContent}/${el && el.getAttribute('data-space')}, expected ${space}`);
  };
  step('Copy atom badges follow the memory movement', () => {
    els.get(`${TAB}-mca-move`).value = 'RMEM>SMEM';
    ctx.setCopyMove(TAB, 'mca');
    expectBadge('mca-src', 'RMEM');
    expectBadge('mca-dst', 'SMEM');
    els.get(`${TAB}-mca-move`).value = 'GMEM>RMEM';
    ctx.setCopyMove(TAB, 'mca');
    expectBadge('mca-src', 'GMEM');
    expectBadge('mca-dst', 'RMEM');
  });
  step('MMA atom and tiled fragment titles name RMEM', () => {
    for (const side of ['a', 'b', 'c']) {
      for (const prefix of [`mma-${side}`, `mtm-warp-${side}`, `mtm-perm-${side}`]) {
        if (!html.includes(`id="${TAB}-${prefix}-space" data-space="RMEM">RMEM</span>`))
          throw new Error(`${prefix} has no RMEM badge`);
      }
    }
  });
  step('memory-space badges have four distinct text colors', () => {
    const css = fs.readFileSync(path.join(ROOT, 'style.css'), 'utf8');
    const colors = ['GMEM', 'SMEM', 'RMEM', 'TMEM'].map(space => {
      const rule = new RegExp(`\\.memory-space-badge\\[data-space="${space}"\\]\\s*\\{([^}]+)\\}`).exec(css);
      const color = rule && /\bcolor:\s*(#[0-9a-f]+)/i.exec(rule[1]);
      if (!color) throw new Error(`${space} has no badge color`);
      return color[1];
    });
    if (new Set(colors).size !== colors.length) throw new Error(`badge colors collide: ${colors}`);
  });
  step('partition_S/D badges follow the selected tensor and move', () => {
    const move = els.get(`${TAB}-psd-move`);
    move.value = 'GMEM>SMEM';
    ctx.setCopyMove(TAB, 'psd');
    ctx.setPsdSide(TAB, 'S');
    for (const level of ['tile', 'sweep', 'extra']) expectBadge(`psd-${level}`, 'GMEM');
    ctx.setPsdSide(TAB, 'D');
    for (const level of ['tile', 'sweep', 'extra']) expectBadge(`psd-${level}`, 'SMEM');
    move.value = 'RMEM>GMEM';
    ctx.setCopyMove(TAB, 'psd');
    for (const level of ['tile', 'sweep', 'extra']) expectBadge(`psd-${level}`, 'GMEM');
    ctx.setPsdSide(TAB, 'S');
    for (const level of ['tile', 'sweep', 'extra']) expectBadge(`psd-${level}`, 'RMEM');
  }, paneIds.filter(id => id.startsWith(`${TAB}-psd-`)));

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

  // ── every .viz-box must be able to carry a fullscreen button ─────────────
  //  attachVizFullscreenButtons looks for a DESCENDANT `div[id]` inside each
  //  .viz-box and silently returns when there isn't one. That silence shipped
  //  three tabs (partition_sd / _abc / _fragment_abc, 8 panels) with no
  //  fullscreen button at all.
  //
  //  The inner div is structural, not cosmetic: the button is appended to the
  //  .viz-box, while every render does `getElementById(hostId).innerHTML = svg`.
  //  Put the id on the .viz-box itself and the host IS the button's parent, so
  //  the first render deletes it — and addOuterTab calls attach BEFORE
  //  renderAllTabs, so it would never survive to be seen.
  step('every .viz-box wraps its SVG host in an inner div[id]', () => {
    const boxes = [...html.matchAll(/<div class="viz-box"([^>]*)>([\s\S]{0,120})/g)];
    const bad = boxes.filter(([, attrs, body]) =>
      /\bid="/.test(attrs) || !/^\s*<div id="/.test(body));
    if (bad.length) {
      throw new Error(
        `${bad.length} of ${boxes.length} viz-box(es) have no inner div[id], so ` +
        `attachVizFullscreenButtons will skip them: ` +
        bad.slice(0, 3).map(b => (b[1] + b[2]).slice(0, 70)).join(' | '));
    }
  });

  step('TiledCopy SRC/DST get fullscreen buttons targeting their SVG hosts', () => {
    const markup = ctx.generateMakeTiledCopyABTabContent(TAB);
    if (!/<div class="comp-viz-item comp-viz-span">\s*<div class="comp-viz-header"><span class="comp-viz-title" id="tab1-mcab-transfer-title"/.test(markup))
      throw new Error('transfer wrapper is excluded from fullscreen setup');
    const boxes = ['src', 'dst'].map(side => {
      const children = [];
      return {
        style: {}, children,
        querySelector(selector) {
          if (selector === 'div[id]') return els.get(`${TAB}-mcab-${side}-svg`);
          if (selector === ':scope > .viz-fullscreen-btn') return children[0] || null;
          return null;
        },
        appendChild(child) { children.push(child); },
      };
    });
    const item = { querySelectorAll: selector => selector === '.viz-box' ? boxes : [] };
    const root = { querySelectorAll: selector => selector === '.comp-viz-item' ? [item] : [] };
    const realCreate = ctx.document.createElement;
    const realView = ctx.viewFullscreen;
    const opened = [];
    ctx.document.createElement = () => ({ style: {}, handlers: {},
      addEventListener(event, fn) { this.handlers[event] = fn; } });
    ctx.viewFullscreen = id => opened.push(id);
    try {
      ctx.attachVizFullscreenButtons(root);
      ctx.attachVizFullscreenButtons(root);
      for (const box of boxes) {
        if (box.children.length !== 1 || box.children[0].className !== 'viz-fullscreen-btn')
          throw new Error('fullscreen button missing or duplicated');
        box.children[0].handlers.click({ stopPropagation() {} });
      }
      if (opened.join(',') !== `${TAB}-mcab-src-svg,${TAB}-mcab-dst-svg`)
        throw new Error('fullscreen buttons opened incorrect hosts');
    } finally {
      ctx.document.createElement = realCreate;
      ctx.viewFullscreen = realView;
    }
  });

  // ── every preset button in the generated markup ──────────────────────────
  const presetSrc = new Set(
    [...html.matchAll(/class="preset-btn"[^>]*onclick="([\s\S]*?)"/g)].map(m => decodeEntities(m[1])));
  const ran = new Set();
  for (const src of handlers) {
    const hit = /^\s*(set[A-Z][A-Za-z]*)\('tab1'\s*,/.exec(src);
    if (!hit || typeof ctx[hit[1]] !== 'function') continue;
    const fnName = hit[1];
    const pfx = { setGemm: 'gemm', setIdx: 'idx', setMCA: 'mca', setMMA: 'mma', setMTC: 'mtc', setMTV: 'mtv', setMTM: 'mtm', setMCAB: 'mcab', setRT: 'rt',
                  setPSD: 'psd', setPSDURL: 'psd', setPsdSide: 'psd', setPsdMode: 'psd',
                  setPABC: 'pabc', setPabcOperand: 'pabc', setPabcMode: 'pabc',
                  setMfrag: 'mfrag', setMfragOperand: 'mfrag' }[fnName];
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

  // The visible result must be the complete layout returned by the selected
  // API, including nested modes and strides. Use the committed CuTeDSL oracle
  // so this checks the displayed result independently of the JS computation.
  const partitionRef = JSON.parse(fs.readFileSync(path.join(ROOT, 'tests/reference.json'), 'utf8'));
  const partitionResults = [
    ['S', 'psd', 'partition_sd', 'basic_2x2', 'partition_S', () =>
      ctx.setPSD(TAB, 'S', 'universal', 32, 'half_t',
                 '((8,4),(2,2)):((16,2),(8,1))', '(8, 16)', '5', '(16, 32):(1, 16)')],
    ['D', 'psd', 'partition_sd', 'basic_2x2_d', 'partition_D', () =>
      ctx.setPSD(TAB, 'D', 'universal', 32, 'half_t',
                 '((8,4),(2,2)):((16,2),(8,1))', '(8, 16)', '5', '(16, 32):(32, 1)')],
    ...[['A', 'A_2x2x1', '(64, 48):(48, 1)'],
        ['B', 'B_2x2x1', '(32, 48):(48, 1)'],
        ['C', 'C_2x2x1', '(64, 32):(32, 1)']].map(([which, ref, tensor]) =>
      [which, 'pabc', 'partition_abc', ref, `partition_${which}`, () =>
        ctx.setPABC(TAB, which, 'f16bf16', 'half_t', 'float', 16,
                    '(2, 2, 1)', '', '5', tensor)]),
  ];
  for (const [which, pfx, section, ref, fn, render] of partitionResults) {
    step(`${fn} displays its returned layout`, () => {
      render();
      const got = els.get(`${TAB}-${pfx}-result`).innerHTML;
      const expected = partitionRef[section][ref].layout.str;
      if (!got.includes(`${fn} = <b>${expected}</b>`))
        throw new Error(`result = ${got}, expected ${fn} = ${expected}`);
    }, paneIds.filter(id => id.startsWith(`${TAB}-${pfx}-`)));
  }

  step('partition_ABC groups the MMA atom and tiling into one TiledMMA section', () => {
    const panelStart = html.indexOf(`id="${TAB}-tab-partition_abc"`);
    const sectionStart = html.indexOf('<summary>1. TiledMMA</summary>', panelStart);
    const section = html.slice(sectionStart, html.indexOf('<summary>2. get_slice(thr_idx)</summary>', sectionStart));
    if (!section) throw new Error('Merged TiledMMA section is missing');
    for (const key of ['op','ab','acc','k','atomlayout','perm'])
      if (!section.includes(`id="${TAB}-pabc-${key}-input"`)) throw new Error(`${key} is outside the TiledMMA section`);
    if (!section.includes(`onclick="pabcDecodePasteboard('${TAB}')"`)) throw new Error('Pasteboard decode is not wired');
    const panel = html.slice(html.indexOf(`id="${TAB}-tab-partition_abc"`), html.indexOf(`id="${TAB}-tab-make_fragment_abc"`));
    for (const title of ['2. get_slice(thr_idx)', '3. The tensor being partitioned'])
      if (!panel.includes(`<summary>${title}</summary>`)) throw new Error(`Missing renumbered section ${title}`);
    if (panel.includes('<summary>1. The MMA Op</summary>') || panel.includes('2. The TiledMMA'))
      throw new Error('Old split constructor sections remain');
  });
  step('partition_ABC decodes real exported TiledMMA URLs and preserves partition inputs', () => {
    ctx.setPABC(TAB, 'B', 'f16bf16', 'half_t', 'float', 16, '(2,2,1)', '', '17', '(64,64,2):(64,1,4096)', '(_,_,_,1)');
    const keep = ['thr','tensor','fragment-slice'].map(key => els.get(`${TAB}-pabc-${key}-input`).value);
    const realExport = ctx.exportURL;
    try {
      for (const [i, args] of [
        ['f16bf16','bfloat16_t','float',16,'(2,2,1):(2,1,4)','(32,32,16)'],
        ['tf32',null,null,8,'(1,1,1)',''],
        ['fp8','float_e5m2_t','float',32,'(2,2,1)','(32,32,32)'],
      ].entries()) {
        ctx.setMTM(TAB, ...args);
        let exported;
        ctx.exportURL = (btn, feature, ...inputs) => { exported = {feature, inputs}; };
        ctx.exportMTM(TAB);
        const query = `?key=${encodeURIComponent([exported.feature,...exported.inputs].join('-'))}`;
        ctx.pabcImportTiledMma(TAB, i === 1 ? query : `file:///project/index.html${query}`);
        const expected = exported.inputs;
        for (const [key, value] of [['op',expected[0]],['k',expected[3]],['atomlayout',expected[4]],['perm',expected[5] === 'na' ? '' : expected[5]]])
          if (els.get(`${TAB}-pabc-${key}-input`).value !== String(value)) throw new Error(`Imported ${key} differs from exported TiledMMA`);
        for (const [key, value] of [['ab',expected[1]],['acc',expected[2]]])
          if (value !== 'na' && els.get(`${TAB}-pabc-${key}-input`).value !== value) throw new Error(`${key} options were not rebuilt before import`);
        if (ctx.pabcState[TAB].which !== 'B') throw new Error('Decode changed the selected operand');
        if (JSON.stringify(['thr','tensor','fragment-slice'].map(key => els.get(`${TAB}-pabc-${key}-input`).value)) !== JSON.stringify(keep))
          throw new Error('Decode changed thread, tensor, or fragment selection');
        if (!els.get(`${TAB}-pabc-result`).innerHTML.includes('partition_B =')) throw new Error('Decode did not render the partition');
        let partitionExport;
        ctx.exportURL = (btn, feature, ...inputs) => { partitionExport = {feature, inputs}; };
        ctx.exportPABC(TAB);
        if (JSON.stringify(partitionExport.inputs.slice(1,7).map(String)) !== JSON.stringify(expected.map(String)))
          throw new Error('Decoded constructor did not survive partition URL export');
      }
      const before = ['op','ab','acc','k','atomlayout','perm'].map(key => els.get(`${TAB}-pabc-${key}-input`).value);
      for (const invalid of ['?key=layout-8:1', '?key=make_tiled_mma-f16bf16-half_t-half_t-99-(2,2,1)-na']) {
        let rejected = false;
        try { ctx.pabcImportTiledMma(TAB, invalid); } catch (_) { rejected = true; }
        if (!rejected || JSON.stringify(before) !== JSON.stringify(['op','ab','acc','k','atomlayout','perm'].map(key => els.get(`${TAB}-pabc-${key}-input`).value)))
          throw new Error('Failed decode altered constructor controls');
      }
    } finally { ctx.exportURL = realExport; }
  }, paneIds.filter(id => id.startsWith(`${TAB}-pabc-`)));

  // The handoff reads the CURRENT partition controls, not cached state from
  // the last Render. The DOM shim has no real tab-bar tree, so capture the
  // navigation call while running both real render paths.
  step('partition_A make_fragment button passes its current result and opens the tab', () => {
    ctx.setPABC(TAB, 'A', 'f16bf16', 'half_t', 'float', 16,
                '(2, 2, 1)', '', '5', '(64, 48):(48, 1)');
    els.get(`${TAB}-pabc-tensor-input`).value = '(64, 48):(1, 64)';
    const realSwitch = ctx.switchInnerTab;
    let switched = '';
    ctx.switchInnerTab = (_, mode) => { switched = mode; };
    try { ctx.pabcOpenFragment(TAB); } finally { ctx.switchInnerTab = realSwitch; }
    const want = partitionRef.partition_abc.A_colmajor.layout.str;
    if (switched !== 'make_fragment_abc') throw new Error(`opened ${switched}`);
    if (els.get(`${TAB}-mfrag-partition-input`).value !== want)
      throw new Error('make_fragment did not receive the newly partitioned layout');
    if (els.get(`${TAB}-mfrag-thr-input`).value !== '5')
      throw new Error('make_fragment did not receive the selected thread ID');
    if (!els.get(`${TAB}-mfrag-svg`).innerHTML.includes('>T5V0</text>'))
      throw new Error('make_fragment did not label the selected thread');
    if (!els.get(`${TAB}-mfrag-result`).innerHTML.includes(partitionRef.partition_abc.A_colmajor.fragment.str))
      throw new Error('make_fragment displayed the wrong returned layout');
  }, [`${TAB}-pabc-tile-svg`, `${TAB}-mfrag-svg`]);
  step('partition make_fragment selection field precedes its shortcut', () => {
    const field = html.indexOf(`id="${TAB}-pabc-fragment-slice-input"`);
    const button = html.indexOf(`id="${TAB}-pabc-make-fragment"`);
    if (field < 0 || field > button) throw new Error('selection field is not above make_fragment');
  });
  step('partition make_fragment defaults to one stage and accepts explicit slices', () => {
    ctx.setPABC(TAB, 'A', 'f16bf16', 'half_t', 'half_t', 16,
      '(2,2,1)', '(32,32,16)', '0', '(128,32,4):(32,1,4096)');
    const realSwitch = ctx.switchInnerTab;
    ctx.switchInnerTab = () => {};
    try {
      ctx.pabcOpenFragment(TAB);
      if (ctx.mfragState[TAB].registers.length !== 32 || ctx.mfragState[TAB].partition.rank() !== 3)
        throw new Error('default handoff included every pipeline stage');
      if (!els.get(`${TAB}-pabc-fragment-slice-preview`).textContent.includes('64 values/thread'))
        throw new Error('slice preview did not report the selected fragment size');
      els.get(`${TAB}-pabc-fragment-slice-input`).value = '(None, None, None, 2)';
      ctx.pabcOpenFragment(TAB);
      if (!els.get(`${TAB}-pabc-fragment-slice-preview`).textContent.includes('+8192 elements'))
        throw new Error('explicit stage lost its source offset');
      if (ctx.mfragState[TAB].registers.length !== 32) throw new Error('explicit stage fragment size wrong');
      els.get(`${TAB}-pabc-fragment-slice-input`).value = '(_,_,_,_)';
      ctx.pabcOpenFragment(TAB);
      if (ctx.mfragState[TAB].registers.length !== 128) throw new Error('explicit keep-all did not retain every stage');
    } finally { ctx.switchInnerTab = realSwitch; }
  }, [`${TAB}-pabc-tile-svg`, `${TAB}-mfrag-svg`]);
  step('partition slice selection survives URL export/import', () => {
    els.get(`${TAB}-pabc-fragment-slice-input`).value = '(_, _, _, 2)';
    const realExport = ctx.exportURL, realSwitch = ctx.switchInnerTab;
    let args;
    ctx.exportURL = (button, feature, ...inputs) => { args = [feature, ...inputs]; };
    ctx.switchInnerTab = () => {};
    try {
      ctx.exportPABC(TAB);
      els.get(`${TAB}-pabc-fragment-slice-input`).value = '_';
      ctx.location.search = `?key=${encodeURIComponent(args.join('-'))}`;
      ctx.applyKeyParam(TAB);
      if (els.get(`${TAB}-pabc-fragment-slice-input`).value !== '(_, _, _, 2)') throw new Error('URL lost selection');
      ctx.pabcOpenFragment(TAB);
      if (ctx.mfragState[TAB].registers.length !== 32) throw new Error('imported selection did not affect handoff');
      ctx.location.search = `?key=${encodeURIComponent(args.slice(0,-1).join('-'))}`;
      ctx.applyKeyParam(TAB);
      if (els.get(`${TAB}-pabc-fragment-slice-input`).value !== '') throw new Error('legacy URL retained stale selection');
    } finally { ctx.exportURL = realExport; ctx.switchInnerTab = realSwitch; ctx.location.search = ''; }
  }, [`${TAB}-pabc-tile-svg`, `${TAB}-mfrag-svg`]);
  step('partition refuses out-of-range fragment selections without navigating', () => {
    els.get(`${TAB}-pabc-fragment-slice-input`).value = '(_,_,_,4)';
    const realSwitch = ctx.switchInnerTab;
    let switched = false;
    ctx.switchInnerTab = () => { switched = true; };
    try { ctx.pabcOpenFragment(TAB); } finally { ctx.switchInnerTab = realSwitch; }
    if (switched) throw new Error('invalid fragment selection navigated away');
    els.get(`${TAB}-pabc-fragment-slice-input`).value = '';
  }, [], { expectError: /out of range/ });
  step('make_fragment rejects a non-partitioned operand layout', () => {
    els.get(`${TAB}-mfrag-partition-input`).value = '(16, 16):(1, 16)';
    ctx.renderMakeFragmentABC(TAB);
  }, [], { expectError: /already-partitioned layout/ });
  step('make_fragment recovers after a bad input', () => {
    ctx.setMfrag(TAB, 'A', 'f16bf16', 'half_t', 'float', 16,
                 partitionRef.partition_abc.A_2x2x1.layout.str);
    const regs = ctx.mfragState[TAB].registers;
    if (regs.length !== 24 || regs[0].slots.map(s => s.source).join(',') !== '0,1' ||
        regs[1].slots.map(s => s.source).join(',') !== '384,385')
      throw new Error('32-bit word packing lost the source positions');
  }, [`${TAB}-mfrag-svg`]);
  step('make_fragment URL imports operand, Op and partitioned layout', () => {
    const key = 'make_fragment_abc-B-fp8-float_e4m3_t-float-32-((4,2),2,3):((1,16),1536,32)';
    ctx.location.search = `?key=${encodeURIComponent(key)}`;
    const realSwitch = ctx.switchInnerTab;
    let switched = '';
    ctx.switchInnerTab = (_, mode) => { switched = mode; };
    try { ctx.applyKeyParam(TAB); } finally { ctx.switchInnerTab = realSwitch; }
    if (switched !== 'make_fragment_abc') throw new Error(`opened ${switched}`);
    if (ctx.mfragState[TAB].which !== 'B' || ctx.mfragState[TAB].dtype !== 'float_e4m3_t')
      throw new Error('fragment URL lost operand or dtype');
    if (ctx.mfragState[TAB].registers[0].slots.map(s => s.source).join(',') !== '0,1,2,3')
      throw new Error('FP8 register did not pack four scalar values');
    if (!els.get(`${TAB}-mfrag-result`).innerHTML.includes(
      JSON.parse(fs.readFileSync(path.join(ROOT, 'tests/reference.json'), 'utf8')).make_fragment_abc.B_fp8.str))
      throw new Error('fragment URL displayed the wrong layout');
    ctx.location.search = '';
  }, [`${TAB}-mfrag-svg`]);
  step('make_fragment B link shows one cell per scalar value', () => {
    const key = 'make_fragment_abc-B-f16bf16-bfloat16_t-float-16-((2,2),2,3):((1,8),768,16)';
    ctx.location.search = `?key=${encodeURIComponent(key)}`;
    const realSwitch = ctx.switchInnerTab;
    ctx.switchInnerTab = () => {};
    try { ctx.applyKeyParam(TAB); } finally { ctx.switchInnerTab = realSwitch; }
    const grid = ctx.mfragValueGrid(ctx.mfragState[TAB]);
    if (grid.rows !== 8 || grid.cols !== 3 || grid.cells.length !== 24)
      throw new Error(`expected an 8×3 scalar grid, got ${grid.rows}×${grid.cols}`);
    const at = (m, n) => grid.cells[m + n * grid.rows];
    if (at(0, 0).register !== 0 || at(0, 0).source !== 0 ||
        at(1, 0).register !== 0 || at(1, 0).source !== 1 ||
        at(4, 0).register !== 6 || at(4, 0).source !== 768 ||
        at(0, 1).register !== 2 || at(0, 1).source !== 16 ||
        at(4, 0).rest !== 1 || at(0, 1).rest !== 2)
      throw new Error('B scalar cells do not match the CuTeDSL fragment mapping');
    const svg = els.get(`${TAB}-mfrag-svg`).innerHTML;
    if (!svg.includes('>(0,0)</text>') || !svg.includes('>T0V0</text>') ||
        !svg.includes('>T0V1</text>') || !svg.includes('>R0</text>') ||
        !svg.includes('>R6</text>') || !svg.includes('>(1,0)</text>'))
      throw new Error('B tile coordinates, T/V labels, or registers did not render');
    for (let r = 0; r < 6; r++) {
      const color = ctx.colorHighlight(r);
      const count = [...svg.matchAll(new RegExp(`<rect[^>]*fill="${color}"`, 'g'))].length;
      if (count !== 4) throw new Error(`Rest block ${r} has ${count} cells in its partition color`);
    }
    ctx.location.search = '';
  }, [`${TAB}-mfrag-svg`]);
  step('make_fragment URL preserves an explicit thread label', () => {
    const key = 'make_fragment_abc-B-f16bf16-bfloat16_t-float-16-((2,2),2,3):((1,8),768,16)-7';
    ctx.location.search = `?key=${encodeURIComponent(key)}`;
    const realSwitch = ctx.switchInnerTab;
    ctx.switchInnerTab = () => {};
    try { ctx.applyKeyParam(TAB); } finally { ctx.switchInnerTab = realSwitch; }
    if (els.get(`${TAB}-mfrag-thr-input`).value !== '7' ||
        !els.get(`${TAB}-mfrag-svg`).innerHTML.includes('>T7V0</text>'))
      throw new Error('thread ID was lost in the fragment URL');
    ctx.location.search = '';
  }, [`${TAB}-mfrag-svg`]);

  // The operand views are identical, while ldmatrix's actual address and
  // destination ownership differ. Toggle and focus must update all four panes.
  const psdPanes= paneIds.filter(id=>id.startsWith(`${TAB}-psd-`));
  step('partition_S/D has pasteboard and folded manual construction without a URL field',()=>{
    if(html.includes(`id="${TAB}-psd-copy-url-input"`) || html.includes('>Decode URL</button>')) throw new Error('URL entry controls remain');
    const manual = html.match(new RegExp(`<details class="cuo-section" id="${TAB}-psd-manual-manual-construction"([^>]*)>`));
    if (!manual || /\bopen\b/.test(manual[1])) throw new Error('Manual TiledCopy section is missing or initially open');
    if(!html.includes(`onclick="psdDecodePasteboard('${TAB}')"`))throw new Error('clipboard decoder missing');
  });
  step('partition_S/D decodes a derived SIMT TiledCopy',()=>{
    ctx.setPSDURL(TAB,'S','?key=make_tiled_copy_tv-universal-32-half_t-(8,4):(1,8)-(2,2):(1,2)','3','(32,32):(32,1)');
    if(ctx.psdState[TAB].thrSize!==32 || ctx.psdState[TAB].copyAtom.threads!==1)
      throw new Error('derived TiledCopy was not decoded');
  },psdPanes);
  step('partition_S/D uses the decoded ldmatrix SRC and DST layouts',()=>{
    const url='file:///index.html?key=make_tiled_copy_ab-A-f16bf16-half_t-half_t-16-(2,2,1)-(32,32,16)-ldmatrix-1-0-16-warp';
    ctx.setPSDURL(TAB,'S',url,'20','(128,32,4):(32,1,4096)');
    const src=ctx.formatLayoutStr(ctx.psdState[TAB].partition.shape,ctx.psdState[TAB].partition.stride);
    if(ctx.psdState[TAB].copyAtom.threads!==32 || ctx.psdState[TAB].frgV!==8)throw new Error('source atom traits were ignored');
    for(const level of ['tile','sweep','extra'])expectBadge(`psd-${level}`,'SMEM');
    ctx.setPsdSide(TAB,'D');
    const dst=ctx.formatLayoutStr(ctx.psdState[TAB].partition.shape,ctx.psdState[TAB].partition.stride);
    if(src===dst || ctx.psdState[TAB].frgV!==2)throw new Error('ldmatrix source and destination incorrectly agree');
    for(const level of ['tile','sweep','extra'])expectBadge(`psd-${level}`,'RMEM');
  },psdPanes);
  step('partition_S/D URL round trip preserves an embedded TiledCopy URL',()=>{
    const before=psdPanes.map(id=>els.get(id).innerHTML);
    const input=ctx.psdState[TAB].copyURL;
    const realExport=ctx.exportURL,realSwitch=ctx.switchInnerTab;
    let args;
    ctx.exportURL=(button,feature,...inputs)=>{args=[feature,...inputs];};ctx.switchInnerTab=()=>{};
    try{
      ctx.exportPSD(TAB);
      ctx.setPSD(TAB,'S','universal',32,'half_t','((8,4),(2,2)):((16,2),(8,1))','(8,16)','5','(16,32):(1,16)');
      ctx.location.search=`?key=${encodeURIComponent(args.join('-'))}`;ctx.applyKeyParam(TAB);
      if(ctx.psdState[TAB].copyURL!==input || psdPanes.some((id,i)=>els.get(id).innerHTML!==before[i]))
        throw new Error('partition URL lost embedded copy or result');
    }finally{ctx.exportURL=realExport;ctx.switchInnerTab=realSwitch;ctx.location.search='';}
  },psdPanes);
  step('partition_S/D displays CuTeDSL nondividing and unvectorizable results with warnings',()=>{
    for(const [id,bits,tensor]of [['cute_nondividing',32,'(12,32):(1,12)'],['cute_unvectorizable',64,'(16,32):(1,16)']]){
      ctx.setPSD(TAB,'S','universal',bits,'half_t','((8,4),(2,2)):((16,2),(8,1))','(8,16)','0',tensor);
      const ref=partitionRef.partition_sd[id].layout.str;
      if(!els.get(`${TAB}-psd-result`).innerHTML.includes(ref) || !els.get(`${TAB}-psd-warning`).textContent)
        throw new Error('partition behavior differs from CuTeDSL: '+id);
    }
  },psdPanes);
  step('partition_S/D failed paste leaves the last decoded copy intact',()=>{
    const before=ctx.psdState[TAB].copyURL;
    const summary=els.get(`${TAB}-psd-tiled-result`).innerHTML;
    try{ctx.psdImportTiledCopy(TAB,'?key=make_copy_atom-universal-32-half_t');}
    catch(e){ctx.showErr(`${TAB}-psd-error`,e.message);}
    if(ctx.psdState[TAB].copyURL!==before || els.get(`${TAB}-psd-tiled-result`).innerHTML!==summary)
      throw new Error('failed paste changed the active TiledCopy');
  },psdPanes,{expectError:/make_tiled_copy/});
  step('partition_S/D pasted copy shows the same summary as make_tiled_copy_A',()=>{
    const markup=ctx.generatePartitionSDTabContent(TAB);
    if(markup.includes('>Decode URL</button>') || markup.includes('psd-copy-url-input'))throw new Error('manual URL controls remain');
    ctx.setMCAB(TAB,'A','f16bf16','half_t','half_t',16,'(2,2,1)','(32,32,16)','ldmatrix',4,0,16);
    const expected=els.get(`${TAB}-mcab-result`).innerHTML;
    ctx.psdImportTiledCopy(TAB,'?key=make_tiled_copy_ab-A-f16bf16-half_t-half_t-16-(2,2,1)-(32,32,16)-ldmatrix-4-0-16-warp');
    if(els.get(`${TAB}-psd-tiled-result`).innerHTML!==expected)throw new Error('decoded copy summary differs from its constructor tab');
    for(const label of ['make_tiled_copy_A(copy_atom, tiled_mma)','Tiler MN = (32:1,16:1)','TV Layout tiled =','TV Layout Src =','TV Layout Dst =','4 warps, 128 threads; 8 destination values/thread; 1 copy instruction per warp.'])
      if(!expected.includes(label))throw new Error('missing decoded summary field '+label);
  },psdPanes);
  step('partition_S/D manual fields are populated for every supported constructor family', () => {
    const corpus = JSON.parse(fs.readFileSync(path.join(__dirname, 'cases.json'), 'utf8')).partition_sd;
    for (const id of ['url_derived_simt','url_strided_tiler','url_A_x4_S','url_A_fp8_D']) {
      const fixture = corpus.find(c => c.id === id);
      ctx.setPSDURL(TAB, fixture.side, fixture.copy_url, '0', fixture.tensor);
      const parsed = ctx.parsePastedKeyParam(fixture.copy_url);
      const manual = ctx.mtcReadManualConstructor(TAB, 'psd-manual');
      if (manual.feature !== parsed.feature || JSON.stringify(manual.inputs) !== JSON.stringify(parsed.inputs))
        throw new Error(`Manual constructor fields differ from decoded ${id}`);
      const expected = partitionRef.partition_sd[id].layout.str;
      if (!els.get(`${TAB}-psd-result`).innerHTML.includes(expected)) throw new Error(`Manual ${id} differs from CuTeDSL`);
    }
  }, psdPanes);
  step('partition_S/D manual edits preserve atom-specific SRC/DST and match CuTeDSL', () => {
    const url = '?key=make_tiled_copy_ab-A-f16bf16-half_t-half_t-16-(2,2,1)-(32,32,16)-ldmatrix-4-0-16-warp';
    ctx.setPSDURL(TAB, 'S', url, '0', '(128,32,4):(32,1,4096)');
    els.get(`${TAB}-psd-manual-atom-nm-input`).value = '1';
    ctx.renderPartitionSD(TAB);
    if (!els.get(`${TAB}-psd-result`).innerHTML.includes(partitionRef.partition_sd.url_A_x1_S.layout.str))
      throw new Error('Manual ldmatrix.x1 SRC differs from CuTeDSL');
    ctx.setPsdSide(TAB, 'D');
    if (!els.get(`${TAB}-psd-result`).innerHTML.includes(partitionRef.partition_sd.url_A_x1_D.layout.str))
      throw new Error('Manual ldmatrix.x1 DST differs from CuTeDSL');
    const parsed = ctx.parsePastedKeyParam(ctx.mtcManualConstructorURL(TAB, 'psd-manual'));
    if (parsed.inputs[8] !== '1') throw new Error('Manual copy edit did not reach the exported constructor');
    ctx.setPSD(TAB, 'S', 'universal', 32, 'half_t', '((8,4),(2,2)):((16,2),(8,1))', '(8,16)', '0', '(16,32):(1,16)');
    els.get(`${TAB}-psd-manual-atom-bits-input`).value = '64';
    ctx.renderPartitionSD(TAB);
    if (!els.get(`${TAB}-psd-result`).innerHTML.includes(partitionRef.partition_sd.cute_unvectorizable.layout.str))
      throw new Error('Manual SIMT edit differs from CuTeDSL');
  }, psdPanes);
  step('partition_S/D manual edits survive URL export and invalid decode preserves fields', () => {
    const read = () => JSON.stringify(ctx.mtcReadManualConstructor(TAB, 'psd-manual'));
    const before = read();
    let rejected = false;
    try { ctx.psdImportTiledCopy(TAB, '?key=make_tiled_copy_ab-A-f16bf16-bfloat16_t-half_t-16-(2,2,1)-(32,32,16)-ldmatrix-1-0-16-warp'); }
    catch (_) { rejected = true; }
    if (!rejected || read() !== before) throw new Error('Rejected decode changed manual fields');
    const realExport = ctx.exportURL, realSwitch = ctx.switchInnerTab;
    try {
      let exported;
      ctx.exportURL = (btn, feature, ...inputs) => { exported = [feature,...inputs]; };
      ctx.switchInnerTab = () => {};
      ctx.exportPSD(TAB);
      ctx.setPSDURL(TAB, 'S', ctx.PSD_DEFAULT_COPY_URL, '0', '(16,32):(1,16)');
      ctx.location.search = `?key=${encodeURIComponent(exported.join('-'))}`;
      ctx.applyKeyParam(TAB);
      if (read() !== before) throw new Error('URL export lost manual copy edits');
    } finally { ctx.exportURL = realExport; ctx.switchInnerTab = realSwitch; ctx.location.search = ''; }
  }, psdPanes);
  step('partition_S/D recovers after invalid decode',()=>{
    ctx.setPSD(TAB,'S','universal',32,'half_t','((8,4),(2,2)):((16,2),(8,1))','(8,16)','5','(16,32):(1,16)');
  },psdPanes);

  const rtPanes = [`${TAB}-rt-src-svg`, `${TAB}-rt-dst-svg`];
  step('retile is in COPY and both decoded inputs have folded manual controls', () => {
    if (!html.includes('data-tab="retile" class="tab" data-scope="copy"')) throw new Error('retile missing from COPY');
    for (const kind of ['copy','input']) if (!html.includes(`onclick="rtDecodePasteboard('${TAB}','${kind}')"`))
      throw new Error(`missing ${kind} paste handler`);
    for (const kind of ['copy','input']) {
      const section = html.match(new RegExp(`<details class="cuo-section" id="${TAB}-rt-${kind}-manual-construction"([^>]*)>`));
      if (!section || /\bopen\b/.test(section[1])) throw new Error(`retile ${kind} manual controls are not folded`);
    }
  });
  step('retile imports TiledCopy A and computes a make_fragment URL', () => {
    ctx.rtImportURL(TAB,'copy','file:///index.html?key=make_tiled_copy_ab-A-f16bf16-half_t-half_t-16-(2,2,1)-(32,32,16)-ldmatrix-4-0-16-warp');
    const key='make_fragment_abc-A-f16bf16-half_t-half_t-16-((2,2,2),4,2):((1,256,8),1024,16)-7';
    ctx.rtImportURL(TAB,'input',`?key=${encodeURIComponent(key)}`);
    const r=ctx.rtState[TAB];
    const fmt=L=>ctx.formatLayoutStr(L.shape,L.stride);
    if (fmt(r.input)!=='((2,2,2),4,2):((1,2,4),16,8)' || fmt(r.layout)!=='((8,1),4,2):((1,0),16,8)')
      throw new Error('fragment URL was not evaluated before retiling');
    expectBadge('rt-src','RMEM'); expectBadge('rt-dst','RMEM');
    if (!els.get(`${TAB}-rt-dst-svg`).innerHTML.includes('>T7V0</text>')) throw new Error('thread label missing');
    const expected=ctx.mfragBuildValueSVG({partition:r.layout,fragment:r.layout,pack:r.pack},7);
    if (els.get(`${TAB}-rt-dst-svg`).innerHTML!==expected) throw new Error('retile does not use shared fragment drawing');
  },rtPanes);
  step('retile manual fields remain editable after decode', () => {
    ctx.setRT(TAB, ctx.RT_DEFAULT_TV, '(32,16)', 8, 'half_t', '((2,2,2),4,2):((1,2,4),16,8)', 0);
    const before = els.get(`${TAB}-rt-result`).innerHTML;
    els.get(`${TAB}-rt-values-input`).value = '2';
    ctx.renderRetile(TAB);
    if (els.get(`${TAB}-rt-result`).innerHTML === before || ctx.rtState[TAB].layout.size() !== ctx.rtState[TAB].input.size())
      throw new Error('Retile did not consume manual copy grouping');
    ctx.rtImportURL(TAB, 'copy', '?key=make_tiled_copy_ab-A-f16bf16-half_t-half_t-16-(2,2,1)-(32,32,16)-ldmatrix-4-0-16-warp');
    if (els.get(`${TAB}-rt-values-input`).value !== '8') throw new Error('Decode did not refill retile manual fields');
  }, rtPanes);
  step('retile B gathers across a Rest mode', () => {
    ctx.setRT(TAB,'((4,8,2,2),((2,2),(2,1))):((64,1,0,8),((32,256),(16,0)))','(32,16)',8,'half_t','((2,2),8,2):((1,2),8,4)',0);
    const r=ctx.rtState[TAB];
    if (ctx.product(r.input.shape[0])!==4 || ctx.product(r.layout.shape[0])!==8 || r.frgMN.size()<=1)
      throw new Error('retile failed to gather outside mode 0');
  },rtPanes);
  step('retile imports primitive and derived SIMT TiledCopy URLs', () => {
    ctx.setRT(TAB,'(4,4):(1,4)','(4,4)',2,'float','(2,2,2):(1,2,4)',0);
    ctx.rtImportURL(TAB,'copy','?key=make_tiled_copy-universal-64-float-(4,4):(1,4)-(4,4)');
    const expected=els.get(`${TAB}-rt-dst-svg`).innerHTML;
    ctx.rtImportURL(TAB,'copy','?key=make_tiled_copy_tv-universal-64-float-(4,1):(1,0)-(1,4):(0,1)');
    if (els.get(`${TAB}-rt-dst-svg`).innerHTML!==expected) throw new Error('derived copy import changed retile');
  },rtPanes);
  step('retile imports fixed-dtype fragment URLs', () => {
    ctx.setRT(TAB,'(4,4):(1,4)','(4,4)',1,'float','((2,2),2,2):((1,2),4,8)',0);
    ctx.rtImportURL(TAB,'input','?key=make_fragment_abc-A-tf32-na-na-8-((2,2),2,4):((256,4),1024,8)');
    if(ctx.rtState[TAB].dtype!=='tfloat32_t')throw new Error('TF32 na dtype was not resolved');
    ctx.rtImportURL(TAB,'input','?key=make_fragment_abc-C-fp8-float_e4m3_t-na-32-((2,2),2,2):((1,256),1024,16)');
    if(ctx.rtState[TAB].dtype!=='float')throw new Error('fixed accumulator dtype was not resolved');
  },rtPanes);
  step('retile URL round trip restores both views', () => {
    ctx.setRT(TAB,ctx.RT_DEFAULT_TV,'(32,16)',2,'half_t','((2,2,2),4,2,3):((1,2,4),16,8,64)',3);
    const before=rtPanes.map(id=>els.get(id).innerHTML);
    const realExport=ctx.exportURL, realSwitch=ctx.switchInnerTab;
    let args,switched;
    ctx.exportURL=(button,feature,...inputs)=>{args=[feature,...inputs];};
    ctx.switchInnerTab=(_,mode)=>{switched=mode;};
    try {
      ctx.exportRT(TAB);
      ctx.setRT(TAB,'(4,4):(1,4)','(4,4)',2,'float','(2,2,2):(1,2,4)',0);
      ctx.location.search=`?key=${encodeURIComponent(args.join('-'))}`;
      ctx.applyKeyParam(TAB);
      if(switched!=='retile'||rtPanes.some((id,i)=>els.get(id).innerHTML!==before[i])) throw new Error('retile URL changed result');
    } finally {ctx.exportURL=realExport;ctx.switchInnerTab=realSwitch;ctx.location.search='';}
  },rtPanes);
  step('retile invalid atom count clears both views', () => {
    els.get(`${TAB}-rt-values-input`).value='3';ctx.renderRetile(TAB);
    if(rtPanes.some(id=>els.get(id).innerHTML))throw new Error('stale retile survived invalid input');
  },[],{expectError:/dividing/});
  step('retile rejects an unrelated URL before changing inputs', () => {
    const before=els.get(`${TAB}-rt-tv-input`).value;
    let error;
    try {ctx.rtImportURL(TAB,'copy','?key=make_copy_atom-universal-16-half_t');}catch(e){error=e.message;}
    if(!/TiledCopy|make_tiled_copy/.test(error||'')||els.get(`${TAB}-rt-tv-input`).value!==before)throw new Error('invalid import was accepted');
    ctx.setRT(TAB,ctx.RT_DEFAULT_TV,'(32,16)',8,'half_t','((2,2,2),4,2):((1,2,4),16,8)',0);
  },rtPanes);

  const mcabPanes = paneIds.filter(id => id.startsWith(`${TAB}-mcab-`));
  for (const operand of ['A', 'B']) {
    step(`make_tiled_copy_${operand}: TVs and SMEM/RMEM badges`, () => {
      ctx.setMCAB(TAB, operand, 'f16bf16', 'half_t', 'float', 16,
        '(2,2,1)', '(32,32,16)', 'ldmatrix', 4, 0, 16, 'tv');
      expectBadge('mcab-src', 'SMEM');
      expectBadge('mcab-dst', 'RMEM');
      const dstSvg = els.get(`${TAB}-mcab-dst-svg`).innerHTML;
      if (!dstSvg.includes('T0/V0')) throw new Error('TV labels did not render');
      if (els.get(`${TAB}-mcab-src-svg`).innerHTML === dstSvg)
        throw new Error('ldmatrix SRC and DST incorrectly agree');
      ctx.setMcabCompare(TAB, 'mma_copy');
      expectBadge('mcab-src', 'RMEM');
      if (els.get(`${TAB}-mcab-dst-svg`).innerHTML !== dstSvg)
        throw new Error('comparison changed the right pane');
      const r = ctx.mcabState[TAB].result;
      const expectedMma = ctx.mtmBuildSVG(ctx.mtmOperandGrid(r.mmaOperand), ...r.tile, { mode: 'tv', focus: null });
      if (els.get(`${TAB}-mcab-src-svg`).innerHTML !== expectedMma)
        throw new Error('left pane does not show the MMA operand');
      if (els.get(`${TAB}-mcab-src-label`).textContent !== 'TiledMMA' ||
          els.get(`${TAB}-mcab-dst-label`).textContent !== 'TiledCopy (DST)')
        throw new Error('comparison labels incorrect');
    }, mcabPanes);
    for (const [mode, focus] of [['warp', '2'], ['tv', '1']]) {
      step(`make_tiled_copy_${operand}: ${mode} focus`, () => {
        els.get(`${TAB}-mcab-focus-input`).value = focus;
        ctx.setMcabMode(TAB, mode);
        const label = els.get(`${TAB}-mcab-focus-label`).textContent;
        if (label !== (mode === 'tv' ? 'Thread ID' : 'Warp id')) throw new Error(`focus label = ${label}`);
        for (const key of ['src', 'dst']) {
          const svg = els.get(`${TAB}-mcab-${key}-svg`).innerHTML;
          if (!svg.includes(mode === 'tv' ? '>T1</text>' : '>W2</text>'))
            throw new Error(`${key} lost the focused ${mode}`);
        }
      }, mcabPanes);
    }
  }
  step('make_tiled_copy_AB: constructor paste buttons are wired', () => {
    for (const [section, feature] of [['mma', 'make_tiled_mma'], ['copy', 'make_copy_atom']]) {
      if (!html.includes(`id="${TAB}-mcab-decode-${section}" onclick="mcabDecodePasteboard('${TAB}','${feature}')">Decode from Pasteboard`))
        throw new Error(`missing ${section} paste button handler`);
    }
  });
  step('make_tiled_copy_AB: decode TiledMMA URL rebuilds dtype/K options', () => {
    ctx.setMCAB(TAB, 'B', 'f16bf16', 'half_t', 'float', 16,
      '(2,2,1)', '(32,32,16)', 's2r', 1, 0, 32, 'tv', '3', 'mma_copy');
    const key = 'make_tiled_mma-tf32-tfloat32_t-float-8-(2,2,1)-na';
    ctx.mcabImportConstructor(TAB, 'make_tiled_mma', `file:///other/index.html?key=${encodeURIComponent(key)}#anchor`);
    for (const [field, expected] of [['op', 'tf32'], ['ab', 'tfloat32_t'], ['k', '8'], ['perm', ''], ['operand', 'B'], ['copy', 's2r']]) {
      if (els.get(`${TAB}-mcab-${field}-input`).value !== expected) throw new Error(`import lost ${field}`);
    }
    if (ctx.mcabState[TAB].mode !== 'tv' || ctx.mcabState[TAB].compare !== 'mma_copy' || ctx.mcabState[TAB].focus !== 3)
      throw new Error('constructor paste changed comparison/focus');
  }, mcabPanes);
  step('make_tiled_copy_AB: decode CopyUniversalOp query', () => {
    ctx.mcabImportConstructor(TAB, 'make_copy_atom', '?key=make_copy_atom-universal-64-tfloat32_t');
    if (els.get(`${TAB}-mcab-copy-input`).value !== 's2r' || els.get(`${TAB}-mcab-bits-input`).value !== '64')
      throw new Error('universal copy parameters not imported');
  }, mcabPanes);
  step('make_tiled_copy_AB: decode ldmatrix query rebuilds matrix/transpose options', () => {
    ctx.mcabImportConstructor(TAB, 'make_tiled_mma', 'key=make_tiled_mma-fp8-float_e4m3_t-float-32-(2,2,1)-(32,32,32)');
    ctx.mcabImportConstructor(TAB, 'make_copy_atom', '?key=make_copy_atom-ldmatrix16x16x8b-128-float_e4m3_t-2-1');
    if (els.get(`${TAB}-mcab-nm-input`).value !== '2' || els.get(`${TAB}-mcab-trans-input`).value !== '1' ||
        !els.get(`${TAB}-mcab-trans-input`).disabled) throw new Error('ldmatrix parameters not imported');
  }, mcabPanes);
  step('make_tiled_copy_AB: bad constructor pastes leave inputs intact', () => {
    const saved = els.get(`${TAB}-mcab-copy-input`).value;
    for (const [feature, text, reason] of [
      ['make_tiled_mma', '?key=make_copy_atom-universal-32-half_t', /Paste a make_tiled_mma/],
      ['make_copy_atom', '?key=make_copy_atom-cpasync-128-float_e4m3_t', /not supported/],
      ['make_copy_atom', '?key=make_copy_atom-ldmatrix-128-half_t-4-0', /does not match/],
      ['make_tiled_mma', 'not a URL', /Paste a/],
    ]) {
      let error;
      try { ctx.mcabImportConstructor(TAB, feature, text); } catch (e) { error = e.message; }
      if (!reason.test(error || '')) throw new Error(`unexpected paste result: ${error}`);
      if (els.get(`${TAB}-mcab-copy-input`).value !== saved) throw new Error('bad paste mutated copy input');
    }
  }, mcabPanes);
  step('make_tiled_copy_AB: CopyAtom manual construction is folded and editable', () => {
    const section = html.match(new RegExp(`<details class="cuo-section" id="${TAB}-mcab-copy-manual-construction"([^>]*)>([\\s\\S]*?)</details>`));
    if (!section || /\bopen\b/.test(section[1])) throw new Error('CopyAtom manual controls are not folded by default');
    for (const key of ['copy','nm','trans','bits'])
      if (!section[2].includes(`id="${TAB}-mcab-${key}-input"`)) throw new Error(`${key} is outside the manual section`);
    ctx.setMCAB(TAB, 'A', 'f16bf16', 'half_t', 'float', 16, '(2,2,1)', '(32,32,16)', 'ldmatrix', 4, 0, 16);
    const before = els.get(`${TAB}-mcab-result`).innerHTML;
    els.get(`${TAB}-mcab-nm-input`).value = '1';
    ctx.renderMakeTiledCopyAB(TAB);
    if (els.get(`${TAB}-mcab-result`).innerHTML === before) throw new Error('Manual CopyAtom edit did not affect the result');
    els.get(`${TAB}-mcab-atomlayout-input`).value = '(1,1,1)';
    ctx.renderMakeTiledCopyAB(TAB);
    if (ctx.mcabState[TAB].mma.warps !== 1) throw new Error('Manual TiledMMA edit did not affect the result');
    ctx.mcabImportConstructor(TAB, 'make_copy_atom', '?key=make_copy_atom-ldmatrix-128-half_t-4-0');
    if (els.get(`${TAB}-mcab-nm-input`).value !== '4') throw new Error('Decode did not refill manual CopyAtom controls');
  }, mcabPanes);
  step('make_tiled_copy_AB: invalid copy width clears all panes', () => {
    ctx.setMCAB(TAB, 'B', 'f16bf16', 'half_t', 'float', 16,
      '(1,1,1)', '', 'ldmatrix', 4, 0, 16);
    if (els.get(`${TAB}-mcab-result`).innerHTML ||
        mcabPanes.some(id => els.get(id).innerHTML)) throw new Error('stale result survived invalid input');
  }, [], { expectError: /too few vals/ });
  step('make_tiled_copy_AB: invalid focus is reported', () => {
    ctx.setMCAB(TAB, 'A', 'f16bf16', 'half_t', 'float', 16,
      '(2,2,1)', '(32,32,16)', 'ldmatrix', 4, 0, 16, 'warp', '99');
  }, mcabPanes, { expectError: /out of range/ });
  step('make_tiled_copy_AB: URL export/import restores B, copy options, view and focus', () => {
    ctx.setMCAB(TAB, 'B', 'f16bf16', 'half_t', 'float', 16,
      '(2,2,1):(2,1,4)', '(32,32,16)', 'ldmatrix', 2, 1, 16, 'tv', '3', 'mma_copy');
    const expected = mcabPanes.map(id => els.get(id).innerHTML);
    const realExport = ctx.exportURL, realSwitch = ctx.switchInnerTab;
    let args;
    ctx.exportURL = (btn, feature, ...inputs) => { args = [feature, ...inputs]; };
    ctx.switchInnerTab = () => {};
    try {
      ctx.exportMCAB(TAB);
      ctx.setMCAB(TAB, 'A', 'tf32', null, null, 8, '(1,1,1)', '', 's2r', 1, 0, 32);
      ctx.location.search = `?key=${encodeURIComponent(args.join('-'))}`;
      ctx.applyKeyParam(TAB);
      if (mcabPanes.some((id, i) => els.get(id).innerHTML !== expected[i]))
        throw new Error('URL changed the ownership picture');
      if (els.get(`${TAB}-mcab-operand-input`).value !== 'B' ||
          ctx.mcabState[TAB].mode !== 'tv' || ctx.mcabState[TAB].focus !== 3 ||
          ctx.mcabState[TAB].compare !== 'mma_copy')
        throw new Error('URL lost operand, mode, focus or comparison');
    } finally {
      ctx.exportURL = realExport;
      ctx.switchInnerTab = realSwitch;
      ctx.location.search = '';
    }
  }, mcabPanes);
  step('make_tiled_copy_AB: SIMT and required-transpose imports rebuild options', () => {
    const realSwitch = ctx.switchInnerTab;
    ctx.switchInnerTab = () => {};
    try {
      ctx.location.search = '?key=make_tiled_copy_ab-A-fp8-float_e5m2_t-float-32-(2,2,1)-(32,32,32)-ldmatrix16x16x8b-1-1-8-warp';
      ctx.applyKeyParam(TAB);
      if (els.get(`${TAB}-mcab-nm-input`).value !== '1' || !els.get(`${TAB}-mcab-trans-input`).disabled)
        throw new Error('copy-dependent controls were not rebuilt');
      if (ctx.mcabState[TAB].compare !== 'src_dst') throw new Error('legacy URL lost default comparison');
    } finally { ctx.switchInnerTab = realSwitch; ctx.location.search = ''; }
  }, mcabPanes);

  // ── the two partition tabs' thread box accepts BLANK = every thread ──────
  // The presets all pass a concrete id, so nothing else here reaches the
  // unfocused path — and it is a whole branch of both grid-1 renderers.
  for (const [pfx, render] of [['psd', 'renderPartitionSD'], ['pabc', 'renderPartitionABC']]) {
    const field = els.get(`${TAB}-${pfx}-thr-input`);
    if (!field || typeof ctx[render] !== 'function') continue;
    const svgs = paneIds.filter(id => id.startsWith(`${TAB}-${pfx}-`));
    for (const [v, label, opts] of [
      ['', 'blank (all threads)', undefined],
      ['3', '#3', undefined],
      ['99999', 'out of range', { expectError: /out of range/ }],
      ['x', 'non-numeric', { expectError: /whole number/ }],
      ['', 'blank again', undefined],
    ]) {
      // Both tabs blank their grids on an error, unlike make_tiled_mma (whose
      // focus box re-reads on every keystroke and must survive typing). So the
      // pane check applies only to the steps expected to succeed.
      step(`${render} thread ${label}`, () => {
        field.value = v;
        ctx[render](TAB);
        if (opts && els.get(`${TAB}-${pfx}-result`).innerHTML)
          throw new Error('stale returned layout remains after an invalid input');
      },
           opts ? [] : svgs, opts);
    }
  }

  // ── the highlight-thread controls ────────────────────────────────────────
  // Live `oninput` handlers that re-render, so nothing else here reaches them:
  // every control is seeded from its markup default and the field ships empty.
  // Both an in-range id and an out-of-range one are driven, since the latter
  // takes the warn-and-show-everything path rather than the filter path.
  for (const [pfx, setter, fieldId] of [['tv', 'setHighlightTid', 'highlight-tid'],
                                       ['mtc', 'setMtcHighlight', 'highlight-tid'],
                                       ['mtv', 'setMtvHighlight', 'highlight-tid'],
                                       ['mma', 'setMmaHighlight', 'highlight-tid'],
                                       ['mtm', 'setMtmHighlight', 'focus-input']]) {
    const field = els.get(`${TAB}-${pfx}-${fieldId}`);
    if (!field || typeof ctx[setter] !== 'function') continue;
    const svgs = paneIds.filter(id => id.startsWith(`${TAB}-${pfx}-`));
    for (const tid of ['0', '3', '9999', '']) {
      // make_tiled_mma REPORTS an out-of-range id rather than ignoring it, and
      // still draws unfocused — the field re-reads on every keystroke, so
      // blanking six grids mid-type would be worse than a red line of text.
      const bad = pfx === 'mtm' && tid === '9999';
      step(`${setter}(${tid === '' ? 'cleared' : `#${tid}`})`, () => {
        field.value = tid;
        ctx[setter](TAB);
      }, svgs, bad ? { expectError: /out of range/ } : undefined);
    }
  }

  // ── make_tiled_mma's focus box is a plain text field, so it can hold junk ──
  {
    const field = els.get(`${TAB}-mtm-focus-input`);
    const svgs = paneIds.filter(id => id.startsWith(`${TAB}-mtm-`));
    step('setMtmHighlight(non-numeric) reports it', () => {
      field.value = 'w3';
      ctx.setMtmHighlight(TAB);
    }, svgs, { expectError: /not a warp id/ });
    // The box's UNIT follows the mode, so the same value can be legal in one
    // mode and out of range in the other — 100 is a thread but not a warp.
    // The box's UNIT follows the mode, and so must its (i) tooltip. The label
    // text lives in its own span precisely so `textContent` cannot eat the
    // icon, and the tooltip is an ATTRIBUTE now — two things a render could
    // silently stop doing.
    const tip = () => (els.get(`${TAB}-mtm-focus-hint`) || {}).getAttribute('data-tooltip') || '';
    step('setMtmMode(tv) relabels the focus box and its tooltip', () => {
      field.value = '100';
      ctx.setMtmMode(TAB, 'tv');
      const label = els.get(`${TAB}-mtm-focus-label`);
      if (label.textContent !== 'Thread ID') throw new Error(`label = ${label.textContent}`);
      if (!/thread/i.test(tip())) throw new Error(`tooltip = ${tip().slice(0, 60)}`);
    }, svgs);
    step('the same id is out of range as a warp', () => {
      ctx.setMtmMode(TAB, 'warp');
      const label = els.get(`${TAB}-mtm-focus-label`);
      if (label.textContent !== 'Warp id') throw new Error(`label = ${label.textContent}`);
      if (!/warp/i.test(tip())) throw new Error(`tooltip = ${tip().slice(0, 60)}`);
    }, svgs, { expectError: /Warp id 100 is out of range/ });
    step('clearing it recovers', () => {
      field.value = '';
      ctx.setMtmHighlight(TAB);
    }, svgs);
  }

  // ── the MMA tabs' Alternative View toggle ────────────────────────────────
  // A plain toggle with no arguments, so the preset driver never reaches it.
  // Both states are driven, and back, since the layout it switches to is the
  // one that re-orients B and it must survive being switched off again.
  for (const [pfx, toggle] of [['mma', 'toggleMmaAltView'], ['mtm', 'toggleMtmAltView']]) {
    if (typeof ctx[toggle] !== 'function') continue;
    const svgs = paneIds.filter(id => id.startsWith(`${TAB}-${pfx}-`));
    const btn = els.get(`${TAB}-${pfx}-alt-btn`);
    for (const want of [true, false]) {
      step(`${toggle}() -> ${want ? 'on' : 'off'}`, () => {
        ctx[toggle](TAB);
        if (btn && btn.classList.contains('active') !== want)
          throw new Error(`button active = ${btn.classList.contains('active')}, want ${want}`);
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

  // ── a blank atom_layout_mnk must RENDER, not error ───────────────────────
  //  Both MMA tabs ship the box filled, so nothing else in the suite reaches
  //  the blank path — and blank is the API default (the argument omitted),
  //  not a missing input.
  for (const [pfx, render] of [['mtm', 'renderMakeTiledMma'], ['pabc', 'renderPartitionABC']]) {
    const field = els.get(`${TAB}-${pfx}-atomlayout-input`);
    if (!field || typeof ctx[render] !== 'function') continue;
    const svgs = paneIds.filter(id => id.startsWith(`${TAB}-${pfx}-`));
    const was = field.value;
    step(`${pfx}: blank atom_layout_mnk renders as (1,1,1)`, () => {
      field.value = '';
      ctx[render](TAB);
    }, svgs);
    step(`${pfx}: restoring it renders again`, () => {
      field.value = was;
      ctx[render](TAB);
    }, svgs);
  }

  // Packed-source unpacking is fixed to None. Legacy URLs must still open,
  // render both panes, and export the current five-input form.
  step('make_copy_atom has no unpack selector or unpack presets', () => {
    const markup = ctx.generateMakeCopyAtomTabContent(TAB);
    if (markup.includes('mca-ub-input') || markup.includes('mca-ub-group'))
      throw new Error('unpack selector remains in the template');
    if (ctx.MCA_PRESETS.some(p => p.ub !== undefined))
      throw new Error('packed-source unpacking preset remains');
  });
  for (const op of ['ldmatrix16x8x8b', 'ldmatrix16x16x8b']) {
    for (const legacyBits of [4, 6]) {
      step(`${op}: legacy unpack_bits=${legacyBits} normalizes to None`, () => {
        const key = `make_copy_atom-${op}-128-int8_t-2-1-${legacyBits}`;
        const realSwitch = ctx.switchInnerTab;
        const realExport = ctx.exportURL;
        ctx.switchInnerTab = () => {};
        try {
          ctx.location.search = `?key=${encodeURIComponent(key)}`;
          ctx.applyKeyParam(TAB);
          if (ctx.mcaState[TAB].unpackBits !== null)
            throw new Error('legacy URL restored an unpacking mode');
          const result = els.get(`${TAB}-mca-atom-result`).innerHTML;
          if (/b4x16_p64|b6x16_p32/.test(result))
            throw new Error('packed-source qualifier remains in the result');
          let exported;
          ctx.exportURL = (btn, feature, ...inputs) => { exported = {feature, inputs}; };
          ctx.exportMCA(TAB);
          if (!exported || exported.feature !== 'make_copy_atom' ||
              exported.inputs.join('-') !== `${op}-128-int8_t-2-1`)
            throw new Error('export did not omit legacy unpack_bits');
          ctx.location.search = `?key=${encodeURIComponent(
            [exported.feature, ...exported.inputs].join('-'))}`;
          ctx.applyKeyParam(TAB);
          if (ctx.mcaState[TAB].unpackBits !== null || ctx.mcaState[TAB].opKey !== op)
            throw new Error('canonical URL did not round trip');
        } finally {
          ctx.switchInnerTab = realSwitch;
          ctx.exportURL = realExport;
          ctx.location.search = '';
        }
      }, paneIds.filter(id => id.startsWith(`${TAB}-mca-`)));
    }
  }

  step('gemm input groups use the shared numbered collapsible sections', () => {
    for (const [key,title,ids] of [
      ['mma','1. TiledMMA',['gemm-mma-summary']],
      ['fragments','2. Per-thread RMEM fragments',['gemm-a-input','gemm-b-input','gemm-c-input','gemm-d-input']]
    ]) {
      const opening = `<details class="cuo-section" id="${TAB}-gemm-${key}-section" open>`;
      const start = html.indexOf(opening);
      if (start < 0) throw new Error(`Missing shared section: ${title}`);
      let depth = 0, end = start;
      for (const match of html.slice(start).matchAll(/<details\b[^>]*>|<\/details>/g)) {
        depth += match[0].startsWith('</') ? -1 : 1;
        if (depth === 0) { end = start + match.index + match[0].length; break; }
      }
      const section = html.slice(start, end);
      if (!section.includes(`<summary>${title}</summary>`) || !section.includes('class="cuo-section-body"'))
        throw new Error(`Missing shared section: ${title}`);
      for (const id of ids) if (!section.includes(`id="${TAB}-${id}"`)) throw new Error(`${id} is outside its input section`);
      if (/onclick="(?:renderGemm|exportGemm)\(/.test(section)) throw new Error('Render/Export actions are inside an input section');
    }
  });
  step('TiledMMA manual construction is folded by default in downstream tabs', () => {
    for (const prefix of ['gemm-mma','pabc','mcab']) {
      const opening = html.match(new RegExp(`<details class="cuo-section" id="${TAB}-${prefix}-manual-construction"([^>]*)>`));
      if (!opening || /\bopen\b/.test(opening[1])) throw new Error(`${prefix} manual controls are not initially folded`);
      for (const key of ['op','ab','acc','k','atomlayout','perm'])
        if (!els.has(`${TAB}-${prefix}-${key}-input`)) throw new Error(`${prefix} is missing editable ${key}`);
    }
  });
  step('gemm manual construction edits affect the visualization and URL export', () => {
    const realExport = ctx.exportURL;
    try {
      ctx.setGemm(TAB, ctx.GEMM_DEFAULT_MMA, '(8,2):(1,8)', '(4,3):(1,4)', '(4,2,3):(1,4,8)', '', 0);
      const before = els.get(`${TAB}-gemm-result`).textContent;
      els.get(`${TAB}-gemm-mma-atomlayout-input`).value = '(1,1,1)';
      ctx.renderGemm(TAB);
      const after = els.get(`${TAB}-gemm-result`).textContent;
      if (after === before || !els.get(`${TAB}-gemm-mma-summary`).textContent.includes('1 warps, 32 threads'))
        throw new Error('Manual tiling did not update the summary');
      if (ctx.gemmState[TAB].result.mma.tiled.warps !== 1) throw new Error('Manual tiling did not reach computation');
      els.get(`${TAB}-gemm-mma-atomlayout-input`).value = '(2,1,1)';
      let exported;
      ctx.exportURL = (btn, feature, ...inputs) => { exported = inputs; };
      ctx.exportGemm(TAB);
      if (exported[4] !== '(2,1,1)') throw new Error('Export used stale constructor state');
    } finally { ctx.exportURL = realExport; }
  }, [`${TAB}-gemm-a-svg`,`${TAB}-gemm-b-svg`,`${TAB}-gemm-tile-svg`]);
  step('MMAAtom and CopyAtom sections use type names and default-folded manual controls', () => {
    for (const prefix of ['mma','mtm','mfrag','gemm-mma','pabc','mcab','psd-manual-mma']) {
      if (!html.includes(`id="${TAB}-${prefix}-decode-atom"`)) throw new Error(`Missing MMAAtom decoder ${prefix}`);
      const tag = html.match(new RegExp(`<details class="cuo-section" id="${TAB}-${prefix}-atom-manual-construction"([^>]*)>`));
      if (!tag || /\bopen\b/.test(tag[1])) throw new Error(`MMAAtom ${prefix} manual section is not folded`);
    }
    for (const prefix of ['mca','mtc','mtv']) {
      if (!html.includes(`id="${TAB}-${prefix}-decode-atom"`)) throw new Error(`Missing CopyAtom decoder ${prefix}`);
      const tag = html.match(new RegExp(`<details class="cuo-section" id="${TAB}-${prefix}-atom-manual-construction"([^>]*)>`));
      if (!tag || /\bopen\b/.test(tag[1])) throw new Error(`CopyAtom ${prefix} manual section is not folded`);
    }
    if (html.includes('<summary>1. The MMA Op</summary>') || html.includes('<summary>1. The Copy Op</summary>') ||
        !html.includes('<summary>1. MMAAtom</summary>') || !html.includes('<summary>1. CopyAtom</summary>'))
      throw new Error('Atom section names are inconsistent');
  });
  step('MMAAtom decode fills every consuming tab and preserves surrounding inputs', () => {
    const url = '?key=make_mma_atom-f16bf16-bfloat16_t-float-16';
    const targets = [
      ['mma','renderMakeMmaAtom', () => ctx.setMMA(TAB,'f16bf16','half_t','float',16)],
      ['mtm','renderMakeTiledMma', () => ctx.setMTM(TAB,'f16bf16','half_t','float',16,'(2,2,1)','')],
      ['mfrag','renderMakeFragmentABC', () => ctx.setMfrag(TAB,'C','f16bf16','half_t','float',16,'((2,2),2,2):((1,256),1024,16)',0)],
      ['pabc','renderPartitionABC', () => ctx.setPABC(TAB,'C','f16bf16','half_t','float',16,'(2,2,1)','','5','(64,64):(1,64)')],
      ['gemm-mma','renderGemm', () => ctx.setGemm(TAB,ctx.GEMM_DEFAULT_MMA,'(8,2):(1,8)','(4,3):(1,4)','(4,2,3):(1,4,8)','',0)],
      ['mcab','renderMakeTiledCopyAB', () => ctx.setMCAB(TAB,'A','f16bf16','half_t','float',16,'(2,2,1)','(32,32,16)','ldmatrix',4,0,16)],
      ['psd-manual-mma','renderPartitionSD', () => ctx.setPSDURL(TAB,'S','?key=make_tiled_copy_ab-A-f16bf16-half_t-float-16-(2,2,1)-(32,32,16)-ldmatrix-4-0-16-warp','0','(64,32):(32,1)')],
    ];
    for (const [prefix, render, setup] of targets) {
      setup();
      const other = ['pabc-tensor-input','mfrag-partition-input','gemm-a-input','mcab-copy-input','psd-tensor-input'].map(x => els.get(`${TAB}-${x}`).value);
      ctx.mmaImportAtom(TAB,prefix,render,url);
      for (const [key,value] of [['op','f16bf16'],['ab','bfloat16_t'],['acc','float'],['k','16']])
        if (els.get(`${TAB}-${prefix}-${key}-input`).value !== value) throw new Error(`${prefix} did not decode ${key}`);
      if (JSON.stringify(other) !== JSON.stringify(['pabc-tensor-input','mfrag-partition-input','gemm-a-input','mcab-copy-input','psd-tensor-input'].map(x => els.get(`${TAB}-${x}`).value)))
        throw new Error('MMAAtom decode changed unrelated inputs');
    }
    const before = ['op','ab','acc','k'].map(x => els.get(`${TAB}-mma-${x}-input`).value);
    let rejected = false;
    try { ctx.mmaImportAtom(TAB,'mma','renderMakeMmaAtom','?key=make_mma_atom-f16bf16-bfloat16_t-half_t-16'); } catch (_) { rejected = true; }
    if (!rejected || JSON.stringify(before) !== JSON.stringify(['op','ab','acc','k'].map(x => els.get(`${TAB}-mma-${x}-input`).value)))
      throw new Error('Rejected MMAAtom decode altered controls');
    for (const [op,ab,acc,k] of [['tf32','na','na',8],['fp8','float_e4m3_t','float',32]]) {
      ctx.mmaImportAtom(TAB,'mma','renderMakeMmaAtom',`?key=make_mma_atom-${op}-${ab}-${acc}-${k}`);
      if (ctx.mmaState[TAB].opKey !== op || els.get(`${TAB}-mma-k-input`).value !== String(k))
        throw new Error('MMAAtom decode did not rebuild dependent options');
    }
  });
  step('WGMMA renders descriptor tiles and 128-thread register ownership in both views', () => {
    for (const [source,am,bm] of [['SMEM','K','K'],['SMEM','MN','MN'],['RMEM','K','MN']]) {
      ctx.setMMA(TAB,'wg_f16bf16','half_t','float',16,24,source,am,bm);
      ctx.mmaState[TAB].altView = false;
      ctx.mmaState[TAB].showValue = false;
      els.get(`${TAB}-mma-highlight-tid`).value = '127';
      ctx.renderMakeMmaAtom(TAB);
      const s = ctx.mmaState[TAB];
      if (s.thrId.shape !== 128 || s.highlightTid !== 127) throw new Error('WGMMA did not accept thread 127');
      const before = ['A','B','C'].map(key => s[key].str).join(';');
      for (const [side,space] of [['a',source],['b','SMEM'],['c','RMEM']]) {
        if (els.get(`${TAB}-mma-${side}-space`).textContent !== space) throw new Error('Wrong WGMMA memory badge');
        const svg = els.get(`${TAB}-mma-${side}-svg`).innerHTML;
        if (space === 'SMEM') {
          if (!svg.includes('fill="#a7f3d0"') || svg.includes('opacity=')) throw new Error('SMEM tile has thread coloring or highlighting');
          if (els.get(`${TAB}-mma-${side}-title`).textContent.includes('per lane')) throw new Error('Descriptor shown as per-lane registers');
        }
      }
      const title = side => els.get(`${TAB}-mma-${side}-title`).textContent;
      if (!title('a').includes(am === 'MN' ? '16×64' : '64×16') || !title('b').includes(bm === 'MN' ? '16×24' : '24×16'))
        throw new Error('Default view does not follow major modes');
      for (const [side,mode,axis] of [['a',am,'M'],['b',bm,'N']]) {
        const label = `${mode === 'MN' ? axis : 'K'}-major`;
        if (!els.get(`${TAB}-mma-${side}-desc`).innerHTML.includes(label) && (side === 'b' || source === 'SMEM'))
          throw new Error('Operand description uses the shared enum name instead of its axis');
        if (!els.get(`${TAB}-mma-${side}major-input`).innerHTML.includes(`>${label}</option>`))
          throw new Error('Major-mode options do not name the operand axis');
      }
      const smemBefore = els.get(`${TAB}-mma-b-svg`).innerHTML;
      ctx.toggleMmaValue(TAB);
      if (els.get(`${TAB}-mma-b-svg`).innerHTML === smemBefore) throw new Error('Value toggle did not update SMEM labels');
      ctx.toggleMmaAltView(TAB);
      if (!title('a').includes('64×16') || !title('b').includes('16×24') || !title('c').includes('64×24'))
        throw new Error('Alternative view does not show MK × KN');
      if (before !== ['A','B','C'].map(key => s[key].str).join(';')) throw new Error('Display rotation changed TV mapping');
    }
    els.get(`${TAB}-mma-highlight-tid`).value = '';
  }, ['a','b','c'].map(side => `${TAB}-mma-${side}-svg`));
  step('WGMMA export, clipboard decode and deep links preserve the editable constructor', () => {
    const realExport = ctx.exportURL, realSwitch = ctx.switchInnerTab;
    try {
      ctx.switchInnerTab = () => {};
      ctx.setMMA(TAB,'wg_fp8','float_e4m3_t','half_t',32,128,'RMEM','K','K','float_e5m2_t');
      // Export must use edits, even before Render.
      els.get(`${TAB}-mma-n-input`).value = '24';
      let exported;
      ctx.exportURL = (btn,feature,...inputs) => { exported = [feature,...inputs]; };
      ctx.exportMMA(TAB);
      const url = '?key=' + encodeURIComponent(exported.join('-'));
      ctx.setMMA(TAB,'f16bf16','half_t','float',16);
      ctx.mmaImportAtom(TAB,'mma','renderMakeMmaAtom',url);
      if (ctx.mmaState[TAB].n !== 24 || ctx.mmaState[TAB].bDtype !== 'float_e5m2_t' || ctx.mmaState[TAB].aSrc !== 'RMEM')
        throw new Error('Clipboard lost WGMMA fields');
      ctx.setMMA(TAB,'f16bf16','half_t','float',16);
      ctx.location.search = url;
      ctx.applyKeyParam(TAB);
      if (JSON.stringify(ctx.mmaReadAtomInputs(TAB)) !== JSON.stringify(exported.slice(1))) throw new Error('Deep link lost WGMMA fields');
      const before = JSON.stringify(ctx.mmaReadAtomInputs(TAB));
      for (const invalid of [url.replace('wg_fp8','missing'),'?key=make_mma_atom-wg_int8-int8_t-int32_t-32-40-SMEM-K-K-int8_t']) {
        let rejected = false;
        try { ctx.mmaImportAtom(TAB,'mma','renderMakeMmaAtom',invalid); } catch (_) { rejected = true; }
        if (!rejected || JSON.stringify(ctx.mmaReadAtomInputs(TAB)) !== before) throw new Error('Failed WGMMA decode altered controls');
      }
      // Source-dependent major options must be rebuilt before MN is assigned.
      ctx.mmaImportAtom(TAB,'mma','renderMakeMmaAtom','?key=make_mma_atom-wg_f16bf16-half_t-float-16-64-SMEM-MN-MN-half_t');
      if (ctx.mmaState[TAB].aMajor !== 'MN') throw new Error('SMEM major mode was assigned before options');
      for (const p of ['mfrag','pabc','gemm-mma','mcab','psd-manual-mma']) {
        const fields = ['op','ab','acc','k'];
        const previous = fields.map(key => els.get(`${TAB}-${p}-${key}-input`).value);
        let rejected = false;
        try { ctx.mmaAssignAtomInputs(TAB,p,exported.slice(1)); } catch (_) { rejected = true; }
        if (!rejected || JSON.stringify(previous) !== JSON.stringify(fields.map(key => els.get(`${TAB}-${p}-${key}-input`).value)))
          throw new Error('Warp-only consumer accepted a WGMMA atom or changed controls');
        if (els.get(`${TAB}-${p}-op-input`).innerHTML.includes('warpgroup.')) throw new Error('Warp-only picker exposes WGMMA');
      }
      ctx.setMMA(TAB,'wg_int8','int8_t','int32_t',32,24,'SMEM','K','K','uint8_t');
      if (ctx.mmaState[TAB].n !== 24) throw new Error('INT8 lost its N input');
      if (els.get(`${TAB}-mma-asrc-input`).innerHTML.includes('RMEM'))
        throw new Error('INT8 exposes the CuTeDSL register signedness failure');
    } finally {
      ctx.exportURL = realExport; ctx.switchInnerTab = realSwitch; ctx.location.search = '';
      ctx.setMMA(TAB,'f16bf16','half_t','float',16);
    }
  }, ['a','b','c'].map(side => `${TAB}-mma-${side}-svg`));
  step('WGMMA N is editable and illegal values are rejected without replacing the input', () => {
    if (!html.includes(`<input type="number" id="${TAB}-mma-n-input"`)) throw new Error('N is not an input box');
    for (const [op,k,ab,acc,invalid] of [
      ['wg_f16bf16',16,'half_t','float',['','0','7','12','264','16.5','abc']],
      ['wg_int8',32,'int8_t','int32_t',['40','56','256.5']]
    ]) {
      ctx.setMMA(TAB,op,ab,acc,k,24);
      const input = els.get(`${TAB}-mma-n-input`);
      for (const value of invalid) {
        input.value = value;
        ctx.renderMakeMmaAtom(TAB);
        if (input.value !== value) throw new Error('Illegal N was silently replaced');
        if (!els.get(`${TAB}-mma-error`).textContent.includes('Illegal N')) throw new Error('Illegal N was not rejected');
        ctx.toggleMmaAltView(TAB);
        ctx.toggleMmaValue(TAB);
        for (const side of ['a','b','c']) if (els.get(`${TAB}-mma-${side}-svg`).innerHTML) throw new Error('Illegal N retained a visualization');
      }
      input.value = '32';
      ctx.renderMakeMmaAtom(TAB);
      if (ctx.mmaState[TAB].n !== 32 || els.get(`${TAB}-mma-error`).textContent) throw new Error('Legal N did not recover');
    }
    ctx.setMMA(TAB,'f16bf16','half_t','float',16);
  }, ['a','b','c'].map(side => `${TAB}-mma-${side}-svg`));
  step('Tiled WGMMA decodes an MMAAtom and offers TVs, warps and warp groups', () => {
    ctx.setMTM(TAB,'f16bf16','half_t','float',16,'(2,1,1)','');
    ctx.mmaImportAtom(TAB,'mtm','renderMakeTiledMma','?key=make_mma_atom-wg_f16bf16-half_t-float-16-16-RMEM-K-MN-half_t');
    const s = ctx.mtmState[TAB];
    if (!s.atom.warpgroup || s.permStage.threads !== 256 || s.permStage.warps !== 8 || s.permStage.atomGroups !== 2)
      throw new Error('WGMMA replication did not count 128 threads per atom');
    if (s.mode !== 'warpgroup' || els.get(`${TAB}-mtm-group-mode-btn`).style.display === 'none')
      throw new Error('Show Warp Groups is missing');
    s.altView = false; s.showValue = false;
    ctx.renderMakeTiledMma(TAB);
    const smem = els.get(`${TAB}-mtm-warp-b-svg`).innerHTML;
    els.get(`${TAB}-mtm-focus-input`).value = '1';
    ctx.setMtmHighlight(TAB);
    if (els.get(`${TAB}-mtm-focus-label`).textContent !== 'Warp group id' || !els.get(`${TAB}-mtm-warp-c-svg`).innerHTML.includes('WG1'))
      throw new Error('Group focus does not label WG1');
    ctx.setMtmMode(TAB,'warp');
    if (els.get(`${TAB}-mtm-focus-label`).textContent !== 'Warp id' || !els.get(`${TAB}-mtm-warp-c-svg`).innerHTML.includes('W1'))
      throw new Error('Physical warp mode was lost');
    els.get(`${TAB}-mtm-focus-input`).value = '255';
    ctx.setMtmMode(TAB,'tv');
    if (!els.get(`${TAB}-mtm-warp-c-svg`).innerHTML.includes('T255')) throw new Error('TV mode lost thread 255');
    if (els.get(`${TAB}-mtm-warp-b-svg`).innerHTML !== smem) throw new Error('Thread/group focus changed SMEM coloring');
    for (const stage of ['warp','perm']) for (const [side,space] of [['a','RMEM'],['b','SMEM'],['c','RMEM']])
      if (els.get(`${TAB}-mtm-${stage}-${side}-space`).textContent !== space) throw new Error('Incorrect tiled memory badge');
    els.get(`${TAB}-mtm-focus-input`).value = '';
    ctx.setMtmMode(TAB,'warpgroup');
  }, ['warp','perm'].flatMap(stage=>['a','b','c'].map(side=>`${TAB}-mtm-${stage}-${side}-svg`)));
  step('Tiled WGMMA default and matrix views preserve operand layouts', () => {
    ctx.setMTM(TAB,'wg_f16bf16','bfloat16_t','float',16,'(1,1,1)','',24,'SMEM','MN','MN');
    const s = ctx.mtmState[TAB]; s.altView = false;
    ctx.renderMakeTiledMma(TAB);
    const tv = ['A','B','C'].map(key=>ctx.formatLayoutStr(s.permStage[key].tv.shape,s.permStage[key].tv.stride)).join(';');
    const title = side=>els.get(`${TAB}-mtm-perm-${side}-title`).textContent;
    if (!title('a').includes('16×64') || !title('b').includes('16×24')) throw new Error('Default view ignores M/N-major modes');
    ctx.toggleMtmAltView(TAB);
    if (!title('a').includes('64×16') || !title('b').includes('16×24')) throw new Error('Matrix view does not show MK × KN');
    if (tv !== ['A','B','C'].map(key=>ctx.formatLayoutStr(ctx.mtmState[TAB].permStage[key].tv.shape,ctx.mtmState[TAB].permStage[key].tv.stride)).join(';'))
      throw new Error('Display orientation changed the TV layouts');
    ctx.toggleMtmValue(TAB);
  }, ['warp','perm'].flatMap(stage=>['a','b','c'].map(side=>`${TAB}-mtm-${stage}-${side}-svg`)));
  step('Tiled WGMMA exports current edits and decodes its complete constructor transactionally', () => {
    const realExport=ctx.exportURL, realSwitch=ctx.switchInnerTab;
    try {
      ctx.switchInnerTab=()=>{};
      ctx.setMTM(TAB,'wg_fp8','float_e4m3_t','float',32,'(2,1,1)','',32,'RMEM','K','K','float_e5m2_t');
      els.get(`${TAB}-mtm-n-input`).value='24';
      let exported;
      ctx.exportURL=(btn,feature,...inputs)=>{exported=[feature,...inputs];};
      ctx.exportMTM(TAB);
      const url='?key='+encodeURIComponent(exported.join('-'));
      ctx.setMTM(TAB,'f16bf16','half_t','float',16,'(1,1,1)','');
      ctx.mtmImportConstructor(TAB,url);
      if (JSON.stringify(ctx.mtmReadConstructorControls(TAB,'mtm'))!==JSON.stringify(exported.slice(1))) throw new Error('Tiled WGMMA clipboard round trip lost fields');
      ctx.setMTM(TAB,'f16bf16','half_t','float',16,'(1,1,1)','');
      ctx.location.search=url;ctx.applyKeyParam(TAB);
      if (ctx.mtmState[TAB].atom.n!==24 || ctx.mtmState[TAB].atom.bDtype!=='float_e5m2_t') throw new Error('Deep link lost WGMMA fields');
      const before=JSON.stringify(ctx.mtmReadConstructorControls(TAB,'mtm'));
      for (const bad of [url.replace('wg_fp8','missing'),'?key=make_tiled_mma-wg_int8-int8_t-int32_t-32-(1,1,1)-na-40-SMEM-K-K-int8_t']) {
        let rejected=false;try{ctx.mtmImportConstructor(TAB,bad);}catch(_){rejected=true;}
        if(!rejected || JSON.stringify(ctx.mtmReadConstructorControls(TAB,'mtm'))!==before) throw new Error('Invalid import changed constructor fields');
      }
      for(const p of ['pabc','mcab','gemm-mma','psd-manual-mma']) {
        let rejected=false;try{ctx.mtmWriteConstructorControls(TAB,p,exported.slice(1));}catch(_){rejected=true;}
        if(!rejected)throw new Error('Unsupported downstream consumer accepted WGMMA');
      }
      els.get(`${TAB}-mtm-perm-input`).value='(128,48,32)';
      ctx.renderMakeTiledMma(TAB);
      const keep=JSON.stringify(ctx.mtmReadConstructorControls(TAB,'mtm'));
      let rejected=false;
      try { ctx.mmaImportAtom(TAB,'mtm','renderMakeTiledMma','?key=make_mma_atom-wg_f16bf16-half_t-float-16-64-SMEM-K-K-half_t'); }
      catch (_) { rejected=true; }
      if(!rejected || JSON.stringify(ctx.mtmReadConstructorControls(TAB,'mtm'))!==keep)
        throw new Error('An atom incompatible with the current tiling changed controls');
      els.get(`${TAB}-mtm-n-input`).value='12';ctx.renderMakeTiledMma(TAB);
      if(!els.get(`${TAB}-mtm-error`).textContent.includes('Illegal N'))throw new Error('Tiled N input did not reject illegal values');
    } finally {
      ctx.exportURL=realExport;ctx.switchInnerTab=realSwitch;ctx.location.search='';
      els.get(`${TAB}-mtm-focus-input`).value='';
      ctx.setMTM(TAB,'f16bf16','half_t','float',16,'(2,2,1)','');
    }
  }, ['warp','perm'].flatMap(stage=>['a','b','c'].map(side=>`${TAB}-mtm-${stage}-${side}-svg`)));
  step('CopyAtom decode updates source and tiled tabs with each consumer domain enforced', () => {
    ctx.setMTC(TAB,'universal',128,'half_t','((8,16),8):((128,1),16)','(16,64)');
    ctx.setMTV(TAB,'universal',128,'half_t','(16,8):(8,1)','(1,8):(1,1)');
    for (const [p,render] of [['mca','renderMakeCopyAtom'],['mtc','renderMakeTiledCopy'],['mtv','renderMakeTiledCopyTv']]) {
      ctx.mcaImportAtom(TAB,p,render,'?key=make_copy_atom-universal-32-half_t');
      if (els.get(`${TAB}-${p}-bits-input`).value !== '32' || els.get(`${TAB}-${p}-dtype-input`).value !== 'half_t')
        throw new Error(`${p} did not decode the CopyAtom`);
    }
    ctx.mcaImportAtom(TAB,'mca','renderMakeCopyAtom','?key=make_copy_atom-ldmatrix16x16x8b-128-int8_t-2-1');
    if (els.get(`${TAB}-mca-nm-input`).value !== '2' || els.get(`${TAB}-mca-trans-input`).value !== '1')
      throw new Error('CopyAtom decoder did not rebuild matrix/transpose options');
    const before = ['op','bits','dtype'].map(x => els.get(`${TAB}-mtc-${x}-input`).value);
    let rejected = false;
    try { ctx.mcaImportAtom(TAB,'mtc','renderMakeTiledCopy','?key=make_copy_atom-ldmatrix-128-half_t-4-0'); } catch (_) { rejected = true; }
    if (!rejected || JSON.stringify(before) !== JSON.stringify(['op','bits','dtype'].map(x => els.get(`${TAB}-mtc-${x}-input`).value)))
      throw new Error('SIMT-only tab accepted or mutated fields for an unsupported CopyAtom');
    ctx.setPSDURL(TAB,'S','?key=make_tiled_copy_ab-A-f16bf16-half_t-half_t-16-(2,2,1)-(32,32,16)-ldmatrix-4-0-16-warp','0','(128,32,4):(32,1,4096)');
    ctx.mtcImportManualCopyAtom(TAB,'psd-manual','renderPartitionSD','?key=make_copy_atom-ldmatrix-128-half_t-1-0');
    const refs = JSON.parse(fs.readFileSync(path.join(__dirname,'reference.json'),'utf8'));
    if (!els.get(`${TAB}-psd-result`).innerHTML.includes(refs.partition_sd.url_A_x1_S.layout.str))
      throw new Error('CopyAtom decode in partition_S/D differs from CuTeDSL');
  });
  step('gemm documents all five dispatches and their normalized forms', () => {
    for (let i = 1; i <= 5; i++) {
      if (!html.includes(`id="${TAB}-gemm-rule-${i}"`) || !html.includes(`Dispatch [${i}]`))
        throw new Error(`Dispatch [${i}] is missing from the reference`);
    }
    for (const form of ['(V,1,1) × (V,1,1) → (V,1,1)', '(1,M,1) × (1,N,1) → (1,M,N)',
                        '(1,M,K) × (1,N,K) → (1,M,N)', '(V,M,1) × (V,N,1) → (V,M,N)',
                        '(V,M,K) × (V,N,K) → (V,M,N)'])
      if (!html.includes(form)) throw new Error(`Missing dispatch form ${form}`);
    if (!html.includes('Use [4] for one K slice, or [5]')) throw new Error('Backend support is not explained');
    const rejected = /<tr id="tab1-gemm-rule-1">([\s\S]*?)<\/tr>/.exec(html);
    if (!rejected || (rejected[1].match(/<s>/g) || []).length !== 3) throw new Error('Rejected rule [1] is not struck through');
  });
  step('gemm Highlight K follows explicit-K dispatches, including singleton K', () => {
    for (const dispatch of [null, 1, 2, 3, 4, 5, 4, null]) {
      ctx.gemmHighlightDispatch(TAB, dispatch);
      if (els.get(`${TAB}-gemm-k-control`).hidden !== !(dispatch === 3 || dispatch === 5))
        throw new Error(`Wrong K visibility for dispatch ${dispatch}`);
    }
  });
  step('gemm: oracle dispatches, complete coverage, and K selection', () => {
    const cases = JSON.parse(fs.readFileSync(path.join(__dirname, 'cases.json'), 'utf8')).gemm;
    const refs = JSON.parse(fs.readFileSync(path.join(__dirname, 'reference.json'), 'utf8')).gemm;
    for (const c of cases) {
      const ref = refs[c.id];
      if (ref.error) continue;
      ctx.setGemm(TAB, c.mma, c.a, c.b, c.c, c.d, ref.mnk[2] - 1);
      const query = `?key=${encodeURIComponent(['make_tiled_mma',...c.mma].join('-'))}`;
      ctx.gemmImportMma(TAB, query);
      if (JSON.stringify(ctx.mtmReadConstructorControls(TAB, 'gemm-mma')) !== JSON.stringify(c.mma))
        throw new Error(`gemm ${c.id} decoded controls differ from imported constructor`);
      const hasK = ref.dispatch === 3 || ref.dispatch === 5;
      if (els.get(`${TAB}-gemm-k-control`).hidden !== !hasK) throw new Error(`gemm ${c.id} K control visibility does not follow dispatch`);
      if (els.get(`${TAB}-gemm-desc`).textContent.includes('Highlighted K') !== hasK) throw new Error(`gemm ${c.id} describes an unused K highlight`);
      const text = els.get(`${TAB}-gemm-result`).textContent;
      const product = ([m,n,k]) => `(${m}, ${k}) @ (${k}, ${n}) = (${m}, ${n})`;
      for (const line of [`Dispatch rule applied: [${ref.dispatch}]`,
                          `TiledMMA: ${product(ref.base_mnk)}`, `GEMM: ${product(ref.mnk)}`,
                          `Final result: ${product(ref.final_mnk)}`]) {
        if (!text.includes(line)) throw new Error(`gemm ${c.id} summary differs from CuTeDSL: ${line}`);
      }
      const d = els.get(`${TAB}-gemm-d-input`);
      if (!d.disabled || d.value !== c.c || ctx.gemmState[TAB].result.D !== undefined &&
          JSON.stringify(ctx.gemmState[TAB].result.calls.map(x => x.d)) !== JSON.stringify(ctx.gemmState[TAB].result.calls.map(x => x.c)))
        throw new Error('D is not disabled and synchronized with C');
      for (let i = 1; i <= 5; i++) {
        if (els.get(`${TAB}-gemm-rule-${i}`).classList.contains('active') !== (i === ref.dispatch))
          throw new Error(`gemm ${c.id} did not highlight only the oracle dispatch`);
      }
      const svg = els.get(`${TAB}-gemm-tile-svg`).innerHTML;
      if ((svg.match(/>TiledMMA<\/text>/g) || []).length !== ref.mnk[0] * ref.mnk[1])
        throw new Error(`gemm ${c.id} grid does not cover M × N`);
      for (const [which, count] of [['a',ref.mnk[0]*ref.mnk[2]],['b',ref.mnk[1]*ref.mnk[2]]]) {
        const pane = els.get(`${TAB}-gemm-${which}-svg`).innerHTML;
        if ((pane.match(/>TiledMMA<\/text>/g) || []).length !== count)
          throw new Error(`gemm ${c.id}/${which} does not show all K blocks`);
      }
      if (els.get(`${TAB}-gemm-warning`).textContent) throw new Error('Semantic GEMM modes incorrectly warned about flattening');
      if (/gemmSelectInvocation|cursor:pointer|#dc2626/.test(svg))
        throw new Error('Coverage grid retained cell selection or a red boundary');
      if (html.includes(`${TAB}-gemm-call`) || html.includes('scalar offsets/thread'))
        throw new Error('Register inspection remains in the gemm UI');
      if (els.get(`${TAB}-gemm-k-input`).value !== String(ref.mnk[2] - 1))
        throw new Error('K selector did not retain the requested block');
    }
  }, [`${TAB}-gemm-tile-svg`]);
  step('gemm: D follows live C edits and the default summary distinguishes scalar and invocation units', () => {
    const d = els.get(`${TAB}-gemm-d-input`);
    els.get(`${TAB}-gemm-c-input`).value = '(4,2,3):(1,4,8)';
    ctx.gemmSyncD(TAB);
    if (!d.disabled || d.value !== '(4,2,3):(1,4,8)') throw new Error('D did not follow C edit');
    const field = html.match(new RegExp(`<input[^>]*id="${TAB}-gemm-d-input"[^>]*>`));
    if (!field || !/\bdisabled\b/.test(field[0])) throw new Error('D was not disabled in the initial markup');
    ctx.setGemm(TAB, ctx.GEMM_DEFAULT_MMA, '((2,2,2),4):((1,2,4),16)', '((2,2),8):((1,2),8)', '((2,2),4,8):((1,2),4,16)', '', 0);
    const text = els.get(`${TAB}-gemm-result`).textContent;
    if (!text.includes('Each element represents a TiledMMA invocation.') || !text.includes('Each element is a scalar.'))
      throw new Error('Result units are not explained');
  }, [`${TAB}-gemm-a-svg`,`${TAB}-gemm-b-svg`,`${TAB}-gemm-tile-svg`]);
  step('gemm: alternative MK / KN / MN view transposes B and preserves the operands', () => {
    ctx.setGemm(TAB, ctx.GEMM_DEFAULT_MMA, '(8,2,2):(1,16,8)', '(4,3,2):(1,8,4)', '(4,2,3):(1,4,8)', '', 1);
    const original = ctx.gemmState[TAB].result;
    const a = els.get(`${TAB}-gemm-a-svg`).innerHTML, c = els.get(`${TAB}-gemm-tile-svg`).innerHTML;
    const b = els.get(`${TAB}-gemm-b-svg`).innerHTML;
    if (!els.get(`${TAB}-gemm-b-title`).textContent.includes('N × K (3 × 2)')) throw new Error('Default B is not NK');
    ctx.toggleGemmAltView(TAB);
    if (!els.get(`${TAB}-gemm-b-title`).textContent.includes('K × N (2 × 3)')) throw new Error('Alternative B is not KN');
    if (!els.get(`${TAB}-gemm-results`).classList.contains('mma-alt') || !els.get(`${TAB}-gemm-alt-btn`).classList.contains('active'))
      throw new Error('Alternative view did not use the shared MMA layout and toggle');
    if (a !== els.get(`${TAB}-gemm-a-svg`).innerHTML || c !== els.get(`${TAB}-gemm-tile-svg`).innerHTML)
      throw new Error('Transposing B changed A or C');
    if (JSON.stringify(original.calls) !== JSON.stringify(ctx.gemmState[TAB].result.calls))
      throw new Error('Alternative view changed invocation operands');
    if (!els.get(`${TAB}-gemm-rule-${original.dispatch}`).classList.contains('active'))
      throw new Error('Alternative view changed the applied dispatch');
    const alt = els.get(`${TAB}-gemm-b-svg`).innerHTML;
    if (alt === b || !els.get(`${TAB}-gemm-desc`).textContent.includes('B[1,:]')) throw new Error('B display was not transposed');
    els.get(`${TAB}-gemm-k-input`).value = '0';
    ctx.renderGemm(TAB);
    if (els.get(`${TAB}-gemm-b-svg`).innerHTML === alt) throw new Error('K highlight did not change B');
    ctx.toggleGemmAltView(TAB);
    els.get(`${TAB}-gemm-k-input`).value = '1';
    ctx.renderGemm(TAB);
    if (b !== els.get(`${TAB}-gemm-b-svg`).innerHTML) throw new Error('Toggling twice did not restore B');
  }, [`${TAB}-gemm-a-svg`,`${TAB}-gemm-b-svg`,`${TAB}-gemm-tile-svg`]);
  step('gemm: pasteboard constructor decode preserves fragment inputs and failed decode preserves MMA', () => {
    const a = els.get(`${TAB}-gemm-a-input`).value;
    const key = 'make_tiled_mma-f16bf16-half_t-half_t-16-(2,2,1)-(32,32,16)';
    ctx.gemmImportMma(TAB, `file:///project/index.html?key=${encodeURIComponent(key)}`);
    if (els.get(`${TAB}-gemm-a-input`).value !== a) throw new Error('Constructor decode overwrote A');
    if (!els.get(`${TAB}-gemm-mma-summary`).textContent.includes('4 warps, 128 threads'))
      throw new Error('Decoded TiledMMA summary missing');
    const before = JSON.stringify(ctx.gemmState[TAB].mmaInputs);
    const controlsBefore = JSON.stringify(ctx.mtmReadConstructorControls(TAB, 'gemm-mma'));
    for (const text of ['?key=layout-8:1', '?key=make_tiled_mma-f16bf16-half_t-half_t-16-(2,2,1)-(0,32,16)']) {
      let failed = false;
      try { ctx.gemmImportMma(TAB, text); } catch (_) { failed = true; }
      if (!failed || JSON.stringify(ctx.gemmState[TAB].mmaInputs) !== before)
        throw new Error('Invalid constructor decode replaced the last valid MMA');
      if (JSON.stringify(ctx.mtmReadConstructorControls(TAB, 'gemm-mma')) !== controlsBefore)
        throw new Error('Invalid decode changed manual fields');
    }
  }, [`${TAB}-gemm-tile-svg`]);
  step('gemm: exported URL restores constructor, A/B/C, shared D, and K', () => {
    const realSwitch = ctx.switchInnerTab, realExport = ctx.exportURL;
    try {
      ctx.switchInnerTab = () => {};
      const a = '(8,2,2):(1,16,8)', b = '(4,3,2):(1,8,4)';
      const c = '(4,2,3):(1,4,8)', d = '(4,2,3):(1,12,4)';
      ctx.setGemm(TAB, ctx.GEMM_DEFAULT_MMA, a, b, c, d, 1, true);
      const before = els.get(`${TAB}-gemm-tile-svg`).innerHTML;
      let exported;
      ctx.exportURL = (btn, feature, ...inputs) => { exported = [feature, ...inputs]; };
      ctx.exportGemm(TAB);
      if (exported[10] !== 'same') throw new Error('Disabled D did not export as same');
      ctx.setGemm(TAB, ctx.GEMM_DEFAULT_MMA, '(8,1):(1,0)', '(4,1):(1,0)', '(4,1,1):(1,0,0)', '', 0);
      ctx.location.search = `?key=${encodeURIComponent(exported.join('-'))}`;
      ctx.applyKeyParam(TAB);
      for (const [x, text] of Object.entries({a,b,c,d:c}))
        if (els.get(`${TAB}-gemm-${x}-input`).value !== text) throw new Error(`gemm URL lost ${x}`);
      if (els.get(`${TAB}-gemm-k-input`).value !== '1') throw new Error('gemm URL lost K selection');
      if (!ctx.gemmState[TAB].altView) throw new Error('gemm URL lost alternative view');
      if (els.get(`${TAB}-gemm-tile-svg`).innerHTML !== before) throw new Error('gemm URL changed coverage');
      if (!html.includes(`onclick="gemmDecodePasteboard('${TAB}')"`)) throw new Error('Pasteboard button is not wired');
    } finally {
      ctx.switchInnerTab = realSwitch;
      ctx.exportURL = realExport;
      ctx.location.search = '';
    }
  }, [`${TAB}-gemm-tile-svg`]);
  step('gemm: legacy URL defaults to MK / NK / MN', () => {
    const realSwitch = ctx.switchInnerTab;
    try {
      ctx.switchInnerTab = () => {};
      const inputs = [...ctx.GEMM_DEFAULT_MMA, '(8,2):(1,8)', '(4,3):(1,4)', '(4,2,3):(1,4,8)', '(4,2,3):(1,12,4)', '0'];
      ctx.location.search = `?key=${encodeURIComponent(['gemm',...inputs].join('-'))}`;
      ctx.applyKeyParam(TAB);
      if (ctx.gemmState[TAB].altView || !els.get(`${TAB}-gemm-b-title`).textContent.includes('N × K'))
        throw new Error('Legacy gemm URL did not restore default organization');
      if (els.get(`${TAB}-gemm-d-input`).value !== inputs[8]) throw new Error('Legacy D override was not normalized to C');
    } finally { ctx.switchInnerTab = realSwitch; ctx.location.search = ''; }
  }, [`${TAB}-gemm-a-svg`,`${TAB}-gemm-b-svg`,`${TAB}-gemm-tile-svg`]);
  step('gemm reports invalid fragment ranks and clears stale coverage', () => {
    ctx.setGemm(TAB, ctx.GEMM_DEFAULT_MMA, '(8,2):(1,8)', '4:1', '(4,2,1):(1,4,8)', '', 0);
    if (['a','b','tile'].some(x => els.get(`${TAB}-gemm-${x}-svg`).innerHTML))
      throw new Error('Invalid GEMM retained stale coverage');
    for (let i = 1; i <= 5; i++)
      if (els.get(`${TAB}-gemm-rule-${i}`).classList.contains('active'))
        throw new Error('Invalid GEMM retained an applied dispatch');
  }, [], {expectError: /A and B must have the same rank/});

  step('Double-click tab rename survives rendering; cancellation preserves the label', () => {
    const label=els.get(`${TAB}-label`), oldPrompt=ctx.prompt;
    ctx.prompt=()=> 'my MMA';
    ctx.renameOuterTab(TAB);
    ctx.updateOuterTabLabel(TAB,'automatic result label');
    if(label.textContent!=='my MMA') throw new Error('Render overwrote custom tab name');
    ctx.prompt=()=>null;ctx.renameOuterTab(TAB);
    if(label.textContent!=='my MMA') throw new Error('Cancel changed the tab name');
    ctx.prompt=()=> '  ';ctx.renameOuterTab(TAB);
    if(label.textContent!=='my MMA') throw new Error('Blank name changed the tab name');
    delete label.dataset.customName;ctx.prompt=oldPrompt;
    const ui=fs.readFileSync(path.join(ROOT,'ui.js'),'utf8');
    if(!ui.includes('ondblclick="event.stopPropagation(); renameOuterTab(')) throw new Error('Missing double-click handler');
  });
  step('TV Layout pasteboard input populates folded manual construction and renders', () => {
    const tag=html.match(new RegExp(`<details class="cuo-section" id="${TAB}-mtv-tv-manual-construction"([^>]*)>`));
    if(!tag || /\bopen\b/.test(tag[1])) throw new Error('TV manual controls must start folded');
    ctx.mtvImportTVLayout(TAB,'?key=tv-2-(4,8):(8,1)-(2,2):(1,2)');
    if(els.get(`${TAB}-mtv-thr-input`).value!=='(4,8):(8,1)' || els.get(`${TAB}-mtv-val-input`).value!=='(2,2):(1,2)') throw new Error('Decoded TV fields were not populated');
    if(els.get(`${TAB}-mtv-error`).textContent) throw new Error(els.get(`${TAB}-mtv-error`).textContent);
    const before=els.get(`${TAB}-mtv-thr-input`).value;
    let failed=false;try{ctx.mtvImportTVLayout(TAB,'?key=tv-1-(4,2):(0,1)-(4,2)');}catch(_){failed=true;}
    if(!failed || els.get(`${TAB}-mtv-thr-input`).value!==before) throw new Error('Failed import mutated controls');
  });
  step('Saved variables can apply a parsed visualization without changing the page URL', () => {
    const realSwitch=ctx.switchInnerTab;ctx.switchInnerTab=()=>{};
    try {
    ctx.applyKeyParam(TAB,{feature:'layout',inputs:['(4,8):(1,4)']});
    if(els.get(`${TAB}-layout-input`).value!=='(4,8):(1,4)') throw new Error('Parsed variable was not applied');
    if(els.get(`${TAB}-layout-error`).textContent) throw new Error(els.get(`${TAB}-layout-error`).textContent);
    ctx.applyKeyParam(TAB,{feature:'tv',method:'2',inputs:['(4,8):(8,1)','(2,2):(1,2)']});
    if(els.get(`${TAB}-tv-error`).textContent) throw new Error(els.get(`${TAB}-tv-error`).textContent);
    } finally {ctx.switchInnerTab=realSwitch;}
  });
  step('Index: oracle results, complete grid, and URL round trip', () => {
    const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, 'reference.json'), 'utf8')).index;
    for (const c of JSON.parse(fs.readFileSync(path.join(__dirname, 'cases.json'), 'utf8')).index) {
      ctx.setIdx(TAB, c.a, c.index);
      const text = els.get(`${TAB}-idx-result`).textContent;
      if (!text.includes(`Returned layout = ${fixture[c.id].layout.str}`) ||
          !text.includes(`Base offset = ${fixture[c.id].offset}`))
        throw new Error(`Index ${c.id} did not display the CuTeDSL result`);
      const svg = els.get(`${TAB}-idx-dst-svg`).innerHTML;
      if (!svg.includes('<svg')) throw new Error(`Index ${c.id} did not render`);
      if ((svg.match(/<rect\b/g) || []).length !== fixture[c.id].layout.size + 1)
        throw new Error(`Index ${c.id} did not draw every selected element`);
    }
    const realSwitch = ctx.switchInnerTab, realExport = ctx.exportURL;
    try {
      ctx.switchInnerTab = () => {};
      let exported;
      ctx.exportURL = (btn, feature, ...inputs) => { exported = [feature, ...inputs]; };
      ctx.setIdx(TAB, '(4,8):(8,1)', '(2,_)');
      ctx.exportIdx(TAB);
      ctx.setIdx(TAB, '8:1', '_');
      ctx.location.search = `?key=${encodeURIComponent(exported.join('-'))}`;
      ctx.applyKeyParam(TAB);
      if (els.get(`${TAB}-idx-layout-input`).value !== '(4,8):(8,1)' ||
          els.get(`${TAB}-idx-coord-input`).value !== '(2,_)') throw new Error('Index URL lost inputs');
    } finally {
      ctx.switchInnerTab = realSwitch;
      ctx.exportURL = realExport;
      ctx.location.search = '';
    }
  }, [`${TAB}-idx-dst-svg`]);
  step('Index rank warning follows the returned layout only', () => {
    const input = '((8,1),4,2,4):((1,0),1024,16,4096)';
    ctx.setIdx(TAB, input, '(_,_,_,0)');
    const warning = els.get(`${TAB}-idx-warning`);
    if (!warning.textContent.includes('Returned layout'))
      throw new Error('Rank-3 returned layout did not warn');
    ctx.setIdx(TAB, input, '(0,_,_,0)');
    if (warning.textContent) throw new Error('Rank-2 returned layout retained a rank warning');
    if (!els.get(`${TAB}-idx-result`).textContent.includes('Returned layout = (4,2):(1024,16)'))
      throw new Error('Rank-2 returned layout was not displayed');
  }, [`${TAB}-idx-dst-svg`]);
  step('Index rejects out-of-bounds coordinates and clears stale visualization', () => {
    ctx.setIdx(TAB, '(4,8):(1,4)', '(4,_)');
    if (els.get(`${TAB}-idx-dst-svg`).innerHTML) throw new Error('Stale Index result remained');
  }, [], {expectError: /whole number/});

  // ── URL round-trip: every ?key= form must parse ──────────────────────────
  for (const key of [
    'layout-(10,10):(1,10)',
    'index-((2,2,2),4,2,4):((1,256,8),1024,16,4096)-(_,_,_,0)',
    'make_copy_atom-universal-128-half_t',
    'make_copy_atom-ldmatrix-128-half_t-4-1',
    'make_copy_atom-g2r-128-half_t',
    'make_mma_atom-f16bf16-half_t-float-16',
    'make_mma_atom-tf32-na-na-8',
    'make_mma_atom-wg_f16bf16-half_t-float-16-64-SMEM-MN-K-half_t',
    'make_mma_atom-wg_fp8-float_e4m3_t-float-32-24-RMEM-K-K-float_e5m2_t',
    'make_mma_atom-wg_int8-int8_t-int32_t-32-256-SMEM-K-K-uint8_t',
    'make_tiled_mma-f16bf16-half_t-float-16-(2, 2, 1)-(32, 32, 16)',
    'make_tiled_mma-tf32-na-na-8-(2, 2, 1)-na',
    'local_tile-(16, 16):(16, 1)-(4, 4)-(1, 2)',
    'local_tile-(32, 64):(64, 1)-(8, 16, 4)-(1, 2, _)-(1, X, 1)',
    'partition_sd-S-universal-32-half_t-((8,4),(2,2)):((16,2),(8,1))-(8, 16)-5-(16, 32):(1, 16)',
    'partition_sd-D-cpasync-128-half_t-((8,16),8):((128,1),16)-(16, 64)-3-(16, 64):(64, 1)',
    'partition_abc-A-f16bf16-half_t-float-16-(2, 2, 1)-na-5-(64, 48):(48, 1)',
    'partition_abc-C-tf32-na-na-8-(2, 2, 1)-(32, 16, 8)-3-(64, 32):(32, 1)',
    'make_fragment_abc-A-f16bf16-half_t-float-16-((2,2,2),2,3):((1,384,8),1536,16)',
    'make_fragment_abc-B-f16bf16-bfloat16_t-float-16-((2,2),2,3):((1,8),768,16)-7',
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
