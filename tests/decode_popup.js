// Decode from Pasteboard failure popup: every decode button in the real tab
// templates is clicked with a rejected paste and must report the reason inline,
// pop up a dialog naming the tab whose Export URL it accepts, and leave the
// controls untouched. Uses dom_smoke's context so templates and handlers are real.
'use strict';
const { makeContext } = require('./dom_smoke');

const TAB = 'dp';

// Button id suffix -> source features whose Export URL that decoder accepts.
// A new Decode from Pasteboard button must be listed here.
const TILED_COPY = ['make_tiled_copy', 'make_tiled_copy_tv', 'make_tiled_copy_ab'];
const EXPECTED = {
  'mca-decode-atom': ['make_copy_atom'],
  'mtc-decode-atom': ['make_copy_atom'],
  'mtv-decode-atom': ['make_copy_atom'],
  'tmc-decode-atom': ['make_copy_atom'],
  'mtv-decode-tv': ['tv'],
  'mcab-decode-mma': ['make_tiled_mma'],
  'mcab-decode-copy': ['make_copy_atom'],
  'rt-decode-copy': TILED_COPY,
  'rt-decode-input': ['make_fragment_abc'],
  'psd-decode-copy': TILED_COPY,
  'mma-decode-atom': ['make_mma_atom'],
  'mtm-decode-atom': ['make_mma_atom'],
  'mtm-decode': ['make_tiled_mma'],
  'pabc-decode-mma': ['make_tiled_mma'],
  'pshp-decode-mma': ['make_tiled_mma'],
  'mfrag-decode-atom': ['make_mma_atom'],
  'gemm-decode': ['make_tiled_mma'],
};

/** Element stub that keeps children, text and listeners so the popup is inspectable. */
function node(tag) {
  const listeners = {};
  return {
    tag, id: '', className: '', textContent: '', type: '', attrs: {}, children: [], parent: null,
    style: {}, dataset: {},
    setAttribute(k, v) { this.attrs[k] = v; }, getAttribute(k) { return this.attrs[k]; },
    appendChild(c) { c.parent = this; this.children.push(c); return c; },
    remove() { if (this.parent) this.parent.children = this.parent.children.filter(c => c !== this); this.parent = null; },
    addEventListener(t, fn) { (listeners[t] ||= []).push(fn); },
    fire(t, e = {}) { for (const fn of listeners[t] || []) fn({ target: this, ...e }); },
    focus() {},
    all() { return [this, ...this.children.flatMap(c => c.all())]; },
    text() { return this.all().map(n => n.textContent).join('\n'); },
  };
}

async function runDecodePopupTests() {
  const results = { pass: 0, fail: 0, failures: [] };
  const check = (label, ok, why = '') => {
    if (ok) results.pass++;
    else { results.fail++; results.failures.push(why ? `${label}: ${why}` : label); }
  };
  const { ctx, els } = makeContext();
  const html = ctx.generateTabContent(TAB);
  const body = node('body');
  const docListeners = {};
  ctx.document.body = body;
  ctx.document.createElement = node;
  ctx.document.addEventListener = (t, fn) => { (docListeners[t] ||= new Set()).add(fn); };
  ctx.document.removeEventListener = (t, fn) => { docListeners[t] && docListeners[t].delete(fn); };
  let clipboard = '';
  ctx.navigator.clipboard = { async readText() { return clipboard; } };
  const popup = () => body.children.find(c => c.id === 'decode-error-popup');
  const switched = [];
  ctx.switchInnerTab = (tabId, mode) => switched.push([tabId, mode]);

  const buttons = [...html.matchAll(/<button\b[^>]*>Decode from Pasteboard<\/button>/g)].map(m => ({
    id: (m[0].match(/\bid="([^"]+)"/) || [])[1],
    onclick: m[0].match(/\bonclick="([^"]+)"/)[1],
  }));
  check('decode buttons found', buttons.length >= Object.keys(EXPECTED).length);
  const covered = new Set();

  for (const btn of buttons) {
    const suffix = btn.id && btn.id.slice(TAB.length + 1);
    const sources = EXPECTED[suffix];
    if (!sources) { check(`decode button ${btn.id || btn.onclick} has an expected source tab`, false); continue; }
    covered.add(suffix);
    for (const pasted of ['not a URL', '?key=layout-(4,4):(1,4)', '']) {
      const label = `${suffix} rejects ${JSON.stringify(pasted)}`;
      ctx.closeDecodeFailurePopup();
      for (const el of els.values()) if (/-error$/.test(el.id)) el.textContent = '';
      const before = [...els.entries()].filter(([id]) => /-input$/.test(id)).map(([id, el]) => [id, el.value]);
      clipboard = pasted;
      try {
        const ok = await ctx.eval(btn.onclick);
        check(`${label}: handler reports failure`, ok === false, `returned ${ok}`);
      } catch (e) { check(label, false, `threw ${e.message}`); continue; }
      const p = popup();
      if (!p) { check(`${label}: popup shown`, false); continue; }
      const text = p.text();
      const expected = ctx.decodeFailureMessage(sources, '');
      check(`${label}: popup says invalid`, text.includes('Invalid pasteboard input'));
      check(`${label}: popup names source tab(s)`, text.includes(expected.hint), text);
      const openButtons = p.all().filter(n => n.tag === 'button' && /^Open /.test(n.textContent));
      check(`${label}: one Open button per source`, openButtons.length === sources.length);
      const inline = [...els.values()].filter(el => /-error$/.test(el.id) && el.textContent);
      check(`${label}: inline error shown`, inline.length === 1 && text.includes(inline[0].textContent),
        inline.map(el => `${el.id}=${el.textContent}`).join('; '));
      const changed = before.filter(([id, v]) => els.get(id).value !== v).map(([id]) => id);
      check(`${label}: controls preserved`, !changed.length, changed.join(', '));
    }
  }
  for (const suffix of Object.keys(EXPECTED))
    check(`decode button ${suffix} exists in templates`, covered.has(suffix));

  // Open buttons jump to the source tab and close the popup.
  ctx.showDecodeFailurePopup(TAB, TILED_COPY, 'bad');
  const opens = popup().all().filter(n => n.tag === 'button' && /^Open /.test(n.textContent));
  opens[2].fire('click');
  check('Open button switches to the source tab',
    JSON.stringify(switched.at(-1)) === JSON.stringify([TAB, 'make_tiled_copy_ab']), JSON.stringify(switched));
  check('Open button closes the popup', !popup());

  // Escape, Close and backdrop dismiss; only one popup at a time.
  ctx.showDecodeFailurePopup(TAB, ['tv'], 'bad');
  ctx.showDecodeFailurePopup(TAB, ['tv'], 'worse');
  check('repeated failure replaces the popup', body.children.filter(c => c.id === 'decode-error-popup').length === 1);
  for (const fn of [...(docListeners.keydown || [])]) fn({ key: 'Escape' });
  check('Escape closes the popup', !popup() && !(docListeners.keydown || new Set()).size);
  ctx.showDecodeFailurePopup(TAB, ['tv'], 'bad');
  popup().all().find(n => n.tag === 'button' && n.textContent === 'Close').fire('click');
  check('Close button closes the popup', !popup());
  ctx.showDecodeFailurePopup(TAB, ['tv'], 'bad');
  popup().fire('click');
  check('backdrop click closes the popup', !popup());

  // Message text and the non-failure paths.
  const msg = ctx.decodeFailureMessage(TILED_COPY, 'x');
  check('multi-source hint lists every tab',
    msg.hint === 'Open COPY → make_tiled_copy, COPY → make_tiled_copy_tv or COPY → make_tiled_copy_A / B / C, click Export URL, then Decode from Pasteboard again.', msg.hint);
  check('single-source hint', ctx.decodeFailureMessage(['tv'], 'x').hint.startsWith('Open BASICS → TV Layout, click Export URL'));
  let threw = false;
  try { ctx.decodeFailureMessage(['nope'], 'x'); } catch (_) { threw = true; }
  check('unknown source is a programming error', threw);

  clipboard = '?key=anything';
  let imported = null;
  const ok = await ctx.decodeFromPasteboard({ tabId: TAB, errorId: `${TAB}-mtm-error`, sources: ['make_tiled_mma'],
    message: '', importFn: t => { imported = t; } });
  check('successful decode shows no popup', ok === true && imported === clipboard && !popup());

  ctx.navigator.clipboard = { async readText() { throw new Error('denied'); } };
  ctx.prompt = () => null;
  const cancelled = await ctx.decodeFromPasteboard({ tabId: TAB, errorId: `${TAB}-mtm-error`, sources: ['make_tiled_mma'],
    message: '', importFn: () => { throw new Error('should not run'); } });
  check('cancelled paste prompt shows no popup', cancelled === false && !popup());

  return results;
}

module.exports = { runDecodePopupTests };

if (require.main === module) {
  runDecodePopupTests().then(r => {
    for (const f of r.failures) console.log(`  FAIL  ${f}`);
    console.log(`\n${r.fail ? 'FAIL' : 'PASS'} ${r.pass}/${r.pass + r.fail} decode-popup checks`);
    process.exit(r.fail ? 1 : 0);
  });
}
