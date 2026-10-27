// Persist editable browser tabs, never generated SVG/HTML or computed layouts.
const TAB_SESSION_STORAGE_KEY = 'cute-layout-visualizer.tabs.v1';
const tabSessionState = {initialized: false, restoring: false, timer: null, launchKey: null};
const TAB_SESSION_PREFERENCES = new Set([
  'which','side','copyURL','mmaInputs','inputs','altView','showValue','mode',
  'bankCheck','coalCheck','bankToHighlight','waveVid','swizzle',
  'tileMode','boxMode','sweepMode','extraMode',
]);

function tabSessionStores() {
  return {index: idxState, swizzle: swState, make_smem_layout_a: smlState.a, make_smem_layout_b: smlState.b,
    make_smem_layout_epi: smlState.epi,
    tv: tvState, make_mma_atom: mmaState, make_tiled_mma: mtmState,
    make_copy_atom: mcaState, make_tiled_copy: mtcState, make_tiled_copy_tv: mtvState,
    make_tiled_copy_ab: mcabState, make_tiled_tma_atom: tmaState, tma_partition: tpState,
    partition_sd: psdState, partition_abc: pabcState, make_fragment_abc: mfragState, gemm: gemmState};
}

function tabSessionEncode(value) {
  if (value instanceof Set) return {set: [...value]};
  if (Array.isArray(value)) return value.map(tabSessionEncode);
  if (value && typeof value === 'object') return Object.fromEntries(
    Object.entries(value).map(([key,v]) => [key,tabSessionEncode(v)]));
  return value;
}

function tabSessionPreference(key, value) {
  const labels = new Set(['value','index','coord']);
  if (['tileMode','boxMode','sweepMode','extraMode'].includes(key) || (key === 'mode' && value && typeof value === 'object'))
    return Array.isArray(value?.set) && value.set.every(label => labels.has(label)) ? new Set(value.set) : undefined;
  if (key === 'mode') return ['','value','index','coord','warp','warpgroup','tv'].includes(value) ? value : undefined;
  if (key === 'which') return ['A','B','C'].includes(value) ? value : undefined;
  if (key === 'side') return ['S','D'].includes(value) ? value : undefined;
  if (['altView','showValue','bankCheck','coalCheck'].includes(key)) return typeof value === 'boolean' ? value : undefined;
  if (['bankToHighlight','waveVid'].includes(key)) return value === null || (Number.isSafeInteger(value) && value >= 0) ? value : undefined;
  if (key === 'copyURL') return typeof value === 'string' ? value : undefined;
  if (key === 'mmaInputs') return Array.isArray(value) && [6,11].includes(value.length) && value.every(v => typeof v === 'string') ? value : undefined;
  if (key === 'inputs' && value && typeof value === 'object') return Object.fromEntries(Object.entries(value).filter(([k,v]) => ['A','B','C'].includes(k) && typeof v === 'string'));
  if (key === 'swizzle') return value === null ? null : value && ['B','M','S'].every(k => Number.isSafeInteger(value[k]) && value[k] >= 0)
    ? parseSwizzleSpec(`${value.B},${value.M},${value.S}`) || undefined : undefined;
}

function captureTabSession() {
  const stores = tabSessionStores();
  const ids = [...document.querySelectorAll('#outer-tab-buttons .outer-tab')].map(el => el.getAttribute('data-tab-id'));
  return {version: 1, active: Math.max(0,ids.indexOf(activeOuterTabId)), launchKey: tabSessionState.launchKey,
    tabs: ids.map(id => {
      const panel = document.getElementById(`${id}-panel`);
      const suffix = el => el.id.slice(id.length+1);
      const preferences = {};
      for (const [feature,store] of Object.entries(stores)) {
        const state = store[id];
        if (state) preferences[feature] = Object.fromEntries(Object.entries(state)
          .filter(([key]) => TAB_SESSION_PREFERENCES.has(key)).map(([key,value]) => [key,tabSessionEncode(value)]));
      }
      return {
        name: document.getElementById(`${id}-label`).dataset.customName || null,
        feature: panel.querySelector('.tab.active')?.getAttribute('data-tab') || 'layout',
        controls: [...panel.querySelectorAll('input[id],select[id],textarea[id]')]
          .filter(el => !['file','password'].includes(el.type))
          .map(el => ({id: suffix(el), value: el.value, checked: !!el.checked})),
        details: [...panel.querySelectorAll('details')].map((el,index) => ({id: el.id ? suffix(el) : null,index,open: el.open})),
        preferences,
      };
    })};
}

function saveTabSession() {
  if (!tabSessionState.initialized || tabSessionState.restoring) return;
  clearTimeout(tabSessionState.timer);
  tabSessionState.timer = null;
  try { localStorage.setItem(TAB_SESSION_STORAGE_KEY,JSON.stringify(captureTabSession())); }
  catch (_) { /* Storage may be disabled/full; the current tabs remain usable. */ }
}

function scheduleTabSessionSave() {
  if (!tabSessionState.initialized || tabSessionState.restoring) return;
  clearTimeout(tabSessionState.timer);
  tabSessionState.timer = setTimeout(saveTabSession,150);
}

function restoreTabSessionTab(saved) {
  const id = addOuterTab();
  const panel = document.getElementById(`${id}-panel`);
  const controls = Array.isArray(saved.controls) ? saved.controls.filter(field => field && typeof field.id === 'string') : [];
  const feature = Object.hasOwn(TAB_RENDER_FN,saved.feature) ? saved.feature : 'layout';
  const stores = tabSessionStores();
  const preferences = saved.preferences || {};
  const applyPreferences = () => {
    for (const [name,store] of Object.entries(stores)) {
      const prefs = preferences[name];
      if (!prefs || typeof prefs !== 'object') continue;
      const state = store[id] || (store[id] = {});
      for (const [key,value] of Object.entries(prefs)) if (TAB_SESSION_PREFERENCES.has(key)) {
        const valid = tabSessionPreference(key,value);
        if (valid !== undefined) state[key] = valid;
      }
    }
  };
  const element = field => typeof field.id === 'string' ? document.getElementById(`${id}-${field.id}`) : null;
  const applyText = () => {
    for (const field of controls) {
      const el = element(field);
      if (!el || el.tagName === 'SELECT' || !['INPUT','TEXTAREA'].includes(el.tagName) || ['file','password'].includes(el.type)) continue;
      if (typeof field.value === 'string') el.value = field.value;
      if (typeof field.checked === 'boolean') el.checked = field.checked;
    }
  };
  applyPreferences();
  applyText();
  // Parent selects precede their dependent fields in every constructor. Drive
  // their real change handlers so supported options are rebuilt before assigning
  // dtype/K/source/major values. Never restore stored option HTML or create options.
  for (const field of controls) {
    const el = element(field);
    if (!el || el.tagName !== 'SELECT' || typeof field.value !== 'string') continue;
    if (![...el.options].some(option => option.value === field.value)) continue;
    el.value = field.value;
    el.dispatchEvent(new Event('change',{bubbles: true}));
  }
  applyText(); // Constructor handlers may derive text; retain the user's edits.
  applyPreferences();
  switchInnerTab(id,feature);
  renderAllTabs(id,feature);
  applyText(); // Invalid or unfinished edits survive too.
  if (typeof saved.name === 'string') setOuterTabName(id,saved.name);
  const details = [...panel.querySelectorAll('details')];
  for (const item of Array.isArray(saved.details) ? saved.details : []) {
    if (!item || typeof item !== 'object') continue;
    const el = item.id ? document.getElementById(`${id}-${item.id}`) : details[item.index];
    if (el?.tagName === 'DETAILS' && typeof item.open === 'boolean') el.open = item.open;
  }
  return id;
}

function initializeTabSession() {
  if (tabSessionState.initialized) return activeOuterTabId;
  let saved = null;
  try {
    const value = JSON.parse(localStorage.getItem(TAB_SESSION_STORAGE_KEY) || 'null');
    if (value?.version === 1 && Array.isArray(value.tabs) && value.tabs.length) saved = value;
  } catch (_) { /* Missing/corrupt/blocked storage starts with a normal tab. */ }
  const parsed = parseKeyParam();
  const launchKey = new URLSearchParams(location.search).get('key');
  tabSessionState.restoring = true;
  const restored = [];
  try {
    if (saved) for (const tab of saved.tabs) if (tab && typeof tab === 'object') restored.push(restoreTabSessionTab(tab));
    if (!restored.length) restored.push(addOuterTab());
    const navigation = window.performance?.getEntriesByType?.('navigation')?.[0]?.type;
    // Reload keeps edits to the current deep link. Explicit new navigation opens
    // the requested visualization alongside the saved workspace.
    if (parsed && (!saved || launchKey !== saved.launchKey || navigation === 'navigate')) {
      const target = saved ? addOuterTab() : restored[0];
      try { applyKeyParam(target,parsed); }
      catch (e) {
        const error = document.getElementById(`${target}-panel`).querySelector('.panel.active .error-msg');
        if (error) showErr(error.id,e.message);
      }
    } else switchOuterTab(restored[Number.isInteger(saved?.active) && saved.active >= 0 && saved.active < restored.length ? saved.active : 0]);
  } finally {
    tabSessionState.launchKey = launchKey;
    tabSessionState.restoring = false;
    tabSessionState.initialized = true;
  }
  for (const type of ['input','change','click','toggle']) document.addEventListener(type,scheduleTabSessionSave,true);
  window.addEventListener('pagehide',saveTabSession);
  document.addEventListener('visibilitychange',() => { if (document.visibilityState === 'hidden') saveTabSession(); });
  saveTabSession();
  return activeOuterTabId;
}
