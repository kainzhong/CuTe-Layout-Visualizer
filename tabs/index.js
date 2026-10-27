// Index a tensor view: preserve the exact sliced layout and its pointer offset.
function generateIndexTabContent(id) {
  return `<div id="${id}-tab-index" class="panel">
    <div class="controls">
      <h2>Index</h2>
      ${layoutInputField({id: `${id}-idx-layout-input`, label: 'Layout', value: '((2,2,2),4,2,4):((1,256,8),1024,16,4096)'})}
      ${layoutInputField({id: `${id}-idx-coord-input`, label: 'Index', value: '(_, _, _, 0)'})}
      ${statusDivs(`${id}-idx`)}
      <div id="${id}-idx-result" class="comp-result-box" style="white-space:pre-wrap"></div>
      <button class="btn btn-render" onclick="renderIndex('${id}')">Render</button>
      <button class="btn btn-render" id="${id}-idx-export" onclick="exportIdx('${id}')">Export URL</button>
      <div class="presets"><h3>Presets</h3><div class="preset-list">
        <button class="preset-btn" onclick="setIdx('${id}','((2,2,2),4,2,4):((1,256,8),1024,16,4096)','(_, _, _, 0)')">Pipeline stage 0</button>
        <button class="preset-btn" onclick="setIdx('${id}','((2,2,2),4,2,4):((1,256,8),1024,16,4096)','(_, _, _, 2)')">Pipeline stage 2</button>
        <button class="preset-btn" onclick="setIdx('${id}','((2,3),4):((1,8),2)','((_,1),_)')">Index inside a nested mode</button>
        <button class="preset-btn" onclick="setIdx('${id}','(4,8):(8,1)','(2,_)')">One row</button>
        <button class="preset-btn" onclick="setIdx('${id}','(4,8):(1,4)','(2,3)')">One element</button>
      </div></div>
      <div class="hint"><code>_</code> or <code>None</code> keeps a mode; a zero-based integer selects it.
      Match the layout's mode structure, e.g. <code>((_,1),_)</code>.
      The returned layout contains relative offsets; the base offset is added to obtain offsets in the original tensor.
      Indexing every mode returns one element (an empty layout).</div>
    </div>
    <div class="comp-results" style="grid-template-columns:1fr">
      ${layoutVisualization({hostId:`${id}-idx-dst-svg`,modeGroupId:`${id}-idx-mode-btns`,
        title:'Indexed view — original tensor offsets',onMode:mode=>`toggleIndexMode('${id}','${mode}')`})}
    </div>
  </div>`;
}

const idxState = {};

// DOM-free entry point shared by rendering and differential tests.
function idxComputeIndex(layoutText, indexText) {
  const input = parseExactLayoutInput(layoutText);
  const shape = input.shape;
  let coord;
  try {
    coord = JSON.parse(String(indexText).trim().replace(/\b(?:None|none|_)\b/g, 'null')
      .replace(/\(/g, '[').replace(/\)/g, ']').replace(/,\s*]/g, ']'));
  } catch (_) {
    throw new Error('Invalid index. Use integers and _ / None, for example (_, _, _, 0).');
  }
  function validate(c, s) {
    if (c === null) return;
    if (Array.isArray(c)) {
      if (!Array.isArray(s) || c.length !== s.length) throw new Error('Index tuple must match the layout mode structure.');
      c.forEach((v, i) => validate(v, s[i]));
    } else if (!Number.isSafeInteger(c) || c < 0 || c >= product(s)) {
      throw new Error(`Index must be a whole number in [0, ${product(s)}) or _.`);
    }
  }
  // A one-mode tuple is useful for conventional tensor indexing of rank-1 layouts.
  if (!Array.isArray(shape) && Array.isArray(coord) && coord.length === 1) coord = coord[0];
  validate(coord, shape);
  const [layout, offset] = coord === null ? [input, 0] : slice_and_offset(coord, input);
  return {input, coord, layout, offset};
}

function renderIndex(tabId) {
  const prefix = `${tabId}-idx`;
  const state=idxState[tabId] ||= {mode:new Set(['value'])};
  const result = document.getElementById(`${prefix}-result`);
  const host = document.getElementById(`${prefix}-dst-svg`);
  showErr(`${prefix}-error`, '');
  try {
    const text = document.getElementById(`${prefix}-layout-input`).value;
    const index = document.getElementById(`${prefix}-coord-input`).value;
    const {layout, offset} = idxComputeIndex(text, index);
    const layoutText = formatLayoutStr(layout.shape, layout.stride);
    // Only the returned view is drawn; input rank does not affect this grid.
    updateRankWarning(`${prefix}-warning`, [['Returned layout', layoutText]]);
    result.textContent = `Returned layout = ${layoutText}\nBase offset = ${offset}\nElements = ${layout.size()}`;
    result.classList.add('visible');
    // Normalize only for display; keep all remaining outer modes in the columns.
    const shape = layout.shape, stride = layout.stride;
    const display = layout.rank() > 1
      ? {shape: [shape[0], shape.slice(1)], stride: [stride[0], stride.slice(1)]}
      : layout.rank() === 1
        ? {shape: [Array.isArray(shape) ? shape[0] : shape, 1],
           stride: [Array.isArray(stride) ? stride[0] : stride, 0]}
        : {shape: [1, 1], stride: [0, 0]};
    renderLayoutVisualization({hostId:`${prefix}-dst-svg`,modeGroupId:`${prefix}-mode-btns`,
      shape:display.shape,stride:display.stride,modes:state.mode,offset,colorFn:colorHighlight});
    updateOuterTabLabel(tabId, `Index:${index.trim()}`);
  } catch (e) {
    showErr(`${prefix}-error`, e.message);
    result.classList.remove('visible');
    host.innerHTML = '';
  }
}

function toggleIndexMode(tabId,mode) {
  const state=idxState[tabId] ||= {mode:new Set(['value'])};
  toggleLayoutVisualizationMode(state.mode,mode);
  renderIndex(tabId);
}

function setIdx(tabId, layout, coord) {
  document.getElementById(`${tabId}-idx-layout-input`).value = layout;
  document.getElementById(`${tabId}-idx-coord-input`).value = coord;
  renderIndex(tabId);
}

function exportIdx(tabId) {
  exportURL(`${tabId}-idx-export`, 'index', document.getElementById(`${tabId}-idx-layout-input`).value,
    document.getElementById(`${tabId}-idx-coord-input`).value);
}
