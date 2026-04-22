// Swizzle tab — visualize a CuTe Swizzle<B, M, S> as a before/after pair of
// grids over a base layout. Top = base layout (cells show logical offset `a`).
// Bottom = swizzled layout (cells show `a → b` where b = Swizzle<B,M,S>.apply(a)).
// Functions become globals on `window` (no module system).

function generateSwizzleTabContent(id) {
  return `
    <!-- Swizzle panel -->
    <div id="${id}-tab-swizzle" class="panel">
      <div class="controls">
        <h2>Swizzle</h2>
        ${layoutInputField({
          id: `${id}-sw-layout-input`,
          label: 'Layout &mdash; the base shape/stride (offsets = layout(m, n))',
          value: '(8, 8):(8, 1)'
        })}
        ${layoutInputField({
          id: `${id}-sw-swizzle-input`,
          label: 'Swizzle &mdash; <code>B, M, S</code> (e.g. <code>3, 0, 3</code>)',
          hint: 'matches CuTe\'s <code>Swizzle&lt;B, M, S&gt;</code>',
          value: '3, 0, 3'
        })}
        ${statusDivs(`${id}-sw`)}
        <div id="${id}-sw-result" class="comp-result-box"></div>
        <button class="btn btn-render" onclick="renderSwizzle('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-sw-export" onclick="exportSW('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setSW('${id}','(8, 8):(8, 1)','3, 0, 3')">8x8 row-maj + Swizzle&lt;3,0,3&gt;</button>
            <button class="preset-btn" onclick="setSW('${id}','(16, 16):(16, 1)','3, 0, 3')">16x16 row-maj + Swizzle&lt;3,0,3&gt;</button>
            <button class="preset-btn" onclick="setSW('${id}','(8, 32):(32, 1)','3, 3, 3')">8x32 row-maj + Swizzle&lt;3,3,3&gt; (typical SMEM matmul)</button>
            <button class="preset-btn" onclick="setSW('${id}','(16, 16):(1, 16)','2, 2, 2')">16x16 col-maj + Swizzle&lt;2,2,2&gt;</button>
            <button class="preset-btn" onclick="setSW('${id}','(8, 8):(8, 1)','0, 0, 0')">8x8 row-maj + Swizzle&lt;0,0,0&gt; (identity — no-op)</button>
          </div>
        </div>

        <div class="hint">
          A <code>Swizzle&lt;B, M, S&gt;</code> is a pure integer-offset permutation:
          it XORs the <code>B</code> bits at position <code>M + S</code> into the
          <code>B</code> bits at position <code>M</code>. The top grid shows the
          base layout (cell <code>(m, n)</code> = logical offset <code>a</code>);
          the bottom grid shows the same coord's swizzled offset as
          <code>a &rarr; b</code> where <code>b = a &oplus; (((a &gt;&gt; (M+S)) &amp; ((1&lt;&lt;B)-1)) &lt;&lt; M)</code>.<br><br>
          Bottom-cell colour is keyed to the <em>swizzled</em> offset, so same
          colour = same post-swizzle address bucket — makes conflict-avoidance
          patterns visible at a glance.<br><br>
          Swizzle is a bijection, so every colour appears exactly as many times
          below as in the top grid (just shuffled).
        </div>
      </div>

      <div class="comp-results" style="grid-template-columns:1fr">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-sw-base-title">Base layout</span>
            <button class="mode-btn" id="${id}-sw-base-svg-zoom" onclick="toggleZoom('${id}-sw-base-svg')">Zoom in</button>
          </div>
          <div class="viz-box"><div id="${id}-sw-base-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-sw-swizzled-title">Swizzled layout</span>
            <button class="mode-btn" id="${id}-sw-swizzled-svg-zoom" onclick="toggleZoom('${id}-sw-swizzled-svg')">Zoom in</button>
          </div>
          <div class="viz-box"><div id="${id}-sw-swizzled-svg"></div></div>
        </div>
      </div>
    </div>`;
}

const swState = {};

// CuTe's Swizzle<B,M,S>::apply: XOR B bits at position M+S into B bits at M.
function swApply(x, sw) {
  const { B, M, S } = sw;
  const mask = (1 << B) - 1;
  const srcBits = (x >>> (M + S)) & mask;
  return x ^ (srcBits << M);
}

// Parse a swizzle spec. Accepts "B,M,S" / "B M S" / "Swizzle<B,M,S>".
function swParse(str) {
  const s = (str || '').trim();
  if (!s) return null;
  const m = s.match(/^\s*(?:Swizzle\s*<\s*)?(\d+)\s*[,\s]\s*(\d+)\s*[,\s]\s*(\d+)\s*>?\s*$/i);
  if (!m) return null;
  const B = parseInt(m[1], 10);
  const M = parseInt(m[2], 10);
  const S = parseInt(m[3], 10);
  if (B < 0 || M < 0 || S < 0 || (B + M + S) >= 31) return null;
  return { B, M, S };
}

function renderSwizzle(tabId) {
  showErr(`${tabId}-sw-error`, '');
  try {
    const layoutStr  = document.getElementById(`${tabId}-sw-layout-input`).value;
    const swizzleStr = document.getElementById(`${tabId}-sw-swizzle-input`).value;

    updateRankWarning(`${tabId}-sw-warning`, [['Layout', layoutStr]]);

    const sw = swParse(swizzleStr);
    if (!sw) {
      throw new Error('Invalid swizzle. Use B, M, S (e.g. "3, 0, 3") ' +
        'with B ≥ 0, M ≥ 0, S ≥ 0, and B + M + S < 31.');
    }

    const parsed = parseLayout(layoutStr);
    swState[tabId] = { parsed, sw, layoutStr };

    const resultEl = document.getElementById(`${tabId}-sw-result`);
    resultEl.textContent = `Swizzle<${sw.B}, ${sw.M}, ${sw.S}> applied to ${layoutStr.trim()}`;
    resultEl.classList.add('visible');

    // Top: plain base layout. Cell colour = colorBW(logical offset).
    document.getElementById(`${tabId}-sw-base-svg`).innerHTML =
      buildLayoutSVG(parsed.shape, parsed.stride, 'value');
    applyZoomState(`${tabId}-sw-base-svg`);
    document.getElementById(`${tabId}-sw-base-title`).textContent =
      `Base layout: ${layoutStr.trim()}`;

    // Bottom: same grid, cell text "a → b". Colour keyed by b so cells that
    // land at the same swizzled offset mod-8 share a hue — makes the permutation
    // structure pop without being the same as the top grid.
    document.getElementById(`${tabId}-sw-swizzled-svg`).innerHTML =
      buildColoredLayoutSVG(parsed.shape, parsed.stride, 'value', (m, n, offset) => {
        const b = swApply(offset, sw);
        return {
          bg: colorBW(b),
          text: [`${offset} → ${b}`],
        };
      });
    applyZoomState(`${tabId}-sw-swizzled-svg`);
    document.getElementById(`${tabId}-sw-swizzled-title`).textContent =
      `Swizzled: Swizzle<${sw.B}, ${sw.M}, ${sw.S}>  (cells show  a → b = swizzle(a))`;

    updateOuterTabLabel(tabId, `Swizzle:${layoutStr.trim()}`);
  } catch (e) {
    showErr(`${tabId}-sw-error`, e.message);
    const resultEl = document.getElementById(`${tabId}-sw-result`);
    if (resultEl) resultEl.classList.remove('visible');
    ['base', 'swizzled'].forEach(w => {
      const el = document.getElementById(`${tabId}-sw-${w}-svg`);
      if (el) el.innerHTML = '';
    });
  }
}

function setSW(tabId, layout, swizzle) {
  document.getElementById(`${tabId}-sw-layout-input`).value = layout;
  document.getElementById(`${tabId}-sw-swizzle-input`).value = swizzle;
  renderSwizzle(tabId);
}

function exportSW(tabId) {
  const layout  = document.getElementById(`${tabId}-sw-layout-input`).value;
  const swizzle = document.getElementById(`${tabId}-sw-swizzle-input`).value;
  exportURL(`${tabId}-sw-export`, 'swizzle', layout, swizzle);
}
