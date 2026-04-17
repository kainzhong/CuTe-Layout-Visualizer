// UI: SVG generation, DOM manipulation, tab management
// Depends on all functions from cute.js (loaded first).

// ═══════════════════════════════════════════════════════
//  SVG generation
// ═══════════════════════════════════════════════════════

const MAX_CELLS = 131072;
const BASE_CS = 56;

function cellSize(M, N) {
  return Math.max(12, Math.min(BASE_CS, Math.floor(1200 / Math.max(M, N, 1))));
}

/** Pick CSS to fit the SVG by longest side (default) or shortest side (zoomed). */
function svgFitStyle(W, H, zoomed) {
  if (zoomed) {
    // Fit by shortest side — image may overflow, container scrolls
    return W >= H
      ? 'height:70vh;width:auto'
      : 'width:100%;height:auto';
  }
  // Fit by longest side — always fully visible
  return W >= H
    ? 'width:100%;height:auto'
    : 'max-height:70vh;width:auto;max-width:100%';
}

/** After rendering SVG into a host, reapply zoom if host is zoomed. */
function applyZoomState(hostId) {
  const host = document.getElementById(hostId);
  if (!host || host.dataset.zoom !== 'true') return;
  const svg = host.querySelector('svg');
  if (svg) {
    const vb = svg.getAttribute('viewBox').split(' ').map(Number);
    svg.setAttribute('style', svgFitStyle(vb[2], vb[3], true));
  }
}

/** Toggle zoom on a viz host and update the SVG + button. */
function toggleZoom(hostId) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const zoomed = host.dataset.zoom !== 'true';
  host.dataset.zoom = zoomed;
  const svg = host.querySelector('svg');
  if (svg) {
    const vb = svg.getAttribute('viewBox').split(' ').map(Number);
    svg.setAttribute('style', svgFitStyle(vb[2], vb[3], zoomed));
  }
  const btn = document.getElementById(hostId + '-zoom');
  if (btn) btn.textContent = zoomed ? 'Zoom out' : 'Zoom in';
}

/** Build cell label lines from a modes Set. Returns an array of strings. */
function buildCellLines(modes, offset, index, coord) {
  const lines = [];
  if (modes.size === 1) {
    if (modes.has('value')) lines.push(String(offset));
    else if (modes.has('index')) lines.push(String(index));
    else if (modes.has('coord')) lines.push(coord);
  } else {
    if (modes.has('value')) lines.push(`val=${offset}`);
    if (modes.has('index'))  lines.push(`idx=${index}`);
    if (modes.has('coord'))  lines.push(coord);
  }
  return lines;
}

/** Emit SVG <text> elements for cell labels, auto-fitting to cell size.
 *  cx, cy = cell center;  cs = cell size in px;  lines = array of strings;  fg = fill color. */
function cellTextSVG(cx, cy, lines, cs, fg) {
  const maxChars = Math.max(...lines.map(l => l.length));
  // ~0.6 char-width ratio for monospace; leave small padding
  const fsByWidth = (cs * 0.9) / (maxChars * 0.6);
  // total height of N lines = (N-1)*lineH + fs = (N-1)*fs*1.25 + fs = fs*(N*1.25 - 0.25)
  const fsByHeight = (cs * 0.9) / (lines.length * 1.25 - 0.25);
  const fs = Math.min(fsByWidth, fsByHeight, 14);
  if (fs < 1) return ''; // too small to render
  const lineH = fs * 1.25;
  const startY = cy - (lines.length - 1) * lineH / 2;
  let out = '';
  for (let i = 0; i < lines.length; i++) {
    out += `<text x="${cx}" y="${startY + i * lineH}" text-anchor="middle" dominant-baseline="middle"
      fill="${fg}" font-size="${fs}" font-family="monospace">${lines[i]}</text>`;
  }
  return out;
}

/** Normalize mode: accept a Set or legacy string, always return a Set. */
function toModeSet(mode) {
  if (mode instanceof Set) return mode;
  return new Set([mode || 'value']);
}

/** Build axis-label and grid SVG for a regular layout.
 *  mode: 'value' (default) shows layout output, 'index' shows 1-D coordinate,
 *  'coord' shows (m,n). */
function buildLayoutSVG(shape, stride, mode) {
  const modes = toModeSet(mode);
  const [M, N] = productEach(shape);
  if (M * N > MAX_CELLS)
    return errSVG(`Grid too large: ${M}\u00d7${N} = ${M*N} cells (max ${MAX_CELLS})`);
  if (M === 0 || N === 0)
    return errSVG(`Empty grid: ${M}\u00d7${N}`);

  const cs = cellSize(M, N);
  const margin = cs;
  const W = margin + N * cs;
  const H = margin + M * cs;
  const axisFs = Math.max(8, Math.min(14, Math.floor(cs * 0.38)));

  let body = '';

  for (let n = 0; n < N; n++) {
    const cx = margin + (n + 0.5) * cs;
    body += `<text x="${cx}" y="${margin * 0.55}" text-anchor="middle" dominant-baseline="middle"
      fill="#555" font-size="${axisFs}" font-family="monospace">${n}</text>`;
  }
  for (let m = 0; m < M; m++) {
    const cy = margin + (m + 0.5) * cs;
    body += `<text x="${margin * 0.5}" y="${cy}" text-anchor="middle" dominant-baseline="middle"
      fill="#555" font-size="${axisFs}" font-family="monospace">${m}</text>`;
  }

  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      const idx = layoutAt(shape, stride, m, n);
      const flatI = m + n * M;
      const lines = buildCellLines(modes, idx, flatI, `(${m},${n})`);
      const bg  = colorBW(idx);
      const fg  = textOnBG(bg);
      const x = margin + n * cs;
      const y = margin + m * cs;
      body += `<rect x="${x}" y="${y}" width="${cs}" height="${cs}"
        fill="${bg}" stroke="#ccc" stroke-width="0.5"/>`;
      body += cellTextSVG(x + cs/2, y + cs/2, lines, cs, fg);
    }
  }

  return `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="${svgFitStyle(W, H)}">
    <rect width="${W}" height="${H}" fill="white"/>
    ${body}
  </svg>`;
}

/** Build axis-label and grid SVG for a TV layout. */
function buildTVSVG(tvShape, tvStride, tileShape, tileStride, showOffset, underlyingLayout, highlightTid) {
  underlyingLayout = underlyingLayout || 'col';
  // highlightTid is a number or null. When set, only cells whose entries
  // include that tid are shown in full color; all others are dimmed.
  if (highlightTid === undefined) highlightTid = null;
  if (!Array.isArray(tvShape) || tvShape.length < 2)
    throw new Error('TV layout must be rank 2: (num_threads, num_values):...');

  const numT = product(tvShape[0]);
  const numV = product(tvShape[1]);
  const M    = product(tileShape[0]);
  const N    = product(tileShape[1]);

  function scalarStride(s) {
    if (typeof s === 'number') return s;
    if (Array.isArray(s)) return scalarStride(s[0]);
    throw new Error('Cannot extract scalar stride from: ' + JSON.stringify(s));
  }
  const sm = scalarStride(tileStride[0]);
  const sn = scalarStride(tileStride[1]);

  if (M * N > MAX_CELLS)
    return errSVG(`Tile too large: ${M}\u00d7${N} = ${M*N} cells (max ${MAX_CELLS})`);
  if (M === 0 || N === 0)
    return errSVG(`Empty tile: ${M}\u00d7${N}`);
  if (numT * numV > 65536)
    return errSVG(`Too many thread\u00d7value combinations: ${numT}\u00d7${numV}`);

  const grid = Array.from({length: M}, () => Array.from({length: N}, () => []));
  for (let tid = 0; tid < numT; tid++) {
    for (let vid = 0; vid < numV; vid++) {
      const c0  = unflatten(tid, tvShape[0]);
      const c1  = unflatten(vid, tvShape[1]);
      const idx = crd2idx(c0, tvShape[0], tvStride[0]) +
                  crd2idx(c1, tvShape[1], tvStride[1]);
      const m = Math.floor(idx / sm) % M;
      const n = Math.floor(idx / sn) % N;
      if (m >= 0 && m < M && n >= 0 && n < N) {
        // Memory offset of cell (m, n) in the underlying data tensor.
        // Row-major: m*N + n. Col-major: m + n*M.
        const offset = underlyingLayout === 'row' ? (m * N + n) : (m + n * M);
        grid[m][n].push({ tid, vid, offset });
      }
    }
  }

  const cs     = cellSize(M, N);
  const margin = cs;
  const W      = margin + N * cs;
  const H      = margin + M * cs;
  const axisFs = Math.max(7, Math.min(12, Math.floor(cs * 0.26)));

  let body = '';

  for (let n = 0; n < N; n++) {
    const cx = margin + (n + 0.5) * cs;
    body += `<text x="${cx}" y="${margin * 0.55}" text-anchor="middle" dominant-baseline="middle"
      fill="#555" font-size="${axisFs}" font-family="monospace">${n}</text>`;
  }
  for (let m = 0; m < M; m++) {
    const cy = margin + (m + 0.5) * cs;
    body += `<text x="${margin * 0.5}" y="${cy}" text-anchor="middle" dominant-baseline="middle"
      fill="#555" font-size="${axisFs}" font-family="monospace">${m}</text>`;
  }

  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      const x = margin + n * cs;
      const y = margin + m * cs;
      const entries = grid[m][n];
      // If a highlight tid is set and no entry in this cell matches, dim it.
      const dimmed = highlightTid !== null
        && entries.length > 0
        && !entries.some(e => e.tid === highlightTid);

      if (entries.length === 0) {
        body += `<rect x="${x}" y="${y}" width="${cs}" height="${cs}"
          fill="#f0f0f0" stroke="#ccc" stroke-width="0.5"/>`;
        body += cellTextSVG(x + cs/2, y + cs/2, ['\u2014'], cs, '#bbb');
      } else if (entries.length === 1) {
        const { tid, vid, offset } = entries[0];
        const bg = dimmed ? '#f0f0f0' : colorTV(tid);
        const fg = dimmed ? '#bbb' : '#111';
        body += `<rect x="${x}" y="${y}" width="${cs}" height="${cs}"
          fill="${bg}" stroke="#ccc" stroke-width="0.5"/>`;
        const lines = [`T${tid}`, `V${vid}`];
        if (showOffset) lines.push(`@${offset}`);
        body += cellTextSVG(x + cs/2, y + cs/2, lines, cs, fg);
      } else {
        const bg = dimmed ? '#f0f0f0' : colorTV(entries[0].tid);
        const fg = dimmed ? '#bbb' : '#111';
        const stroke = dimmed ? '#ccc' : '#e53e3e';
        const sw = dimmed ? 0.5 : 1.5;
        body += `<rect x="${x}" y="${y}" width="${cs}" height="${cs}"
          fill="${bg}" stroke="${stroke}" stroke-width="${sw}"/>`;
        const lines = [];
        for (const e of entries) {
          let label = `T${e.tid}/V${e.vid}`;
          if (showOffset) label += `@${e.offset}`;
          lines.push(label);
        }
        body += cellTextSVG(x + cs/2, y + cs/2, lines, cs, fg);
      }
    }
  }

  return `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="${svgFitStyle(W, H)}">
    <rect width="${W}" height="${H}" fill="white"/>
    ${body}
  </svg>`;
}

/** Like buildLayoutSVG but cells NOT in highlightSet are greyed out. */
function buildHighlightedLayoutSVG(shape, stride, highlightSet, mode, edgeSet, shadowMap) {
  const modes = toModeSet(mode);
  const [M, N] = productEach(shape);
  if (M * N > MAX_CELLS) return errSVG(`Grid too large: ${M}x${N}`);
  if (M === 0 || N === 0) return errSVG(`Empty grid: ${M}x${N}`);

  const cs = cellSize(M, N);
  const margin = cs;
  const W = margin + N * cs;
  const H = margin + M * cs;
  const axisFs = Math.max(8, Math.min(14, Math.floor(cs * 0.38)));

  let body = '';

  for (let n = 0; n < N; n++) {
    const cx = margin + (n + 0.5) * cs;
    body += `<text x="${cx}" y="${margin * 0.55}" text-anchor="middle" dominant-baseline="middle"
      fill="#555" font-size="${axisFs}" font-family="monospace">${n}</text>`;
  }
  for (let m = 0; m < M; m++) {
    const cy = margin + (m + 0.5) * cs;
    body += `<text x="${margin * 0.5}" y="${cy}" text-anchor="middle" dominant-baseline="middle"
      fill="#555" font-size="${axisFs}" font-family="monospace">${m}</text>`;
  }

  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      const flatPos = m + n * M;
      const idx = layoutAt(shape, stride, m, n);
      const x = margin + n * cs;
      const y = margin + m * cs;
      const lit = highlightSet.has(flatPos);
      const shadowSrc = (!lit && shadowMap) ? shadowMap.get(idx) : undefined;
      const inShadow = shadowSrc !== undefined;

      let bg, fg, stroke, sw, fillOpacity;
      if (lit) {
        bg = colorHighlight(idx);
        fg = textOnBG(bg);
        stroke = '#1e3a5f';
        sw = 2;
        fillOpacity = 1;
      } else if (inShadow) {
        // Same color as the corresponding cell in the original R tile, faded
        bg = colorHighlight(shadowSrc);
        fg = '#555';
        stroke = '#ccc';
        sw = 0.5;
        fillOpacity = 0.35;
      } else {
        bg = '#e8e8e8';
        fg = '#bbb';
        stroke = '#ddd';
        sw = 0.5;
        fillOpacity = 1;
      }

      body += `<rect x="${x}" y="${y}" width="${cs}" height="${cs}"
        fill="${bg}" fill-opacity="${fillOpacity}" stroke="${stroke}" stroke-width="${sw}"/>`;
      const lines = buildCellLines(modes, idx, flatPos, `(${m},${n})`);
      body += cellTextSVG(x + cs/2, y + cs/2, lines, cs, fg);
      // If this cell's offset is in the edge set (complement anchor), draw an edge marker
      if (edgeSet && edgeSet.has(idx)) {
        body += `<rect x="${x}" y="${y}" width="${cs}" height="${cs}"
          fill="none" stroke="#f59e0b" stroke-width="3"/>`;
      }
    }
  }

  return `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="${svgFitStyle(W, H)}">
    <rect width="${W}" height="${H}" fill="white"/>
    ${body}
  </svg>`;
}

/** Render a pre-computed 2-D grid of values (used for composition result R). */
function buildGridSVG(grid, M, N, mode) {
  const modes = toModeSet(mode);
  if (M * N > MAX_CELLS) return errSVG(`Grid too large: ${M}x${N}`);
  if (M === 0 || N === 0) return errSVG(`Empty grid: ${M}x${N}`);

  const cs = cellSize(M, N);
  const margin = cs;
  const W = margin + N * cs;
  const H = margin + M * cs;
  const axisFs = Math.max(8, Math.min(14, Math.floor(cs * 0.38)));

  let body = '';

  for (let n = 0; n < N; n++) {
    const cx = margin + (n + 0.5) * cs;
    body += `<text x="${cx}" y="${margin * 0.55}" text-anchor="middle" dominant-baseline="middle"
      fill="#555" font-size="${axisFs}" font-family="monospace">${n}</text>`;
  }
  for (let m = 0; m < M; m++) {
    const cy = margin + (m + 0.5) * cs;
    body += `<text x="${margin * 0.5}" y="${cy}" text-anchor="middle" dominant-baseline="middle"
      fill="#555" font-size="${axisFs}" font-family="monospace">${m}</text>`;
  }

  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      const idx = grid[m][n];
      const flatI = m + n * M;
      const lines = buildCellLines(modes, idx, flatI, `(${m},${n})`);
      const bg = colorHighlight(idx);
      const fg = textOnBG(bg);
      const x = margin + n * cs;
      const y = margin + m * cs;
      body += `<rect x="${x}" y="${y}" width="${cs}" height="${cs}"
        fill="${bg}" stroke="#1e3a5f" stroke-width="2"/>`;
      body += cellTextSVG(x + cs/2, y + cs/2, lines, cs, fg);
    }
  }

  return `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="${svgFitStyle(W, H)}">
    <rect width="${W}" height="${H}" fill="white"/>
    ${body}
  </svg>`;
}

function errSVG(msg) {
  return `<svg viewBox="0 0 400 60" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto">
    <rect width="400" height="60" fill="white"/>
    <text x="10" y="35" fill="#e53e3e" font-size="13" font-family="monospace">${msg}</text>
  </svg>`;
}

// ═══════════════════════════════════════════════════════
//  Multi-tab management
// ═══════════════════════════════════════════════════════

let tabCounter = 0;
let activeOuterTabId = null;

function generateTabContent(id) {
  return `
  <div class="container">
    <div class="tab-bar">
      <div class="tab active" onclick="switchInnerTab('${id}', 'layout')">Layout</div>
      <div class="tab" onclick="switchInnerTab('${id}', 'tv')">TV Layout</div>
      <div class="tab" onclick="switchInnerTab('${id}', 'composition')">Composition</div>
    </div>

    <!-- Layout panel -->
    <div id="${id}-tab-layout" class="panel active">
      <div class="controls">
        <h2>Layout</h2>
        <div class="form-group">
          <label>Shape : Stride</label>
          <input type="text" id="${id}-layout-input" value="(10, 10):(1, 10)">
        </div>
        <div id="${id}-layout-error" class="error-msg"></div>
        <button class="btn btn-render" onclick="renderLayout('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-layout-export" onclick="exportLayout('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setL('${id}','(10, 10):(1, 10)')">Column-major (10x10)</button>
            <button class="preset-btn" onclick="setL('${id}','(10, 10):(10, 1)')">Row-major (10x10)</button>
            <button class="preset-btn" onclick="setL('${id}','(8, 8):(1, 8)')">Column-major (8x8)</button>
            <button class="preset-btn" onclick="setL('${id}','(8, 8):(8, 1)')">Row-major (8x8)</button>
            <button class="preset-btn" onclick="setL('${id}','(8, 8):(2, 16)')">Strided (8x8, s=2)</button>
            <button class="preset-btn" onclick="setL('${id}','((2,2),(2,2)):((1,4),(2,8))')">Nested ((2,2),(2,2))</button>
            <button class="preset-btn" onclick="setL('${id}','((4,2),(4,2)):((1,8),(2,32))')">Nested ((4,2),(4,2))</button>
            <button class="preset-btn" onclick="setL('${id}','(4, 32):(32, 1)')">Warp row-major (4x32)</button>
          </div>
        </div>

        <div class="hint">
          Format: <code>shape:stride</code><br>
          Examples:<br>
          <code>(M, N):(s0, s1)</code><br>
          <code>((a,b), c):((sa,sb), sc)</code><br>
          Omit <code>:stride</code> for column-major default.
        </div>
      </div>

      <div class="visualization">
        <div class="viz-header">
          <span class="viz-title" id="${id}-layout-title">&mdash;</span>
          <span style="display:flex;align-items:center;gap:8px">
            <span class="mode-btn-group" id="${id}-layout-mode-btns">
              <button class="mode-btn active" onclick="setLayoutMode('${id}','value')">value</button>
              <button class="mode-btn" onclick="setLayoutMode('${id}','index')">index</button>
              <button class="mode-btn" onclick="setLayoutMode('${id}','coord')">coord</button>
            </span>
            <button class="btn" id="${id}-layout-svg-host-zoom" onclick="toggleZoom('${id}-layout-svg-host')">Zoom in</button>
            <button class="btn" onclick="downloadSVG('${id}-layout-svg-host', 'layout.svg')">Download SVG</button>
          </span>
        </div>
        <div class="viz-box">
          <div id="${id}-layout-svg-host"></div>
        </div>
      </div>
    </div>

    <!-- TV Layout panel -->
    <div id="${id}-tab-tv" class="panel">
      <div class="controls">
        <h2>TV Layout</h2>
        <div class="form-group">
          <label>TV Layout &mdash; (num_threads, num_values):(t_stride, v_stride)</label>
          <input type="text" id="${id}-tv-layout-input" value="(32, 4):(1, 32)">
        </div>
        <div class="form-group">
          <label>Tile &mdash; (M, N):(stride_M, stride_N)
            <span style="color:#6b7280;font-weight:normal"> &mdash; CuTe uses col-major (1, M)</span>
          </label>
          <input type="text" id="${id}-tv-tile-input" value="(8, 16):(1, 8)">
        </div>

        <div class="form-group">
          <label>Underlying data is</label>
          <div style="display:flex;gap:5px;margin-top:5px">
            <button class="btn" id="${id}-tv-data-row" style="flex:1;font-size:0.75rem;padding:4px" onclick="setUnderlyingLayout('${id}','row')">Row-major</button>
            <button class="btn" id="${id}-tv-data-col" style="flex:1;font-size:0.75rem;padding:4px" onclick="setUnderlyingLayout('${id}','col')">Col-major</button>
          </div>
        </div>

        <div class="form-group">
          <label>Highlight thread (empty = none)</label>
          <input type="text" id="${id}-tv-highlight-tid" value="" placeholder="e.g. 3" oninput="setHighlightTid('${id}')">
        </div>

        <div class="form-group" style="border-top:1px solid #374151;padding-top:12px">
          <label style="color:#93c5fd;letter-spacing:0.5px">&mdash; OR compute from thr/val &mdash;</label>
        </div>
        <div class="form-group">
          <label>thr_layout</label>
          <input type="text" id="${id}-tv-thr-input" value="" placeholder="e.g. (2, 3):(3, 1)">
        </div>
        <div class="form-group">
          <label>val_layout</label>
          <input type="text" id="${id}-tv-val-input" value="" placeholder="e.g. (2, 2):(2, 1)">
        </div>
        <button class="btn" style="width:100%;font-size:0.8rem" onclick="computeTVFromThrVal('${id}')">Compute TV + Tile from thr/val</button>
        <div id="${id}-tv-error" class="error-msg"></div>
        <button class="btn btn-render" onclick="renderTV('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-tv-export" onclick="exportTV('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setTV('${id}','(4,(4,4)):(4,(16,1))','(16, 4):(4, 1)')">4Tx16V nested -> 16x4 row-major</button>
            <button class="preset-btn" onclick="setTV('${id}','(32, 4):(1, 32)','(8, 16):(16, 1)')">32Tx4V -> 8x16 row-major</button>
            <button class="preset-btn" onclick="setTV('${id}','(32, 4):(1, 32)','(8, 16):(1, 8)')">32Tx4V -> 8x16 col-major</button>
            <button class="preset-btn" onclick="setTV('${id}','(8, 8):(1, 8)','(8, 8):(8, 1)')">8Tx8V -> 8x8 tile</button>
            <button class="preset-btn" onclick="setTV('${id}','(4, 32):(32, 1)','(4, 32):(32, 1)')">Warp 4Tx32V</button>
            <button class="preset-btn" onclick="setTV('${id}','(128, 4):(1, 128)','(16, 32):(32, 1)')">128Tx4V -> 16x32</button>
          </div>
        </div>

        <div class="hint">
          TV layout maps each <code>(tid, vid)</code> pair to a
          position in the MxN tile.<br><br>
          Cell color = thread ID.<br>
          Cell text = T<i>tid</i> / V<i>vid</i>.<br>
          Empty cells (&mdash;) are not covered by any thread.<br><br>
          <b>Tiler strides matter.</b> Tilers returned by
          CuTe's <code>make_layout_tv</code> are typically
          <b>row-major</b> <code>(N, 1)</code>. Use the buttons
          above to quickly switch, or enter strides manually.<br><br>
          In Python, print the full tiler with:<br>
          <code>print(f"{tiler_mn}")</code><br>
          to see <code>(M, N):(sm, sn)</code>.
        </div>
      </div>

      <div class="visualization">
        <div class="viz-header">
          <span class="viz-title" id="${id}-tv-title">&mdash;</span>
          <span style="display:flex;align-items:center;gap:8px">
            <button class="mode-btn" id="${id}-tv-offset-btn" onclick="toggleTVOffset('${id}')">Show offset</button>
            <button class="btn" id="${id}-tv-svg-host-zoom" onclick="toggleZoom('${id}-tv-svg-host')">Zoom in</button>
            <button class="btn" onclick="downloadSVG('${id}-tv-svg-host', 'tv_layout.svg')">Download SVG</button>
          </span>
        </div>
        <div class="viz-box">
          <div id="${id}-tv-svg-host"></div>
        </div>
        <div class="legend" id="${id}-tv-legend"></div>
      </div>
    </div>

    <!-- Composition panel -->
    <div id="${id}-tab-composition" class="panel">
      <div class="controls">
        <h2>Composition &amp; Complement</h2>
        <div class="form-group">
          <label>Layout A &mdash; the outer layout</label>
          <input type="text" id="${id}-comp-a-input" value="(4, 4):(4, 1)">
        </div>
        <div class="form-group">
          <label>Layout B &mdash; layout or tiler (one layout per line for by-mode tiler)</label>
          <textarea id="${id}-comp-b-input" rows="2">(2, 2):(1, 2)</textarea>
        </div>
        <div id="${id}-comp-error" class="error-msg"></div>
        <div id="${id}-comp-result" class="comp-result-box"></div>
        <button class="btn btn-render" onclick="renderComposition('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-comp-complement-btn" onclick="toggleComplement('${id}')">Render complement</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-comp-export" onclick="exportComp('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setComp('${id}','(4,4):(4,1)','(2,2):(1,2)')">Row-maj A(4x4), Col-maj B(2x2)</button>
            <button class="preset-btn" onclick="setComp('${id}','(8,8):(1,8)','(4,4):(1,4)')">Col-maj A(8x8), Col-maj B(4x4)</button>
            <button class="preset-btn" onclick="setComp('${id}','(8,8):(1,8)','(2,4):(2,8)')">Col-maj A(8x8), Strided B(2x4)</button>
            <button class="preset-btn" onclick="setComp('${id}','(4,8):(8,1)','(2,4):(1,2)')">Row-maj A(4x8), Col-maj B(2x4)</button>
            <button class="preset-btn" onclick="setComp('${id}','(4,4):(1,4)','(4,4):(1,0)')">Broadcast: B stride-0 col</button>
            <button class="preset-btn" onclick="setComp('${id}','(12,(4,8)):(59,(13,1))','(3):(4)\\n(8):(2)')">Tiler &lt;3:4, 8:2&gt; on A(12x32)</button>
            <button class="preset-btn" onclick="setComp('${id}','(12,(4,8)):(59,(13,1))','3\\n8')">Shape-tiler &lt;3:1, 8:1&gt; on A(12x32)</button>
            <button class="preset-btn" onclick="setComp('${id}','(4,8):(8,1)','2\\n4')">Shape-tiler &lt;2:1, 4:1&gt; on A(4x8)</button>
          </div>
        </div>

        <div class="hint">
          B can be a single layout or multiple layouts
          (one per line) for by-mode tiler composition.<br><br>
          <b>Single layout</b>: <code>R(c) = A(B(c))</code> &mdash;
          B's output is a 1-D coordinate into A.<br>
          <b>Tiler (multi-line)</b>: each line applies to one mode
          of A independently. E.g. line 1 = <code>(3):(4)</code>,
          line 2 = <code>(8):(2)</code> gives tiler
          <code>&lt;3:4, 8:2&gt;</code>.<br>
          A shape like <code>3</code> is treated as <code>3:1</code>
          (stride-1).<br><br>
          Non-injective B (e.g. stride 0) gives broadcasting behavior.
        </div>
      </div>

      <div class="comp-results">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-comp-a-title">A</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-comp-a-mode-btns">
                <button class="mode-btn active" onclick="setCompMode('${id}','a','value')">value</button>
                <button class="mode-btn" onclick="setCompMode('${id}','a','index')">index</button>
                <button class="mode-btn" onclick="setCompMode('${id}','a','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-comp-a-svg-zoom" onclick="toggleZoom('${id}-comp-a-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-comp-a-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-comp-b-title">B</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-comp-b-mode-btns">
                <button class="mode-btn active" onclick="setCompMode('${id}','b','value')">value</button>
                <button class="mode-btn" onclick="setCompMode('${id}','b','index')">index</button>
                <button class="mode-btn" onclick="setCompMode('${id}','b','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-comp-b-svg-zoom" onclick="toggleZoom('${id}-comp-b-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-comp-b-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label">A highlighted by B</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-comp-highlight-mode-btns">
                <button class="mode-btn active" onclick="setCompMode('${id}','highlight','value')">value</button>
                <button class="mode-btn" onclick="setCompMode('${id}','highlight','index')">index</button>
                <button class="mode-btn" onclick="setCompMode('${id}','highlight','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-comp-highlight-svg-zoom" onclick="toggleZoom('${id}-comp-highlight-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-comp-highlight-svg"></div></div>
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-comp-r-title">R = A(B)</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-comp-r-mode-btns">
                <button class="mode-btn active" onclick="setCompMode('${id}','r','value')">value</button>
                <button class="mode-btn" onclick="setCompMode('${id}','r','index')">index</button>
                <button class="mode-btn" onclick="setCompMode('${id}','r','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-comp-r-svg-zoom" onclick="toggleZoom('${id}-comp-r-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-comp-r-svg"></div></div>
        </div>
      </div>

      <div id="${id}-comp-complement-section" style="display:none;margin-top:16px;width:100%">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label">Complement layout &mdash; tiles R over A to cover the full coordinate space</span>
            <span style="display:flex;align-items:center;gap:4px">
              <span class="mode-btn-group" id="${id}-comp-complement-mode-btns">
                <button class="mode-btn active" onclick="setCompMode('${id}','complement','value')">value</button>
                <button class="mode-btn" onclick="setCompMode('${id}','complement','index')">index</button>
                <button class="mode-btn" onclick="setCompMode('${id}','complement','coord')">coord</button>
              </span>
              <button class="mode-btn" id="${id}-comp-complement-svg-zoom" onclick="toggleZoom('${id}-comp-complement-svg')">Zoom in</button>
            </span>
          </div>
          <div class="viz-box"><div id="${id}-comp-complement-svg"></div></div>
        </div>
      </div>
    </div>
  </div>`;
}

function addOuterTab() {
  tabCounter++;
  const id = `ot${tabCounter}`;

  const btn = document.createElement('div');
  btn.className = 'outer-tab';
  btn.id = `${id}-btn`;
  btn.setAttribute('data-tab-id', id);
  btn.onclick = () => switchOuterTab(id);
  btn.innerHTML = `<span class="outer-tab-label" id="${id}-label">New Tab</span><span class="close-tab-btn" onclick="event.stopPropagation(); closeOuterTab('${id}')">&times;</span>`;
  document.getElementById('outer-tab-buttons').appendChild(btn);

  const panel = document.createElement('div');
  panel.className = 'outer-panel';
  panel.id = `${id}-panel`;
  panel.innerHTML = generateTabContent(id);
  document.getElementById('outer-panels').appendChild(panel);

  switchOuterTab(id);
  renderLayout(id);
  return id;
}

function switchOuterTab(id) {
  document.querySelectorAll('.outer-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.outer-panel').forEach(p => p.classList.remove('active'));
  document.getElementById(`${id}-btn`).classList.add('active');
  document.getElementById(`${id}-panel`).classList.add('active');
  activeOuterTabId = id;
}

function closeOuterTab(id) {
  const allTabs = document.querySelectorAll('#outer-tab-buttons .outer-tab');
  if (allTabs.length <= 1) return;

  const btn = document.getElementById(`${id}-btn`);
  const panel = document.getElementById(`${id}-panel`);

  if (activeOuterTabId === id) {
    const tabIds = Array.from(allTabs).map(t => t.getAttribute('data-tab-id'));
    const idx = tabIds.indexOf(id);
    const nextId = tabIds[idx === tabIds.length - 1 ? idx - 1 : idx + 1];
    switchOuterTab(nextId);
  }

  btn.remove();
  panel.remove();
}

// ═══════════════════════════════════════════════════════
//  UI (per-tab)
// ═══════════════════════════════════════════════════════

function switchInnerTab(tabId, mode) {
  const panel = document.getElementById(`${tabId}-panel`);
  panel.querySelectorAll('.tab-bar .tab').forEach(t => t.classList.remove('active'));
  panel.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  const tabs = panel.querySelectorAll('.tab-bar .tab');
  const modeIndex = { layout: 0, tv: 1, composition: 2 };
  tabs[modeIndex[mode]].classList.add('active');
  document.getElementById(`${tabId}-tab-${mode}`).classList.add('active');
}

function showErr(id, msg) {
  const el = document.getElementById(id);
  el.textContent = msg;
  el.style.display = msg ? 'block' : 'none';
}

function updateOuterTabLabel(tabId, text) {
  const label = document.getElementById(`${tabId}-label`);
  if (label) label.textContent = text;
}

/** Sync button highlights with a modes Set. */
function updateModeBtns(groupId, modes) {
  const group = document.getElementById(groupId);
  if (!group) return;
  const modeSet = (modes instanceof Set) ? modes : new Set([modes]);
  group.querySelectorAll('.mode-btn').forEach(btn => {
    btn.classList.toggle('active', modeSet.has(btn.textContent));
  });
}

// ── Layout tab ──

const layoutState = {};

function renderLayout(tabId) {
  showErr(`${tabId}-layout-error`, '');
  try {
    const inputVal = document.getElementById(`${tabId}-layout-input`).value;
    let { shape, stride } = parseLayout(inputVal);
    const [M, N] = productEach(shape);
    document.getElementById(`${tabId}-layout-title`).textContent =
      `shape=${JSON.stringify(shape).replace(/"/g,'')}  stride=${JSON.stringify(stride).replace(/"/g,'')}  \u2014  ${M}\u00d7${N} grid`;
    layoutState[tabId] = { shape, stride, modes: new Set(['value']) };
    document.getElementById(`${tabId}-layout-svg-host`).innerHTML = buildLayoutSVG(shape, stride, new Set(['value']));
    applyZoomState(`${tabId}-layout-svg-host`);
    updateModeBtns(`${tabId}-layout-mode-btns`, layoutState[tabId].modes);
    updateOuterTabLabel(tabId, `Layout:${inputVal.trim()}`);
  } catch (e) {
    showErr(`${tabId}-layout-error`, e.message);
    document.getElementById(`${tabId}-layout-svg-host`).innerHTML = '';
  }
}

function setLayoutMode(tabId, mode) {
  const s = layoutState[tabId];
  if (!s) return;
  if (s.modes.has(mode)) {
    if (s.modes.size > 1) s.modes.delete(mode);
  } else {
    s.modes.add(mode);
  }
  document.getElementById(`${tabId}-layout-svg-host`).innerHTML =
    buildLayoutSVG(s.shape, s.stride, s.modes);
  applyZoomState(`${tabId}-layout-svg-host`);
  updateModeBtns(`${tabId}-layout-mode-btns`, s.modes);
}

// ── TV Layout tab ──

const tvState = {};

function renderTV(tabId) {
  showErr(`${tabId}-tv-error`, '');
  try {
    const tvInput = document.getElementById(`${tabId}-tv-layout-input`).value;
    const tvL   = parseLayout(tvInput);
    const tileL = parseLayout(document.getElementById(`${tabId}-tv-tile-input`).value);

    const numT = product(tvL.shape[0]);
    const numV = product(tvL.shape[1]);
    const M    = product(tileL.shape[0]);
    const N    = product(tileL.shape[1]);

    const prev = tvState[tabId] || {};
    const showOffset = prev.showOffset || false;
    const underlyingLayout = prev.underlyingLayout || 'col';
    // Read the highlight-thread input; empty or invalid → null (no filter).
    const highlightRaw = document.getElementById(`${tabId}-tv-highlight-tid`).value.trim();
    const highlightTid = highlightRaw === '' ? null : parseInt(highlightRaw, 10);
    const highlightValid = highlightTid !== null && !isNaN(highlightTid);
    tvState[tabId] = { tvL, tileL, showOffset, underlyingLayout };

    const titleHL = highlightValid ? `  \u2014  highlight T${highlightTid}` : '';
    document.getElementById(`${tabId}-tv-title`).textContent =
      `${numT} threads \u00d7 ${numV} values  \u2014  ${M}\u00d7${N} tile (${underlyingLayout}-major data)${titleHL}`;

    document.getElementById(`${tabId}-tv-svg-host`).innerHTML =
      buildTVSVG(tvL.shape, tvL.stride, tileL.shape, tileL.stride, showOffset, underlyingLayout,
                 highlightValid ? highlightTid : null);
    applyZoomState(`${tabId}-tv-svg-host`);

    const btn = document.getElementById(`${tabId}-tv-offset-btn`);
    if (btn) btn.classList.toggle('active', showOffset);
    const rowBtn = document.getElementById(`${tabId}-tv-data-row`);
    const colBtn = document.getElementById(`${tabId}-tv-data-col`);
    if (rowBtn) rowBtn.classList.toggle('active', underlyingLayout === 'row');
    if (colBtn) colBtn.classList.toggle('active', underlyingLayout === 'col');

    buildLegend(tabId, numT);
    updateOuterTabLabel(tabId, `TV-Layout:${tvInput.trim()}`);
  } catch (e) {
    showErr(`${tabId}-tv-error`, e.message);
    document.getElementById(`${tabId}-tv-svg-host`).innerHTML = '';
    document.getElementById(`${tabId}-tv-legend`).innerHTML = '';
  }
}

/** Set the underlying data layout ('row' | 'col') for offset display. */
function setUnderlyingLayout(tabId, layout) {
  if (!tvState[tabId]) tvState[tabId] = {};
  tvState[tabId].underlyingLayout = layout;
  if (tvState[tabId].tvL) renderTV(tabId);
}

function toggleTVOffset(tabId) {
  const s = tvState[tabId];
  if (!s) return;
  s.showOffset = !s.showOffset;
  renderTV(tabId);
}

/** Re-render when the highlight-thread input changes (live update). */
function setHighlightTid(tabId) {
  if (tvState[tabId] && tvState[tabId].tvL) renderTV(tabId);
}

function buildLegend(tabId, numT) {
  const legend = document.getElementById(`${tabId}-tv-legend`);
  const shown = Math.min(numT, 16);
  let html = '';
  for (let tid = 0; tid < shown; tid++) {
    html += `<div class="legend-item">
      <div class="legend-swatch" style="background:${TV_COLORS[tid % 8]}"></div>
      T${tid}
    </div>`;
  }
  if (numT > 16) html += `<div class="legend-item">\u2026 (${numT} threads total)</div>`;
  legend.innerHTML = html;
}

function setL(tabId, val) {
  document.getElementById(`${tabId}-layout-input`).value = val;
  renderLayout(tabId);
}

function setTV(tabId, tv, tile) {
  document.getElementById(`${tabId}-tv-layout-input`).value = tv;
  document.getElementById(`${tabId}-tv-tile-input`).value = tile;
  renderTV(tabId);
}

function setTileStride(tabId, mode) {
  const el = document.getElementById(`${tabId}-tv-tile-input`);
  try {
    const {shape} = parseLayout(el.value);
    const [M, N] = productEach(shape);
    el.value = mode === 'row' ? `(${M}, ${N}):(${N}, 1)` : `(${M}, ${N}):(1, ${M})`;
    renderTV(tabId);
  } catch (e) {
    showErr(`${tabId}-tv-error`, 'Could not parse tile shape: ' + e.message);
  }
}

/** Format a (possibly nested) shape/stride pair as CuTe-style "(...)":"(...)". */
function formatLayoutStr(shape, stride) {
  function fmt(x) {
    if (typeof x === 'number') return String(x);
    return '(' + x.map(fmt).join(',') + ')';
  }
  return `${fmt(shape)}:${fmt(stride)}`;
}

/** parseLayout normalizes to rank-2. Strip trailing (shape=1, stride=0) modes
 *  to recover the user's intended rank. */
function stripTrivialTrailing(shape, stride) {
  if (!Array.isArray(shape)) return { shape, stride };
  const s = shape.slice(), d = stride.slice();
  while (s.length > 1 && s[s.length - 1] === 1 && d[d.length - 1] === 0) {
    s.pop(); d.pop();
  }
  if (s.length === 1) return { shape: s[0], stride: d[0] };
  return { shape: s, stride: d };
}

/** Fill TV Layout + Tile inputs from thr_layout and val_layout via make_layout_tv. */
function computeTVFromThrVal(tabId) {
  showErr(`${tabId}-tv-error`, '');
  try {
    const thrStr = document.getElementById(`${tabId}-tv-thr-input`).value;
    const valStr = document.getElementById(`${tabId}-tv-val-input`).value;
    const thrRaw = parseLayout(thrStr);
    const valRaw = parseLayout(valStr);
    const thrP = stripTrivialTrailing(thrRaw.shape, thrRaw.stride);
    const valP = stripTrivialTrailing(valRaw.shape, valRaw.stride);
    const thr = new Layout(thrP.shape, thrP.stride);
    const val = new Layout(valP.shape, valP.stride);

    const { tiler_mn, layout_tv } = make_layout_tv(thr, val);

    const tvString = formatLayoutStr(layout_tv.shape, layout_tv.stride);
    const [M, N] = tiler_mn;
    // layout_tv outputs col-major flat indices into tiler_mn. parseLayout
    // defaults shape-only inputs to col-major strides (1, M), so emit the shape.
    const tileString = `(${M}, ${N})`;

    document.getElementById(`${tabId}-tv-layout-input`).value = tvString;
    document.getElementById(`${tabId}-tv-tile-input`).value = tileString;
    renderTV(tabId);
  } catch (e) {
    showErr(`${tabId}-tv-error`, 'Failed to compute from thr/val: ' + e.message);
  }
}

// ═══════════════════════════════════════════════════════
//  Composition
// ═══════════════════════════════════════════════════════

const compState = {};

function renderComposition(tabId) {
  showErr(`${tabId}-comp-error`, '');
  const svgIds = ['comp-a-svg', 'comp-b-svg', 'comp-highlight-svg', 'comp-r-svg'];

  try {
    const aInput = document.getElementById(`${tabId}-comp-a-input`).value;
    const bRaw = document.getElementById(`${tabId}-comp-b-input`).value;
    const aL = parseLayout(aInput);
    const [M_A, N_A] = productEach(aL.shape);

    const bLines = bRaw.split('\n').map(s => s.trim()).filter(s => s.length > 0);
    if (bLines.length === 0) throw new Error('B input is empty');

    const isTiler = bLines.length > 1;
    let rGrid, highlightSet, M_R, N_R, bL, bLayouts;

    if (isTiler) {
      if (bLines.length !== 2) {
        throw new Error(
          `Tiler must have exactly 2 layouts (one per mode of rank-2 A), got ${bLines.length}`);
      }
      bLayouts = bLines.map(line => parseLayout(line));
      bL = null;

      const sizeA0 = product(aL.shape[0]);
      const sizeA1 = product(aL.shape[1]);

      const sizeB0 = product(bLayouts[0].shape[0]) * product(bLayouts[0].shape[1]);
      const r0Vals = [], b0Out = new Set();
      for (let i = 0; i < sizeB0; i++) {
        const v = evalLayoutFlat(bLayouts[0], i);
        if (v < 0 || v >= sizeA0)
          throw new Error(`B[0](${i}) = ${v} out of range [0, ${sizeA0}) for A mode-0`);
        b0Out.add(v);
        r0Vals.push(evalModeAt(aL.shape[0], aL.stride[0], v));
      }

      const sizeB1 = product(bLayouts[1].shape[0]) * product(bLayouts[1].shape[1]);
      const r1Vals = [], b1Out = new Set();
      for (let j = 0; j < sizeB1; j++) {
        const v = evalLayoutFlat(bLayouts[1], j);
        if (v < 0 || v >= sizeA1)
          throw new Error(`B[1](${j}) = ${v} out of range [0, ${sizeA1}) for A mode-1`);
        b1Out.add(v);
        r1Vals.push(evalModeAt(aL.shape[1], aL.stride[1], v));
      }

      M_R = sizeB0;
      N_R = sizeB1;

      rGrid = [];
      for (let i = 0; i < M_R; i++) {
        rGrid[i] = [];
        for (let j = 0; j < N_R; j++)
          rGrid[i][j] = r0Vals[i] + r1Vals[j];
      }

      highlightSet = new Set();
      for (const m of b0Out)
        for (const n of b1Out)
          highlightSet.add(m + n * M_A);

    } else {
      bL = parseLayout(bLines[0]);
      bLayouts = null;

      const [M_B, N_B] = productEach(bL.shape);
      const sizeA = M_A * N_A;

      highlightSet = new Set();
      for (let m = 0; m < M_B; m++) {
        for (let n = 0; n < N_B; n++) {
          const bVal = layoutAt(bL.shape, bL.stride, m, n);
          if (bVal < 0 || bVal >= sizeA)
            throw new Error(`B(${m},${n}) = ${bVal} out of range [0, ${sizeA}) for A (size ${M_A}x${N_A})`);
          highlightSet.add(bVal);
        }
      }

      rGrid = [];
      for (let m = 0; m < M_B; m++) {
        rGrid[m] = [];
        for (let n = 0; n < N_B; n++) {
          const bVal = layoutAt(bL.shape, bL.stride, m, n);
          rGrid[m][n] = layoutAt(aL.shape, aL.stride, bVal % M_A, Math.floor(bVal / M_A));
        }
      }
      M_R = M_B;
      N_R = N_B;
    }

    compState[tabId] = {
      aL, bL, bLayouts, isTiler, bInputLines: bLines,
      rGrid, highlightSet, M_R, N_R,
      modes: {
        a: new Set(['value']), b: new Set(['value']),
        highlight: new Set(['value']), r: new Set(['value']),
        complement: new Set(['value'])
      }
    };

    // Hide complement section on fresh render; user must click "Render complement"
    const compSection = document.getElementById(`${tabId}-comp-complement-section`);
    if (compSection) compSection.style.display = 'none';
    const compBtn = document.getElementById(`${tabId}-comp-complement-btn`);
    if (compBtn) { compBtn.textContent = 'Render complement'; compBtn.classList.remove('active'); }

    document.getElementById(`${tabId}-comp-a-title`).textContent = `A: ${aInput.trim()}`;
    document.getElementById(`${tabId}-comp-b-title`).textContent =
      isTiler ? `Tiler: <${bLines.join(', ')}>` : `B: ${bLines[0]}`;
    document.getElementById(`${tabId}-comp-r-title`).textContent =
      isTiler ? 'R = A(Tiler)' : 'R = A(B)';

    renderCompGrid(tabId, 'a');
    renderCompGrid(tabId, 'b');
    renderCompGrid(tabId, 'highlight');
    renderCompGrid(tabId, 'r');

    // Algebraic composition R = A(B) via layout.js
    const resultEl = document.getElementById(`${tabId}-comp-result`);
    try {
      const aStripped = stripTrivialTrailing(aL.shape, aL.stride);
      const aLayout = new Layout(aStripped.shape, aStripped.stride);
      let R;
      if (isTiler) {
        const bTuple = bLayouts.map(b => {
          const p = stripTrivialTrailing(b.shape, b.stride);
          return new Layout(p.shape, p.stride);
        });
        R = composition(aLayout, bTuple);
      } else {
        const bStripped = stripTrivialTrailing(bL.shape, bL.stride);
        const bLayout = new Layout(bStripped.shape, bStripped.stride);
        R = composition(aLayout, bLayout);
      }
      const label = isTiler ? 'R = A(Tiler)' : 'R = A(B)';
      resultEl.textContent = `${label} = ${formatLayoutStr(R.shape, R.stride)}`;
      resultEl.classList.add('visible');
    } catch (e) {
      resultEl.textContent = `Algebraic composition unavailable: ${e.message}`;
      resultEl.classList.add('visible');
    }

    updateOuterTabLabel(tabId, `Compose:${aInput.trim()}`);
  } catch (e) {
    showErr(`${tabId}-comp-error`, e.message);
    svgIds.forEach(h => {
      const el = document.getElementById(`${tabId}-${h}`);
      if (el) el.innerHTML = '';
    });
    const resultEl = document.getElementById(`${tabId}-comp-result`);
    if (resultEl) resultEl.classList.remove('visible');
  }
}

function renderCompGrid(tabId, which) {
  const s = compState[tabId];
  if (!s) return;
  const modes = s.modes[which];
  const host = document.getElementById(`${tabId}-comp-${which}-svg`);

  if (which === 'b' && s.isTiler) {
    let html = '';
    s.bLayouts.forEach((bl, i) => {
      html += `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin:${i > 0 ? '10px' : '0'} 0 4px">B[${i}]: ${s.bInputLines[i]}</div>`;
      html += buildLayoutSVG(bl.shape, bl.stride, modes);
    });
    host.innerHTML = html;
  } else {
    let svg;
    switch (which) {
      case 'a':
        svg = buildLayoutSVG(s.aL.shape, s.aL.stride, modes);
        break;
      case 'b':
        svg = buildLayoutSVG(s.bL.shape, s.bL.stride, modes);
        break;
      case 'highlight':
        svg = buildHighlightedLayoutSVG(s.aL.shape, s.aL.stride, s.highlightSet, modes, s.complementAnchors, s.shadowMap);
        break;
      case 'r':
        svg = buildGridSVG(s.rGrid, s.M_R, s.N_R, modes);
        break;
      case 'complement':
        if (s.complementShape) {
          svg = buildLayoutSVG(s.complementShape, s.complementStride, modes);
        } else {
          svg = '';
        }
        break;
    }
    host.innerHTML = svg;
  }

  applyZoomState(`${tabId}-comp-${which}-svg`);
  updateModeBtns(`${tabId}-comp-${which}-mode-btns`, modes);
}

function setCompMode(tabId, which, mode) {
  const s = compState[tabId];
  if (!s) return;
  const modes = s.modes[which];
  if (modes.has(mode)) {
    if (modes.size > 1) modes.delete(mode);
  } else {
    modes.add(mode);
  }
  renderCompGrid(tabId, which);
}

/** Compute complement(R, size(A)) where R = A(B), and visualize it.
 *  - Appends "complement(R, A) = ..." to the result box
 *  - Shows a new viz below the 4-panel grid with the complement layout
 *  - Adds amber edge markers to "A highlighted by B" at each anchor offset
 *    (cells whose A(m,n) equals one of the complement's output values). */
function toggleComplement(tabId) {
  const s = compState[tabId];
  if (!s) {
    showErr(`${tabId}-comp-error`, 'Click Render first before Render complement.');
    return;
  }
  if (s.shadowMap) hideComplement(tabId);
  else             showComplement(tabId);
}

function hideComplement(tabId) {
  const s = compState[tabId];
  if (!s) return;
  delete s.shadowMap;
  delete s.complementAnchors;
  delete s.complementShape;
  delete s.complementStride;
  // Restore the result text (strip the complement line)
  const resultEl = document.getElementById(`${tabId}-comp-result`);
  if (s.preComplementResultText !== undefined) {
    resultEl.textContent = s.preComplementResultText;
    delete s.preComplementResultText;
  }
  // Hide the complement viz section
  document.getElementById(`${tabId}-comp-complement-section`).style.display = 'none';
  // Re-render the highlighted A to remove edges + shadow tiles
  renderCompGrid(tabId, 'highlight');
  // Update button state
  const btn = document.getElementById(`${tabId}-comp-complement-btn`);
  if (btn) { btn.textContent = 'Render complement'; btn.classList.remove('active'); }
}

function showComplement(tabId) {
  const s = compState[tabId];
  if (!s) return;
  showErr(`${tabId}-comp-error`, '');
  try {
    // Reconstruct R = A(B) via layout.js composition
    const aStripped = stripTrivialTrailing(s.aL.shape, s.aL.stride);
    const aLayout = new Layout(aStripped.shape, aStripped.stride);
    let R;
    if (s.isTiler) {
      const bTuple = s.bLayouts.map(b => {
        const p = stripTrivialTrailing(b.shape, b.stride);
        return new Layout(p.shape, p.stride);
      });
      R = composition(aLayout, bTuple);
    } else {
      const bStripped = stripTrivialTrailing(s.bL.shape, s.bL.stride);
      const bLayout = new Layout(bStripped.shape, bStripped.stride);
      R = composition(aLayout, bLayout);
    }

    const sizeA = aLayout.size();
    const C = complement(R, sizeA);

    // Collect anchor offsets: C(i) for i in [0, size(C))
    const anchors = new Set();
    const cSize = C.size();
    for (let i = 0; i < cSize; i++) {
      anchors.add(C.call(i));
    }

    // R's output VALUES (not B's flat positions). These are A's values at the
    // cells the original R tile selects — what we need to color-match against.
    const rOutputs = new Set();
    for (let m = 0; m < s.M_R; m++) {
      for (let n = 0; n < s.N_R; n++) {
        rOutputs.add(s.rGrid[m][n]);
      }
    }

    // Build shadowMap: for each NON-zero anchor c, for each r in R's outputs,
    // map (c + r) -> r. These cells are the shifted copies of the original R tile;
    // they get colored like their corresponding r, but faded.
    const shadowMap = new Map();
    for (const c of anchors) {
      if (c === 0) continue;
      for (const r of rOutputs) {
        shadowMap.set(c + r, r);
      }
    }

    // Normalize the complement Layout to a rank-2 shape/stride for rendering
    const cStr = formatLayoutStr(C.shape, C.stride);
    const cParsed = parseLayout(cStr);

    s.complementAnchors = anchors;
    s.shadowMap = shadowMap;
    s.complementShape = cParsed.shape;
    s.complementStride = cParsed.stride;

    // Append complement info to the result text box (remember prev to allow restore)
    const resultEl = document.getElementById(`${tabId}-comp-result`);
    s.preComplementResultText = resultEl.textContent;
    resultEl.textContent = `${resultEl.textContent}\ncomplement(R, A) = ${cStr}`;
    resultEl.style.whiteSpace = 'pre-wrap';

    // Show the complement viz section and render it
    document.getElementById(`${tabId}-comp-complement-section`).style.display = 'block';
    renderCompGrid(tabId, 'complement');
    // Re-render the highlighted A so the edge markers and shadow tiles appear
    renderCompGrid(tabId, 'highlight');
    // Update button state
    const btn = document.getElementById(`${tabId}-comp-complement-btn`);
    if (btn) { btn.textContent = 'Hide complement'; btn.classList.add('active'); }
  } catch (e) {
    showErr(`${tabId}-comp-error`, 'Could not compute complement: ' + e.message);
  }
}

function setComp(tabId, a, b) {
  document.getElementById(`${tabId}-comp-a-input`).value = a;
  document.getElementById(`${tabId}-comp-b-input`).value = b;
  renderComposition(tabId);
}

function downloadSVG(hostId, filename) {
  const svg = document.getElementById(hostId).querySelector('svg');
  if (!svg) return;
  const blob = new Blob([svg.outerHTML], { type: 'image/svg+xml' });
  const a = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(blob), download: filename
  });
  a.click();
  URL.revokeObjectURL(a.href);
}

// ═══════════════════════════════════════════════════════
//  URL param: ?key={feature}[-{method}]-{input1}[-{input2}]
//    layout-<shape:stride>
//    tv-1-<tv_layout>-<tile>       (method 1: direct TV + Tile)
//    tv-2-<thr_layout>-<val_layout> (method 2: compute via make_layout_tv)
//    composition-<A>-<B>
//  Legacy accepted: tv-<tv_layout>-<tile>  (treated as method 1)
// ═══════════════════════════════════════════════════════

// feature -> { expectedInputs, methods? }
// If `methods` is set, the key must be "<feature>-<method>-<inputs...>".
const FEATURE_SPEC = {
  layout:      { inputs: 1 },
  tv:          { inputs: 2, methods: ['1', '2'] },
  composition: { inputs: 2 },
};

function parseKeyParam() {
  const key = new URLSearchParams(location.search).get('key');
  if (!key) return null;
  const parts = key.split('-');
  if (parts.length < 2) return null;
  const feature = parts[0];
  const spec = FEATURE_SPEC[feature];
  if (!spec) return null;

  let method = null;
  let inputs = parts.slice(1);
  if (spec.methods && spec.methods.includes(parts[1])) {
    method = parts[1];
    inputs = parts.slice(2);
  }
  if (inputs.length !== spec.inputs) return null;
  return { feature, method, inputs };
}

/** Build a shareable URL for the given feature and inputs, then copy to clipboard. */
async function exportURL(btnId, feature, ...inputs) {
  const key = [feature, ...inputs].join('-');
  const url = location.origin + location.pathname +
              '?' + new URLSearchParams({ key }).toString();
  const btn = document.getElementById(btnId);
  const showFeedback = (text) => {
    if (!btn) return;
    const orig = btn.textContent;
    btn.textContent = text;
    setTimeout(() => { btn.textContent = orig; }, 1500);
  };
  try {
    await navigator.clipboard.writeText(url);
    showFeedback('Copied!');
  } catch (e) {
    prompt('Copy this URL:', url);
  }
}

function exportLayout(tabId) {
  const val = document.getElementById(`${tabId}-layout-input`).value;
  exportURL(`${tabId}-layout-export`, 'layout', val);
}

function exportTV(tabId) {
  const thr = document.getElementById(`${tabId}-tv-thr-input`).value.trim();
  const val = document.getElementById(`${tabId}-tv-val-input`).value.trim();
  // If both thr_layout and val_layout are provided, prefer method 2
  // (TV+Tile can be derived from them). Otherwise fall back to method 1.
  if (thr && val) {
    exportURL(`${tabId}-tv-export`, 'tv-2', thr, val);
  } else {
    const tv = document.getElementById(`${tabId}-tv-layout-input`).value;
    const tile = document.getElementById(`${tabId}-tv-tile-input`).value;
    exportURL(`${tabId}-tv-export`, 'tv-1', tv, tile);
  }
}

function exportComp(tabId) {
  const a = document.getElementById(`${tabId}-comp-a-input`).value;
  const b = document.getElementById(`${tabId}-comp-b-input`).value;
  exportURL(`${tabId}-comp-export`, 'composition', a, b);
}

function applyKeyParam(tabId) {
  const parsed = parseKeyParam();
  if (!parsed) return;
  const { feature, method, inputs } = parsed;
  switch (feature) {
    case 'layout':
      document.getElementById(`${tabId}-layout-input`).value = inputs[0];
      switchInnerTab(tabId, 'layout');
      renderLayout(tabId);
      break;
    case 'tv':
      switchInnerTab(tabId, 'tv');
      if (method === '2') {
        // Method 2: thr_layout + val_layout → compute TV + Tile
        document.getElementById(`${tabId}-tv-thr-input`).value = inputs[0];
        document.getElementById(`${tabId}-tv-val-input`).value = inputs[1];
        computeTVFromThrVal(tabId);
      } else {
        // Method 1 (or legacy `tv-<X>-<Y>`): direct TV + Tile
        document.getElementById(`${tabId}-tv-layout-input`).value = inputs[0];
        document.getElementById(`${tabId}-tv-tile-input`).value = inputs[1];
        renderTV(tabId);
      }
      break;
    case 'composition':
      document.getElementById(`${tabId}-comp-a-input`).value = inputs[0];
      document.getElementById(`${tabId}-comp-b-input`).value = inputs[1];
      switchInnerTab(tabId, 'composition');
      renderComposition(tabId);
      break;
  }
}

// Initialize first tab on page load
const firstTabId = addOuterTab();
applyKeyParam(firstTabId);
