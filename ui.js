// UI: SVG generation, DOM manipulation, tab management (shared infrastructure).
// Per-tab rendering, state, and HTML templates live under tabs/*.js.
// Depends on all functions from cute.js (loaded first).

// ═══════════════════════════════════════════════════════
//  SVG generation
// ═══════════════════════════════════════════════════════

const MAX_CELLS = 131072;
const BASE_CS = 56;

function cellSize(M, N) {
  return Math.max(12, Math.min(BASE_CS, Math.floor(1200 / Math.max(M, N, 1))));
}

/** Pick CSS for the SVG. Default state fits by longest side so the whole viz
 *  is visible without scrolling. Zoomed state always fits the viz-box's WIDTH
 *  and lets the container scroll vertically if the viz is taller than fits —
 *  matches the natural "make this thing fill the column and let me scroll
 *  down" expectation, and keeps horizontal scrollbars out of grid layouts. */
function svgFitStyle(W, H, zoomed) {
  if (zoomed) {
    return 'width:100%;height:auto';
  }
  // Default: fit by longest side — always fully visible
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
 *  'coord' shows (m,n).
 *  cellTextColor: if set, overrides the default luminance-based cell text color
 *                 (used to draw tiler cells with an accent color).
 *  allCellsEdgeColor: if set, draws a colored border around every cell
 *                    (used to mark all cells as "anchors" in complement viz). */
/** Draw a basis-strided ("coordinate") layout: every cell shows the COORDINATE
 *  the layout maps it to, rather than a 1-D offset. Colour keys off output axis
 *  0, so which logical mode feeds which output dimension is visible at a glance
 *  — identity gives horizontal bands, a transpose gives vertical ones.
 *
 *  `origin` is the iterator's value (the `(0,0) o ...` prefix): the constant
 *  that slicing accumulates, added to every cell. */
function buildBasisLayoutSVG(shape, stride, ndim, origin, modes) {
  modes = toModeSet(modes);
  const [M, N] = productEach(shape);
  if (M * N > MAX_CELLS)
    return errSVG(`Grid too large: ${M}\u00d7${N} = ${M*N} cells (max ${MAX_CELLS})`);
  if (M === 0 || N === 0) return errSVG(`Empty grid: ${M}\u00d7${N}`);

  const cs = cellSize(M, N);
  const margin = cs;
  const W = margin + N * cs;
  const H = margin + M * cs;
  const axisFs = Math.max(8, Math.min(14, Math.floor(cs * 0.38)));
  let body = '';
  for (let n = 0; n < N; n++) {
    body += `<text x="${margin + (n + 0.5) * cs}" y="${margin * 0.55}" text-anchor="middle"
      dominant-baseline="middle" fill="#555" font-size="${axisFs}" font-family="monospace">${n}</text>`;
  }
  for (let m = 0; m < M; m++) {
    body += `<text x="${margin * 0.5}" y="${margin + (m + 0.5) * cs}" text-anchor="middle"
      dominant-baseline="middle" fill="#555" font-size="${axisFs}" font-family="monospace">${m}</text>`;
  }
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      const out = basisAt(shape, stride, m, n, ndim, origin);
      const flatI = m + n * M;
      // Same prefixing rule as buildCellLines: bare when one mode is on, tagged
      // when several are, because here `value` and `coord` are BOTH tuples and
      // would otherwise be indistinguishable.
      const val = `(${out.join(',')})`;
      const wanted = modes.size === 0 ? new Set(['value']) : modes;
      const lines = [];
      if (wanted.size === 1) {
        if (wanted.has('value')) lines.push(val);
        else if (wanted.has('index')) lines.push(String(flatI));
        else if (wanted.has('coord')) lines.push(`(${m},${n})`);
      } else {
        if (wanted.has('value')) lines.push(`val=${val}`);
        if (wanted.has('index')) lines.push(`idx=${flatI}`);
        if (wanted.has('coord')) lines.push(`crd=(${m},${n})`);
      }
      const bg = colorBW(out[0]);
      const fg = textOnBG(bg);
      const x = margin + n * cs, y = margin + m * cs;
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

function buildLayoutSVG(shape, stride, mode, cellTextColor, allCellsEdgeColor) {
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
      const fg  = cellTextColor || textOnBG(bg);
      const x = margin + n * cs;
      const y = margin + m * cs;
      body += `<rect x="${x}" y="${y}" width="${cs}" height="${cs}"
        fill="${bg}" stroke="#ccc" stroke-width="0.5"/>`;
      body += cellTextSVG(x + cs/2, y + cs/2, lines, cs, fg);
      if (allCellsEdgeColor) {
        body += `<rect x="${x}" y="${y}" width="${cs}" height="${cs}"
          fill="none" stroke="${allCellsEdgeColor}" stroke-width="3"/>`;
      }
    }
  }

  return `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="${svgFitStyle(W, H)}">
    <rect width="${W}" height="${H}" fill="white"/>
    ${body}
  </svg>`;
}

/** Build axis-label and grid SVG for a TV layout. */
function buildTVSVG(tvShape, tvStride, tileShape, tileStride, showOffset, underlyingLayout, highlightTid, labelMode) {
  // labelMode === 'value' → show the TV layout's output at each cell, i.e. the
  // col-major flat position (m + n*M). Anything else (including '' / undefined)
  // → no cell text. Thread id remains encoded by cell color.
  const showIdx = labelMode === 'value';
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
        const { tid, vid } = entries[0];
        const bg = dimmed ? '#f0f0f0' : colorTV(tid);
        const fg = dimmed ? '#bbb' : '#111';
        body += `<rect x="${x}" y="${y}" width="${cs}" height="${cs}"
          fill="${bg}" stroke="#ccc" stroke-width="0.5"/>`;
        // T/V labels always visible; 'value' picker adds the col-major flat
        // position (= the TV layout's output at this cell) on top.
        const lines = [`T${tid}`, `V${vid}`];
        if (showIdx) lines.push(String(m + n * M));
        body += cellTextSVG(x + cs/2, y + cs/2, lines, cs, fg);
      } else {
        const bg = dimmed ? '#f0f0f0' : colorTV(entries[0].tid);
        const fg = dimmed ? '#bbb' : '#111';
        const stroke = dimmed ? '#ccc' : '#e53e3e';
        const sw = dimmed ? 0.5 : 1.5;
        body += `<rect x="${x}" y="${y}" width="${cs}" height="${cs}"
          fill="${bg}" stroke="${stroke}" stroke-width="${sw}"/>`;
        const lines = entries.map(e => `T${e.tid}/V${e.vid}`);
        if (showIdx) lines.push(String(m + n * M));
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

/** Render a 2-D layout with a caller-supplied per-cell style resolver.
 *  `cellFn(m, n, offset)` returns `{ bg?, fg?, stroke?, sw?, text? }`:
 *    text === null     → no cell text
 *    text === [strs]   → use these strings
 *    text === undefined → default label from `modes` (value / index / coord)
 *  `opts.pixelScale` (default 1) forces the SVG to render at `pixelScale ×`
 *  its natural pixel size (square cells preserved). Values > 1 bypass the
 *  default `width:100%` fit, which means the containing `.viz-box` may need
 *  to scroll horizontally when the tensor is wider than the container —
 *  this is the only way to get both "bigger cells" and "square cells" since
 *  `width:100%` fundamentally couples cell size to container width.
 *  Used by the Copy Atom visualizations. */
function buildColoredLayoutSVG(shape, stride, modes, cellFn, opts) {
  opts = opts || {};
  modes = toModeSet(modes);
  const [M, N] = productEach(shape);
  if (M * N > MAX_CELLS)
    return errSVG(`Grid too large: ${M}×${N} = ${M*N} cells (max ${MAX_CELLS})`);
  if (M === 0 || N === 0)
    return errSVG(`Empty grid: ${M}×${N}`);

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
      const offset = layoutAt(shape, stride, m, n);
      const flatI = m + n * M;
      const style = cellFn(m, n, offset, flatI) || {};
      const bg = style.bg || '#f0f0f0';
      const stroke = style.stroke || '#ccc';
      const sw = (style.sw != null) ? style.sw : 0.5;
      const x = margin + n * cs;
      const y = margin + m * cs;
      body += `<rect x="${x}" y="${y}" width="${cs}" height="${cs}"
        fill="${bg}" stroke="${stroke}" stroke-width="${sw}"/>`;
      let lines;
      if (style.text === null)            lines = null;
      else if (Array.isArray(style.text)) lines = style.text;
      else                                 lines = buildCellLines(modes, offset, flatI, `(${m},${n})`);
      if (lines && lines.length > 0) {
        const fg = style.fg || textOnRGB(bg);
        body += cellTextSVG(x + cs/2, y + cs/2, lines, cs, fg);
      }
    }
  }

  const scale = opts.pixelScale || 1;
  const styleStr = scale > 1
    ? `width:${Math.round(W * scale)}px;height:${Math.round(H * scale)}px;max-width:none;max-height:none;display:block;margin:0 auto`
    : svgFitStyle(W, H);

  return `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="${styleStr}">
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

/** Like buildLayoutSVG, but the background color comes from a per-cell
 *  tile index (via `tileIdxFn(m, n)` returning an integer or null).
 *  Cells with null/undefined tileIdx are rendered in grey.
 *
 *  `selection` (optional) marks "first tile" picks on top of the coloring:
 *    - edgeCells: Set<flatPos>  — draw amber 3px borders on these cells
 *    - rows:      Set<m>        — highlight these row axis labels
 *    - cols:      Set<n>        — highlight these column axis labels
 */
function buildTiledLayoutSVG(shape, stride, mode, tileIdxFn, selection) {
  const modes = toModeSet(mode);
  const [M, N] = productEach(shape);
  if (M * N > MAX_CELLS)
    return errSVG(`Grid too large: ${M}\u00d7${N} = ${M*N} cells (max ${MAX_CELLS})`);
  if (M === 0 || N === 0)
    return errSVG(`Empty grid: ${M}\u00d7${N}`);

  selection = selection || {};
  const edgeCells = selection.edgeCells || null;
  const highlightRows = selection.rows || null;
  const highlightCols = selection.cols || null;
  const edgeColor = selection.edgeColor || '#f59e0b';
  const rowColor  = selection.rowColor  || '#f59e0b';
  const colColor  = selection.colColor  || '#f59e0b';

  const cs = cellSize(M, N);
  const margin = cs;
  const W = margin + N * cs;
  const H = margin + M * cs;
  const axisFs = Math.max(8, Math.min(14, Math.floor(cs * 0.38)));

  let body = '';

  // Column labels (top axis)
  for (let n = 0; n < N; n++) {
    const cx = margin + (n + 0.5) * cs;
    const hl = highlightCols && highlightCols.has(n);
    const fill = hl ? colColor : '#555';
    const weight = hl ? 'bold' : 'normal';
    body += `<text x="${cx}" y="${margin * 0.55}" text-anchor="middle" dominant-baseline="middle"
      fill="${fill}" font-size="${axisFs}" font-weight="${weight}" font-family="monospace">${n}</text>`;
  }
  // Row labels (left axis)
  for (let m = 0; m < M; m++) {
    const cy = margin + (m + 0.5) * cs;
    const hl = highlightRows && highlightRows.has(m);
    const fill = hl ? rowColor : '#555';
    const weight = hl ? 'bold' : 'normal';
    body += `<text x="${margin * 0.5}" y="${cy}" text-anchor="middle" dominant-baseline="middle"
      fill="${fill}" font-size="${axisFs}" font-weight="${weight}" font-family="monospace">${m}</text>`;
  }

  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      const idx = layoutAt(shape, stride, m, n);
      const flatI = m + n * M;
      const lines = buildCellLines(modes, idx, flatI, `(${m},${n})`);
      const tileIdx = tileIdxFn(m, n);
      const inTile = tileIdx !== null && tileIdx !== undefined;
      const bg = inTile ? colorHighlight(tileIdx) : '#e8e8e8';
      const fg = inTile ? textOnBG(bg) : '#bbb';
      const x = margin + n * cs;
      const y = margin + m * cs;
      body += `<rect x="${x}" y="${y}" width="${cs}" height="${cs}"
        fill="${bg}" stroke="#ccc" stroke-width="0.5"/>`;
      body += cellTextSVG(x + cs/2, y + cs/2, lines, cs, fg);
      if (edgeCells && edgeCells.has(flatI)) {
        body += `<rect x="${x}" y="${y}" width="${cs}" height="${cs}"
          fill="none" stroke="${edgeColor}" stroke-width="3"/>`;
      }
    }
  }

  return `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="${svgFitStyle(W, H)}">
    <rect width="${W}" height="${H}" fill="white"/>
    ${body}
  </svg>`;
}

// ═══════════════════════════════════════════════════════
//  Multi-tab management
// ═══════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════
//  Render shortcut (Cmd/Ctrl + Enter)
// ═══════════════════════════════════════════════════════

/** Inner-tab name -> its render function's global name. The `data-tab`
 *  attribute on each `.tab` div is the key, so adding a tab means adding one
 *  entry here as well as to `modeIndex` in switchInnerTab. */
const TAB_RENDER_FN = {
  layout:             'renderLayout',
  tv:                 'renderTV',
  swizzle:            'renderSwizzle',
  composition:        'renderComposition',
  complement:         'renderComplementFeature',
  divide:             'renderLogicalDivide',
  zipped:             'renderZippedDivide',
  local_tile:         'renderLocalTile',
  product:            'renderLogicalProduct',
  zipped_product:     'renderZippedProduct',
  blocked_product:    'renderBlockedProduct',
  raked_product:      'renderRakedProduct',
  make_copy_atom:     'renderMakeCopyAtom',
  make_tiled_copy:    'renderMakeTiledCopy',
  make_tiled_copy_tv: 'renderMakeTiledCopyTv',
  make_tiled_tma_atom: 'renderMakeTiledTmaAtom',
};

/** True on Apple platforms, where the modifier is ⌘ rather than Ctrl. */
function isMacPlatform() {
  const p = (navigator.userAgentData && navigator.userAgentData.platform) ||
            navigator.platform || navigator.userAgent || '';
  return /mac|iphone|ipad|ipod/i.test(p);
}

/** Plain-text label for the render shortcut, matching the user's platform. */
function renderShortcutLabel() {
  return isMacPlatform() ? '\u2318\u21A9' : 'Ctrl+\u21A9';
}

/** The same shortcut as <kbd> chips, for embedding in a button's label. */
function renderShortcutKeys() {
  const mod = isMacPlatform() ? '\u2318' : 'Ctrl';
  return `<kbd>${mod}</kbd><kbd>\u21A9</kbd>`;
}

/** Run the render function of whichever inner tab is currently visible in the
 *  active outer tab. Returns false when there is nothing to render. */
function renderActiveInnerTab() {
  if (!activeOuterTabId) return false;
  const panel = document.getElementById(`${activeOuterTabId}-panel`);
  if (!panel) return false;
  const tab = panel.querySelector('.tab.active');
  const name = tab && tab.getAttribute('data-tab');
  const fn = name && window[TAB_RENDER_FN[name]];
  if (typeof fn !== 'function') return false;
  fn(activeOuterTabId);
  return true;
}

// Cmd+Enter (mac) / Ctrl+Enter renders without scrolling down to the button.
// preventDefault matters inside the multi-line tiler <textarea>s, where Enter
// would otherwise insert a newline before we ever see it.
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Enter' || !(e.metaKey || e.ctrlKey) || e.altKey || e.shiftKey) return;
  e.preventDefault();
  renderActiveInnerTab();
});

/** Fold the keyboard hint into every "Render" button's own label in `root`, so
 *  it reads `Render (or press ⌘↵)`. Done here, once per new panel, so tabs
 *  don't each have to remember it — and so the modifier follows the user's
 *  platform. Buttons whose label is a variant ("Render Inverse", "Render
 *  complement") are toggles, not the tab's main render, and are skipped.
 *  `data-kbd-hint` makes this idempotent if a panel is ever re-decorated. */
function attachRenderHints(root) {
  root.querySelectorAll('button').forEach(btn => {
    if (btn.hasAttribute('data-kbd-hint') || btn.textContent.trim() !== 'Render') return;
    btn.setAttribute('data-kbd-hint', '');
    btn.innerHTML =
      `Render <span class="btn-kbd-hint">(or press ${renderShortcutKeys()})</span>`;
  });
}

let tabCounter = 0;
let activeOuterTabId = null;

function generateTabContent(id) {
  return `
  <div class="container">
    <div class="tab-nav" data-scope="basics">
      <div class="tab-scope-bar">
        <div class="tab-scope-btn active" data-scope="basics" onclick="switchTabGroup('${id}', 'basics')">
          <span class="tab-scope-icon">\u2756</span>Basics
          <span class="tab-scope-count">3</span>
        </div>
        <div class="tab-scope-btn" data-scope="operations" onclick="switchTabGroup('${id}', 'operations')">
          <span class="tab-scope-icon">\u25A3</span>Layout Operations
          <span class="tab-scope-count">9</span>
        </div>
        <div class="tab-scope-btn" data-scope="copy" onclick="switchTabGroup('${id}', 'copy')">
          <span class="tab-scope-icon">⇄</span>Copy
          <span class="tab-scope-count">4</span>
        </div>
      </div>
    <div class="tab-bar" data-scope="basics">
      <div data-tab="layout" class="tab active" data-scope="basics" onclick="switchInnerTab('${id}', 'layout')">Layout</div>
      <div data-tab="tv" class="tab" data-scope="basics" onclick="switchInnerTab('${id}', 'tv')">TV Layout</div>
      <div data-tab="swizzle" class="tab" data-scope="basics" onclick="switchInnerTab('${id}', 'swizzle')">Swizzle</div>
      <div data-tab="composition" class="tab" data-scope="operations" onclick="switchInnerTab('${id}', 'composition')">Composition</div>
      <div data-tab="complement" class="tab" data-scope="operations" onclick="switchInnerTab('${id}', 'complement')">Complement</div>
      <div data-tab="divide" class="tab" data-scope="operations" onclick="switchInnerTab('${id}', 'divide')">Logical Divide</div>
      <div data-tab="zipped" class="tab" data-scope="operations" onclick="switchInnerTab('${id}', 'zipped')">Zipped / Tiled / Flat Divide</div>
      <div data-tab="local_tile" class="tab" data-scope="operations" onclick="switchInnerTab('${id}', 'local_tile')">Local Tile</div>
      <div data-tab="product" class="tab" data-scope="operations" onclick="switchInnerTab('${id}', 'product')">Logical Product</div>
      <div data-tab="zipped_product" class="tab" data-scope="operations" onclick="switchInnerTab('${id}', 'zipped_product')">Zipped / Tiled / Flat Product</div>
      <div data-tab="blocked_product" class="tab" data-scope="operations" onclick="switchInnerTab('${id}', 'blocked_product')">Blocked Product</div>
      <div data-tab="raked_product" class="tab" data-scope="operations" onclick="switchInnerTab('${id}', 'raked_product')">Raked Product</div>
      <div data-tab="make_copy_atom" class="tab" data-scope="copy" onclick="switchInnerTab('${id}', 'make_copy_atom')">make_copy_atom</div>
      <div data-tab="make_tiled_copy" class="tab" data-scope="copy" onclick="switchInnerTab('${id}', 'make_tiled_copy')">make_tiled_copy</div>
      <div data-tab="make_tiled_copy_tv" class="tab" data-scope="copy" onclick="switchInnerTab('${id}', 'make_tiled_copy_tv')">make_tiled_copy_tv</div>
      <div data-tab="make_tiled_tma_atom" class="tab" data-scope="copy" onclick="switchInnerTab('${id}', 'make_tiled_tma_atom')">make_tiled_tma_atom</div>
    </div>
    </div>
    ${generateLayoutTabContent(id)}
    ${generateTVTabContent(id)}
    ${generateSwizzleTabContent(id)}
    ${generateCompositionTabContent(id)}
    ${generateComplementTabContent(id)}
    ${generateDivideTabContent(id)}
    ${generateZippedDivideTabContent(id)}
    ${generateLocalTileTabContent(id)}
    ${generateLogicalProductTabContent(id)}
    ${generateZippedProductTabContent(id)}
    ${generateBlockedProductTabContent(id)}
    ${generateRakedProductTabContent(id)}
    ${generateMakeCopyAtomTabContent(id)}
    ${generateMakeTiledCopyTabContent(id)}
    ${generateMakeTiledCopyTvTabContent(id)}
    ${generateMakeTiledTmaAtomTabContent(id)}
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

  attachVizCollapsibles(panel);
  attachVizFullscreenButtons(panel);
  attachRenderHints(panel);
  initCopyPanes(id);

  switchOuterTab(id);
  renderAllTabs(id, 'layout');
  return id;
}

/** Inject a collapse/expand chevron into every visualization header inside
 *  `root`, so users can fold any viz they're not looking at. Handles both
 *  layout families: `.comp-viz-item > .comp-viz-header` (multi-viz grid tabs
 *  like Composition, Divide, Product, Copy) and `.visualization > .viz-header`
 *  (single-viz tabs like Layout, TV Layout). Collapsed state is carried on
 *  the item element via `.collapsed`, and CSS hides the body content — so the
 *  state survives in-place re-renders that only update the SVG contents. */
function attachVizCollapsibles(root) {
  const items = [
    ...root.querySelectorAll('.comp-viz-item'),
    ...root.querySelectorAll('.visualization'),
  ];
  items.forEach(item => {
    const header = item.querySelector('.comp-viz-header, .viz-header');
    if (!header || header.querySelector('.viz-collapse-btn')) return;  // idempotent
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'viz-collapse-btn';
    btn.title = 'Collapse / expand';
    // Respect a `.collapsed` class already on the item, so a tab can ship a viz
    // folded (reference material you don't want on every render) without the
    // chevron pointing the wrong way.
    btn.textContent = item.classList.contains('collapsed') ? '▸' : '▾';
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const nowCollapsed = item.classList.toggle('collapsed');
      btn.textContent = nowCollapsed ? '▸' : '▾';
    });
    header.insertBefore(btn, header.firstChild);
  });
}

/** Inject a "Fullscreen" button into every viz inside `root`. The button is
 *  placed at the top-right corner of the viz-box itself (overlaid on the
 *  diagram) so it's always visually paired with the visualization it controls.
 *  Clicking it opens the SVG in a viewport-filling overlay. Idempotent. */
function attachVizFullscreenButtons(root) {
  const items = [
    ...root.querySelectorAll('.comp-viz-item'),
    ...root.querySelectorAll('.visualization'),
  ];
  items.forEach(item => {
    // EVERY viz-box in the item, not just the first: the Copy tabs put SRC and
    // DST side by side inside ONE `.comp-viz-item`, so taking only the first
    // left the DST pane without a button.
    item.querySelectorAll('.viz-box').forEach(vizBox => {
      if (vizBox.querySelector(':scope > .viz-fullscreen-btn')) return;  // idempotent
      // Every viz puts its SVG inside a div with an id (the same id toggleZoom /
      // downloadSVG already key off). Find it so we can pass it to viewFullscreen.
      const host = vizBox.querySelector('div[id]');
      if (!host || !host.id) return;

      // Make the viz-box a positioning ancestor so the absolutely-positioned
      // button anchors to its top-right corner. Done inline so we don't have to
      // touch style.css and so it's idempotent.
      if (!vizBox.style.position) vizBox.style.position = 'relative';

      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'viz-fullscreen-btn';
      btn.title = 'Open in fullscreen overlay (Esc to close)';
      btn.textContent = '⛶';
      btn.style.cssText =
        'position:absolute;top:8px;right:8px;z-index:2;' +
        'font-size:14px;line-height:1;padding:5px 8px;' +
        'background:rgba(17,24,39,0.78);color:white;' +
        'border:none;border-radius:4px;cursor:pointer;' +
        'opacity:0.55;transition:opacity 0.15s';
      btn.addEventListener('mouseenter', () => { btn.style.opacity = '1'; });
      btn.addEventListener('mouseleave', () => { btn.style.opacity = '0.55'; });
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        viewFullscreen(host.id);
      });
      vizBox.appendChild(btn);
    });
  });
}

/** Open the SVG inside `hostId` in a viewport-filling overlay. The SVG is
 *  scaled so its LONGEST axis matches the corresponding viewport dimension,
 *  which keeps cells as large as possible while letting the short axis scroll
 *  if the SVG is more lopsided than the viewport. Click the close button,
 *  click the empty background, or hit Esc to dismiss. */
function viewFullscreen(hostId) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const origSvg = host.querySelector('svg');
  if (!origSvg) return;

  closeFullscreen();  // tear down any prior overlay

  const overlay = document.createElement('div');
  overlay.id = 'viz-fullscreen-overlay';
  overlay.style.cssText =
    'position:fixed;inset:0;z-index:9999;background:#fff;' +
    'overflow:auto;padding:48px 16px 16px';
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) closeFullscreen();
  });

  const cloned = origSvg.cloneNode(true);
  const vb = (cloned.getAttribute('viewBox') || '0 0 100 100').split(/\s+/).map(Number);
  const vbW = vb[2] || 1, vbH = vb[3] || 1;
  const vw = window.innerWidth, vh = window.innerHeight;
  // Compare aspect ratios. If SVG is wider-than-tall relative to the viewport,
  // fit by height (so width overflows -> horizontal scroll, cells as big as
  // possible). Otherwise fit by width (vertical scroll on the tall axis).
  if (vbW / vbH >= vw / vh) {
    // Wide SVG -> fit by viewport height; allow horizontal scroll
    cloned.setAttribute('style',
      'height:calc(100vh - 64px);width:auto;display:block;max-width:none;max-height:none');
  } else {
    cloned.setAttribute('style',
      'width:calc(100vw - 32px);height:auto;display:block;max-width:none;max-height:none');
  }
  overlay.appendChild(cloned);

  const closeBtn = document.createElement('button');
  closeBtn.type = 'button';
  closeBtn.textContent = '× Close (Esc)';
  closeBtn.style.cssText =
    'position:fixed;top:12px;right:12px;z-index:10000;' +
    'font-size:13px;padding:6px 12px;background:#111827;color:white;' +
    'border:none;border-radius:6px;cursor:pointer;' +
    'box-shadow:0 2px 8px rgba(0,0,0,0.25)';
  closeBtn.addEventListener('click', closeFullscreen);
  overlay.appendChild(closeBtn);

  document.body.appendChild(overlay);
  document.addEventListener('keydown', vizFullscreenEscHandler);
}

function vizFullscreenEscHandler(e) {
  if (e.key === 'Escape') closeFullscreen();
}

function closeFullscreen() {
  document.removeEventListener('keydown', vizFullscreenEscHandler);
  const overlay = document.getElementById('viz-fullscreen-overlay');
  if (overlay) overlay.remove();
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
  const modeIndex = { layout: 0, tv: 1, swizzle: 2, composition: 3, complement: 4, divide: 5, zipped: 6, local_tile: 7, product: 8, zipped_product: 9, blocked_product: 10, raked_product: 11, make_copy_atom: 12, make_tiled_copy: 13, make_tiled_copy_tv: 14, make_tiled_tma_atom: 15 };
  const activeTab = tabs[modeIndex[mode]];
  activeTab.classList.add('active');
  document.getElementById(`${tabId}-tab-${mode}`).classList.add('active');

  // Keep the scope selector in sync with whichever tab is now active, so
  // opening a tab from a URL also flips the scope buttons / visible row.
  const scope = activeTab.getAttribute('data-scope') || 'basics';
  const tabNav = panel.querySelector('.tab-nav');
  const tabBar = panel.querySelector('.tab-bar');
  if (tabNav) tabNav.setAttribute('data-scope', scope);
  if (tabBar) tabBar.setAttribute('data-scope', scope);
  panel.querySelectorAll('.tab-scope-btn').forEach(b => {
    b.classList.toggle('active', b.getAttribute('data-scope') === scope);
  });
}

function switchTabGroup(tabId, scope) {
  const panel = document.getElementById(`${tabId}-panel`);
  const tabNav = panel.querySelector('.tab-nav');
  const tabBar = panel.querySelector('.tab-bar');
  if (tabNav) tabNav.setAttribute('data-scope', scope);
  tabBar.setAttribute('data-scope', scope);
  panel.querySelectorAll('.tab-scope-btn').forEach(b => {
    b.classList.toggle('active', b.getAttribute('data-scope') === scope);
  });
  // If the currently-active tab belongs to the other scope, jump to the
  // first tab of the scope we just switched to so something stays visible.
  const activeTab = tabBar.querySelector('.tab.active');
  if (!activeTab || activeTab.getAttribute('data-scope') !== scope) {
    const firstTab = tabBar.querySelector(`.tab[data-scope="${scope}"]`);
    if (firstTab) firstTab.click();
  }
}

function showErr(id, msg) {
  const el = document.getElementById(id);
  el.textContent = msg;
  el.style.display = msg ? 'block' : 'none';
}

function showWarn(id, msg) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = msg;
  el.style.display = msg ? 'block' : 'none';
}

/** True if the layout string has an outer rank > 2 (e.g. "(2,3,4):(1,2,6)").
 *  The visualizer collapses such inputs to a rank-2 column for display. */
function isHighRankLayout(str) {
  try {
    str = (str || '').trim();
    if (!str) return false;
    const ci = topLevelColon(str);
    const shapePart = ci === -1 ? str : str.slice(0, ci).trim();
    const shape = parseValue(shapePart);
    return Array.isArray(shape) && shape.length > 2;
  } catch (e) {
    return false;
  }
}

/** Check a list of [name, str] input pairs and return names of any high-rank ones. */
function collectHighRank(inputs) {
  const offenders = [];
  for (const [name, str] of inputs) {
    if (isHighRankLayout(str)) offenders.push(name);
  }
  return offenders;
}

// ═══════════════════════════════════════════════════════
//  Layout input component — reusable across tabs.
//
//  ANY tab that accepts a layout/shape string should use `layoutInputField()`
//  to render the input, follow it with `statusDivs()` for the error + warning
//  panels, and call `updateRankWarning()` in its render function to flag
//  rank > 2 inputs. See CLAUDE.md for the full convention.
// ═══════════════════════════════════════════════════════

/** HTML for a layout-aware input field.
 *  opts: { id, label, value?, hint?, textarea?, rows?, placeholder?, oninput? } */
function layoutInputField(opts) {
  const {
    id, label, value = '', hint = '', textarea = false, rows = 2, placeholder = '', oninput = ''
  } = opts;
  const hintSpan = hint
    ? ` <span style="color:#6b7280;font-weight:normal">&mdash; ${hint}</span>`
    : '';
  const phAttr = placeholder ? ` placeholder="${placeholder}"` : '';
  const oiAttr = oninput ? ` oninput="${oninput}"` : '';
  const field = textarea
    ? `<textarea id="${id}" rows="${rows}"${phAttr}${oiAttr}>${value}</textarea>`
    : `<input type="text" id="${id}" value="${value}"${phAttr}${oiAttr}>`;
  return `<div class="form-group">
      <label>${label}${hintSpan}</label>
      ${field}
    </div>`;
}

/** HTML for the standard error + rank-warning div pair. Use `${prefix}` where
 *  your render function calls `showErr('${prefix}-error', ...)` and
 *  `updateRankWarning('${prefix}-warning', ...)`. */
function statusDivs(prefix) {
  return `<div id="${prefix}-error" class="error-msg"></div>
    <div id="${prefix}-warning" class="warning-msg"></div>`;
}

/** Show the rank-warning for a tab if any inputs have rank > 2. */
function updateRankWarning(warnId, inputs) {
  const offenders = collectHighRank(inputs);
  if (offenders.length === 0) {
    showWarn(warnId, '');
  } else {
    showWarn(warnId,
      `Note: ${offenders.join(', ')} has rank > 2. The math is still correct, ` +
      `but the visualization is designed for 1D/2D layouts and may not faithfully ` +
      `show the multi-dimensional structure (it gets flattened to a 2D grid).`);
  }
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

// ═══════════════════════════════════════════════════════
//  Copy SRC / DST panes (shared by the three Copy-scope tabs)
// ═══════════════════════════════════════════════════════

/** Which memory movements each Copy Op can legally perform. Keyed by the same
 *  Op key the Copy tabs use, so the two op tables can't drift apart.
 *
 *  `cpasync.CopyG2SOp` is `cp.async` — GMEM to SMEM, and nothing else; the name
 *  says so and the hardware has no other form. `CopyUniversalOp` is CuTe's
 *  generic `a = b`, so it works between any two spaces a *thread can address*:
 *  global, shared and its own registers. TMEM is deliberately absent — it isn't
 *  thread-addressable at all (tcgen05's Ld/St src layout is stride-0 across all
 *  32 lanes), so reaching it needs a tcgen05 Op rather than a universal copy.
 *  Same-space moves (GMEM->GMEM) are omitted as degenerate. */
const COPY_OP_MOVES = {
  universal: [
    ['GMEM', 'RMEM'], ['RMEM', 'GMEM'],
    ['SMEM', 'RMEM'], ['RMEM', 'SMEM'],
    ['GMEM', 'SMEM'], ['SMEM', 'GMEM'],
  ],
  cpasync: [
    ['GMEM', 'SMEM'],
  ],
  // Bulk tensor (TMA) load. One direction by construction: the "G2S" is in the
  // Op's name, and the store is a different Op entirely.
  tma_g2s: [
    ['GMEM', 'SMEM'],
  ],
};

/** Section-0 control: pick one of the Op's legal memory movements. Options are
 *  rebuilt by `syncCopyMoves` whenever the Op changes, so an illegal pairing
 *  simply cannot be selected. */
function copyMoveField(id, p) {
  return `
            <div class="form-group">
              <label>Memory movement<span style="color:#6b7280;font-weight:normal">&nbsp;&mdash; only the pairings this Op supports are listed</span></label>
              <select id="${id}-${p}-move" onchange="setCopyMove('${id}','${p}')"></select>
            </div>`;
}

/** Repopulate the movement picker for `opKey`, keeping the current selection if
 *  the new Op still allows it, and refresh the pane titles. */
function syncCopyMoves(tabId, p, opKey) {
  const sel = document.getElementById(`${tabId}-${p}-move`);
  if (!sel) return;
  const moves = COPY_OP_MOVES[opKey] || COPY_OP_MOVES.universal;
  const want = sel.value;
  sel.innerHTML = moves.map(([a, b]) => {
    const v = `${a}>${b}`;
    return `<option value="${v}">${a} \u2192 ${b}</option>`;
  }).join('');
  sel.value = moves.some(([a, b]) => `${a}>${b}` === want) ? want : `${moves[0][0]}>${moves[0][1]}`;
  sel.disabled = moves.length === 1;
  updateCopyPaneTitles(tabId, p);
}

/** The selected movement as [src, dst]. */
function copyMove(tabId, p) {
  const sel = document.getElementById(`${tabId}-${p}-move`);
  const v = (sel && sel.value) || 'GMEM>SMEM';
  return v.split('>');
}

function setCopyMove(tabId, p) { updateCopyPaneTitles(tabId, p); }

/** Push the selected movement onto the two pane headers. */
function updateCopyPaneTitles(tabId, p) {
  const [src, dst] = copyMove(tabId, p);
  const a = document.getElementById(`${tabId}-${p}-src-space`);
  const b = document.getElementById(`${tabId}-${p}-dst-space`);
  if (a) a.textContent = src;
  if (b) b.textContent = dst;
}

/** The SRC / DST / BOTH button group. Switching is pure CSS — both SVGs are
 *  always rendered, `data-dir` on the pane container decides what's shown — so
 *  no re-render is needed and the toggle is instant. */
function copyDirButtons(id, p) {
  return `<span class="mode-btn-group" id="${id}-${p}-dir-btns">
      <button class="mode-btn" onclick="setCopyDir('${id}','${p}','src')">SRC</button>
      <button class="mode-btn" onclick="setCopyDir('${id}','${p}','dst')">DST</button>
      <button class="mode-btn active" onclick="setCopyDir('${id}','${p}','both')">BOTH</button>
    </span>`;
}

/** The side-by-side pane pair. Each pane's title is a plain label driven by the
 *  section-0 movement picker — not an independent control, because which spaces
 *  a copy can connect is a property of the Op, not a free choice.
 *
 *  In BOTH mode the two panes are flex siblings at 1fr each, so every SVG's
 *  `width:100%` resolves against half the container — same aspect ratio, half
 *  the box. */
function copyPanes(id, p) {
  const pane = (side) => `
        <div class="copy-pane" data-side="${side}">
          <div class="copy-pane-head">
            <span class="copy-pane-side">${side.toUpperCase()}</span>
            <span class="copy-pane-space" id="${id}-${p}-${side}-space">&mdash;</span>
          </div>
          <div class="viz-box"><div id="${id}-${p}-${side}-svg"></div></div>
        </div>`;
  return `
      <div class="copy-panes" id="${id}-${p}-panes" data-dir="both">
${pane('src')}
        <div class="copy-arrow" aria-hidden="true">&rarr;</div>
${pane('dst')}
      </div>`;
}

/** Render every tab once with its shipped defaults, so switching to a tab shows
 *  a picture instead of an empty box. Safe because every tab's defaults are a
 *  working configuration — the same rule that governs presets (see CLAUDE.md).
 *  The active tab is rendered LAST so its `updateOuterTabLabel` call is the one
 *  that sticks, and each render is isolated so one failure can't block the rest. */
function renderAllTabs(tabId, activeTab) {
  for (const [name, fn] of Object.entries(TAB_RENDER_FN)) {
    if (name === activeTab) continue;
    try { window[fn](tabId); } catch (e) { console.warn(`initial render of "${name}" failed:`, e); }
  }
  const fn = TAB_RENDER_FN[activeTab];
  if (fn) { try { window[fn](tabId); } catch (e) { console.warn(`initial render of "${activeTab}" failed:`, e); } }
}

/** Populate every Copy tab's movement picker and pane titles when a panel is
 *  first built, so the section-0 select and the SRC/DST headers are correct
 *  before the user has pressed Render. Prefixes that don't exist are skipped. */
function initCopyPanes(tabId) {
  for (const p of ['mca', 'mtc', 'mtv', 'tma']) {
    const op = document.getElementById(`${tabId}-${p}-op-input`);
    if (op) syncCopyMoves(tabId, p, op.value);
  }
}

/** Flip which pane(s) are visible. */
function setCopyDir(tabId, p, dir) {
  const panes = document.getElementById(`${tabId}-${p}-panes`);
  if (panes) panes.setAttribute('data-dir', dir);
  const group = document.getElementById(`${tabId}-${p}-dir-btns`);
  if (group) group.querySelectorAll('.mode-btn').forEach(b =>
    b.classList.toggle('active', b.textContent.trim().toLowerCase() === dir));
}

/** Current direction, defaulting to 'both'. */
function copyDir(tabId, p) {
  const panes = document.getElementById(`${tabId}-${p}-panes`);
  return (panes && panes.getAttribute('data-dir')) || 'both';
}

/** Zoom both panes together, so a side-by-side comparison stays comparable. */
function toggleCopyZoom(tabId, p) {
  toggleZoom(`${tabId}-${p}-src-svg`);
  toggleZoom(`${tabId}-${p}-dst-svg`);
}

// ═══════════════════════════════════════════════════════
//  Shared element-type + swizzle helpers (used by the TV,
//  Copy_Atom and make_tiled_copy tabs)
// ═══════════════════════════════════════════════════════

/** Element types exposed in every tensor_dtype dropdown, keyed to bit width. */
const DTYPE_BITS = {
  'int8_t': 8, 'uint8_t': 8,
  'half_t': 16, 'bfloat16_t': 16, 'int16_t': 16, 'uint16_t': 16,
  'float': 32, 'int32_t': 32, 'uint32_t': 32, 'tfloat32_t': 32,
  'double': 64, 'int64_t': 64, 'uint64_t': 64,
  'uint128_t': 128,
};

/** <option> list for a tensor_dtype <select>. Single source of truth so the
 *  tabs can never drift apart on which dtypes they accept. */
function dtypeOptions(selected) {
  return Object.entries(DTYPE_BITS).map(([name, bits]) =>
    `<option value="${name}"${name === selected ? ' selected' : ''}>${name} (${bits})</option>`
  ).join('');
}

/** Apply a CuTe `Swizzle<B, M, S>` to an element offset: the B bits starting at
 *  position M+S of `x` get XOR'd into the B bits at position M — same semantics
 *  as `cute::Swizzle<B, M, S>::apply(x)`. Returns `x` unchanged when `sw` is
 *  null. Matches CuTe's convention of swizzling the layout's *element* index
 *  (not a byte address), so callers convert to bytes afterwards. */
function applySwizzleOffset(x, sw) {
  if (!sw) return x;
  const { B, M, S } = sw;
  const mask = (1 << B) - 1;
  const srcBits = (x >>> (M + S)) & mask;
  return x ^ (srcBits << M);
}

/** Parse a `B, M, S` swizzle spec (commas or whitespace, angle brackets ok).
 *  Returns `{B, M, S}` or null for empty / unparseable input. */
function parseSwizzleSpec(raw) {
  const t = (raw || '').trim();
  if (t === '') return null;
  const m = t.match(/^\s*(?:Sw(?:izzle)?\s*<\s*)?(\d+)\s*[,\s]\s*(\d+)\s*[,\s]\s*(\d+)\s*>?\s*$/i);
  if (!m) return null;
  const B = parseInt(m[1], 10), M = parseInt(m[2], 10), S = parseInt(m[3], 10);
  if (B < 0 || M < 0 || S < 0 || (M + S + B) >= 31) return null;
  return { B, M, S };
}

// ═══════════════════════════════════════════════════════
//  Shared layout-string helpers (used by multiple tabs)
// ═══════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════
//  Download SVG
// ═══════════════════════════════════════════════════════

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
//    complement-<layout>-<cotarget>
//    logical_divide-<A>-<tiler>
//    zipped_divide-<A>-<tiler>
//    logical_product-<A>-<tiler>
//    zipped_product-<A>-<tiler>
//    blocked_product-<A>-<tiler>
//    raked_product-<A>-<tiler>
//    make_copy_atom-<op>-<num_bits>-<dtype>
//    make_tiled_copy-<op>-<bits>-<dtype>-<layout_tv>-<tiler_mn>
//    make_tiled_copy_tv-<op>-<bits>-<dtype>-<thr>-<val>
//    swizzle-<layout>-<swizzle>
//  Legacy accepted: tv-<tv_layout>-<tile>  (treated as method 1)
// ═══════════════════════════════════════════════════════

// feature -> { expectedInputs, methods? }
// If `methods` is set, the key must be "<feature>-<method>-<inputs...>".
const FEATURE_SPEC = {
  layout:         { inputs: 1 },
  tv:             { inputs: 2, methods: ['1', '2'] },
  composition:    { inputs: 2 },
  complement:     { inputs: 2 },
  logical_divide: { inputs: 2 },
  zipped_divide:  { inputs: 2 },
  local_tile:     { inputs: 3 },  // A, tiler, coord
  logical_product: { inputs: 2 },
  zipped_product:  { inputs: 2 },
  blocked_product: { inputs: 2 },
  raked_product:   { inputs: 2 },
  make_copy_atom:     { inputs: 3 },  // op, bits, dtype
  make_tiled_copy:    { inputs: 5 },  // op, bits, dtype, layout_tv, tiler_mn
  make_tiled_copy_tv: { inputs: 5 },  // op, bits, dtype, thr, val
  make_tiled_tma_atom: { inputs: 5 },  // dtype, gmem, swizzle, smem, tiler
  swizzle:         { inputs: 2 },
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
    case 'complement':
      document.getElementById(`${tabId}-cpl-layout-input`).value = inputs[0];
      document.getElementById(`${tabId}-cpl-cotarget-input`).value = inputs[1];
      switchInnerTab(tabId, 'complement');
      renderComplementFeature(tabId);
      break;
    case 'logical_divide':
      document.getElementById(`${tabId}-ld-a-input`).value = inputs[0];
      document.getElementById(`${tabId}-ld-tiler-input`).value = inputs[1];
      switchInnerTab(tabId, 'divide');
      renderLogicalDivide(tabId);
      break;
    case 'zipped_divide':
      document.getElementById(`${tabId}-zd-a-input`).value = inputs[0];
      document.getElementById(`${tabId}-zd-tiler-input`).value = inputs[1];
      switchInnerTab(tabId, 'zipped');
      renderZippedDivide(tabId);
      break;
    case 'local_tile':
      document.getElementById(`${tabId}-lt-a-input`).value     = inputs[0];
      document.getElementById(`${tabId}-lt-tiler-input`).value = inputs[1];
      document.getElementById(`${tabId}-lt-coord-input`).value = inputs[2];
      switchInnerTab(tabId, 'local_tile');
      renderLocalTile(tabId);
      break;
    case 'logical_product':
      document.getElementById(`${tabId}-lp-a-input`).value = inputs[0];
      document.getElementById(`${tabId}-lp-tiler-input`).value = inputs[1];
      switchInnerTab(tabId, 'product');
      renderLogicalProduct(tabId);
      break;
    case 'zipped_product':
      document.getElementById(`${tabId}-zp-a-input`).value = inputs[0];
      document.getElementById(`${tabId}-zp-tiler-input`).value = inputs[1];
      switchInnerTab(tabId, 'zipped_product');
      renderZippedProduct(tabId);
      break;
    case 'blocked_product':
      document.getElementById(`${tabId}-bp-a-input`).value = inputs[0];
      document.getElementById(`${tabId}-bp-tiler-input`).value = inputs[1];
      switchInnerTab(tabId, 'blocked_product');
      renderBlockedProduct(tabId);
      break;
    case 'raked_product':
      document.getElementById(`${tabId}-rp-a-input`).value = inputs[0];
      document.getElementById(`${tabId}-rp-tiler-input`).value = inputs[1];
      switchInnerTab(tabId, 'raked_product');
      renderRakedProduct(tabId);
      break;
    case 'make_copy_atom':
      document.getElementById(`${tabId}-mca-op-input`).value    = inputs[0];
      document.getElementById(`${tabId}-mca-bits-input`).value  = inputs[1];
      document.getElementById(`${tabId}-mca-dtype-input`).value = inputs[2];
      switchInnerTab(tabId, 'make_copy_atom');
      renderMakeCopyAtom(tabId);
      break;
    case 'make_tiled_copy':
      document.getElementById(`${tabId}-mtc-op-input`).value    = inputs[0];
      document.getElementById(`${tabId}-mtc-bits-input`).value  = inputs[1];
      document.getElementById(`${tabId}-mtc-dtype-input`).value = inputs[2];
      document.getElementById(`${tabId}-mtc-tv-input`).value    = inputs[3];
      document.getElementById(`${tabId}-mtc-tiler-input`).value = inputs[4];
      switchInnerTab(tabId, 'make_tiled_copy');
      renderMakeTiledCopy(tabId);
      break;
    case 'make_tiled_copy_tv':
      document.getElementById(`${tabId}-mtv-op-input`).value    = inputs[0];
      document.getElementById(`${tabId}-mtv-bits-input`).value  = inputs[1];
      document.getElementById(`${tabId}-mtv-dtype-input`).value = inputs[2];
      document.getElementById(`${tabId}-mtv-thr-input`).value   = inputs[3];
      document.getElementById(`${tabId}-mtv-val-input`).value   = inputs[4];
      switchInnerTab(tabId, 'make_tiled_copy_tv');
      renderMakeTiledCopyTv(tabId);
      break;
    case 'make_tiled_tma_atom':
      document.getElementById(`${tabId}-tma-dtype-input`).value   = inputs[0];
      document.getElementById(`${tabId}-tma-gmem-input`).value    = inputs[1];
      document.getElementById(`${tabId}-tma-swizzle-input`).value = inputs[2];
      document.getElementById(`${tabId}-tma-smem-input`).value    = inputs[3];
      document.getElementById(`${tabId}-tma-tiler-input`).value   = inputs[4];
      switchInnerTab(tabId, 'make_tiled_tma_atom');
      renderMakeTiledTmaAtom(tabId);
      break;
    case 'swizzle':
      document.getElementById(`${tabId}-sw-layout-input`).value  = inputs[0];
      document.getElementById(`${tabId}-sw-swizzle-input`).value = inputs[1];
      switchInnerTab(tabId, 'swizzle');
      renderSwizzle(tabId);
      break;
  }
}
