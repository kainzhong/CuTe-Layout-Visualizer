// CuTe layout math, parser, and color utilities (pure logic, no DOM)

// ═══════════════════════════════════════════════════════
//  Layout arithmetic  (mirrors CuTe's core operations)
// ═══════════════════════════════════════════════════════

/** Product of all leaf values in a (possibly nested) number/array. */
function product(x) {
  if (typeof x === 'number') return x;
  return x.reduce((a, v) => a * product(v), 1);
}

/** For rank-2 layout shape [s0, s1], return [product(s0), product(s1)]. */
function productEach(shape) {
  return [product(shape[0]), product(shape[1])];
}

/**
 * Unflatten a linear index `idx` into the coordinate space of `shape`.
 * shape may be a scalar or a nested array.
 * Returns a scalar (if shape is scalar) or an array matching shape structure.
 */
function unflatten(idx, shape) {
  if (typeof shape === 'number') return idx;
  const result = [];
  let rem = idx;
  for (let i = 0; i < shape.length; i++) {
    const sz = product(shape[i]);
    result.push(unflatten(rem % sz, shape[i]));
    rem = Math.floor(rem / sz);
  }
  return result;
}

/**
 * Compute the linear index for coordinate `crd` in a layout (shape, stride).
 * crd, shape, stride must have matching structure (scalar or nested array).
 */
function crd2idx(crd, shape, stride) {
  if (typeof shape === 'number') return crd * stride;
  let r = 0;
  for (let i = 0; i < shape.length; i++) r += crd2idx(crd[i], shape[i], stride[i]);
  return r;
}

/**
 * Evaluate a rank-2 layout at grid position (m, n).
 * shape and stride are arrays [mode0, mode1] where each mode can be nested.
 */
function layoutAt(shape, stride, m, n) {
  const c0 = unflatten(m, shape[0]);
  const c1 = unflatten(n, shape[1]);
  return crd2idx(c0, shape[0], stride[0]) + crd2idx(c1, shape[1], stride[1]);
}

/** Evaluate a rank-2 layout at a flat 1-D index (column-major unflattening). */
function evalLayoutFlat(layout, flatIdx) {
  const M = product(layout.shape[0]);
  return layoutAt(layout.shape, layout.stride, flatIdx % M, Math.floor(flatIdx / M));
}

/** Evaluate a single mode of a layout at a 1-D coordinate. */
function evalModeAt(modeShape, modeStride, idx) {
  return crd2idx(unflatten(idx, modeShape), modeShape, modeStride);
}

/**
 * Auto column-major stride starting at `base`.
 * Returns [computed_stride, next_base].
 */
function autoStrideHelper(shape, base) {
  if (typeof shape === 'number') return [base, base * shape];
  const strides = [];
  let cur = base;
  for (const s of shape) {
    const [st, next] = autoStrideHelper(s, cur);
    strides.push(st);
    cur = next;
  }
  return [strides, cur];
}
function autoStride(shape) {
  return autoStrideHelper(shape, 1)[0];
}

// ═══════════════════════════════════════════════════════
//  Parser — handles CuTe notation like (10,10):(1,10)
//           and nested ((2,2),(2,2)):((1,4),(2,8))
// ═══════════════════════════════════════════════════════

function parseValue(str) {
  str = str.trim();
  if (!str) throw new Error('Empty value');
  if (str[0] === '(') {
    let depth = 0, start = 1;
    const els = [];
    for (let i = 0; i < str.length; i++) {
      if (str[i] === '(') depth++;
      else if (str[i] === ')') {
        depth--;
        if (depth === 0) {
          const sub = str.slice(start, i).trim();
          if (sub) els.push(parseValue(sub));
          break;
        }
      } else if (str[i] === ',' && depth === 1) {
        els.push(parseValue(str.slice(start, i)));
        start = i + 1;
      }
    }
    // Unwrap single-element parens: (10) -> 10
    return els.length === 1 ? els[0] : els;
  }
  const n = parseInt(str, 10);
  if (isNaN(n)) throw new Error(`Not a number: "${str}"`);
  return n;
}

/** Find the colon that separates shape from stride at depth 0. */
function topLevelColon(str) {
  let depth = 0;
  for (let i = 0; i < str.length; i++) {
    if (str[i] === '(') depth++;
    else if (str[i] === ')') depth--;
    else if (str[i] === ':' && depth === 0) return i;
  }
  return -1;
}

/**
 * Parse a CuTe layout string into {shape, stride}.
 * Always returns shape as a 2-element array [mode0, mode1].
 *
 * Accepted formats:
 *   shape only:    3  |  (M,N)  |  ((a,b),c)
 *   shape:stride:  3:4  |  (M,N):(s0,s1)  |  ((a,b),c):((sa,sb),sc)
 * The ":" must be at the top level, not inside parens.
 */
function parseLayout(str) {
  str = str.trim();
  if (!str) throw new Error('Empty layout string');

  const ci = topLevelColon(str);

  // Reject colons buried inside parens — e.g. (3:4) — which this parser
  // cannot handle.
  if (ci === -1 && str.includes(':')) {
    throw new Error(
      `Unexpected ":" inside parentheses. This parser expects either:\n` +
      `  shape only:   3  or  (M,N)  or  ((a,b),c)\n` +
      `  shape:stride: 3:4  or  (M,N):(s0,s1)\n` +
      `The ":" must be at the top level, not inside parens like (3:4). ` +
      `Write 3:4 instead.`);
  }

  let shape, stride;
  if (ci === -1) {
    shape = parseValue(str);
    stride = null; // auto
  } else {
    shape = parseValue(str.slice(0, ci).trim());
    stride = parseValue(str.slice(ci + 1).trim());
  }

  // Normalise to rank-2 [mode0, mode1]
  if (typeof shape === 'number') {
    // 1-D scalar -> column vector
    shape = [shape, 1];
    stride = stride === null ? [1, 0] : [stride, 0];
  } else if (!Array.isArray(shape[0]) && typeof shape[0] !== 'number') {
    throw new Error('Unexpected shape format');
  } else {
    if (stride === null) stride = autoStride(shape);
    // If shape has > 2 modes, treat as 1-D for now (wrap as [shape, 1])
    if (shape.length > 2) {
      shape = [shape, 1];
      stride = [stride, 0];
    }
  }

  return { shape, stride };
}

// ═══════════════════════════════════════════════════════
//  Colors  (exactly match the LaTeX tikz_color_* functions)
// ═══════════════════════════════════════════════════════

const BW_COLORS = [
  '#ffffff', '#999999', '#cccccc', '#666666',
  '#e6e6e6', '#808080', '#b3b3b3', '#4d4d4d',
];

const TV_COLORS = [
  'rgb(175,175,255)', 'rgb(175,255,175)',
  'rgb(255,255,175)', 'rgb(255,175,175)',
  'rgb(210,210,255)', 'rgb(210,255,210)',
  'rgb(255,255,210)', 'rgb(255,210,210)',
];

const TV_COLORS_HEX = [
  '#afafff','#afffaf','#ffffaf','#ffafaf',
  '#d2d2ff','#d2ffd2','#ffffd2','#ffd2d2',
];

const HIGHLIGHT_COLORS = [
  '#60a5fa', '#34d399', '#fbbf24', '#f87171',
  '#a78bfa', '#2dd4bf', '#fb923c', '#e879f9',
];

function colorBW(idx) { return BW_COLORS[((idx % 8) + 8) % 8]; }
function colorTV(tid) { return TV_COLORS[((tid % 8) + 8) % 8]; }
function colorHighlight(idx) { return HIGHLIGHT_COLORS[((idx % 8) + 8) % 8]; }

/** Pick black or white text based on background luminance. */
function textOnBG(hex) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  const lum = (0.299*r + 0.587*g + 0.114*b) / 255;
  return lum > 0.45 ? '#000' : '#fff';
}

/** Convert rgb(...) string to hex. */
function rgbToHex(str) {
  const m = str.match(/\d+/g);
  if (!m) return '#ffffff';
  return '#' + m.slice(0,3).map(n => parseInt(n).toString(16).padStart(2,'0')).join('');
}
