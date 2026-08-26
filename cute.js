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

/** A CuTe scaled-basis stride: `k@i` means k units along basis direction i,
 *  i.e. the tuple (0,..,k,..,0). See media/docs/cpp/cute/0z_tma_tensors.md.
 *  Strides built from these make a layout map a coordinate to a COORDINATE
 *  rather than to a 1-D offset — which is what TMA and identity/predication
 *  tensors need. */
function makeBasis(k, axis) { return { basis: true, k, axis }; }
function isBasis(x) { return !!(x && x.basis === true); }

/** Does any stride in this (possibly nested) tuple use a basis element? */
function hasBasisStride(x) {
  if (isBasis(x)) return true;
  return Array.isArray(x) && x.some(hasBasisStride);
}

/** Highest basis axis referenced, so callers know the output rank. */
function basisRank(x) {
  if (isBasis(x)) return x.axis + 1;
  if (Array.isArray(x)) return x.reduce((a, y) => Math.max(a, basisRank(y)), 0);
  return 0;
}

function parseValue(str, allowBasis) {
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
          if (sub) els.push(parseValue(sub, allowBasis));
          break;
        }
      } else if (str[i] === ',' && depth === 1) {
        els.push(parseValue(str.slice(start, i), allowBasis));
        start = i + 1;
      }
    }
    // Unwrap single-element parens: (10) -> 10
    return els.length === 1 ? els[0] : els;
  }
  // `k@i` — a scaled basis element. Only legal in a stride, and only where the
  // caller opted in, so every other tab keeps its "strides are integers"
  // assumption and fails loudly rather than silently producing NaN.
  const at = str.indexOf('@');
  if (at !== -1) {
    if (!allowBasis) {
      throw new Error(
        `Basis stride "${str}" (k@i) is only supported by the Layout tab. ` +
        `Elsewhere a stride must be a plain integer.`);
    }
    if (str.indexOf('@', at + 1) !== -1) {
      throw new Error(
        `Nested basis stride "${str}" (k@i@j) produces a hierarchical coordinate, ` +
        `which this visualization cannot draw. Only flat k@i is supported.`);
    }
    const k = parseInt(str.slice(0, at), 10);
    const axis = parseInt(str.slice(at + 1), 10);
    if (isNaN(k) || isNaN(axis) || axis < 0) {
      throw new Error(`Malformed basis stride "${str}" — expected k@i with integer k and axis i >= 0.`);
    }
    return makeBasis(k, axis);
  }
  const n = parseInt(str, 10);
  if (isNaN(n)) throw new Error(`Not a number: "${str}"`);
  return n;
}

/** Split a `<origin> o <layout>` printout at the top-level `o`, so a CuTe
 *  coordinate-tensor printout can be pasted verbatim. Returns -1 when absent. */
function topLevelCompose(str) {
  let depth = 0;
  for (let i = 0; i < str.length; i++) {
    const c = str[i];
    if (c === '(') depth++;
    else if (c === ')') depth--;
    else if (c === 'o' && depth === 0) {
      const before = i === 0 ? ' ' : str[i - 1];
      const after = i + 1 >= str.length ? ' ' : str[i + 1];
      if (/\s/.test(before) && /\s/.test(after)) return i;
    }
  }
  return -1;
}

/** Evaluate a basis-strided layout: the ordinary inner product, but the
 *  accumulator is a vector. `out` is mutated.
 *
 *  NOTE the name. `crd2crd` was the obvious choice and is already taken by
 *  layout.js's pycute port — and since layout.js loads AFTER cute.js it wins,
 *  silently. Keep new globals here distinct from every name in layout.js. */
function crd2basis(crd, shape, stride, out) {
  if (typeof shape === 'number') {
    if (isBasis(stride)) out[stride.axis] += crd * stride.k;
    else out[0] += crd * stride;      // a plain integer stride lands on axis 0
    return out;
  }
  for (let i = 0; i < shape.length; i++) crd2basis(crd[i], shape[i], stride[i], out);
  return out;
}

/** (m, n) -> output coordinate, for a rank-2 basis layout. */
function basisAt(shape, stride, m, n, ndim, origin) {
  const out = (origin && origin.length === ndim) ? origin.slice() : new Array(ndim).fill(0);
  crd2basis(unflatten(m, shape[0]), shape[0], stride[0], out);
  crd2basis(unflatten(n, shape[1]), shape[1], stride[1], out);
  return out;
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
function parseLayout(str, opts) {
  opts = opts || {};
  const allowBasis = !!opts.basis;
  str = str.trim();
  if (!str) throw new Error('Empty layout string');

  // Optional `<origin> o <layout>` prefix, so a CuTe coordinate-tensor printout
  // pastes in verbatim: `(0,0) o (4,5):(1@0,1@1)`. The origin is the iterator's
  // value — the constant that slicing accumulates and a layout cannot hold.
  let origin = null;
  const oi = topLevelCompose(str);
  if (oi !== -1) {
    if (!allowBasis) {
      throw new Error(
        `"<origin> o <layout>" form is only supported by the Layout tab. ` +
        `Enter just the layout here.`);
    }
    const originStr = str.slice(0, oi).trim().replace(/^ArithTuple/i, '').trim();
    const parsedOrigin = parseValue(originStr, false);
    origin = Array.isArray(parsedOrigin) ? parsedOrigin.slice() : [parsedOrigin];
    str = str.slice(oi + 1).trim();
    if (!str) throw new Error('Nothing after the "o" — expected a layout.');
  }

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
    shape = parseValue(str, false);
    stride = null; // auto
  } else {
    shape  = parseValue(str.slice(0, ci).trim(), false);
    stride = parseValue(str.slice(ci + 1).trim(), allowBasis);
  }

  if (!Array.isArray(shape)) shape = [shape, 1];
  if (shape.length === 1) shape = [shape[0], 1];
  if (stride === null) {
    stride = autoStride(shape);
  } else {
    if (!Array.isArray(stride)) stride = [stride, 0];
    if (stride.length === 1) stride = [stride[0], 0];
  }
  if (shape.length !== stride.length) {
    throw new Error(
      `Shape rank (${shape.length}) does not match stride rank (${stride.length}).`);
  }

  const basis = hasBasisStride(stride);
  return { shape, stride, basis, origin, ndim: basis ? Math.max(basisRank(stride), origin ? origin.length : 0) : 0 };
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

/** Extract [r,g,b] from a "#rrggbb" or "rgb(r,g,b)" string, or null if unparseable. */
function parseColor(c) {
  if (!c) return null;
  if (c[0] === '#' && c.length === 7) {
    return [parseInt(c.slice(1,3),16), parseInt(c.slice(3,5),16), parseInt(c.slice(5,7),16)];
  }
  const m = c.match(/rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)/);
  if (!m) return null;
  return [parseInt(m[1],10), parseInt(m[2],10), parseInt(m[3],10)];
}

/** Text color for any supported color string (hex or rgb()). */
function textOnRGB(color) {
  const rgb = parseColor(color);
  if (!rgb) return '#000';
  const [r, g, b] = rgb;
  const lum = (0.299*r + 0.587*g + 0.114*b) / 255;
  return lum > 0.45 ? '#000' : '#fff';
}

/** Interpolate toward black by `factor` in [0, 1]. factor=0 → original, factor=1 → black. */
function darkenRGB(color, factor) {
  const rgb = parseColor(color);
  if (!rgb) return color;
  const [r, g, b] = rgb;
  const k = 1 - Math.max(0, Math.min(1, factor));
  return `rgb(${Math.round(r*k)},${Math.round(g*k)},${Math.round(b*k)})`;
}

/** Interpolate toward white by `factor` in [0, 1]. factor=0 → original, factor=1 → white. */
function lightenRGB(color, factor) {
  const rgb = parseColor(color);
  if (!rgb) return color;
  const [r, g, b] = rgb;
  const k = Math.max(0, Math.min(1, factor));
  return `rgb(${Math.round(r + (255-r)*k)},${Math.round(g + (255-g)*k)},${Math.round(b + (255-b)*k)})`;
}

/** Gray with a given 0..255 lightness. */
function grayRGB(shade) {
  const s = Math.max(0, Math.min(255, Math.round(shade)));
  return `rgb(${s},${s},${s})`;
}

/** Convert rgb(...) string to hex. */
function rgbToHex(str) {
  const m = str.match(/\d+/g);
  if (!m) return '#ffffff';
  return '#' + m.slice(0,3).map(n => parseInt(n).toString(16).padStart(2,'0')).join('');
}
