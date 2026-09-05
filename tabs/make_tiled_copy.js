// make_tiled_copy tab (Copy scope): the primitive constructor —
// `make_tiled_copy(atom, layout_tv, tiler_mn)`. You supply the TV layout and
// the tiler yourself; nothing is derived.
//
// This file ALSO holds the pieces shared with tabs/make_tiled_copy_tv.js
// (atom section, tile lookup, coloring, vectorization check, viz renderers),
// since the two tabs differ only in where (layout_tv, tiler_mn) come from.
// Functions become globals on `window` (no module system).
//
// python/CuTeDSL/cutlass/cute/atom.py:1011 — `_make_tiled_copy` is the ONLY
// function that builds a TiledCopy. Every public constructor
// (make_tiled_copy, make_tiled_copy_tv, make_cotiled_copy, make_tiled_copy_A/
// B/C/S/D) computes (layout_tv, tiler_mn) some other way and then calls it.
//
// What CuTe checks here (include/cute/atom/copy_atom.hpp:205-206) is ONLY the
// atom relationship:
//   TiledNumThr % AtomNumThr == 0
//   TiledNumVal % AtomNumVal == 0
// Nothing ties layout_tv to tiler_mn. That coupling is implicit: layout_tv's
// output is a flat index into the tile that `zipped_divide(tensor, Tiler_MN)`
// carves out, so the codomain has to land inside it — see mtcCoverageCheck.

// The Ops both tiled-copy tabs offer: ui.js's SIMT_COPY_OPS, unfiltered.
// Every one of them is `ThrID = 1:0` with `ValLayoutSrc == ValLayoutDst`, which
// is precisely what mtcReadAtom assumes when it reports `AtomNumThr = 1` and a
// single atom layout — so the SIMT set is the whole set these tabs can take.
//
// The ldmatrix Ops in MCA_OPS are deliberately NOT here: they are warp-collective
// (ThrID 32:1) with divergent src/dst, so `AtomNumThr = 1` would be a lie and
// mtcRequireAtomDivides would be checking the wrong assert. Supporting them means
// teaching these tabs a second atom shape, not adding a dropdown entry.
const MTC_OPS = SIMT_COPY_OPS;

const MTC_CPASYNC_BITS = [32, 64, 128];

// ═══════════════════════════════════════════════════════
//  Shared: atom section (HTML + read/validate)
// ═══════════════════════════════════════════════════════

/** HTML for the "which atom are you tiling" section. `p` is the id prefix
 *  ('mtc' or 'mtv') so the two tabs get distinct element ids. */
function mtcAtomSection(id, p, summary) {
  return `
        <details class="cuo-section" open>
          <summary>0. Memory movement</summary>
          <div class="cuo-section-body">
${copyMoveField(id, p)}
          </div>
        </details>

        <details class="cuo-section" open>
          <summary>${summary}</summary>
          <div class="cuo-section-body">
            <div class="form-group">
              <label>Copy operation</label>
              <select id="${id}-${p}-op-input" onchange="syncCopyMoves('${id}','${p}',this.value)">
                ${simtCopyOpOptions('universal')}
              </select>
            </div>
            <div class="form-group">
              <label>num_bits_per_copy</label>
              <input type="number" id="${id}-${p}-bits-input" value="128" min="1" step="1">
            </div>
            <div class="form-group">
              <label>tensor_dtype</label>
              <select id="${id}-${p}-dtype-input">${dtypeOptions('half_t')}</select>
            </div>
            <div id="${id}-${p}-atom-result" class="cuo-result"></div>
          </div>
        </details>`;
}

/** Read + validate the atom inputs, paint the atom result div, and return the
 *  derived facts both tabs need. Throws on invalid bits/dtype. */
function mtcReadAtom(tabId, p) {
  const opKey   = document.getElementById(`${tabId}-${p}-op-input`).value;
  const bitsStr = document.getElementById(`${tabId}-${p}-bits-input`).value;
  const dtype   = document.getElementById(`${tabId}-${p}-dtype-input`).value;
  const op = MTC_OPS[opKey] || MTC_OPS.universal;
  const numBits = parseInt(bitsStr, 10);
  if (!Number.isFinite(numBits) || numBits <= 0) {
    throw new Error(`num_bits_per_copy must be a positive integer, got "${bitsStr}"`);
  }
  const elemBits = DTYPE_BITS[dtype];
  if (!elemBits) throw new Error(`Unknown tensor_dtype "${dtype}"`);
  if (numBits % elemBits !== 0) {
    throw new Error(
      `num_bits_per_copy (${numBits}) must be a multiple of sizeof_bits(${dtype}) = ${elemBits}`);
  }
  const atomNumVal = numBits / elemBits;
  const atomStr = formatLayoutStr(new Layout([1, atomNumVal]).shape, new Layout([1, atomNumVal]).stride);
  syncCopyMoves(tabId, p, opKey);
  // The atom strip above the tile viz is a second pane pair with no picker of
  // its own; it follows this tab's movement selection.
  updateCopyPaneTitles(tabId, `${p}-atom`, p);

  document.getElementById(`${tabId}-${p}-atom-result`).innerHTML =
    `<div class="cuo-result-line"><b>Copy_Atom&lt;${op.label}&lt;${numBits}b&gt;, ${dtype}&gt;</b></div>` +
    `<div class="cuo-result-line">AtomNumThr = 1, AtomNumVal = ${numBits} / ${elemBits} = <b>${atomNumVal}</b></div>` +
    `<div class="cuo-result-line">ThrID = 1:0, ValLayoutSrc = ValLayoutDst = ${atomStr}</div>` +
    // The directional Ops' memory attributes are named, not offered: they change
    // the Atom's MLIR type and the emitted PTX but not a single cell of any
    // layout, so they cannot affect the tiling this tab is about.
    (op.attrs
      ? `<div class="cuo-result-line" style="color:#9ca3af">Also accepts ` +
        op.attrs.map(a => `<code>${a}</code>`).join(', ') +
        ` as <code>make_copy_atom</code> kwargs. None of them changes the Atom's layouts, ` +
        `so none of them changes anything below.</div>`
      : '');

  const cpasyncWarn = (op.cpasync && !MTC_CPASYNC_BITS.includes(numBits))
    ? `cp.async hardware only accepts num_bits_per_copy ∈ {32, 64, 128} — ${numBits} would be ` +
      `rejected by cpasync.CopyG2SOp (CopyUniversalOp would take it).`
    : '';
  return { opKey, op, numBits, dtype, elemBits, atomNumVal, atomStr, cpasyncWarn };
}

// ═══════════════════════════════════════════════════════
//  Shared: checks
// ═══════════════════════════════════════════════════════

/** CuTe's own static assert, copy_atom.hpp:206. AtomNumThr is 1 for every Op
 *  these tabs accept (they are all SIMT), so only the value side can fail. */
function mtcRequireAtomDivides(tiledNumVal, atomNumVal, numBits, elemBits, dtype) {
  if (tiledNumVal % atomNumVal === 0) return;
  // Spell out the way out: any num_bits whose AtomNumVal divides TiledNumVal
  // works, so enumerate them rather than leaving the user to factor by hand.
  const ok = [];
  for (let n = 1; n <= tiledNumVal; n++) {
    if (tiledNumVal % n === 0) ok.push(n * elemBits);
  }
  throw new Error(
    `TiledNumVal (${tiledNumVal}) must be a multiple of AtomNumVal ` +
    `(${atomNumVal} = ${numBits}/${elemBits}). This is CuTe's own static assert ` +
    `"TiledCopy uses too few vals for selected CopyAtom" ` +
    `(include/cute/atom/copy_atom.hpp:206) — each thread issues ` +
    `TiledNumVal / AtomNumVal atom invocations, so the division must be exact. ` +
    `Your layout_tv gives each thread only ${tiledNumVal} value${tiledNumVal === 1 ? '' : 's'}, ` +
    `but one ${numBits}-bit atom already moves ${atomNumVal} of them. ` +
    `Either set num_bits_per_copy to one of {${ok.join(', ')}} for ${dtype}, ` +
    `or widen layout_tv's value mode to a multiple of ${atomNumVal}.`);
}

/** `cosize == size` means the layout's image is exactly [0, size) with no holes.
 *  CuTe's thr/val layouts are position maps: their strides express ordering, so
 *  a gap is always a bug rather than a deliberate stride. */
function mtcRequireCompact(name, L) {
  const size = L.size(), cosize = L.cosize();
  if (cosize === size) return;
  const kind = cosize > size
    ? `cosize > size, so raked_product(thr, val) is left with holes`
    : `cosize < size, so distinct value coordinates collide onto the same slot ` +
      `(only ${cosize} distinct outputs for ${size} coordinates)`;
  throw new Error(
    `${name} = ${formatLayoutStr(L.shape, L.stride)} is not compact ` +
    `(size = ${size}, cosize = ${cosize}) — ${kind}. make_tiled_copy_tv requires compact ` +
    `thread and value layouts: strides here encode ordering, not gaps. ` +
    `right_inverse() then returns a partial inverse and threads silently overlap. ` +
    `CuTe and CuTeDSL do NOT check this; they fail much later, at cute.copy.`);
}

/** flat (m, n) in the tile → { tid, vid } (first claimant), plus coverage
 *  bookkeeping. `counts[cell]` is how many (tid, vid) pairs claim that cell —
 *  normally 1, but legitimately >1 when layout_tv has a stride-0 mode (see
 *  mtcCoverageCheck). */
function mtcBuildTileLookup(layout_tv, tiler_mn, thrSize, valSize) {
  const M_tile = tiler_mn[0];
  const cells = tiler_mn[0] * tiler_mn[1];
  const lookup = new Array(cells).fill(null);
  const counts = new Array(cells).fill(0);
  let covered = 0, dupes = 0, outOfTile = 0;
  for (let tid = 0; tid < thrSize; tid++) {
    for (let vid = 0; vid < valSize; vid++) {
      const flat = layout_tv.call(tid, vid);
      if (flat < 0 || flat >= cells) { outOfTile++; continue; }
      if (counts[flat] === 0) { lookup[flat] = { tid, vid }; covered++; }
      else { dupes++; }
      counts[flat]++;
    }
  }
  return { lookup, counts, covered, dupes, outOfTile, M_tile };
}

/** A stride-0 mode in layout_tv means DELIBERATE replication: several threads
 *  read the same tile element. That is not a bug — `make_tiled_copy_A(atom,
 *  tiled_mma)` produces exactly this, because the A operand is shared across the
 *  N-dimension warps of the TiledMMA. Verified on an SM100: a 2x2x1 atom layout
 *  over MmaF16BF16Op gives tv_layout_A_tiled of size 1024 against a 32x16 = 512
 *  tile, i.e. every cell claimed exactly twice, all strides but two nonzero.
 *  So duplicates are only an error when NO stride is zero. */
function mtcHasBroadcast(layout_tv) {
  return flatten(layout_tv.stride).some(x => x === 0);
}

/** The implicit layout_tv <-> tiler_mn contract that CuTe never states. Both
 *  failure directions are silent at compile time and wrong at runtime:
 *    - codomain spills past the tile -> the copy reads into the NEXT tile
 *    - codomain covers only part of the tile -> those elements never get copied
 *  Verified against CuTeDSL on an SM100: an oversized tiler compiles, runs, and
 *  leaves half the tensor untouched. */
function mtcCoverageCheck(look, tiler_mn, thrSize, valSize, derived, broadcast) {
  const cells = tiler_mn[0] * tiler_mn[1];
  const pairs = thrSize * valSize;
  if (look.outOfTile > 0) {
    throw new Error(
      `layout_tv reads outside its tile: ${look.outOfTile} of ${pairs} (tid, vid) pairs map past ` +
      `the end of the ${tiler_mn[0]}×${tiler_mn[1]} tile (${cells} cells). ` +
      `layout_tv's output is a flat index into the tile that zipped_divide(tensor, Tiler_MN) ` +
      `carves out, so those values land in the NEXT tile. CuTe does not check this — ` +
      `it compiles and silently copies the wrong elements. Enlarge Tiler_MN or shrink layout_tv.`);
  }
  if (look.dupes > 0 && !broadcast) {
    throw new Error(
      `layout_tv is not injective: ${look.dupes} of ${pairs} (tid, vid) pairs land on a cell that ` +
      `another pair already claimed, so only ${look.covered} of ${cells} tile cells are covered. ` +
      `No mode of layout_tv has stride 0, so this is accidental overlap rather than deliberate ` +
      `broadcast. ` +
      (derived
        ? `right_inverse(layout_mn) is a partial inverse here, which CuTe does not check — ` +
          `the copy would read and write overlapping elements.`
        : `Threads would read and write overlapping elements.`));
  }
  if (broadcast) {
    // Replication is intended, but it must be UNIFORM: every covered cell read
    // by the same number of threads. A ragged count means the stride-0 mode
    // doesn't line up with the tile and some elements get extra traffic.
    const mult = new Set(look.counts.filter(c => c > 0));
    if (mult.size > 1) {
      throw new Error(
        `layout_tv has a stride-0 (broadcast) mode, but the replication is not uniform: tile cells ` +
        `are claimed ${[...mult].sort((a, b) => a - b).join(', ')} times respectively. A broadcast ` +
        `TiledCopy must read every element the same number of times.`);
    }
  }
  if (look.covered !== cells) {
    throw new Error(
      `layout_tv covers only ${look.covered} of the ${cells} cells in the ` +
      `${tiler_mn[0]}×${tiler_mn[1]} tile. The remaining ${cells - look.covered} elements are never ` +
      `copied — CuTe compiles this happily and silently leaves them untouched. ` +
      `Tiler_MN must have exactly size(layout_tv) = ${pairs} cells for a full partition.`);
  }
}

/** The atom moves `atomNumVal` CONTIGUOUS elements. Within the tile alone we
 *  can't know a memory layout, but we can say which tile axis T0's first atom
 *  invocation runs along — and therefore which major-ness the src/dst tensor
 *  must have for CuTe's `upcast<atom_num_val>` to accept the copy. If the vids
 *  walk neither axis contiguously, no tensor layout can rescue it. */
function mtcVectorizationCheck(atomNumVal, layout_tv, tiler_mn) {
  if (atomNumVal <= 1) return { kind: 'trivial' };
  const M = tiler_mn[0];
  const coords = [];
  for (let v = 0; v < atomNumVal; v++) {
    const flat = layout_tv.call(0, v);
    coords.push([flat % M, Math.floor(flat / M)]);
  }
  const [m0, n0] = coords[0];
  if (coords.every(([m, n], i) => n === n0 && m === m0 + i)) return { kind: 'm', coords, m0, n0 };
  if (coords.every(([m, n], i) => m === m0 && n === n0 + i)) return { kind: 'n', coords, m0, n0 };
  return { kind: 'none', coords };
}

function mtcFormatVecCheck(check, atomNumVal) {
  if (check.kind === 'trivial') {
    return `<div class="cuo-result-line" style="color:#9ca3af">` +
      `Scalar atom (AtomNumVal = 1) — no vectorization constraint; each thread just issues ` +
      `more instructions.</div>`;
  }
  const fmt = check.coords.map(([m, n]) => `(${m},${n})`).join(' ');
  if (check.kind === 'none') {
    return `<div class="cuo-result-line" style="color:#ef4444">` +
      `<b>✗ Cannot vectorize</b> — the ${atomNumVal}-element atom needs T0's first ` +
      `${atomNumVal} vids on a stride-1 run along one tile axis, but they sit at ` +
      `${fmt} — contiguous along neither M nor N. No tensor layout can make this ` +
      `vectorize; CuTe's <code>upcast&lt;${atomNumVal}&gt;</code> rejects it at JIT time ` +
      `("<i>cannot vectorize copy to ${atomNumVal} elements</i>").</div>`;
  }
  const axis  = check.kind === 'm' ? `M (down column n=${check.n0})` : `N (across row m=${check.m0})`;
  const needs = check.kind === 'm' ? 'M-major (column-major)' : 'N-major (row-major)';
  return `<div class="cuo-result-line" style="color:#10b981">` +
    `<b>✓ Vectorizable</b> — T0's first ${atomNumVal} vids run along <b>${axis}</b> at ${fmt}. ` +
    `<code>upcast&lt;${atomNumVal}&gt;</code> will coalesce them to <code>(${atomNumVal}):(1)</code> ` +
    `provided the src/dst tensor is <b>${needs}</b> over the tile.</div>`;
}

// ═══════════════════════════════════════════════════════
//  Tiler parsing
// ═══════════════════════════════════════════════════════

/** Parse a Tiler_MN. Unlike a Layout, a Tiler's modes are INDEPENDENT — each is
 *  its own extent or layout — so `parseLayout` is the wrong tool: it demands a
 *  single top-level `shape:stride` split and rejects a colon inside parens.
 *  That would refuse `(8:1, 16:2)`, which is exactly the form CuTeDSL prints a
 *  tiler in (`tc.tiler_mn` → `(8:1, 16:2)`), so you could not paste one back.
 *
 *  Accepts:
 *    (8, 16)          shape per mode
 *    (8:1, 16:2)      layout per mode — a strided sub-tile
 *    ((2,4), 16)      nested shape mode, folded to its product
 *    128  |  (128,)   rank 1
 *
 *  Returns { extents, strides, hadStride }. The strides are reported back to the
 *  user but do not change this tab's picture: a tiler's strides say where the
 *  tile's cells SIT IN THE TENSOR, and this tab never involves a tensor. The
 *  tile is extents[0] x extents[1] cells either way. */
function mtcParseTiler(str) {
  const t = (str || '').trim();
  if (!t) throw new Error('Tiler_MN is empty — give it a shape like (8, 16).');

  // Strip one layer of parens only if it wraps the WHOLE string, so that
  // "(8, 16)" unwraps but "(2,4), 16" (already a mode list) is left alone.
  let body = t, depth = 0;
  if (body[0] === '(') {
    for (let i = 0; i < body.length; i++) {
      if (body[i] === '(') depth++;
      else if (body[i] === ')' && --depth === 0) {
        if (i === body.length - 1) body = body.slice(1, -1);
        break;
      }
    }
  }

  // Split on top-level commas — each piece is one independent tiler mode.
  const modes = [];
  depth = 0;
  let start = 0;
  for (let i = 0; i < body.length; i++) {
    const c = body[i];
    if (c === '(') depth++;
    else if (c === ')') depth--;
    else if (c === ',' && depth === 0) { modes.push(body.slice(start, i)); start = i + 1; }
  }
  const tail = body.slice(start).trim();
  if (tail) modes.push(tail);           // a trailing comma, as in "(128,)", is fine

  const extents = [], strides = [];
  let hadStride = false;
  for (const raw of modes) {
    const m = raw.trim();
    if (!m) continue;
    const ci = topLevelColon(m);
    const shapeStr = ci === -1 ? m : m.slice(0, ci).trim();
    let extent;
    try {
      extent = product(parseValue(shapeStr));
    } catch (e) {
      throw new Error(
        `Cannot read Tiler_MN mode "${m}": ${e.message}. Each mode is an extent (8), ` +
        `a layout (8:1), or a nested shape ((2,4)) — separated by top-level commas.`);
    }
    if (!Number.isFinite(extent) || extent <= 0) {
      throw new Error(`Tiler_MN mode "${m}" has non-positive extent ${extent}.`);
    }
    extents.push(extent);
    if (ci === -1) {
      strides.push(null);
    } else {
      hadStride = true;
      strides.push(m.slice(ci + 1).trim());
    }
  }

  if (extents.length === 0) throw new Error('Tiler_MN has no modes.');
  if (extents.length === 1) { extents.push(1); strides.push(null); }  // rank-1 tiler -> M x 1
  if (extents.length > 2) {
    throw new Error(
      `Tiler_MN has rank ${extents.length}; this visualization draws a 2-D tile. ` +
      `CuTe accepts higher-rank tilers — collapse it to (M, N) to view it here.`);
  }
  return { extents, strides, hadStride };
}

// ═══════════════════════════════════════════════════════
//  Shared: visualization
// ═══════════════════════════════════════════════════════

/** HTML for the tile viz + per-thread panel. Both tabs render into these. */
function mtcVizSection(id, p) {
  return `
      <div class="comp-results" style="grid-template-columns:1fr">
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-${p}-atom-title">Copy_Atom &mdash; the unit being tiled</span>
            <span style="display:flex;align-items:center;gap:4px">
              ${copyDirButtons(id, `${p}-atom`)}
              <button class="mode-btn" id="${id}-${p}-atom-src-svg-zoom" onclick="toggleCopyZoom('${id}','${p}-atom')">Zoom in</button>
            </span>
          </div>
          <div class="cuo-viz-desc">
            One invocation of the atom, drawn exactly as the <b>make_copy_atom</b>
            tab draws it: <code>V<i>k</i></code> is the value index within the
            instruction, and <code>ThrID = 1:0</code> means one thread issues the
            whole thing. This is the <em>unit</em>; everything below is that unit
            replicated over <code>Tiler_MN</code> by <code>layout_tv</code>. The
            colour is the one T0's first invocation gets in the tile grid, so the
            two pictures line up.
          </div>
${copyPanes(id, `${p}-atom`)}
        </div>
        <div class="comp-viz-item">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-${p}-tile-title">TiledCopy tile</span>
            <span style="display:flex;align-items:center;gap:4px">
              ${copyDirButtons(id, p)}
              <span class="mode-btn-group" id="${id}-${p}-tile-mode-btns">
                <button class="mode-btn" onclick="setMtcMode('${id}','${p}','value')">value</button>
              </span>
              <button class="mode-btn" id="${id}-${p}-src-svg-zoom" onclick="toggleCopyZoom('${id}','${p}')">Zoom in</button>
              <button class="mode-btn" onclick="downloadSVG('${id}-${p}-src-svg', 'tiled_copy.svg')">Download SVG</button>
            </span>
          </div>
          <div class="cuo-viz-desc">
            One tile of <code>Tiler_MN</code>. Cell <code>(m, n)</code> shows the
            <code>(t, v)</code> pair that owns it; the <code>value</code> toggle
            adds the col-major flat offset, which is what
            <code>layout_tv(t, v)</code> actually returns. Both Ops here have
            <code>ValLayoutSrc == ValLayoutDst</code>, so the two panes match.
          </div>
${copyPanes(id, p)}
        </div>
        <div class="comp-viz-item" id="${id}-${p}-thread-item" style="display:none">
          <div class="comp-viz-header">
            <span class="comp-viz-label" id="${id}-${p}-thread-title">Per-thread view</span>
          </div>
          <div class="viz-box" style="background:#0b1220;color:#d1d5db;font-family:monospace;font-size:0.82rem;padding:14px 16px">
            <div id="${id}-${p}-thread-body"></div>
          </div>
        </div>
      </div>`;
}

// Thread 0 gets the "initial" color; other threads cycle through TV_COLORS.
// Within a thread, successive atom invocations darken the base color so the
// first atom is the lightest and later ones progressively more saturated.
function mtcThreadAtomColor(tid, atomIdx, totalAtoms) {
  const base = colorTV(tid);
  if (totalAtoms <= 1 || atomIdx === 0) return base;
  return darkenRGB(base, (atomIdx / (totalAtoms - 1)) * 0.45);
}

/** The atom strip above the tile grid. Identical to what make_copy_atom draws —
 *  same `simtAtomPaneHTML`, so the two tabs cannot drift. `ValLayoutSrc ==
 *  ValLayoutDst` for every Op these tabs accept, so the panes are equal; they
 *  are still rendered as a pair because the SRC/DST headers carry the memory
 *  movement, which now differs per Op. */
function mtcRenderAtomViz(tabId, p, s) {
  for (const side of ['src', 'dst']) {
    const host = document.getElementById(`${tabId}-${p}-atom-${side}-svg`);
    if (!host) continue;
    host.innerHTML = simtAtomPaneHTML(side, s.atomStr, s.atomNumVal, s.dtype);
    applyZoomState(`${tabId}-${p}-atom-${side}-svg`);
  }
  const title = document.getElementById(`${tabId}-${p}-atom-title`);
  if (title) {
    title.textContent =
      `${s.op.label} — ${s.numBits}b / ${s.dtype} → ${s.atomNumVal} ` +
      `element${s.atomNumVal === 1 ? '' : 's'} per invocation`;
  }
}

function mtcRenderTileViz(tabId, p, s) {
  const M_tile = s.tiler_mn[0];
  const filterTid = s.highlightTid;
  const modes = s.tileMode instanceof Set ? s.tileMode : new Set();
  const header =
    `<div style="font-size:0.78rem;color:#9ca3af;font-family:monospace;margin-bottom:4px">` +
    `Tile ${s.tiler_mn[0]}×${s.tiler_mn[1]} &mdash; same color = same thread; ` +
    `brightness = atom invocation (FrgX=${s.frgX}${s.frgX > 1 ? ', darker for later atoms' : ''})` +
    (s.broadcast ? ` &mdash; <b>${s.bcastFactor}-way broadcast</b>, cell shows the lowest-id claimant` : '') +
    (filterTid !== null ? ` &mdash; <b>filtered to T${filterTid}</b>` : '') +
    `</div>`;
  const svg = buildColoredLayoutSVG(s.tiler_mn.slice(), [1, M_tile], 'value', (m, n, offset) => {
      const e = s.lookup[m + n * M_tile];
      // No owner at all -- the coverage check has already said so; draw it the
      // way the TV tab draws an unclaimed cell rather than as a blank box.
      if (!e) return { bg: '#f0f0f0', fg: '#bbb', text: ['\u2014'] };
      const lines = [`T${e.tid}`, `V${e.vid}`];
      if (modes.has('value')) lines.push(String(offset));
      // Highlighting is a FOCUS, not a mask: a dimmed cell keeps its T/V labels
      // in muted grey so the surrounding ownership stays readable, which is what
      // buildTVSVG does for the TV tab. Erasing the text instead made the rest of
      // the tile unreadable the moment a thread was picked. Same palette as
      // buildTVSVG (#f0f0f0 on #bbb) so the two tabs look like one tool.
      if (filterTid !== null && e.tid !== filterTid) {
        return { bg: '#f0f0f0', fg: '#bbb', text: lines };
      }
      return { bg: mtcThreadAtomColor(e.tid, Math.floor(e.vid / s.atomNumVal), s.frgX), text: lines };
    });
  // Same picture on both sides: layout_tv describes the tile, and for these Ops
  // ValLayoutSrc == ValLayoutDst so src and dst partition it identically.
  for (const side of ['src', 'dst']) {
    document.getElementById(`${tabId}-${p}-${side}-svg`).innerHTML = header + svg;
    applyZoomState(`${tabId}-${p}-${side}-svg`);
  }
  updateModeBtns(`${tabId}-${p}-tile-mode-btns`, modes);
  document.getElementById(`${tabId}-${p}-tile-title`).textContent =
    `TiledCopy tile (${s.tiler_mn[0]}×${s.tiler_mn[1]}) — ${s.thrSize} threads × ${s.valSize} values ` +
    `(FrgV=${s.atomNumVal}, FrgX=${s.frgX})` +
    (filterTid !== null ? ` — filtered to T${filterTid}` : '');
}

// Text panel below the tile, shown only when a valid thread is highlighted.
// Everything here is tile-local — no tensor is involved.
function mtcRenderThreadPanel(tabId, p, s) {
  const item = document.getElementById(`${tabId}-${p}-thread-item`);
  const body = document.getElementById(`${tabId}-${p}-thread-body`);
  if (!item || !body) return;
  if (!s || s.highlightTid === null) {
    item.style.display = 'none';
    body.innerHTML = '';
    return;
  }
  const tid = s.highlightTid, M_tile = s.tiler_mn[0];
  const rows = [];
  for (let a = 0; a < s.frgX; a++) {
    const coords = [];
    for (let k = 0; k < s.atomNumVal; k++) {
      const flat = s.layout_tv.call(tid, a * s.atomNumVal + k);
      coords.push(`(${flat % M_tile},${Math.floor(flat / M_tile)})`);
    }
    rows.push(`<div class="cuo-result-line">atom #${a}: V${a * s.atomNumVal}..V${(a + 1) * s.atomNumVal - 1} ` +
      `&rarr; ${coords.join(' ')}</div>`);
  }
  item.style.display = '';
  document.getElementById(`${tabId}-${p}-thread-title`).textContent = `Per-thread view — T${tid}`;
  body.innerHTML =
    `<div class="cuo-result-line">Thread <b>T${tid}</b> of ${s.thrSize}, ` +
    `<b>${s.valSize}</b> values per tile in <b>${s.frgX}</b> atom invocation${s.frgX === 1 ? '' : 's'}</div>` +
    rows.join('') +
    `<div class="cuo-result-line" style="color:#9ca3af">Coordinates are (m, n) in the ` +
    `${s.tiler_mn[0]}×${s.tiler_mn[1]} tile.</div>`;
}

/** Wipe both SRC/DST panes — used on every error path. */
function mtcClearPanes(tabId, p) {
  // Both pane pairs: the tile grid and the atom strip above it. Leaving a stale
  // atom picture up while the tile went blank would read as "the atom is fine,
  // the tiling failed" — sometimes true, but not something to imply by accident.
  for (const prefix of [p, `${p}-atom`]) {
    for (const side of ['src', 'dst']) {
      const el = document.getElementById(`${tabId}-${prefix}-${side}-svg`);
      if (el) el.innerHTML = '';
    }
  }
}

/** Shared mode toggle — `p` picks which tab's state to mutate. */
function setMtcMode(tabId, p, mode) {
  const s = (p === 'mtc' ? mtcState : mtvState)[tabId];
  if (!s) return;
  let modes = s.tileMode;
  if (!(modes instanceof Set)) { modes = new Set(); s.tileMode = modes; }
  if (modes.has(mode)) modes.delete(mode); else modes.add(mode);
  mtcRenderTileViz(tabId, p, s);
}

// ═══════════════════════════════════════════════════════
//  The make_tiled_copy tab itself
// ═══════════════════════════════════════════════════════

function generateMakeTiledCopyTabContent(id) {
  return `
    <!-- make_tiled_copy panel -->
    <div id="${id}-tab-make_tiled_copy" class="panel">
      <div class="controls">
        <h2>make_tiled_copy</h2>
${mtcAtomSection(id, 'mtc', '1. The Copy_Atom you are tiling')}

        <details class="cuo-section" open>
          <summary>2. Supply layout_tv and Tiler_MN directly</summary>
          <div class="cuo-section-body">
            ${layoutInputField({
              id: `${id}-mtc-tv-input`,
              label: 'layout_tv &mdash; (tid, vid) &rarr; flat (m, n) in the tile',
              hint: 'mode 0 = threads, mode 1 = values',
              value: '((8,16),8):((128,1),16)'
            })}
            ${layoutInputField({
              id: `${id}-mtc-tiler-input`,
              label: 'Tiler_MN &mdash; the (M, N) tile zipped_divide carves out',
              hint: 'a Tiler: (8, 16) or per-mode layouts (8:1, 16:2)',
              value: '(16, 64)'
            })}
            <div id="${id}-mtc-tile-result" class="cuo-result"></div>
          </div>
        </details>

        <div class="form-group">
          <label>Highlight thread (empty = show all threads)</label>
          <input type="text" id="${id}-mtc-highlight-tid" value="" placeholder="e.g. 5" oninput="setMtcHighlight('${id}')">
        </div>

        ${statusDivs(`${id}-mtc`)}
        <button class="btn btn-render" onclick="renderMakeTiledCopy('${id}')">Render</button>
        <button class="btn btn-render" style="margin-top:6px;background:#111827" id="${id}-mtc-export" onclick="exportMTC('${id}')">Export URL</button>

        <div class="presets">
          <h3>Presets</h3>
          <div class="preset-list">
            <button class="preset-btn" onclick="setMTC('${id}','universal',32,'half_t','((8,4),(2,2)):((16,2),(8,1))','(8, 16)')">32b half, 32thr x 4val, tile 8x16</button>
            <button class="preset-btn" onclick="setMTC('${id}','cpasync',128,'half_t','((8,16),8):((128,1),16)','(16, 64)')">sgemm_sm80 TV layout, tile 16x64</button>
            <button class="preset-btn" onclick="setMTC('${id}','universal',32,'half_t','((4,8,2,2),((2,2,2),(1,1))):((64,1,16,0),((32,8,256),(0,0)))','(32, 16)')">make_tiled_copy_A shape &mdash; 2-way broadcast (stride-0 mode)</button>
            <button class="preset-btn" onclick="setMTC('${id}','universal',128,'float','(256,4):(4,1)','(128, 8)')">sgemm_2.cu TV layout &mdash; 128b float, tile 128x8</button>
            <button class="preset-btn" onclick="setMTC('${id}','g2r',128,'half_t','((8,16),8):((128,1),16)','(16, 64)')">CopyG2ROp &mdash; same tiling, GMEM&rarr;RMEM</button>
            <button class="preset-btn" onclick="setMTC('${id}','r2s',64,'half_t','((8,4),(2,2)):((16,2),(8,1))','(8, 16)')">CopyR2SOp &mdash; 64b half, RMEM&rarr;SMEM</button>
          </div>
        </div>

        <div class="hint">
          <b>The primitive constructor.</b>
          <code>make_tiled_copy(atom, layout_tv, tiler_mn)</code> is the only thing
          that actually builds a <code>TiledCopy</code>
          (<code>_make_tiled_copy</code>, <code>cutlass/cute/atom.py:1011</code>).
          Every other constructor &mdash; <code>make_tiled_copy_tv</code>,
          <code>make_cotiled_copy</code>, <code>make_tiled_copy_A/B/C/S/D</code>
          &mdash; just computes these two arguments some other way and calls it.
          Use the <b>make_tiled_copy_tv</b> tab if you'd rather derive them from a
          thr/val pair.<br><br>
          <b>Must Tiler_MN match layout_tv's shape?</b> No &mdash; they describe
          different things. <code>layout_tv</code>'s shape is
          <code>(num_threads, num_values)</code>; <code>Tiler_MN</code> is the
          <code>(M, N)</code> tile. CuTe's only static asserts here
          (<code>copy_atom.hpp:205-206</code>) relate <code>layout_tv</code> to the
          <em>atom</em>, never to the tiler:
          <code>TiledNumThr % AtomNumThr == 0</code> and
          <code>TiledNumVal % AtomNumVal == 0</code>.<br><br>
          What <em>is</em> required, and never checked, is that
          <code>layout_tv</code>'s codomain exactly fills the tile &mdash; every
          cell claimed, none out of bounds. Note two ways the sizes can
          legitimately differ:<br>
          &bull; <b>Broadcast.</b> A stride-0 mode in <code>layout_tv</code> makes
          several threads read the same element, so
          <code>size(layout_tv)</code> is a multiple of
          <code>size(Tiler_MN)</code>. <code>make_tiled_copy_A/B(atom, tiled_mma)</code>
          does exactly this &mdash; an MMA's A operand is shared across the
          N-dimension warps.<br>
          &bull; <b>Tiler_MN is a Tiler, not a shape.</b> Each mode is independent
          and may be a whole layout, so <code>(8:1, 16:2)</code> selects a strided
          sub-tile &mdash; tile 0 becomes every <em>other</em> column instead of a
          contiguous block. The input above accepts that form (it is how CuTeDSL
          prints a tiler), and it does not change this picture: tiler strides say
          where the tile's cells sit in the <em>tensor</em>, which this tab never
          involves. The tile is <code>M &times; N</code> cells either way. Verified on
          an SM100 against CuTeDSL &mdash; a tiler that is too big compiles, runs,
          and silently leaves the uncovered elements untouched; one that is too
          small compiles and reads into the neighbouring tile. Both presets above
          reproduce it. Note also that reshaping the tiler at constant size (e.g.
          <code>(8,16)</code> &rarr; <code>(16,8)</code>) changes the strides that
          come out of <code>zipped_divide</code>, and can break vectorization even
          though the cell count is identical.
        </div>
      </div>
${mtcVizSection(id, 'mtc')}
    </div>`;
}

const mtcState = {};

function renderMakeTiledCopy(tabId) {
  showErr(`${tabId}-mtc-error`, '');
  showWarn(`${tabId}-mtc-warning`, '');
  try {
    const atom = mtcReadAtom(tabId, 'mtc');

    const tvStr    = document.getElementById(`${tabId}-mtc-tv-input`).value;
    const tilerStr = document.getElementById(`${tabId}-mtc-tiler-input`).value;
    updateRankWarning(`${tabId}-mtc-warning`, [['layout_tv', tvStr], ['Tiler_MN', tilerStr]]);

    const tvP  = parseLayout(tvStr);
    const tvSP = stripTrivialTrailing(tvP.shape, tvP.stride);
    const layout_tv = new Layout(tvSP.shape, tvSP.stride);
    if (layout_tv.rank() !== 2) {
      throw new Error(
        `layout_tv must have rank 2 — mode 0 is the thread mode, mode 1 the value mode. ` +
        `Got rank ${layout_tv.rank()} from ${formatLayoutStr(layout_tv.shape, layout_tv.stride)}.`);
    }
    const thrSize = product(layout_tv.shape[0]);
    const valSize = product(layout_tv.shape[1]);

    // Tiler_MN is a Tiler, not a Layout: independent modes, each an extent or a
    // layout. mtcParseTiler handles both, including CuTeDSL's own "(8:1, 16:2)"
    // print form, which parseLayout would reject.
    const tilerP = mtcParseTiler(tilerStr);
    const tiler_mn = tilerP.extents;

    mtcRequireAtomDivides(valSize, atom.atomNumVal, atom.numBits, atom.elemBits, atom.dtype);
    const frgX = valSize / atom.atomNumVal;

    const broadcast = mtcHasBroadcast(layout_tv);
    const look = mtcBuildTileLookup(layout_tv, tiler_mn, thrSize, valSize);
    mtcCoverageCheck(look, tiler_mn, thrSize, valSize, false, broadcast);
    const bcastFactor = broadcast ? look.counts.find(c => c > 0) : 1;

    const vecCheck = mtcVectorizationCheck(atom.atomNumVal, layout_tv, tiler_mn);
    document.getElementById(`${tabId}-mtc-tile-result`).innerHTML =
      `<div class="cuo-result-line">layout_tv = <b>${formatLayoutStr(layout_tv.shape, layout_tv.stride)}</b></div>` +
      `<div class="cuo-result-line">Tiler_MN  = <b>(${tiler_mn.join(', ')})</b> = ${tiler_mn[0] * tiler_mn[1]} cells</div>` +
      `<div class="cuo-result-line">TiledNumThr = ${thrSize}, TiledNumVal = ${valSize}, ` +
      `FrgV = ${atom.atomNumVal}, FrgX = ${frgX}</div>` +
      (broadcast
        ? `<div class="cuo-result-line" style="color:#93c5fd">layout_tv has a stride-0 mode &mdash; ` +
          `<b>${bcastFactor}-way broadcast</b>: every tile element is read by ${bcastFactor} threads. ` +
          `That is deliberate, not overlap (this is what <code>make_tiled_copy_A/B</code> produce, ` +
          `since an MMA's A operand is shared across the N-dimension warps).</div>`
        : '') +
      mtcFormatVecCheck(vecCheck, atom.atomNumVal);

    const hl = readHighlightTid(tabId, 'mtc', thrSize);
    const tilerNote = tilerP.hadStride
      ? `Tiler_MN modes carry strides (${tilerP.strides.map((x, i) => x === null ? `${tiler_mn[i]}` : `${tiler_mn[i]}:${x}`).join(', ')}). ` +
        `They are accepted and do not change this view — a tiler's strides say where the tile's ` +
        `cells sit in the TENSOR, and this tab never involves a tensor. The tile is ` +
        `${tiler_mn[0]}×${tiler_mn[1]} cells either way.`
      : '';
    const warn = [atom.cpasyncWarn, hl.warn, tilerNote].filter(Boolean).join(' ');
    if (warn) showWarn(`${tabId}-mtc-warning`, warn);

    const prev = mtcState[tabId] || {};
    mtcState[tabId] = {
      ...atom, layout_tv, tiler_mn, thrSize, valSize, frgX,
      lookup: look.lookup, counts: look.counts, broadcast, bcastFactor,
      highlightTid: hl.tid, vecCheck,
      tileMode: (prev.tileMode instanceof Set) ? prev.tileMode : new Set(),
    };

    mtcRenderAtomViz(tabId, 'mtc', mtcState[tabId]);
    mtcRenderTileViz(tabId, 'mtc', mtcState[tabId]);
    mtcRenderThreadPanel(tabId, 'mtc', mtcState[tabId]);
    updateOuterTabLabel(tabId, `make_tiled_copy:${tiler_mn[0]}x${tiler_mn[1]}`);
  } catch (e) {
    showErr(`${tabId}-mtc-error`, e.message);
    mtcClearPanes(tabId, 'mtc');
    const item = document.getElementById(`${tabId}-mtc-thread-item`);
    if (item) item.style.display = 'none';
  }
}

function setMtcHighlight(tabId) {
  const s = mtcState[tabId];
  if (!s || !s.layout_tv) return;
  renderMakeTiledCopy(tabId);
}

function setMTC(tabId, opKey, bits, dtype, tv, tiler) {
  document.getElementById(`${tabId}-mtc-op-input`).value    = opKey;
  document.getElementById(`${tabId}-mtc-bits-input`).value  = bits;
  document.getElementById(`${tabId}-mtc-dtype-input`).value = dtype;
  document.getElementById(`${tabId}-mtc-tv-input`).value    = tv;
  document.getElementById(`${tabId}-mtc-tiler-input`).value = tiler;
  renderMakeTiledCopy(tabId);
}

function exportMTC(tabId) {
  exportURL(`${tabId}-mtc-export`, 'make_tiled_copy',
    document.getElementById(`${tabId}-mtc-op-input`).value,
    document.getElementById(`${tabId}-mtc-bits-input`).value,
    document.getElementById(`${tabId}-mtc-dtype-input`).value,
    document.getElementById(`${tabId}-mtc-tv-input`).value,
    document.getElementById(`${tabId}-mtc-tiler-input`).value);
}
