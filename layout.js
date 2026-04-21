/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

//
// Port of python/pycute/int_tuple.py and python/pycute/layout.py to JavaScript.
//
// Conventions:
//   Python tuple  →  JS Array
//   Python int    →  JS number (typeof x === 'number')
//   Python None   →  JS null
//   Python assert →  JS assert() helper (throws on failure)
//
//   Layout.__call__(*args) → layout.call(...args)
//   Layout.__getitem__(i)  → layout.mode(i)
//   Layout.__len__()       → layout.rank()
//


// ═══════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════

function assert(cond, msg) {
  if (!cond) throw new Error(msg || 'Assertion failed');
}


// ═══════════════════════════════════════════════════════
//  Functions for manipulating IntTuples
//  (port of python/pycute/int_tuple.py)
// ═══════════════════════════════════════════════════════

function is_int(x) {
  return typeof x === 'number';
}

function is_tuple(x) {
  return Array.isArray(x);
}

function flatten(t) {
  if (is_tuple(t)) {
    if (t.length === 0) {
      return [];
    } else {
      return [].concat(...t.map(a => flatten(a)));
    }
  } else {
    return [t];
  }
}

function signum(a) {
  return (a > 0) - (a < 0);
}

function product(a) {
  if (is_tuple(a)) {
    return a.reduce((val, elem) => val * product(elem), 1);
  } else {
    return a;
  }
}

function inner_product(a, b) {
  if (is_tuple(a)) {                      // tuple tuple
    assert(a.length === b.length);
    return a.reduce((sum, x, i) => sum + inner_product(x, b[i]), 0);
  } else {                                // "int" "int"
    assert(!is_tuple(b));
    return a * b;
  }
}

function tuple_max(a) {
  if (is_tuple(a)) {
    return Math.max(...a.map(x => tuple_max(x)));
  } else {
    return a;
  }
}

function elem_scale(a, b) {
  if (is_tuple(a)) {
    if (is_tuple(b)) {                     // tuple tuple
      assert(a.length === b.length);
      return a.map((x, i) => elem_scale(x, b[i]));
    } else {                               // tuple "int"
      assert(false, "elem_scale: tuple scaled by int");  // Error
    }
  } else {
    if (is_tuple(b)) {                     // "int" tuple
      return elem_scale(a, product(b));
    } else {                               // "int" "int"
      return a * b;
    }
  }
}


// Inclusive prefix ceil div with output congruent to input a
function shape_div(a, b) {
  if (is_tuple(a)) {
    if (is_tuple(b)) {                    // tuple tuple
      assert(a.length === b.length);
      return a.map((x, i) => shape_div(x, b[i]));
    } else {                              // tuple "int"
      //r = [shape_div(a[0],b)] + [shape_div(a[i],b := shape_div(b, product(a[i-1]))) for i in range(1,len(a))]
      const r = [];
      for (const v of a) {
        r.push(shape_div(v, b));
        b = shape_div(b, product(v));
      }
      return r;
    }
  } else {
    if (is_tuple(b)) {                    // "int" tuple
      return shape_div(a, product(b));
    } else {                              // "int" "int"
      assert(a % b === 0 || b % a === 0);
      return Math.floor((a + b - 1) / b);
    }
  }
}

// Exclusive prefix product with output congruent to input a
function prefix_product(a, init) {
  if (init === undefined) init = 1;

  if (is_tuple(a)) {
    if (is_tuple(init)) {                 // tuple tuple
      assert(a.length === init.length);
      return a.map((x, i) => prefix_product(x, init[i]));
    } else {                              // tuple "int"
      //r = [prefix_product(a[0],init)] + [prefix_product(a[i],init := init * product(a[i-1])) for i in range(1,len(a))]
      const r = [];
      for (const v of a) {
        r.push(prefix_product(v, init));
        init = init * product(v);
      }
      return r;
    }
  } else {
    if (is_tuple(init)) {                 // "int" tuple
      assert(false, "prefix_product: int with tuple init");  // Error
    } else {                              // "int" "int"
      return init;
    }
  }
}


function idx2crd(idx, shape, stride) {
  if (stride === undefined) {
    stride = prefix_product(shape);
  }

  if (is_tuple(idx)) {
    if (is_tuple(shape)) {                // tuple tuple tuple
      assert(idx.length === shape.length && idx.length === stride.length);
      return idx.map((idxI, i) => idx2crd(idxI, shape[i], stride[i]));
    } else {                              // tuple "int" "int"
      assert(false, "idx2crd: tuple idx with int shape");  // Error
    }
  } else {
    if (is_tuple(shape)) {                // "int" tuple tuple
      assert(shape.length === stride.length);
      return shape.map((s, i) => idx2crd(idx, s, stride[i]));
    } else {                              // "int" "int" "int"
      return Math.floor(idx / stride) % shape;
    }
  }
}


function crd2idx(crd, shape, stride) {
  if (stride === undefined) {
    stride = prefix_product(shape);
  }

  if (is_tuple(crd)) {
    if (is_tuple(shape)) {                // tuple tuple tuple
      assert(crd.length === shape.length && crd.length === stride.length);
      return crd.reduce((sum, c, i) => sum + crd2idx(c, shape[i], stride[i]), 0);
    } else {                              // tuple "int" "int"
      assert(false, `crd2idx: crd=${crd}, shape=${shape}`);  // Error
    }
  } else {
    if (crd === null) {
      crd = 0;
    }

    if (is_tuple(shape)) {                // "int" tuple tuple
      assert(shape.length === stride.length);
      let result = 0;
      for (let i = 0; i < shape.length - 1; i++) {
        result += crd2idx(crd % product(shape[i]), shape[i], stride[i]);
        crd = Math.floor(crd / product(shape[i]));
      }
      return result + crd2idx(crd, shape[shape.length - 1], stride[stride.length - 1]);
    } else {                              // "int" "int" "int"
      return crd * stride;
    }
  }
}


// Transform crd into the dst_shape's iteration space
function crd2crd(crd, dst_shape, src_shape) {
  if (is_tuple(crd)) {
    if (is_tuple(dst_shape)) {            // tuple tuple
      assert(crd.length === dst_shape.length);
      return crd.map((x, i) => crd2crd(x, dst_shape[i]));
    } else {                              // tuple "int"
      // Ambiguous unless we have src_shape
      assert(src_shape !== undefined);
      return crd2idx(crd, src_shape);
    }
  } else {
    if (is_tuple(dst_shape)) {            // "int" tuple
      return idx2crd(crd, dst_shape);
    } else {                              // "int" "int"
      assert(crd < dst_shape);
      return crd;
    }
  }
}


// Filter trg according to crd: keep only elements of trg that are paired with null (None)
function slice_(crd, trg) {
  if (is_tuple(crd)) {
    if (is_tuple(trg)) {                  // tuple tuple
      assert(crd.length === trg.length);
      // match C++ behavior of `filter_tuple` using `tuple_cat(...)`
      const parts = crd.map((c, i) => slice_(c, trg[i])).filter(x => x.length > 0);
      return [].concat(...parts);
    } else {
      assert(false, "slice_: tuple crd with non-tuple trg");  // tuple "int" : Error
    }
  } else if (crd === null) {
    // match C++ behavior `return cute::tuple<B>{b};`
    return [trg];
  } else {
    return [];
  }
}


// Determine if null (None) appears at any of an int_tuple's terminals
function has_none(a) {
  if (is_tuple(a)) {
    return a.some(v => has_none(v));
  } else {
    return a === null;
  }
}


// ═══════════════════════════════════════════════════════
//  Definition of CuTe Layouts and functions to manipulate them
//  (port of python/pycute/layout.py)
// ═══════════════════════════════════════════════════════

function is_layout(x) {
  return x instanceof Layout;
}


class Layout {
  constructor(_shape, _stride) {
    this.shape = _shape;
    if (_stride === undefined) {
      this.stride = prefix_product(this.shape);
    } else {
      this.stride = _stride;
    }
  }

  // operator ==
  equals(other) {
    return JSON.stringify(this.shape) === JSON.stringify(other.shape) &&
           JSON.stringify(this.stride) === JSON.stringify(other.stride);
  }

  // operator len(L)  (len [rank] like tuples)
  rank() {
    if (is_tuple(this.shape)) {
      return this.shape.length;
    } else {
      return 1;
    }
  }

  // operator ()    (map coord to idx)
  // Python: layout(*args) → JS: layout.call(...args)
  //
  // Map a logical coordinate to a linear index (Coord has no null slice operators)
  // OR
  // Slice the layout and return the sublayout (Coord has a null slice op)
  //
  // Follow the same behavior of `Layout::operator(Coord const&)` in cute C++
  call(...args) {
    if (has_none(args)) {
      if (args.length === 1) {
        return new Layout(slice_(args[0], this.shape), slice_(args[0], this.stride));
      } else {
        return new Layout(slice_(args, this.shape), slice_(args, this.stride));
      }
    } else {
      if (args.length === 1) {
        return crd2idx(args[0], this.shape, this.stride);
      } else {
        return crd2idx(args, this.shape, this.stride);
      }
    }
  }

  // operator []    (get-i like tuples)
  // Python: layout[i] → JS: layout.mode(i)
  mode(i) {
    if (is_tuple(this.shape)) {
      return new Layout(this.shape[i], this.stride[i]);
    } else {
      assert(i === 0);
      return new Layout(this.shape, this.stride);
    }
  }

  // size(layout)   Size of the domain
  size() {
    return product(this.shape);
  }

  // cosize(layout)   Size of the codomain
  cosize() {
    return this.call(this.size() - 1) + 1;
  }

  // print and str
  toString() {
    return `${JSON.stringify(this.shape)}:${JSON.stringify(this.stride)}`;
  }

  // error msgs and representation
  repr() {
    return `Layout(${JSON.stringify(this.shape)},${JSON.stringify(this.stride)})`;
  }
}


// Make Layout from a list of layouts (each layout its own mode in the result)
function make_layout(...layouts) {
  if (layouts.length === 1 && !is_layout(layouts[0])) {
    // Accept an iterable (array of layouts)
    layouts = [...layouts[0]];
  }

  const shape = layouts.map(a => a.shape);
  const stride = layouts.map(a => a.stride);
  return new Layout(shape, stride);
}


// Size of the domain
function size(layout) {
  if (is_layout(layout)) {
    return layout.size();
  }
  return product(layout);
}


// Size of the codomain
function cosize(layout) {
  return layout.cosize();
}


// Layout coalesce -- flatten and combine as many modes as possible while preserving the int-to-int function
function coalesce(layout, profile) {
  if (is_tuple(profile)) {
    assert(layout.rank() >= profile.length);
    const parts = [];
    for (let i = 0; i < profile.length; i++) {
      parts.push(coalesce(layout.mode(i), profile[i]));
    }
    for (let i = profile.length; i < layout.rank(); i++) {
      parts.push(layout.mode(i));
    }
    return make_layout(parts);
  }

  const result_shape  = [1];
  const result_stride = [0];
  const flat_shapes  = flatten(layout.shape);
  const flat_strides = flatten(layout.stride);
  for (let k = 0; k < flat_shapes.length; k++) {
    const shape  = flat_shapes[k];
    const stride = flat_strides[k];
    // skip their shape-1s
    if (shape === 1) {
      continue;
    }
    // replace our shape-1 with anything
    else if (result_shape[result_shape.length - 1] === 1) {
      result_shape[result_shape.length - 1]  = shape;
      result_stride[result_stride.length - 1] = stride;
    }
    // merge modes if the shape*stride match
    else if (result_shape[result_shape.length - 1] * result_stride[result_stride.length - 1] === stride) {
      result_shape[result_shape.length - 1] = result_shape[result_shape.length - 1] * shape;
    }
    // append a new mode
    else {
      result_shape.push(shape);
      result_stride.push(stride);
    }
  }

  if (result_shape.length === 1) {
    return new Layout(result_shape[0], result_stride[0]);
  } else {
    return new Layout(result_shape, result_stride);
  }
}


// Layout filter -- replace all stride-0 modes with size-1 and then coalesce to remove them
function filter(layout, profile) {
  if (is_tuple(profile)) {
    assert(layout.rank() >= profile.length);
    const parts = [];
    for (let i = 0; i < profile.length; i++) {
      parts.push(filter(layout.mode(i), profile[i]));
    }
    for (let i = profile.length; i < layout.rank(); i++) {
      parts.push(layout.mode(i));
    }
    return make_layout(parts);
  }

  const result_shape  = [];
  const result_stride = [];
  const flat_shapes  = flatten(layout.shape);
  const flat_strides = flatten(layout.stride);
  for (let k = 0; k < flat_shapes.length; k++) {
    const shape  = flat_shapes[k];
    const stride = flat_strides[k];
    // skip their shape-1s and stride-0s
    if (!(shape === 1 || stride === 0)) {
      result_shape.push(shape);
      result_stride.push(stride);
    }
  }

  if (result_shape.length === 0) {
    return new Layout(1, 0);
  } else {
    return coalesce(new Layout(result_shape, result_stride));
  }
}


// Layout composition
// Use tuples-of-layouts to perform this operation by-mode and null as no-op
function composition(layoutA, layoutB) {
  if (layoutB === null) {
    return layoutA;
  } else if (is_int(layoutB)) {
    return composition(layoutA, new Layout(layoutB));
  } else if (is_tuple(layoutB)) {
    assert(layoutA.rank() >= layoutB.length);
    const parts = [];
    for (let i = 0; i < layoutB.length; i++) {
      parts.push(composition(layoutA.mode(i), layoutB[i]));
    }
    for (let i = layoutB.length; i < layoutA.rank(); i++) {
      parts.push(layoutA.mode(i));
    }
    return make_layout(parts);
  } else if (is_tuple(layoutB.shape)) {
    return make_layout(layoutB.shape.map((_, i) => composition(layoutA, layoutB.mode(i))));
  }

  if (layoutB.stride === 0) {
    return new Layout(layoutB.shape, 0);
  } else {
    const result_shape  = [];
    const result_stride = [];
    let rest_shape  = layoutB.shape;
    let rest_stride = layoutB.stride;
    const flat_A = coalesce(layoutA);
    const flatA_shapes  = flatten(flat_A.shape);
    const flatA_strides = flatten(flat_A.stride);
    for (let k = 0; k < flatA_shapes.length - 1; k++) {
      const curr_shape  = flatA_shapes[k];
      const curr_stride = flatA_strides[k];
      assert(curr_shape % rest_stride === 0 || rest_stride % curr_shape === 0);
      const new_shape = Math.min(Math.max(1, Math.floor(curr_shape / rest_stride)), rest_shape);

      if (new_shape !== 1) {
        result_shape.push(new_shape);
        result_stride.push(rest_stride * curr_stride);
      }

      rest_shape  = Math.floor(rest_shape / new_shape);
      // Python exclusive impl: "//" is always floor div so == ceil_div(abs(rest_stride), curr_shape) * signum(rest_stride)
      rest_stride = -Math.floor(-rest_stride / curr_shape);
    }

    if (rest_shape !== 1 || result_shape.length === 0) {
      result_shape.push(rest_shape);
      result_stride.push(rest_stride * flatA_strides[flatA_strides.length - 1]);
    }

    if (result_shape.length === 1) {
      return new Layout(result_shape[0], result_stride[0]);
    } else {
      return new Layout(result_shape, result_stride);
    }
  }
}


// Layout complement
function complement(layout, max_idx) {
  if (max_idx === undefined) max_idx = 1;

  if (is_int(layout)) {
    return complement(new Layout(layout));
  }

  const result_shape  = [];
  const result_stride = [];
  let current_idx = 1;

  const flat_strides = flatten(layout.stride);
  const flat_shapes  = flatten(layout.shape);
  // sorted_DS = sorted(zip(flatten(layout.stride), flatten(layout.shape)))
  const sorted_DS = flat_strides.map((s, i) => [s, flat_shapes[i]]);
  sorted_DS.sort((a, b) => a[0] - b[0]);

  for (const [stride, shape] of sorted_DS) {
    if (stride === 0 || shape === 1) {
      continue;
    }

    const in_bound = current_idx <= shape * stride;
    // To support symbolic value which can't be evaluated now
    assert(in_bound);

    result_shape.push(Math.floor(stride / current_idx));
    result_stride.push(current_idx);
    current_idx = shape * stride;
  }

  result_shape.push(Math.floor((max_idx + current_idx - 1) / current_idx));  // ceil_div
  result_stride.push(current_idx);

  return coalesce(new Layout(result_shape, result_stride));
}


// Layout right inverse
function right_inverse(layout) {
  if (layout === null) {
    return null;
  } else if (is_int(layout)) {
    return new Layout(layout);
  }

  const result_shape  = [];
  const result_stride = [];
  let current_idx = 1;

  const flat_shape  = flatten(layout.shape);
  const flat_stride = flatten(layout.stride);
  const flat_pp     = prefix_product(flat_shape);
  // sorted_DSA = sorted(zip(flat_stride, flat_shape, prefix_product(flat_shape)))
  const sorted_DSA = flat_stride.map((s, i) => [s, flat_shape[i], flat_pp[i]]);
  sorted_DSA.sort((a, b) => a[0] - b[0]);

  for (const [stride, shape, rstride] of sorted_DSA) {
    if (shape === 1) {
      continue;
    }
    if (current_idx !== stride) {
      break;
    }

    result_shape.push(shape);
    result_stride.push(rstride);
    current_idx = shape * stride;
  }

  return coalesce(new Layout(result_shape, result_stride));
}


// Layout left inverse
function left_inverse(layout) {
  if (layout === null) {
    return null;
  } else if (is_int(layout)) {
    return new Layout(layout);
  }
  return right_inverse(make_layout(layout, complement(layout)));
}


// Split a layout by the composition of B and the "rest"
// Use tuples-of-layouts to perform this operation by-mode and null as no-op
function logical_divide(layoutA, layoutB) {
  if (layoutB === null) {
    return layoutA;
  } else if (is_int(layoutB)) {
    return logical_divide(layoutA, new Layout(layoutB));
  } else if (is_tuple(layoutB)) {
    assert(layoutA.rank() >= layoutB.length);
    const parts = [];
    for (let i = 0; i < layoutB.length; i++) {
      parts.push(logical_divide(layoutA.mode(i), layoutB[i]));
    }
    for (let i = layoutB.length; i < layoutA.rank(); i++) {
      parts.push(layoutA.mode(i));
    }
    return make_layout(parts);
  }

  return composition(layoutA, make_layout(layoutB, complement(layoutB, size(layoutA))));
}


// Reproduce a layoutA over a layoutB
// Use tuples-of-layouts to perform this operation by-mode and null as no-op
function logical_product(layoutA, layoutB) {
  if (layoutB === null) {
    return layoutA;
  } else if (is_int(layoutB)) {
    return logical_divide(layoutA, new Layout(layoutB));
  } else if (is_tuple(layoutB)) {
    assert(layoutA.rank() >= layoutB.length);
    const parts = [];
    for (let i = 0; i < layoutB.length; i++) {
      parts.push(logical_product(layoutA.mode(i), layoutB[i]));
    }
    for (let i = layoutB.length; i < layoutA.rank(); i++) {
      parts.push(layoutA.mode(i));
    }
    return make_layout(parts);
  }

  return make_layout(layoutA, composition(complement(layoutA, size(layoutA) * cosize(layoutB)), layoutB));
}


// Gather the modes from a hierarchical logical_divide or logical_product
function hier_unzip(splitter, layoutA, layoutB) {
  if (layoutB === null) {
    return make_layout(new Layout(1, 0), layoutA);
  } else if (is_tuple(layoutB)) {
    assert(layoutA.rank() >= layoutB.length);
    // A layout with shape ((A,a),(B,b),(C,c))
    const split = make_layout(
      layoutB.map((_, i) => hier_unzip(splitter, layoutA.mode(i), layoutB[i]))
    );
    // Gather to shape ((A,B,C,...),(a,b,c,...,y,z))
    const firstParts = [];
    for (let i = 0; i < layoutB.length; i++) {
      firstParts.push(split.mode(i).mode(0));
    }
    const secondParts = [];
    for (let i = 0; i < layoutB.length; i++) {
      secondParts.push(split.mode(i).mode(1));
    }
    for (let i = layoutB.length; i < layoutA.rank(); i++) {
      secondParts.push(layoutA.mode(i));
    }
    return make_layout(make_layout(firstParts), make_layout(secondParts));
  }

  // splitter must return a rank-2 layout
  return splitter(layoutA, layoutB);
}


// Apply logical divide hierarchically and gather the split modes into two modes
function zipped_divide(layoutA, layoutB) {
  return hier_unzip(logical_divide, layoutA, layoutB);
}


// Perform logical divide hierarchically and gather tiles (B-layouts) into a new mode
function tiled_divide(layoutA, layoutB) {
  const result = zipped_divide(layoutA, layoutB);
  const parts = [result.mode(0)];
  for (let i = 0; i < result.mode(1).rank(); i++) {
    parts.push(result.mode(1).mode(i));
  }
  return make_layout(parts);
}


// Fully flatten zipped_divide's output -- every leaf of mode(0) and mode(1)
// becomes a top-level mode. Port of include/cute/layout.hpp:1635.
function flat_divide(layoutA, layoutB) {
  const result = zipped_divide(layoutA, layoutB);
  return new Layout(flatten(result.shape), flatten(result.stride));
}


// Apply logical product hierarchically and gather the split modes into two modes
function zipped_product(layoutA, layoutB) {
  return hier_unzip(logical_product, layoutA, layoutB);
}


// Perform logical product hierarchically and gather tiles (B-layouts) into a new mode
function tiled_product(layoutA, layoutB) {
  const result = zipped_product(layoutA, layoutB);
  const parts = [result.mode(0)];
  for (let i = 0; i < result.mode(1).rank(); i++) {
    parts.push(result.mode(1).mode(i));
  }
  return make_layout(parts);
}


// Fully flatten zipped_product's output -- every leaf of mode(0) and mode(1)
// becomes a top-level mode. Port of include/cute/layout.hpp:1712.
function flat_product(layoutA, layoutB) {
  const result = zipped_product(layoutA, layoutB);
  return new Layout(flatten(result.shape), flatten(result.stride));
}


function slice_and_offset(crd, layout) {
  return [
    new Layout(slice_(crd, layout.shape), slice_(crd, layout.stride)),
    crd2idx(crd, layout.shape, layout.stride)
  ];
}


// ═══════════════════════════════════════════════════════
//  Additional helpers: zip, append_layout, product_each, raked_product, make_layout_tv
//  (ported from include/cute/layout.hpp and include/cute/algorithm/tuple_algorithms.hpp)
// ═══════════════════════════════════════════════════════

// Return a rank(t) tuple `result` such that `result[i] = product(t[i])`
// Matches product_each in include/cute/int_tuple.hpp:249
function product_each(t) {
  if (!is_tuple(t)) return [product(t)];
  return t.map(ti => product(ti));
}

// Zip on a nested tuple: transpose rank-R0 x rank-R1 to rank-R1 x rank-R0
// Take       ((a,b,c,...),(x,y,z,...),...)        rank-R0 x rank-R1 input
// to produce ((a,x,...),(b,y,...),(c,z,...),...)  rank-R1 x rank-R0 output
function zip_tuple(t) {
  if (!is_tuple(t)) return t;
  if (!is_tuple(t[0])) return [t];
  const outerRank = t.length;
  const innerRank = t[0].length;
  for (const ti of t) {
    assert(is_tuple(ti) && ti.length === innerRank, "Mismatched ranks in zip");
  }
  const result = [];
  for (let j = 0; j < innerRank; j++) {
    const entry = [];
    for (let i = 0; i < outerRank; i++) entry.push(t[i][j]);
    result.push(entry);
  }
  return result;
}

// Two-arg zip: zip(t0, t1) = zip_tuple([t0, t1])
function zip_tuples(...ts) {
  return zip_tuple(ts);
}

// Zip two or more layouts, pairing up mode i of each
function zip_layouts(...layouts) {
  const shapes  = layouts.map(l => l.shape);
  const strides = layouts.map(l => l.stride);
  return new Layout(zip_tuple(shapes), zip_tuple(strides));
}

// Append a layout to rank N by appending trivial modes (shape 1, stride 0)
function append_layout(layout, N) {
  const shape  = is_tuple(layout.shape)  ? layout.shape.slice()  : [layout.shape];
  const stride = is_tuple(layout.stride) ? layout.stride.slice() : [layout.stride];
  while (shape.length < N) { shape.push(1); stride.push(0); }
  return new Layout(shape, stride);
}

// blocked_product -- Reproduce a block over a tiler with block-contiguity.
// Each output axis carries (block_i, tile_i) merged, so copies of the block
//   are laid down as contiguous sub-blocks of a larger matrix-shaped layout.
// post: rank(result) == max(rank(block), rank(tiler), 2)
// Port of include/cute/layout.hpp:1734. R is floored at 2 so single-mode
// inputs still produce a 2-D-renderable result.
function blocked_product(block, tiler) {
  const R = Math.max(block.rank(), tiler.rank(), 2);
  const result = logical_product(append_layout(block, R), append_layout(tiler, R));
  return zip_layouts(result.mode(0), result.mode(1));
}

// raked_product -- Reproduce a block over a tiler with block-interleaving.
// Think of every element of "tiler" as a "block", interleave those blocks,
//   and return the layout of the resulting structure.
// post: rank(result) == max(rank(block), rank(tiler))
// Port of include/cute/layout.hpp:1752
function raked_product(block, tiler) {
  const R = Math.max(block.rank(), tiler.rank());
  const result = logical_product(append_layout(block, R), append_layout(tiler, R));
  // zip(get<1>(result), get<0>(result)) -- tiler mode paired first, block mode second
  return zip_layouts(result.mode(1), result.mode(0));
}

// Create a thread-value layout by repeating the val_layout over the thr_layout.
// Returns { tiler_mn, layout_tv } where:
//   tiler_mn  -- shape of the MN tile (e.g. [4, 6])
//   layout_tv -- layout mapping (tid, vid) -> flat index into the MN tile
// Port of make_layout_tv in python/CuTeDSL/cutlass/cute/core.py:4096
function make_layout_tv(thr_layout, val_layout) {
  // layout_mn maps (M, N) coords -> (thr_idx, val_idx)
  const layout_mn = raked_product(thr_layout, val_layout);
  const thr_size = size(thr_layout);
  const val_size = size(val_layout);
  const tmp = new Layout([thr_size, val_size]);          // auto col-major
  // layout_tv maps (tid, vid) -> flat MN index (col-major into tiler_mn)
  const layout_tv = composition(right_inverse(layout_mn), tmp);
  const tiler_mn = product_each(layout_mn.shape);
  return { tiler_mn, layout_tv };
}

/** Check if a layout is bijective over its full codomain [0, size(layout)).
 *  Sorted leaf (stride, shape) pairs must form a gap-free basis from stride 1. */
function isBijective(layout) {
  const shapes = flatten(layout.shape);
  const strides = flatten(layout.stride);
  const pairs = shapes.map((s, i) => [strides[i], s])
                      .filter(([_, s]) => s > 1);
  pairs.sort((a, b) => a[0] - b[0]);
  let current = 1;
  for (const [stride, shape] of pairs) {
    if (stride !== current) return false;  // gap or stride-0 collision
    current *= shape;
  }
  return current === layout.size();
}
