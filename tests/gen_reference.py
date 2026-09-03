#!/usr/bin/env python3
"""Generate tests/reference.json by running tests/cases.json through CuTeDSL.

CuTeDSL is the oracle: every layout op in layout.js / cute.js and every copy
construction in tabs/*.js is a port of something in `cutlass.cute`, so the port
is correct exactly when it agrees with this.

Everything here runs at TRACE time inside a single @cute.jit function -- the
layouts are static Python objects, no GPU and no kernel launch is involved, so
this runs anywhere the `nvidia-cutlass-dsl` wheel installs.

    python3 tests/gen_reference.py            # rewrite tests/reference.json
    python3 tests/gen_reference.py --check    # regenerate and diff, exit 1 on drift

The output is committed so `npm test` works on machines without CuTeDSL. Rerun
this whenever cases.json changes or CuTeDSL is upgraded.
"""

import argparse
import json
import os
import re
import sys

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import common, cpasync, warp

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "cases.json")
REFERENCE = os.path.join(HERE, "reference.json")

DTYPES = {
    "int8_t": cutlass.Int8,
    "uint8_t": cutlass.Uint8,
    "half_t": cutlass.Float16,
    "bfloat16_t": cutlass.BFloat16,
    "int16_t": cutlass.Int16,
    "uint16_t": cutlass.Uint16,
    "float": cutlass.Float32,
    "int32_t": cutlass.Int32,
    "uint32_t": cutlass.Uint32,
    "tfloat32_t": cutlass.TFloat32,
    "double": cutlass.Float64,
    "int64_t": cutlass.Int64,
    "uint64_t": cutlass.Uint64,
    "uint128_t": cutlass.Int128,
}

COPY_OPS = {
    "universal": lambda: common.CopyUniversalOp(),
    "cpasync": lambda: cpasync.CopyG2SOp(),
}

EVAL_CAP = 4096


# ── layout-string parsing ─────────────────────────────────────────────────────
# The corpus stores layouts the way a user types them into the tool. This is the
# Python twin of cute.js's parseValue/topLevelColon; keep the two in step.

def _top_level_colon(s):
    depth = 0
    for i, c in enumerate(s):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == ":" and depth == 0:
            return i
    return -1


def _parse_value(s, allow_basis=False):
    s = s.strip()
    if not s:
        raise ValueError("empty value")
    if s[0] == "(":
        depth, start, els = 0, 1, []
        for i, c in enumerate(s):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    sub = s[start:i].strip()
                    if sub:
                        els.append(_parse_value(sub, allow_basis))
                    break
            elif c == "," and depth == 1:
                els.append(_parse_value(s[start:i], allow_basis))
                start = i + 1
        # cute.js unwraps single-element parens: (10) -> 10
        return els[0] if len(els) == 1 else tuple(els)
    if "@" in s:
        if not allow_basis:
            raise ValueError("basis stride not allowed here: %r" % s)
        k, axis = s.split("@")
        return cute.ScaledBasis(int(k), [int(axis)])
    return int(s)


def parse_layout(s, allow_basis=False):
    """A layout string -> cute.Layout, without any rank padding."""
    s = s.strip()
    ci = _top_level_colon(s)
    if ci == -1:
        return cute.make_layout(_parse_value(s))
    shape = _parse_value(s[:ci].strip())
    stride = _parse_value(s[ci + 1:].strip(), allow_basis)
    return cute.make_layout(shape, stride=stride)


def parse_tiler(spec):
    """A string is a single layout tiler; a list is a by-mode Tiler."""
    if isinstance(spec, list):
        return tuple(parse_layout(x) for x in spec)
    return parse_layout(spec)


def parse_swizzle(spec):
    if spec in (None, "none", ""):
        return None
    b, m, s = [int(x) for x in re.split(r"[,\s]+", spec.strip()) if x != ""]
    return (b, m, s)


# ── canonical stringification ─────────────────────────────────────────────────

def canon(x):
    """Whitespace-free string, the comparison key shared with the JS side."""
    return re.sub(r"\s+", "", str(x))


def _leaves(x):
    """Flatten to a list of scalars. cute.flatten returns the scalar itself for a
    rank-1 layout, so it cannot be iterated blindly."""
    if isinstance(x, tuple):
        return [leaf for e in x for leaf in _leaves(e)]
    return [x]


def has_basis(layout):
    return any(isinstance(s, cute.ScaledBasis) for s in _leaves(cute.flatten(layout.stride)))


def eval_all(layout, cap=EVAL_CAP):
    """layout(i) for every i in the domain (capped), as strings.

    A coordinate (basis-strided) layout maps to a COORDINATE, not a 1-D offset,
    so its values are emitted as "a|b" -- axis-ordered and rank-explicit. That
    sidesteps Python's `(0,)` vs JS's `(0)` printing of a 1-tuple, which is a
    formatting difference and not something worth failing a test over."""
    n = min(cute.size(layout), cap)
    if has_basis(layout):
        return ["|".join(str(int(v)) for v in _leaves(cute.crd2idx(i, layout)))
                for i in range(n)]
    return [canon(cute.crd2idx(i, layout)) for i in range(n)]


def record(layout):
    out = {"str": canon(layout), "size": int(cute.size(layout))}
    if not has_basis(layout):
        out["cosize"] = int(cute.cosize(layout))
    out["eval"] = eval_all(layout)
    return out


# ── the op dispatch ───────────────────────────────────────────────────────────

def run_layout_op(c, allow_basis=False):
    op = c["op"]
    if op == "coalesce":
        a = parse_layout(c["a"], allow_basis)
        if "profile" in c:
            return record(cute.coalesce(a, target_profile=tuple(c["profile"])))
        return record(cute.coalesce(a))
    if op == "filter":
        # NB: cute.filter takes no target_profile, though layout.js's port does
        # (it mirrors the C++ overload). The profiled form is pinned in
        # tests/unit.js instead.
        return record(cute.filter(parse_layout(c["a"], allow_basis)))
    if op in ("right_inverse", "left_inverse"):
        return record(getattr(cute, op)(parse_layout(c["a"], allow_basis)))
    if op == "composition":
        return record(cute.composition(parse_layout(c["a"], allow_basis), parse_tiler(c["b"])))
    if op == "complement":
        return record(cute.complement(parse_layout(c["a"]), c["n"]))
    if op in ("logical_divide", "zipped_divide", "tiled_divide", "flat_divide",
              "logical_product", "zipped_product", "tiled_product", "flat_product",
              "blocked_product", "raked_product"):
        a = parse_layout(c["a"], allow_basis)
        return record(getattr(cute, op)(a, parse_tiler(c["b"])))
    if op == "size":
        return {"value": int(cute.size(parse_layout(c["a"])))}
    if op == "cosize":
        return {"value": int(cute.cosize(parse_layout(c["a"])))}
    if op == "slice_and_offset":
        layout, offset = cute.slice_and_offset(to_crd(c["crd"]), parse_layout(c["a"]))
        return {"str": canon(layout), "offset": int(offset)}
    if op == "product_each":
        return {"value": canon(cute.product_each(_parse_value(c["shape"])))}
    if op == "shape_div":
        return {"value": canon(cute.shape_div(_parse_value(c["shape"]),
                                              _parse_value(c["shape_b"])))}
    if op == "idx2crd":
        return {"value": canon(cute.idx2crd(c["idx"], _parse_value(c["shape"])))}
    if op == "crd2idx":
        return {"value": canon(cute.crd2idx(to_crd(c["crd"]), parse_layout(c["a"])))}
    raise ValueError("unknown op %r" % op)


def to_crd(x):
    """JSON coord -> CuTe coord. null is the slice marker (`None` / `_`)."""
    if isinstance(x, list):
        return tuple(to_crd(e) for e in x)
    return x


def run_make_layout_tv(c):
    thr, val = parse_layout(c["thr"]), parse_layout(c["val"])
    tiler_mn, layout_tv = cute.make_layout_tv(thr, val)
    out = {"tiler_mn": canon(tiler_mn), "layout_tv": record(layout_tv)}
    # Cross-check against the constructor the tab actually models.
    atom = cute.make_copy_atom(common.CopyUniversalOp(), DTYPES[c["dtype"]],
                               num_bits_per_copy=c["bits"])
    tc = cute.make_tiled_copy_tv(atom, thr, val)
    out["tiled_copy_layout_tv"] = canon(tc.layout_tv_tiled)
    out["tiled_copy_tiler_mn"] = canon(tc.tiler_mn)
    return out


def run_make_tiled_copy(c):
    atom = cute.make_copy_atom(common.CopyUniversalOp(), DTYPES[c["dtype"]],
                               num_bits_per_copy=c["bits"])
    layout_tv = parse_layout(c["tv"])
    tiler = _parse_value(c["tiler"])
    tc = cute.make_tiled_copy(atom, layout_tv, tiler)
    return {
        "layout_tv": canon(tc.layout_tv_tiled),
        "tiler_mn": canon(tc.tiler_mn),
        "atom_num_val": int(cute.size(atom.layout_src_tv, mode=[1])),
        "eval": eval_all(layout_tv),
    }


def run_copy_atom(c):
    atom = cute.make_copy_atom(COPY_OPS[c["op"]](), DTYPES[c["dtype"]],
                               num_bits_per_copy=c["bits"])
    return {
        "layout_src_tv": canon(atom.layout_src_tv),
        "layout_dst_tv": canon(atom.layout_dst_tv),
        "num_val": int(cute.size(atom.layout_src_tv, mode=[1])),
    }


LDSM_OPS = {
    "ldsm8x8x16b": warp.LdMatrix8x8x16bOp,
    "ldsm16x8x8b": warp.LdMatrix16x8x8bOp,
    "ldsm16x16x8b": warp.LdMatrix16x16x8bOp,
}


def run_ldmatrix_atom(c):
    # No num_bits_per_copy: _make_trait never reads it for any of these Ops, the
    # instruction width being fixed by the Op. Passing one would silently do
    # nothing, which is exactly the confusion the tab's disabled field avoids.
    kwargs = {"transpose": bool(c["transpose"]),
              "num_matrices": c["num_matrices"]}
    # unpack_bits is only legal on the two 8-bit Ops; where it IS legal it
    # selects the LdsmSzPattern and changes no layout, which the JS side asserts.
    if c.get("unpack_bits"):
        kwargs["unpack_bits"] = c["unpack_bits"]
    op = LDSM_OPS[c.get("op", "ldsm8x8x16b")](**kwargs)
    atom = cute.make_copy_atom(op, DTYPES[c["dtype"]])
    return {
        "thr_id": canon(atom.thr_id),
        "src": record(atom.layout_src_tv),
        "dst": record(atom.layout_dst_tv),
    }


def _smem_layout(spec, sw):
    base = parse_layout(spec)
    if sw is None:
        return base
    return cute.make_composed_layout(cute.make_swizzle(*sw), 0, base)


def _tma_inputs(c):
    dtype = DTYPES[c["dtype"]]
    g = cute.make_tensor(
        cute.make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=16),
        parse_layout(c["gmem"]))
    sw = parse_swizzle(c["swizzle"])
    return dtype, g, _smem_layout(c["smem"], sw), _parse_value(c["tiler"]), sw


def run_tma_atom(c):
    dtype, g, slay, tiler, _sw = _tma_inputs(c)
    atom, tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), g, slay, tiler)
    num_val = int(cute.size(atom.layout_src_tv, mode=[1]))
    return {
        "num_val": num_val,
        "num_bits": num_val * dtype.width,
        "layout_src_tv": canon(atom.layout_src_tv),
        "tma_tensor": canon(tma_tensor.layout),
    }


def run_tma_partition(c):
    dtype, g, slay, tiler, sw = _tma_inputs(c)
    # atom_tiler builds the atom over a SMALLER box than the tile being
    # partitioned, which is what makes TMA_Iter > 1. Without it the atom covers
    # the whole tile and TMA_Iter is degenerate.
    if "atom_tiler" in c:
        atom, gA = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), g,
            _smem_layout(c["atom_smem"], sw), _parse_value(c["atom_tiler"]))
    else:
        atom, gA = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), g, slay, tiler)
    sA = cute.make_tensor(
        cute.make_ptr(dtype, 0, cute.AddressSpace.smem, assumed_align=16), slay)
    sT = cute.group_modes(sA, 0, 2)          # ((tile)) -- no Rest on the smem side
    gT = cute.zipped_divide(gA, tiler)       # ((tile), (rest...)) -- mode 0 IS the tile
    tAsA, tAgA = cpasync.tma_partition(atom, 0, cute.make_layout(1), sT, gT)
    return {
        "num_val": int(cute.size(atom.layout_src_tv, mode=[1])),
        # The pre-grouped tensors, exactly as tma_partition received them. The JS
        # side is fed these strings, so it is tested on CuTe's own output.
        "smem_grouped": canon(sT.layout),
        "gmem_grouped": canon(gT.layout),
        "tAsA": canon(tAsA.layout),
        "tAgA": canon(tAgA.layout),
    }


def run_swizzle(c):
    b, m, s = parse_swizzle(c["bms"])
    base = parse_layout(c["layout"])
    composed = cute.make_composed_layout(cute.make_swizzle(b, m, s), 0, base)
    n = min(cute.size(base), EVAL_CAP)
    return {
        "plain": [int(cute.crd2idx(i, base)) for i in range(n)],
        "swizzled": [int(cute.crd2idx(i, composed)) for i in range(n)],
    }


SECTIONS = [
    ("layout_ops", lambda c: run_layout_op(c, allow_basis=False)),
    ("basis_ops", lambda c: run_layout_op(c, allow_basis=True)),
    ("make_layout_tv", run_make_layout_tv),
    ("make_tiled_copy", run_make_tiled_copy),
    ("copy_atom", run_copy_atom),
    ("ldmatrix_atom", run_ldmatrix_atom),
    ("tma_atom", run_tma_atom),
    ("tma_partition", run_tma_partition),
    ("swizzle", run_swizzle),
]


def build(cases, out, failures):
    for name, fn in SECTIONS:
        section = out.setdefault(name, {})
        for c in cases.get(name, []):
            try:
                section[c["id"]] = fn(c)
            except Exception as e:                       # noqa: BLE001
                failures.append("%s/%s: %s: %s" % (name, c["id"], type(e).__name__, e))


def missing_cases(cases, out, failures):
    """Case ids the trace never produced -- i.e. ones that died mid-trace.
    A case that raised is recorded in `failures` and reported separately."""
    failed = {f.split(":", 1)[0] for f in failures}
    gone = []
    for name, _ in SECTIONS:
        have = out.get(name, {})
        for c in cases.get(name, []):
            key = "%s/%s" % (name, c["id"])
            if c["id"] not in have and key not in failed:
                gone.append(key)
    return gone


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="regenerate and fail if it differs from the committed file")
    args = ap.parse_args()

    with open(CASES) as f:
        cases = json.load(f)

    data, failures = {}, []

    @cute.jit
    def trace():
        # Every layout here is static, so the whole body runs at TRACE time and
        # `data` is fully populated before any MLIR is emitted.
        build(cases, data, failures)

    try:
        trace()
    except Exception as e:                               # noqa: BLE001
        # We only ever wanted the trace-time Python values. Some CuTeDSL entry
        # points (make_tiled_tma_atom in particular) also emit IR, and compiling
        # a few dozen unrelated TMA descriptors in one function can fail to
        # legalize even though every individual trace succeeded. That is a
        # property of this harness, not of the cases -- so it is only an error
        # if a case is actually missing from `data`.
        gone = missing_cases(cases, data, failures)
        if gone:
            print("Trace aborted at %s: %s" % (type(e).__name__, e), file=sys.stderr)
            print("Cases never reached:", file=sys.stderr)
            for g in gone:
                print("  " + g, file=sys.stderr)
            return 1
        print("note: MLIR compilation of the trace failed (%s), but every case was "
              "evaluated at trace time -- results are complete." % type(e).__name__,
              file=sys.stderr)

    if failures:
        print("Cases CuTeDSL could not evaluate:", file=sys.stderr)
        for f in failures:
            print("  " + f, file=sys.stderr)
        return 1

    text = json.dumps(
        {"_generated_by": "tests/gen_reference.py (CuTeDSL oracle) -- do not edit by hand",
         "_cutlass_version": getattr(cutlass, "__version__", "unknown"),
         **data},
        indent=1, sort_keys=True) + "\n"

    if args.check:
        if not os.path.exists(REFERENCE):
            print("reference.json is missing; run without --check", file=sys.stderr)
            return 1
        with open(REFERENCE) as f:
            old = f.read()
        if old != text:
            print("reference.json is out of date -- rerun tests/gen_reference.py",
                  file=sys.stderr)
            return 1
        print("reference.json is up to date")
        return 0

    with open(REFERENCE, "w") as f:
        f.write(text)
    total = sum(len(v) for v in data.values())
    print("wrote %s (%d cases across %d sections)" % (REFERENCE, total, len(data)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
