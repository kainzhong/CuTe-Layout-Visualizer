#!/usr/bin/env python3
"""OPTIONAL GPU gate: does each Op the tabs offer actually EXECUTE?

    python3 tests/gpu_check.py

NOT part of `npm test`, and not part of `npm run reference` -- both of those are
deliberately GPU-free (see tests/README.md). This is the check you run BEFORE
adding an Op to a tab, because three different things can be true of a DSL Op and
only the third makes it worth shipping:

    1. it CONSTRUCTS      -- the dataclass accepts your parameters
    2. it TRACES          -- make_*_atom returns layouts (this is what
                             gen_reference.py exercises, and all it can exercise)
    3. it LOWERS and RUNS -- MLIR -> NVVM -> PTX, and the numbers come out right

`tests/cases.json` pins (2). Only this script can tell you about (3), and the gap
is real: `cute.nvgpu.warp.mma`'s own support matrix marks TF32 as "N / no DSL
class", yet `MmaTF32Op` exists, lowers, and computes correctly -- that comment
table is stale. Trust the hardware over the docstring, and re-run this when
CuTeDSL is upgraded.

No torch: it is broken in some CUDA images (nccl symbol mismatch) and pulling it
in just to allocate three buffers is not worth the fragility. cuda-python plus
numpy/ml_dtypes is enough.
"""

import sys
import os

if "--warpgroup" in sys.argv:
    os.environ["CUTE_DSL_ARCH"] = "sm_90a"

import numpy as np

try:
    import ml_dtypes
    import cuda.bindings.driver as drv
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import warp
    from cutlass import Float16, BFloat16, Float32, TFloat32, Float8E4M3FN, Float8E5M2
except ImportError as e:                                   # noqa: BLE001
    sys.exit(f"skipped: {e} (needs nvidia-cutlass-dsl, cuda-python, ml_dtypes)")


def ck(ret):
    """Unwrap the cuda-python `(CUresult, *values)` convention."""
    res, *vals = ret if isinstance(ret, tuple) else (ret,)
    if res != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(str(res))
    return vals[0] if len(vals) == 1 else (vals or None)


def to_device(arr):
    ptr = ck(drv.cuMemAlloc(arr.nbytes))
    ck(drv.cuMemcpyHtoD(ptr, arr.ctypes.data, arr.nbytes))
    return ptr


def from_device(ptr, like):
    out = np.empty_like(like)
    ck(drv.cuMemcpyDtoH(out.ctypes.data, ptr, out.nbytes))
    return out


def run_mma(make_op, np_ab, cute_ab, m, n, k):
    """One warp, one tiled MMA, C = A @ B^T. Returns the relative error."""

    @cute.kernel
    def kern(pa: cute.Pointer, pb: cute.Pointer, pc: cute.Pointer):
        tid, _, _ = cute.arch.thread_idx()
        ga = cute.make_tensor(pa, cute.make_layout((m, k), stride=(k, 1)))
        gb = cute.make_tensor(pb, cute.make_layout((n, k), stride=(k, 1)))
        gc = cute.make_tensor(pc, cute.make_layout((m, n), stride=(n, 1)))
        mma = cute.make_tiled_mma(make_op())
        thr = mma.get_slice(tid)
        frg_a = mma.make_fragment_A(thr.partition_A(ga))
        frg_b = mma.make_fragment_B(thr.partition_B(gb))
        gc_thr = thr.partition_C(gc)
        acc = mma.make_fragment_C(gc_thr)
        acc.fill(0.0)
        cute.autovec_copy(thr.partition_A(ga), frg_a)
        cute.autovec_copy(thr.partition_B(gb), frg_b)
        cute.gemm(mma, acc, frg_a, frg_b, acc)
        cute.autovec_copy(acc, gc_thr)

    @cute.jit
    def launch(pa: cute.Pointer, pb: cute.Pointer, pc: cute.Pointer):
        kern(pa, pb, pc).launch(grid=(1, 1, 1), block=(32, 1, 1))

    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k)).astype(np_ab)
    b = rng.standard_normal((n, k)).astype(np_ab)
    c = np.zeros((m, n), np.float32)
    pa, pb, pc = to_device(a), to_device(b), to_device(c)
    ptr = lambda p, t: cute.runtime.make_ptr(  # noqa: E731
        t, int(p), cute.AddressSpace.gmem, assumed_align=16)
    args = (ptr(pa, cute_ab), ptr(pb, cute_ab), ptr(pc, Float32))
    cute.compile(launch, *args)(*args)          # compile == the lowering gate
    ck(drv.cuCtxSynchronize())

    got = from_device(pc, c)
    # Reference from the ALREADY-ROUNDED inputs, so what is measured is the
    # instruction, not the input quantization. tf32 still shows ~1e-3 because
    # the hardware rounds A and B to a 10-bit mantissa internally and numpy does
    # not -- which is itself the evidence that tf32 math really happened.
    ref = a.astype(np.float32) @ b.astype(np.float32).T
    return np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9)


# (label, op factory, numpy dtype, cute dtype, (M,N,K), relative tolerance)
BF16, E4M3, E5M2 = ml_dtypes.bfloat16, ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e5m2
MMA_CASES = [
    ("MmaF16BF16Op f16  m16n8k16", lambda: warp.MmaF16BF16Op(Float16, Float32, (16, 8, 16)),
     np.float16, Float16, (16, 8, 16), 5e-3),
    ("MmaF16BF16Op f16  m16n8k8 ", lambda: warp.MmaF16BF16Op(Float16, Float32, (16, 8, 8)),
     np.float16, Float16, (16, 8, 8), 5e-3),
    ("MmaF16BF16Op bf16 m16n8k16", lambda: warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
     BF16, BFloat16, (16, 8, 16), 5e-2),
    ("MmaF16BF16Op bf16 m16n8k8 ", lambda: warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 8)),
     BF16, BFloat16, (16, 8, 8), 5e-2),
    ("MmaTF32Op         m16n8k8 ", lambda: warp.MmaTF32Op((16, 8, 8)),
     np.float32, TFloat32, (16, 8, 8), 5e-3),
    ("MmaTF32Op         m16n8k4 ", lambda: warp.MmaTF32Op((16, 8, 4)),
     np.float32, TFloat32, (16, 8, 4), 5e-3),
    ("MmaFP8Op     e4m3 m16n8k32", lambda: warp.MmaFP8Op(Float8E4M3FN, Float32, (16, 8, 32)),
     E4M3, Float8E4M3FN, (16, 8, 32), 2e-1),
    ("MmaFP8Op     e4m3 m16n8k16", lambda: warp.MmaFP8Op(Float8E4M3FN, Float32, (16, 8, 16)),
     E4M3, Float8E4M3FN, (16, 8, 16), 2e-1),
    ("MmaFP8Op     e5m2 m16n8k32", lambda: warp.MmaFP8Op(Float8E5M2, Float32, (16, 8, 32)),
     E5M2, Float8E5M2, (16, 8, 32), 4e-1),
    ("MmaFP8Op     e5m2 m16n8k16", lambda: warp.MmaFP8Op(Float8E5M2, Float32, (16, 8, 16)),
     E5M2, Float8E5M2, (16, 8, 16), 4e-1),
]


def run_warpgroup_mma(case, compile_only=False):
    """One 128-thread WGMMA. Stage SMEM operands with scalar stores so this
    gate tests the MMA independently of TMA/cp.async. Check D = A @ B.T + C,
    including nonzero C, mixed input types, major modes and register A.
    """
    from cutlass.cute.nvgpu import warpgroup as wg, OperandMajorMode
    from cutlass.memory import SmemAllocator
    from gen_reference import parse_layout, parse_tiled_mma_perm
    import ast
    import math

    types = {'half_t': (np.float16, Float16), 'bfloat16_t': (BF16, BFloat16),
             'float_e4m3_t': (E4M3, Float8E4M3FN), 'float_e5m2_t': (E5M2, Float8E5M2),
             'int8_t': (np.int8, cutlass.Int8), 'uint8_t': (np.uint8, cutlass.Uint8),
             'float': (np.float32, Float32), 'int32_t': (np.int32, cutlass.Int32)}
    np_a, type_a = types[case['ab_dtype']]
    np_b, type_b = types[case['b_dtype']]
    _, type_c = types[case['acc_dtype']]
    integer = case['op'] == 'wg_int8'
    np_out, type_out = (np.int32, cutlass.Int32) if integer else (np.float32, Float32)
    instruction = (64, case['n'], case['k'])
    m, n, k = case.get('tile_mnk', instruction)
    atom_layout = case.get('atom_layout', '(1,1,1)')
    threads = 128 * math.prod(ast.literal_eval(atom_layout.split(':')[0]))
    register_a = case['a_src'] == 'RMEM'
    am, bm = (getattr(OperandMajorMode, case[x]) for x in ['a_major','b_major'])
    a_kind = getattr(wg.SmemLayoutAtomKind, case['a_major'] + '_INTER')
    b_kind = getattr(wg.SmemLayoutAtomKind, case['b_major'] + '_INTER')

    @cute.kernel
    def kern(pa: cute.Pointer, pb: cute.Pointer, pc: cute.Pointer):
        tid, _, _ = cute.arch.thread_idx()
        ga = cute.make_tensor(pa, cute.make_layout((m,k), stride=(k,1)))
        gb = cute.make_tensor(pb, cute.make_layout((n,k), stride=(k,1)))
        gc = cute.make_tensor(pc, cute.make_layout((m,n), stride=(n,1)))
        if cutlass.const_expr(case['op'] == 'wg_f16bf16'):
            op = wg.MmaF16BF16Op(type_a,type_c,instruction,getattr(wg.OperandSource,case['a_src']),am,bm)
        elif cutlass.const_expr(case['op'] == 'wg_fp8'):
            op = wg.MmaF8Op(type_a,type_b,type_c,instruction,getattr(wg.OperandSource,case['a_src']),am,bm)
        else:
            op = wg.MmaI8Op(type_a,type_b,type_c,instruction,getattr(wg.OperandSource,case['a_src']),am,bm)
        mma = cute.make_tiled_mma(op,parse_layout(atom_layout),parse_tiled_mma_perm(case.get('perm')))
        thr = mma.get_slice(tid)
        allocator = SmemAllocator()
        b_atom = wg.make_smem_layout_atom(b_kind,type_b)
        b_layout = cute.tile_to_shape(b_atom,(n,k),order=(1,0) if cutlass.const_expr(case['b_major'] == 'K') else (0,1))
        sb = allocator.allocate_tensor(type_b,b_layout,byte_alignment=128)
        for j in cutlass.range_constexpr((n*k+threads-1)//threads):
            i = tid + j*threads
            if i < n*k:
                sb[i//k,i%k] = gb[i//k,i%k]
        if cutlass.const_expr(register_a):
            a_part = thr.partition_A(ga)
            fa = cute.make_rmem_tensor(a_part.shape,type_a)
            cute.autovec_copy(a_part,fa)
        else:
            a_atom = wg.make_smem_layout_atom(a_kind,type_a)
            a_layout = cute.tile_to_shape(a_atom,(m,k),order=(1,0) if cutlass.const_expr(case['a_major'] == 'K') else (0,1))
            sa = allocator.allocate_tensor(type_a,a_layout,byte_alignment=128)
            for j in cutlass.range_constexpr((m*k+threads-1)//threads):
                i = tid + j*threads
                if i < m*k:
                    sa[i//k,i%k] = ga[i//k,i%k]
            fa = mma.make_fragment_A(thr.partition_A(sa))
        cute.arch.fence_proxy('async.shared',space='cta')
        cute.arch.sync_threads()
        fb = mma.make_fragment_B(thr.partition_B(sb))
        c_part = thr.partition_C(gc)
        acc = cute.make_rmem_tensor(c_part.shape,type_c)
        acc.fill(1)
        mma.set(wg.Field.ACCUMULATE,True)
        wg.fence()
        cute.gemm(mma,acc,fa,fb,acc)
        wg.commit_group()
        wg.wait_group(0)
        result = cute.make_rmem_tensor(acc.shape,type_out)
        result.store(acc.load().to(type_out))
        cute.autovec_copy(result,c_part)

    @cute.jit
    def launch(pa: cute.Pointer, pb: cute.Pointer, pc: cute.Pointer):
        kern(pa,pb,pc).launch(grid=(1,1,1),block=(threads,1,1))

    rng = np.random.default_rng(42)
    def inputs(dtype, shape):
        # High unsigned values distinguish u8 from accidentally signed s8.
        return rng.integers(128,133,size=shape).astype(dtype) if dtype == np.uint8 else rng.integers(-2,3,size=shape).astype(dtype)
    a, b = inputs(np_a,(m,k)), inputs(np_b,(n,k))
    c = np.zeros((m,n),np_out)
    ptr = lambda p,t: cute.runtime.make_ptr(t,int(p),cute.AddressSpace.gmem,assumed_align=16)
    if compile_only:
        cute.compile(launch,ptr(16,type_a),ptr(16,type_b),ptr(16,type_out))
        return 0.0
    pointers = [to_device(x) for x in [a,b,c]]
    try:
        args = tuple(ptr(p,t) for p,t in zip(pointers,[type_a,type_b,type_out]))
        cute.compile(launch,*args)(*args)
        ck(drv.cuCtxSynchronize())
        got = from_device(pointers[2],c)
        ref = a.astype(np_out) @ b.astype(np_out).T + 1
        return np.abs(got-ref).max() / max(np.abs(ref).max(),1)
    finally:
        for pointer in pointers:
            ck(drv.cuMemFree(pointer))


def warpgroup_main():
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--warpgroup',action='store_true')
    parser.add_argument('--compile-only',action='store_true',help='Lower without executing; does NOT satisfy the GPU gate')
    parser.add_argument('--case',default='',help='Filter corpus case IDs')
    parser.add_argument('--tiled',action='store_true',help='Execute representative WGMMA TiledMMA cases')
    args = parser.parse_args()
    if not args.compile_only:
        ck(drv.cuInit(0))
        dev = ck(drv.cuDeviceGet(0))
        major = ck(drv.cuDeviceGetAttribute(drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev))
        minor = ck(drv.cuDeviceGetAttribute(drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev))
        if (major,minor) != (9,0):
            sys.exit(f'WGMMA requires Hopper SM90a; this device is SM{major}{minor}.')
        ck(drv.cuCtxCreate(0,dev))
    section = 'warpgroup_tiled_mma' if args.tiled else 'warpgroup_mma_atom'
    corpus = json.loads(Path(__file__).with_name('cases.json').read_text())[section]
    if args.tiled:
        reference = json.loads(Path(__file__).with_name('reference.json').read_text())[section]
        # K-split ownership is checked against partition_C by the layout suite;
        # these numerical GEMMs use independent output tiles, with no reduction.
        corpus = [dict(c,tile_mnk=reference[c['id']]['tile_mnk']) for c in corpus if c.get('gpu')]
    cases = [c for c in corpus if args.case in c['id']]
    if not cases:
        sys.exit('No warpgroup cases match --case.')
    failures = []
    for case in cases:
        try:
            error = run_warpgroup_mma(case,args.compile_only)
            if error != 0:
                raise AssertionError(f'relative error {error}: small integer-valued inputs must agree exactly')
            print(f"  ok  {case['id']}",flush=True)
        except Exception as e:
            print(f"  FAIL {case['id']} {type(e).__name__}: {e}",flush=True)
            failures.append(case['id'])
    verb = 'lowered (execution NOT checked)' if args.compile_only else 'compiled and executed'
    print(f'{len(cases)-len(failures)}/{len(cases)} WGMMA cases {verb}')
    return 1 if failures else 0


def main():
    ck(drv.cuInit(0))
    dev = ck(drv.cuDeviceGet(0))
    ck(drv.cuCtxCreate(0, dev))
    name = ck(drv.cuDeviceGetName(64, dev)).decode(errors="replace").strip("\x00")
    major = ck(drv.cuDeviceGetAttribute(
        drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev))
    minor = ck(drv.cuDeviceGetAttribute(
        drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev))
    print(f"{name}  sm_{major}{minor}\n")
    print("make_mma_atom -- every Op the tab offers, compiled and executed")

    failures = []
    for label, make_op, np_ab, cute_ab, (m, n, k), tol in MMA_CASES:
        try:
            rel = run_mma(make_op, np_ab, cute_ab, m, n, k)
        except Exception as e:                              # noqa: BLE001
            print(f"  FAIL  {label}  {type(e).__name__}: {' '.join(str(e).split())[:120]}")
            failures.append(label)
            continue
        ok = rel < tol
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}   rel {rel:.2e}  (tol {tol:.0e})")
        if not ok:
            failures.append(label)

    print()
    if failures:
        print(f"{len(failures)} of {len(MMA_CASES)} did not execute correctly:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"all {len(MMA_CASES)} execute correctly")
    return 0


if __name__ == "__main__":
    sys.exit(warpgroup_main() if "--warpgroup" in sys.argv else main())
