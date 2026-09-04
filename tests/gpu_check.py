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
    sys.exit(main())
