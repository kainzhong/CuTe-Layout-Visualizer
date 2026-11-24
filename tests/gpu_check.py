#!/usr/bin/env python3
"""OPTIONAL GPU gate: does each Op the tabs offer actually EXECUTE?

    python3 tests/gpu_check.py
    python3 tests/gpu_check.py --tma     # TMA store, SM90+
    python3 tests/gpu_check.py --stmatrix  # StMatrix epilogue store, SM90+

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


def run_tma_store(case):
    """One CTA, one TMA store (cpasync.CopyBulkTensorTileS2GOp) of the tile at
    the origin. Threads fill the SMEM tile from `src`, the TMA unit writes it to
    `dst`; the tile must arrive bit-exact and nothing else may be written.
    Returns the number of mismatched elements."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gen_reference import DTYPES, parse_layout, parse_swizzle, _parse_value
    from cutlass.cute.nvgpu import cpasync
    from cutlass.memory import SmemAllocator

    dtype = DTYPES[case["dtype"]]
    gmem_str, smem_str, sw = case["gmem"], case["smem"], parse_swizzle(case["swizzle"])
    tiler_val = _parse_value(case["tiler"])
    # A tma_partition case builds its atom over a smaller box than the tile
    # (atom_smem / atom_tiler), so the tile takes several instructions.
    atom_smem_str = case.get("atom_smem", smem_str)
    atom_tiler_val = _parse_value(case["atom_tiler"]) if "atom_tiler" in case else tiler_val
    tile_size = int(np.prod(tiler_val))
    layout_shape = [int(x) for x in gmem_str.split(":")[0].strip("() ").split(",")]
    origin = (0,) * len(layout_shape)   # the tile at index 0 of every mode, batch included
    threads = min(tile_size, 1024)
    assert tile_size % threads == 0
    smem_bytes = tile_size * dtype.width // 8

    @cute.kernel
    def kern(atom: cute.CopyAtom, tma_dst: cute.Tensor, src: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        alloc = SmemAllocator()
        base = parse_layout(smem_str)
        if cutlass.const_expr(sw is not None):
            sC = alloc.allocate_tensor(dtype, base, 1024, swizzle=cute.make_swizzle(*sw))
        else:
            sC = alloc.allocate_tensor(dtype, base, 128)
        gS = cute.local_tile(src, tiler_val, origin)
        for k in cutlass.range_constexpr(tile_size // threads):
            sC[tid + k * threads] = gS[tid + k * threads]
        # Generic-proxy SMEM writes must be visible to the async (TMA) proxy.
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier()
        gD = cute.local_tile(tma_dst, tiler_val, origin)
        tAsC, tAgC = cpasync.tma_partition(atom, 0, cute.make_layout(1),
                                           cute.group_modes(sC, 0, 2), cute.group_modes(gD, 0, 2))
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp == 0:
            cute.copy(atom, tAsC, tAgC)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=False)

    @cute.jit
    def launch(psrc: cute.Pointer, pdst: cute.Pointer):
        layout = parse_layout(gmem_str)
        src = cute.make_tensor(psrc, layout)
        dst = cute.make_tensor(pdst, layout)
        slay = parse_layout(atom_smem_str)
        if cutlass.const_expr(sw is not None):
            slay = cute.make_composed_layout(cute.make_swizzle(*sw), 0, slay)
        atom, tma_dst = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), dst, slay, atom_tiler_val)
        kern(atom, tma_dst, src).launch(grid=(1, 1, 1), block=(threads, 1, 1),
                                        smem=smem_bytes + 1024)

    stride = [int(x) for x in gmem_str.split(":")[1].strip("() ").split(",")]
    n = sum((e - 1) * d for e, d in zip(layout_shape, stride)) + 1   # cosize: padded batch strides
    nbytes = dtype.width // 8
    rng = np.random.default_rng(1)
    src = rng.integers(1, 255, size=n * nbytes, dtype=np.uint8)
    dst = np.zeros_like(src)
    psrc, pdst = to_device(src), to_device(dst)
    ptr = lambda p: cute.runtime.make_ptr(  # noqa: E731
        dtype, int(p), cute.AddressSpace.gmem, assumed_align=128)
    args = (ptr(psrc), ptr(pdst))
    cute.compile(launch, *args)(*args)
    ck(drv.cuCtxSynchronize())
    got = from_device(pdst, dst).reshape(n, nbytes)
    want = np.zeros_like(got)
    # Expected: exactly the origin tile copied (modes 0 and 1, index 0 of the
    # rest). Offsets come from the flat GMEM layout on the host.
    for i in range(tiler_val[0]):
        for j in range(tiler_val[1]):
            off = i * stride[0] + j * stride[1]
            want[off] = src.reshape(n, nbytes)[off]
    return int(np.any(got != want, axis=1).sum())


def tma_case_isolated(case_id):
    """Run one case in a fresh process. A TMA fault (misaligned or illegal
    address) is sticky: it poisons the CUDA context, so every later case in the
    same process would fail too. Returns (passed, message)."""
    import subprocess
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--tma-one", case_id],
                       capture_output=True, text=True, timeout=600)
    msg = " ".join((p.stdout + p.stderr).split())[-200:]
    return p.returncode == 0, msg


def tma_one_main():
    import json
    from pathlib import Path
    ck(drv.cuInit(0))
    ck(drv.cuCtxCreate(0, ck(drv.cuDeviceGet(0))))
    case_id = sys.argv[sys.argv.index("--tma-one") + 1]
    corpus = json.loads(Path(__file__).with_name("cases.json").read_text())
    case = next(c for c in corpus["tma_atom"] + corpus["tma_partition"] if c["id"] == case_id)
    bad = run_tma_store(case)
    if bad:
        sys.exit(f"{bad} elements differ")
    return 0


def tma_main():
    import json
    from pathlib import Path
    ck(drv.cuInit(0))
    dev = ck(drv.cuDeviceGet(0))
    major = ck(drv.cuDeviceGetAttribute(
        drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev))
    if major < 9:
        sys.exit(f"TMA requires SM90 or newer; this device is SM{major}x.")
    ck(drv.cuCtxCreate(0, dev))
    case_filter = sys.argv[sys.argv.index("--case") + 1] if "--case" in sys.argv else ""
    corpus = json.loads(Path(__file__).with_name("cases.json").read_text())
    # tma_partition cases add atoms smaller than the tile: several instructions
    # per tile, i.e. the TMA_Iter mode that tma_partition produces.
    cases = [c for c in corpus["tma_atom"] + corpus["tma_partition"] if case_filter in c["id"]]
    print("make_tiled_tma_atom -- cpasync.CopyBulkTensorTileS2GOp, compiled and executed")
    failures = []
    for case in cases:
        if case.get("hw_invalid"):
            # CuTe builds this atom, but the hardware must reject it: the tab
            # flags it as an error, and that claim is checked here.
            passed, msg = tma_case_isolated(case["id"])
            if passed:
                print(f"  FAIL  {case['id']}  executed correctly, but is marked hw_invalid "
                      f"({case['hw_invalid']}); the tab's diagnostic is wrong", flush=True)
                failures.append(case["id"])
            else:
                print(f"  ok    {case['id']}  fails on hardware as flagged: {case['hw_invalid']}", flush=True)
            continue
        try:
            bad = run_tma_store(case)
            if bad:
                raise AssertionError(f"{bad} elements differ")
            print(f"  ok    {case['id']}", flush=True)
        except Exception as e:                              # noqa: BLE001
            print(f"  FAIL  {case['id']}  {type(e).__name__}: {' '.join(str(e).split())[:200]}", flush=True)
            failures.append(case["id"])
    invalid = sum(1 for c in cases if c.get("hw_invalid") and c["id"] not in failures)
    valid = len(cases) - len(failures) - invalid
    print(f"{len(cases)} TMA store cases: {valid} executed bit-exact, {invalid} hardware-invalid "
          f"failed as the tab flags them, {len(failures)} FAILED")
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


def run_stmatrix_tiled_c(case, compile_only=False):
    """make_tiled_copy_C with a StMatrix atom, executed: every thread fills its
    accumulator fragment with each element's logical index m*N + n, retiles it
    to the TiledCopy, stmatrix-stores it into SMEM, and the CTA copies SMEM
    out. Every element must land at its own index. stmatrix moves bits, so the
    kernel uses the same-width integer type (16-bit indices stay exact; 8-bit
    ones are compared mod 256). Returns the number of wrong elements."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gen_reference import make_any_mma_op, parse_tiled_mma_perm, parse_layout
    from cutlass.memory import SmemAllocator
    cfg = {"op": case["op"], "ab_dtype": case["ab_dtype"], "acc_dtype": case["acc_dtype"], "k": case["k"]}
    width = {"half_t": 16, "bfloat16_t": 16, "uint16_t": 16, "int8_t": 8, "uint8_t": 8}[
        case.get("copy_dtype") or case["acc_dtype"]]
    ity, nty = (cutlass.Int16, np.int16) if width == 16 else (cutlass.Int8, np.int8)
    st = {"stmatrix": warp.StMatrix8x8x16bOp, "stmatrix16x8x8b": warp.StMatrix16x8x8bOp}[case["copy_op"]]
    trans, nm = bool(case["transpose"]), case["num_matrices"]

    def build():
        tmma = cute.make_tiled_mma(cute.make_mma_atom(make_any_mma_op(cfg)),
                                   parse_layout(case["atom_layout"]), parse_tiled_mma_perm(case["perm"]))
        return tmma, cute.make_tiled_copy_C(cute.make_copy_atom(st(transpose=trans, num_matrices=nm), ity), tmma)

    # Host-side sizes (the same construction, traced once).
    sizes = {}

    @cute.jit
    def probe():
        tmma, _ = build()
        sizes["M"], sizes["N"] = int(tmma.get_tile_size(0)), int(tmma.get_tile_size(1))
        sizes["T"] = int(cute.size(tmma))
    try:
        probe()
    except Exception:                                       # noqa: BLE001
        pass
    M, N, T = sizes["M"], sizes["N"], sizes["T"]
    # stmatrix writes 16 B rows: along N for the plain form, along M for .trans,
    # so the SMEM tile is N-major or M-major to match.
    slay_args = ((M, N), (1, M)) if trans else ((M, N), (N, 1))

    @cute.kernel
    def kern(pout: cute.Pointer):
        tid, _, _ = cute.arch.thread_idx()
        tmma, tc = build()
        thr_mma, thr_copy = tmma.get_slice(tid), tc.get_slice(tid)
        sC = SmemAllocator().allocate_tensor(ity, cute.make_layout(slay_args[0], stride=slay_args[1]), 128)
        tCcC = thr_mma.partition_C(cute.make_identity_tensor((M, N)))
        frag = cute.make_fragment_like(tCcC, ity)
        for i in cutlass.range_constexpr(cute.size(frag)):
            m, n = tCcC[i]
            frag[i] = ity(m * N + n)
        cute.copy(tc, thr_copy.retile(frag), thr_copy.partition_D(sC))
        cute.arch.barrier()
        gOut = cute.make_tensor(pout, cute.make_layout((M, N), stride=(N, 1)))
        for k in cutlass.range_constexpr((M * N + T - 1) // T):
            i = tid + k * T
            if i < M * N:
                gOut[i] = sC[i]

    @cute.jit
    def launch(pout: cute.Pointer):
        kern(pout).launch(grid=(1, 1, 1), block=(T, 1, 1), smem=M * N * width // 8 + 256)

    out = np.full((M, N), -1, nty)
    pout = to_device(out) if not compile_only else 0x1000
    p = cute.runtime.make_ptr(ity, int(pout), cute.AddressSpace.gmem, assumed_align=16)
    compiled = cute.compile(launch, p)
    if compile_only:
        return 0
    compiled(p)
    ck(drv.cuCtxSynchronize())
    got = from_device(pout, out).astype(np.int64)
    want = (np.arange(M * N).reshape(M, N) & (0xFFFF if width == 16 else 0xFF)).astype(np.int64)
    if width == 16:
        want = np.where(want >= 0x8000, want - 0x10000, want)
    else:
        want = np.where(want >= 0x80, want - 0x100, want)
    return int((got != want).sum())


def run_ldmatrix_tiled_ab(case, compile_only=False):
    """make_tiled_copy_A/B with an LdMatrix atom, executed: SMEM holds each
    element's index r*C + c, the TiledCopy ldmatrix-loads it into the MMA
    fragment, and every thread writes each fragment value back to GMEM at the
    fragment's own logical coordinate. Every element must come back as its own
    index. Same-width integer types, as for stmatrix. Returns wrong elements."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gen_reference import make_any_mma_op, parse_tiled_mma_perm, parse_layout
    from cutlass.memory import SmemAllocator
    cfg = {"op": case["op"], "ab_dtype": case["ab_dtype"], "acc_dtype": case["acc_dtype"], "k": case["k"]}
    # The element width, not the Op's unit: FP8 through the b16 Op is 8-bit data.
    width = 8 if case["ab_dtype"].startswith("float_e") or case["ab_dtype"] in ("int8_t", "uint8_t") else 16
    ity, nty = (cutlass.Int16, np.int16) if width == 16 else (cutlass.Int8, np.int8)
    ops = {"ldmatrix": warp.LdMatrix8x8x16bOp, "ldmatrix16x8x8b": warp.LdMatrix16x8x8bOp,
           "ldmatrix16x16x8b": warp.LdMatrix16x16x8bOp, "ldmatrix8x16x8b": warp.LdMatrix8x16x8bOp}
    kwargs = {"transpose": bool(case["transpose"]), "num_matrices": case["num_matrices"]}
    if case.get("unpack_bits"):
        kwargs["unpack_bits"] = case["unpack_bits"]
    operand, trans = case["operand"], bool(case["transpose"])

    def build():
        tmma = cute.make_tiled_mma(cute.make_mma_atom(make_any_mma_op(cfg)),
                                   parse_layout(case["atom_layout"]), parse_tiled_mma_perm(case["perm"]))
        atom = cute.make_copy_atom(ops[case["copy_op"]](**kwargs), ity)
        maker = cute.make_tiled_copy_A if operand == "A" else cute.make_tiled_copy_B
        return tmma, maker(atom, tmma)

    sizes = {}

    @cute.jit
    def probe():
        tmma, _ = build()
        mnk = [int(tmma.get_tile_size(i)) for i in range(3)]
        sizes["R"], sizes["C"] = (mnk[0], mnk[2]) if operand == "A" else (mnk[1], mnk[2])
        sizes["T"] = int(cute.size(tmma))
    try:
        probe()
    except Exception:                                       # noqa: BLE001
        pass
    R, C, T = sizes["R"], sizes["C"], sizes["T"]
    # ldmatrix reads 16 B rows: along K for the plain form, along M/N for .trans.
    # The 8-bit transposing Ops can want the other orientation, so the copy is
    # compiled with the expected one first and the other if that fails.
    orders = [((R, C), (1, R)), ((R, C), (C, 1))] if trans else [((R, C), (C, 1)), ((R, C), (1, R))]
    for slay in orders:
        try:
            return _run_ldmatrix(case, build, ity, nty, width, operand, R, C, T, slay, compile_only)
        except Exception as e:                              # noqa: BLE001
            if "alignment" not in str(e) or slay is orders[-1]:
                raise


def _run_ldmatrix(case, build, ity, nty, width, operand, R, C, T, slay, compile_only):
    from cutlass.memory import SmemAllocator

    @cute.kernel
    def kern(pout: cute.Pointer):
        tid, _, _ = cute.arch.thread_idx()
        tmma, tc = build()
        thr_mma, thr_copy = tmma.get_slice(tid), tc.get_slice(tid)
        sT = SmemAllocator().allocate_tensor(ity, cute.make_layout(slay[0], stride=slay[1]), 128)
        for k in cutlass.range_constexpr((R * C + T - 1) // T):
            i = tid + k * T
            if i < R * C:
                sT[i] = ity((i % R) * C + i // R)        # colexicographic i -> (r, c)
        cute.arch.barrier()
        ident = cute.make_identity_tensor((R, C))
        tCc = thr_mma.partition_A(ident) if operand == "A" else thr_mma.partition_B(ident)
        frag = cute.make_fragment_like(tCc, ity)
        cute.copy(tc, thr_copy.partition_S(sT), thr_copy.retile(frag))
        gOut = cute.make_tensor(pout, cute.make_layout(R * C))
        for i in cutlass.range_constexpr(cute.size(frag)):
            r, c = tCc[i]
            gOut[r * C + c] = frag[i]

    @cute.jit
    def launch(pout: cute.Pointer):
        kern(pout).launch(grid=(1, 1, 1), block=(T, 1, 1), smem=R * C * width // 8 + 256)

    out = np.full(R * C, -1, nty)
    pout = to_device(out) if not compile_only else 0x1000
    p = cute.runtime.make_ptr(ity, int(pout), cute.AddressSpace.gmem, assumed_align=16)
    compiled = cute.compile(launch, p)
    if compile_only:
        return 0
    compiled(p)
    ck(drv.cuCtxSynchronize())
    got = from_device(pout, out).astype(np.int64)
    mask = 0xFFFF if width == 16 else 0xFF
    want = (np.arange(R * C) & mask).astype(np.int64)
    want = np.where(want > mask // 2, want - mask - 1, want)
    return int((got != want).sum())


def ldmatrix_one_main():
    import json
    from pathlib import Path
    ck(drv.cuInit(0))
    ck(drv.cuCtxCreate(0, ck(drv.cuDeviceGet(0))))
    case_id = sys.argv[sys.argv.index("--ldmatrix-one") + 1]
    case = next(c for c in json.loads(Path(__file__).with_name("cases.json").read_text())["make_tiled_copy_ab"]
                if c["id"] == case_id)
    bad = run_ldmatrix_tiled_ab(case)
    if bad:
        sys.exit(f"{bad} elements misplaced")
    return 0


def ldmatrix_main():
    import json
    from pathlib import Path
    compile_only = "--compile-only" in sys.argv
    if not compile_only:
        ck(drv.cuInit(0))
        ck(drv.cuCtxCreate(0, ck(drv.cuDeviceGet(0))))
    case_filter = sys.argv[sys.argv.index("--case") + 1] if "--case" in sys.argv else ""
    cases = [c for c in json.loads(Path(__file__).with_name("cases.json").read_text())["make_tiled_copy_ab"]
             if c["operand"] in ("A", "B") and c["copy_op"].startswith("ldmatrix") and case_filter in c["id"]]
    verb = "compiled for " + os.environ.get("CUTE_DSL_ARCH", "the default arch") if compile_only else "executed"
    print(f"make_tiled_copy_A/B(LdMatrix) -- {verb}")
    failures = []
    for case in cases:
        if compile_only:
            try:
                run_ldmatrix_tiled_ab(case, True)
                print(f"  ok    {case['id']}", flush=True)
            except Exception as e:                          # noqa: BLE001
                print(f"  FAIL  {case['id']}  {type(e).__name__}: {' '.join(str(e).split())[:220]}", flush=True)
                failures.append(case["id"])
            continue
        # One process per case: a fault poisons the CUDA context for the rest.
        import subprocess
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--ldmatrix-one", case["id"]],
                           capture_output=True, text=True, timeout=600)
        if p.returncode == 0:
            print(f"  ok    {case['id']}", flush=True)
        else:
            msg = " ".join((p.stdout + p.stderr).split())
            print(f"  FAIL  {case['id']}  {msg[-220:]}", flush=True)
            failures.append(case["id"])
    print(f"{len(cases) - len(failures)}/{len(cases)} LdMatrix cases {verb}")
    return 1 if failures else 0


TMEM_OPS = ["Ld32x32b", "Ld16x64b", "Ld16x128b", "Ld16x256b", "Ld16x32bx2",
            "St32x32b", "St16x64b", "St16x128b", "St16x256b", "St16x32bx2"]
TMEM_REPEATS = {"Ld16x128b": 64, "St16x128b": 64, "Ld16x256b": 32, "St16x256b": 32}   # CuTeDSL's caps


def run_tmem_roundtrip(op_name, repeat, compile_only=False):
    """One tcgen05.ld or .st Op, executed through make_tmem_copy over a real
    128 x 256 float accumulator in TMEM. The other direction uses the 32x32b
    Op, whose mapping is the PTX definition (thread t <-> lane t). A store test
    writes every element's logical index through the Op under test and reads
    it back with Ld32x32b; a load test writes with St32x32b and reads through
    the Op under test. Every element must come back as its own index, so the
    Op's thread/value ownership (what make_copy_atom draws) is confirmed on the
    hardware. Returns the number of wrong elements."""
    from cutlass.cute.nvgpu import tcgen05
    from cutlass.memory import SmemAllocator
    M, N = 128, 256
    R = getattr(tcgen05.Repetition, f"x{repeat}")
    test = getattr(tcgen05, op_name + "Op")(R)
    is_store = op_name.startswith("St")
    st_op = test if is_store else tcgen05.St32x32bOp(tcgen05.Repetition.x32)
    ld_op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x32) if is_store else test

    @cute.kernel
    def kern(pout: cute.Pointer):
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        holder = SmemAllocator().allocate_tensor(cutlass.Int32, cute.make_layout(4), 16)
        if warp == 0:
            cute.arch.alloc_tmem(N, holder.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.barrier()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, holder.iterator)
        tmma = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            cutlass.Float16, cutlass.Float32, (M, N, 16), tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM, cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K))
        acc_shape = tmma.partition_shape_C((M, N))
        tAcc = cute.make_tensor(tmem_ptr, tmma.make_fragment_C(acc_shape).layout)
        ident = cute.make_identity_tensor(acc_shape)

        st = tcgen05.make_tmem_copy(cute.make_copy_atom(st_op, cutlass.Float32), tAcc)
        thr_st = st.get_slice(tid)
        cS = thr_st.partition_S(ident)
        regs = cute.make_fragment_like(cS, cutlass.Float32)
        for i in cutlass.range_constexpr(cute.size(regs)):
            c = cS[i]
            regs[i] = cutlass.Float32(c[0][0] * N + c[0][1])
        cute.copy(st, regs, thr_st.partition_D(tAcc))
        cute.arch.fence_view_async_tmem_store()
        cute.arch.barrier()

        ld = tcgen05.make_tmem_copy(cute.make_copy_atom(ld_op, cutlass.Float32), tAcc)
        thr_ld = ld.get_slice(tid)
        cD = thr_ld.partition_D(ident)
        back = cute.make_fragment_like(cD, cutlass.Float32)
        cute.copy(ld, thr_ld.partition_S(tAcc), back)
        cute.arch.fence_view_async_tmem_load()
        gOut = cute.make_tensor(pout, cute.make_layout(M * N))
        for i in cutlass.range_constexpr(cute.size(back)):
            c = cD[i]
            gOut[c[0][0] * N + c[0][1]] = back[i]
        cute.arch.barrier()
        if warp == 0:
            cute.arch.dealloc_tmem(tmem_ptr, N)

    @cute.jit
    def launch(pout: cute.Pointer):
        kern(pout).launch(grid=(1, 1, 1), block=(128, 1, 1), smem=256)

    out = np.full(M * N, -1.0, np.float32)
    pout = to_device(out) if not compile_only else 0x1000
    p = cute.runtime.make_ptr(cutlass.Float32, int(pout), cute.AddressSpace.gmem, assumed_align=16)
    compiled = cute.compile(launch, p)
    if compile_only:
        return 0
    compiled(p)
    ck(drv.cuCtxSynchronize())
    got = from_device(pout, out)
    return int((got != np.arange(M * N, dtype=np.float32)).sum())


def run_tmem_copy_case(case, compile_only=False):
    """One `tmem_copy` corpus case on the hardware: TMEM is allocated and viewed
    through the case's own tensor layout and element width; every element's
    index m*N + n is written through one make_tmem_copy TiledCopy and read back
    through another, with the Op under test on its side and a plain partner Op
    (32x32b, or 16x256b when a lane quarter holds only 16 rows) on the other.
    Same-width integer types, compared mod 2^bits. Returns wrong elements."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gen_reference import parse_layout
    from cutlass.cute.nvgpu import tcgen05
    from cutlass.memory import SmemAllocator
    width = {"float": 32, "half_t": 16, "bfloat16_t": 16, "int8_t": 8, "uint8_t": 8}[case["dtype"]]
    ity, nty = {32: (cutlass.Int32, np.int32), 16: (cutlass.Int16, np.int16), 8: (cutlass.Int8, np.int8)}[width]
    is_store = case["op"].startswith("St")
    test = getattr(tcgen05, case["op"] + "Op")(getattr(tcgen05.Repetition, "x%d" % case["repeat"]))
    partner = "16x256b" if case["tensor"].startswith("((16,4)") else "32x32b"
    other = getattr(tcgen05, ("Ld" if is_store else "St") + partner + "Op")(tcgen05.Repetition.x1)
    st_op, ld_op = (test, other) if is_store else (other, test)
    sizes = {}

    @cute.jit
    def probe():
        L = parse_layout(case["tensor"])
        sizes["M"], sizes["N"] = int(cute.size(L, mode=[0])), int(cute.size(L, mode=[1]))
    probe()
    M, N = sizes["M"], sizes["N"]
    need = max(32, 1 << (max(1, (N * width + 31) // 32) - 1).bit_length())   # 32-bit columns, power of 2

    @cute.kernel
    def kern(pout: cute.Pointer):
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        holder = SmemAllocator().allocate_tensor(cutlass.Int32, cute.make_layout(4), 16)
        if warp == 0:
            cute.arch.alloc_tmem(need, holder.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.barrier()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(ity, 16, holder.iterator)
        tT = cute.make_tensor(tmem_ptr, parse_layout(case["tensor"]))
        ident = cute.make_identity_tensor((M, N))
        st = tcgen05.make_tmem_copy(cute.make_copy_atom(st_op, ity), tT)
        thr_st = st.get_slice(tid)
        cS = thr_st.partition_S(ident)
        regs = cute.make_fragment_like(cS, ity)
        for i in cutlass.range_constexpr(cute.size(regs)):
            c = cS[i]
            regs[i] = ity(c[0] * N + c[1])
        cute.copy(st, regs, thr_st.partition_D(tT))
        cute.arch.fence_view_async_tmem_store()
        cute.arch.barrier()
        ld = tcgen05.make_tmem_copy(cute.make_copy_atom(ld_op, ity), tT)
        thr_ld = ld.get_slice(tid)
        cD = thr_ld.partition_D(ident)
        back = cute.make_fragment_like(cD, ity)
        cute.copy(ld, thr_ld.partition_S(tT), back)
        cute.arch.fence_view_async_tmem_load()
        gOut = cute.make_tensor(pout, cute.make_layout(M * N))
        for i in cutlass.range_constexpr(cute.size(back)):
            c = cD[i]
            gOut[c[0] * N + c[1]] = back[i]
        cute.arch.barrier()
        if warp == 0:
            cute.arch.dealloc_tmem(tmem_ptr, need)

    @cute.jit
    def launch(pout: cute.Pointer):
        kern(pout).launch(grid=(1, 1, 1), block=(128, 1, 1), smem=256)

    out = np.full(M * N, -1, nty)
    pout = to_device(out) if not compile_only else 0x1000
    p = cute.runtime.make_ptr(ity, int(pout), cute.AddressSpace.gmem, assumed_align=16)
    compiled = cute.compile(launch, p)
    if compile_only:
        return 0
    compiled(p)
    ck(drv.cuCtxSynchronize())
    got = from_device(pout, out).astype(np.int64)
    mask = (1 << width) - 1
    want = (np.arange(M * N) & mask).astype(np.int64)
    want = np.where(want > mask // 2, want - mask - 1, want)
    return int((got != want).sum())


def tmem_copy_one_main():
    import json
    from pathlib import Path
    ck(drv.cuInit(0))
    ck(drv.cuCtxCreate(0, ck(drv.cuDeviceGet(0))))
    case_id = sys.argv[sys.argv.index("--tmem-copy-one") + 1]
    case = next(c for c in json.loads(Path(__file__).with_name("cases.json").read_text())["tmem_copy"]
                if c["id"] == case_id)
    bad = run_tmem_copy_case(case)
    if bad:
        sys.exit(f"{bad} elements misplaced")
    return 0


def tmem_copy_main():
    """Every executable `tmem_copy` case, one process each. Skipped: the cases
    CuTe rejects, and the ones the tab warns cannot run (fewer than 128 lanes,
    so warps alias onto rows outside their quarter; or a 32-bit lane stride on
    a 16-bit type, whose rows are not TMEM lanes)."""
    import json
    import subprocess
    from pathlib import Path
    case_filter = sys.argv[sys.argv.index("--case") + 1] if "--case" in sys.argv else ""
    ref = json.loads(Path(__file__).with_name("reference.json").read_text())["tmem_copy"]
    cases = [c for c in json.loads(Path(__file__).with_name("cases.json").read_text())["tmem_copy"]
             if case_filter in c["id"] and not ref[c["id"]].get("rejected")
             and "lanes-" not in c["id"] and not c["id"].endswith("float-dp")]
    print("tcgen05.make_tmem_copy -- write every index through TMEM and read it back")
    failures = []
    for case in cases:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--tmem-copy-one", case["id"]],
                           capture_output=True, text=True, timeout=600)
        if p.returncode == 0:
            print(f"  ok    {case['id']}", flush=True)
        else:
            print(f"  FAIL  {case['id']}  {' '.join((p.stdout + p.stderr).split())[-220:]}", flush=True)
            failures.append(case["id"])
    print(f"{len(cases) - len(failures)}/{len(cases)} make_tmem_copy cases executed correctly")
    return 1 if failures else 0


def tmem_one_main():
    ck(drv.cuInit(0))
    ck(drv.cuCtxCreate(0, ck(drv.cuDeviceGet(0))))
    i = sys.argv.index("--tmem-one")
    bad = run_tmem_roundtrip(sys.argv[i + 1], int(sys.argv[i + 2]))
    if bad:
        sys.exit(f"{bad} elements misplaced")
    return 0


def tmem_main():
    """Every tcgen05 ld/st Op at every repetition CuTeDSL allows, each in its
    own process (a TMEM fault poisons the CUDA context)."""
    import subprocess
    case_filter = sys.argv[sys.argv.index("--case") + 1] if "--case" in sys.argv else ""
    cases = [(op, r) for op in TMEM_OPS for r in (1, 2, 4, 8, 16, 32, 64, 128)
             if r <= TMEM_REPEATS.get(op, 128) and case_filter in f"{op}-x{r}"]
    print("make_copy_atom(tcgen05 ld/st) -- round trip through a 128x256 TMEM accumulator")
    failures = []
    for op, r in cases:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--tmem-one", op, str(r)],
                           capture_output=True, text=True, timeout=600)
        if p.returncode == 0:
            print(f"  ok    {op} x{r}", flush=True)
        else:
            print(f"  FAIL  {op} x{r}  {' '.join((p.stdout + p.stderr).split())[-220:]}", flush=True)
            failures.append(f"{op} x{r}")
    print(f"{len(cases) - len(failures)}/{len(cases)} TMEM cases executed correctly")
    return 1 if failures else 0


def stmatrix_main():
    import json
    from pathlib import Path
    compile_only = "--compile-only" in sys.argv
    if not compile_only:
        ck(drv.cuInit(0))
        dev = ck(drv.cuDeviceGet(0))
        major = ck(drv.cuDeviceGetAttribute(
            drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev))
        if major < 9:
            sys.exit(f"stmatrix requires SM90 or newer; this device is SM{major}x.")
        ck(drv.cuCtxCreate(0, dev))
    case_filter = sys.argv[sys.argv.index("--case") + 1] if "--case" in sys.argv else ""
    # Warp-MMA make_tiled_copy_C cases with a StMatrix atom. (WGMMA cases share
    # the instruction; their layouts are oracle-checked in the SM90a worker.)
    cases = [c for c in json.loads(Path(__file__).with_name("cases.json").read_text())["make_tiled_copy_ab"]
             if c["operand"] == "C" and c["copy_op"].startswith("stmatrix") and case_filter in c["id"]]
    print("make_tiled_copy_C(StMatrix) -- retile, stmatrix to SMEM, compare every element")
    failures = []
    for case in cases:
        try:
            bad = run_stmatrix_tiled_c(case, compile_only)
            if bad:
                raise AssertionError(f"{bad} elements misplaced")
            print(f"  ok    {case['id']}", flush=True)
        except Exception as e:                              # noqa: BLE001
            print(f"  FAIL  {case['id']}  {type(e).__name__}: {' '.join(str(e).split())[:220]}", flush=True)
            failures.append(case["id"])
    print(f"{len(cases) - len(failures)}/{len(cases)} StMatrix cases executed correctly")
    return 1 if failures else 0


def run_warp_tiled_copy(case, order, compile_only=False):
    """make_tiled_copy / make_tiled_copy_tv over an ldmatrix or stmatrix atom
    (corpus section `warp_tiled_copy`), executed over an SMEM tile of the given
    orientation: 'n' = (M, N):(N, 1) row-major, 'm' = (M, N):(1, M).

    Load: SMEM holds each element's col-major index m + n*M; the TiledCopy
    loads it into a fragment shaped by partition_D of the identity tensor, and
    every thread writes each register back at the coordinate CuTe assigns it.
    Store: the fragment is filled at partition_S's coordinates, stored, and
    SMEM is copied out. Either way every element must hold its own index, which
    is exactly the claim the tab's SRC / DST panes make. Same-width integer
    types; 8-bit indices compare mod 256. Returns the number of wrong elements."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gen_reference import make_warp_copy_atom, parse_layout, _parse_value
    from cutlass.memory import SmemAllocator
    width = {"half_t": 16, "bfloat16_t": 16, "float": 32, "int8_t": 8, "uint8_t": 8}[case["dtype"]]
    ity, nty = {16: (cutlass.Int16, np.int16), 32: (cutlass.Int32, np.int32), 8: (cutlass.Int8, np.int8)}[width]
    store = case["copy_op"].startswith("stmatrix")

    def build():
        atom = make_warp_copy_atom(case, ity)
        if case["ctor"] == "tv":
            return cute.make_tiled_copy_tv(atom, parse_layout(case["thr"]), parse_layout(case["val"]))
        return cute.make_tiled_copy(atom, parse_layout(case["tv"]), _parse_value(case["tiler"]))

    sizes = {}

    @cute.jit
    def probe():
        tc = build()
        sizes["M"], sizes["N"] = int(cute.size(tc.tiler_mn[0])), int(cute.size(tc.tiler_mn[1]))
        sizes["T"] = int(cute.size(tc.layout_tv_tiled, mode=[0]))
    try:
        probe()
    except Exception:                                       # noqa: BLE001
        pass
    M, N, T = sizes["M"], sizes["N"], sizes["T"]
    slay = ((M, N), (N, 1)) if order == "n" else ((M, N), (1, M))

    @cute.kernel
    def kern(pout: cute.Pointer):
        tid, _, _ = cute.arch.thread_idx()
        tc = build()
        thr = tc.get_slice(tid)
        sT = SmemAllocator().allocate_tensor(ity, cute.make_layout(slay[0], stride=slay[1]), 128)
        ident = cute.make_identity_tensor((M, N))
        gOut = cute.make_tensor(pout, cute.make_layout(M * N))
        if cutlass.const_expr(store):
            tSc = thr.partition_S(ident)
            frag = cute.make_fragment_like(tSc, ity)
            for i in cutlass.range_constexpr(cute.size(frag)):
                m, n = tSc[i]
                frag[i] = ity(m + n * M)
            cute.copy(tc, frag, thr.partition_D(sT))
            cute.arch.barrier()
            for k in cutlass.range_constexpr((M * N + T - 1) // T):
                i = tid + k * T
                if i < M * N:
                    gOut[i] = sT[i]                     # flat i is the colexicographic (m, n)
        else:
            for k in cutlass.range_constexpr((M * N + T - 1) // T):
                i = tid + k * T
                if i < M * N:
                    sT[i] = ity(i)
            cute.arch.barrier()
            tDc = thr.partition_D(ident)
            frag = cute.make_fragment_like(tDc, ity)
            cute.copy(tc, thr.partition_S(sT), frag)
            for i in cutlass.range_constexpr(cute.size(frag)):
                m, n = tDc[i]
                gOut[m + n * M] = frag[i]

    @cute.jit
    def launch(pout: cute.Pointer):
        kern(pout).launch(grid=(1, 1, 1), block=(T, 1, 1), smem=M * N * width // 8 + 256)

    out = np.full(M * N, -1, nty)
    pout = to_device(out) if not compile_only else 0x1000
    p = cute.runtime.make_ptr(ity, int(pout), cute.AddressSpace.gmem, assumed_align=16)
    compiled = cute.compile(launch, p)
    if compile_only:
        return 0
    compiled(p)
    ck(drv.cuCtxSynchronize())
    got = from_device(pout, out).astype(np.int64)
    mask = (1 << width) - 1
    want = (np.arange(M * N) & mask).astype(np.int64)
    want = np.where(want > mask // 2, want - mask - 1, want)
    return int((got != want).sum())


def warp_tiled_one_main():
    import json
    from pathlib import Path
    ck(drv.cuInit(0))
    ck(drv.cuCtxCreate(0, ck(drv.cuDeviceGet(0))))
    i = sys.argv.index("--warp-tiled-one")
    case_id, order = sys.argv[i + 1], sys.argv[i + 2]
    case = next(c for c in json.loads(Path(__file__).with_name("cases.json").read_text())["warp_tiled_copy"]
                if c["id"] == case_id)
    bad = run_warp_tiled_copy(case, order)
    if bad:
        sys.exit(f"{bad} elements misplaced")
    return 0


def warp_tiled_main():
    """Every warp_tiled_copy case over both SMEM orientations, one process per
    run (a fault poisons the CUDA context). The gate fails only when an
    execution misplaces data or errors unexpectedly: the compiler rejecting an
    orientation whose 16 B address rows are not contiguous is correct
    behaviour, and the per-orientation table is what the tab's address-row
    check (mtcWarpRowCheck) predicts."""
    import json
    import subprocess
    from pathlib import Path
    case_filter = sys.argv[sys.argv.index("--case") + 1] if "--case" in sys.argv else ""
    cases = [c for c in json.loads(Path(__file__).with_name("cases.json").read_text())["warp_tiled_copy"]
             if case_filter in c["id"]]
    print("make_tiled_copy(_tv) over LdMatrix / StMatrix -- executed over N-major and M-major SMEM")
    failures, executed = [], 0
    for case in cases:
        cells = []
        for order in ("n", "m"):
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "--warp-tiled-one", case["id"], order],
                               capture_output=True, text=True, timeout=600)
            msg = " ".join((p.stdout + p.stderr).split())
            cells.append("ok" if p.returncode == 0
                         else "misplaced" if "misplaced" in msg
                         else "rejected" if ("vectoriz" in msg or "alignment" in msg or "divisib" in msg)
                         else "error: " + msg[-160:])
        bad = any(c not in ("ok", "rejected") for c in cells)
        runs = "ok" in cells
        print(f"  {'FAIL' if bad else 'ok  '}  {case['id']:<22} N-major: {cells[0]:<10} M-major: {cells[1]}", flush=True)
        if bad:
            failures.append(case["id"])
        executed += runs
    print(f"{executed}/{len(cases)} execute correctly in a plain SMEM orientation; the rest are rejected "
          f"by the compiler. {len(failures)} misplaced or errored.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(warpgroup_main() if "--warpgroup" in sys.argv
             else warp_tiled_one_main() if "--warp-tiled-one" in sys.argv
             else warp_tiled_main() if "--warp-tiled" in sys.argv
             else tma_one_main() if "--tma-one" in sys.argv
             else tmem_copy_one_main() if "--tmem-copy-one" in sys.argv
             else tmem_copy_main() if "--tmem-copy" in sys.argv
             else tmem_one_main() if "--tmem-one" in sys.argv
             else tmem_main() if "--tmem" in sys.argv
             else stmatrix_main() if "--stmatrix" in sys.argv
             else ldmatrix_one_main() if "--ldmatrix-one" in sys.argv
             else ldmatrix_main() if "--ldmatrix" in sys.argv
             else tma_main() if "--tma" in sys.argv else main())
