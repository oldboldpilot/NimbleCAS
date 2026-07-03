"""Correctness + throughput harness for the Triton batch polynomial evaluator.

@author Olumuyiwa Oluwasanmi

Compares the Triton GPU kernel in poly_eval.py against a NumPy/torch-CPU Horner
reference across several polynomials (constant, linear, degree-4, degree-8) and
batch sizes (small and 20M), using an allclose tolerance. Prints PASS/FAIL per
case and reports achieved throughput (Melem/s) for the large batch.

Run with the project venv, e.g.
    /scratch/NimbleCAS/.venv/bin/python test_poly_eval.py
"""

from __future__ import annotations

import time

import torch

from poly_eval import poly_eval


def horner_reference(coeffs, x: torch.Tensor) -> torch.Tensor:
    """CPU Horner reference matching poly_eval's low-degree-first convention."""
    ref = torch.zeros_like(x)
    for c in reversed(list(coeffs)):
        ref = ref * x + torch.as_tensor(c, dtype=x.dtype)
    return ref


def _tol(dtype: torch.dtype) -> dict:
    if dtype == torch.float64:
        return {"rtol": 1e-11, "atol": 1e-9}
    return {"rtol": 1e-4, "atol": 1e-3}


CASES = [
    ("constant  p=7",             [7.0]),
    ("linear    2 + 3x",          [2.0, 3.0]),
    ("degree-4  1-2x+3x^2-x^3+.5x^4", [1.0, -2.0, 3.0, -1.0, 0.5]),
    ("degree-8",                  [1.0, 0.5, -0.25, 0.125, -0.0625,
                                   0.03125, -0.015625, 0.0078125, -0.00390625]),
]

SMALL_N = 4096
LARGE_N = 20_000_000


def run_case(name: str, coeffs: list, n: int, dtype: torch.dtype) -> bool:
    # Bounded inputs keep degree-8 float32 within tolerance while still
    # exercising all coefficients.
    torch.manual_seed(0)
    x_cpu = (torch.rand(n, dtype=dtype) * 2.0 - 1.0)  # in [-1, 1]

    got = poly_eval(coeffs, x_cpu, dtype=dtype).cpu()
    ref = horner_reference(coeffs, x_cpu)

    ok = torch.allclose(got, ref, **_tol(dtype))
    if not ok:
        diff = (got - ref).abs()
        idx = int(torch.argmax(diff))
        print(f"      max_abs_err={diff.max().item():.3e} at i={idx} "
              f"got={got[idx].item():.6g} ref={ref[idx].item():.6g}")
    tag = f"[{str(dtype).replace('torch.', '')}] {name} n={n}"
    print(f"  {'PASS' if ok else 'FAIL'}  {tag}")
    return ok


def benchmark_large(dtype: torch.dtype) -> None:
    coeffs = CASES[2][1]  # degree-4
    n = LARGE_N
    x_cpu = (torch.rand(n, dtype=dtype) * 2.0 - 1.0)
    x_gpu = x_cpu.cuda()
    torch.cuda.synchronize()

    # Warm up (triggers Triton JIT compile) then time the kernel-only path.
    _ = poly_eval(coeffs, x_gpu, dtype=dtype)
    torch.cuda.synchronize()

    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = poly_eval(coeffs, x_gpu, dtype=dtype)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    per_call = (t1 - t0) / reps
    melem_s = n / per_call / 1e6
    print(f"  throughput [{str(dtype).replace('torch.', '')}] degree-4 "
          f"n={n}: {melem_s:,.1f} Melem/s ({per_call * 1e3:.3f} ms/call, "
          f"kernel-only, inputs preloaded on GPU)")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available -- cannot run Triton kernel")
        return 2

    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"Device: {dev}  (sm_{cap[0]}{cap[1]})")
    print(f"torch {torch.__version__}")
    try:
        import triton
        print(f"triton {triton.__version__}")
    except Exception:
        pass

    all_ok = True

    print("\n== Correctness (small batch) ==")
    for name, coeffs in CASES:
        for dtype in (torch.float32, torch.float64):
            all_ok &= run_case(name, coeffs, SMALL_N, dtype)

    print("\n== Correctness (large batch, 20M) ==")
    for name, coeffs in CASES:
        all_ok &= run_case(name, coeffs, LARGE_N, torch.float32)
    all_ok &= run_case(CASES[2][0], CASES[2][1], LARGE_N, torch.float64)

    print("\n== Throughput ==")
    benchmark_large(torch.float32)
    benchmark_large(torch.float64)

    print(f"\n{'ALL PASS' if all_ok else 'SOME FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
