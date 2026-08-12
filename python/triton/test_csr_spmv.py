"""Correctness harness for the Triton CSR sparse matrix-vector product.

@author Olumuyiwa Oluwasanmi

Compares the Triton GPU SpMV in csr_spmv.py against a dense torch-CPU reference
(A_dense @ x) over a few small matrices with KNOWN dense equivalents — a
hand-verified 3x3 with zeros, a diagonal, and a random sparse case — using an
allclose tolerance. Prints PASS/FAIL per case.

Run with the project venv, e.g.
    /scratch/NimbleCAS/.venv/bin/python test_csr_spmv.py
"""

from __future__ import annotations

import torch

from csr_spmv import csr_spmv


def dense_to_csr(a: torch.Tensor):
    """Build a CSR triple (row_ptr, col_indices, values) from a dense matrix."""
    n_rows, n_cols = a.shape
    row_ptr = [0]
    col_indices: list[int] = []
    values: list[float] = []
    for i in range(n_rows):
        for j in range(n_cols):
            v = a[i, j].item()
            if v != 0.0:
                col_indices.append(j)
                values.append(v)
        row_ptr.append(len(values))
    return row_ptr, col_indices, values


def _tol(dtype: torch.dtype) -> dict:
    if dtype == torch.float64:
        return {"rtol": 0.0, "atol": 1e-9}
    return {"rtol": 1e-4, "atol": 1e-3}


def run_dense_case(name: str, a: torch.Tensor, x: torch.Tensor,
                   dtype: torch.dtype) -> bool:
    a_d = a.to(dtype)
    x_d = x.to(dtype)
    row_ptr, col_indices, values = dense_to_csr(a_d)

    got = csr_spmv(row_ptr, col_indices, values, x_d, dtype=dtype).cpu()
    ref = a_d @ x_d

    ok = torch.allclose(got, ref, **_tol(dtype))
    if not ok:
        diff = (got - ref).abs()
        idx = int(torch.argmax(diff))
        print(f"      max_abs_err={diff.max().item():.3e} at i={idx} "
              f"got={got[idx].item():.6g} ref={ref[idx].item():.6g}")
    tag = f"[{str(dtype).replace('torch.', '')}] {name}"
    print(f"  {'PASS' if ok else 'FAIL'}  {tag}")
    return ok


def run_oracle() -> bool:
    """Hand-verified tiny oracle: A=[[2,0,1],[0,3,0],[1,0,4]], x=[1,2,3].

    row0: 2*1 + 1*3 = 5;  row1: 3*2 = 6;  row2: 1*1 + 4*3 = 13  ->  y=[5,6,13].
    """
    row_ptr = [0, 2, 3, 5]
    col_indices = [0, 2, 1, 0, 2]
    values = [2.0, 1.0, 3.0, 1.0, 4.0]
    x = [1.0, 2.0, 3.0]
    expected = torch.tensor([5.0, 6.0, 13.0], dtype=torch.float64)

    got = csr_spmv(row_ptr, col_indices, values, x, dtype=torch.float64).cpu()
    ok = torch.allclose(got, expected, rtol=0.0, atol=1e-12)
    if not ok:
        print(f"      got={got.tolist()} expected={expected.tolist()}")
    print(f"  {'PASS' if ok else 'FAIL'}  [float64] oracle A*x -> [5, 6, 13]")
    return ok


# (name, dense A, x) cases with known dense equivalents.
def _cases():
    a1 = torch.tensor([[2.0, 0.0, 1.0],
                       [0.0, 3.0, 0.0],
                       [1.0, 0.0, 4.0]])
    x1 = torch.tensor([1.0, 2.0, 3.0])

    a2 = torch.diag(torch.tensor([5.0, -2.0, 7.0, 0.5]))
    x2 = torch.tensor([2.0, 3.0, -1.0, 8.0])

    torch.manual_seed(0)
    a3 = torch.rand(6, 6) * 2.0 - 1.0
    a3[a3.abs() < 0.5] = 0.0            # ~half the entries zeroed -> genuinely sparse
    x3 = torch.rand(6) * 2.0 - 1.0

    return [
        ("3x3 with zeros", a1, x1),
        ("4x4 diagonal", a2, x2),
        ("6x6 random sparse", a3, x3),
    ]


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

    print("\n== Oracle (hand-verified) ==")
    all_ok &= run_oracle()

    print("\n== Correctness vs dense reference ==")
    for name, a, x in _cases():
        for dtype in (torch.float32, torch.float64):
            all_ok &= run_dense_case(name, a, x, dtype)

    print(f"\n{'ALL PASS' if all_ok else 'SOME FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
