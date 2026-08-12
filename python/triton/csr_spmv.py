"""Portable Triton GPU kernel for CSR sparse matrix-vector product (SpMV).

@author Olumuyiwa Oluwasanmi

The GPU numerical mirror of nimblecas.krylov's CPU csr_matvec factory
(src/krylov/krylov.cppm). It computes y = A*x for A held as a Compressed
Sparse Row triple (row_ptr, col_indices, values): one Triton program owns one
row, walks that row's nonzeros in BLOCK_SIZE-wide chunks, gathers the matching
x entries, and reduces the products into a single output scalar y[row]. The
grid is 1-D over the rows, so an arbitrary sparsity pattern is handled without
a rebuild, and because Triton JIT-compiles per device the same source runs on
an sm_120 Blackwell part and older architectures alike.

HONESTY: the authoritative SpMV is the CPU csr_matvec, which accumulates each
row with a sequential std::fma. This kernel reduces each row on the GPU with a
tree/parallel sum (tl.sum) in a DIFFERENT order, so a row's result may differ
from the CPU value in the last bits — each is a valid double, but the CPU path
stays authoritative. float32 and float64 are both supported (float64 exercises
the double-precision path); the last-bit disagreement is only meaningful there.

Run / profile (needs a torch + triton install with CUDA):
    python python/triton/csr_spmv.py
    ncu --set full python python/triton/csr_spmv.py                 # kernel counters
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _csr_spmv_kernel(
    row_ptr_ptr,   # *i32 : CSR row offsets, length n_rows + 1
    col_ptr,       # *i32 : column indices, length nnz
    val_ptr,       # *T   : nonzero values, length nnz
    x_ptr,         # *T   : dense input vector
    y_ptr,         # *T   : dense output vector, length n_rows
    BLOCK_SIZE: tl.constexpr,
):
    """Dot one CSR row against x: y[row] = sum_k vals[k] * x[cols[k]].

    A single program owns the row given by program_id(0). It reads the row's
    [start, end) slice of the value/column arrays, strides over it in
    BLOCK_SIZE-wide chunks masking the ragged tail with idx < (end - start),
    gathers x at the chunk's column indices, and folds the products into a
    scalar accumulator via tl.sum.
    """
    row = tl.program_id(axis=0)
    start = tl.load(row_ptr_ptr + row)
    end = tl.load(row_ptr_ptr + row + 1)
    n = end - start

    acc = tl.zeros((), dtype=val_ptr.dtype.element_ty)
    for base in range(0, n, BLOCK_SIZE):
        idx = base + tl.arange(0, BLOCK_SIZE)
        mask = idx < n
        cols = tl.load(col_ptr + start + idx, mask=mask, other=0)
        vals = tl.load(val_ptr + start + idx, mask=mask, other=0.0)
        xs = tl.load(x_ptr + cols, mask=mask, other=0.0)
        acc += tl.sum(vals * xs, axis=0)

    tl.store(y_ptr + row, acc)


def csr_spmv(
    row_ptr,
    col_indices,
    values,
    x,
    *,
    dtype: torch.dtype | None = None,
    block_size: int = 128,
) -> torch.Tensor:
    """Compute y = A*x for A in CSR form on the GPU.

    Parameters
    ----------
    row_ptr : sequence or 1-D tensor
        CSR row offsets, length n_rows + 1, monotone non-decreasing. Moved to
        int32 on device.
    col_indices : sequence or 1-D tensor
        Column index of every nonzero, length nnz. Moved to int32 on device.
    values : sequence or 1-D tensor
        Nonzero values, length nnz, in CSR order.
    x : sequence or 1-D tensor
        Dense input vector, length equal to the matrix's column count.
    dtype : torch.dtype, optional
        Compute/output float dtype. Defaults to ``x``'s float dtype if it is
        already floating point, otherwise float32. float32 and float64 are both
        supported (float64 exercises the double-precision path).
    block_size : int
        Nonzeros processed per chunk within a row. Must be a power of two.

    Returns
    -------
    torch.Tensor
        A*x on the CUDA device, shape (n_rows,).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("csr_spmv requires a CUDA device")
    device = torch.device("cuda")

    x_t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if dtype is None:
        dtype = x_t.dtype if x_t.is_floating_point() else torch.float32

    row_ptr_t = (
        row_ptr if isinstance(row_ptr, torch.Tensor) else torch.as_tensor(row_ptr)
    ).to(device=device, dtype=torch.int32).contiguous().reshape(-1)
    col_t = (
        col_indices if isinstance(col_indices, torch.Tensor)
        else torch.as_tensor(col_indices)
    ).to(device=device, dtype=torch.int32).contiguous().reshape(-1)
    val_t = (
        values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
    ).to(device=device, dtype=dtype).contiguous().reshape(-1)
    x_t = x_t.to(device=device, dtype=dtype).contiguous().reshape(-1)

    n_rows = row_ptr_t.numel() - 1

    # Empty matrix (no rows) or a degenerate row_ptr: nothing to launch.
    if n_rows <= 0:
        return torch.empty(0, dtype=dtype, device=device)

    y = torch.empty(n_rows, dtype=dtype, device=device)

    # No nonzeros at all: every row dots an empty slice, so y is all zeros.
    if val_t.numel() == 0:
        y.zero_()
        return y

    grid = (n_rows,)
    _csr_spmv_kernel[grid](
        row_ptr_t,
        col_t,
        val_t,
        x_t,
        y,
        BLOCK_SIZE=block_size,
    )
    return y


__all__ = ["csr_spmv"]
