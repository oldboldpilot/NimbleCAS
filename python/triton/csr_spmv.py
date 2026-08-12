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


def _spmv_configs():
    """Autotune space: (ROWS_PER_PROG, BLOCK_SIZE, num_warps).

    ncu on the earlier one-row-per-program kernel (BLOCK_SIZE=32, num_warps=1)
    showed theoretical occupancy capped at 50 % — block-count-limited: a 1-warp
    block admits only 24 warps/SM (of 48). Letting each program own several rows
    (`ROWS_PER_PROG`) with a 2-D [ROWS_PER_PROG, BLOCK_SIZE] tile puts more warps
    in a block (raising occupancy) and amortizes launch + reduction latency,
    while the BLOCK_SIZE dimension still vectorizes each row's nonzeros. The
    tuner picks the point per mean-nnz/row."""
    configs = []
    for rows in (1, 2, 4, 8, 16):
        for bs in (16, 32, 64):
            for nw in (1, 2, 4):
                configs.append(
                    triton.Config({"ROWS_PER_PROG": rows, "BLOCK_SIZE": bs}, num_warps=nw))
    return configs


@triton.autotune(configs=_spmv_configs(), key=["avg_nnz"])
@triton.jit
def _csr_spmv_kernel(
    row_ptr_ptr,   # *i32 : CSR row offsets, length n_rows + 1
    col_ptr,       # *i32 : column indices, length nnz
    val_ptr,       # *T   : nonzero values, length nnz
    x_ptr,         # *T   : dense input vector
    y_ptr,         # *T   : dense output vector, length n_rows
    n_rows,        # i32  : number of rows
    avg_nnz,       # i32  : nnz // n_rows, the autotune key (not read in-kernel)
    ROWS_PER_PROG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """SpMV over a tile of ROWS_PER_PROG rows: y[r] = sum_k vals[k]*x[cols[k]].

    program_id(0) owns rows [row0, row0 + ROWS_PER_PROG). It loads each row's
    [start, end), then walks the widest row in the tile in BLOCK_SIZE-wide
    chunks, forming a 2-D [ROWS_PER_PROG, BLOCK_SIZE] index masked per row by
    `k < len[r]`, gathering vals and x[cols], and reducing along the nonzero
    axis into a per-row accumulator. Owning several rows per program raises the
    warps-per-block (hence occupancy) that the one-row kernel left on the table.
    """
    pid = tl.program_id(axis=0)
    rows = pid * ROWS_PER_PROG + tl.arange(0, ROWS_PER_PROG)       # [R]
    row_mask = rows < n_rows
    starts = tl.load(row_ptr_ptr + rows, mask=row_mask, other=0)   # [R]
    ends = tl.load(row_ptr_ptr + rows + 1, mask=row_mask, other=0)  # [R]
    lens = ends - starts                                            # [R]
    maxlen = tl.max(lens, axis=0)                                   # scalar loop bound

    acc = tl.zeros([ROWS_PER_PROG], dtype=val_ptr.dtype.element_ty)
    kcol = tl.arange(0, BLOCK_SIZE)                                 # [B]
    for base in range(0, maxlen, BLOCK_SIZE):
        k = base + kcol                                            # [B]
        mask = (k[None, :] < lens[:, None]) & row_mask[:, None]    # [R, B]
        idx = starts[:, None] + k[None, :]                         # [R, B]
        cols = tl.load(col_ptr + idx, mask=mask, other=0)          # [R, B]
        vals = tl.load(val_ptr + idx, mask=mask, other=0.0)        # [R, B]
        xs = tl.load(x_ptr + cols, mask=mask, other=0.0)          # [R, B]
        acc += tl.sum(vals * xs, axis=1)                          # [R]

    tl.store(y_ptr + rows, acc, mask=row_mask)


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
        Advisory only. The chunk width (and warp count) is now chosen by
        ``@triton.autotune`` keyed on the mean nonzeros-per-row; this argument is
        kept for API stability and otherwise ignored.

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

    # ROWS_PER_PROG, BLOCK_SIZE and num_warps are chosen by @triton.autotune
    # (keyed on the mean nonzeros-per-row); the `block_size` argument is retained
    # for API stability but is advisory only. Each program owns ROWS_PER_PROG
    # rows, so the grid divides the row count by that tile height.
    avg_nnz = int(val_t.numel() // n_rows)
    grid = lambda meta: (triton.cdiv(n_rows, meta["ROWS_PER_PROG"]),)
    _csr_spmv_kernel[grid](
        row_ptr_t,
        col_t,
        val_t,
        x_t,
        y,
        n_rows,
        avg_nnz,
    )
    return y


__all__ = ["csr_spmv"]
