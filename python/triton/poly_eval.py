"""Portable Triton GPU kernel for batch polynomial evaluation (Horner's method).

@author Olumuyiwa Oluwasanmi

Complements the raw-CUDA kernel in src/gpu/gpu_kernels.cu. Because Triton
JIT-compiles per device, the same source accommodates different kinds of GPU
(e.g. an sm_120 Blackwell part and an sm_90 Hopper part) without a rebuild.

The kernel evaluates p(x) = sum_k coeffs[k] * x**k  (coeffs low-degree-first)
at every element of an input tensor x, returning a tensor of p(x_i). It tiles
the input into BLOCK_SIZE-wide programs indexed by a 1-D program-id grid and
masks the ragged tail so an arbitrary element count is handled correctly.

The coefficient vector is passed as a small device tensor and consumed by an
in-kernel Horner loop; n_coeffs is a `tl.constexpr` so the loop is unrolled and
specialized at compile time for each distinct polynomial degree.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _poly_eval_kernel(
    coeffs_ptr,  # *T   : coefficients, low-degree-first, length n_coeffs
    x_ptr,       # *T   : input points
    out_ptr,     # *T   : output p(x_i)
    n_elements,  # i32  : number of points
    n_coeffs: tl.constexpr,   # unrolled Horner length
    BLOCK_SIZE: tl.constexpr,
):
    """Horner evaluation of one BLOCK_SIZE-wide tile of points.

    acc starts at the highest-degree coefficient and folds downward:
        acc = ((c_{n-1} * x + c_{n-2}) * x + ...) * x + c_0
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Seed accumulator with the leading coefficient in the element dtype, then
    # fold the remaining coefficients in descending degree. tl.load of a single
    # scalar element keeps the coefficient on-device and dtype-correct.
    acc = tl.load(coeffs_ptr + (n_coeffs - 1))
    acc = acc + tl.zeros_like(x)
    for k in tl.static_range(n_coeffs - 2, -1, -1):
        ck = tl.load(coeffs_ptr + k)
        acc = acc * x + ck

    tl.store(out_ptr + offsets, acc, mask=mask)


def poly_eval(
    coeffs,
    x,
    *,
    dtype: torch.dtype | None = None,
    block_size: int = 1024,
) -> torch.Tensor:
    """Evaluate p(x) = sum_k coeffs[k] * x**k at every element of ``x`` on GPU.

    Parameters
    ----------
    coeffs : sequence or 1-D tensor
        Polynomial coefficients, low-degree-first (coeffs[0] is the constant
        term). The empty vector denotes the zero polynomial.
    x : sequence or tensor
        Points to evaluate. Any shape; the result matches its shape.
    dtype : torch.dtype, optional
        Compute/output dtype. Defaults to ``x``'s float dtype if it already is
        floating point, otherwise float32. float32 and float64 are both
        supported (float64 exercises the double-precision path).
    block_size : int
        Tile width (elements per Triton program). Must be a power of two.

    Returns
    -------
    torch.Tensor
        p(x_i) on the same CUDA device, shaped like ``x``.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("poly_eval requires a CUDA device")
    device = torch.device("cuda")

    x_t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if dtype is None:
        dtype = x_t.dtype if x_t.is_floating_point() else torch.float32
    x_t = x_t.to(device=device, dtype=dtype).contiguous()

    out_shape = x_t.shape
    x_flat = x_t.reshape(-1)
    n = x_flat.numel()
    out_flat = torch.empty_like(x_flat)

    coeffs_t = (
        coeffs if isinstance(coeffs, torch.Tensor) else torch.as_tensor(coeffs)
    ).to(device=device, dtype=dtype).contiguous().reshape(-1)
    n_coeffs = coeffs_t.numel()

    # Zero polynomial (or nothing to do): result is all zeros.
    if n_coeffs == 0 or n == 0:
        out_flat.zero_()
        return out_flat.reshape(out_shape)

    grid = (triton.cdiv(n, block_size),)
    _poly_eval_kernel[grid](
        coeffs_t,
        x_flat,
        out_flat,
        n,
        n_coeffs=n_coeffs,
        BLOCK_SIZE=block_size,
    )
    return out_flat.reshape(out_shape)


__all__ = ["poly_eval"]
