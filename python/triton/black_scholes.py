"""Portable Triton GPU kernel for batched Black-Scholes-Merton option pricing.

@author Olumuyiwa Oluwasanmi

The GPU closed-form counterpart of nimblecas.pricing's Black-Scholes (src/pricing/
pricing.cppm) and the in-engine CUDA `black_scholes_batch` kernel (src/gpu/gpu_kernels.cu):
it prices a whole array of options — one program per BLOCK_SIZE contracts — in a single
launch. It MIRRORS the authoritative CPU closed form to floating-point tolerance (same
d1/d2, same degenerate T==0 / vol==0 collapse to discounted intrinsic); it is a
batch-valuation accelerator, never a second source of truth.

Because Triton JIT-compiles per device, the same source runs on an sm_120 Blackwell part
(e.g. the RTX 5090) and older architectures without a rebuild — the same portability the
existing poly_eval.py / mc_option.py Triton kernels rely on.

HONESTY: the price is a correctly-rounded numerical value (transcendental erf/exp/log), not
claimed exact — identical to the CPU module's boundary.

Run / profile (needs a torch + triton install with CUDA):
    python python/triton/black_scholes.py
    nsys profile -o bs_batch python python/triton/black_scholes.py     # timeline
    ncu --set full python python/triton/black_scholes.py               # kernel counters
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _ncdf(x):
    """Standard-normal CDF Phi(x) = 0.5*(1 + erf(x/sqrt(2)))."""
    return 0.5 * (1.0 + tl.erf(x * 0.7071067811865475244))


@triton.jit
def _bs_kernel(
    spot_ptr, strike_ptr, rate_ptr, div_ptr, vol_ptr, time_ptr, iscall_ptr,
    out_ptr, n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    # Masked-out lanes take harmless defaults (S=K=1 -> log 0; v=0.2, T=1 avoid div-by-zero);
    # their results are never stored.
    S = tl.load(spot_ptr + offs, mask=mask, other=1.0)
    K = tl.load(strike_ptr + offs, mask=mask, other=1.0)
    r = tl.load(rate_ptr + offs, mask=mask, other=0.0)
    q = tl.load(div_ptr + offs, mask=mask, other=0.0)
    v = tl.load(vol_ptr + offs, mask=mask, other=0.2)
    T = tl.load(time_ptr + offs, mask=mask, other=1.0)
    is_call = tl.load(iscall_ptr + offs, mask=mask, other=1)

    sq = tl.sqrt(T)
    dr = tl.exp(-r * T)
    dq = tl.exp(-q * T)
    d1 = (tl.log(S / K) + (r - q + 0.5 * v * v) * T) / (v * sq)
    d2 = d1 - v * sq
    call_px = S * dq * _ncdf(d1) - K * dr * _ncdf(d2)
    put_px = K * dr * _ncdf(-d2) - S * dq * _ncdf(-d1)
    # Degenerate T==0 or vol==0: discounted intrinsic on the forward.
    fwd = S * tl.exp((r - q) * T)
    deg = (T == 0.0) | (v == 0.0)
    px_call = tl.where(deg, dr * tl.maximum(fwd - K, 0.0), call_px)
    px_put = tl.where(deg, dr * tl.maximum(K - fwd, 0.0), put_px)
    price = tl.where(is_call != 0, px_call, px_put)
    tl.store(out_ptr + offs, price, mask=mask)


def black_scholes_batch(spot, strike, rate, dividend, vol, time, is_call,
                        block_size=1024, device="cuda"):
    """Price a batch of options given parallel float64 torch tensors. Returns a prices tensor.

    `is_call` is an int tensor (1 call / 0 put). All inputs must be the same length and live
    on `device`.
    """
    n = spot.numel()
    out = torch.empty(n, dtype=torch.float64, device=device)
    grid = (triton.cdiv(n, block_size),)
    _bs_kernel[grid](
        spot, strike, rate, dividend, vol, time, is_call.to(torch.int32),
        out, n, BLOCK_SIZE=block_size,
    )
    return out


def _cpu_black_scholes(spot, strike, rate, q, vol, T, is_call):
    """Reference closed form for the validation check below."""
    from math import log, sqrt, exp, erf
    def ncdf(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))
    d1 = (log(spot / strike) + (rate - q + 0.5 * vol * vol) * T) / (vol * sqrt(T))
    d2 = d1 - vol * sqrt(T)
    if is_call:
        return spot * exp(-q * T) * ncdf(d1) - strike * exp(-rate * T) * ncdf(d2)
    return strike * exp(-rate * T) * ncdf(-d2) - spot * exp(-q * T) * ncdf(-d1)


if __name__ == "__main__":
    dev = "cuda"
    # A grid of calls and puts over a spot sweep at K=100, r=5%, vol=20%, T=1.
    spots, strikes, rates, divs, vols, times, calls = [], [], [], [], [], [], []
    for s in (80.0, 90.0, 100.0, 110.0, 120.0):
        for c in (1, 0):
            spots.append(s); strikes.append(100.0); rates.append(0.05)
            divs.append(0.0); vols.append(0.2); times.append(1.0); calls.append(c)
    t = lambda xs, dt: torch.tensor(xs, dtype=dt, device=dev)
    prices = black_scholes_batch(t(spots, torch.float64), t(strikes, torch.float64),
                                 t(rates, torch.float64), t(divs, torch.float64),
                                 t(vols, torch.float64), t(times, torch.float64),
                                 t(calls, torch.int32)).cpu().tolist()
    max_err = 0.0
    for i in range(len(spots)):
        ref = _cpu_black_scholes(spots[i], strikes[i], rates[i], divs[i], vols[i], times[i],
                                 bool(calls[i]))
        max_err = max(max_err, abs(prices[i] - ref))
    atm_call = prices[4]  # S=100 call
    print(f"Triton batch BS: {len(spots)} options, max |error| vs CPU closed form = {max_err:.3e}")
    print(f"ATM 1y call = {atm_call:.5f}  (textbook 10.45058)")
    assert max_err < 1e-9, "Triton batch Black-Scholes disagrees with the CPU closed form"
    assert abs(atm_call - 10.4505835) < 1e-4, "ATM call off"
    print("OK: Triton batched Black-Scholes mirrors the CPU closed form.")
