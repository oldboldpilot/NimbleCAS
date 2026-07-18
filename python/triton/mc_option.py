"""Portable Triton GPU kernel for Monte Carlo European option pricing.

@author Olumuyiwa Oluwasanmi

The GPU counterpart of nimblecas.pricing's CPU Monte Carlo (src/pricing/pricing.cppm).
It prices a European call/put under geometric Brownian motion by simulating terminal
underlying prices and averaging the discounted payoff. Like the CPU engine it is
REPRODUCIBLE and PARTITION-INDEPENDENT: each path's normal draw is a pure function of its
global index via a counter-based hash (a Triton port of splitmix64 + the inverse-normal
CDF), so equal seeds reproduce equal results regardless of grid shape, and antithetic
variates halve the variance.

HONESTY: the price is a STATISTICAL estimate carrying Monte Carlo standard error, not an
exact value — identical to the CPU module's boundary. The kernel returns the per-block
partial sums of the discounted payoff and its square; the host reduces them to the mean and
the standard error.

Because Triton JIT-compiles per device, the same source runs on an sm_120 Blackwell part
(e.g. the RTX 5090) and older architectures without a rebuild.

Run / profile (needs a torch + triton install with CUDA):
    python python/triton/mc_option.py
    nsys profile -o mc_option python python/triton/mc_option.py     # timeline
    ncu --set full python python/triton/mc_option.py                # kernel counters
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _splitmix64(x):
    """Triton port of splitmix64 — a counter -> well-distributed 64-bit hash (in-kernel RNG)."""
    x = x + 0x9E3779B97F4A7C15
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


@triton.jit
def _counter_u64(key, counter):
    """Two-round counter mix matching the reproducibility contract of nimblecas.rng."""
    return _splitmix64(key ^ _splitmix64(counter))


@triton.jit
def _inv_norm(p):
    """Acklam's rational inverse standard-normal CDF (no Halley step), matching the CPU
    hot-path fast_inv_norm in src/pricing/pricing.cppm."""
    a0, a1, a2 = -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02
    a3, a4, a5 = 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00
    b0, b1, b2 = -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02
    b3, b4 = 6.680131188771972e01, -1.328068155288572e01
    c0, c1, c2 = -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00
    c3, c4, c5 = -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00
    d0, d1, d2, d3 = 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00
    plow = 0.02425
    # Central region.
    q = p - 0.5
    r = q * q
    central = (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q / \
              (((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1.0)
    # Lower tail.
    ql = tl.sqrt(-2.0 * tl.log(tl.where(p < plow, p, plow)))
    lower = (((((c0 * ql + c1) * ql + c2) * ql + c3) * ql + c4) * ql + c5) / \
            ((((d0 * ql + d1) * ql + d2) * ql + d3) * ql + 1.0)
    # Upper tail.
    qu = tl.sqrt(-2.0 * tl.log(tl.where(p > 1.0 - plow, 1.0 - p, plow)))
    upper = -(((((c0 * qu + c1) * qu + c2) * qu + c3) * qu + c4) * qu + c5) / \
            ((((d0 * qu + d1) * qu + d2) * qu + d3) * qu + 1.0)
    z = tl.where(p < plow, lower, tl.where(p > 1.0 - plow, upper, central))
    return z


@triton.jit
def _mc_kernel(
    sum_ptr,      # *f64 : per-program partial sum of discounted payoff
    sumsq_ptr,    # *f64 : per-program partial sum of squares
    n_paths,      # i32  : number of antithetic path pairs
    spot, strike, drift, vol_sqrtT, disc, is_call, seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_paths
    # A normal per path index (partition-independent), then the antithetic pair (+z, -z).
    key = _counter_u64(seed.to(tl.uint64), tl.zeros_like(offs).to(tl.uint64))
    bits = _counter_u64(key, offs.to(tl.uint64))
    u = (bits >> 11).to(tl.float64) * (1.0 / 9007199254740992.0)   # 53-bit uniform in [0,1)
    u = tl.maximum(tl.minimum(u, 1.0 - 1e-15), 1e-15)
    z = _inv_norm(u)
    sp = spot * tl.exp(drift + vol_sqrtT * z)
    sm = spot * tl.exp(drift - vol_sqrtT * z)
    if is_call:
        payoff = 0.5 * (tl.maximum(sp - strike, 0.0) + tl.maximum(sm - strike, 0.0)) * disc
    else:
        payoff = 0.5 * (tl.maximum(strike - sp, 0.0) + tl.maximum(strike - sm, 0.0)) * disc
    payoff = tl.where(mask, payoff, 0.0)
    tl.store(sum_ptr + pid, tl.sum(payoff, axis=0))
    tl.store(sumsq_ptr + pid, tl.sum(payoff * payoff, axis=0))


def mc_european(spot, strike, rate, dividend_yield, volatility, expiry, is_call,
                paths, seed, block_size=1024, device="cuda"):
    """Price a European option by GPU Monte Carlo. Returns (price, std_error)."""
    n_blocks = triton.cdiv(paths, block_size)
    part_sum = torch.zeros(n_blocks, dtype=torch.float64, device=device)
    part_sumsq = torch.zeros(n_blocks, dtype=torch.float64, device=device)
    drift = (rate - dividend_yield - 0.5 * volatility * volatility) * expiry
    vol_sqrtT = volatility * math.sqrt(expiry)
    disc = math.exp(-rate * expiry)
    _mc_kernel[(n_blocks,)](
        part_sum, part_sumsq, paths,
        float(spot), float(strike), float(drift), float(vol_sqrtT), float(disc),
        1 if is_call else 0, int(seed),
        BLOCK_SIZE=block_size,
    )
    total = part_sum.sum().item()
    total_sq = part_sumsq.sum().item()
    mean = total / paths
    var = max((total_sq - paths * mean * mean) / max(paths - 1, 1), 0.0)
    return mean, math.sqrt(var / paths)


def _black_scholes(spot, strike, rate, q, vol, T, is_call):
    """Reference closed form for the validation check below."""
    from math import log, sqrt, exp, erf
    def ncdf(x): return 0.5 * (1.0 + erf(x / sqrt(2.0)))
    d1 = (log(spot / strike) + (rate - q + 0.5 * vol * vol) * T) / (vol * sqrt(T))
    d2 = d1 - vol * sqrt(T)
    if is_call:
        return spot * exp(-q * T) * ncdf(d1) - strike * exp(-rate * T) * ncdf(d2)
    return strike * exp(-rate * T) * ncdf(-d2) - spot * exp(-q * T) * ncdf(-d1)


if __name__ == "__main__":
    price, se = mc_european(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, True, 8_000_000, 42)
    bs = _black_scholes(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, True)
    print(f"GPU MC call  = {price:.5f} +/- {se:.5f}")
    print(f"Black-Scholes = {bs:.5f}   (error {price - bs:+.5f}, {abs(price-bs)/se:.2f} std errors)")
    assert abs(price - bs) < 5.0 * se, "GPU MC disagrees with Black-Scholes beyond 5 standard errors"
    print("OK: GPU Monte Carlo agrees with Black-Scholes within Monte Carlo error.")
