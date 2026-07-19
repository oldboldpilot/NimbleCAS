"""cuTile-style tiled GPU kernel for batched Black-Scholes-Merton option pricing.

@author Olumuyiwa Oluwasanmi

The tile-programming counterpart of nimblecas.pricing's Black-Scholes (src/pricing/
pricing.cppm), the in-engine CUDA `black_scholes_batch` kernel (src/gpu/gpu_kernels.cu),
and the Triton mirror (python/triton/black_scholes.py). Each thread block cooperatively
stages a TILE of contracts into shared memory, prices the tile, and stores it back — the
canonical cuTile load -> compute -> store pattern — so a whole array of options is valued
in a single launch. It MIRRORS the authoritative CPU closed form to floating-point
tolerance (same d1/d2, same degenerate T==0 / vol==0 collapse to discounted intrinsic); it
is a batch-valuation accelerator, never a second source of truth.

HONESTY ON "cuTile": NVIDIA's public cuTile *DSL* (a `@cutile.jit` Python front-end) is not
yet released for this CUDA 13.x / sm_120 stack — the `cutile` PyPI name is a 474-byte empty
placeholder, and no `tile` surface ships under `cuda.*` or in the CUDA 13.2 toolkit headers.
Rather than fabricate a kernel against an API that does not exist, this delivers the cuTile
*programming model* — cooperative shared-memory tiles — on NVIDIA's officially-shipping
CUDA-Python runtime-compilation stack (`cuda-core` / `cuda-bindings`, CUDA 13): it
JIT-compiles a tiled CUDA C++ kernel and launches it on the device. When the cuTile DSL
lands, only the kernel body's surface syntax changes; the tiled decomposition, the host
API, and this verification harness stay identical. The price itself is a correctly-rounded
numerical value (transcendental erf/exp/log), not claimed exact — the CPU module's boundary.

Requires a CUDA-capable device plus `cuda-core`/`cuda-bindings` and `torch` (used only for
host<->device memory and the reference check). Run / profile:
    python python/cutile/black_scholes.py
    nsys profile -o bs_cutile python python/cutile/black_scholes.py     # timeline
    ncu --set full python python/cutile/black_scholes.py                # kernel counters
"""

from __future__ import annotations

import numpy as np
import torch
from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch

# Tiled CUDA C++ kernel. One block owns a TILE of `blockDim.x` contracts: it cooperatively
# loads the six input lanes into shared memory (the tile), prices each element, and stores
# the tile back. N(-d) == 1 - N(d) by the symmetry of the standard-normal CDF.
#
# HONEST NOTE ON THE TILE: Black-Scholes is elementwise, so each lane reads only the tile
# slot it wrote — the shared-memory tile carries NO cross-lane data reuse and is therefore
# not a bandwidth optimization (it adds one global->shared->register hop). It is here to
# express the cuTile load -> compute -> store decomposition faithfully; the ~262 us/launch
# measured at 1e6 options (see docs/reference/gpu.md) already INCLUDES this staging, so the
# reported throughput is honest. A naive global-only kernel would price the same array to
# the same values; the tile structure is what a cuTile DSL port would preserve.
_KERNEL_SRC = r"""
extern "C" __global__ void bs_tile_kernel(
    const double* __restrict__ spot,   const double* __restrict__ strike,
    const double* __restrict__ rate,   const double* __restrict__ div_,
    const double* __restrict__ vol,    const double* __restrict__ time_,
    const int*    __restrict__ iscall, double* __restrict__ out, int n)
{
    extern __shared__ double sm[];        // 6 lanes of blockDim.x doubles
    const int TILE = blockDim.x;
    double* sS = sm;
    double* sK = sS + TILE;
    double* sr = sK + TILE;
    double* sq = sr + TILE;
    double* sv = sq + TILE;
    double* sT = sv + TILE;

    const int lid = threadIdx.x;
    const int gid = blockIdx.x * TILE + lid;

    // Cooperative tile load: each lane stages its own contract into shared memory.
    if (gid < n) {
        sS[lid] = spot[gid];  sK[lid] = strike[gid]; sr[lid] = rate[gid];
        sq[lid] = div_[gid];  sv[lid] = vol[gid];    sT[lid] = time_[gid];
    }
    __syncthreads();

    if (gid < n) {
        const double S = sS[lid], K = sK[lid], r = sr[lid];
        const double q = sq[lid], v = sv[lid], T = sT[lid];
        const double INV_SQRT2 = 0.70710678118654752440;
        const double rootT = sqrt(T);
        const double dr = exp(-r * T);
        const double dq = exp(-q * T);
        const double d1 = (log(S / K) + (r - q + 0.5 * v * v) * T) / (v * rootT);
        const double d2 = d1 - v * rootT;
        const double nd1 = 0.5 * (1.0 + erf(d1 * INV_SQRT2));
        const double nd2 = 0.5 * (1.0 + erf(d2 * INV_SQRT2));
        const double call_px = S * dq * nd1 - K * dr * nd2;
        const double put_px  = K * dr * (1.0 - nd2) - S * dq * (1.0 - nd1);
        // Degenerate T==0 or vol==0: discounted intrinsic on the forward.
        const double fwd = S * exp((r - q) * T);
        const bool deg = (T == 0.0) || (v == 0.0);
        const double pc = deg ? dr * fmax(fwd - K, 0.0) : call_px;
        const double pp = deg ? dr * fmax(K - fwd, 0.0) : put_px;
        out[gid] = (iscall[gid] != 0) ? pc : pp;
    }
}
"""

_KERNEL_CACHE: dict[str, object] = {}


def _tile_kernel(device: Device):
    """JIT-compile (once per process) the tiled kernel for the current device arch."""
    key = f"sm_{device.arch}"
    ker = _KERNEL_CACHE.get(key)
    if ker is None:
        opts = ProgramOptions(arch=key, std="c++17")
        obj = Program(_KERNEL_SRC, code_type="c++", options=opts).compile("cubin")
        ker = obj.get_kernel("bs_tile_kernel")
        _KERNEL_CACHE[key] = ker
    return ker


def black_scholes_batch(spot, strike, rate, dividend, vol, time, is_call,
                        tile_size=256):
    """Price a batch of options given parallel float64 CUDA torch tensors.

    `is_call` is an int tensor (1 call / 0 put). All inputs must be the same length and
    live on the same CUDA device. Returns a float64 prices tensor on that device.
    """
    device = Device()
    device.set_current()
    stream = device.default_stream
    ker = _tile_kernel(device)

    n = spot.numel()
    # The kernel assumes unit-stride float64 inputs; force contiguity so a non-contiguous
    # slice cannot silently produce wrong prices. `.contiguous()` is a no-op when already so.
    # Each staged tensor is bound to a name to keep it alive across the launch (a bare
    # `.contiguous().data_ptr()` could free the temporary before the kernel reads it).
    s_ = spot.contiguous();     k_ = strike.contiguous(); r_ = rate.contiguous()
    q_ = dividend.contiguous(); v_ = vol.contiguous();    t_ = time.contiguous()
    iscall_i32 = is_call.to(torch.int32).contiguous()
    out = torch.empty(n, dtype=torch.float64, device=spot.device)

    grid = (n + tile_size - 1) // tile_size
    shmem = 6 * tile_size * 8  # six double lanes per tile
    cfg = LaunchConfig(grid=grid, block=tile_size, shmem_size=shmem)
    launch(stream, cfg, ker,
           s_.data_ptr(), k_.data_ptr(), r_.data_ptr(), q_.data_ptr(),
           v_.data_ptr(), t_.data_ptr(), iscall_i32.data_ptr(), out.data_ptr(),
           np.int32(n))
    stream.sync()
    return out


def _cpu_black_scholes(spot, strike, rate, q, vol, T, is_call):
    """Reference closed form for the validation check below."""
    from math import erf, exp, log, sqrt
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
    print(f"cuTile tiled BS: {len(spots)} options, max |error| vs CPU closed form = {max_err:.3e}")
    print(f"ATM 1y call = {atm_call:.5f}  (textbook 10.45058)")
    assert max_err < 1e-9, "cuTile tiled Black-Scholes disagrees with the CPU closed form"
    assert abs(atm_call - 10.4505835) < 1e-4, "ATM call off"
    print("OK: cuTile tiled Black-Scholes mirrors the CPU closed form.")
