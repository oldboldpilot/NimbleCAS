// NimbleCAS Metropolis-Hastings MCMC (ROADMAP 7.8).
// @author Olumuyiwa Oluwasanmi
//
// Deterministic (seeded) random-walk Metropolis-Hastings sampling built on the
// parallelisable RNG substrate (nimblecas.rng) and its Monte Carlo companions. Targets are
// supplied as an UNNORMALISED log-density: working in log space avoids underflow for
// sharply peaked targets, and the unknown normalising constant cancels in the acceptance
// ratio, so only differences of log-densities are ever needed. A single seeded `Rng` drives
// each chain (two draws per step: the symmetric proposal, then the acceptance test), which
// makes every chain bit-reproducible from its seed. `run_parallel_chains` derives one
// independent seed per chain via `splitmix64(seed ^ c)`, so the chains are mutually
// independent and reproducible regardless of how many are launched or in what order — they
// are summed serially here, but the design admits parallel execution. There is no time or
// entropy seeding anywhere, and all failure travels the railway (Result<T> / MathError),
// never an exception.

export module nimblecas.mcmc;

import std;
import nimblecas.core;
import nimblecas.rng;

export namespace nimblecas {

// The outcome of a single Metropolis-Hastings run: the retained samples together with the
// accept/propose bookkeeping. The empirical acceptance rate is accepted / proposed, counted
// over ALL iterations (burn-in included), while `chain` holds only the post-burn-in samples.
struct McmcResult {
    std::vector<double> chain;  // the `samples` retained draws (post burn-in), in order
    std::uint64_t accepted;     // proposals accepted over all burn_in + samples steps
    std::uint64_t proposed;     // total proposals attempted == burn_in + samples
};

// Random-walk Metropolis with a SYMMETRIC uniform proposal. From the current state x a
// candidate is drawn as x' = x + step · (2U − 1) with U ~ Uniform[0, 1), so the proposal is
// symmetric and its density cancels — the Metropolis (not full Hastings) acceptance ratio
// applies. A second draw V ~ Uniform[0, 1) tests acceptance:
//
//     accept  <=>  log(V) <= log_density(x') − log_density(x)
//
// Because U, V are in [0, 1), log(V) <= 0, so the move is ALWAYS accepted when the proposed
// log-density is not smaller than the current one. The comparison also handles −inf support
// correctly via IEEE semantics: a proposal into a zero-support region (log_density(x') =
// −inf) gives a difference of −inf and is rejected; if the current point already has −inf
// density, the difference is +inf and any finite proposal is accepted (letting a chain that
// started outside the support walk back in); and −inf minus −inf is NaN, whose comparison is
// false, so such a step is rejected and the chain holds its place.
//
// The routine runs exactly `burn_in + samples` iterations, discards the first `burn_in`,
// and returns the final `samples` states in `chain`; accepted/proposed count every
// iteration. A single Rng::seeded(seed) supplies the whole chain (proposal draw then
// acceptance draw each step), so equal seeds reproduce a bit-identical chain and identical
// counts. Because the loop count is fixed the routine always terminates. Returns
// domain_error if step <= 0 or samples == 0.
[[nodiscard]] auto metropolis_hastings(std::function<double(double)> log_density, double x0,
                                       double step, std::uint64_t samples,
                                       std::uint64_t burn_in, std::uint64_t seed)
    -> Result<McmcResult>;

// Launch `chains` INDEPENDENT Metropolis-Hastings chains against the same target, each with
// the same starting point / step / sample budget but its own reproducible seed.
//
// PARALLELISATION CONTRACT: chain c is seeded with splitmix64(seed ^ c). SplitMix64 is a
// bijective bit-mixer, so distinct c yield distinct, well-separated seeds and hence
// mutually independent streams; and because chain c's seed depends only on (seed, c) — not
// on the total number of chains nor on the order they run — the set of chains is fully
// reproducible and each chain is identical no matter how many others are launched. The
// chains are run serially here, but this per-chain-index seeding is exactly what lets a
// parallel executor hand each chain to its own worker with no coordination.
//
// Returns domain_error if chains == 0, or propagates the per-chain domain_error if the
// shared arguments are invalid (step <= 0 or samples_per_chain == 0).
[[nodiscard]] auto run_parallel_chains(std::function<double(double)> log_density, double x0,
                                       double step, std::uint64_t samples_per_chain,
                                       std::uint64_t burn_in, std::uint64_t seed,
                                       std::uint64_t chains) -> Result<std::vector<McmcResult>>;

// Arithmetic mean of a chain. Self-contained (no montecarlo dependency). Returns
// domain_error on an empty span.
[[nodiscard]] auto chain_mean(std::span<const double> chain) -> Result<double>;

// Population variance of a chain (sum of squared deviations divided by n, not n − 1).
// Self-contained. Returns domain_error on an empty span.
[[nodiscard]] auto chain_variance(std::span<const double> chain) -> Result<double>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {

auto metropolis_hastings(std::function<double(double)> log_density, double x0, double step,
                         std::uint64_t samples, std::uint64_t burn_in, std::uint64_t seed)
    -> Result<McmcResult> {
    if (step <= 0.0 || samples == 0) {
        return make_error<McmcResult>(MathError::domain_error);
    }

    auto rng = Rng::seeded(seed);

    double x = x0;
    double current_log = log_density(x);

    McmcResult result{};
    result.chain.reserve(static_cast<std::size_t>(samples));
    result.accepted = 0;
    result.proposed = 0;

    const std::uint64_t total = burn_in + samples;
    for (std::uint64_t i = 0; i < total; ++i) {
        // Symmetric uniform proposal: x' = x + step · (2U − 1), U ~ Uniform[0, 1).
        const double u = rng.next_unit();
        const double proposal = x + step * (2.0 * u - 1.0);
        const double proposal_log = log_density(proposal);

        // Acceptance draw. log(V) <= Δlog accepts; the IEEE compare rejects −inf proposals
        // and (via NaN) −inf-from-−inf steps, and always accepts when Δlog >= 0.
        const double v = rng.next_unit();
        if (std::log(v) <= proposal_log - current_log) {
            x = proposal;
            current_log = proposal_log;
            ++result.accepted;
        }
        ++result.proposed;

        if (i >= burn_in) {
            result.chain.push_back(x);
        }
    }

    return result;
}

auto run_parallel_chains(std::function<double(double)> log_density, double x0, double step,
                         std::uint64_t samples_per_chain, std::uint64_t burn_in,
                         std::uint64_t seed, std::uint64_t chains)
    -> Result<std::vector<McmcResult>> {
    if (chains == 0) {
        return make_error<std::vector<McmcResult>>(MathError::domain_error);
    }

    std::vector<McmcResult> results;
    results.reserve(static_cast<std::size_t>(chains));

    for (std::uint64_t c = 0; c < chains; ++c) {
        // Per-chain-index seed: bijective mixing of (seed, c) gives independent, reproducible
        // streams that do not depend on the chain count or scheduling order.
        const std::uint64_t chain_seed = splitmix64(seed ^ c);
        auto run = metropolis_hastings(log_density, x0, step, samples_per_chain, burn_in,
                                       chain_seed);
        if (!run) {
            // Propagate the per-chain domain_error (invalid step or samples_per_chain).
            return make_error<std::vector<McmcResult>>(run.error());
        }
        results.push_back(std::move(run.value()));
    }

    return results;
}

auto chain_mean(std::span<const double> chain) -> Result<double> {
    if (chain.empty()) {
        return make_error<double>(MathError::domain_error);
    }
    double sum = 0.0;
    for (const double v : chain) {
        sum += v;
    }
    return sum / static_cast<double>(chain.size());
}

auto chain_variance(std::span<const double> chain) -> Result<double> {
    const std::size_t n = chain.size();
    if (n == 0) {
        return make_error<double>(MathError::domain_error);
    }

    // Two-pass: mean first, then the sum of squared deviations (more stable than E[x²] −
    // E[x]²). Population variance divides by n, so a single point is allowed (variance 0).
    double sum = 0.0;
    for (const double v : chain) {
        sum += v;
    }
    const double mean = sum / static_cast<double>(n);

    double sq = 0.0;
    for (const double v : chain) {
        const double d = v - mean;
        sq += d * d;
    }
    return sq / static_cast<double>(n);
}

}  // namespace nimblecas
