// NimbleCAS distributed modular polynomial GCD driver (Brown's modular GCD for Z[x]).
// @author Olumuyiwa Oluwasanmi
//
// Distributed driver and operation registration for Brown's modular GCD algorithm over
// univariate polynomials in Z[x]. Decomposes per-round GCD image evaluations into
// literal-bearing wavefront tasks scheduled across an abstract Executor seam (e.g.
// serial, local parallel, or SGEE distributed), collecting and merging the resulting
// modular images coordinator-locally via symmetric CRT and trial division verification.
//
// HONESTY BOUNDARY:
// - Exact and complete: returns the exact polynomial GCD or an honest MathError.
// - Bit-identical to the in-process modular_gcd and Polynomial::gcd references.
// - Returns MathError::not_converged if the prime budget is exhausted without
//   producing a trial-division-verified candidate.
// - Returns MathError::overflow if reconstructed coefficients or content scaling
//   exceed the int64 range.
// - Transport failures from the executor are propagated directly as
//   MathError::distributed_error without remapping into math errors.
// - NEVER returns an unverified or approximate candidate.

export module nimblecas.modgcd_dist;

import std;
import nimblecas.core;
import nimblecas.polynomial;
import nimblecas.bigint;
import nimblecas.numbertheory;
import nimblecas.taskdag;
import nimblecas.modgcd;

export namespace nimblecas {

// Registers the modgcd image task op ("nimblecas.modgcd.image/v1") into reg.
[[nodiscard]] auto register_modgcd_ops(TaskRegistry& reg) -> Result<void>;

// Distributed driver for Brown's modular GCD over Z[x], executed over an abstract Executor.
[[nodiscard]] auto modular_gcd_with(const Polynomial& a, const Polynomial& b,
                                    Executor& exec, const TaskRegistry& reg,
                                    std::size_t max_primes = k_default_prime_budget)
    -> Result<Polynomial>;

}  // namespace nimblecas

// ===========================================================================
// Implementation
// ===========================================================================
namespace nimblecas {
namespace {

// std::gcd guard for INT64_MIN
[[nodiscard]] auto checked_gcd(std::int64_t a, std::int64_t b) -> std::optional<std::int64_t> {
    constexpr std::int64_t int64_min = std::numeric_limits<std::int64_t>::min();
    if (a == int64_min || b == int64_min) {
        return std::nullopt;
    }
    return std::gcd(a, b);
}

// PrimeSchedule now comes from nimblecas.modgcd -- one schedule, shared by both drivers.

}  // namespace

// ---------------------------------------------------------------------------
// Operation Registration
// ---------------------------------------------------------------------------

auto register_modgcd_ops(TaskRegistry& reg) -> Result<void> {
    return reg.register_op("nimblecas.modgcd.image/v1", [](std::span<const Payload> args) -> Result<Payload> {
        if (args.size() != 1) {
            return make_error<Payload>(MathError::syntax_error);
        }
        auto req_res = decode_image_request(args[0]);
        if (!req_res) {
            return make_error<Payload>(req_res.error());
        }
        auto img_res = gcd_image_mod_p(req_res->a, req_res->b, req_res->p, req_res->gamma);
        if (!img_res) {
            return make_error<Payload>(img_res.error());
        }
        auto enc_res = encode_image_result(*img_res);
        if (!enc_res) {
            return make_error<Payload>(enc_res.error());
        }
        return *enc_res;
    });
}

// ---------------------------------------------------------------------------
// Distributed Driver
// ---------------------------------------------------------------------------

auto modular_gcd_with(const Polynomial& a, const Polynomial& b,
                      Executor& exec, const TaskRegistry& reg,
                      std::size_t max_primes) -> Result<Polynomial> {
    auto normalized = [](const Polynomial& p) -> Result<Polynomial> {
        return p.leading_coefficient() < 0 ? p.scale(-1) : Result<Polynomial>{p};
    };
    if (a.is_zero()) {
        return normalized(b);
    }
    if (b.is_zero()) {
        return normalized(a);
    }

    // Content split
    auto ca = a.content();
    auto cb = b.content();
    if (!ca || !cb) {
        return make_error<Polynomial>(MathError::overflow);
    }
    auto d_opt = checked_gcd(*ca, *cb);
    if (!d_opt) {
        return make_error<Polynomial>(MathError::overflow);
    }
    const std::int64_t d = *d_opt;

    auto a_prim = a.primitive_part();
    auto b_prim = b.primitive_part();
    if (!a_prim || !b_prim) {
        return make_error<Polynomial>(a_prim ? b_prim.error() : a_prim.error());
    }
    const Polynomial A = *a_prim;
    const Polynomial B = *b_prim;

    // Constant primitive parts
    if (A.degree() == 0 || B.degree() == 0) {
        return Polynomial::constant(d);
    }

    // gamma = gcd(lc(A), lc(B))
    auto gamma_opt = checked_gcd(A.leading_coefficient(), B.leading_coefficient());
    if (!gamma_opt) {
        return make_error<Polynomial>(MathError::overflow);
    }
    const std::int64_t gamma = *gamma_opt;

    // Coefficient bound
    auto b_lm_res = landau_mignotte_bound(A, B);
    if (!b_lm_res) {
        return make_error<Polynomial>(b_lm_res.error());
    }
    const BigInt b_lm = *b_lm_res;

    // Sizing: least k with 2^k > 2 * B_lm
    const BigInt target_prod = BigInt::from_u64(2).multiply(b_lm);
    std::size_t k = 1;
    BigInt pow2 = BigInt::from_u64(2);
    while (!(pow2 > target_prod)) {
        pow2 = pow2.multiply(BigInt::from_u64(2));
        ++k;
    }
    const std::size_t m_target = std::max<std::size_t>(1, (k + 29) / 30);

    PrimeSchedule sched(A.leading_coefficient(), B.leading_coefficient());
    std::vector<ZpImage> all_images;
    std::size_t primes_consumed = 0;
    bool first_round = true;

    while (primes_consumed < max_primes) {
        std::size_t m_request = 0;
        if (first_round) {
            m_request = m_target + 2;
            first_round = false;
        } else {
            // Count accepted images with minimal degree so far
            std::size_t d_min = std::numeric_limits<std::size_t>::max();
            for (const auto& img : all_images) {
                if (!img.coeffs.empty() && img.coeffs.size() - 1 < d_min) {
                    d_min = img.coeffs.size() - 1;
                }
            }
            std::size_t m_accepted = 0;
            for (const auto& img : all_images) {
                if (img.coeffs.size() - 1 == d_min) {
                    ++m_accepted;
                }
            }
            const std::int64_t diff = static_cast<std::int64_t>(m_target) -
                                      static_cast<std::int64_t>(m_accepted);
            m_request = static_cast<std::size_t>(std::max<std::int64_t>(diff, 2) + 2);
        }

        // Cap m_request by remaining prime budget
        if (primes_consumed + m_request > max_primes) {
            m_request = max_primes - primes_consumed;
        }
        if (m_request == 0) {
            break;
        }

        std::vector<std::uint64_t> round_primes;
        round_primes.reserve(m_request);
        for (std::size_t i = 0; i < m_request; ++i) {
            auto p_res = sched.next();
            if (!p_res) {
                return make_error<Polynomial>(p_res.error());
            }
            round_primes.push_back(*p_res);
            ++primes_consumed;
        }

        TaskGraph g;
        for (std::size_t i = 0; i < m_request; ++i) {
            const std::uint64_t p = round_primes[i];
            ImageRequest req{
                .p = p,
                .gamma = gamma,
                .a = A,
                .b = B,
            };
            auto enc_res = encode_image_request(req);
            if (!enc_res) {
                return make_error<Polynomial>(enc_res.error());
            }
            CostHint hint{
                .mean_seconds = 1e-6 * static_cast<double>(A.degree() + B.degree()),
                .variance = 0.0,
            };
            auto add_res = g.add_named_task(reg, "nimblecas.modgcd.image/v1",
                                            std::vector<Payload>{std::move(*enc_res)},
                                            /*deps=*/{}, hint);
            if (!add_res) {
                return make_error<Polynomial>(add_res.error());
            }
        }

        auto run_res = exec.run(g);
        if (!run_res) {
            return make_error<Polynomial>(run_res.error());
        }

        if (run_res->outputs.size() != m_request) {
            return make_error<Polynomial>(MathError::domain_error);
        }

        for (std::size_t i = 0; i < m_request; ++i) {
            const auto& out = run_res->outputs[i];
            if (!out.has_value()) {
                return make_error<Polynomial>(out.error());
            }
            auto img_res = decode_image_result(*out);
            if (!img_res) {
                return make_error<Polynomial>(img_res.error());
            }
            if (img_res->p != round_primes[i]) {
                return make_error<Polynomial>(MathError::domain_error);
            }
            all_images.push_back(std::move(*img_res));
        }

        auto merge_res = merge_images(all_images, gamma, b_lm);
        if (!merge_res) {
            return make_error<Polynomial>(merge_res.error());
        }

        if (std::holds_alternative<CoprimeProven>(*merge_res)) {
            return Polynomial::constant(d);
        }

        if (const auto* cand = std::get_if<Candidate>(&*merge_res)) {
            // Trial division verification (§3.8)
            auto div_a = A.divide_exact(cand->polynomial);
            auto div_b = B.divide_exact(cand->polynomial);
            if (div_a.has_value() && div_b.has_value()) {
                // Exact candidate verified! Scale by content gcd d
                auto scaled = cand->polynomial.scale(d);
                if (!scaled) {
                    return make_error<Polynomial>(MathError::overflow);
                }
                return scaled;
            }
            // Trial division failed: all accepted primes were unlucky.
            // Discard accepted set by clearing all images and continuing.
            all_images.clear();
        }
    }

    return make_error<Polynomial>(MathError::not_converged);
}

}  // namespace nimblecas
