// NimbleCAS arbitrary-precision mathematical constants (pi, e, gamma, ln2, ...).
// @author Olumuyiwa Oluwasanmi
//
// The classical constants of analysis -- pi, Euler's number e, the Euler-Mascheroni
// constant gamma, ln2, ln10, sqrt2, the golden ratio, and Catalan's constant G -- are
// transcendental or irrational: NONE of them has an exact representation over the
// rationals, so there is no "exact" BigInt/Rational answer to return. The honest
// deliverable is therefore a high-precision BigFloat that is correctly rounded to the
// caller's requested precision. This module is the NUMERIC high-precision provider; a
// SYMBOLIC pi/e (a leaf that stays exact inside an expression tree) belongs to the
// symbolic layer, not here.
//
// PRECISION DISCIPLINE (every entry point):
//   * The caller asks for `prec` significant bits. We compute at an ELEVATED working
//     precision work = prec + 32 + O(log2 prec) guard bits (see working_precision):
//     the fixed 32 bits absorb the final rounding, and the log-sized slack absorbs the
//     per-term round-off accumulated over the O(prec) series terms. The final result is
//     rounded once to `prec` with BigFloat::with_precision, so it is accurate to ~prec
//     bits. This is a rounded floating-point value, NOT an exact number.
//   * STOPPING RULE (shared by every series): keep an explicit threshold epsilon = 2^-work
//     (epsilon_at) and stop as soon as the magnitude of the newly added term drops below
//     epsilon -- at that point the untruncated tail is below the working ULP and cannot
//     affect the rounded result. (The Brent-McMillan sum for gamma has terms that RISE to
//     a peak near k = N before decaying, so its stop test additionally requires k > N.)
//
// ALGORITHMS:
//   e            Sum_{n>=0} 1/n!, the reciprocal factorial maintained by dividing the
//                previous term by n.
//   pi           Machin's formula pi = 16*arctan(1/5) - 4*arctan(1/239) with
//                arctan(1/x) = Sum_{n>=0} (-1)^n / ((2n+1) x^(2n+1)), a running 1/x^2 power.
//   ln2          2*atanh(1/3) = 2*Sum_{n>=0} 1/((2n+1) 3^(2n+1)).
//   ln10         3*ln2 + 2*atanh(1/9): 3*ln2 = ln8 and 2*atanh(1/9) = ln(5/4), so the sum
//                is ln(8 * 5/4) = ln10 exactly.
//   sqrt2        BigFloat::sqrt(2) (correctly rounded integer-sqrt of the scaled mantissa).
//   golden_ratio (1 + sqrt(5)) / 2 (algebraic, via BigFloat::sqrt).
//   catalan      G = (pi/8) ln(2+sqrt3) + (3/8) Sum_{n>=0} 1/((2n+1)^2 C(2n,n)). The binomial
//                series converges ~2 bits/term (C(2n,n) ~ 4^n); ln(2+sqrt3) is evaluated as
//                (2/sqrt3) Sum_{n>=0} 1/((2n+1) 3^n) since atanh(1/sqrt3) = ln(2+sqrt3)/2.
//                This replaces the plain Sum (-1)^n/(2n+1)^2, which converges only linearly.
//   gamma        Brent-McMillan algorithm B1: with N ~ ceil(work * ln2 / 4) (so the tail
//                error ~ pi*e^(-4N) < 2^-work), gamma = S/I - ln(N) where
//                t_k = (N^k/k!)^2, I = Sum_{k>=0} t_k, S = Sum_{k>=0} t_k H_k, H_k the k-th
//                harmonic number, and ln(N) uses the argument-reduced atanh log below.
//
// Railway-oriented (Rule 32): every entry point returns Result<BigFloat>; a non-positive
// precision yields domain_error, and every underlying BigFloat Result error is propagated.

export module nimblecas.constants;

import std;
import nimblecas.core;
import nimblecas.bigint;
import nimblecas.bigfloat;

export namespace nimblecas {

// Each computes the named constant to ~`prec` significant bits as a correctly-rounded
// BigFloat. prec must be > 0 (else domain_error). See the module header for the algorithm
// and stopping rule behind each one.
[[nodiscard]] auto e(std::int64_t prec) -> Result<BigFloat>;
[[nodiscard]] auto pi(std::int64_t prec) -> Result<BigFloat>;
[[nodiscard]] auto ln2(std::int64_t prec) -> Result<BigFloat>;
[[nodiscard]] auto ln10(std::int64_t prec) -> Result<BigFloat>;
[[nodiscard]] auto sqrt2(std::int64_t prec) -> Result<BigFloat>;
[[nodiscard]] auto golden_ratio(std::int64_t prec) -> Result<BigFloat>;
[[nodiscard]] auto catalan(std::int64_t prec) -> Result<BigFloat>;
[[nodiscard]] auto euler_mascheroni(std::int64_t prec) -> Result<BigFloat>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// |x| as a BigFloat (a pure sign flip; precision unchanged).
[[nodiscard]] auto magnitude(const BigFloat& x) -> BigFloat {
    return x.sign() < 0 ? x.negate() : x;
}

// Elevated working precision: prec + 32 fixed guard bits (for the single final rounding)
// + a slack that grows like 4*log2(prec) to cover the round-off accumulated across the
// O(prec) summed terms. Comfortably conservative -- the accumulated error only needs
// ~log2(#terms) extra bits, so leading digits are stable as `prec` rises.
[[nodiscard]] auto working_precision(std::int64_t prec) -> std::int64_t {
    const auto safe = static_cast<std::uint64_t>(prec > 0 ? prec : 1);
    const auto bw = static_cast<std::int64_t>(std::bit_width(safe));
    return prec + 32 + 4 * bw + 16;
}

// The convergence threshold 2^-work as a BigFloat: a series term whose magnitude falls
// below this is below the working ULP and its tail cannot change the rounded result.
[[nodiscard]] auto epsilon_at(std::int64_t work) -> Result<BigFloat> {
    auto one = BigFloat::from_i64(1, work);
    if (!one) {
        return one;
    }
    const BigInt two_pow = BigInt::from_i64(2).pow(static_cast<std::uint64_t>(work));
    auto denom = BigFloat::from_bigint(two_pow, work);
    if (!denom) {
        return denom;
    }
    return one->divide(*denom, work);
}

// A generous per-series iteration cap; every series here converges long before it, so the
// cap only guards against a pathological non-terminating loop, never against correctness.
[[nodiscard]] auto iteration_cap(std::int64_t work) -> std::int64_t { return 16 * work + 100; }

// atanh(1/x) = Sum_{n>=0} 1/((2n+1) x^(2n+1)), positive terms, a running power of 1/x^2.
[[nodiscard]] auto atanh_recip(std::int64_t x, std::int64_t work) -> Result<BigFloat> {
    auto eps = epsilon_at(work);
    if (!eps) {
        return eps;
    }
    auto fx = BigFloat::from_i64(x, work);
    if (!fx) {
        return fx;
    }
    auto fx2 = BigFloat::from_i64(x * x, work);
    if (!fx2) {
        return fx2;
    }
    auto one = BigFloat::from_i64(1, work);
    if (!one) {
        return one;
    }
    auto power = one->divide(*fx, work);  // 1/x^(2n+1), starts at 1/x
    if (!power) {
        return power;
    }
    auto sum = power;  // n = 0 term is power / 1
    const std::int64_t cap = iteration_cap(work);
    for (std::int64_t n = 1; n <= cap; ++n) {
        auto np = power->divide(*fx2, work);
        if (!np) {
            return np;
        }
        power = np;
        auto d = BigFloat::from_i64(2 * n + 1, work);
        if (!d) {
            return d;
        }
        auto term = power->divide(*d, work);
        if (!term) {
            return term;
        }
        auto s = sum->add(*term, work);
        if (!s) {
            return s;
        }
        sum = s;
        if (magnitude(*term) < *eps) {
            break;
        }
    }
    return sum;
}

// arctan(1/x) = Sum_{n>=0} (-1)^n / ((2n+1) x^(2n+1)), a running power of 1/x^2 with the
// alternating sign applied per term.
[[nodiscard]] auto arctan_recip(std::int64_t x, std::int64_t work) -> Result<BigFloat> {
    auto eps = epsilon_at(work);
    if (!eps) {
        return eps;
    }
    auto fx = BigFloat::from_i64(x, work);
    if (!fx) {
        return fx;
    }
    auto fx2 = BigFloat::from_i64(x * x, work);
    if (!fx2) {
        return fx2;
    }
    auto one = BigFloat::from_i64(1, work);
    if (!one) {
        return one;
    }
    auto power = one->divide(*fx, work);  // 1/x^(2n+1), starts at 1/x
    if (!power) {
        return power;
    }
    auto sum = power;  // n = 0 term (+)
    const std::int64_t cap = iteration_cap(work);
    for (std::int64_t n = 1; n <= cap; ++n) {
        auto np = power->divide(*fx2, work);
        if (!np) {
            return np;
        }
        power = np;
        auto d = BigFloat::from_i64(2 * n + 1, work);
        if (!d) {
            return d;
        }
        auto term = power->divide(*d, work);
        if (!term) {
            return term;
        }
        // (-1)^n: subtract on odd n, add on even n.
        if ((n & 1) != 0) {
            auto s = sum->subtract(*term, work);
            if (!s) {
                return s;
            }
            sum = s;
        } else {
            auto s = sum->add(*term, work);
            if (!s) {
                return s;
            }
            sum = s;
        }
        if (magnitude(*term) < *eps) {
            break;
        }
    }
    return sum;
}

// atanh(y) = Sum_{n>=0} y^(2n+1)/(2n+1) for a general BigFloat |y| < 1, a running power of
// y^2. Used by ln of a reduced argument (where y in [0, 1/3), fast convergence).
[[nodiscard]] auto atanh_bf(const BigFloat& y, std::int64_t work) -> Result<BigFloat> {
    auto eps = epsilon_at(work);
    if (!eps) {
        return eps;
    }
    auto y2 = y.multiply(y, work);
    if (!y2) {
        return y2;
    }
    auto power = y.with_precision(work);  // y^(2n+1), starts at y
    if (!power) {
        return power;
    }
    auto sum = power;  // n = 0 term is y / 1
    const std::int64_t cap = iteration_cap(work);
    for (std::int64_t n = 1; n <= cap; ++n) {
        auto np = power->multiply(*y2, work);
        if (!np) {
            return np;
        }
        power = np;
        auto d = BigFloat::from_i64(2 * n + 1, work);
        if (!d) {
            return d;
        }
        auto term = power->divide(*d, work);
        if (!term) {
            return term;
        }
        auto s = sum->add(*term, work);
        if (!s) {
            return s;
        }
        sum = s;
        if (magnitude(*term) < *eps) {
            break;
        }
    }
    return sum;
}

// ln2 = 2*atanh(1/3).
[[nodiscard]] auto ln2_val(std::int64_t work) -> Result<BigFloat> {
    auto a = atanh_recip(3, work);
    if (!a) {
        return a;
    }
    auto two = BigFloat::from_i64(2, work);
    if (!two) {
        return two;
    }
    return a->multiply(*two, work);
}

// ln(n) for a positive integer n, by argument reduction: write n = m * 2^e with e =
// floor(log2 n) so m = n / 2^e lies in [1, 2); then ln(n) = e*ln2 + 2*atanh((m-1)/(m+1)),
// and (m-1)/(m+1) in [0, 1/3) keeps the atanh series fast. Used for ln(N) in gamma.
[[nodiscard]] auto ln_int(std::int64_t n, std::int64_t work) -> Result<BigFloat> {
    if (n <= 0) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    if (n == 1) {
        return BigFloat::from_i64(0, work);  // ln(1) = 0
    }
    const std::int64_t e =
        static_cast<std::int64_t>(std::bit_width(static_cast<std::uint64_t>(n))) - 1;
    const BigInt two_e = BigInt::from_i64(2).pow(static_cast<std::uint64_t>(e));
    auto fn = BigFloat::from_i64(n, work);
    if (!fn) {
        return fn;
    }
    auto f2e = BigFloat::from_bigint(two_e, work);
    if (!f2e) {
        return f2e;
    }
    auto m = fn->divide(*f2e, work);  // in [1, 2)
    if (!m) {
        return m;
    }
    auto one = BigFloat::from_i64(1, work);
    if (!one) {
        return one;
    }
    auto num = m->subtract(*one, work);
    if (!num) {
        return num;
    }
    auto den = m->add(*one, work);
    if (!den) {
        return den;
    }
    auto y = num->divide(*den, work);  // in [0, 1/3)
    if (!y) {
        return y;
    }
    auto at = atanh_bf(*y, work);
    if (!at) {
        return at;
    }
    auto l2 = ln2_val(work);
    if (!l2) {
        return l2;
    }
    auto fe = BigFloat::from_i64(e, work);
    if (!fe) {
        return fe;
    }
    auto e_ln2 = fe->multiply(*l2, work);  // e * ln2
    if (!e_ln2) {
        return e_ln2;
    }
    auto two = BigFloat::from_i64(2, work);
    if (!two) {
        return two;
    }
    auto two_at = at->multiply(*two, work);  // 2 * atanh(y)
    if (!two_at) {
        return two_at;
    }
    return e_ln2->add(*two_at, work);
}

// Machin: pi = 16*arctan(1/5) - 4*arctan(1/239).
[[nodiscard]] auto pi_machin(std::int64_t work) -> Result<BigFloat> {
    auto a5 = arctan_recip(5, work);
    if (!a5) {
        return a5;
    }
    auto a239 = arctan_recip(239, work);
    if (!a239) {
        return a239;
    }
    auto s16 = BigFloat::from_i64(16, work);
    if (!s16) {
        return s16;
    }
    auto s4 = BigFloat::from_i64(4, work);
    if (!s4) {
        return s4;
    }
    auto t1 = a5->multiply(*s16, work);
    if (!t1) {
        return t1;
    }
    auto t2 = a239->multiply(*s4, work);
    if (!t2) {
        return t2;
    }
    return t1->subtract(*t2, work);
}

// ln10 = 3*ln2 + 2*atanh(1/9) = ln8 + ln(5/4) = ln10.
[[nodiscard]] auto ln10_val(std::int64_t work) -> Result<BigFloat> {
    auto l2 = ln2_val(work);
    if (!l2) {
        return l2;
    }
    auto three = BigFloat::from_i64(3, work);
    if (!three) {
        return three;
    }
    auto t1 = l2->multiply(*three, work);  // 3*ln2 = ln8
    if (!t1) {
        return t1;
    }
    auto a9 = atanh_recip(9, work);
    if (!a9) {
        return a9;
    }
    auto two = BigFloat::from_i64(2, work);
    if (!two) {
        return two;
    }
    auto t2 = a9->multiply(*two, work);  // 2*atanh(1/9) = ln(5/4)
    if (!t2) {
        return t2;
    }
    return t1->add(*t2, work);
}

// e = Sum_{n>=0} 1/n!, the running reciprocal factorial divided by n each step.
[[nodiscard]] auto e_series(std::int64_t work) -> Result<BigFloat> {
    auto eps = epsilon_at(work);
    if (!eps) {
        return eps;
    }
    auto sum = BigFloat::from_i64(1, work);  // 1/0!
    if (!sum) {
        return sum;
    }
    auto term = BigFloat::from_i64(1, work);  // current 1/n!, starts at 1/0!
    if (!term) {
        return term;
    }
    const std::int64_t cap = iteration_cap(work);
    for (std::int64_t n = 1; n <= cap; ++n) {
        auto fn = BigFloat::from_i64(n, work);
        if (!fn) {
            return fn;
        }
        auto nt = term->divide(*fn, work);  // 1/n! = (1/(n-1)!) / n
        if (!nt) {
            return nt;
        }
        term = nt;
        auto ns = sum->add(*term, work);
        if (!ns) {
            return ns;
        }
        sum = ns;
        if (magnitude(*term) < *eps) {
            break;
        }
    }
    return sum;
}

// Catalan's constant G = (pi/8) ln(2+sqrt3) + (3/8) Sum_{n>=0} 1/((2n+1)^2 C(2n,n)).
[[nodiscard]] auto catalan_val(std::int64_t work) -> Result<BigFloat> {
    auto eps = epsilon_at(work);
    if (!eps) {
        return eps;
    }
    auto one = BigFloat::from_i64(1, work);
    if (!one) {
        return one;
    }
    const std::int64_t cap = iteration_cap(work);

    // sqrt(3), needed both for ln(2+sqrt3) and its 2/sqrt3 prefactor.
    auto three_bf = BigFloat::from_i64(3, work);
    if (!three_bf) {
        return three_bf;
    }
    auto sqrt3 = three_bf->sqrt(work);
    if (!sqrt3) {
        return sqrt3;
    }

    // slog = Sum_{n>=0} 1/((2n+1) 3^n); atanh(1/sqrt3) = (1/sqrt3) * slog, so
    // ln(2+sqrt3) = 2*atanh(1/sqrt3) = (2/sqrt3) * slog. Exact integer denominators.
    auto slog = BigFloat::from_i64(1, work);  // n = 0: 1/(1*1)
    if (!slog) {
        return slog;
    }
    BigInt pow3 = BigInt::from_i64(1);  // 3^n
    for (std::int64_t n = 1; n <= cap; ++n) {
        pow3 = pow3.multiply(BigInt::from_i64(3));
        const BigInt denom = BigInt::from_i64(2 * n + 1).multiply(pow3);
        auto fden = BigFloat::from_bigint(denom, work);
        if (!fden) {
            return fden;
        }
        auto term = one->divide(*fden, work);
        if (!term) {
            return term;
        }
        auto s = slog->add(*term, work);
        if (!s) {
            return s;
        }
        slog = s;
        if (magnitude(*term) < *eps) {
            break;
        }
    }
    auto two = BigFloat::from_i64(2, work);
    if (!two) {
        return two;
    }
    auto two_over_r = two->divide(*sqrt3, work);
    if (!two_over_r) {
        return two_over_r;
    }
    auto ln_arg = two_over_r->multiply(*slog, work);  // ln(2+sqrt3)
    if (!ln_arg) {
        return ln_arg;
    }

    // s2 = Sum_{n>=0} 1/((2n+1)^2 C(2n,n)); C(2n,n) maintained exactly by the recurrence
    // C(2n,n) = C(2n-2,n-1) * 2(2n-1) / n. Denominators are exact BigInts.
    auto s2 = BigFloat::from_i64(1, work);  // n = 0: 1/(1*1)
    if (!s2) {
        return s2;
    }
    BigInt binom = BigInt::from_i64(1);  // C(0,0)
    for (std::int64_t n = 1; n <= cap; ++n) {
        binom = binom.multiply(BigInt::from_i64(2 * (2 * n - 1)));
        auto bq = binom.divide(BigInt::from_i64(n));  // exact
        if (!bq) {
            return make_error<BigFloat>(bq.error());
        }
        binom = *bq;
        const BigInt odd_sq = BigInt::from_i64(2 * n + 1).pow(2);
        const BigInt denom = odd_sq.multiply(binom);
        auto fden = BigFloat::from_bigint(denom, work);
        if (!fden) {
            return fden;
        }
        auto term = one->divide(*fden, work);
        if (!term) {
            return term;
        }
        auto s = s2->add(*term, work);
        if (!s) {
            return s;
        }
        s2 = s;
        if (magnitude(*term) < *eps) {
            break;
        }
    }

    // G = (pi * ln(2+sqrt3) + 3 * s2) / 8.
    auto pi_v = pi_machin(work);
    if (!pi_v) {
        return pi_v;
    }
    auto pi_log = pi_v->multiply(*ln_arg, work);
    if (!pi_log) {
        return pi_log;
    }
    auto three = BigFloat::from_i64(3, work);
    if (!three) {
        return three;
    }
    auto s2_3 = s2->multiply(*three, work);
    if (!s2_3) {
        return s2_3;
    }
    auto numer = pi_log->add(*s2_3, work);
    if (!numer) {
        return numer;
    }
    auto eight = BigFloat::from_i64(8, work);
    if (!eight) {
        return eight;
    }
    return numer->divide(*eight, work);
}

// Euler-Mascheroni gamma via Brent-McMillan B1. t_k = (N^k/k!)^2 is built by the recurrence
// t_k = t_{k-1} * N^2 / k^2 (done as two multiplies by N and two divides by k, so no integer
// N^2/k^2 ever overflows). I = Sum t_k, S = Sum t_k H_k with H_k the harmonic numbers, and
// gamma = S/I - ln(N). The t_k RISE to a peak near k = N before decaying, so the stop test
// requires k > N in addition to t_k < 2^-work.
[[nodiscard]] auto gamma_val(std::int64_t work) -> Result<BigFloat> {
    auto eps = epsilon_at(work);
    if (!eps) {
        return eps;
    }
    // N ~ ceil(work * ln2 / 4) + margin: the B1 tail error ~ pi*e^(-4N) is then < 2^-work.
    const std::int64_t N =
        static_cast<std::int64_t>(std::ceil(static_cast<double>(work) * 0.17328679513998632)) + 2;
    auto fN = BigFloat::from_i64(N, work);
    if (!fN) {
        return fN;
    }
    auto one = BigFloat::from_i64(1, work);
    if (!one) {
        return one;
    }

    auto t = BigFloat::from_i64(1, work);  // t_0 = 1
    if (!t) {
        return t;
    }
    auto I = t;  // Sum t_k, includes the k = 0 term
    auto S = BigFloat::from_i64(0, work);  // Sum t_k H_k (H_0 = 0)
    if (!S) {
        return S;
    }
    auto H = BigFloat::from_i64(0, work);  // H_0 = 0
    if (!H) {
        return H;
    }

    const std::int64_t cap = iteration_cap(work);
    for (std::int64_t k = 1; k <= cap; ++k) {
        auto fk = BigFloat::from_i64(k, work);
        if (!fk) {
            return fk;
        }
        // t_k = t_{k-1} * N^2 / k^2.
        auto a1 = t->multiply(*fN, work);
        if (!a1) {
            return a1;
        }
        auto a2 = a1->multiply(*fN, work);
        if (!a2) {
            return a2;
        }
        auto a3 = a2->divide(*fk, work);
        if (!a3) {
            return a3;
        }
        auto a4 = a3->divide(*fk, work);
        if (!a4) {
            return a4;
        }
        t = a4;
        // H_k = H_{k-1} + 1/k.
        auto invk = one->divide(*fk, work);
        if (!invk) {
            return invk;
        }
        auto hh = H->add(*invk, work);
        if (!hh) {
            return hh;
        }
        H = hh;
        // Accumulate S += t_k H_k and I += t_k.
        auto tH = t->multiply(*H, work);
        if (!tH) {
            return tH;
        }
        auto ns = S->add(*tH, work);
        if (!ns) {
            return ns;
        }
        S = ns;
        auto ni = I->add(*t, work);
        if (!ni) {
            return ni;
        }
        I = ni;
        if (k > N && magnitude(*t) < *eps) {
            break;  // past the peak and the tail is below the working ULP
        }
    }
    auto ratio = S->divide(*I, work);
    if (!ratio) {
        return ratio;
    }
    auto lnN = ln_int(N, work);
    if (!lnN) {
        return lnN;
    }
    return ratio->subtract(*lnN, work);
}

}  // namespace

// --- public entry points: elevate precision, compute, round once to `prec` -------------

auto e(std::int64_t prec) -> Result<BigFloat> {
    if (prec <= 0) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    auto v = e_series(working_precision(prec));
    if (!v) {
        return v;
    }
    return v->with_precision(prec);
}

auto pi(std::int64_t prec) -> Result<BigFloat> {
    if (prec <= 0) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    auto v = pi_machin(working_precision(prec));
    if (!v) {
        return v;
    }
    return v->with_precision(prec);
}

auto ln2(std::int64_t prec) -> Result<BigFloat> {
    if (prec <= 0) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    auto v = ln2_val(working_precision(prec));
    if (!v) {
        return v;
    }
    return v->with_precision(prec);
}

auto ln10(std::int64_t prec) -> Result<BigFloat> {
    if (prec <= 0) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    auto v = ln10_val(working_precision(prec));
    if (!v) {
        return v;
    }
    return v->with_precision(prec);
}

auto sqrt2(std::int64_t prec) -> Result<BigFloat> {
    if (prec <= 0) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    const std::int64_t work = working_precision(prec);
    auto two = BigFloat::from_i64(2, work);
    if (!two) {
        return two;
    }
    auto r = two->sqrt(work);
    if (!r) {
        return r;
    }
    return r->with_precision(prec);
}

auto golden_ratio(std::int64_t prec) -> Result<BigFloat> {
    if (prec <= 0) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    const std::int64_t work = working_precision(prec);
    auto five = BigFloat::from_i64(5, work);
    if (!five) {
        return five;
    }
    auto r = five->sqrt(work);  // sqrt(5)
    if (!r) {
        return r;
    }
    auto one = BigFloat::from_i64(1, work);
    if (!one) {
        return one;
    }
    auto sum = r->add(*one, work);  // 1 + sqrt(5)
    if (!sum) {
        return sum;
    }
    auto two = BigFloat::from_i64(2, work);
    if (!two) {
        return two;
    }
    auto phi = sum->divide(*two, work);  // (1 + sqrt(5)) / 2
    if (!phi) {
        return phi;
    }
    return phi->with_precision(prec);
}

auto catalan(std::int64_t prec) -> Result<BigFloat> {
    if (prec <= 0) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    auto v = catalan_val(working_precision(prec));
    if (!v) {
        return v;
    }
    return v->with_precision(prec);
}

auto euler_mascheroni(std::int64_t prec) -> Result<BigFloat> {
    if (prec <= 0) {
        return make_error<BigFloat>(MathError::domain_error);
    }
    auto v = gamma_val(working_precision(prec));
    if (!v) {
        return v;
    }
    return v->with_precision(prec);
}

}  // namespace nimblecas
