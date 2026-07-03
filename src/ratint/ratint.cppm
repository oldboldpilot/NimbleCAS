// NimbleCAS rational-function integration — Hermite reduction (ROADMAP 7.19).
// @author Olumuyiwa Oluwasanmi
//
// Hermite reduction splits the integral of a rational function A(x)/B(x) over Q into an
// already-integrated *rational part* g(x) and a leftover integrand whose denominator is
// square-free, WITHOUT fully factoring B:
//
//     int A/B dx = g  +  int h dx,   h = numerator/denominator, denominator square-free.
//
// The rational part is elementary and computed exactly here; the remaining int h dx is
// the logarithmic/transcendental part, later handled by Rothstein-Trager (which needs a
// square-free denominator, precisely what this pass leaves behind).
//
// Method. Using the square-free partial-fraction towers from nimblecas.pfd, the proper
// part becomes a sum of M(x)/V(x)^k with V monic square-free and deg M < k*deg V. For
// k >= 2, since gcd(V, V') = 1 (V square-free), solve V*S + V'*T = M with deg T < deg V,
// and integrate by parts:
//
//     int M/V^k dx = -T/((k-1) V^{k-1})  +  int (S + T'/(k-1)) / V^{k-1} dx.
//
// The new numerator S + T'/(k-1) again has degree < (k-1)*deg V, so each step lowers the
// pole order by one while staying proper. At k = 1 the term joins the square-free
// integrand h. The polynomial part of A/B integrates directly into g. Every operation is
// exact and overflow-checked (Result / MathError, Rule 32).

export module nimblecas.ratint;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.pfd;

export namespace nimblecas {

// The result of Hermite reduction:
//     int numerator/denominator dx
//         == rational_num/rational_den  +  int integrand_num/integrand_den dx,
// where integrand_den is square-free and deg integrand_num < deg integrand_den. When the
// whole integral is elementary-rational (a square-free-denominator remainder that is the
// zero function) integrand_num is zero.
struct HermiteReduction {
    RationalPoly rational_num;   // g = rational_num / rational_den (already integrated)
    RationalPoly rational_den;   // monic-ish; never zero (the constant 1 when g == 0)
    RationalPoly integrand_num;  // remaining integrand numerator (deg < deg integrand_den)
    RationalPoly integrand_den;  // square-free; the input to the logarithmic part
};

// Hermite-reduce numerator/denominator over Q(x). Fails with division_by_zero (a zero
// denominator) or overflow (an int64 coefficient limit), matching nimblecas.ratpoly.
[[nodiscard]] auto hermite_reduce(const RationalPoly& numerator,
                                  const RationalPoly& denominator)
    -> Result<HermiteReduction>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

[[nodiscard]] auto one_poly() -> RationalPoly {
    return RationalPoly::constant(Rational::from_int(1));
}

[[nodiscard]] auto exact_quotient(const RationalPoly& a, const RationalPoly& b)
    -> Result<RationalPoly> {
    auto dm = a.divide(b);
    if (!dm) {
        return make_error<RationalPoly>(dm.error());
    }
    return std::move(dm->quotient);
}

[[nodiscard]] auto pow_poly(const RationalPoly& b, std::int64_t n) -> Result<RationalPoly> {
    RationalPoly acc = one_poly();
    for (std::int64_t k = 0; k < n; ++k) {
        auto next = acc.multiply(b);
        if (!next) {
            return next;
        }
        acc = std::move(*next);
    }
    return acc;
}

// --- Extended Euclid over Q[x]: cofactor of V' in V*S + V'*T = gcd(V, V') ----

struct ExtGcd {
    RationalPoly g;  // gcd(a, b) (not necessarily monic)
    RationalPoly t;  // s*a + t*b == g (only the b-cofactor t is needed here)
};

auto ext_gcd_t(const RationalPoly& a, const RationalPoly& b) -> Result<ExtGcd> {
    RationalPoly old_r = a;
    RationalPoly r = b;
    RationalPoly old_t;       // 0
    RationalPoly t = one_poly();
    while (!r.is_zero()) {
        auto dm = old_r.divide(r);
        if (!dm) {
            return make_error<ExtGcd>(dm.error());
        }
        const RationalPoly& q = dm->quotient;
        old_r = std::move(r);
        r = std::move(dm->remainder);  // = old_r - q*r
        auto qt = q.multiply(t);
        if (!qt) {
            return make_error<ExtGcd>(qt.error());
        }
        auto nt = old_t.subtract(*qt);
        if (!nt) {
            return make_error<ExtGcd>(nt.error());
        }
        old_t = std::move(t);
        t = std::move(*nt);
    }
    return ExtGcd{.g = std::move(old_r), .t = std::move(old_t)};
}

// --- Rational-function accumulator (num/den in lowest terms) ------------------

struct Frac {
    RationalPoly num;
    RationalPoly den;  // never zero
};

// Reduce num/den by their gcd; the zero numerator normalises to 0/1.
[[nodiscard]] auto reduce(Frac f) -> Result<Frac> {
    if (f.num.is_zero()) {
        return Frac{.num = RationalPoly{}, .den = one_poly()};
    }
    auto g = f.num.gcd(f.den);  // monic
    if (!g) {
        return make_error<Frac>(g.error());
    }
    if (g->degree() <= 0) {
        return f;  // already coprime (gcd is a unit)
    }
    auto n = exact_quotient(f.num, *g);
    if (!n) {
        return make_error<Frac>(n.error());
    }
    auto d = exact_quotient(f.den, *g);
    if (!d) {
        return make_error<Frac>(d.error());
    }
    return Frac{.num = std::move(*n), .den = std::move(*d)};
}

// a/b + c/d = (a*d + c*b)/(b*d), reduced.
[[nodiscard]] auto add(const Frac& x, const Frac& y) -> Result<Frac> {
    auto ad = x.num.multiply(y.den);
    if (!ad) {
        return make_error<Frac>(ad.error());
    }
    auto cb = y.num.multiply(x.den);
    if (!cb) {
        return make_error<Frac>(cb.error());
    }
    auto num = ad->add(*cb);
    if (!num) {
        return make_error<Frac>(num.error());
    }
    auto den = x.den.multiply(y.den);
    if (!den) {
        return make_error<Frac>(den.error());
    }
    return reduce(Frac{.num = std::move(*num), .den = std::move(*den)});
}

// --- Polynomial antiderivative (constant of integration 0) -------------------

// int (sum a_i x^i) dx = sum a_i/(i+1) x^{i+1}.
[[nodiscard]] auto integrate_polynomial(const RationalPoly& p) -> Result<RationalPoly> {
    if (p.is_zero()) {
        return RationalPoly{};
    }
    const auto coeffs = p.coefficients();
    std::vector<Rational> out(coeffs.size() + 1);  // out[0] = 0 (the integration constant)
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        auto term = coeffs[i].divide(Rational::from_int(static_cast<std::int64_t>(i) + 1));
        if (!term) {
            return make_error<RationalPoly>(term.error());
        }
        out[i + 1] = *term;
    }
    return RationalPoly::from_coeffs(std::move(out));
}

}  // namespace

auto hermite_reduce(const RationalPoly& numerator, const RationalPoly& denominator)
    -> Result<HermiteReduction> {
    // Square-free tower decomposition: A/B = P + sum_i M_i / V_i^{k_i}.
    auto sf = square_free_partial_fractions(numerator, denominator);
    if (!sf) {
        return make_error<HermiteReduction>(sf.error());
    }

    // Rational part g starts at the polynomial antiderivative int P dx.
    auto poly_int = integrate_polynomial(sf->polynomial_part);
    if (!poly_int) {
        return make_error<HermiteReduction>(poly_int.error());
    }
    Frac g{.num = std::move(*poly_int), .den = one_poly()};
    Frac h{.num = RationalPoly{}, .den = one_poly()};  // the square-free integrand

    for (auto& tower : sf->towers) {
        const RationalPoly& v = tower.base;
        auto vprime = v.derivative();
        if (!vprime) {
            return make_error<HermiteReduction>(vprime.error());
        }
        // Bezout cofactor: V*S + V'*T = 1 (V square-free => gcd(V, V') is a nonzero
        // constant). We need only T, the cofactor of V'.
        auto eg = ext_gcd_t(v, *vprime);
        if (!eg) {
            return make_error<HermiteReduction>(eg.error());
        }
        auto inv_g = Rational::from_int(1).divide(eg->g.coefficient(0));  // g is a constant
        if (!inv_g) {
            return make_error<HermiteReduction>(inv_g.error());
        }
        auto tau = eg->t.scale(*inv_g);  // V*sigma + V'*tau == 1
        if (!tau) {
            return make_error<HermiteReduction>(tau.error());
        }

        RationalPoly m = tower.numerator;
        std::int64_t k = tower.exponent;
        while (k >= 2) {
            // T = (M * tau) mod V, deg T < deg V.
            auto mtau = m.multiply(*tau);
            if (!mtau) {
                return make_error<HermiteReduction>(mtau.error());
            }
            auto dm = mtau->divide(v);
            if (!dm) {
                return make_error<HermiteReduction>(dm.error());
            }
            const RationalPoly big_t = std::move(dm->remainder);
            // S = (M - T*V') / V  (exact).
            auto tvp = big_t.multiply(*vprime);
            if (!tvp) {
                return make_error<HermiteReduction>(tvp.error());
            }
            auto m_minus = m.subtract(*tvp);
            if (!m_minus) {
                return make_error<HermiteReduction>(m_minus.error());
            }
            auto s = exact_quotient(*m_minus, v);
            if (!s) {
                return make_error<HermiteReduction>(s.error());
            }
            // Rational piece: -T / ((k-1) * V^{k-1}).
            auto inv_km1 = Rational::make(-1, k - 1);  // k-1 >= 1, cannot fail
            if (!inv_km1) {
                return make_error<HermiteReduction>(inv_km1.error());
            }
            auto piece_num = big_t.scale(*inv_km1);
            if (!piece_num) {
                return make_error<HermiteReduction>(piece_num.error());
            }
            auto piece_den = pow_poly(v, k - 1);
            if (!piece_den) {
                return make_error<HermiteReduction>(piece_den.error());
            }
            auto g_next = add(g, Frac{.num = std::move(*piece_num),
                                      .den = std::move(*piece_den)});
            if (!g_next) {
                return make_error<HermiteReduction>(g_next.error());
            }
            g = std::move(*g_next);
            // New integrand numerator: S + T'/(k-1).
            auto tprime = big_t.derivative();
            if (!tprime) {
                return make_error<HermiteReduction>(tprime.error());
            }
            auto scale_km1 = Rational::make(1, k - 1);  // k-1 >= 1, cannot fail
            if (!scale_km1) {
                return make_error<HermiteReduction>(scale_km1.error());
            }
            auto tprime_scaled = tprime->scale(*scale_km1);
            if (!tprime_scaled) {
                return make_error<HermiteReduction>(tprime_scaled.error());
            }
            auto m_next = s->add(*tprime_scaled);
            if (!m_next) {
                return make_error<HermiteReduction>(m_next.error());
            }
            m = std::move(*m_next);
            --k;
        }
        // k == 1: the leftover M/V joins the square-free integrand h.
        auto h_next = add(h, Frac{.num = std::move(m), .den = v});
        if (!h_next) {
            return make_error<HermiteReduction>(h_next.error());
        }
        h = std::move(*h_next);
    }

    return HermiteReduction{.rational_num = std::move(g.num),
                            .rational_den = std::move(g.den),
                            .integrand_num = std::move(h.num),
                            .integrand_den = std::move(h.den)};
}

}  // namespace nimblecas
