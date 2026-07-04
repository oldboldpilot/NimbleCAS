// NimbleCAS symbolic limits / L'Hopital (ROADMAP 7).
// @author Olumuyiwa Oluwasanmi
//
// This module computes EXACT symbolic limits for a deliberately narrow, decidable
// class of expressions. It is NOT a general limit oracle: anything it cannot decide
// soundly returns an honest MathError rather than a guessed value (Rule 32).
//
// WHAT IS HANDLED
//   * Continuity / direct substitution. If substituting var := point and simplifying
//     yields a finite constant with no hidden division-by-zero, that constant IS the
//     limit (elementary functions are continuous on their domain).
//   * Removable 0/0 indeterminacies at a finite point via iterated L'Hopital. The
//     expression is split into a quotient num/den (a product whose negatively-powered
//     factors form the denominator); when num(point) = den(point) = 0 we replace
//     num/den by num'/den' and retry, up to a HARD cap (kLHopitalCap) after which we
//     return not_implemented rather than loop forever.
//   * Rational functions at +/-infinity by comparing leading polynomial degrees:
//       deg(num) <  deg(den)  -> 0
//       deg(num) == deg(den)  -> exact ratio of leading coefficients (over Q)
//       deg(num) >  deg(den)  -> a signed infinity (see convention below)
//
// INFINITY CONVENTION
//   Signed infinities are represented as the nullary function markers
//       +infinity  ==  apply("inf", {})
//       -infinity  ==  apply("neg_inf", {})
//   These are the ONLY places this module ever emits an "infinity"; a finite limit is
//   always a ConstantNode. limit() also accepts either marker as its `point` argument
//   and forwards to limit_at_infinity().
//
// HONESTY / OUT OF SCOPE (all return a MathError, never a wrong value)
//   * Oscillatory / essential singularities (e.g. sin(1/x) as x->0)          -> not_implemented
//   * A genuine pole at a finite point, num(point) != 0 and den(point) = 0
//     (no finite two-sided limit)                                            -> domain_error
//   * Non-rational behaviour at infinity that is not L'Hopital-reducible here -> not_implemented
//   * Multivariate / symbolic-coefficient forms whose value or sign we cannot
//     pin down to a constant                                                 -> not_implemented
//   * Limits that genuinely depend on one-sided direction. `LimitDirection` is
//     accepted for API completeness; for the decidable class above the two-sided
//     value equals both one-sided values, so it is returned for any direction, and
//     direction-sensitive cases fall into the pole/not_implemented branches.
//   * Any int64 overflow in exact rational arithmetic                        -> overflow
//
// Everything numeric stays exact over the integers / Q; a double constant only ever
// arises if the input already carried one.

export module nimblecas.limits;

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.diff;
import nimblecas.simplify;

export namespace nimblecas {

// Direction of approach for a finite-point limit. See the header note on honesty:
// the decidable class is direction-independent, so `dir` is accepted but does not
// change a successfully-computed finite value.
enum class LimitDirection : std::uint8_t { both, left, right };

// Limit of f as var -> point (a finite constant, or an infinity marker which is
// forwarded to limit_at_infinity). Returns the exact limit value, or an honest
// MathError for anything outside the decidable class documented above.
[[nodiscard]] auto limit(const Expr& f, std::string_view var, const Expr& point,
                         LimitDirection dir = LimitDirection::both) -> Result<Expr>;

// Limit of f as var -> +infinity (positive == true) or -infinity (positive == false).
// Handles rational functions in `var` exactly; other forms return not_implemented.
[[nodiscard]] auto limit_at_infinity(const Expr& f, std::string_view var,
                                     bool positive = true) -> Result<Expr>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// Hard iteration cap on repeated L'Hopital differentiation: past this we admit
// defeat with not_implemented rather than risk a non-terminating reduction.
inline constexpr std::int64_t kLHopitalCap = 16;

// Cap on integer exponents expanded when reading a polynomial: a crafted x^(huge)
// must not blow up memory/time — beyond this the at-infinity path returns
// not_implemented instead.
inline constexpr std::int64_t kPolyPowCap = 256;

// Cap on the TOTAL expanded degree of a polynomial power. kPolyPowCap bounds a single
// exponent, but deg(base) * exp can still explode (e.g. (x^256+1)^256 -> degree 65536,
// ~10^9 nodes) from a tiny input. This bounds the product so no crafted power can blow
// up memory/time; beyond it the at-infinity path returns not_implemented.
inline constexpr std::int64_t kPolyDegreeCap = 4096;

// --- infinity markers ------------------------------------------------------
[[nodiscard]] auto pos_inf() -> Expr { return Expr::apply("inf", {}); }
[[nodiscard]] auto neg_inf() -> Expr { return Expr::apply("neg_inf", {}); }

// nullopt if e is not an infinity marker; true for +inf, false for -inf.
[[nodiscard]] auto is_inf_marker(const Expr& e) -> std::optional<bool> {
    auto fn = as<FunctionNode>(e.node().value);
    if (!fn || !(*fn)->args.empty()) {
        return std::nullopt;
    }
    if ((*fn)->name == "inf") {
        return true;
    }
    if ((*fn)->name == "neg_inf") {
        return false;
    }
    return std::nullopt;
}

// --- constant inspection ---------------------------------------------------
[[nodiscard]] auto is_zero_const(const Expr& e) -> bool {
    auto c = as<ConstantNode>(e.node().value);
    if (!c) {
        return false;
    }
    return std::visit(
        []<typename V>(const V& v) -> bool {
            if constexpr (std::is_same_v<V, std::int64_t>) {
                return v == 0;
            } else if constexpr (std::is_same_v<V, double>) {
                return v == 0.0;
            } else {  // std::pair<int64,int64>
                return v.first == 0;
            }
        },
        (*c)->value);
}

// A finite constant is a ConstantNode that is not a non-finite double. Integers and
// exact rationals are always finite; the infinity markers are FunctionNodes and are
// therefore (correctly) excluded here.
[[nodiscard]] auto is_finite_const(const Expr& e) -> bool {
    auto c = as<ConstantNode>(e.node().value);
    if (!c) {
        return false;
    }
    return std::visit(
        []<typename V>(const V& v) -> bool {
            if constexpr (std::is_same_v<V, double>) {
                return std::isfinite(v);
            } else {
                return true;  // int64 or exact rational
            }
        },
        (*c)->value);
}

// Sign of a constant (-1/0/+1), or nullopt if e is not a finite constant.
[[nodiscard]] auto const_sign(const Expr& e) -> std::optional<int> {
    auto c = as<ConstantNode>(e.node().value);
    if (!c) {
        return std::nullopt;
    }
    return std::visit(
        []<typename V>(const V& v) -> std::optional<int> {
            if constexpr (std::is_same_v<V, std::int64_t>) {
                return static_cast<int>((v > 0) - (v < 0));
            } else if constexpr (std::is_same_v<V, double>) {
                if (!std::isfinite(v)) {
                    return std::nullopt;
                }
                return static_cast<int>((v > 0.0) - (v < 0.0));
            } else {  // pair, denominator > 0 by construction
                return static_cast<int>((v.first > 0) - (v.first < 0));
            }
        },
        (*c)->value);
}

// Extract e as a non-negative integer constant (integer, or exact rational with
// denominator 1), else nullopt.
[[nodiscard]] auto nonneg_int(const Expr& e) -> std::optional<std::int64_t> {
    auto c = as<ConstantNode>(e.node().value);
    if (!c) {
        return std::nullopt;
    }
    return std::visit(
        []<typename V>(const V& v) -> std::optional<std::int64_t> {
            if constexpr (std::is_same_v<V, std::int64_t>) {
                return v >= 0 ? std::optional<std::int64_t>(v) : std::nullopt;
            } else if constexpr (std::is_same_v<V, double>) {
                return std::nullopt;
            } else {  // pair
                return (v.second == 1 && v.first >= 0)
                           ? std::optional<std::int64_t>(v.first)
                           : std::nullopt;
            }
        },
        (*c)->value);
}

// --- evaluation & quotient split -------------------------------------------

// f with var := point, automatically simplified. A division-by-zero / undefined form
// surfaces as the corresponding MathError (used to detect indeterminacy upstream).
[[nodiscard]] auto value_at(const Expr& f, std::string_view var, const Expr& point)
    -> Result<Expr> {
    return simplify(substitute(f, Expr::symbol(std::string(var)), point));
}

// Negate a finite constant exponent (used when moving a negatively-powered factor
// into the denominator). Overflow-guarded (Rule 32).
[[nodiscard]] auto negate_const(const Expr& e) -> Result<Expr> {
    auto c = as<ConstantNode>(e.node().value);
    if (!c) {
        return make_error<Expr>(MathError::not_implemented);
    }
    constexpr std::int64_t int64_min = std::numeric_limits<std::int64_t>::min();
    return std::visit(
        [&]<typename V>(const V& v) -> Result<Expr> {
            if constexpr (std::is_same_v<V, std::int64_t>) {
                if (v == int64_min) {
                    return make_error<Expr>(MathError::overflow);
                }
                return Expr::integer(-v);
            } else if constexpr (std::is_same_v<V, double>) {
                return Expr::real(-v);
            } else {  // pair
                if (v.first == int64_min) {
                    return make_error<Expr>(MathError::overflow);
                }
                return Expr::rational(-v.first, v.second);
            }
        },
        (*c)->value);
}

// Split f into (numerator, denominator): every top-level factor carrying a negative
// constant exponent contributes base^(-exp) to the denominator; all other factors go
// to the numerator. A non-product f is treated as a single factor.
[[nodiscard]] auto as_fraction(const Expr& f) -> Result<std::pair<Expr, Expr>> {
    std::vector<Expr> nums;
    std::vector<Expr> dens;

    auto handle = [&](const Expr& g) -> std::optional<MathError> {
        if (auto p = as<PowerNode>(g.node().value)) {
            auto s = const_sign((*p)->exponent);
            if (s && *s < 0) {
                auto ne = negate_const((*p)->exponent);
                if (!ne) {
                    return ne.error();
                }
                dens.push_back(Expr::power((*p)->base, *ne));
                return std::nullopt;
            }
        }
        nums.push_back(g);
        return std::nullopt;
    };

    if (auto m = as<MulNode>(f.node().value)) {
        for (const Expr& factor : (*m)->factors) {
            if (auto err = handle(factor)) {
                return make_error<std::pair<Expr, Expr>>(*err);
            }
        }
    } else if (auto err = handle(f)) {
        return make_error<std::pair<Expr, Expr>>(*err);
    }

    Expr num = nums.empty() ? Expr::integer(1)
                            : (nums.size() == 1 ? nums.front() : Expr::product(std::move(nums)));
    Expr den = dens.empty() ? Expr::integer(1)
                            : (dens.size() == 1 ? dens.front() : Expr::product(std::move(dens)));
    return std::pair<Expr, Expr>{std::move(num), std::move(den)};
}

// --- L'Hopital at a finite point -------------------------------------------

// Compute lim num/den as var -> point, assuming a possible 0/0 indeterminacy.
// Recurses through L'Hopital, bounded by kLHopitalCap.
[[nodiscard]] auto lhopital(const Expr& num, const Expr& den, std::string_view var,
                            const Expr& point, std::int64_t depth) -> Result<Expr> {
    if (depth > kLHopitalCap) {
        return make_error<Expr>(MathError::not_implemented);
    }

    auto nv = value_at(num, var, point);
    if (!nv) {
        return make_error<Expr>(nv.error() == MathError::overflow ? MathError::overflow
                                                                  : MathError::not_implemented);
    }
    auto dv = value_at(den, var, point);
    if (!dv) {
        return make_error<Expr>(dv.error() == MathError::overflow ? MathError::overflow
                                                                  : MathError::not_implemented);
    }
    if (!is_finite_const(*nv) || !is_finite_const(*dv)) {
        return make_error<Expr>(MathError::not_implemented);
    }

    if (!is_zero_const(*dv)) {
        // Denominator tends to a finite non-zero constant: the limit is num/den,
        // whether or not the numerator vanishes.
        return simplify(Expr::product({*nv, Expr::power(*dv, Expr::integer(-1))}));
    }
    if (!is_zero_const(*nv)) {
        // finite non-zero / zero: a pole, no finite two-sided limit.
        return make_error<Expr>(MathError::domain_error);
    }

    // 0/0 -> differentiate numerator and denominator and retry.
    auto dnum = differentiate(num, var);
    if (!dnum) {
        return dnum;
    }
    auto dden = differentiate(den, var);
    if (!dden) {
        return dden;
    }
    return lhopital(*dnum, *dden, var, point, depth + 1);
}

// --- dense univariate polynomial extraction (for the at-infinity path) -----
// Coefficients are stored low-degree first: coeffs[i] is the coefficient of var^i.
// They are left unsimplified; poly_degree_lead simplifies only what it inspects.

[[nodiscard]] auto poly_add(const std::vector<Expr>& a, const std::vector<Expr>& b)
    -> std::vector<Expr> {
    const std::size_t n = std::max(a.size(), b.size());
    std::vector<Expr> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<Expr> terms;
        if (i < a.size()) {
            terms.push_back(a[i]);
        }
        if (i < b.size()) {
            terms.push_back(b[i]);
        }
        out.push_back(terms.size() == 1 ? terms.front() : Expr::sum(std::move(terms)));
    }
    return out;
}

[[nodiscard]] auto poly_mul(const std::vector<Expr>& a, const std::vector<Expr>& b)
    -> std::vector<Expr> {
    if (a.empty() || b.empty()) {
        return {};
    }
    const std::size_t n = a.size() + b.size() - 1;
    std::vector<std::vector<Expr>> acc(n);
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < b.size(); ++j) {
            acc[i + j].push_back(Expr::product({a[i], b[j]}));
        }
    }
    std::vector<Expr> out;
    out.reserve(n);
    for (auto& terms : acc) {
        out.push_back(terms.size() == 1 ? terms.front() : Expr::sum(std::move(terms)));
    }
    return out;
}

[[nodiscard]] auto poly_pow(const std::vector<Expr>& base, std::int64_t exp)
    -> std::vector<Expr> {
    std::vector<Expr> result{Expr::integer(1)};
    for (std::int64_t k = 0; k < exp; ++k) {
        result = poly_mul(result, base);
    }
    return result;
}

// Dense coefficient vector of e as a polynomial in `var`, or nullopt if e is not a
// polynomial in var (contains var inside a function, a negative/fractional power, etc.).
[[nodiscard]] auto poly_in_var(const Expr& e, std::string_view var)
    -> std::optional<std::vector<Expr>> {
    const Expr vsym = Expr::symbol(std::string(var));
    if (free_of(e, vsym)) {
        return std::vector<Expr>{e};  // degree-0: a var-free coefficient
    }
    if (as<SymbolNode>(e.node().value)) {
        // free_of was false, so this symbol IS var.
        return std::vector<Expr>{Expr::integer(0), Expr::integer(1)};
    }
    if (auto a = as<AddNode>(e.node().value)) {
        std::optional<std::vector<Expr>> acc;
        for (const Expr& term : (*a)->terms) {
            auto pt = poly_in_var(term, var);
            if (!pt) {
                return std::nullopt;
            }
            if (acc) {
                acc = poly_add(*acc, *pt);
            } else {
                acc = std::move(pt);
            }
        }
        return acc;
    }
    if (auto m = as<MulNode>(e.node().value)) {
        std::vector<Expr> acc{Expr::integer(1)};
        for (const Expr& factor : (*m)->factors) {
            auto pf = poly_in_var(factor, var);
            if (!pf) {
                return std::nullopt;
            }
            acc = poly_mul(acc, *pf);
        }
        return acc;
    }
    if (auto p = as<PowerNode>(e.node().value)) {
        auto exp = nonneg_int((*p)->exponent);
        if (!exp || *exp > kPolyPowCap) {
            return std::nullopt;
        }
        auto bp = poly_in_var((*p)->base, var);
        if (!bp) {
            return std::nullopt;
        }
        // Bound the TOTAL expanded degree deg(base)*exp, not just the exponent, so a
        // small crafted power like (x^256+1)^256 cannot blow up into a degree-65536
        // dense polynomial. Written to avoid overflow in the multiplication.
        const std::int64_t base_deg = static_cast<std::int64_t>(bp->size()) - 1;
        if (base_deg > 0 && *exp > 0 && base_deg > kPolyDegreeCap / *exp) {
            return std::nullopt;
        }
        return poly_pow(*bp, *exp);
    }
    // FunctionNode (or anything else) containing var: not a polynomial.
    return std::nullopt;
}

// (degree, leading coefficient) of a dense coefficient vector; inner nullopt means the
// polynomial is identically zero. Propagates simplify errors (e.g. overflow).
[[nodiscard]] auto poly_degree_lead(const std::vector<Expr>& coeffs)
    -> Result<std::optional<std::pair<std::size_t, Expr>>> {
    for (std::size_t i = coeffs.size(); i-- > 0;) {
        auto s = simplify(coeffs[i]);
        if (!s) {
            return make_error<std::optional<std::pair<std::size_t, Expr>>>(s.error());
        }
        if (!is_zero_const(*s)) {
            return std::optional<std::pair<std::size_t, Expr>>{
                std::pair<std::size_t, Expr>{i, *s}};
        }
    }
    return std::optional<std::pair<std::size_t, Expr>>{std::nullopt};
}

// --- continuity domain safety ----------------------------------------------

// Is `exp` a provable integer exponent (any sign)? Integer powers are continuous
// wherever the base is defined; only NON-integer powers (roots) carry a restricted
// real domain (base >= 0), so only those need a boundary check.
[[nodiscard]] auto is_integer_exponent(const Expr& exp) -> bool {
    auto c = as<ConstantNode>(exp.node().value);
    if (!c) {
        return false;  // symbolic exponent: treat conservatively as non-integer
    }
    return std::visit(
        []<typename V>(const V& v) -> bool {
            if constexpr (std::is_same_v<V, std::int64_t>) {
                return true;
            } else if constexpr (std::is_same_v<V, double>) {
                return std::isfinite(v) && std::floor(v) == v;
            } else {  // pair num/den
                return v.second == 1;
            }
        },
        (*c)->value);
}

// Elementary functions whose real domain is arg > 0 (interior). Substitution is a valid
// limit only strictly inside the domain.
[[nodiscard]] auto is_positive_domain_fn(std::string_view name) -> bool {
    return name == "log" || name == "ln" || name == "log2" || name == "log10" ||
           name == "sqrt";
}
// Functions whose domain is a bounded interval (asin/acos: |arg|<1; acosh: arg>1;
// atanh: |arg|<1) — we do not cheaply verify their interior, so a var-dependent
// argument is treated conservatively as unsafe.
[[nodiscard]] auto is_bounded_domain_fn(std::string_view name) -> bool {
    return name == "asin" || name == "acos" || name == "acosh" || name == "atanh";
}

// Conservative continuity-safety check for direct substitution. A finite substituted
// value is the limit ONLY at an INTERIOR point of the real domain: at a boundary a root
// or logarithm is one-sided-only or undefined, so substitution must not be trusted
// there. Returns false whenever `e` contains such a feature at `point` that we cannot
// rule out — a non-integer power base^(p/q) or a domain-restricted function whose
// var-dependent argument is not provably strictly interior. Overflow while probing the
// argument propagates on the railway.
[[nodiscard]] auto continuity_safe(const Expr& e, std::string_view var, const Expr& point)
    -> Result<bool> {
    const Expr vsym = Expr::symbol(std::string(var));
    const ExprNode& n = e.node();

    if (auto p = as<PowerNode>(n.value)) {
        if (!is_integer_exponent((*p)->exponent) && !free_of((*p)->base, vsym)) {
            auto bval = value_at((*p)->base, var, point);
            if (!bval) {
                if (bval.error() == MathError::overflow) {
                    return make_error<bool>(MathError::overflow);
                }
                return false;  // base undefined at point -> not an interior point
            }
            auto sgn = const_sign(*bval);
            if (!sgn || *sgn <= 0) {
                return false;  // boundary (0), outside (<0), or non-constant -> unsafe
            }
        }
        for (const Expr& child : {(*p)->base, (*p)->exponent}) {
            auto r = continuity_safe(child, var, point);
            if (!r) {
                return r;
            }
            if (!*r) {
                return false;
            }
        }
        return true;
    }

    if (auto fn = as<FunctionNode>(n.value)) {
        const std::string& name = (*fn)->name;
        for (const Expr& arg : (*fn)->args) {
            if (free_of(arg, vsym)) {
                continue;
            }
            if (is_positive_domain_fn(name)) {
                auto av = value_at(arg, var, point);
                if (!av) {
                    if (av.error() == MathError::overflow) {
                        return make_error<bool>(MathError::overflow);
                    }
                    return false;
                }
                auto sgn = const_sign(*av);
                if (!sgn || *sgn <= 0) {
                    return false;
                }
            } else if (is_bounded_domain_fn(name)) {
                return false;  // bounded-interval domain we cannot cheaply verify
            }
        }
        for (const Expr& arg : (*fn)->args) {
            auto r = continuity_safe(arg, var, point);
            if (!r) {
                return r;
            }
            if (!*r) {
                return false;
            }
        }
        return true;
    }

    if (auto a = as<AddNode>(n.value)) {
        for (const Expr& t : (*a)->terms) {
            auto r = continuity_safe(t, var, point);
            if (!r) {
                return r;
            }
            if (!*r) {
                return false;
            }
        }
        return true;
    }

    if (auto m = as<MulNode>(n.value)) {
        for (const Expr& factor : (*m)->factors) {
            auto r = continuity_safe(factor, var, point);
            if (!r) {
                return r;
            }
            if (!*r) {
                return false;
            }
        }
        return true;
    }

    // SymbolNode / ConstantNode: always continuity-safe.
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry points.
// ---------------------------------------------------------------------------

auto limit(const Expr& f, std::string_view var, const Expr& point,
           [[maybe_unused]] LimitDirection dir) -> Result<Expr> {
    // A limit "at" an infinity marker forwards to the dedicated routine.
    if (auto marker = is_inf_marker(point)) {
        return limit_at_infinity(f, var, *marker);
    }
    if (!is_finite_const(point)) {
        // Symbolic / non-constant limit points are out of scope.
        return make_error<Expr>(MathError::not_implemented);
    }

    auto sf = simplify(f);
    if (!sf) {
        return sf;
    }

    // 1. Continuity: direct substitution to a finite constant is the answer — but only
    //    at an INTERIOR point of the real domain. A root or logarithm evaluated at its
    //    domain boundary folds to a finite value that is one-sided-only or undefined, so
    //    we must not trust it (nor fall through to L'Hopital, which would recompute the
    //    same num/den value); we admit not_implemented instead.
    auto direct = value_at(*sf, var, point);
    if (direct && is_finite_const(*direct)) {
        auto safe = continuity_safe(*sf, var, point);
        if (!safe) {
            return make_error<Expr>(safe.error());
        }
        if (*safe) {
            return *direct;
        }
        return make_error<Expr>(MathError::not_implemented);
    }
    if (!direct && direct.error() == MathError::overflow) {
        return make_error<Expr>(MathError::overflow);
    }

    // 2. Indeterminate / non-constant: split into a quotient and apply L'Hopital.
    auto frac = as_fraction(*sf);
    if (!frac) {
        return make_error<Expr>(frac.error());
    }
    return lhopital(frac->first, frac->second, var, point, 0);
}

auto limit_at_infinity(const Expr& f, std::string_view var, bool positive) -> Result<Expr> {
    auto sf = simplify(f);
    if (!sf) {
        return sf;
    }
    auto frac = as_fraction(*sf);
    if (!frac) {
        return make_error<Expr>(frac.error());
    }
    const auto& [num, den] = *frac;

    auto pnum = poly_in_var(num, var);
    auto pden = poly_in_var(den, var);
    if (!pnum || !pden) {
        // Not a rational function in var: outside this module's decidable class.
        return make_error<Expr>(MathError::not_implemented);
    }

    auto dlP = poly_degree_lead(*pnum);
    if (!dlP) {
        return make_error<Expr>(dlP.error());
    }
    auto dlQ = poly_degree_lead(*pden);
    if (!dlQ) {
        return make_error<Expr>(dlQ.error());
    }
    if (!dlQ->has_value()) {
        return make_error<Expr>(MathError::domain_error);  // denominator identically zero
    }
    if (!dlP->has_value()) {
        return Expr::integer(0);  // numerator identically zero
    }

    const auto [degP, leadP] = **dlP;
    const auto [degQ, leadQ] = **dlQ;

    // Soundness guard: the degree comparison is valid only when the denominator's
    // leading coefficient is a DEFINITE nonzero constant. A symbolic leadQ (e.g. `a` in
    // a*x^2 + x) could vanish, dropping the denominator's true degree and flipping the
    // comparison, so emitting a value on that locus would be wrong (x/(a*x^2+x) is 0 for
    // a!=0 but 1 at a=0). Return an honest not_implemented instead. (poly_degree_lead
    // already skipped provably-zero coefficients, so a constant leadQ is necessarily
    // nonzero.) A symbolic leadP is safe: degP can only drop, and the degP==degQ result
    // leadP/leadQ still equals the true limit 0 on the leadP==0 locus; the degP>degQ
    // branch is separately guarded by const_sign below.
    if (!is_finite_const(leadQ)) {
        return make_error<Expr>(MathError::not_implemented);
    }

    if (degP < degQ) {
        return Expr::integer(0);
    }

    // ratio of leading coefficients (exact over Q).
    auto ratio = simplify(Expr::product({leadP, Expr::power(leadQ, Expr::integer(-1))}));
    if (!ratio) {
        return ratio;
    }
    if (degP == degQ) {
        return *ratio;  // finite: leadP / leadQ
    }

    // degP > degQ: diverges to a signed infinity. Sign at +inf is sign(leadP/leadQ);
    // at -inf it additionally flips when the degree gap is odd.
    auto s = const_sign(*ratio);
    if (!s) {
        return make_error<Expr>(MathError::not_implemented);  // indeterminate sign
    }
    int sign = *s;
    if (!positive && ((degP - degQ) % 2 == 1)) {
        sign = -sign;
    }
    return sign >= 0 ? pos_inf() : neg_inf();
}

}  // namespace nimblecas
