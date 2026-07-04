// NimbleCAS tensor calculus / differential geometry (ROADMAP 7, extends the
// vector-calculus of 7.19).
// @author Olumuyiwa Oluwasanmi
//
// This module lifts the flat, Cartesian vector operators of nimblecas.vectorcalc onto a
// Riemannian (or pseudo-Riemannian) manifold described by a symbolic metric tensor. The
// metric components are EXACT symbolic expressions (Expr) — e.g. the round 2-sphere metric
// is diag(a^2, a^2*sin(theta)^2) — so we deliberately do NOT reuse nimblecas.matrix (which is
// Rational-only). Everything is computed by composing the exact symbolic differentiation
// engine (nimblecas.diff) with automatic simplification (nimblecas.simplify); there is no
// floating point and no tolerance anywhere. Failure is reported only on the railway
// (Result<T> = std::expected<T, MathError>, Rule 32).
//
// What is computed (all indices are 0-based over the coordinate list):
//   * metric_determinant / inverse_metric  — a small SYMBOLIC determinant and inverse over
//     Expr by cofactor (Laplace) expansion, simplifying each entry. Bounded to dimension
//     n <= 5 (beyond that -> not_implemented); a metric whose determinant simplifies to the
//     exact constant 0 is singular and has no inverse (-> domain_error).
//   * christoffel                Gamma^k_{ij} = (1/2) g^{kl}(d_i g_{jl} + d_j g_{il} - d_l g_{ij}),
//                                a rank-3 array indexed [k][i][j].
//   * covariant_derivative_vector nabla_i v^k = d_i v^k + Gamma^k_{ij} v^j (contravariant field).
//   * riemann_tensor             R^rho_{sigma mu nu} = d_mu Gamma^rho_{nu sigma} - d_nu Gamma^rho_{mu sigma}
//                                + Gamma^rho_{mu lam}Gamma^lam_{nu sigma} - Gamma^rho_{nu lam}Gamma^lam_{mu sigma},
//                                indexed [rho][sigma][mu][nu].
//   * ricci_tensor  R_{mu nu} = R^rho_{mu rho nu};  scalar_curvature R = g^{mu nu} R_{mu nu};
//     einstein_tensor  G_{mu nu} = R_{mu nu} - (1/2) R g_{mu nu}.
//   * geodesic_coefficients      the Gamma^k_{ij} packaged for x''^k + Gamma^k_{ij} x'^i x'^j = 0.
//   * metric_gradient (raise the index of d_i f), covariant_divergence, and laplace_beltrami
//     (1/sqrt(g)) d_i(sqrt(g) g^{ij} d_j f) — the curvilinear generalisations of 7.19.
//
// HONESTY BOUNDARY. Index raising and sqrt(g) introduce roots and reciprocals; these are
// represented EXACTLY as Expr (power(g, 1/2), power(x, -1)) — never as decimals. Because the
// simplifier performs no trig identities and combines same-base powers only after their
// exponents become integral, a single simplify() pass can leave residual, un-fused factors
// such as r^2 * r^(-2) or (r^2)^(1/2) * (r^2)^(-1/2). We therefore drive simplify() to a
// bounded fixed point (simp_to_fixed_point) and, for the divergence-form operators, distribute
// the outer 1/sqrt(g) across the differentiated sum so the sqrt factors cancel structurally.
// This is still exact: every step is an identity. If a result cannot be reduced to a pretty
// closed form it is STILL returned as the exact (merely un-simplified) Expr — never fudged.

export module nimblecas.tensor;

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.diff;
import nimblecas.simplify;

export namespace nimblecas {

using ExprVector = std::vector<Expr>;
using ExprMatrix = std::vector<std::vector<Expr>>;
using ExprRank3 = std::vector<ExprMatrix>;              // [i][j][k]
using ExprRank4 = std::vector<ExprRank3>;               // [i][j][k][l]

// A rank-2 covariant metric tensor g_{ij} over a named coordinate chart. The components are
// exact symbolic expressions; the tensor is assumed symmetric (g_{ij} = g_{ji}).
struct Metric {
    std::vector<std::string> coords;  // coordinate names x^0, x^1, ...
    ExprMatrix g;                     // g_{ij}, coords.size() x coords.size()

    [[nodiscard]] auto dim() const noexcept -> std::size_t { return coords.size(); }
};

// Validate squareness/size and build a Metric. Fails with domain_error on an empty chart or
// a non-square component matrix.
[[nodiscard]] auto make_metric(std::vector<std::string> coords, ExprMatrix g) -> Result<Metric>;

// det(g_{ij}) via exact symbolic cofactor expansion (n <= 5, else not_implemented).
[[nodiscard]] auto metric_determinant(const Metric& m) -> Result<Expr>;

// The inverse metric g^{ij} (n <= 5, else not_implemented; singular metric -> domain_error).
[[nodiscard]] auto inverse_metric(const Metric& m) -> Result<ExprMatrix>;

// Christoffel symbols of the second kind, Gamma^k_{ij}, indexed [k][i][j].
[[nodiscard]] auto christoffel(const Metric& m) -> Result<ExprRank3>;

// Covariant derivative of a contravariant vector field v^k: result[i][k] = nabla_i v^k.
// Requires v.size() == dim().
[[nodiscard]] auto covariant_derivative_vector(const Metric& m, const ExprVector& v)
    -> Result<ExprMatrix>;

// Riemann curvature tensor R^rho_{sigma mu nu}, indexed [rho][sigma][mu][nu].
[[nodiscard]] auto riemann_tensor(const Metric& m) -> Result<ExprRank4>;

// Ricci tensor R_{mu nu} = R^rho_{mu rho nu}, indexed [mu][nu].
[[nodiscard]] auto ricci_tensor(const Metric& m) -> Result<ExprMatrix>;

// Ricci scalar R = g^{mu nu} R_{mu nu}.
[[nodiscard]] auto scalar_curvature(const Metric& m) -> Result<Expr>;

// Einstein tensor G_{mu nu} = R_{mu nu} - (1/2) R g_{mu nu}, indexed [mu][nu].
[[nodiscard]] auto einstein_tensor(const Metric& m) -> Result<ExprMatrix>;

// Geodesic coefficients: the Gamma^k_{ij} for x''^k + Gamma^k_{ij} x'^i x'^j = 0
// (identical data to christoffel(), named for the geodesic-equation use site).
[[nodiscard]] auto geodesic_coefficients(const Metric& m) -> Result<ExprRank3>;

// Raised-index gradient grad^i f = g^{ij} d_j f.
[[nodiscard]] auto metric_gradient(const Metric& m, const Expr& f) -> Result<ExprVector>;

// Covariant divergence of a contravariant field: div v = (1/sqrt(g)) d_i(sqrt(g) v^i).
// Requires field.size() == dim().
[[nodiscard]] auto covariant_divergence(const Metric& m, const ExprVector& field) -> Result<Expr>;

// Laplace-Beltrami operator: (1/sqrt(g)) d_i(sqrt(g) g^{ij} d_j f).
[[nodiscard]] auto laplace_beltrami(const Metric& m, const Expr& f) -> Result<Expr>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// Cofactor expansion is exponential; cap the dimension so a mistyped metric cannot spin.
inline constexpr std::size_t max_cofactor_dim = 5;

// Extract e as an integer exponent (int64, or exact rational with denominator 1), else nullopt.
[[nodiscard]] auto as_int_exponent(const Expr& e) -> std::optional<std::int64_t> {
    const auto* c = std::get_if<ConstantNode>(&e.node().value);
    if (c == nullptr) {
        return std::nullopt;
    }
    return std::visit(
        []<typename V>(const V& v) -> std::optional<std::int64_t> {
            if constexpr (std::is_same_v<V, std::int64_t>) {
                return v;
            } else if constexpr (std::is_same_v<V, double>) {
                return std::nullopt;
            } else {  // pair<int64,int64>
                return v.second == 1 ? std::optional<std::int64_t>(v.first) : std::nullopt;
            }
        },
        c->value);
}

// Distribute an INTEGER power over a product, (f_0 * ... * f_m)^k -> f_0^k * ... * f_m^k, and
// flatten (a^m)^k -> a^(m*k). This is an exact identity for integer k. simplify() does NOT do
// it, yet it is exactly what frees same-base factors trapped inside an inverse-metric
// denominator like (a^4 * sin(theta)^2)^(-1) so they cancel against the numerator's a^4 /
// sin(theta)^3. Every other node is rebuilt from expanded children. Pure structural rewrite.
[[nodiscard]] auto expand_powers(const Expr& e) -> Expr {
    const ExprNode& node = e.node();
    if (const auto* pw = std::get_if<PowerNode>(&node.value)) {
        Expr base = expand_powers(pw->base);
        Expr exponent = expand_powers(pw->exponent);
        if (auto k = as_int_exponent(exponent)) {
            if (const auto* mul = std::get_if<MulNode>(&base.node().value)) {
                std::vector<Expr> factors;
                factors.reserve(mul->factors.size());
                for (const Expr& f : mul->factors) {
                    factors.push_back(Expr::power(f, Expr::integer(*k)));
                }
                return Expr::product(std::move(factors));
            }
            if (const auto* inner = std::get_if<PowerNode>(&base.node().value)) {
                if (auto m = as_int_exponent(inner->exponent)) {
                    std::int64_t prod = 0;
                    if (!__builtin_mul_overflow(*m, *k, &prod)) {
                        return Expr::power(inner->base, Expr::integer(prod));
                    }
                }
            }
        }
        return Expr::power(std::move(base), std::move(exponent));
    }
    if (const auto* mul = std::get_if<MulNode>(&node.value)) {
        std::vector<Expr> factors;
        factors.reserve(mul->factors.size());
        for (const Expr& f : mul->factors) {
            factors.push_back(expand_powers(f));
        }
        return Expr::product(std::move(factors));
    }
    if (const auto* add = std::get_if<AddNode>(&node.value)) {
        std::vector<Expr> terms;
        terms.reserve(add->terms.size());
        for (const Expr& tm : add->terms) {
            terms.push_back(expand_powers(tm));
        }
        return Expr::sum(std::move(terms));
    }
    if (const auto* fn = std::get_if<FunctionNode>(&node.value)) {
        std::vector<Expr> args;
        args.reserve(fn->args.size());
        for (const Expr& arg : fn->args) {
            args.push_back(expand_powers(arg));
        }
        return Expr::apply(fn->name, std::move(args));
    }
    return e;  // SymbolNode / ConstantNode
}

// Fully distribute products over sums, (a+b)*(c+d) -> ac+ad+bc+bd, recursively (and expand
// integer powers first). simplify() never distributes, so like terms trapped inside an
// undistributed product such as ((-cot^2 + -1) * -1) + (-cot^2) stay uncancelled; flattening to
// a sum-of-products lets simplify combine ALL like terms (here to 1). Exact identity. A soft cap
// keeps a pathological product from exploding — beyond it that factor is left undistributed
// (still exact; simplify runs regardless). Powers of a SUM are deliberately NOT expanded (that is
// not needed here and (a+b)^(-1) must never become a^(-1)+b^(-1)).
[[nodiscard]] auto distribute_all(const Expr& e) -> Expr {
    Expr pe = expand_powers(e);
    const ExprNode& node = pe.node();
    if (const auto* add = std::get_if<AddNode>(&node.value)) {
        std::vector<Expr> terms;
        terms.reserve(add->terms.size());
        for (const Expr& t : add->terms) {
            terms.push_back(distribute_all(t));
        }
        return Expr::sum(std::move(terms));
    }
    if (const auto* mul = std::get_if<MulNode>(&node.value)) {
        constexpr std::size_t cap = 8192;
        std::vector<Expr> result{Expr::integer(1)};
        for (const Expr& f : mul->factors) {
            Expr df = distribute_all(f);
            std::vector<Expr> fterms;
            if (const auto* fadd = std::get_if<AddNode>(&df.node().value)) {
                fterms.assign(fadd->terms.begin(), fadd->terms.end());
            } else {
                fterms.push_back(df);
            }
            std::vector<Expr> next;
            if (result.size() * fterms.size() > cap) {
                next.reserve(result.size());
                for (const Expr& r : result) {
                    next.push_back(Expr::product({r, df}));  // leave this factor undistributed
                }
            } else {
                next.reserve(result.size() * fterms.size());
                for (const Expr& r : result) {
                    for (const Expr& ft : fterms) {
                        next.push_back(Expr::product({r, ft}));
                    }
                }
            }
            result = std::move(next);
        }
        return result.size() == 1 ? std::move(result.front()) : Expr::sum(std::move(result));
    }
    if (const auto* fn = std::get_if<FunctionNode>(&node.value)) {
        std::vector<Expr> args;
        args.reserve(fn->args.size());
        for (const Expr& arg : fn->args) {
            args.push_back(distribute_all(arg));
        }
        return Expr::apply(fn->name, std::move(args));
    }
    return pe;  // SymbolNode / ConstantNode / power of a non-product base
}

// Drive simplify() to a bounded fixed point. A single pass can leave residual same-base
// powers un-fused (e.g. r^2 * r^(-2)) and never distributes powers over products or products
// over sums, so we first distribute_all() (which also expands integer powers) then simplify(),
// iterating until the expression stops changing. Every step is an exact identity, so the fixed
// point is exact; the bound merely guarantees termination.
[[nodiscard]] auto simp_to_fixed_point(const Expr& e) -> Result<Expr> {
    constexpr int max_passes = 8;
    Expr current = e;
    for (int pass = 0; pass < max_passes; ++pass) {
        auto next = simplify(distribute_all(current));
        if (!next) {
            return next;
        }
        if (next->is_equivalent_to(current)) {
            return *next;  // fixed point reached
        }
        current = std::move(*next);
    }
    return current;  // exact, just not provably minimal within the pass budget
}

// True iff e is the exact integer constant 0 (used for singular-metric detection).
[[nodiscard]] auto is_exact_zero(const Expr& e) -> bool {
    return e.is_equivalent_to(Expr::integer(0));
}

// a - b as a symbolic expression.
[[nodiscard]] auto sub(const Expr& a, const Expr& b) -> Expr {
    return Expr::sum({a, Expr::product({Expr::integer(-1), b})});
}

// The rational literals 1/2 and -1/2 are always well-formed (non-zero denominator).
[[nodiscard]] auto one_half() -> Expr { return Expr::rational(1, 2).value(); }
[[nodiscard]] auto neg_one_half() -> Expr { return Expr::rational(-1, 2).value(); }

// A single partial derivative d(expr)/d(coords[k]); differentiate() already simplifies.
[[nodiscard]] auto partial(const Expr& expr, std::string_view var) -> Result<Expr> {
    return differentiate(expr, var);
}

// True iff the metric's component matrix is a non-empty square of the coordinate count.
[[nodiscard]] auto dimensions_ok(const Metric& m) -> bool {
    const std::size_t n = m.coords.size();
    if (n == 0 || m.g.size() != n) {
        return false;
    }
    return std::ranges::all_of(m.g, [n](const std::vector<Expr>& row) { return row.size() == n; });
}

// The (n-1)x(n-1) submatrix of mat with row r and column c removed.
[[nodiscard]] auto minor_matrix(const ExprMatrix& mat, std::size_t r, std::size_t c)
    -> ExprMatrix {
    ExprMatrix out;
    out.reserve(mat.size() - 1);
    for (std::size_t i = 0; i < mat.size(); ++i) {
        if (i == r) {
            continue;
        }
        std::vector<Expr> row;
        row.reserve(mat.size() - 1);
        for (std::size_t j = 0; j < mat[i].size(); ++j) {
            if (j == c) {
                continue;
            }
            row.push_back(mat[i][j]);
        }
        out.push_back(std::move(row));
    }
    return out;
}

// Exact symbolic determinant by Laplace expansion along the first row. Recurses over minors;
// every intermediate is simplified so cancellations collapse. Assumes a square matrix.
[[nodiscard]] auto symbolic_determinant(const ExprMatrix& mat) -> Result<Expr> {
    const std::size_t n = mat.size();
    if (n == 0) {
        return Expr::integer(1);  // determinant of the empty matrix
    }
    if (n == 1) {
        return simp_to_fixed_point(mat[0][0]);
    }
    std::vector<Expr> terms;
    terms.reserve(n);
    for (std::size_t j = 0; j < n; ++j) {
        auto minor_det = symbolic_determinant(minor_matrix(mat, 0, j));
        if (!minor_det) {
            return minor_det;
        }
        const std::int64_t sign = (j % 2 == 0) ? 1 : -1;
        terms.push_back(Expr::product({Expr::integer(sign), mat[0][j], *minor_det}));
    }
    return simp_to_fixed_point(Expr::sum(std::move(terms)));
}

// Distribute a single factor across the additive terms of expr, simplifying each product to
// a fixed point. This is what makes the outer 1/sqrt(g) cancel the inner sqrt(g) structurally
// in the divergence-form operators: simplify() never distributes a product over a sum on its
// own, so factor * (A + B) would otherwise stay factored and the sqrt powers would not meet.
[[nodiscard]] auto distribute_factor(const Expr& factor, const Expr& expr) -> Result<Expr> {
    auto reduced = simp_to_fixed_point(expr);
    if (!reduced) {
        return reduced;
    }
    std::vector<Expr> source_terms;
    if (const auto* add = std::get_if<AddNode>(&reduced->node().value)) {
        source_terms = add->terms;
    } else {
        source_terms.push_back(*reduced);
    }
    std::vector<Expr> out;
    out.reserve(source_terms.size());
    for (const Expr& term : source_terms) {
        auto product = simp_to_fixed_point(Expr::product({factor, term}));
        if (!product) {
            return product;
        }
        out.push_back(std::move(*product));
    }
    return simp_to_fixed_point(Expr::sum(std::move(out)));
}

}  // namespace

auto make_metric(std::vector<std::string> coords, ExprMatrix g) -> Result<Metric> {
    Metric m{.coords = std::move(coords), .g = std::move(g)};
    if (!dimensions_ok(m)) {
        return make_error<Metric>(MathError::domain_error);
    }
    return m;
}

auto metric_determinant(const Metric& m) -> Result<Expr> {
    if (!dimensions_ok(m)) {
        return make_error<Expr>(MathError::domain_error);
    }
    if (m.dim() > max_cofactor_dim) {
        return make_error<Expr>(MathError::not_implemented);
    }
    return symbolic_determinant(m.g);
}

auto inverse_metric(const Metric& m) -> Result<ExprMatrix> {
    if (!dimensions_ok(m)) {
        return make_error<ExprMatrix>(MathError::domain_error);
    }
    const std::size_t n = m.dim();
    if (n > max_cofactor_dim) {
        return make_error<ExprMatrix>(MathError::not_implemented);
    }
    auto det = symbolic_determinant(m.g);
    if (!det) {
        return make_error<ExprMatrix>(det.error());
    }
    if (is_exact_zero(*det)) {
        return make_error<ExprMatrix>(MathError::domain_error);  // singular: no inverse exists
    }
    const Expr det_inverse = Expr::power(*det, Expr::integer(-1));

    // (g^{-1})_{ij} = cofactor_{ji} / det = ((-1)^{i+j} det(minor_{ji})) / det.
    ExprMatrix inverse(n, std::vector<Expr>(n, Expr::integer(0)));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            auto cof = symbolic_determinant(minor_matrix(m.g, j, i));
            if (!cof) {
                return make_error<ExprMatrix>(cof.error());
            }
            const std::int64_t sign = ((i + j) % 2 == 0) ? 1 : -1;
            auto entry = simp_to_fixed_point(
                Expr::product({Expr::integer(sign), *cof, det_inverse}));
            if (!entry) {
                return make_error<ExprMatrix>(entry.error());
            }
            inverse[i][j] = std::move(*entry);
        }
    }
    return inverse;
}

auto christoffel(const Metric& m) -> Result<ExprRank3> {
    if (!dimensions_ok(m)) {
        return make_error<ExprRank3>(MathError::domain_error);
    }
    const std::size_t n = m.dim();
    auto ginv = inverse_metric(m);
    if (!ginv) {
        return make_error<ExprRank3>(ginv.error());
    }

    // Precompute d_l g_{ij} to avoid recomputing shared partial derivatives.
    // dg[l][i][j] = d(g_{ij})/d(x^l).
    ExprRank3 dg(n, ExprMatrix(n, std::vector<Expr>(n, Expr::integer(0))));
    for (std::size_t l = 0; l < n; ++l) {
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                auto d = partial(m.g[i][j], m.coords[l]);
                if (!d) {
                    return make_error<ExprRank3>(d.error());
                }
                dg[l][i][j] = std::move(*d);
            }
        }
    }

    ExprRank3 gamma(n, ExprMatrix(n, std::vector<Expr>(n, Expr::integer(0))));
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                // Gamma^k_{ij} = (1/2) sum_l g^{kl} (d_i g_{jl} + d_j g_{il} - d_l g_{ij}).
                std::vector<Expr> contraction;
                contraction.reserve(n);
                for (std::size_t l = 0; l < n; ++l) {
                    const Expr bracket =
                        Expr::sum({dg[i][j][l], dg[j][i][l], sub(Expr::integer(0), dg[l][i][j])});
                    contraction.push_back(Expr::product({(*ginv)[k][l], bracket}));
                }
                auto value = simp_to_fixed_point(
                    Expr::product({one_half(), Expr::sum(std::move(contraction))}));
                if (!value) {
                    return make_error<ExprRank3>(value.error());
                }
                gamma[k][i][j] = std::move(*value);
            }
        }
    }
    return gamma;
}

auto covariant_derivative_vector(const Metric& m, const ExprVector& v) -> Result<ExprMatrix> {
    if (!dimensions_ok(m)) {
        return make_error<ExprMatrix>(MathError::domain_error);
    }
    const std::size_t n = m.dim();
    if (v.size() != n) {
        return make_error<ExprMatrix>(MathError::domain_error);
    }
    auto gamma = christoffel(m);
    if (!gamma) {
        return make_error<ExprMatrix>(gamma.error());
    }

    // result[i][k] = nabla_i v^k = d_i v^k + sum_j Gamma^k_{ij} v^j.
    ExprMatrix result(n, std::vector<Expr>(n, Expr::integer(0)));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            auto dv = partial(v[k], m.coords[i]);
            if (!dv) {
                return make_error<ExprMatrix>(dv.error());
            }
            std::vector<Expr> terms;
            terms.reserve(n + 1);
            terms.push_back(std::move(*dv));
            for (std::size_t j = 0; j < n; ++j) {
                terms.push_back(Expr::product({(*gamma)[k][i][j], v[j]}));
            }
            auto value = simp_to_fixed_point(Expr::sum(std::move(terms)));
            if (!value) {
                return make_error<ExprMatrix>(value.error());
            }
            result[i][k] = std::move(*value);
        }
    }
    return result;
}

auto riemann_tensor(const Metric& m) -> Result<ExprRank4> {
    if (!dimensions_ok(m)) {
        return make_error<ExprRank4>(MathError::domain_error);
    }
    const std::size_t n = m.dim();
    auto gamma = christoffel(m);
    if (!gamma) {
        return make_error<ExprRank4>(gamma.error());
    }

    // Precompute d_mu Gamma^rho_{nu sigma}: dgamma[mu][rho][nu][sigma].
    ExprRank4 dgamma(
        n, ExprRank3(n, ExprMatrix(n, std::vector<Expr>(n, Expr::integer(0)))));
    for (std::size_t mu = 0; mu < n; ++mu) {
        for (std::size_t rho = 0; rho < n; ++rho) {
            for (std::size_t a = 0; a < n; ++a) {
                for (std::size_t b = 0; b < n; ++b) {
                    auto d = partial((*gamma)[rho][a][b], m.coords[mu]);
                    if (!d) {
                        return make_error<ExprRank4>(d.error());
                    }
                    dgamma[mu][rho][a][b] = std::move(*d);
                }
            }
        }
    }

    ExprRank4 riemann(
        n, ExprRank3(n, ExprMatrix(n, std::vector<Expr>(n, Expr::integer(0)))));
    for (std::size_t rho = 0; rho < n; ++rho) {
        for (std::size_t sigma = 0; sigma < n; ++sigma) {
            for (std::size_t mu = 0; mu < n; ++mu) {
                for (std::size_t nu = 0; nu < n; ++nu) {
                    // R^rho_{sigma mu nu} = d_mu Gamma^rho_{nu sigma} - d_nu Gamma^rho_{mu sigma}
                    //   + sum_l Gamma^rho_{mu l} Gamma^l_{nu sigma}
                    //   - sum_l Gamma^rho_{nu l} Gamma^l_{mu sigma}.
                    std::vector<Expr> terms;
                    terms.reserve(2 + 2 * n);
                    terms.push_back(dgamma[mu][rho][nu][sigma]);
                    terms.push_back(Expr::product(
                        {Expr::integer(-1), dgamma[nu][rho][mu][sigma]}));
                    for (std::size_t l = 0; l < n; ++l) {
                        terms.push_back(Expr::product(
                            {(*gamma)[rho][mu][l], (*gamma)[l][nu][sigma]}));
                        terms.push_back(Expr::product(
                            {Expr::integer(-1), (*gamma)[rho][nu][l], (*gamma)[l][mu][sigma]}));
                    }
                    auto value = simp_to_fixed_point(Expr::sum(std::move(terms)));
                    if (!value) {
                        return make_error<ExprRank4>(value.error());
                    }
                    riemann[rho][sigma][mu][nu] = std::move(*value);
                }
            }
        }
    }
    return riemann;
}

auto ricci_tensor(const Metric& m) -> Result<ExprMatrix> {
    if (!dimensions_ok(m)) {
        return make_error<ExprMatrix>(MathError::domain_error);
    }
    const std::size_t n = m.dim();
    auto riemann = riemann_tensor(m);
    if (!riemann) {
        return make_error<ExprMatrix>(riemann.error());
    }

    // R_{mu nu} = R^rho_{mu rho nu} (contract the first upper with the second-to-last lower).
    ExprMatrix ricci(n, std::vector<Expr>(n, Expr::integer(0)));
    for (std::size_t mu = 0; mu < n; ++mu) {
        for (std::size_t nu = 0; nu < n; ++nu) {
            std::vector<Expr> terms;
            terms.reserve(n);
            for (std::size_t rho = 0; rho < n; ++rho) {
                terms.push_back((*riemann)[rho][mu][rho][nu]);
            }
            auto value = simp_to_fixed_point(Expr::sum(std::move(terms)));
            if (!value) {
                return make_error<ExprMatrix>(value.error());
            }
            ricci[mu][nu] = std::move(*value);
        }
    }
    return ricci;
}

auto scalar_curvature(const Metric& m) -> Result<Expr> {
    if (!dimensions_ok(m)) {
        return make_error<Expr>(MathError::domain_error);
    }
    const std::size_t n = m.dim();
    auto ginv = inverse_metric(m);
    if (!ginv) {
        return make_error<Expr>(ginv.error());
    }
    auto ricci = ricci_tensor(m);
    if (!ricci) {
        return make_error<Expr>(ricci.error());
    }

    // R = g^{mu nu} R_{mu nu}.
    std::vector<Expr> terms;
    terms.reserve(n * n);
    for (std::size_t mu = 0; mu < n; ++mu) {
        for (std::size_t nu = 0; nu < n; ++nu) {
            terms.push_back(Expr::product({(*ginv)[mu][nu], (*ricci)[mu][nu]}));
        }
    }
    return simp_to_fixed_point(Expr::sum(std::move(terms)));
}

auto einstein_tensor(const Metric& m) -> Result<ExprMatrix> {
    if (!dimensions_ok(m)) {
        return make_error<ExprMatrix>(MathError::domain_error);
    }
    const std::size_t n = m.dim();
    auto ricci = ricci_tensor(m);
    if (!ricci) {
        return make_error<ExprMatrix>(ricci.error());
    }
    auto scalar = scalar_curvature(m);
    if (!scalar) {
        return make_error<ExprMatrix>(scalar.error());
    }

    // G_{mu nu} = R_{mu nu} - (1/2) R g_{mu nu}.
    ExprMatrix einstein(n, std::vector<Expr>(n, Expr::integer(0)));
    for (std::size_t mu = 0; mu < n; ++mu) {
        for (std::size_t nu = 0; nu < n; ++nu) {
            const Expr correction =
                Expr::product({neg_one_half(), *scalar, m.g[mu][nu]});
            auto value = simp_to_fixed_point(Expr::sum({(*ricci)[mu][nu], correction}));
            if (!value) {
                return make_error<ExprMatrix>(value.error());
            }
            einstein[mu][nu] = std::move(*value);
        }
    }
    return einstein;
}

auto geodesic_coefficients(const Metric& m) -> Result<ExprRank3> {
    // The geodesic equation x''^k + Gamma^k_{ij} x'^i x'^j = 0 is governed exactly by the
    // Christoffel symbols; expose them under the geodesic name.
    return christoffel(m);
}

auto metric_gradient(const Metric& m, const Expr& f) -> Result<ExprVector> {
    if (!dimensions_ok(m)) {
        return make_error<ExprVector>(MathError::domain_error);
    }
    const std::size_t n = m.dim();
    auto ginv = inverse_metric(m);
    if (!ginv) {
        return make_error<ExprVector>(ginv.error());
    }

    // Precompute the covariant gradient d_j f once.
    std::vector<Expr> df;
    df.reserve(n);
    for (std::size_t j = 0; j < n; ++j) {
        auto d = partial(f, m.coords[j]);
        if (!d) {
            return make_error<ExprVector>(d.error());
        }
        df.push_back(std::move(*d));
    }

    // grad^i f = sum_j g^{ij} d_j f.
    ExprVector grad;
    grad.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<Expr> terms;
        terms.reserve(n);
        for (std::size_t j = 0; j < n; ++j) {
            terms.push_back(Expr::product({(*ginv)[i][j], df[j]}));
        }
        auto value = simp_to_fixed_point(Expr::sum(std::move(terms)));
        if (!value) {
            return make_error<ExprVector>(value.error());
        }
        grad.push_back(std::move(*value));
    }
    return grad;
}

auto covariant_divergence(const Metric& m, const ExprVector& field) -> Result<Expr> {
    if (!dimensions_ok(m)) {
        return make_error<Expr>(MathError::domain_error);
    }
    const std::size_t n = m.dim();
    if (field.size() != n) {
        return make_error<Expr>(MathError::domain_error);
    }
    auto det = metric_determinant(m);
    if (!det) {
        return det;
    }
    if (is_exact_zero(*det)) {
        return make_error<Expr>(MathError::domain_error);  // sqrt(g) / (1/sqrt(g)) undefined
    }
    const Expr sqrt_g = Expr::power(*det, one_half());
    const Expr inv_sqrt_g = Expr::power(*det, neg_one_half());

    // div v = (1/sqrt(g)) sum_i d_i(sqrt(g) v^i).
    std::vector<Expr> derivative_terms;
    derivative_terms.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto inner = simp_to_fixed_point(Expr::product({sqrt_g, field[i]}));
        if (!inner) {
            return inner;
        }
        auto d = partial(*inner, m.coords[i]);
        if (!d) {
            return d;
        }
        derivative_terms.push_back(std::move(*d));
    }
    return distribute_factor(inv_sqrt_g, Expr::sum(std::move(derivative_terms)));
}

auto laplace_beltrami(const Metric& m, const Expr& f) -> Result<Expr> {
    // The Laplace-Beltrami operator is the covariant divergence of the raised-index gradient.
    auto grad = metric_gradient(m, f);
    if (!grad) {
        return make_error<Expr>(grad.error());
    }
    return covariant_divergence(m, *grad);
}

}  // namespace nimblecas
