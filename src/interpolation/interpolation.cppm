// NimbleCAS exact polynomial interpolation over the rationals (ROADMAP 7).
// @author Olumuyiwa Oluwasanmi
//
// Given distinct nodes x_0,...,x_{n-1} in Q and values y_0,...,y_{n-1} in Q, there is
// a UNIQUE polynomial of degree <= n-1 through the points (x_i, y_i). This module
// builds that polynomial (and evaluates it) EXACTLY over Q — no floating point — via
// four classical, mathematically-equivalent constructions:
//
//   * Lagrange   — the basis L_i(x) = Π_{j!=i} (x - x_j)/(x_i - x_j), and the sum
//                  Σ y_i L_i(x). Direct, O(n^2) basis products.
//   * Newton     — divided differences and the nested Newton form, with cheap
//                  incremental addition of a new point.
//   * Barycentric— the barycentric weights w_i and the stable evaluation form
//                  (here computed exactly rather than for numerical stability).
//   * Neville    — evaluate the interpolant at a point without forming the polynomial.
//
// Because the same unique polynomial underlies all four, they agree exactly:
// lagrange_polynomial == newton_polynomial, and barycentric/Neville evaluations match
// the polynomial's value at every point.
//
// Hermite interpolation additionally matches first derivatives: given (x_i, y_i, y'_i)
// it returns the unique degree <= 2n-1 polynomial with p(x_i)=y_i and p'(x_i)=y'_i,
// via confluent (repeated-node) divided differences.
//
// Every operation is overflow-checked: an int64 boundary inside any Rational step
// propagates as MathError::overflow rather than wrapping (Rule 32). A size mismatch,
// an empty input, or a duplicated node is rejected with MathError::domain_error.
//
// HONESTY BOUNDARY. For rational data this interpolant is the EXACT unique polynomial —
// there is no approximation error to report. The Runge phenomenon and ill-conditioning
// are properties of the CHOICE OF NODES (equispaced high-degree nodes make the true
// interpolant oscillate wildly between them), NOT numerical errors introduced here: the
// coefficients returned are the mathematically exact ones. No floats are used anywhere
// on this path.

export module nimblecas.interpolation;

import std;
import nimblecas.core;
import nimblecas.ratpoly;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// Evaluation of an exact rational polynomial (Horner over Q).
// ---------------------------------------------------------------------------
// p(x) evaluated exactly at a rational point. Handy for checking that an
// interpolant passes through its nodes; fails only on overflow.
[[nodiscard]] auto poly_evaluate(const RationalPoly& p, const Rational& x) -> Result<Rational>;

// ---------------------------------------------------------------------------
// Lagrange interpolation.
// ---------------------------------------------------------------------------
// The i-th Lagrange basis polynomial L_i(x) = Π_{j!=i} (x - x_j)/(x_i - x_j): the
// unique degree <= n-1 polynomial with L_i(x_i) = 1 and L_i(x_j) = 0 for j != i.
// Fails with domain_error on an empty node set, an out-of-range index, or duplicate
// nodes (which would divide by zero).
[[nodiscard]] auto lagrange_basis(std::span<const Rational> nodes, std::size_t i)
    -> Result<RationalPoly>;

// All n Lagrange basis polynomials L_0,...,L_{n-1} for the given nodes.
[[nodiscard]] auto lagrange_basis_polynomials(std::span<const Rational> nodes)
    -> Result<std::vector<RationalPoly>>;

// The Lagrange interpolant Σ_i y_i L_i(x): the unique degree <= n-1 polynomial through
// the points. Fails with domain_error on a size mismatch, empty input, or duplicate
// nodes.
[[nodiscard]] auto lagrange_polynomial(std::span<const Rational> nodes,
                                       std::span<const Rational> values)
    -> Result<RationalPoly>;

// ---------------------------------------------------------------------------
// Newton divided differences.
// ---------------------------------------------------------------------------
// The Newton coefficients c_k = f[x_0,...,x_k] (the top diagonal of the divided-
// difference table), so the interpolant is c_0 + c_1(x-x_0) + c_2(x-x_0)(x-x_1) + ...
[[nodiscard]] auto divided_differences(std::span<const Rational> nodes,
                                       std::span<const Rational> values)
    -> Result<std::vector<Rational>>;

// The interpolant in Newton form (identical polynomial to lagrange_polynomial).
[[nodiscard]] auto newton_polynomial(std::span<const Rational> nodes,
                                     std::span<const Rational> values)
    -> Result<RationalPoly>;

// A Newton interpolant that supports cheap incremental point addition: adding one node
// costs O(current size) rather than rebuilding the whole table. Immutable/fluent —
// with_point returns a new interpolant rather than mutating in place (railway style).
class NewtonInterpolant {
public:
    NewtonInterpolant() = default;  // the empty interpolant (zero polynomial)

    // Build directly from a full point set (domain_error on mismatch / empty / dup).
    [[nodiscard]] static auto from_points(std::span<const Rational> nodes,
                                          std::span<const Rational> values)
        -> Result<NewtonInterpolant>;

    // Return a copy extended by one more point. The new Newton coefficient is computed
    // from the existing coefficients in O(size) exact rational steps. Fails with
    // domain_error if x duplicates an existing node, or overflow on an int64 boundary.
    [[nodiscard]] auto with_point(const Rational& x, const Rational& y) const
        -> Result<NewtonInterpolant>;

    [[nodiscard]] auto is_empty() const noexcept -> bool { return nodes_.empty(); }
    [[nodiscard]] auto size() const noexcept -> std::size_t { return nodes_.size(); }
    [[nodiscard]] auto nodes() const noexcept -> std::span<const Rational> { return nodes_; }
    // Newton divided-difference coefficients c_k = f[x_0,...,x_k].
    [[nodiscard]] auto coefficients() const noexcept -> std::span<const Rational> {
        return coeffs_;
    }

    // The interpolant as an explicit polynomial in Q[x].
    [[nodiscard]] auto polynomial() const -> Result<RationalPoly>;
    // Exact evaluation at a rational point via nested Newton (Horner) form.
    [[nodiscard]] auto evaluate(const Rational& x) const -> Result<Rational>;

private:
    std::vector<Rational> nodes_;   // x_0,...,x_{k-1}, all distinct
    std::vector<Rational> coeffs_;  // c_i = f[x_0,...,x_i]
};

// ---------------------------------------------------------------------------
// Barycentric form.
// ---------------------------------------------------------------------------
// The barycentric weights w_i = 1 / Π_{j!=i} (x_i - x_j).
[[nodiscard]] auto barycentric_weights(std::span<const Rational> nodes)
    -> Result<std::vector<Rational>>;

// Exact barycentric interpolant: precomputes the weights, then evaluates
// p(x) = [Σ_i w_i/(x-x_i) · y_i] / [Σ_i w_i/(x-x_i)] exactly, with the value returned
// directly when x coincides with a node.
class BarycentricInterpolant {
public:
    [[nodiscard]] static auto make(std::span<const Rational> nodes,
                                   std::span<const Rational> values)
        -> Result<BarycentricInterpolant>;

    [[nodiscard]] auto nodes() const noexcept -> std::span<const Rational> { return nodes_; }
    [[nodiscard]] auto values() const noexcept -> std::span<const Rational> { return values_; }
    [[nodiscard]] auto weights() const noexcept -> std::span<const Rational> { return weights_; }

    [[nodiscard]] auto evaluate(const Rational& x) const -> Result<Rational>;

private:
    BarycentricInterpolant() = default;
    std::vector<Rational> nodes_;
    std::vector<Rational> values_;
    std::vector<Rational> weights_;
};

// ---------------------------------------------------------------------------
// Neville's algorithm.
// ---------------------------------------------------------------------------
// Evaluate the interpolant at x by recombining the tableau of lower-order interpolants
// — the polynomial is never formed. Exact over Q; agrees with every other method.
[[nodiscard]] auto neville_evaluate(std::span<const Rational> nodes,
                                    std::span<const Rational> values, const Rational& x)
    -> Result<Rational>;

// ---------------------------------------------------------------------------
// Hermite interpolation (value + first derivative).
// ---------------------------------------------------------------------------
// The unique degree <= 2n-1 polynomial with p(x_i) = values[i] and p'(x_i) =
// derivatives[i] for every node, built from confluent divided differences (each node
// doubled). Fails with domain_error on a size mismatch, empty input, or duplicate node.
[[nodiscard]] auto hermite_polynomial(std::span<const Rational> nodes,
                                      std::span<const Rational> values,
                                      std::span<const Rational> derivatives)
    -> Result<RationalPoly>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

[[nodiscard]] auto rat_one() -> Rational { return Rational::from_int(1); }

// The linear factor (x - root) as a RationalPoly. Fails (overflow) only when negating
// an INT64_MIN numerator.
[[nodiscard]] auto linear_factor(const Rational& root) -> Result<RationalPoly> {
    auto neg = root.negate();
    if (!neg) {
        return make_error<RationalPoly>(neg.error());
    }
    return RationalPoly::from_coeffs({*neg, rat_one()});
}

// All nodes pairwise distinct? Duplicates make interpolation ill-posed (division by
// zero), so they are reported as domain_error.
[[nodiscard]] auto nodes_distinct(std::span<const Rational> nodes) -> bool {
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        for (std::size_t j = i + 1; j < nodes.size(); ++j) {
            if (nodes[i] == nodes[j]) {
                return false;
            }
        }
    }
    return true;
}

// Shared precondition for the (nodes, values) interpolators: equal, non-empty, distinct.
[[nodiscard]] auto validate_points(std::span<const Rational> nodes,
                                   std::span<const Rational> values) -> std::optional<MathError> {
    if (nodes.size() != values.size() || nodes.empty()) {
        return MathError::domain_error;
    }
    if (!nodes_distinct(nodes)) {
        return MathError::domain_error;
    }
    return std::nullopt;
}

}  // namespace

// --- Polynomial evaluation --------------------------------------------------

auto poly_evaluate(const RationalPoly& p, const Rational& x) -> Result<Rational> {
    const std::span<const Rational> c = p.coefficients();
    Rational acc{};  // 0/1
    for (std::size_t k = c.size(); k-- > 0;) {  // high degree -> low (Horner)
        auto m = acc.multiply(x);
        if (!m) {
            return make_error<Rational>(m.error());
        }
        auto a = m->add(c[k]);
        if (!a) {
            return make_error<Rational>(a.error());
        }
        acc = *a;
    }
    return acc;
}

// --- Lagrange ---------------------------------------------------------------

auto lagrange_basis(std::span<const Rational> nodes, std::size_t i) -> Result<RationalPoly> {
    if (nodes.empty() || i >= nodes.size() || !nodes_distinct(nodes)) {
        return make_error<RationalPoly>(MathError::domain_error);
    }
    RationalPoly numerator = RationalPoly::constant(rat_one());  // Π_{j!=i} (x - x_j)
    Rational denominator = rat_one();                            // Π_{j!=i} (x_i - x_j)
    for (std::size_t j = 0; j < nodes.size(); ++j) {
        if (j == i) {
            continue;
        }
        auto factor = linear_factor(nodes[j]);
        if (!factor) {
            return make_error<RationalPoly>(factor.error());
        }
        auto num = numerator.multiply(*factor);
        if (!num) {
            return make_error<RationalPoly>(num.error());
        }
        numerator = *num;

        auto diff = nodes[i].subtract(nodes[j]);  // x_i - x_j, nonzero (distinct)
        if (!diff) {
            return make_error<RationalPoly>(diff.error());
        }
        auto den = denominator.multiply(*diff);
        if (!den) {
            return make_error<RationalPoly>(den.error());
        }
        denominator = *den;
    }
    auto inv = rat_one().divide(denominator);  // denominator != 0 here
    if (!inv) {
        return make_error<RationalPoly>(inv.error());
    }
    return numerator.scale(*inv);
}

auto lagrange_basis_polynomials(std::span<const Rational> nodes)
    -> Result<std::vector<RationalPoly>> {
    if (nodes.empty() || !nodes_distinct(nodes)) {
        return make_error<std::vector<RationalPoly>>(MathError::domain_error);
    }
    std::vector<RationalPoly> basis;
    basis.reserve(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        auto li = lagrange_basis(nodes, i);
        if (!li) {
            return make_error<std::vector<RationalPoly>>(li.error());
        }
        basis.push_back(std::move(*li));
    }
    return basis;
}

auto lagrange_polynomial(std::span<const Rational> nodes, std::span<const Rational> values)
    -> Result<RationalPoly> {
    if (auto err = validate_points(nodes, values)) {
        return make_error<RationalPoly>(*err);
    }
    RationalPoly p{};  // zero
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        auto li = lagrange_basis(nodes, i);
        if (!li) {
            return make_error<RationalPoly>(li.error());
        }
        auto term = li->scale(values[i]);
        if (!term) {
            return make_error<RationalPoly>(term.error());
        }
        auto sum = p.add(*term);
        if (!sum) {
            return make_error<RationalPoly>(sum.error());
        }
        p = *sum;
    }
    return p;
}

// --- Newton -----------------------------------------------------------------

auto divided_differences(std::span<const Rational> nodes, std::span<const Rational> values)
    -> Result<std::vector<Rational>> {
    if (auto err = validate_points(nodes, values)) {
        return make_error<std::vector<Rational>>(*err);
    }
    // In-place table: c[i] starts as y_i and becomes f[x_{i-j},...,x_i] after column j.
    std::vector<Rational> c(values.begin(), values.end());
    const std::size_t n = c.size();
    for (std::size_t j = 1; j < n; ++j) {
        for (std::size_t i = n; i-- > j;) {  // i = n-1 .. j (descending, in place)
            auto num = c[i].subtract(c[i - 1]);
            if (!num) {
                return make_error<std::vector<Rational>>(num.error());
            }
            auto den = nodes[i].subtract(nodes[i - j]);  // distinct -> nonzero
            if (!den) {
                return make_error<std::vector<Rational>>(den.error());
            }
            auto q = num->divide(*den);
            if (!q) {
                return make_error<std::vector<Rational>>(q.error());
            }
            c[i] = *q;
        }
    }
    return c;  // c[k] = f[x_0,...,x_k], the Newton coefficients
}

namespace {

// Assemble a polynomial from Newton coefficients c_k over the given nodes:
// Σ_k c_k · Π_{m<k} (x - nodes[m]).
[[nodiscard]] auto newton_form(std::span<const Rational> nodes, std::span<const Rational> coeffs)
    -> Result<RationalPoly> {
    RationalPoly p{};                                        // zero
    RationalPoly basis = RationalPoly::constant(rat_one());  // Π_{m<k} (x - x_m)
    for (std::size_t k = 0; k < coeffs.size(); ++k) {
        auto term = basis.scale(coeffs[k]);
        if (!term) {
            return make_error<RationalPoly>(term.error());
        }
        auto sum = p.add(*term);
        if (!sum) {
            return make_error<RationalPoly>(sum.error());
        }
        p = *sum;
        if (k + 1 < coeffs.size()) {
            auto factor = linear_factor(nodes[k]);
            if (!factor) {
                return make_error<RationalPoly>(factor.error());
            }
            auto nb = basis.multiply(*factor);
            if (!nb) {
                return make_error<RationalPoly>(nb.error());
            }
            basis = *nb;
        }
    }
    return p;
}

}  // namespace

auto newton_polynomial(std::span<const Rational> nodes, std::span<const Rational> values)
    -> Result<RationalPoly> {
    auto coeffs = divided_differences(nodes, values);  // validates inputs
    if (!coeffs) {
        return make_error<RationalPoly>(coeffs.error());
    }
    return newton_form(nodes, *coeffs);
}

auto NewtonInterpolant::from_points(std::span<const Rational> nodes,
                                    std::span<const Rational> values)
    -> Result<NewtonInterpolant> {
    auto coeffs = divided_differences(nodes, values);  // validates inputs
    if (!coeffs) {
        return make_error<NewtonInterpolant>(coeffs.error());
    }
    NewtonInterpolant r;
    r.nodes_.assign(nodes.begin(), nodes.end());
    r.coeffs_ = std::move(*coeffs);
    return r;
}

auto NewtonInterpolant::with_point(const Rational& x, const Rational& y) const
    -> Result<NewtonInterpolant> {
    // A repeated node would divide by zero in the update below and is ill-posed anyway.
    for (const Rational& node : nodes_) {
        if (node == x) {
            return make_error<NewtonInterpolant>(MathError::domain_error);
        }
    }
    // Extend the divided-difference diagonal: the new coefficient is obtained by folding
    // the existing coefficients c_k in via (bottom - c_k) / (x - x_k).
    Rational bottom = y;
    for (std::size_t k = 0; k < nodes_.size(); ++k) {
        auto sub = bottom.subtract(coeffs_[k]);
        if (!sub) {
            return make_error<NewtonInterpolant>(sub.error());
        }
        auto den = x.subtract(nodes_[k]);  // nonzero: x differs from every node
        if (!den) {
            return make_error<NewtonInterpolant>(den.error());
        }
        auto div = sub->divide(*den);
        if (!div) {
            return make_error<NewtonInterpolant>(div.error());
        }
        bottom = *div;
    }
    NewtonInterpolant r = *this;
    r.nodes_.push_back(x);
    r.coeffs_.push_back(bottom);
    return r;
}

auto NewtonInterpolant::polynomial() const -> Result<RationalPoly> {
    return newton_form(nodes_, coeffs_);
}

auto NewtonInterpolant::evaluate(const Rational& x) const -> Result<Rational> {
    if (coeffs_.empty()) {
        return Rational{};  // empty interpolant is the zero polynomial
    }
    // Nested Newton (Horner): acc = c_{n-1}; acc = c_k + (x - x_k)·acc for k = n-2 .. 0.
    Rational acc = coeffs_.back();
    for (std::size_t k = coeffs_.size() - 1; k-- > 0;) {
        auto d = x.subtract(nodes_[k]);
        if (!d) {
            return make_error<Rational>(d.error());
        }
        auto m = acc.multiply(*d);
        if (!m) {
            return make_error<Rational>(m.error());
        }
        auto a = m->add(coeffs_[k]);
        if (!a) {
            return make_error<Rational>(a.error());
        }
        acc = *a;
    }
    return acc;
}

// --- Barycentric ------------------------------------------------------------

auto barycentric_weights(std::span<const Rational> nodes) -> Result<std::vector<Rational>> {
    if (nodes.empty() || !nodes_distinct(nodes)) {
        return make_error<std::vector<Rational>>(MathError::domain_error);
    }
    std::vector<Rational> w(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        Rational prod = rat_one();  // Π_{j!=i} (x_i - x_j)
        for (std::size_t j = 0; j < nodes.size(); ++j) {
            if (j == i) {
                continue;
            }
            auto diff = nodes[i].subtract(nodes[j]);  // nonzero (distinct)
            if (!diff) {
                return make_error<std::vector<Rational>>(diff.error());
            }
            auto p = prod.multiply(*diff);
            if (!p) {
                return make_error<std::vector<Rational>>(p.error());
            }
            prod = *p;
        }
        auto inv = rat_one().divide(prod);  // prod != 0
        if (!inv) {
            return make_error<std::vector<Rational>>(inv.error());
        }
        w[i] = *inv;
    }
    return w;
}

auto BarycentricInterpolant::make(std::span<const Rational> nodes,
                                  std::span<const Rational> values)
    -> Result<BarycentricInterpolant> {
    if (auto err = validate_points(nodes, values)) {
        return make_error<BarycentricInterpolant>(*err);
    }
    auto w = barycentric_weights(nodes);
    if (!w) {
        return make_error<BarycentricInterpolant>(w.error());
    }
    BarycentricInterpolant r;
    r.nodes_.assign(nodes.begin(), nodes.end());
    r.values_.assign(values.begin(), values.end());
    r.weights_ = std::move(*w);
    return r;
}

auto BarycentricInterpolant::evaluate(const Rational& x) const -> Result<Rational> {
    // At a node the barycentric quotient is 0/0; the interpolated value is simply y_i.
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i] == x) {
            return values_[i];
        }
    }
    Rational numerator{};    // Σ_i w_i/(x - x_i) · y_i
    Rational denominator{};  // Σ_i w_i/(x - x_i)
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        auto diff = x.subtract(nodes_[i]);  // nonzero: x is not a node here
        if (!diff) {
            return make_error<Rational>(diff.error());
        }
        auto term = weights_[i].divide(*diff);  // w_i / (x - x_i)
        if (!term) {
            return make_error<Rational>(term.error());
        }
        auto dsum = denominator.add(*term);
        if (!dsum) {
            return make_error<Rational>(dsum.error());
        }
        denominator = *dsum;

        auto wy = term->multiply(values_[i]);
        if (!wy) {
            return make_error<Rational>(wy.error());
        }
        auto nsum = numerator.add(*wy);
        if (!nsum) {
            return make_error<Rational>(nsum.error());
        }
        numerator = *nsum;
    }
    return numerator.divide(denominator);  // denominator != 0 for a valid node set
}

// --- Neville ----------------------------------------------------------------

auto neville_evaluate(std::span<const Rational> nodes, std::span<const Rational> values,
                      const Rational& x) -> Result<Rational> {
    if (auto err = validate_points(nodes, values)) {
        return make_error<Rational>(*err);
    }
    // p[i] holds the value at x of the interpolant through a growing window of nodes.
    std::vector<Rational> p(values.begin(), values.end());
    const std::size_t n = p.size();
    for (std::size_t j = 1; j < n; ++j) {
        for (std::size_t i = n; i-- > j;) {  // i = n-1 .. j
            // p[i] <- ((x - x_{i-j})·p[i] - (x - x_i)·p[i-1]) / (x_i - x_{i-j}).
            auto a = x.subtract(nodes[i - j]);
            auto b = x.subtract(nodes[i]);
            if (!a || !b) {
                return make_error<Rational>(a ? b.error() : a.error());
            }
            auto t1 = a->multiply(p[i]);
            auto t2 = b->multiply(p[i - 1]);
            if (!t1 || !t2) {
                return make_error<Rational>(t1 ? t2.error() : t1.error());
            }
            auto numer = t1->subtract(*t2);
            if (!numer) {
                return make_error<Rational>(numer.error());
            }
            auto den = nodes[i].subtract(nodes[i - j]);  // distinct -> nonzero
            if (!den) {
                return make_error<Rational>(den.error());
            }
            auto q = numer->divide(*den);
            if (!q) {
                return make_error<Rational>(q.error());
            }
            p[i] = *q;
        }
    }
    return p[n - 1];
}

// --- Hermite ----------------------------------------------------------------

auto hermite_polynomial(std::span<const Rational> nodes, std::span<const Rational> values,
                        std::span<const Rational> derivatives) -> Result<RationalPoly> {
    if (nodes.size() != values.size() || nodes.size() != derivatives.size() || nodes.empty() ||
        !nodes_distinct(nodes)) {
        return make_error<RationalPoly>(MathError::domain_error);
    }
    const std::size_t n = nodes.size();
    const std::size_t m = 2 * n;  // each node contributes two conditions (value + slope)

    // Confluent divided differences on the doubled node sequence z (Burden & Faires):
    // z[2i] = z[2i+1] = x_i. The Newton coefficients are the table's diagonal Q[i][i].
    std::vector<Rational> z(m);
    std::vector<std::vector<Rational>> q(m, std::vector<Rational>(m));
    for (std::size_t i = 0; i < n; ++i) {
        z[2 * i] = nodes[i];
        z[2 * i + 1] = nodes[i];
        q[2 * i][0] = values[i];
        q[2 * i + 1][0] = values[i];
    }
    for (std::size_t i = 1; i < m; ++i) {
        for (std::size_t j = 1; j <= i; ++j) {
            if (j == 1 && z[i] == z[i - 1]) {
                // Two copies of the same node: the first divided difference is f'(x).
                q[i][1] = derivatives[i / 2];
                continue;
            }
            // z[i] - z[i-j] is nonzero here: equal adjacent copies are handled above,
            // and for j >= 2 the endpoints belong to distinct original nodes.
            auto num = q[i][j - 1].subtract(q[i - 1][j - 1]);
            if (!num) {
                return make_error<RationalPoly>(num.error());
            }
            auto den = z[i].subtract(z[i - j]);
            if (!den) {
                return make_error<RationalPoly>(den.error());
            }
            auto d = num->divide(*den);
            if (!d) {
                return make_error<RationalPoly>(d.error());
            }
            q[i][j] = *d;
        }
    }
    // Assemble Σ_i Q[i][i] · Π_{k<i} (x - z[k]).
    std::vector<Rational> diag(m);
    for (std::size_t i = 0; i < m; ++i) {
        diag[i] = q[i][i];
    }
    return newton_form(z, diag);
}

}  // namespace nimblecas
