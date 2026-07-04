// NimbleCAS exact splines & Bezier / B-spline / NURBS curves over the rationals (ROADMAP 7).
// @author Olumuyiwa Oluwasanmi
//
// Piecewise-polynomial geometry done EXACTLY over the field Q. For rational knots,
// values, control points, weights and a rational parameter, every construction below is
// exact — there is no floating point and no tolerance anywhere in this module. Failure is
// reported only on the railway (Result<T> / MathError); nothing throws.
//
// Four families:
//
//   * CUBIC SPLINE interpolation. The C^2-continuity conditions on the second derivatives
//     (the "moments" M_i = S''(x_i)) form a TRIDIAGONAL linear system. It is assembled and
//     solved EXACTLY over Q by the Thomas algorithm in nimblecas.bandsolve
//     (solve_tridiagonal). Three boundary conditions are supported:
//       - NATURAL  : S''=0 at both ends (M_0 = M_n = 0), interior moments solved.
//       - CLAMPED  : given end first-derivatives S'(x_0), S'(x_n); all moments solved.
//       - PERIODIC : S, S', S'' wrap around (requires y_0 == y_n). The wrap makes the
//                    system CYCLIC-tridiagonal; it is reduced to two ordinary tridiagonal
//                    Thomas solves by the exact Sherman-Morrison correction — still exact
//                    over Q.
//     The result is a piecewise cubic: one RationalPoly per interval (in absolute x) plus
//     the moment array. evaluate(x) locates the interval and evaluates the exact cubic.
//
//   * HermiteSpline — a piecewise cubic Hermite (C^1) curve. from_slopes(x,y,d) is the
//     EXACT interpolant matching value y_i and slope d_i at each knot. pchip() derives the
//     slopes by the Fritsch-Carlson shape-preserving (monotonicity-tending) rule. HONESTY:
//     every PCHIP arithmetic step is exact over Q, but the SLOPE CHOICE itself is a
//     heuristic (a shape decision), not the unique slope forced by an interpolation
//     condition — documented here and reproducible bit-for-bit over Q.
//
//   * BEZIER curves in the Bernstein basis over rational control points. De Casteljau
//     evaluation (repeated convex combinations) is exact at a rational t in [0,1]; degree
//     elevation, de Casteljau subdivision (split at t), and the exact Bezier<->power-basis
//     (RationalPoly) conversions are provided. Scalar (1-D) and 2-D (Point2) control data.
//
//   * B-SPLINES via the Cox-de Boor recursion for the basis functions N_{i,p}(u), exact
//     over Q; curve evaluation C(u) = sum_i N_{i,p}(u) P_i. Open/clamped knot vectors are
//     supported (the endpoint special case makes a clamped curve interpolate its end
//     control points and gives partition of unity sum_i N_{i,p}(u) == 1 exactly).
//
//   * NURBS — rational B-splines with rational weights w_i:
//     C(u) = (sum_i w_i N_{i,p}(u) P_i) / (sum_i w_i N_{i,p}(u)), evaluated exactly over Q.
//     HONESTY: the exact-rational core covers the RATIONAL-WEIGHT case. Some classical
//     shapes need irrational data (an exact circle needs weight cos(theta) at a corner),
//     which lies outside the rational core and is deliberately not represented here.
//
// HONESTY BOUNDARY. Everything above is EXACT over Q for rational data: evaluation at a
// rational parameter is the exact rational value, with no numerical error to report.
// Conditioning and shape (Runge-like oscillation of a high-degree interpolant, the shape of
// a spline for badly spaced knots) are properties of the DATA/KNOTS, not errors introduced
// here. Curves requiring irrational data are outside this rational core. Degenerate inputs —
// unsorted/duplicate knots where strict monotonicity is required, mismatched sizes, a
// parameter t outside [0,1] or u outside the knot domain, too few points, or a non-periodic
// (y_0 != y_n) periodic request — are rejected with MathError::domain_error. Any int64
// boundary inside a Rational step propagates as MathError::overflow (Rule 32).

export module nimblecas.splines;

import std;
import nimblecas.core;
import nimblecas.ratpoly;
import nimblecas.bandsolve;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// A 2-D point with exact rational coordinates (for 2-D Bezier control data).
// ---------------------------------------------------------------------------
struct Point2 {
    Rational x{};
    Rational y{};
    [[nodiscard]] auto operator==(const Point2& o) const noexcept -> bool {
        return x == o.x && y == o.y;
    }
};

// ---------------------------------------------------------------------------
// Cubic spline interpolation (moment / tridiagonal method), exact over Q.
// ---------------------------------------------------------------------------

// Which boundary condition a CubicSpline was built with.
enum class SplineBoundary : std::uint8_t { natural, clamped, periodic };

// A C^2 interpolating cubic spline through knots (x_i, y_i). Holds the knots, values, the
// moment array M_i = S''(x_i), and one cubic RationalPoly per interval (expressed in the
// absolute variable x). evaluate(x) is exact for a rational x in [x_0, x_n].
class CubicSpline {
public:
    // NATURAL spline: S''(x_0) = S''(x_n) = 0. Requires >= 2 strictly increasing knots and a
    // matching value count, else domain_error.
    [[nodiscard]] static auto natural(std::span<const Rational> xs, std::span<const Rational> ys)
        -> Result<CubicSpline>;

    // CLAMPED spline: prescribed end slopes S'(x_0) = deriv_left, S'(x_n) = deriv_right.
    // Requires >= 2 strictly increasing knots and a matching value count, else domain_error.
    [[nodiscard]] static auto clamped(std::span<const Rational> xs, std::span<const Rational> ys,
                                      const Rational& deriv_left, const Rational& deriv_right)
        -> Result<CubicSpline>;

    // PERIODIC spline: S, S', S'' agree at the two ends. Requires >= 3 strictly increasing
    // knots, a matching value count, and y_0 == y_n (else domain_error).
    [[nodiscard]] static auto periodic(std::span<const Rational> xs, std::span<const Rational> ys)
        -> Result<CubicSpline>;

    [[nodiscard]] auto knots() const noexcept -> std::span<const Rational> { return xs_; }
    [[nodiscard]] auto values() const noexcept -> std::span<const Rational> { return ys_; }
    // The moment array M_i = S''(x_i) (n+1 entries).
    [[nodiscard]] auto moments() const noexcept -> std::span<const Rational> { return moments_; }
    [[nodiscard]] auto boundary() const noexcept -> SplineBoundary { return boundary_; }
    [[nodiscard]] auto piece_count() const noexcept -> std::size_t { return pieces_.size(); }
    // The cubic on interval [x_i, x_{i+1}] as a RationalPoly in absolute x (i < piece_count()).
    [[nodiscard]] auto piece(std::size_t i) const -> const RationalPoly& { return pieces_[i]; }
    [[nodiscard]] auto pieces() const noexcept -> std::span<const RationalPoly> { return pieces_; }

    // Exact evaluation at a rational x in [x_0, x_n]; x outside that range is domain_error.
    [[nodiscard]] auto evaluate(const Rational& x) const -> Result<Rational>;

private:
    CubicSpline() = default;
    std::vector<Rational> xs_;
    std::vector<Rational> ys_;
    std::vector<Rational> moments_;
    std::vector<RationalPoly> pieces_;
    SplineBoundary boundary_{SplineBoundary::natural};
};

// ---------------------------------------------------------------------------
// Piecewise cubic Hermite spline (C^1) — exact given slopes; PCHIP shape slopes.
// ---------------------------------------------------------------------------
class HermiteSpline {
public:
    // EXACT interpolant matching value ys[i] and first-derivative slopes[i] at every knot.
    // Requires >= 2 strictly increasing knots and equal-length ys / slopes, else domain_error.
    [[nodiscard]] static auto from_slopes(std::span<const Rational> xs,
                                          std::span<const Rational> ys,
                                          std::span<const Rational> slopes)
        -> Result<HermiteSpline>;

    // PCHIP: derive shape-preserving (monotonicity-tending) slopes by the Fritsch-Carlson
    // rule, then build the exact Hermite interpolant. The arithmetic is exact over Q; the
    // slope CHOICE is a documented heuristic, not a unique interpolation condition.
    [[nodiscard]] static auto pchip(std::span<const Rational> xs, std::span<const Rational> ys)
        -> Result<HermiteSpline>;

    [[nodiscard]] auto knots() const noexcept -> std::span<const Rational> { return xs_; }
    [[nodiscard]] auto values() const noexcept -> std::span<const Rational> { return ys_; }
    [[nodiscard]] auto slopes() const noexcept -> std::span<const Rational> { return slopes_; }
    [[nodiscard]] auto piece_count() const noexcept -> std::size_t { return pieces_.size(); }
    [[nodiscard]] auto piece(std::size_t i) const -> const RationalPoly& { return pieces_[i]; }
    [[nodiscard]] auto pieces() const noexcept -> std::span<const RationalPoly> { return pieces_; }

    [[nodiscard]] auto evaluate(const Rational& x) const -> Result<Rational>;

private:
    HermiteSpline() = default;
    std::vector<Rational> xs_;
    std::vector<Rational> ys_;
    std::vector<Rational> slopes_;
    std::vector<RationalPoly> pieces_;
};

// ---------------------------------------------------------------------------
// Bezier curves (scalar / 1-D control values), Bernstein basis over Q.
// ---------------------------------------------------------------------------
class BezierCurve {
public:
    // Build from control points b_0..b_n (n = degree). At least one control point is
    // required (else domain_error).
    [[nodiscard]] static auto make(std::vector<Rational> control_points) -> Result<BezierCurve>;

    [[nodiscard]] auto degree() const noexcept -> std::size_t { return control_.size() - 1; }
    [[nodiscard]] auto control_points() const noexcept -> std::span<const Rational> {
        return control_;
    }

    // De Casteljau evaluation at rational t in [0,1] (t outside -> domain_error). Exact.
    [[nodiscard]] auto evaluate(const Rational& t) const -> Result<Rational>;

    // Bernstein-basis evaluation sum_i C(n,i) t^i (1-t)^{n-i} b_i at t in [0,1]. Equal to
    // evaluate() mathematically; provided so the two forms can be cross-checked exactly.
    [[nodiscard]] auto evaluate_bernstein(const Rational& t) const -> Result<Rational>;

    // Degree elevation: an equal-shape curve of degree+1.
    [[nodiscard]] auto elevate() const -> Result<BezierCurve>;

    // The power-basis polynomial B(t) = sum_j c_j t^j (a RationalPoly in t). Exact.
    [[nodiscard]] auto to_power_basis() const -> Result<RationalPoly>;

    // The degree-`degree` Bezier whose power-basis polynomial is `p` (deg(p) <= degree).
    // Inverse of to_power_basis(); fails with domain_error if deg(p) > degree.
    [[nodiscard]] static auto from_power_basis(const RationalPoly& p, std::size_t degree)
        -> Result<BezierCurve>;

private:
    BezierCurve() = default;
    explicit BezierCurve(std::vector<Rational> c) : control_(std::move(c)) {}
    std::vector<Rational> control_;  // b_0..b_n, always non-empty
};

// De Casteljau split of a Bezier at t: the left [0,t] and right [t,1] sub-curves, each a
// Bezier of the same degree; together they reproduce the original.
struct BezierSplit {
    BezierCurve left;
    BezierCurve right;
};

// Subdivide `curve` at t in [0,1] into its left and right halves (exact). t out of range
// -> domain_error.
[[nodiscard]] auto bezier_subdivide(const BezierCurve& curve, const Rational& t)
    -> Result<BezierSplit>;

// ---------------------------------------------------------------------------
// Bezier curves with 2-D control points (pairs of rationals).
// ---------------------------------------------------------------------------
// Every Bezier operation is affine in the control coordinates, so a 2-D curve is exactly a
// pair of independent scalar curves (one per coordinate). This class is that pair.
class BezierCurve2 {
public:
    [[nodiscard]] static auto make(std::vector<Point2> control_points) -> Result<BezierCurve2>;

    [[nodiscard]] auto degree() const noexcept -> std::size_t { return x_.degree(); }
    [[nodiscard]] auto control_points() const -> std::vector<Point2>;

    [[nodiscard]] auto evaluate(const Rational& t) const -> Result<Point2>;
    [[nodiscard]] auto elevate() const -> Result<BezierCurve2>;

private:
    BezierCurve2(BezierCurve x, BezierCurve y) : x_(std::move(x)), y_(std::move(y)) {}
    BezierCurve x_;
    BezierCurve y_;
};

// ---------------------------------------------------------------------------
// B-splines (Cox-de Boor), exact over Q.
// ---------------------------------------------------------------------------
class BSpline {
public:
    // Build from a non-decreasing knot vector U, control points P, and degree p. Requires
    // p >= 1, at least one control point, and the size relation U.size() == P.size() + p + 1.
    // A strictly-decreasing step in U (unsorted), a size-relation violation, or an empty
    // control set is domain_error.
    [[nodiscard]] static auto make(std::vector<Rational> knots,
                                   std::vector<Rational> control_points, std::size_t degree)
        -> Result<BSpline>;

    [[nodiscard]] auto degree() const noexcept -> std::size_t { return degree_; }
    [[nodiscard]] auto knots() const noexcept -> std::span<const Rational> { return knots_; }
    [[nodiscard]] auto control_points() const noexcept -> std::span<const Rational> {
        return control_;
    }
    // The valid parameter domain [u_min, u_max] = [U_p, U_{m-p}].
    [[nodiscard]] auto domain_min() const -> Rational { return knots_[degree_]; }
    [[nodiscard]] auto domain_max() const -> Rational {
        return knots_[knots_.size() - 1 - degree_];
    }

    // The basis function N_{i,p}(u) via Cox-de Boor, exact over Q. i must index a control
    // point (i < control_points().size()), else domain_error.
    [[nodiscard]] auto basis_function(std::size_t i, const Rational& u) const -> Result<Rational>;

    // Curve evaluation C(u) = sum_i N_{i,p}(u) P_i, exact; u outside the domain -> domain_error.
    [[nodiscard]] auto evaluate(const Rational& u) const -> Result<Rational>;

private:
    BSpline() = default;
    std::vector<Rational> knots_;
    std::vector<Rational> control_;
    std::size_t degree_{1};
};

// ---------------------------------------------------------------------------
// NURBS — rational B-splines with rational weights, exact over Q.
// ---------------------------------------------------------------------------
class NurbsCurve {
public:
    // Build from knots U, control points P, rational weights w (one per control point), and
    // degree p. Same structural requirements as BSpline; additionally weights must match the
    // control-point count and every weight must be nonzero (a zero weight is domain_error).
    [[nodiscard]] static auto make(std::vector<Rational> knots,
                                   std::vector<Rational> control_points,
                                   std::vector<Rational> weights, std::size_t degree)
        -> Result<NurbsCurve>;

    [[nodiscard]] auto degree() const noexcept -> std::size_t { return spline_.degree(); }
    [[nodiscard]] auto knots() const noexcept -> std::span<const Rational> {
        return spline_.knots();
    }
    [[nodiscard]] auto control_points() const noexcept -> std::span<const Rational> {
        return spline_.control_points();
    }
    [[nodiscard]] auto weights() const noexcept -> std::span<const Rational> { return weights_; }

    // C(u) = (sum_i w_i N_{i,p}(u) P_i) / (sum_i w_i N_{i,p}(u)), exact over Q. u outside the
    // domain, or a zero denominator, is domain_error.
    [[nodiscard]] auto evaluate(const Rational& u) const -> Result<Rational>;

private:
    NurbsCurve(BSpline s, std::vector<Rational> w) : spline_(std::move(s)), weights_(std::move(w)) {}
    BSpline spline_;
    std::vector<Rational> weights_;
};

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// --- small rational conveniences -------------------------------------------

[[nodiscard]] auto rat_zero() -> Rational { return Rational{}; }
[[nodiscard]] auto rat_one() -> Rational { return Rational::from_int(1); }
[[nodiscard]] auto ri(std::int64_t v) -> Rational { return Rational::from_int(v); }

// -1, 0, +1 sign of a Rational (denominator is always > 0, so the numerator decides).
[[nodiscard]] auto rat_sign(const Rational& r) -> int {
    if (r.is_zero()) {
        return 0;
    }
    return r.numerator() > 0 ? 1 : -1;
}

// Exact three-way comparison of two Rationals; overflow in the difference propagates.
[[nodiscard]] auto rat_cmp(const Rational& a, const Rational& b) -> Result<int> {
    auto d = a.subtract(b);
    if (!d) {
        return make_error<int>(d.error());
    }
    return rat_sign(*d);
}

// C(n, k) as an exact Rational (an integer value, kept as a fraction for uniform arithmetic).
// Built multiplicatively so every intermediate stays reduced; overflow propagates.
[[nodiscard]] auto binomial(std::size_t n, std::size_t k) -> Result<Rational> {
    if (k > n) {
        return rat_zero();
    }
    if (k > n - k) {
        k = n - k;
    }
    Rational result = rat_one();
    for (std::size_t i = 0; i < k; ++i) {
        // result *= (n - k + 1 + i) / (i + 1)
        const auto num = ri(static_cast<std::int64_t>(n - k + 1 + i));
        const auto den = ri(static_cast<std::int64_t>(i + 1));
        auto step = num.divide(den);
        if (!step) {
            return make_error<Rational>(step.error());
        }
        auto r = result.multiply(*step);
        if (!r) {
            return make_error<Rational>(r.error());
        }
        result = *r;
    }
    return result;
}

// Horner evaluation of a RationalPoly at a rational point (exact).
[[nodiscard]] auto poly_eval(const RationalPoly& p, const Rational& x) -> Result<Rational> {
    const std::span<const Rational> c = p.coefficients();
    Rational acc{};  // 0/1
    for (std::size_t k = c.size(); k-- > 0;) {
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

// Exact linear interpolation a + t*(b - a) over Q (the de Casteljau / de Boor blend
// step). Overflow in any intermediate propagates as MathError::overflow.
[[nodiscard]] auto lerp(const Rational& a, const Rational& b, const Rational& t)
    -> Result<Rational> {
    auto diff = b.subtract(a);
    if (!diff) {
        return make_error<Rational>(diff.error());
    }
    auto scaled = diff->multiply(t);
    if (!scaled) {
        return make_error<Rational>(scaled.error());
    }
    return a.add(*scaled);
}

// --- validation helpers -----------------------------------------------------

// Are the knots strictly increasing? Overflow in a comparison propagates as overflow.
[[nodiscard]] auto strictly_increasing(std::span<const Rational> xs) -> Result<bool> {
    for (std::size_t i = 1; i < xs.size(); ++i) {
        auto c = rat_cmp(xs[i - 1], xs[i]);
        if (!c) {
            return make_error<bool>(c.error());
        }
        if (*c >= 0) {
            return false;
        }
    }
    return true;
}

// Are the knots non-decreasing (equal entries allowed, e.g. B-spline multiplicities)?
[[nodiscard]] auto non_decreasing(std::span<const Rational> xs) -> Result<bool> {
    for (std::size_t i = 1; i < xs.size(); ++i) {
        auto c = rat_cmp(xs[i - 1], xs[i]);
        if (!c) {
            return make_error<bool>(c.error());
        }
        if (*c > 0) {
            return false;
        }
    }
    return true;
}

// Interval spacings h_i = x_{i+1} - x_i (all > 0 for strictly increasing knots).
[[nodiscard]] auto spacings(std::span<const Rational> xs) -> Result<std::vector<Rational>> {
    std::vector<Rational> h(xs.size() - 1);
    for (std::size_t i = 0; i + 1 < xs.size(); ++i) {
        auto d = xs[i + 1].subtract(xs[i]);
        if (!d) {
            return make_error<std::vector<Rational>>(d.error());
        }
        h[i] = *d;
    }
    return h;
}

// Secant slopes delta_i = (y_{i+1} - y_i) / h_i.
[[nodiscard]] auto secants(std::span<const Rational> ys, std::span<const Rational> h)
    -> Result<std::vector<Rational>> {
    std::vector<Rational> d(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) {
        auto dy = ys[i + 1].subtract(ys[i]);
        if (!dy) {
            return make_error<std::vector<Rational>>(dy.error());
        }
        auto s = dy->divide(h[i]);  // h_i != 0
        if (!s) {
            return make_error<std::vector<Rational>>(s.error());
        }
        d[i] = *s;
    }
    return d;
}

// Locate the interval index i with xs[i] <= x <= xs[i+1] for a strictly increasing xs of at
// least two entries. domain_error if x lies outside [xs.front(), xs.back()].
[[nodiscard]] auto find_interval(std::span<const Rational> xs, const Rational& x)
    -> Result<std::size_t> {
    auto lo = rat_cmp(x, xs.front());
    auto hi = rat_cmp(x, xs.back());
    if (!lo || !hi) {
        return make_error<std::size_t>(lo ? hi.error() : lo.error());
    }
    if (*lo < 0 || *hi > 0) {
        return make_error<std::size_t>(MathError::domain_error);
    }
    for (std::size_t i = 0; i + 2 < xs.size(); ++i) {
        auto c = rat_cmp(x, xs[i + 1]);
        if (!c) {
            return make_error<std::size_t>(c.error());
        }
        if (*c <= 0) {
            return i;
        }
    }
    return xs.size() - 2;  // last interval (covers x == xs.back())
}

// Evaluate a piecewise polynomial given its knots and per-interval pieces (in absolute x).
[[nodiscard]] auto eval_piecewise(std::span<const Rational> xs,
                                  std::span<const RationalPoly> pieces, const Rational& x)
    -> Result<Rational> {
    auto i = find_interval(xs, x);
    if (!i) {
        return make_error<Rational>(i.error());
    }
    return poly_eval(pieces[*i], x);
}

// --- piecewise-cubic assembly ----------------------------------------------

// The cubic on [x_i, x_{i+1}] from the moment (second-derivative) endpoints, in absolute x:
//   S(x) = M_i (x_{i+1}-x)^3/(6h) + M_{i+1}(x-x_i)^3/(6h)
//        + (y_i    - M_i    h^2/6)(x_{i+1}-x)/h
//        + (y_{i+1} - M_{i+1}h^2/6)(x-x_i)/h,   h = x_{i+1}-x_i.
[[nodiscard]] auto moment_piece(const Rational& xi, const Rational& xi1, const Rational& yi,
                                const Rational& yi1, const Rational& mi, const Rational& mi1,
                                const Rational& h) -> Result<RationalPoly> {
    // A = (x_{i+1} - x), B = (x - x_i) as linear RationalPolys.
    auto negxi = xi.negate();
    if (!negxi) {
        return make_error<RationalPoly>(negxi.error());
    }
    const RationalPoly a = RationalPoly::from_coeffs({xi1, ri(-1)});
    const RationalPoly b = RationalPoly::from_coeffs({*negxi, ri(1)});

    auto a2 = a.multiply(a);
    auto b2 = b.multiply(b);
    if (!a2 || !b2) {
        return make_error<RationalPoly>(a2 ? b2.error() : a2.error());
    }
    auto a3 = a2->multiply(a);
    auto b3 = b2->multiply(b);
    if (!a3 || !b3) {
        return make_error<RationalPoly>(a3 ? b3.error() : a3.error());
    }

    // Cubic coefficients c3i = M_i/(6h), c3j = M_{i+1}/(6h).
    auto six_h = h.multiply(ri(6));
    if (!six_h) {
        return make_error<RationalPoly>(six_h.error());
    }
    auto c3i = mi.divide(*six_h);
    auto c3j = mi1.divide(*six_h);
    if (!c3i || !c3j) {
        return make_error<RationalPoly>(c3i ? c3j.error() : c3i.error());
    }

    // Linear coefficients lin_i = y_i/h - M_i h/6, lin_j = y_{i+1}/h - M_{i+1} h/6.
    auto mih = mi.multiply(h);
    auto mjh = mi1.multiply(h);
    if (!mih || !mjh) {
        return make_error<RationalPoly>(mih ? mjh.error() : mih.error());
    }
    auto mih6 = mih->divide(ri(6));
    auto mjh6 = mjh->divide(ri(6));
    auto yih = yi.divide(h);
    auto yjh = yi1.divide(h);
    if (!mih6 || !mjh6 || !yih || !yjh) {
        return make_error<RationalPoly>(MathError::overflow);
    }
    auto lin_i = yih->subtract(*mih6);
    auto lin_j = yjh->subtract(*mjh6);
    if (!lin_i || !lin_j) {
        return make_error<RationalPoly>(lin_i ? lin_j.error() : lin_i.error());
    }

    // S = c3i*A^3 + c3j*B^3 + lin_i*A + lin_j*B.
    auto ta3 = a3->scale(*c3i);
    auto tb3 = b3->scale(*c3j);
    auto ta = a.scale(*lin_i);
    auto tb = b.scale(*lin_j);
    if (!ta3 || !tb3 || !ta || !tb) {
        return make_error<RationalPoly>(MathError::overflow);
    }
    auto s1 = ta3->add(*tb3);
    if (!s1) {
        return make_error<RationalPoly>(s1.error());
    }
    auto s2 = s1->add(*ta);
    if (!s2) {
        return make_error<RationalPoly>(s2.error());
    }
    return s2->add(*tb);
}

// The cubic Hermite piece on [x_i, x_{i+1}] with values y_i,y_{i+1} and slopes d_i,d_{i+1}
// (in absolute x). Uses the Hermite basis with t = (x - x_i)/h:
//   H = h00(t) y_i + h10(t) (h d_i) + h01(t) y_{i+1} + h11(t) (h d_{i+1}),
//   h00 = 2t^3-3t^2+1, h10 = t^3-2t^2+t, h01 = -2t^3+3t^2, h11 = t^3-t^2.
[[nodiscard]] auto hermite_piece(const Rational& xi, const Rational& xi1, const Rational& yi,
                                 const Rational& yi1, const Rational& di, const Rational& di1,
                                 const Rational& h) -> Result<RationalPoly> {
    auto inv_h = rat_one().divide(h);  // h != 0
    if (!inv_h) {
        return make_error<RationalPoly>(inv_h.error());
    }
    auto negxi = xi.negate();
    if (!negxi) {
        return make_error<RationalPoly>(negxi.error());
    }
    auto c0 = negxi->multiply(*inv_h);  // -x_i / h
    if (!c0) {
        return make_error<RationalPoly>(c0.error());
    }
    const RationalPoly t = RationalPoly::from_coeffs({*c0, *inv_h});  // (x - x_i)/h
    auto t2 = t.multiply(t);
    if (!t2) {
        return make_error<RationalPoly>(t2.error());
    }
    auto t3 = t2->multiply(t);
    if (!t3) {
        return make_error<RationalPoly>(t3.error());
    }
    const RationalPoly one = RationalPoly::constant(rat_one());

    // Build each basis polynomial from t, t^2, t^3.
    auto two_t3 = t3->scale(ri(2));
    auto three_t2 = t2->scale(ri(3));
    auto neg2_t3 = t3->scale(ri(-2));
    auto two_t2 = t2->scale(ri(2));
    if (!two_t3 || !three_t2 || !neg2_t3 || !two_t2) {
        return make_error<RationalPoly>(MathError::overflow);
    }
    // h00 = 2t^3 - 3t^2 + 1
    auto h00a = two_t3->subtract(*three_t2);
    auto h00 = h00a ? h00a->add(one) : make_error<RationalPoly>(h00a.error());
    // h10 = t^3 - 2t^2 + t
    auto h10a = t3->subtract(*two_t2);
    auto h10 = h10a ? h10a->add(t) : make_error<RationalPoly>(h10a.error());
    // h01 = -2t^3 + 3t^2
    auto h01 = neg2_t3->add(*three_t2);
    // h11 = t^3 - t^2
    auto h11 = t3->subtract(*t2);
    if (!h00 || !h10 || !h01 || !h11) {
        return make_error<RationalPoly>(MathError::overflow);
    }

    auto hdi = h.multiply(di);
    auto hdi1 = h.multiply(di1);
    if (!hdi || !hdi1) {
        return make_error<RationalPoly>(hdi ? hdi1.error() : hdi.error());
    }
    // H = y_i*h00 + (h d_i)*h10 + y_{i+1}*h01 + (h d_{i+1})*h11.
    auto t0 = h00->scale(yi);
    auto t1 = h10->scale(*hdi);
    auto t4 = h01->scale(yi1);
    auto t5 = h11->scale(*hdi1);
    if (!t0 || !t1 || !t4 || !t5) {
        return make_error<RationalPoly>(MathError::overflow);
    }
    auto s1 = t0->add(*t1);
    if (!s1) {
        return make_error<RationalPoly>(s1.error());
    }
    auto s2 = s1->add(*t4);
    if (!s2) {
        return make_error<RationalPoly>(s2.error());
    }
    return s2->add(*t5);
}

// Assemble every moment_piece for a solved spline.
[[nodiscard]] auto build_moment_pieces(std::span<const Rational> xs, std::span<const Rational> ys,
                                       std::span<const Rational> moments,
                                       std::span<const Rational> h)
    -> Result<std::vector<RationalPoly>> {
    std::vector<RationalPoly> pieces;
    pieces.reserve(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) {
        auto piece =
            moment_piece(xs[i], xs[i + 1], ys[i], ys[i + 1], moments[i], moments[i + 1], h[i]);
        if (!piece) {
            return make_error<std::vector<RationalPoly>>(piece.error());
        }
        pieces.push_back(std::move(*piece));
    }
    return pieces;
}

}  // namespace

// --- CubicSpline::natural ---------------------------------------------------

auto CubicSpline::natural(std::span<const Rational> xs, std::span<const Rational> ys)
    -> Result<CubicSpline> {
    if (xs.size() != ys.size() || xs.size() < 2) {
        return make_error<CubicSpline>(MathError::domain_error);
    }
    auto inc = strictly_increasing(xs);
    if (!inc) {
        return make_error<CubicSpline>(inc.error());
    }
    if (!*inc) {
        return make_error<CubicSpline>(MathError::domain_error);
    }
    auto h = spacings(xs);
    if (!h) {
        return make_error<CubicSpline>(h.error());
    }
    auto delta = secants(ys, *h);
    if (!delta) {
        return make_error<CubicSpline>(delta.error());
    }
    const std::size_t n = xs.size() - 1;  // number of intervals
    std::vector<Rational> moments(xs.size(), rat_zero());  // M_0 = M_n = 0 (natural)

    if (n >= 2) {
        const std::size_t s = n - 1;  // interior unknowns M_1..M_{n-1}
        std::vector<Rational> sub(s == 0 ? 0 : s - 1);
        std::vector<Rational> diag(s);
        std::vector<Rational> super(s == 0 ? 0 : s - 1);
        std::vector<Rational> rhs(s);
        for (std::size_t r = 0; r < s; ++r) {
            const std::size_t i = r + 1;  // interior knot index
            auto hh = (*h)[i - 1].add((*h)[i]);
            auto d = hh ? hh->multiply(ri(2)) : make_error<Rational>(hh.error());
            auto dd = (*delta)[i].subtract((*delta)[i - 1]);
            auto rr = dd ? dd->multiply(ri(6)) : make_error<Rational>(dd.error());
            if (!d || !rr) {
                return make_error<CubicSpline>(MathError::overflow);
            }
            diag[r] = *d;
            rhs[r] = *rr;
            if (r > 0) {
                sub[r - 1] = (*h)[i - 1];  // coefficient of M_{i-1}
            }
            if (r + 1 < s) {
                super[r] = (*h)[i];  // coefficient of M_{i+1}
            }
        }
        auto interior = solve_tridiagonal(sub, diag, super, rhs);
        if (!interior) {
            return make_error<CubicSpline>(interior.error());
        }
        for (std::size_t r = 0; r < s; ++r) {
            moments[r + 1] = (*interior)[r];
        }
    }

    auto pieces = build_moment_pieces(xs, ys, moments, *h);
    if (!pieces) {
        return make_error<CubicSpline>(pieces.error());
    }
    CubicSpline out;
    out.xs_.assign(xs.begin(), xs.end());
    out.ys_.assign(ys.begin(), ys.end());
    out.moments_ = std::move(moments);
    out.pieces_ = std::move(*pieces);
    out.boundary_ = SplineBoundary::natural;
    return out;
}

// --- CubicSpline::clamped ---------------------------------------------------

auto CubicSpline::clamped(std::span<const Rational> xs, std::span<const Rational> ys,
                          const Rational& deriv_left, const Rational& deriv_right)
    -> Result<CubicSpline> {
    if (xs.size() != ys.size() || xs.size() < 2) {
        return make_error<CubicSpline>(MathError::domain_error);
    }
    auto inc = strictly_increasing(xs);
    if (!inc) {
        return make_error<CubicSpline>(inc.error());
    }
    if (!*inc) {
        return make_error<CubicSpline>(MathError::domain_error);
    }
    auto h = spacings(xs);
    if (!h) {
        return make_error<CubicSpline>(h.error());
    }
    auto delta = secants(ys, *h);
    if (!delta) {
        return make_error<CubicSpline>(delta.error());
    }
    const std::size_t n = xs.size() - 1;  // intervals
    const std::size_t s = n + 1;          // unknowns M_0..M_n
    std::vector<Rational> sub(s - 1);
    std::vector<Rational> diag(s);
    std::vector<Rational> super(s - 1);
    std::vector<Rational> rhs(s);

    // Row 0: 2 h_0 M_0 + h_0 M_1 = 6(delta_0 - deriv_left).
    auto d0 = (*h)[0].multiply(ri(2));
    auto r0a = (*delta)[0].subtract(deriv_left);
    auto r0 = r0a ? r0a->multiply(ri(6)) : make_error<Rational>(r0a.error());
    if (!d0 || !r0) {
        return make_error<CubicSpline>(MathError::overflow);
    }
    diag[0] = *d0;
    super[0] = (*h)[0];
    rhs[0] = *r0;

    // Interior rows i = 1..n-1.
    for (std::size_t i = 1; i < n; ++i) {
        auto hh = (*h)[i - 1].add((*h)[i]);
        auto d = hh ? hh->multiply(ri(2)) : make_error<Rational>(hh.error());
        auto dd = (*delta)[i].subtract((*delta)[i - 1]);
        auto rr = dd ? dd->multiply(ri(6)) : make_error<Rational>(dd.error());
        if (!d || !rr) {
            return make_error<CubicSpline>(MathError::overflow);
        }
        diag[i] = *d;
        rhs[i] = *rr;
        sub[i - 1] = (*h)[i - 1];
        super[i] = (*h)[i];
    }

    // Row n: h_{n-1} M_{n-1} + 2 h_{n-1} M_n = 6(deriv_right - delta_{n-1}).
    auto dn = (*h)[n - 1].multiply(ri(2));
    auto rna = deriv_right.subtract((*delta)[n - 1]);
    auto rn = rna ? rna->multiply(ri(6)) : make_error<Rational>(rna.error());
    if (!dn || !rn) {
        return make_error<CubicSpline>(MathError::overflow);
    }
    diag[n] = *dn;
    sub[n - 1] = (*h)[n - 1];
    rhs[n] = *rn;

    auto moments = solve_tridiagonal(sub, diag, super, rhs);
    if (!moments) {
        return make_error<CubicSpline>(moments.error());
    }
    auto pieces = build_moment_pieces(xs, ys, *moments, *h);
    if (!pieces) {
        return make_error<CubicSpline>(pieces.error());
    }
    CubicSpline out;
    out.xs_.assign(xs.begin(), xs.end());
    out.ys_.assign(ys.begin(), ys.end());
    out.moments_ = std::move(*moments);
    out.pieces_ = std::move(*pieces);
    out.boundary_ = SplineBoundary::clamped;
    return out;
}

// --- CubicSpline::periodic (cyclic tridiagonal via Sherman-Morrison) --------

auto CubicSpline::periodic(std::span<const Rational> xs, std::span<const Rational> ys)
    -> Result<CubicSpline> {
    if (xs.size() != ys.size() || xs.size() < 3) {
        return make_error<CubicSpline>(MathError::domain_error);
    }
    auto inc = strictly_increasing(xs);
    if (!inc) {
        return make_error<CubicSpline>(inc.error());
    }
    if (!*inc) {
        return make_error<CubicSpline>(MathError::domain_error);
    }
    if (!(ys.front() == ys.back())) {  // periodicity requires y_0 == y_n
        return make_error<CubicSpline>(MathError::domain_error);
    }
    auto h = spacings(xs);
    if (!h) {
        return make_error<CubicSpline>(h.error());
    }
    auto delta = secants(ys, *h);
    if (!delta) {
        return make_error<CubicSpline>(delta.error());
    }
    const std::size_t nn = xs.size() - 1;  // intervals; unknowns M_0..M_{nn-1}, M_nn == M_0
    // Cyclic-tridiagonal rows i = 0..nn-1 (indices mod nn):
    //   a_i M_{i-1} + b_i M_i + c_i M_{i+1} = d_i,
    //   a_i = h_{i-1}, b_i = 2(h_{i-1}+h_i), c_i = h_i,
    //   d_i = 6(delta_i - delta_{i-1}),   h_{-1}=h_{nn-1}, delta_{-1}=delta_{nn-1}.
    std::vector<Rational> avec(nn);
    std::vector<Rational> bvec(nn);
    std::vector<Rational> cvec(nn);
    std::vector<Rational> dvec(nn);
    for (std::size_t i = 0; i < nn; ++i) {
        const std::size_t im1 = (i + nn - 1) % nn;  // previous interval, wrapped
        avec[i] = (*h)[im1];
        cvec[i] = (*h)[i];
        auto hh = (*h)[im1].add((*h)[i]);
        auto bb = hh ? hh->multiply(ri(2)) : make_error<Rational>(hh.error());
        auto dd = (*delta)[i].subtract((*delta)[im1]);
        auto d6 = dd ? dd->multiply(ri(6)) : make_error<Rational>(dd.error());
        if (!bb || !d6) {
            return make_error<CubicSpline>(MathError::overflow);
        }
        bvec[i] = *bb;
        dvec[i] = *d6;
    }

    // Sherman-Morrison: A = T + u v^T with the two corner entries alpha = A[nn-1][0] = c_{nn-1}
    // and beta = A[0][nn-1] = a_0 folded into u v^T. gamma = -b_0 (nonzero, b_0 > 0).
    const Rational alpha = cvec[nn - 1];
    const Rational beta = avec[0];
    auto gamma_r = bvec[0].negate();
    if (!gamma_r) {
        return make_error<CubicSpline>(gamma_r.error());
    }
    const Rational gamma = *gamma_r;

    // Tridiagonal T: band = A's band; diag[0] -= gamma; diag[nn-1] -= alpha*beta/gamma.
    std::vector<Rational> sub(nn - 1);
    std::vector<Rational> diag(nn);
    std::vector<Rational> super(nn - 1);
    for (std::size_t i = 0; i < nn; ++i) {
        diag[i] = bvec[i];
    }
    for (std::size_t i = 1; i < nn; ++i) {
        sub[i - 1] = avec[i];  // subdiagonal coefficient a_i of row i
    }
    for (std::size_t i = 0; i + 1 < nn; ++i) {
        super[i] = cvec[i];  // superdiagonal coefficient c_i of row i
    }
    auto d0mod = diag[0].subtract(gamma);
    auto ab = alpha.multiply(beta);
    auto abg = ab ? ab->divide(gamma) : make_error<Rational>(ab.error());
    auto dnmod = abg ? diag[nn - 1].subtract(*abg) : make_error<Rational>(abg.error());
    if (!d0mod || !dnmod) {
        return make_error<CubicSpline>(MathError::overflow);
    }
    diag[0] = *d0mod;
    diag[nn - 1] = *dnmod;

    // u = gamma e_0 + alpha e_{nn-1}.
    std::vector<Rational> uvec(nn, rat_zero());
    uvec[0] = gamma;
    uvec[nn - 1] = alpha;

    auto yv = solve_tridiagonal(sub, diag, super, dvec);  // T y = d
    auto zv = solve_tridiagonal(sub, diag, super, uvec);  // T z = u
    if (!yv) {
        return make_error<CubicSpline>(yv.error());
    }
    if (!zv) {
        return make_error<CubicSpline>(zv.error());
    }

    // v = e_0 + (beta/gamma) e_{nn-1}. fact = (v.y)/(1 + v.z); x = y - fact z.
    auto bg = beta.divide(gamma);
    if (!bg) {
        return make_error<CubicSpline>(bg.error());
    }
    auto vy_tail = (*yv)[nn - 1].multiply(*bg);
    auto vz_tail = (*zv)[nn - 1].multiply(*bg);
    if (!vy_tail || !vz_tail) {
        return make_error<CubicSpline>(vy_tail ? vz_tail.error() : vy_tail.error());
    }
    auto vy = (*yv)[0].add(*vy_tail);
    auto vz = (*zv)[0].add(*vz_tail);
    if (!vy || !vz) {
        return make_error<CubicSpline>(vy ? vz.error() : vy.error());
    }
    auto denom = vz->add(rat_one());
    if (!denom) {
        return make_error<CubicSpline>(denom.error());
    }
    if (denom->is_zero()) {
        return make_error<CubicSpline>(MathError::domain_error);  // singular periodic system
    }
    auto fact = vy->divide(*denom);
    if (!fact) {
        return make_error<CubicSpline>(fact.error());
    }

    std::vector<Rational> moments(xs.size());
    for (std::size_t i = 0; i < nn; ++i) {
        auto fz = fact->multiply((*zv)[i]);
        auto xi = fz ? (*yv)[i].subtract(*fz) : make_error<Rational>(fz.error());
        if (!xi) {
            return make_error<CubicSpline>(MathError::overflow);
        }
        moments[i] = *xi;
    }
    moments[nn] = moments[0];  // M_n == M_0 (periodicity)

    auto pieces = build_moment_pieces(xs, ys, moments, *h);
    if (!pieces) {
        return make_error<CubicSpline>(pieces.error());
    }
    CubicSpline out;
    out.xs_.assign(xs.begin(), xs.end());
    out.ys_.assign(ys.begin(), ys.end());
    out.moments_ = std::move(moments);
    out.pieces_ = std::move(*pieces);
    out.boundary_ = SplineBoundary::periodic;
    return out;
}

auto CubicSpline::evaluate(const Rational& x) const -> Result<Rational> {
    return eval_piecewise(xs_, pieces_, x);
}

// --- HermiteSpline ----------------------------------------------------------

auto HermiteSpline::from_slopes(std::span<const Rational> xs, std::span<const Rational> ys,
                                std::span<const Rational> slopes) -> Result<HermiteSpline> {
    if (xs.size() != ys.size() || xs.size() != slopes.size() || xs.size() < 2) {
        return make_error<HermiteSpline>(MathError::domain_error);
    }
    auto inc = strictly_increasing(xs);
    if (!inc) {
        return make_error<HermiteSpline>(inc.error());
    }
    if (!*inc) {
        return make_error<HermiteSpline>(MathError::domain_error);
    }
    auto h = spacings(xs);
    if (!h) {
        return make_error<HermiteSpline>(h.error());
    }
    std::vector<RationalPoly> pieces;
    pieces.reserve(h->size());
    for (std::size_t i = 0; i < h->size(); ++i) {
        auto piece = hermite_piece(xs[i], xs[i + 1], ys[i], ys[i + 1], slopes[i], slopes[i + 1],
                                   (*h)[i]);
        if (!piece) {
            return make_error<HermiteSpline>(piece.error());
        }
        pieces.push_back(std::move(*piece));
    }
    HermiteSpline out;
    out.xs_.assign(xs.begin(), xs.end());
    out.ys_.assign(ys.begin(), ys.end());
    out.slopes_.assign(slopes.begin(), slopes.end());
    out.pieces_ = std::move(pieces);
    return out;
}

namespace {

// Fritsch-Carlson shape-preserving limit for a one-sided endpoint slope d, given the
// adjacent secant delta_end and the next secant delta_next. Keeps the endpoint from
// introducing a spurious extremum: zero it if it opposes delta_end, else clamp its magnitude
// to 3|delta_end| when the two adjacent secants disagree in sign. Exact over Q.
[[nodiscard]] auto limit_endpoint(const Rational& d, const Rational& delta_end,
                                  const Rational& delta_next) -> Result<Rational> {
    const int sd = rat_sign(d);
    if (sd == 0) {
        return d;
    }
    const int se = rat_sign(delta_end);
    if (sd != se) {
        return rat_zero();
    }
    if (se != rat_sign(delta_next)) {
        auto three = delta_end.multiply(ri(3));
        if (!three) {
            return make_error<Rational>(three.error());
        }
        auto c = rat_cmp(d, *three);
        if (!c) {
            return make_error<Rational>(c.error());
        }
        // same sign as delta_end: clamp |d| down to |3 delta_end| when it exceeds it.
        if ((se > 0 && *c > 0) || (se < 0 && *c < 0)) {
            return *three;
        }
    }
    return d;
}

// The PCHIP interior slope at knot i (weighted harmonic mean of the two adjacent secants),
// or 0 where the data is not locally monotone. Exact over Q.
[[nodiscard]] auto pchip_interior_slope(const Rational& hL, const Rational& hR,
                                        const Rational& dL, const Rational& dR)
    -> Result<Rational> {
    const int sL = rat_sign(dL);
    const int sR = rat_sign(dR);
    if (sL == 0 || sR == 0 || sL != sR) {
        return rat_zero();  // local extremum / flat: zero slope preserves monotonicity
    }
    // w1 = 2 hR + hL, w2 = hR + 2 hL; d = (w1 + w2) / (w1/dL + w2/dR).
    auto twohR = hR.multiply(ri(2));
    auto twohL = hL.multiply(ri(2));
    if (!twohR || !twohL) {
        return make_error<Rational>(twohR ? twohL.error() : twohR.error());
    }
    auto w1 = twohR->add(hL);
    auto w2 = hR.add(*twohL);
    if (!w1 || !w2) {
        return make_error<Rational>(w1 ? w2.error() : w1.error());
    }
    auto wsum = w1->add(*w2);
    auto q1 = w1->divide(dL);
    auto q2 = w2->divide(dR);
    if (!wsum || !q1 || !q2) {
        return make_error<Rational>(MathError::overflow);
    }
    auto qsum = q1->add(*q2);
    if (!qsum) {
        return make_error<Rational>(qsum.error());
    }
    return wsum->divide(*qsum);  // qsum != 0 (both q_k share dL/dR sign)
}

}  // namespace

auto HermiteSpline::pchip(std::span<const Rational> xs, std::span<const Rational> ys)
    -> Result<HermiteSpline> {
    if (xs.size() != ys.size() || xs.size() < 2) {
        return make_error<HermiteSpline>(MathError::domain_error);
    }
    auto inc = strictly_increasing(xs);
    if (!inc) {
        return make_error<HermiteSpline>(inc.error());
    }
    if (!*inc) {
        return make_error<HermiteSpline>(MathError::domain_error);
    }
    auto h = spacings(xs);
    if (!h) {
        return make_error<HermiteSpline>(h.error());
    }
    auto delta = secants(ys, *h);
    if (!delta) {
        return make_error<HermiteSpline>(delta.error());
    }
    const std::size_t n = xs.size() - 1;  // intervals
    std::vector<Rational> slopes(xs.size());

    if (n == 1) {
        // Two points: the unique monotone choice is the linear slope at both ends.
        slopes[0] = (*delta)[0];
        slopes[1] = (*delta)[0];
    } else {
        // Interior slopes.
        for (std::size_t i = 1; i < n; ++i) {
            auto d = pchip_interior_slope((*h)[i - 1], (*h)[i], (*delta)[i - 1], (*delta)[i]);
            if (!d) {
                return make_error<HermiteSpline>(d.error());
            }
            slopes[i] = *d;
        }
        // Left endpoint: non-centered three-point slope, then shape-limited.
        // d0 = ((2h_0 + h_1) delta_0 - h_0 delta_1) / (h_0 + h_1).
        {
            auto twoh0 = (*h)[0].multiply(ri(2));
            auto a = twoh0 ? twoh0->add((*h)[1]) : make_error<Rational>(twoh0.error());
            auto term1 = a ? a->multiply((*delta)[0]) : make_error<Rational>(a.error());
            auto term2 = (*h)[0].multiply((*delta)[1]);
            auto numer = (term1 && term2) ? term1->subtract(*term2)
                                          : make_error<Rational>(MathError::overflow);
            auto den = (*h)[0].add((*h)[1]);
            auto d0 = (numer && den) ? numer->divide(*den) : make_error<Rational>(MathError::overflow);
            if (!d0) {
                return make_error<HermiteSpline>(d0.error());
            }
            auto lim = limit_endpoint(*d0, (*delta)[0], (*delta)[1]);
            if (!lim) {
                return make_error<HermiteSpline>(lim.error());
            }
            slopes[0] = *lim;
        }
        // Right endpoint: mirror formula.
        // dn = ((2h_{n-1} + h_{n-2}) delta_{n-1} - h_{n-1} delta_{n-2}) / (h_{n-1} + h_{n-2}).
        {
            auto twohn = (*h)[n - 1].multiply(ri(2));
            auto a = twohn ? twohn->add((*h)[n - 2]) : make_error<Rational>(twohn.error());
            auto term1 = a ? a->multiply((*delta)[n - 1]) : make_error<Rational>(a.error());
            auto term2 = (*h)[n - 1].multiply((*delta)[n - 2]);
            auto numer = (term1 && term2) ? term1->subtract(*term2)
                                          : make_error<Rational>(MathError::overflow);
            auto den = (*h)[n - 1].add((*h)[n - 2]);
            auto dn = (numer && den) ? numer->divide(*den) : make_error<Rational>(MathError::overflow);
            if (!dn) {
                return make_error<HermiteSpline>(dn.error());
            }
            auto lim = limit_endpoint(*dn, (*delta)[n - 1], (*delta)[n - 2]);
            if (!lim) {
                return make_error<HermiteSpline>(lim.error());
            }
            slopes[n] = *lim;
        }
    }

    return from_slopes(xs, ys, slopes);
}

auto HermiteSpline::evaluate(const Rational& x) const -> Result<Rational> {
    return eval_piecewise(xs_, pieces_, x);
}

// --- BezierCurve ------------------------------------------------------------

auto BezierCurve::make(std::vector<Rational> control_points) -> Result<BezierCurve> {
    if (control_points.empty()) {
        return make_error<BezierCurve>(MathError::domain_error);
    }
    return BezierCurve{std::move(control_points)};
}

auto BezierCurve::evaluate(const Rational& t) const -> Result<Rational> {
    auto lo = rat_cmp(t, rat_zero());
    auto hi = rat_cmp(t, rat_one());
    if (!lo || !hi) {
        return make_error<Rational>(lo ? hi.error() : lo.error());
    }
    if (*lo < 0 || *hi > 0) {
        return make_error<Rational>(MathError::domain_error);
    }
    std::vector<Rational> b(control_.begin(), control_.end());
    for (std::size_t r = 1; r < b.size(); ++r) {
        for (std::size_t i = 0; i + r < b.size(); ++i) {
            auto v = lerp(b[i], b[i + 1], t);
            if (!v) {
                return make_error<Rational>(v.error());
            }
            b[i] = *v;
        }
    }
    return b[0];
}

auto BezierCurve::evaluate_bernstein(const Rational& t) const -> Result<Rational> {
    auto lo = rat_cmp(t, rat_zero());
    auto hi = rat_cmp(t, rat_one());
    if (!lo || !hi) {
        return make_error<Rational>(lo ? hi.error() : lo.error());
    }
    if (*lo < 0 || *hi > 0) {
        return make_error<Rational>(MathError::domain_error);
    }
    const std::size_t n = degree();
    auto omt = rat_one().subtract(t);
    if (!omt) {
        return make_error<Rational>(omt.error());
    }
    // Precompute t^i and (1-t)^j.
    std::vector<Rational> tp(n + 1);
    std::vector<Rational> op(n + 1);
    tp[0] = rat_one();
    op[0] = rat_one();
    for (std::size_t i = 1; i <= n; ++i) {
        auto a = tp[i - 1].multiply(t);
        auto c = op[i - 1].multiply(*omt);
        if (!a || !c) {
            return make_error<Rational>(a ? c.error() : a.error());
        }
        tp[i] = *a;
        op[i] = *c;
    }
    Rational acc{};
    for (std::size_t i = 0; i <= n; ++i) {
        auto cni = binomial(n, i);
        if (!cni) {
            return make_error<Rational>(cni.error());
        }
        auto term = cni->multiply(tp[i]);
        auto term2 = term ? term->multiply(op[n - i]) : make_error<Rational>(term.error());
        auto term3 = term2 ? term2->multiply(control_[i]) : make_error<Rational>(term2.error());
        if (!term3) {
            return make_error<Rational>(term3.error());
        }
        auto s = acc.add(*term3);
        if (!s) {
            return make_error<Rational>(s.error());
        }
        acc = *s;
    }
    return acc;
}

auto BezierCurve::elevate() const -> Result<BezierCurve> {
    const std::size_t n = degree();
    std::vector<Rational> out(n + 2);
    out[0] = control_[0];
    out[n + 1] = control_[n];
    for (std::size_t i = 1; i <= n; ++i) {
        // out[i] = (i/(n+1)) b[i-1] + (1 - i/(n+1)) b[i].
        auto frac = ri(static_cast<std::int64_t>(i)).divide(ri(static_cast<std::int64_t>(n + 1)));
        if (!frac) {
            return make_error<BezierCurve>(frac.error());
        }
        auto omf = rat_one().subtract(*frac);
        auto left = omf ? frac->multiply(control_[i - 1]) : make_error<Rational>(omf.error());
        auto right = omf ? omf->multiply(control_[i]) : make_error<Rational>(omf.error());
        auto v = (left && right) ? left->add(*right) : make_error<Rational>(MathError::overflow);
        if (!v) {
            return make_error<BezierCurve>(v.error());
        }
        out[i] = *v;
    }
    return BezierCurve{std::move(out)};
}

auto BezierCurve::to_power_basis() const -> Result<RationalPoly> {
    const std::size_t n = degree();
    std::vector<Rational> coeffs(n + 1, rat_zero());
    for (std::size_t j = 0; j <= n; ++j) {
        // c_j = C(n,j) * sum_{i=0}^{j} (-1)^{j-i} C(j,i) b_i.
        Rational inner{};
        for (std::size_t i = 0; i <= j; ++i) {
            auto cji = binomial(j, i);
            if (!cji) {
                return make_error<RationalPoly>(cji.error());
            }
            auto term = cji->multiply(control_[i]);
            if (!term) {
                return make_error<RationalPoly>(term.error());
            }
            Rational signed_term = *term;
            if (((j - i) & 1U) != 0U) {
                auto neg = term->negate();
                if (!neg) {
                    return make_error<RationalPoly>(neg.error());
                }
                signed_term = *neg;
            }
            auto s = inner.add(signed_term);
            if (!s) {
                return make_error<RationalPoly>(s.error());
            }
            inner = *s;
        }
        auto cnj = binomial(n, j);
        if (!cnj) {
            return make_error<RationalPoly>(cnj.error());
        }
        auto cj = cnj->multiply(inner);
        if (!cj) {
            return make_error<RationalPoly>(cj.error());
        }
        coeffs[j] = *cj;
    }
    return RationalPoly::from_coeffs(std::move(coeffs));
}

auto BezierCurve::from_power_basis(const RationalPoly& p, std::size_t degree)
    -> Result<BezierCurve> {
    if (p.degree() > static_cast<std::int64_t>(degree)) {
        return make_error<BezierCurve>(MathError::domain_error);
    }
    const std::size_t n = degree;
    std::vector<Rational> b(n + 1, rat_zero());
    for (std::size_t i = 0; i <= n; ++i) {
        // b_i = sum_{j=0}^{i} (C(i,j)/C(n,j)) c_j.
        Rational bi{};
        for (std::size_t j = 0; j <= i; ++j) {
            auto cij = binomial(i, j);
            auto cnj = binomial(n, j);
            if (!cij || !cnj) {
                return make_error<BezierCurve>(cij ? cnj.error() : cij.error());
            }
            auto ratio = cij->divide(*cnj);  // C(n,j) != 0 for j <= n
            auto term = ratio ? ratio->multiply(p.coefficient(j))
                              : make_error<Rational>(ratio.error());
            auto s = term ? bi.add(*term) : make_error<Rational>(term.error());
            if (!s) {
                return make_error<BezierCurve>(s.error());
            }
            bi = *s;
        }
        b[i] = bi;
    }
    return BezierCurve{std::move(b)};
}

auto bezier_subdivide(const BezierCurve& curve, const Rational& t) -> Result<BezierSplit> {
    auto lo = rat_cmp(t, rat_zero());
    auto hi = rat_cmp(t, rat_one());
    if (!lo || !hi) {
        return make_error<BezierSplit>(lo ? hi.error() : lo.error());
    }
    if (*lo < 0 || *hi > 0) {
        return make_error<BezierSplit>(MathError::domain_error);
    }
    const std::span<const Rational> cp = curve.control_points();
    const std::size_t n = cp.size() - 1;
    std::vector<Rational> work(cp.begin(), cp.end());
    std::vector<Rational> left(n + 1);
    std::vector<Rational> right(n + 1);
    left[0] = work[0];
    right[0] = work[n];
    for (std::size_t r = 1; r <= n; ++r) {
        for (std::size_t i = 0; i + r <= n; ++i) {
            auto v = lerp(work[i], work[i + 1], t);
            if (!v) {
                return make_error<BezierSplit>(v.error());
            }
            work[i] = *v;
        }
        left[r] = work[0];
        right[r] = work[n - r];
    }
    // Right control points run from the apex back to the original endpoint.
    std::vector<Rational> right_ordered(n + 1);
    for (std::size_t r = 0; r <= n; ++r) {
        right_ordered[r] = right[n - r];
    }
    auto lc = BezierCurve::make(std::move(left));
    auto rc = BezierCurve::make(std::move(right_ordered));
    if (!lc || !rc) {
        return make_error<BezierSplit>(MathError::domain_error);
    }
    return BezierSplit{.left = std::move(*lc), .right = std::move(*rc)};
}

// --- BezierCurve2 -----------------------------------------------------------

auto BezierCurve2::make(std::vector<Point2> control_points) -> Result<BezierCurve2> {
    if (control_points.empty()) {
        return make_error<BezierCurve2>(MathError::domain_error);
    }
    std::vector<Rational> xs;
    std::vector<Rational> ys;
    xs.reserve(control_points.size());
    ys.reserve(control_points.size());
    for (const Point2& p : control_points) {
        xs.push_back(p.x);
        ys.push_back(p.y);
    }
    auto cx = BezierCurve::make(std::move(xs));
    auto cy = BezierCurve::make(std::move(ys));
    if (!cx || !cy) {
        return make_error<BezierCurve2>(MathError::domain_error);
    }
    return BezierCurve2{std::move(*cx), std::move(*cy)};
}

auto BezierCurve2::control_points() const -> std::vector<Point2> {
    const std::span<const Rational> xs = x_.control_points();
    const std::span<const Rational> ys = y_.control_points();
    std::vector<Point2> out(xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i) {
        out[i] = Point2{.x = xs[i], .y = ys[i]};
    }
    return out;
}

auto BezierCurve2::evaluate(const Rational& t) const -> Result<Point2> {
    auto px = x_.evaluate(t);
    auto py = y_.evaluate(t);
    if (!px || !py) {
        return make_error<Point2>(px ? py.error() : px.error());
    }
    return Point2{.x = *px, .y = *py};
}

auto BezierCurve2::elevate() const -> Result<BezierCurve2> {
    auto ex = x_.elevate();
    auto ey = y_.elevate();
    if (!ex || !ey) {
        return make_error<BezierCurve2>(ex ? ey.error() : ex.error());
    }
    return BezierCurve2{std::move(*ex), std::move(*ey)};
}

// --- B-spline Cox-de Boor ---------------------------------------------------

namespace {

// N_{i,p}(u) by the Cox-de Boor recursion (The NURBS Book, Algorithm A2.4 "OneBasisFun"),
// exact over Q. U is the full knot vector (indices 0..m), p the degree, i the basis index
// (0..m-p-1). Divisions with a zero denominator correspond to zero basis contributions and
// are skipped (the 0/0 := 0 convention), so no division_by_zero can escape.
[[nodiscard]] auto one_basis(std::span<const Rational> U, std::size_t p, std::size_t i,
                             const Rational& u) -> Result<Rational> {
    const std::size_t m = U.size() - 1;
    const std::size_t last = m - p - 1;  // last basis / control-point index
    // Endpoint special cases. A CLAMPED end (U[0]==U[p] on the left, U[m]==U[m-p] on the
    // right) makes the curve interpolate that end control point, with N==1 exactly there.
    // This shortcut is ONLY valid for a clamped end: for an OPEN (non-clamped) knot vector
    // the true value at U[0]/U[m] follows from the local-support recursion below, so we must
    // not short-circuit. Unconditionally returning 1 was wrong for open vectors, e.g.
    // U={0,1,2,3,4}, p=1: N_{0,1}(0) is 0 (the (u-U0)/(U1-U0) factor vanishes), not 1.
    const bool left_clamped = (U[0] == U[p]);
    const bool right_clamped = (U[m] == U[m - p]);
    if ((i == 0 && left_clamped && u == U[0]) ||
        (i == last && right_clamped && u == U[m])) {
        return rat_one();
    }
    // Local support: N_{i,p}(u) = 0 outside [U_i, U_{i+p+1}).
    auto c1 = rat_cmp(u, U[i]);
    auto c2 = rat_cmp(u, U[i + p + 1]);
    if (!c1 || !c2) {
        return make_error<Rational>(c1 ? c2.error() : c1.error());
    }
    if (*c1 < 0 || *c2 >= 0) {
        return rat_zero();
    }
    // Zeroth-degree functions over the p+1 relevant spans.
    std::vector<Rational> N(p + 1);
    for (std::size_t j = 0; j <= p; ++j) {
        auto a = rat_cmp(u, U[i + j]);
        auto b = rat_cmp(u, U[i + j + 1]);
        if (!a || !b) {
            return make_error<Rational>(a ? b.error() : a.error());
        }
        N[j] = (*a >= 0 && *b < 0) ? rat_one() : rat_zero();
    }
    // Triangular table up to degree p.
    for (std::size_t k = 1; k <= p; ++k) {
        Rational saved = rat_zero();
        if (!N[0].is_zero()) {
            auto den = U[i + k].subtract(U[i]);  // nonzero: N[0] != 0 => nonempty span
            auto unum = u.subtract(U[i]);
            auto prod = (unum && den) ? unum->multiply(N[0]) : make_error<Rational>(MathError::overflow);
            auto s = (prod && den) ? prod->divide(*den) : make_error<Rational>(MathError::overflow);
            if (!s) {
                return make_error<Rational>(s.error());
            }
            saved = *s;
        }
        for (std::size_t j = 0; j + k <= p; ++j) {
            const Rational& u_left = U[i + j + 1];
            const Rational& u_right = U[i + j + k + 1];
            if (N[j + 1].is_zero()) {
                N[j] = saved;
                saved = rat_zero();
            } else {
                auto den = u_right.subtract(u_left);  // nonzero when N[j+1] != 0
                auto temp = den ? N[j + 1].divide(*den) : make_error<Rational>(den.error());
                auto ru = u_right.subtract(u);
                auto addend = (temp && ru) ? ru->multiply(*temp) : make_error<Rational>(MathError::overflow);
                auto nj = addend ? saved.add(*addend) : make_error<Rational>(addend.error());
                auto ul = u.subtract(u_left);
                auto sv = (temp && ul) ? ul->multiply(*temp) : make_error<Rational>(MathError::overflow);
                if (!nj || !sv) {
                    return make_error<Rational>(MathError::overflow);
                }
                N[j] = *nj;
                saved = *sv;
            }
        }
    }
    return N[0];
}

}  // namespace

auto BSpline::make(std::vector<Rational> knots, std::vector<Rational> control_points,
                   std::size_t degree) -> Result<BSpline> {
    if (degree < 1 || control_points.empty()) {
        return make_error<BSpline>(MathError::domain_error);
    }
    // U.size() == P.size() + p + 1.
    if (knots.size() != control_points.size() + degree + 1) {
        return make_error<BSpline>(MathError::domain_error);
    }
    auto nd = non_decreasing(knots);
    if (!nd) {
        return make_error<BSpline>(nd.error());
    }
    if (!*nd) {
        return make_error<BSpline>(MathError::domain_error);
    }
    BSpline out;
    out.knots_ = std::move(knots);
    out.control_ = std::move(control_points);
    out.degree_ = degree;
    return out;
}

auto BSpline::basis_function(std::size_t i, const Rational& u) const -> Result<Rational> {
    if (i >= control_.size()) {
        return make_error<Rational>(MathError::domain_error);
    }
    return one_basis(knots_, degree_, i, u);
}

auto BSpline::evaluate(const Rational& u) const -> Result<Rational> {
    // Domain check: u in [U_p, U_{m-p}].
    auto lo = rat_cmp(u, domain_min());
    auto hi = rat_cmp(u, domain_max());
    if (!lo || !hi) {
        return make_error<Rational>(lo ? hi.error() : lo.error());
    }
    if (*lo < 0 || *hi > 0) {
        return make_error<Rational>(MathError::domain_error);
    }
    Rational acc{};
    for (std::size_t i = 0; i < control_.size(); ++i) {
        auto ni = one_basis(knots_, degree_, i, u);
        if (!ni) {
            return make_error<Rational>(ni.error());
        }
        auto term = ni->multiply(control_[i]);
        if (!term) {
            return make_error<Rational>(term.error());
        }
        auto s = acc.add(*term);
        if (!s) {
            return make_error<Rational>(s.error());
        }
        acc = *s;
    }
    return acc;
}

// --- NURBS ------------------------------------------------------------------

auto NurbsCurve::make(std::vector<Rational> knots, std::vector<Rational> control_points,
                      std::vector<Rational> weights, std::size_t degree) -> Result<NurbsCurve> {
    if (weights.size() != control_points.size()) {
        return make_error<NurbsCurve>(MathError::domain_error);
    }
    for (const Rational& w : weights) {
        if (w.is_zero()) {
            return make_error<NurbsCurve>(MathError::domain_error);  // zero weight ill-posed
        }
    }
    auto s = BSpline::make(std::move(knots), std::move(control_points), degree);
    if (!s) {
        return make_error<NurbsCurve>(s.error());
    }
    return NurbsCurve{std::move(*s), std::move(weights)};
}

auto NurbsCurve::evaluate(const Rational& u) const -> Result<Rational> {
    const std::span<const Rational> cp = spline_.control_points();
    const std::span<const Rational> kn = spline_.knots();
    // Domain check via the underlying B-spline domain.
    auto lo = rat_cmp(u, spline_.domain_min());
    auto hi = rat_cmp(u, spline_.domain_max());
    if (!lo || !hi) {
        return make_error<Rational>(lo ? hi.error() : lo.error());
    }
    if (*lo < 0 || *hi > 0) {
        return make_error<Rational>(MathError::domain_error);
    }
    Rational numer{};
    Rational denom{};
    for (std::size_t i = 0; i < cp.size(); ++i) {
        auto ni = one_basis(kn, spline_.degree(), i, u);
        if (!ni) {
            return make_error<Rational>(ni.error());
        }
        auto wn = weights_[i].multiply(*ni);  // w_i N_i
        if (!wn) {
            return make_error<Rational>(wn.error());
        }
        auto dsum = denom.add(*wn);
        if (!dsum) {
            return make_error<Rational>(dsum.error());
        }
        denom = *dsum;
        auto wnp = wn->multiply(cp[i]);  // w_i N_i P_i
        if (!wnp) {
            return make_error<Rational>(wnp.error());
        }
        auto nsum = numer.add(*wnp);
        if (!nsum) {
            return make_error<Rational>(nsum.error());
        }
        numer = *nsum;
    }
    if (denom.is_zero()) {
        return make_error<Rational>(MathError::domain_error);
    }
    return numer.divide(denom);
}

}  // namespace nimblecas
