// NimbleCAS differential forms / exterior calculus (ROADMAP 7).
// @author Olumuyiwa Oluwasanmi
//
// A DifferentialForm is a p-form over a fixed, ordered list of coordinate variable
// names `coords` (dimension n = coords.size()). It is stored sparsely as a map from
// the CANONICAL strictly-increasing index tuples (i_0 < i_1 < ... < i_{p-1}) in
// [0, n) to an exact symbolic Expr coefficient. General antisymmetry is DERIVED, not
// stored: querying an arbitrary index list canonicalises it (each transposition
// flips the sign, a repeated index makes the term identically zero), so only the
// n-choose-p basis components are ever materialised. Every coefficient is passed
// through nimblecas.simplify, so a form's stored components are simplified and any
// component that reduces to 0 is dropped (a form is zero iff its component map is
// empty). All arithmetic is EXACT symbolic — no floating point, no tolerance; the
// only failure channel is the railway (Result<T> / MathError, Rule 32).
//
// Provided operations:
//   * wedge                — graded-antisymmetric exterior product (degree p+q, the
//                            zero form when p+q > n). Sign = permutation parity of the
//                            merged index list; a shared index annihilates the term.
//   * exterior_derivative  — d raising a p-form to a (p+1)-form via nimblecas.diff:
//                            d(sum a_I dx^I) = sum_I sum_j (d a_I / d x_j) dx^j ^ dx^I.
//   * hodge_star           — metric Hodge dual. SCOPE / HONESTY BOUNDARY (below).
//   * interior_product     — contraction ι_V with a vector field V (degree p -> p-1).
//   * is_closed            — decides d w == 0 (sound and total).
//   * is_exact             — sound but INTENTIONALLY PARTIAL (see below).
//
// HODGE-STAR SCOPE (honesty boundary): hodge_star SOUNDLY supports ONLY the Euclidean
// / orthonormal metric, i.e. the identity matrix. hodge_star(w, metric) validates the
// supplied symbolic metric: if every entry simplifies exactly to the identity
// (metric[i][j] == 1 for i==j, 0 otherwise) it computes the exact Euclidean dual;
// for ANY other (e.g. general, curved, or even diagonal-non-unit) metric it returns
// MathError::not_implemented rather than emit a wrong dual. A general metric Hodge
// star needs sqrt(det g) and the inverse metric contracted against the Levi-Civita
// symbol; sqrt(det) is not representable soundly in the current simplifier's exact
// domain, so it is deliberately NOT faked here. Use hodge_star_euclidean(w) directly
// for the supported case. On the Euclidean metric the classical identities hold
// exactly: *(dx)=dy^dz, *(dx^dy)=dz, and **w = (-1)^{p(n-p)} w.
//
// IS_EXACT scope (honesty boundary): deciding whether a closed form is exact
// (w = d alpha) in general requires solving the coboundary equation / de Rham
// cohomology, which is not decided here. is_exact therefore returns Ok(true) ONLY for
// the identically-zero form (trivially exact) and MathError::not_implemented for every
// nonzero form — it NEVER returns a possibly-wrong true/false. Closedness (a necessary
// condition, sound and total) is available via is_closed.
//
// Bounds / errors: mismatched coordinate lists between two forms, an index >= n, or an
// index-tuple whose length != the form's degree are MathError::domain_error. A degree
// p > n (or a wedge/derivative whose result degree exceeds n) is the zero form, not an
// error. Coefficient overflow inside a Rational step propagates as MathError::overflow
// from the underlying simplify/diff railway.

export module nimblecas.forms;

import std;
import nimblecas.core;
import nimblecas.symbolic;
import nimblecas.diff;
import nimblecas.simplify;

export namespace nimblecas {

// ---------------------------------------------------------------------------
// DifferentialForm — a p-form over an ordered coordinate list, stored as the
// canonical strictly-increasing basis components with exact symbolic coefficients.
// ---------------------------------------------------------------------------
class DifferentialForm {
public:
    // Map from a canonical strictly-increasing index tuple to its coefficient.
    using ComponentMap = std::map<std::vector<std::size_t>, Expr>;

    // 0-form (scalar field) over the given coordinates.
    [[nodiscard]] static auto scalar(std::vector<std::string> coords, Expr value)
        -> Result<DifferentialForm>;

    // The identically-zero p-form (empty component map).
    [[nodiscard]] static auto zero(std::vector<std::string> coords, std::size_t degree)
        -> DifferentialForm;

    // Single basis component: coeff * dx_{indices[0]} ^ ... ^ dx_{indices.back()}.
    // The indices need not be sorted (antisymmetry supplies the sign) and a repeated
    // index yields the zero form. Out-of-range index -> domain_error.
    [[nodiscard]] static auto basis(std::vector<std::string> coords,
                                    std::vector<std::size_t> indices, Expr coeff)
        -> Result<DifferentialForm>;

    // General builder from a list of {index-tuple, coefficient} terms. Each tuple is
    // canonicalised (signed) and like tuples are summed; coefficients are simplified
    // and zero components dropped. Every tuple must have length == degree and indices
    // in [0, n); otherwise domain_error. Repeated indices in a tuple drop that term.
    [[nodiscard]] static auto from_components(
        std::vector<std::string> coords, std::size_t degree,
        std::vector<std::pair<std::vector<std::size_t>, Expr>> terms)
        -> Result<DifferentialForm>;

    [[nodiscard]] auto dimension() const noexcept -> std::size_t { return coords_.size(); }
    [[nodiscard]] auto degree() const noexcept -> std::size_t { return degree_; }
    [[nodiscard]] auto coordinates() const noexcept -> const std::vector<std::string>& {
        return coords_;
    }
    [[nodiscard]] auto components() const noexcept -> const ComponentMap& { return components_; }
    [[nodiscard]] auto is_zero() const noexcept -> bool { return components_.empty(); }

    // Coefficient of an arbitrary (possibly unsorted / repeated) index tuple, with the
    // antisymmetry sign applied. A repeated index or an absent component gives 0.
    // Wrong-length tuple or an index >= n is domain_error.
    [[nodiscard]] auto component(const std::vector<std::size_t>& indices) const -> Result<Expr>;

    [[nodiscard]] auto to_string() const -> std::string;

private:
    DifferentialForm(std::vector<std::string> coords, std::size_t degree, ComponentMap comps)
        : coords_(std::move(coords)), degree_(degree), components_(std::move(comps)) {}

    std::vector<std::string> coords_;
    std::size_t degree_{0};
    ComponentMap components_;
};

// Structural equality: same coordinates, same degree, and structurally-equal
// (already-simplified) components.
[[nodiscard]] auto operator==(const DifferentialForm& lhs, const DifferentialForm& rhs) -> bool;

// Exterior (wedge) product a ^ b. Requires identical coordinate lists (else
// domain_error). Degree p+q; the zero form when p+q > n. Graded-antisymmetric.
[[nodiscard]] auto wedge(const DifferentialForm& a, const DifferentialForm& b)
    -> Result<DifferentialForm>;

// Exterior derivative d, raising a p-form to a (p+1)-form (the zero form when p == n).
[[nodiscard]] auto exterior_derivative(const DifferentialForm& w) -> Result<DifferentialForm>;

// Euclidean / orthonormal Hodge dual (the supported metric — see file header). Maps a
// p-form to an (n-p)-form with the Levi-Civita sign of the (I, complement) split.
[[nodiscard]] auto hodge_star_euclidean(const DifferentialForm& w) -> Result<DifferentialForm>;

// Metric Hodge dual. SOUND ONLY for the identity metric: validates that `metric` is an
// n-by-n symbolic matrix that simplifies exactly to the identity, then delegates to
// hodge_star_euclidean; any other metric returns MathError::not_implemented (never a
// wrong dual). A wrong-sized metric is domain_error.
[[nodiscard]] auto hodge_star(const DifferentialForm& w,
                              const std::vector<std::vector<Expr>>& metric)
    -> Result<DifferentialForm>;

// Interior product (contraction) ι_V with a vector field V = (V^0, ..., V^{n-1}),
// lowering a p-form to a (p-1)-form. A 0-form contracts to the zero scalar. Requires
// field.size() == n (else domain_error).
[[nodiscard]] auto interior_product(const DifferentialForm& w, const std::vector<Expr>& field)
    -> Result<DifferentialForm>;

// True iff d w == 0 (sound and total).
[[nodiscard]] auto is_closed(const DifferentialForm& w) -> Result<bool>;

// Sound but partial (see file header): Ok(true) only for the zero form, otherwise
// MathError::not_implemented. Never returns a possibly-wrong verdict.
[[nodiscard]] auto is_exact(const DifferentialForm& w) -> Result<bool>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// -(a) as a symbolic factor.
[[nodiscard]] auto negate(const Expr& a) -> Expr {
    return Expr::product({Expr::integer(-1), a});
}

// A simplified coefficient counts as absent when it is structurally the integer 0.
[[nodiscard]] auto is_zero_expr(const Expr& e) -> bool {
    return e.is_equivalent_to(Expr::integer(0));
}

// Result of canonicalising an arbitrary index tuple to strictly-increasing order.
struct Canonical {
    std::vector<std::size_t> indices;  // sorted ascending
    int sign{1};                       // permutation parity applied to reach that order
    bool vanishes{false};              // a repeated index makes the whole term zero
};

// Sort `idx` ascending while tracking the permutation parity (product of adjacent
// transpositions == parity of the inversion count) and detecting any repeated index.
[[nodiscard]] auto canonicalize_indices(std::vector<std::size_t> idx) -> Canonical {
    int sign = 1;
    for (std::size_t i = 0; i < idx.size(); ++i) {
        for (std::size_t j = i + 1; j < idx.size(); ++j) {
            if (idx[i] == idx[j]) {
                return Canonical{.indices = {}, .sign = 0, .vanishes = true};
            }
            if (idx[i] > idx[j]) {
                sign = -sign;  // one inversion flips the orientation
            }
        }
    }
    std::ranges::sort(idx);
    return Canonical{.indices = std::move(idx), .sign = sign, .vanishes = false};
}

// Sign of the permutation that sorts a tuple of DISTINCT indices ascending (used for
// the wedge merge and the Hodge (I, complement) split). +1 / -1.
[[nodiscard]] auto permutation_sign(const std::vector<std::size_t>& idx) -> int {
    int sign = 1;
    for (std::size_t i = 0; i < idx.size(); ++i) {
        for (std::size_t j = i + 1; j < idx.size(); ++j) {
            if (idx[i] > idx[j]) {
                sign = -sign;
            }
        }
    }
    return sign;
}

}  // namespace

auto DifferentialForm::from_components(
    std::vector<std::string> coords, std::size_t degree,
    std::vector<std::pair<std::vector<std::size_t>, Expr>> terms) -> Result<DifferentialForm> {
    const std::size_t n = coords.size();
    ComponentMap accumulated;
    for (const auto& [raw_indices, coeff] : terms) {
        if (raw_indices.size() != degree) {
            return make_error<DifferentialForm>(MathError::domain_error);
        }
        for (const std::size_t idx : raw_indices) {
            if (idx >= n) {
                return make_error<DifferentialForm>(MathError::domain_error);
            }
        }
        const Canonical canon = canonicalize_indices(raw_indices);
        if (canon.vanishes) {
            continue;  // repeated index -> this term is identically zero
        }
        Expr signed_coeff = (canon.sign == 1) ? coeff : negate(coeff);
        auto it = accumulated.find(canon.indices);
        if (it == accumulated.end()) {
            accumulated.emplace(canon.indices, std::move(signed_coeff));
        } else {
            it->second = Expr::sum({it->second, std::move(signed_coeff)});
        }
    }

    // Simplify every coefficient and drop the ones that reduce to zero.
    ComponentMap out;
    for (const auto& [key, value] : accumulated) {
        auto simplified = simplify(value);
        if (!simplified) {
            return make_error<DifferentialForm>(simplified.error());
        }
        if (!is_zero_expr(*simplified)) {
            out.emplace(key, std::move(*simplified));
        }
    }
    return DifferentialForm(std::move(coords), degree, std::move(out));
}

auto DifferentialForm::scalar(std::vector<std::string> coords, Expr value)
    -> Result<DifferentialForm> {
    std::vector<std::pair<std::vector<std::size_t>, Expr>> terms;
    terms.emplace_back(std::vector<std::size_t>{}, std::move(value));
    return from_components(std::move(coords), 0, std::move(terms));
}

auto DifferentialForm::zero(std::vector<std::string> coords, std::size_t degree)
    -> DifferentialForm {
    return DifferentialForm(std::move(coords), degree, ComponentMap{});
}

auto DifferentialForm::basis(std::vector<std::string> coords, std::vector<std::size_t> indices,
                             Expr coeff) -> Result<DifferentialForm> {
    const std::size_t degree = indices.size();
    std::vector<std::pair<std::vector<std::size_t>, Expr>> terms;
    terms.emplace_back(std::move(indices), std::move(coeff));
    return from_components(std::move(coords), degree, std::move(terms));
}

auto DifferentialForm::component(const std::vector<std::size_t>& indices) const -> Result<Expr> {
    if (indices.size() != degree_) {
        return make_error<Expr>(MathError::domain_error);
    }
    const std::size_t n = coords_.size();
    for (const std::size_t idx : indices) {
        if (idx >= n) {
            return make_error<Expr>(MathError::domain_error);
        }
    }
    const Canonical canon = canonicalize_indices(indices);
    if (canon.vanishes) {
        return Expr::integer(0);
    }
    auto it = components_.find(canon.indices);
    if (it == components_.end()) {
        return Expr::integer(0);
    }
    return (canon.sign == 1) ? it->second : negate(it->second);
}

auto DifferentialForm::to_string() const -> std::string {
    if (components_.empty()) {
        return "0";
    }
    std::string out;
    bool first = true;
    for (const auto& [key, coeff] : components_) {
        if (!first) {
            out += " + ";
        }
        first = false;
        out += std::format("({})", coeff.to_string());
        for (const std::size_t idx : key) {
            out += std::format(" d{}", coords_[idx]);
        }
    }
    return out;
}

auto operator==(const DifferentialForm& lhs, const DifferentialForm& rhs) -> bool {
    return lhs.coordinates() == rhs.coordinates() && lhs.degree() == rhs.degree() &&
           lhs.components() == rhs.components();
}

auto wedge(const DifferentialForm& a, const DifferentialForm& b) -> Result<DifferentialForm> {
    if (a.coordinates() != b.coordinates()) {
        return make_error<DifferentialForm>(MathError::domain_error);
    }
    const std::size_t n = a.dimension();
    const std::size_t result_degree = a.degree() + b.degree();
    if (result_degree > n) {
        return DifferentialForm::zero(a.coordinates(), result_degree);  // identically zero
    }
    std::vector<std::pair<std::vector<std::size_t>, Expr>> terms;
    terms.reserve(a.components().size() * b.components().size());
    for (const auto& [ia, ca] : a.components()) {
        for (const auto& [ib, cb] : b.components()) {
            std::vector<std::size_t> merged = ia;
            merged.insert(merged.end(), ib.begin(), ib.end());
            terms.emplace_back(std::move(merged), Expr::product({ca, cb}));
        }
    }
    return DifferentialForm::from_components(a.coordinates(), result_degree, std::move(terms));
}

auto exterior_derivative(const DifferentialForm& w) -> Result<DifferentialForm> {
    const std::vector<std::string>& coords = w.coordinates();
    const std::size_t n = coords.size();
    const std::size_t result_degree = w.degree() + 1;

    std::vector<std::pair<std::vector<std::size_t>, Expr>> terms;
    for (const auto& [indices, coeff] : w.components()) {
        for (std::size_t j = 0; j < n; ++j) {
            auto derivative = differentiate(coeff, coords[j]);
            if (!derivative) {
                return make_error<DifferentialForm>(derivative.error());
            }
            if (is_zero_expr(*derivative)) {
                continue;
            }
            // dx^j ^ dx^I : prepend j; from_components supplies the reordering sign and
            // annihilates the term if j already occurs in I.
            std::vector<std::size_t> new_indices;
            new_indices.reserve(indices.size() + 1);
            new_indices.push_back(j);
            new_indices.insert(new_indices.end(), indices.begin(), indices.end());
            terms.emplace_back(std::move(new_indices), std::move(*derivative));
        }
    }
    return DifferentialForm::from_components(coords, result_degree, std::move(terms));
}

auto hodge_star_euclidean(const DifferentialForm& w) -> Result<DifferentialForm> {
    const std::vector<std::string>& coords = w.coordinates();
    const std::size_t n = coords.size();
    const std::size_t p = w.degree();
    if (p > n) {
        return DifferentialForm::zero(coords, 0);  // only the zero form exists in degree > n
    }
    const std::size_t result_degree = n - p;

    std::vector<std::pair<std::vector<std::size_t>, Expr>> terms;
    for (const auto& [indices, coeff] : w.components()) {
        // Complement J = [0, n) \ I, kept ascending.
        std::vector<std::size_t> complement;
        complement.reserve(result_degree);
        for (std::size_t k = 0; k < n; ++k) {
            if (!std::ranges::binary_search(indices, k)) {
                complement.push_back(k);
            }
        }
        // Levi-Civita sign of the ordered concatenation (I, J) as a permutation of
        // 0..n-1; I and J are individually sorted, so this is their interleave parity.
        std::vector<std::size_t> concat = indices;
        concat.insert(concat.end(), complement.begin(), complement.end());
        const int sign = permutation_sign(concat);
        Expr signed_coeff = (sign == 1) ? coeff : negate(coeff);
        terms.emplace_back(std::move(complement), std::move(signed_coeff));
    }
    return DifferentialForm::from_components(coords, result_degree, std::move(terms));
}

auto hodge_star(const DifferentialForm& w, const std::vector<std::vector<Expr>>& metric)
    -> Result<DifferentialForm> {
    const std::size_t n = w.dimension();
    if (metric.size() != n) {
        return make_error<DifferentialForm>(MathError::domain_error);
    }
    for (const auto& row : metric) {
        if (row.size() != n) {
            return make_error<DifferentialForm>(MathError::domain_error);
        }
    }
    // Supported scope: the identity metric only. Validate exactly, then delegate.
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            auto entry = simplify(metric[i][j]);
            if (!entry) {
                return make_error<DifferentialForm>(entry.error());
            }
            const Expr expected = Expr::integer(i == j ? 1 : 0);
            if (!entry->is_equivalent_to(expected)) {
                // Non-Euclidean metric: not faked (see file header honesty boundary).
                return make_error<DifferentialForm>(MathError::not_implemented);
            }
        }
    }
    return hodge_star_euclidean(w);
}

auto interior_product(const DifferentialForm& w, const std::vector<Expr>& field)
    -> Result<DifferentialForm> {
    const std::vector<std::string>& coords = w.coordinates();
    const std::size_t n = coords.size();
    if (field.size() != n) {
        return make_error<DifferentialForm>(MathError::domain_error);
    }
    const std::size_t p = w.degree();
    if (p == 0) {
        return DifferentialForm::zero(coords, 0);  // ι_V of a 0-form is 0
    }

    std::vector<std::pair<std::vector<std::size_t>, Expr>> terms;
    for (const auto& [indices, coeff] : w.components()) {
        for (std::size_t r = 0; r < indices.size(); ++r) {
            // ι_V(dx^{i_0} ^ ... ^ dx^{i_{p-1}})
            //   = sum_r (-1)^r V^{i_r} dx^{i_0} ^ ... ^ [omit i_r] ^ ... ^ dx^{i_{p-1}}.
            const std::size_t removed = indices[r];
            std::vector<std::size_t> rest;
            rest.reserve(indices.size() - 1);
            for (std::size_t s = 0; s < indices.size(); ++s) {
                if (s != r) {
                    rest.push_back(indices[s]);
                }
            }
            const Expr parity = Expr::integer((r % 2 == 0) ? 1 : -1);
            terms.emplace_back(std::move(rest),
                               Expr::product({parity, coeff, field[removed]}));
        }
    }
    return DifferentialForm::from_components(coords, p - 1, std::move(terms));
}

auto is_closed(const DifferentialForm& w) -> Result<bool> {
    auto dw = exterior_derivative(w);
    if (!dw) {
        return make_error<bool>(dw.error());
    }
    return dw->is_zero();
}

auto is_exact(const DifferentialForm& w) -> Result<bool> {
    if (w.is_zero()) {
        return true;  // the zero form is d(0): trivially exact
    }
    // General exactness (w = d alpha) is not decided here — see file header.
    return make_error<bool>(MathError::not_implemented);
}

}  // namespace nimblecas
