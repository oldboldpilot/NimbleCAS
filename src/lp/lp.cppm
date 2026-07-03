// NimbleCAS exact-rational linear programming: the Simplex method (ROADMAP 7.22).
// @author Olumuyiwa Oluwasanmi
//
// Solves a standard-form linear program in *exact* rationals:
//
//     maximize  c . x   subject to  A x <= b,  x >= 0,   with b >= 0 given.
//
// Because every b_i >= 0, the slack basis (x = 0, slacks = b) is already feasible, so
// a single-phase Simplex suffices — there is no Phase I. The whole point of a rational
// Simplex is that pivoting is closed under exact arithmetic: unlike a floating-point
// tableau it accumulates no round-off, so an optimum reported as p/q is exactly p/q and
// the termination proof (Bland's rule) is not defeated by comparison noise.
//
// Entering variable: the smallest-index column with a strictly positive reduced cost
// (Bland's rule) — this anti-cycling choice guarantees termination. Leaving variable:
// the minimum-ratio row, ties broken again by smallest basic-variable index. A column
// with no positive entry certifies unboundedness. Every comparison keys off the sign
// of a numerator (denominators are kept positive and canonical by Rational), and every
// arithmetic step is overflow-checked and returned via Result (Rule 32).

export module nimblecas.lp;

import std;
import nimblecas.core;
import nimblecas.ratpoly;

export namespace nimblecas {

// Outcome classification for maximize(). A standard-form maximisation over a feasible,
// non-empty region is either bounded (optimal) or unbounded above.
enum class LpStatus : std::uint8_t { optimal, unbounded };

// The result of a solved LP. `solution` carries one value per *original* variable
// (slacks are internal and dropped) and is empty when the problem is unbounded, in
// which case `value` is left at its default 0/1 and is not meaningful.
struct LpSolution {
    LpStatus status;
    Rational value;
    std::vector<Rational> solution;
};

// Maximise c . x over { x : A x <= b, x >= 0 } with A an m x n matrix (m constraints,
// n variables), b of length m with every b_i >= 0, and c of length n.
//
// Domain errors (MathError::domain_error): a ragged A (some row's width != n), a b or c
// whose length disagrees with the derived dimensions, or any b_i < 0. Overflow in the
// exact rational tableau propagates as MathError::overflow.
[[nodiscard]] auto maximize(const std::vector<std::vector<Rational>>& A,
                            const std::vector<Rational>& b,
                            const std::vector<Rational>& c) -> Result<LpSolution>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

// A Rational is kept canonical with a strictly positive denominator, so the sign of the
// whole fraction is exactly the sign of its numerator — no arithmetic (and hence no
// overflow risk) is needed to test positivity or negativity.
[[nodiscard]] auto is_positive(const Rational& r) noexcept -> bool {
    return r.numerator() > 0;
}
[[nodiscard]] auto is_negative(const Rational& r) noexcept -> bool {
    return r.numerator() < 0;
}

// Three-way comparison of a and b, exact and overflow-checked. Both denominators are
// positive (Rational's canonical form), so a < b  <=>  a.num*b.den < b.num*a.den; the
// cross-multiplication cannot flip a sign the way it would with a negative denominator.
// Returns -1, 0, or +1 for a < b, a == b, a > b. (Rational exposes no operator<, so the
// ordering is reconstructed here rather than borrowed from the type.)
[[nodiscard]] auto compare(const Rational& a, const Rational& b) -> Result<int> {
    std::int64_t lhs = 0;
    std::int64_t rhs = 0;
    if (__builtin_mul_overflow(a.numerator(), b.denominator(), &lhs) ||
        __builtin_mul_overflow(b.numerator(), a.denominator(), &rhs)) {
        return make_error<int>(MathError::overflow);
    }
    if (lhs < rhs) {
        return -1;
    }
    if (lhs > rhs) {
        return 1;
    }
    return 0;
}

}  // namespace

auto maximize(const std::vector<std::vector<Rational>>& A, const std::vector<Rational>& b,
              const std::vector<Rational>& c) -> Result<LpSolution> {
    const std::size_t m = A.size();
    const std::size_t n = c.size();

    // --- Dimension and domain validation ---------------------------------------------
    if (b.size() != m) {
        return make_error<LpSolution>(MathError::domain_error);
    }
    for (const auto& row : A) {
        if (row.size() != n) {  // ragged / width mismatch against c
            return make_error<LpSolution>(MathError::domain_error);
        }
    }
    for (const auto& bi : b) {
        if (is_negative(bi)) {  // b >= 0 is required for the slack basis to be feasible
            return make_error<LpSolution>(MathError::domain_error);
        }
    }

    // --- Tableau construction ---------------------------------------------------------
    // Columns 0..n-1 are the original variables; columns n..n+m-1 are the slacks, laid
    // out as the identity so row i owns slack column n+i. total = n + m columns.
    const std::size_t total = n + m;
    const Rational zero{};
    const Rational one = Rational::from_int(1);

    std::vector<std::vector<Rational>> tab(m, std::vector<Rational>(total, zero));
    std::vector<Rational> rhs(m, zero);        // current value of each basic variable
    std::vector<std::size_t> basis(m, 0);      // basis[i] = column basic in row i
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            tab[i][j] = A[i][j];
        }
        tab[i][n + i] = one;   // slack coefficient
        rhs[i] = b[i];
        basis[i] = n + i;      // initial basis is the slacks
    }

    // Reduced costs. With the slack basis the basic costs are all zero, so the initial
    // reduced cost of column j is just c_j (0 for slacks); the objective value is 0.
    std::vector<Rational> reduced(total, zero);
    for (std::size_t j = 0; j < n; ++j) {
        reduced[j] = c[j];
    }
    Rational obj = zero;

    // --- Simplex iterations -----------------------------------------------------------
    for (;;) {
        // Entering column: smallest index with a strictly positive reduced cost
        // (Bland's rule). None => optimality.
        std::size_t entering = total;
        for (std::size_t j = 0; j < total; ++j) {
            if (is_positive(reduced[j])) {
                entering = j;
                break;
            }
        }
        if (entering == total) {
            break;  // optimal
        }

        // Leaving row: minimum of rhs[i]/tab[i][entering] over rows with a positive
        // pivot entry, ties broken by smallest basic-variable index (Bland's rule). If
        // no row has a positive entry the objective can grow without bound.
        std::size_t leaving = m;
        Rational best_ratio = zero;
        for (std::size_t i = 0; i < m; ++i) {
            if (!is_positive(tab[i][entering])) {
                continue;
            }
            auto ratio = rhs[i].divide(tab[i][entering]);  // tab[i][entering] != 0
            if (!ratio) {
                return make_error<LpSolution>(ratio.error());
            }
            if (leaving == m) {
                leaving = i;
                best_ratio = *ratio;
                continue;
            }
            auto cmp = compare(*ratio, best_ratio);
            if (!cmp) {
                return make_error<LpSolution>(cmp.error());
            }
            if (*cmp < 0 || (*cmp == 0 && basis[i] < basis[leaving])) {
                leaving = i;
                best_ratio = *ratio;
            }
        }
        if (leaving == m) {
            return LpSolution{.status = LpStatus::unbounded,
                              .value = zero,
                              .solution = {}};
        }

        // --- Pivot on (leaving, entering), all in exact rationals --------------------
        const Rational pivot = tab[leaving][entering];

        // 1. Normalise the pivot row so the pivot element becomes 1.
        for (std::size_t j = 0; j < total; ++j) {
            auto v = tab[leaving][j].divide(pivot);
            if (!v) {
                return make_error<LpSolution>(v.error());
            }
            tab[leaving][j] = *v;
        }
        {
            auto v = rhs[leaving].divide(pivot);
            if (!v) {
                return make_error<LpSolution>(v.error());
            }
            rhs[leaving] = *v;
        }

        // 2. Eliminate the entering column from every other row.
        for (std::size_t i = 0; i < m; ++i) {
            if (i == leaving) {
                continue;
            }
            const Rational factor = tab[i][entering];
            if (factor.is_zero()) {
                continue;
            }
            for (std::size_t j = 0; j < total; ++j) {
                auto term = factor.multiply(tab[leaving][j]);
                if (!term) {
                    return make_error<LpSolution>(term.error());
                }
                auto diff = tab[i][j].subtract(*term);
                if (!diff) {
                    return make_error<LpSolution>(diff.error());
                }
                tab[i][j] = *diff;
            }
            auto term = factor.multiply(rhs[leaving]);
            if (!term) {
                return make_error<LpSolution>(term.error());
            }
            auto diff = rhs[i].subtract(*term);
            if (!diff) {
                return make_error<LpSolution>(diff.error());
            }
            rhs[i] = *diff;
        }

        // 3. Eliminate the entering column from the reduced-cost row and lift the
        //    objective by (reduced cost) * (new value of the entering variable).
        const Rational rc = reduced[entering];
        if (!rc.is_zero()) {
            for (std::size_t j = 0; j < total; ++j) {
                auto term = rc.multiply(tab[leaving][j]);
                if (!term) {
                    return make_error<LpSolution>(term.error());
                }
                auto diff = reduced[j].subtract(*term);
                if (!diff) {
                    return make_error<LpSolution>(diff.error());
                }
                reduced[j] = *diff;
            }
            auto gain = rc.multiply(rhs[leaving]);
            if (!gain) {
                return make_error<LpSolution>(gain.error());
            }
            auto next = obj.add(*gain);
            if (!next) {
                return make_error<LpSolution>(next.error());
            }
            obj = *next;
        }

        // 4. Record the basis change.
        basis[leaving] = entering;
    }

    // --- Read the optimum -------------------------------------------------------------
    // Each original variable is either basic (its value is the rhs of its row) or
    // nonbasic (zero).
    std::vector<Rational> solution(n, zero);
    for (std::size_t i = 0; i < m; ++i) {
        if (basis[i] < n) {
            solution[basis[i]] = rhs[i];
        }
    }
    return LpSolution{.status = LpStatus::optimal, .value = obj, .solution = std::move(solution)};
}

}  // namespace nimblecas
