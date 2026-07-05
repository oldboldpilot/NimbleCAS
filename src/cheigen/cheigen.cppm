// NimbleCAS numeric eigenvalues of a COMPLEX matrix — the Hermitian / skew-Hermitian /
// unitary / normal companion to the real-only nimblecas.numeigen (ROADMAP 7.2 spectral).
// @author Olumuyiwa Oluwasanmi
//
// nimblecas.numeigen answers the same question for a REAL matrix. This module extends the
// numeric spectrum to an exact ComplexMatrix (entries in the Gaussian rationals Q + Qi)
// without writing a second QR: it uses the classical real-embedding trick. An n x n complex
// matrix M = A + iB (A, B real n x n) maps to the 2n x 2n REAL matrix
//
//         R = [ A  -B ]
//             [ B   A ]
//
// whose spectrum is exactly the spectrum of M together with its complex conjugate — i.e.
// every eigenvalue of M appears in R alongside its conjugate. Feeding R to
// nimblecas.numeigen::eigenvalues_qr therefore yields all 2n numbers, and M's own n
// eigenvalues are recovered from them (see below). No new eigensolver is introduced; the
// battle-tested real Jacobi / Francis QR of nimblecas.numeigen does all the arithmetic.
//
// HONESTY BOUNDARY (Rule 32). The eigenvalues of a general complex matrix are irrational /
// complex in general and have NO exact Gaussian-rational representation, so — exactly like
// nimblecas.numeigen — the results here are NUMERIC (std::complex<double>) approximations,
// never dressed up as exact. The one place exactness is lost is the entrywise conversion of
// each Gaussian-rational Complex to a double when building the embedding R; everything
// downstream is floating point.
//
// STRUCTURE is exploited for BOTH correctness and honesty, decided on the EXACT matrix
// before any float ever appears:
//   * Hermitian (verified by ComplexMatrix::is_hermitian on the exact entries) => the
//     spectrum is REAL. R is then symmetric, so numeigen's Jacobi path returns real
//     eigenvalues with imaginary part EXACTLY 0.0 (not "≈ 0"), each doubled; we de-duplicate
//     the doubled spectrum and return honest reals.
//   * skew-Hermitian => the spectrum is purely IMAGINARY; the recovered real parts are
//     snapped to EXACTLY 0.0 (justified because the structure was proven on the exact M).
//   * unitary => the eigenvalues lie on the unit circle |λ| = 1 (documented; not faked —
//     the magnitudes are the numeric ones the algorithm produced).
// A block that fails to converge surfaces as an honest MathError::not_implemented (inherited
// from numeigen), never a partial or invented spectrum.
//
// RECOVERY of M's n eigenvalues from R's 2n. R is real, so its spectrum is closed under
// conjugation: it is M's spectrum ⊎ conj(M's spectrum). To pick M's own half we pair each
// candidate with its conjugate partner and keep, from each pair, the representative that
// actually annihilates M's characteristic polynomial — the one minimizing |det(λI − M)|,
// evaluated in complex double arithmetic on M itself. This is what makes every returned λ
// satisfy det(λI − M) ≈ 0 (rather than det(λI − M̄) ≈ 0). The lone genuinely ambiguous case
// is a matrix whose EXACT spectrum is itself closed under conjugation (e.g. a real-spectrum
// or ±-pair spectrum): there both representatives are true eigenvalues, so any pick is a
// valid eigenvalue — the returned set still satisfies the characteristic equation.

export module nimblecas.cheigen;

import std;
import nimblecas.core;
import nimblecas.ratpoly;    // Rational (real/imag parts of a Complex)
import nimblecas.complex;    // Complex — the entry type
import nimblecas.cmatrix;    // ComplexMatrix — the input matrix and its exact predicates
import nimblecas.numeigen;   // eigenvalues_qr — the real eigensolver reused on the embedding

export namespace nimblecas {

// All (numeric) eigenvalues of a Hermitian complex matrix, returned REAL and ascending.
// The matrix is verified Hermitian on its EXACT Gaussian-rational entries first: a Hermitian
// matrix has an entirely real spectrum, so the result is std::vector<double> with no
// imaginary component at all. A NON-Hermitian matrix is rejected with domain_error (asking
// for the real Hermitian spectrum of a non-Hermitian matrix is a category error — use
// eigenvalues() instead). A non-square matrix is domain_error; a 0x0 matrix yields an empty
// vector. `tol` / `max_iter` are forwarded to nimblecas.numeigen; non-convergence surfaces
// as not_implemented. Eigenvalues are numeric approximations (see the honesty boundary).
[[nodiscard]] auto hermitian_eigenvalues(const ComplexMatrix& m, double tol = 1e-12,
                                         std::size_t max_iter = 1000)
    -> Result<std::vector<double>>;

// All (numeric) eigenvalues of a general complex matrix, as std::complex<double>. Dispatches
// on the exact structure for correctness and honesty: a Hermitian matrix returns a purely
// real spectrum (imaginary part exactly 0.0); a skew-Hermitian matrix returns a purely
// imaginary spectrum (real part exactly 0.0); every other matrix (unitary, normal, or wholly
// unstructured) goes through the real-embedding + conjugate-recovery path, and each returned
// λ satisfies det(λI − M) ≈ 0. A non-square matrix is domain_error; a 0x0 matrix yields an
// empty vector. `tol` / `max_iter` are forwarded to nimblecas.numeigen; a non-converging
// block surfaces as an honest not_implemented, never a partial spectrum.
[[nodiscard]] auto eigenvalues(const ComplexMatrix& m, double tol = 1e-12,
                               std::size_t max_iter = 1000)
    -> Result<std::vector<std::complex<double>>>;

}  // namespace nimblecas

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas {
namespace {

using cd = std::complex<double>;

// Exact-to-double projection of one Rational (this is where exactness ends). A canonical
// Rational has a non-zero denominator, so the division is well defined.
[[nodiscard]] auto to_double(const Rational& r) -> double {
    return static_cast<double>(r.numerator()) / static_cast<double>(r.denominator());
}

// Project a Complex entry to a std::complex<double>.
[[nodiscard]] auto to_cd(const Complex& z) -> cd {
    return cd{to_double(z.real()), to_double(z.imag())};
}

// Determinant of an n x n complex (double) matrix `a` (row-major, taken by value) by
// Gaussian elimination with partial pivoting. Used only to score how well a candidate λ
// satisfies M's characteristic equation, det(λI − M) ≈ 0; the magnitude, not the value, is
// what matters. A structurally singular column yields exactly 0.
[[nodiscard]] auto complex_det(std::vector<cd> a, std::size_t n) -> cd {
    cd det{1.0, 0.0};
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t piv = k;
        double best = std::abs(a[k * n + k]);
        for (std::size_t i = k + 1; i < n; ++i) {
            const double v = std::abs(a[i * n + k]);
            if (v > best) {
                best = v;
                piv = i;
            }
        }
        if (best == 0.0) {
            return cd{0.0, 0.0};  // singular: λ is (numerically) an exact eigenvalue
        }
        if (piv != k) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(a[piv * n + j], a[k * n + j]);
            }
            det = -det;
        }
        det *= a[k * n + k];
        for (std::size_t i = k + 1; i < n; ++i) {
            const cd f = a[i * n + k] / a[k * n + k];
            for (std::size_t j = k; j < n; ++j) {
                a[i * n + j] -= f * a[k * n + j];
            }
        }
    }
    return det;
}

// Build the 2n x 2n real embedding R = [[A, -B], [B, A]] (row-major) from M = A + iB, with
// A = Re(M), B = Im(M). Guards the 2n and (2n)^2 index arithmetic against size_t wrap.
[[nodiscard]] auto real_embedding(const ComplexMatrix& m)
    -> Result<std::vector<double>> {
    const std::size_t n = m.rows();
    const std::size_t smax = std::numeric_limits<std::size_t>::max();
    if (n > smax / 2) {
        return make_error<std::vector<double>>(MathError::overflow);
    }
    const std::size_t sz = 2 * n;
    if (sz != 0 && sz > smax / sz) {
        return make_error<std::vector<double>>(MathError::overflow);
    }
    std::vector<double> r(sz * sz, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const cd e = to_cd(m.at(i, j));
            const double a = e.real();
            const double b = e.imag();
            r[i * sz + j] = a;                     // top-left     A
            r[i * sz + (j + n)] = -b;              // top-right   -B
            r[(i + n) * sz + j] = b;               // bottom-left  B
            r[(i + n) * sz + (j + n)] = a;         // bottom-right A
        }
    }
    return r;
}

// From R's 2n eigenvalues, recover M's n by conjugate pairing + characteristic-residual
// selection. `mc` is M as complex doubles (row-major, n x n).
[[nodiscard]] auto recover_from_embedding(const std::vector<cd>& evs, std::size_t n,
                                          const std::vector<cd>& mc) -> std::vector<cd> {
    const std::size_t m = evs.size();  // == 2n
    // |det(λI − M)|: how well λ satisfies M's characteristic equation.
    auto residual = [&](cd lam) -> double {
        std::vector<cd> a(n * n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                a[i * n + j] = (i == j ? lam : cd{0.0, 0.0}) - mc[i * n + j];
            }
        }
        return std::abs(complex_det(std::move(a), n));
    };

    std::vector<cd> out;
    out.reserve(n);
    std::vector<char> used(m, 0);
    for (std::size_t i = 0; i < m; ++i) {
        if (used[i]) {
            continue;
        }
        // Find i's conjugate partner: the unused candidate closest to conj(evs[i]).
        const cd target = std::conj(evs[i]);
        std::size_t partner = m;
        double best = std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < m; ++j) {
            if (used[j] || j == i) {
                continue;
            }
            const double dist = std::abs(evs[j] - target);
            if (dist < best) {
                best = dist;
                partner = j;
            }
        }
        used[i] = 1;
        if (partner == m) {  // no partner left (unreachable for even m); keep self
            out.push_back(evs[i]);
            continue;
        }
        used[partner] = 1;
        // Keep the representative that best annihilates M's characteristic polynomial.
        out.push_back(residual(evs[i]) <= residual(evs[partner]) ? evs[i] : evs[partner]);
    }
    return out;
}

}  // namespace

auto hermitian_eigenvalues(const ComplexMatrix& m, double tol, std::size_t max_iter)
    -> Result<std::vector<double>> {
    if (!m.is_square()) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    const std::size_t n = m.rows();
    if (n == 0) {
        return std::vector<double>{};
    }
    // Honesty: only claim a real spectrum after proving the matrix Hermitian on exact entries.
    auto herm = m.is_hermitian();
    if (!herm) {
        return make_error<std::vector<double>>(herm.error());
    }
    if (!*herm) {
        return make_error<std::vector<double>>(MathError::domain_error);
    }
    // R is symmetric for a Hermitian M => numeigen's Jacobi path => real, exact-0 imaginary.
    auto r = real_embedding(m);
    if (!r) {
        return make_error<std::vector<double>>(r.error());
    }
    auto evs = eigenvalues_qr(*r, 2 * n, tol, max_iter);
    if (!evs) {
        return make_error<std::vector<double>>(evs.error());
    }
    // The 2n eigenvalues are the n real eigenvalues of M, each doubled. Sort and collapse
    // adjacent pairs (averaging the two floating-point estimates of each true eigenvalue).
    std::vector<double> reals;
    reals.reserve(2 * n);
    for (const cd& z : *evs) {
        reals.push_back(z.real());
    }
    std::ranges::sort(reals);
    std::vector<double> out;
    out.reserve(n);
    for (std::size_t k = 0; k + 1 < reals.size(); k += 2) {
        out.push_back(0.5 * (reals[k] + reals[k + 1]));
    }
    return out;
}

auto eigenvalues(const ComplexMatrix& m, double tol, std::size_t max_iter)
    -> Result<std::vector<cd>> {
    if (!m.is_square()) {
        return make_error<std::vector<cd>>(MathError::domain_error);
    }
    const std::size_t n = m.rows();
    if (n == 0) {
        return std::vector<cd>{};
    }

    // Hermitian => real spectrum: reuse the real path and lift to complex with exact-0 imag.
    auto herm = m.is_hermitian();
    if (!herm) {
        return make_error<std::vector<cd>>(herm.error());
    }
    if (*herm) {
        auto reals = hermitian_eigenvalues(m, tol, max_iter);
        if (!reals) {
            return make_error<std::vector<cd>>(reals.error());
        }
        std::vector<cd> out;
        out.reserve(reals->size());
        for (double e : *reals) {
            out.emplace_back(e, 0.0);
        }
        return out;
    }

    // General path: embed, solve the 2n x 2n real problem, recover M's own n eigenvalues.
    auto r = real_embedding(m);
    if (!r) {
        return make_error<std::vector<cd>>(r.error());
    }
    auto evs = eigenvalues_qr(*r, 2 * n, tol, max_iter);
    if (!evs) {
        return make_error<std::vector<cd>>(evs.error());
    }
    std::vector<cd> mc(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            mc[i * n + j] = to_cd(m.at(i, j));
        }
    }
    std::vector<cd> out = recover_from_embedding(*evs, n, mc);

    // skew-Hermitian => purely imaginary spectrum: snap the real parts to exactly 0.0
    // (justified: the structure was proven on the exact matrix, not merely observed).
    auto skew = m.is_skew_hermitian();
    if (!skew) {
        return make_error<std::vector<cd>>(skew.error());
    }
    if (*skew) {
        for (cd& z : out) {
            z = cd{0.0, z.imag()};
        }
    }
    return out;
}

}  // namespace nimblecas
