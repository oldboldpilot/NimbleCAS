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
// before any float ever appears, via three disjoint dispatch paths:
//   * Hermitian (verified by ComplexMatrix::is_hermitian on the exact entries) => the
//     spectrum is REAL. R is then symmetric, so numeigen's Jacobi path returns real
//     eigenvalues with imaginary part EXACTLY 0.0 (not "≈ 0"), each doubled; we de-duplicate
//     the doubled spectrum and return honest reals.
//   * skew-Hermitian (is_skew_hermitian) => the spectrum is purely IMAGINARY, recovered
//     EXACTLY: iM is Hermitian when M is skew-Hermitian ((iM)† = -i M† = -i(-M) = iM), so we
//     run the Hermitian path on iM for its real eigenvalues μₖ and return λₖ = -iμₖ (real part
//     EXACTLY 0.0). This is unambiguous — it never touches the general embedding recovery.
//   * general (neither of the above; includes unitary and normal) => the eigenvalues are
//     recovered from R's spectrum by selecting the candidates that satisfy M's characteristic
//     equation, det(λI − M) ≈ 0. For a unitary M these additionally lie on the unit circle
//     |λ| = 1 (documented; not faked — the magnitudes are the numeric ones produced).
// A block that fails to converge surfaces as an honest MathError::not_implemented (inherited
// from numeigen), never a partial or invented spectrum.
//
// RECOVERY (general path) and its HONEST LIMIT. R is real, so spec(R) is closed under
// conjugation: spec(R) = spec(M) ⊎ conj(spec(M)). M's own half is the subset annihilating
// M's characteristic polynomial, so we select the R-eigenvalues with |det(λI − M)| ≈ 0. In
// GENERAL POSITION exactly n of the 2n candidates qualify and none is the conjugate of
// another — those n ARE spec(M), each satisfying det(λI − M) ≈ 0. But when M's EXACT spectrum
// is itself CLOSED UNDER CONJUGATION, the embedding is genuinely lossy: diag(i,i) and
// diag(i,−i) both produce R-spectrum {i,i,−i,−i}, so M is not recoverable from R alone. This
// case — which includes EVERY real matrix that has complex-conjugate eigenvalues (e.g.
// [[1,−2],[2,1]] with spectrum {1±2i}) — is detected (the selection has size ≠ n, or a
// selected non-real λ has its conjugate also selected) and returns an honest
// MathError::not_implemented. It NEVER returns a wrong multiset. For a REAL matrix with
// complex eigenvalues, call nimblecas.numeigen::eigenvalues_qr directly, which solves the
// real problem without the embedding's conjugate ambiguity.

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
// imaginary spectrum (real part exactly 0.0, via the exact iM-Hermitian reduction); every
// other matrix (unitary, normal, or wholly unstructured) is recovered from the real embedding
// by selecting the R-eigenvalues satisfying det(λI − M) ≈ 0. When that spectrum is closed
// under conjugation the embedding cannot distinguish M from M̄ — this INCLUDES every real
// matrix with complex-conjugate eigenvalues — and the function returns an honest
// MathError::not_implemented rather than a wrong multiset (use nimblecas.numeigen for real
// matrices). A non-square matrix is domain_error; a 0x0 matrix yields an empty vector. `tol`
// / `max_iter` are forwarded to nimblecas.numeigen; a non-converging block also surfaces as
// not_implemented, never a partial spectrum.
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

// Scale-free characteristic residual of a candidate λ against M: |det(λI − M)| divided by
// the Hadamard bound ∏ᵢ ‖rowᵢ(λI − M)‖₂ (which upper-bounds |det|). The quotient lies in
// [0, 1] and is ~machine-eps precisely when λI − M is (numerically) singular — i.e. when λ
// is a genuine eigenvalue of M — independent of the matrix's overall scale. `mc` is M as
// complex doubles (row-major, n x n).
[[nodiscard]] auto char_residual(cd lam, std::size_t n, const std::vector<cd>& mc) -> double {
    std::vector<cd> b(n * n);
    double hadamard = 1.0;
    for (std::size_t i = 0; i < n; ++i) {
        double row_norm2 = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            const cd v = (i == j ? lam : cd{0.0, 0.0}) - mc[i * n + j];
            b[i * n + j] = v;
            row_norm2 += std::norm(v);  // |v|^2
        }
        hadamard *= std::sqrt(row_norm2);
    }
    const double det = std::abs(complex_det(std::move(b), n));
    if (hadamard == 0.0) {
        return 0.0;  // a zero row of λI − M => exactly singular => λ is an eigenvalue
    }
    return det / hadamard;
}

// Recover M's n eigenvalues from R's 2n. R = [[A,−B],[B,A]] is real, so spec(R) is
// spec(M) ⊎ conj(spec(M)); M's own half is the subset annihilating M's characteristic
// polynomial. Select the candidates with normalized residual ≤ tol (counting multiplicity).
//
// Rule 32 honesty: the selection is trustworthy ONLY when it is unambiguous. Return the
// selected set exactly when it has size n AND is not conjugate-closed (no selected non-real
// λ has its conjugate also selected). Otherwise — the conjugate-closed case, in which R
// genuinely cannot distinguish M from M̄ (e.g. diag(i,i) and diag(i,−i) share the R-spectrum
// {i,i,−i,−i}), and which includes every real matrix with complex-conjugate eigenvalues —
// return an honest MathError::not_implemented rather than a wrong multiset.
[[nodiscard]] auto recover_general(const std::vector<cd>& evs, std::size_t n,
                                   const std::vector<cd>& mc, double tol)
    -> Result<std::vector<cd>> {
    std::vector<cd> selected;
    for (const cd& z : evs) {
        if (char_residual(z, n, mc) <= tol) {
            selected.push_back(z);
        }
    }
    // In general position exactly n of the 2n candidates satisfy det(λI − M) ≈ 0; any other
    // count is the ambiguous conjugate-closed situation (a real eigenvalue doubles, a
    // conjugate pair in spec(M) is selected on both sides, etc.).
    if (selected.size() != n) {
        return make_error<std::vector<cd>>(MathError::not_implemented);
    }
    // Reject a spectrum that is closed under conjugation: a genuine non-real λ whose conjugate
    // is also among the selected eigenvalues cannot be pinned to M rather than M̄.
    for (std::size_t i = 0; i < selected.size(); ++i) {
        const double mag = 1.0 + std::abs(selected[i]);
        if (std::abs(selected[i].imag()) <= 1e-9 * mag) {
            continue;  // real λ is its own conjugate; multiplicity is caught by the size test
        }
        const cd conj_i = std::conj(selected[i]);
        for (std::size_t j = 0; j < selected.size(); ++j) {
            if (j != i && std::abs(selected[j] - conj_i) <= 1e-9 * mag) {
                return make_error<std::vector<cd>>(MathError::not_implemented);
            }
        }
    }
    return selected;
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

    // skew-Hermitian => purely imaginary spectrum, recovered EXACTLY (no ambiguous embedding).
    // If M is skew-Hermitian then iM is Hermitian: (iM)† = -i M† = -i(-M) = iM. So the
    // Hermitian path gives the real eigenvalues μₖ of iM, and M's eigenvalues are λₖ = -iμₖ
    // (real part exactly 0). This sidesteps the conjugate-closed ambiguity of the general path
    // — e.g. diag(i,−i) maps to the Hermitian diag(1,−1), μ = {1,−1}, λ = {−i, i}.
    auto skew = m.is_skew_hermitian();
    if (!skew) {
        return make_error<std::vector<cd>>(skew.error());
    }
    if (*skew) {
        auto im = m.scale(Complex::i());  // iM, which is Hermitian
        if (!im) {
            return make_error<std::vector<cd>>(im.error());
        }
        auto mu = hermitian_eigenvalues(*im, tol, max_iter);
        if (!mu) {
            return make_error<std::vector<cd>>(mu.error());
        }
        std::vector<cd> out;
        out.reserve(mu->size());
        for (double e : *mu) {
            out.emplace_back(0.0, -e);  // λ = -i·μ : purely imaginary, real part exactly 0
        }
        return out;
    }

    // General path: embed, solve the 2n x 2n real problem, recover M's own n eigenvalues.
    // Recovery returns an honest not_implemented when the spectrum is conjugate-closed and
    // therefore genuinely unrecoverable from the real embedding (see recover_general).
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
    return recover_general(*evs, n, mc, tol);
}

}  // namespace nimblecas
