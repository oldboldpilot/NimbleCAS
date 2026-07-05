"""Smoke test for the NimbleCAS nanobind Python bindings.
@author Olumuyiwa Oluwasanmi
"""

import nimblecas_ext as ncas


def main() -> int:
    x = ncas.Expr.symbol("x")
    y = ncas.Expr.symbol("y")

    assert x.to_string() == "x"
    assert x == ncas.Expr.symbol("x")
    assert x != y

    # u = x^2 + x
    u = x.pow(ncas.Expr.integer(2)).add(x)
    assert not ncas.free_of(u, x), "x^2 + x contains x"
    assert ncas.free_of(u, ncas.Expr.symbol("z")), "x^2 + x is free of z"

    # substitute x -> y  =>  y^2 + y  (and u is unchanged)
    r = ncas.substitute(u, x, y)
    expected = y.pow(ncas.Expr.integer(2)).add(y)
    assert r.is_equivalent_to(expected), f"got {r.to_string()}"
    assert u.is_equivalent_to(x.pow(ncas.Expr.integer(2)).add(x)), "u must be unchanged"

    # operator sugar
    assert (x + y).is_equivalent_to(ncas.Expr.sum([x, y]))
    assert (x * y).is_equivalent_to(ncas.Expr.product([x, y]))

    # comparing against a non-Expr returns False, does not raise
    assert (x == 5) is False
    assert (x != 5) is True

    # __hash__ is consistent with structural equality: dedupes in a set
    assert hash(x) == hash(ncas.Expr.symbol("x"))
    assert len({x, ncas.Expr.symbol("x"), y}) == 2

    # automatic simplification: x + x -> 2*x, and x + 0 -> x
    assert ncas.simplify(x + x).is_equivalent_to(ncas.Expr.integer(2).mul(x))
    assert ncas.simplify(x + ncas.Expr.integer(0)).is_equivalent_to(x)

    # differentiation: d/dx x^2 = 2*x
    d = ncas.differentiate(x.pow(ncas.Expr.integer(2)), "x")
    assert d.is_equivalent_to(ncas.Expr.integer(2).mul(x)), d.to_string()

    # polynomial gcd: gcd(x^2 - 1, x - 1) is x - 1 (verify by simplifying the difference)
    x2m1 = x.pow(ncas.Expr.integer(2)).add(ncas.Expr.integer(-1))
    xm1 = x.add(ncas.Expr.integer(-1))
    g = ncas.polynomial_gcd(x2m1, xm1, "x")
    assert ncas.simplify(g.add(ncas.Expr.integer(1))).is_equivalent_to(x), g.to_string()

    # square-free factorization: (x+1)^2 -> one factor with multiplicity 2
    factors = ncas.square_free_factor(x.add(ncas.Expr.integer(1)).pow(ncas.Expr.integer(2)), "x")
    assert len(factors) == 1 and factors[0][1] == 2, str(factors)

    # rational: canonicalisation and zero-denominator error
    assert ncas.Expr.rational(2, 4).is_equivalent_to(ncas.Expr.rational(1, 2))
    try:
        ncas.Expr.rational(1, 0)
        raise AssertionError("expected division by zero to raise")
    except ValueError as exc:
        assert "division by zero" in str(exc), str(exc)

    _test_value_types()
    _test_solve_and_factor()
    _test_calculus()
    _test_linalg()
    _test_contfrac()
    _test_stats()
    _test_stats_widened()
    _test_definite_integral()
    _test_probdist()
    _test_qrschur()
    _test_cheigen()
    _test_laurent()
    _test_hyptest_widened()
    _test_gpu()

    print("python bindings OK:", r.to_string())
    return 0


def _test_value_types() -> None:
    """Rational and RationalPoly value types: arithmetic, equality, hashing."""
    R = ncas.Rational
    # 2/4 reduces to 1/2, canonical and equal.
    half = R.make(2, 4)
    assert half.numerator() == 1 and half.denominator() == 2, half.to_string()
    assert half == R.make(1, 2)
    assert hash(half) == hash(R.make(1, 2))
    assert len({half, R.make(1, 2), R.from_int(3)}) == 2

    # exact arithmetic: 1/2 + 1/3 == 5/6, and operator sugar agrees.
    assert half.add(R.make(1, 3)) == R.make(5, 6)
    assert (half + R.make(1, 3)) == R.make(5, 6)
    assert (half * R.from_int(4)) == R.from_int(2)
    assert (-half) == R.make(-1, 2)

    # zero-denominator raises at the boundary.
    try:
        R.make(1, 0)
        raise AssertionError("expected division by zero to raise")
    except ValueError as exc:
        assert "division by zero" in str(exc), str(exc)

    # comparing against a non-Rational returns False, does not raise.
    assert (half == 5) is False

    # RationalPoly: p = x^2 - 2, degree 2, leading coefficient 1.
    p = ncas.RationalPoly.from_coeffs([R.from_int(-2), R.from_int(0), R.from_int(1)])
    assert p.degree() == 2
    assert p.leading_coefficient() == R.from_int(1)
    assert p.coefficient(0) == R.from_int(-2)
    # divide x^2 - 2 by x gives quotient x, remainder -2.
    x_poly = ncas.RationalPoly.from_coeffs([R.from_int(0), R.from_int(1)])
    q, rem = p.divide(x_poly)
    assert q == x_poly
    assert rem == ncas.RationalPoly.constant(R.from_int(-2)), rem.to_string()


def _test_solve_and_factor() -> None:
    """solve_poly on x^2 - 2 (two exact roots); factor_over_Q on a reducible poly."""
    R = ncas.Rational
    # x^2 - 2: two exact (radical) roots.
    p = ncas.RationalPoly.from_coeffs([R.from_int(-2), R.from_int(0), R.from_int(1)])
    roots = ncas.solve_poly(p)
    assert len(roots) == 2, str([r.value.to_string() for r in roots])
    assert all(r.exact for r in roots), "sqrt(2) roots must be exact"

    # x^2 - 1 = (x - 1)(x + 1): two rational roots, each multiplicity 1.
    p2 = ncas.RationalPoly.from_coeffs([R.from_int(-1), R.from_int(0), R.from_int(1)])
    roots2 = ncas.solve_poly(p2)
    assert len(roots2) == 2 and all(r.exact for r in roots2)
    assert all(r.multiplicity == 1 for r in roots2), str(roots2)

    # factor x^2 - 1 over Q into two linear irreducibles.
    factors = ncas.factor_over_Q(p2)
    assert len(factors) == 2, str(factors)
    assert all(mult == 1 for _, mult in factors)
    assert all(fac.degree() == 1 for fac, _ in factors)


def _test_calculus() -> None:
    """expand of (a+b)^2 and a definite/indefinite integral."""
    a = ncas.Expr.symbol("a")
    b = ncas.Expr.symbol("b")
    x = ncas.Expr.symbol("x")

    # expand (a + b)^2 -> a^2 + 2ab + b^2 (verified structurally against the manual sum).
    expanded = ncas.expand((a + b).pow(ncas.Expr.integer(2)))
    manual = ncas.simplify(
        ncas.Expr.sum(
            [
                a.pow(ncas.Expr.integer(2)),
                ncas.Expr.integer(2).mul(a).mul(b),
                b.pow(ncas.Expr.integer(2)),
            ]
        )
    )
    assert expanded.is_equivalent_to(manual), expanded.to_string()

    # integrate x^2 dx -> x^3/3, and differentiating back recovers x^2.
    integ = ncas.integrate(x.pow(ncas.Expr.integer(2)), "x")
    back = ncas.differentiate(integ, "x")
    assert back.is_equivalent_to(x.pow(ncas.Expr.integer(2))), integ.to_string()


def _test_linalg() -> None:
    """Matrix determinant / inverse / solve, plus a Kronecker product."""
    R = ncas.Rational
    # A = [[1, 2], [3, 4]], det = -2.
    A = ncas.Matrix.from_rows(
        [[R.from_int(1), R.from_int(2)], [R.from_int(3), R.from_int(4)]]
    )
    assert A.rows() == 2 and A.cols() == 2
    assert A.determinant() == R.from_int(-2), A.determinant().to_string()
    assert A.rank() == 2

    # A * A^{-1} == I.
    inv = A.inverse()
    prod = A.multiply(inv)
    ident = ncas.Matrix.identity(2)
    assert prod == ident, prod.to_string()

    # solve A x = b for b = [[1], [1]] and check A x == b.
    b = ncas.Matrix.from_rows([[R.from_int(1)], [R.from_int(1)]])
    sol = A.solve(b)
    assert A.multiply(sol) == b, sol.to_string()

    # Kronecker product of two 2x2s is 4x4.
    B = ncas.Matrix.from_rows(
        [[R.from_int(0), R.from_int(1)], [R.from_int(1), R.from_int(0)]]
    )
    kron = ncas.kronecker_product(A, B)
    assert kron.rows() == 4 and kron.cols() == 4, kron.to_string()
    # top-left block is A(0,0) * B == 1 * B, so entry (0,1) == B(0,1) == 1.
    assert kron.at(0, 1) == R.from_int(1), kron.to_string()


def _test_contfrac() -> None:
    """Continued fraction of a rational round-trips exactly."""
    R = ncas.Rational
    cf = ncas.cf_from_rational(R.make(415, 93))  # [4; 2, 6, 7]
    assert cf == [4, 2, 6, 7], str(cf)
    assert ncas.cf_reconstruct(cf) == R.make(415, 93)
    convs = ncas.cf_convergents(cf)
    assert convs[-1] == R.make(415, 93), convs[-1].to_string()

    # sqrt(2) = [1; (2)] : prefix [1], purely-periodic block [2].
    pcf = ncas.quadratic_irrational_cf(2)
    assert list(pcf.prefix) == [1], str(list(pcf.prefix))
    assert list(pcf.period) == [2], str(list(pcf.period))
    # A perfect square is not a quadratic irrational -> raises.
    try:
        ncas.quadratic_irrational_cf(9)
        raise AssertionError("expected perfect-square D to raise")
    except ValueError:
        pass

    # Viskovatov C-fraction of the geometric series {1,1,1,1} is (b0; a) = (1; 1, -1).
    scf = ncas.viskovatov([R.from_int(1), R.from_int(1), R.from_int(1), R.from_int(1)])
    assert scf.b0 == R.from_int(1), scf.b0.to_string()
    assert list(scf.a) == [R.from_int(1), R.make(-1, 1)], str([v.to_string() for v in scf.a])


def _test_stats() -> None:
    """A chi-squared statistic and an exact MLE point estimate."""
    R = ncas.Rational
    # Uniform expectation, all observed equal to expected -> statistic 0.
    observed = [R.from_int(10), R.from_int(10), R.from_int(10)]
    expected = [R.from_int(10), R.from_int(10), R.from_int(10)]
    chi = ncas.chi_squared_goodness_of_fit(observed, expected)
    assert chi.value == R.from_int(0), chi.value.to_string()
    assert chi.df1 == 2, chi.df1

    # A non-trivial chi-squared: O = [12, 8], E = [10, 10] -> (2^2/10)+(2^2/10) = 4/5.
    chi2 = ncas.chi_squared_goodness_of_fit(
        [R.from_int(12), R.from_int(8)], [R.from_int(10), R.from_int(10)]
    )
    assert chi2.value == R.make(4, 5), chi2.value.to_string()

    # Bernoulli MLE p-hat = sample mean of {1, 0, 1, 1} = 3/4.
    data = [R.from_int(1), R.from_int(0), R.from_int(1), R.from_int(1)]
    assert ncas.bernoulli_mle(data) == R.make(3, 4), ncas.bernoulli_mle(data).to_string()
    assert ncas.mean(data) == R.make(3, 4)

    # Symbolic MLE model exposes an Expr estimator (p-hat == the mean symbol m).
    model = ncas.bernoulli_mle_model()
    assert model.mle.is_equivalent_to(ncas.Expr.symbol("m")), model.mle.to_string()


def _test_stats_widened() -> None:
    """The widened exact descriptive statistics (weighted mean, moments, quantiles)."""
    R = ncas.Rational
    data = [R.from_int(1), R.from_int(2), R.from_int(3)]

    # weighted mean of {1,2,3} with weights {1,1,2} = (1 + 2 + 6)/4 = 9/4.
    wm = ncas.weighted_mean(data, [R.from_int(1), R.from_int(1), R.from_int(2)])
    assert wm == R.make(9, 4), wm.to_string()

    # second raw moment (1 + 4 + 9)/3 = 14/3; second central moment (population var) = 2/3.
    assert ncas.raw_moment(data, 2) == R.make(14, 3), ncas.raw_moment(data, 2).to_string()
    assert ncas.central_moment(data, 2) == R.make(2, 3), ncas.central_moment(data, 2).to_string()

    # type-7 median (p = 1/2) of {1,2,3,4} = 5/2; range of {1,5,3} = 4.
    q = ncas.quantile([R.from_int(1), R.from_int(2), R.from_int(3), R.from_int(4)], R.make(1, 2))
    assert q == R.make(5, 2), q.to_string()
    assert ncas.data_range([R.from_int(1), R.from_int(5), R.from_int(3)]) == R.from_int(4)

    # mode of {1,2,2,3} = 2; r^2 of perfectly correlated data = 1.
    assert ncas.mode([R.from_int(1), R.from_int(2), R.from_int(2), R.from_int(3)]) == R.from_int(2)
    r2 = ncas.pearson_correlation_squared(
        data, [R.from_int(2), R.from_int(4), R.from_int(6)]
    )
    assert r2 == R.from_int(1), r2.to_string()


def _test_definite_integral() -> None:
    """A definite integral: integral_0^2 3*x^2 dx = [x^3]_0^2 = 8 (exact)."""
    x = ncas.Expr.symbol("x")
    integrand = ncas.Expr.integer(3).mul(x.pow(ncas.Expr.integer(2)))  # 3*x^2
    result = ncas.integrate_definite(
        integrand, "x", ncas.Expr.integer(0), ncas.Expr.integer(2)
    )
    assert result.is_equivalent_to(ncas.Expr.integer(8)), result.to_string()


def _test_probdist() -> None:
    """The exact symbolic distribution catalog: DistInfo mean / variance / mgf."""
    lam = ncas.Expr.symbol("lambda")
    pois = ncas.probdist.poisson(lam)
    # Poisson mean == variance == lambda; both an MGF and a PGF exist.
    assert pois.mean.is_equivalent_to(lam), pois.mean.to_string()
    assert pois.variance.is_equivalent_to(lam), pois.variance.to_string()
    assert pois.mgf is not None
    assert pois.pgf is not None

    # A continuous family has no PGF; Student-t honestly reports no MGF (it does not exist).
    p = ncas.Expr.symbol("p")
    assert ncas.probdist.bernoulli(p).mgf is not None
    assert ncas.probdist.exponential(lam).pgf is None
    assert ncas.probdist.student_t(ncas.Expr.symbol("nu")).mgf is None


def _test_qrschur() -> None:
    """Exact orthogonal QR reconstructs A = Q*R exactly over Q."""
    R = ncas.Rational
    # A = [[1, 1], [0, 1]] is full rank, so the exact Gram-Schmidt QR exists.
    A = ncas.Matrix.from_rows(
        [[R.from_int(1), R.from_int(1)], [R.from_int(0), R.from_int(1)]]
    )
    qr = ncas.exact_orthogonal_qr(A)
    # Reconstruction is exact (not "up to rounding") because Q, R are exact Rationals.
    assert qr.q.multiply(qr.r) == A, qr.q.to_string()
    # R is upper-triangular with a unit diagonal.
    assert qr.r.at(0, 0) == R.from_int(1) and qr.r.at(1, 1) == R.from_int(1)
    assert qr.r.at(1, 0) == R.from_int(0), qr.r.to_string()

    # A rank-deficient matrix has no exact orthogonal QR (a zero pseudo-norm).
    singular = ncas.Matrix.from_rows(
        [[R.from_int(1), R.from_int(2)], [R.from_int(2), R.from_int(4)]]
    )
    try:
        ncas.exact_orthogonal_qr(singular)
        raise AssertionError("expected rank-deficient QR to raise")
    except ValueError:
        pass


def _test_cheigen() -> None:
    """Hermitian complex matrix has a real spectrum: [[2, i], [-i, 2]] -> {1, 3}."""
    C = ncas.Complex
    cm = ncas.ComplexMatrix.from_rows(
        [[C.from_int(2), C.i()], [C.i().negate(), C.from_int(2)]]
    )
    assert cm.is_hermitian(), cm.to_string()
    evs = ncas.hermitian_eigenvalues(cm)
    assert len(evs) == 2, str(evs)
    assert abs(evs[0] - 1.0) < 1e-9 and abs(evs[1] - 3.0) < 1e-9, str(evs)

    # complex_eigenvalues lifts the same real spectrum to complex (imag part exactly 0).
    cevs = ncas.complex_eigenvalues(cm)
    assert all(abs(z.imag) < 1e-12 for z in cevs), str(cevs)


def _test_laurent() -> None:
    """Laurent expansion of 1/x about 0 has valuation -1 and residue 1."""
    R = ncas.Rational
    num = ncas.RationalPoly.from_coeffs([R.from_int(1)])          # 1
    den = ncas.RationalPoly.from_coeffs([R.from_int(0), R.from_int(1)])  # x
    lau = ncas.Laurent.from_rational_function(num, den, R.from_int(0), 3)
    assert lau.valuation() == -1, lau.to_string()
    assert lau.residue() == R.from_int(1), lau.residue().to_string()
    # The principal part carries the x^{-1} term; the residue lives there.
    assert lau.principal_part().residue() == R.from_int(1)


def _test_hyptest_widened() -> None:
    """A variance-ratio F statistic and a chi-squared test of independence."""
    R = ncas.Rational
    # var(x) = 1, var(y) = 4 (sample), so F = 1/4 with df1 = df2 = 2.
    x = [R.from_int(1), R.from_int(2), R.from_int(3)]
    y = [R.from_int(2), R.from_int(4), R.from_int(6)]
    f = ncas.variance_ratio_f(x, y)
    assert f.value == R.make(1, 4), f.value.to_string()
    assert f.df1 == 2 and f.df2 == 2, str((f.df1, f.df2))

    # A perfectly homogeneous 2x2 table gives chi-squared 0 with df1 = 1.
    table = ncas.Matrix.from_rows(
        [[R.from_int(10), R.from_int(10)], [R.from_int(10), R.from_int(10)]]
    )
    ts = ncas.chi_squared_independence(table)
    assert ts.value == R.from_int(0), ts.value.to_string()
    assert ts.df1 == 1, ts.df1

    # A paired t^2 statistic is an exact rational (differences {1,1,1} have zero variance -> raises).
    try:
        ncas.paired_t_squared(
            [R.from_int(2), R.from_int(3), R.from_int(4)],
            [R.from_int(1), R.from_int(2), R.from_int(3)],
        )
        raise AssertionError("expected zero-variance paired t^2 to raise")
    except ValueError:
        pass


def _test_gpu() -> None:
    """GPU submodule: present only in a CUDA build (HAS_GPU); kernels skipped CPU-only."""
    if not ncas.HAS_GPU:
        return  # CPU-only build: the gpu submodule is absent by design.
    dc = ncas.gpu.device_count()
    assert isinstance(dc, int) and dc >= 0
    assert ncas.gpu.available() == (dc > 0)
    if dc == 0:
        return  # No device present: skip the kernel assertion so the test still passes.
    # p(x) = 1 + 2x evaluated at x = 3 is 7.
    out = ncas.gpu.poly_eval([1.0, 2.0], [3.0])
    assert abs(out[0] - 7.0) < 1e-9, str(out)


if __name__ == "__main__":
    raise SystemExit(main())
