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


if __name__ == "__main__":
    raise SystemExit(main())
