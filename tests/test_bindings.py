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

    # rational: canonicalisation and zero-denominator error
    assert ncas.Expr.rational(2, 4).is_equivalent_to(ncas.Expr.rational(1, 2))
    try:
        ncas.Expr.rational(1, 0)
        raise AssertionError("expected division by zero to raise")
    except ValueError as exc:
        assert "division by zero" in str(exc), str(exc)

    print("python bindings OK:", r.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
