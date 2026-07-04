# A nonlinear Volterra equation: Neumann, ADM, HPM, HAM

phi(x) = f(x) + integral_0^x phi(t)^2 dt has phi(x) = x as its exact fixed point when
f(x) = x - x^3/3 (since integral_0^x t^2 dt = x^3/3). We verify this algebraic
identity directly.

```nimblecas
f = x - x^3/3
check = simplify(f + x^3/3 - x)
```

For the LINEAR case phi(x) = 1 + integral_0^x phi(t) dt (i.e. phi' = phi, phi(0) = 1),
Neumann/Picard/ADM/HPM/HAM(hbar=-1) all return the exact Taylor series of e^x. The
degree-4 truncation phi_4 satisfies phi_4' = phi_4 only up to the missing next-order
term -- the residual is exactly that term, -x^4/24, a precise measure of the
truncation error rather than a vague "approximately":

```nimblecas
phi4 = 1 + x + x^2/2 + x^3/6 + x^4/24
phi4prime = diff(phi4, x)
residual4 = simplify(phi4prime - phi4)
```

The full Neumann == ADM == HPM == HAM(hbar=-1) cross-check (bit-identical rational
coefficients) and the nonlinear ADM/HPM/HAM convergence toward the exact phi(x) = x are
exact-rational tests in tests/inteq_crossmethod_tests.cpp; this document exercises the
same mathematics through the executable-document engine's symbolic (diff / simplify)
surface.

---

This document is executed and its rendered LaTeX output is asserted verbatim by
`tests/execdoc_multimethod_tests.cpp` (see `inteq_doc` and the
`inteq_document_renders_and_contains_latex` test case), so it is not just prose — it is a
live, checked executable document.
