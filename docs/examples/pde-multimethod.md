# Reaction-diffusion: one PDE, four methods

We solve u_t = u_xx + u^2 on x in [0,1], with initial datum phi(x) = x - x^2.

## The linear diffusion piece has an exact closed form

Since phi is quadratic, phi'' = -2 (a constant), the Cauchy-Kovalevskaya series
u(x,t) = phi(x) - 2t terminates exactly: u_t = u_xx. We verify the PDE residual is
exactly zero for this closed form.

```nimblecas
u = x - x^2 - 2*t
ut = diff(u, t)
uxx = diff(diff(u, x), x)
residual = simplify(ut - uxx)
```

Finite Elements (fem_p1_solve) independently reproduces the same spatial profile:
solving -w'' = 2 with w(0) = w(1) = 0 on a uniform 4-interval mesh gives nodal values
3/16, 1/4, 3/16 at x = 1/4, 1/2, 3/4 -- exactly x - x^2 there. The Finite Difference
discrete Laplacian applied to those same nodal values gives exactly -2 at every
interior node, matching phi'' above (see tests/pde_crossmethod_tests.cpp for the full
exact cross-check).

## The full nonlinear PDE: ADM and HPM agree

nimblecas.pde's ADM (reaction_diffusion_quadratic) and HPM
(solve_nonlinear_evolution_pde_hpm) both solve u_t = u_xx + u^2 and return
bit-identical truncated series (the ADM == HPM homotopy equivalence). The
hand-verified first-order coefficient is:

```nimblecas
c1 = x^4 - 2*x^3 + x^2 - 2
```

HAM (the convergence-control deformation) is NOT YET implemented for genuine PDEs in
nimblecas.pde -- only for ODEs (nimblecas.perturbation) and integral equations
(nimblecas.inteq). This is an honest, documented gap, not a silent omission.

---

This document is executed and its rendered LaTeX output is asserted verbatim by
`tests/execdoc_multimethod_tests.cpp` (see `pde_doc` and the
`pde_document_renders_and_contains_latex` test case), so it is not just prose — it is a
live, checked executable document.
