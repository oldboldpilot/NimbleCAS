# NimbleCAS Documentation

**Author:** Olumuyiwa Oluwasanmi

Welcome to the documentation hub for **NimbleCAS** — a high-performance,
modern Computer Algebra System in **C++23**. It combines exact symbolic algebra
(Joel S. Cohen's algorithms) with a high-performance numeric path (multi-register
SIMD, dense polynomial arithmetic) on a foundation of immutable copy-on-write
expression trees, railway-oriented error handling (`std::expected`), and
deterministic TBB/PPL parallelism. This page links to every documentation topic
in the repository.

## Overview & product

| Document | What it covers |
| :--- | :--- |
| [Product Requirement Document](PRD.md) | Product scope, vision, and mathematical/algorithmic goals. |
| [Roadmap](ROADMAP.md) | Technical architecture, module structures, and the implementation plan. |
| [README](../README.md) | Project summary, current status, and build entry points. |
| [Agent guide](../AGENTS.md) | Conventions, policy pointers, and workflow for AI coding agents contributing to this repo. |

## Getting started

| Document | What it covers |
| :--- | :--- |
| [Quickstart](QUICKSTART.md) | Prerequisites, build & test on Linux/macOS and Windows, the uv Python environment. |

## Architecture

| Document | What it covers |
| :--- | :--- |
| [Architecture overview](architecture/overview.md) | The module graph, COW data model, error model, `import std` strategy, and parallel strategy. |
| [Parallel tree computation](architecture/parallel-tree-computation.md) | Why tree manipulation parallelizes by construction; fork–join, grain control, determinism, hash-consing. |

## Module reference

The symbolic chain (`core → symbolic → {simplify, cache} → diff → vectorcalc`):

| Module | Reference | Summary |
| :--- | :--- | :--- |
| `nimblecas.core` | [core.md](reference/core.md) | `MathError`/`Result`, `make_error`, `CowPtr<T>`, `cache_line_size`. |
| `nimblecas.symbolic` | [symbolic.md](reference/symbolic.md) | Immutable `Expr` trees, node kinds, structural equality/hashing, `free_of`, `substitute`. |
| `nimblecas.simplify` | [simplify.md](reference/simplify.md) | Cohen automatic simplification (ASAE): folding, identities, canonical order, like-term combination. |
| `nimblecas.expand` | [expand.md](reference/expand.md) | Algebraic expansion (§7.20): distribute products over sums and expand integer powers of sums via the multinomial theorem (capped), then simplify — the distributing companion `simplify` deliberately omits. |
| `nimblecas.cache` | [cache.md](reference/cache.md) | `ExprMemo` sharded concurrent hash-consing / memoization. |
| `nimblecas.diff` | [diff.md](reference/diff.md) | Symbolic differentiation with an elementary + special-function derivative table. |
| `nimblecas.series` | [series.md](reference/series.md) | Taylor series over `diff` + `simplify`: `taylor_coefficients` (`c_k = f^(k)(point)/k!`) and the truncated `taylor_polynomial`, overflow-checked `k!`. |
| `nimblecas.laplace` | [laplace.md](reference/laplace.md) | Table-driven symbolic Laplace transform `L{f(t)} = F(s)` by linearity over the elementary forms, simplified; `not_implemented` off-table. |
| `nimblecas.vectorcalc` | [vectorcalc.md](reference/vectorcalc.md) | Vector calculus over `diff`: gradient, divergence, curl, Laplacian, Jacobian, Hessian, directional / total derivatives. |
| `nimblecas.limits` | [limits.md](reference/limits.md) | Symbolic limits over `diff` + `simplify`: continuity substitution, iterated L'Hôpital for `0/0`, rational functions at `±∞`; a narrow decidable class — anything undecidable returns an honest `MathError`. |
| `nimblecas.tensor` | [tensor.md](reference/tensor.md) | Tensor calculus / differential geometry over `Q`: Christoffel symbols, covariant derivative, Riemann/Ricci/Einstein tensors, scalar curvature, geodesics, Laplace–Beltrami — exact symbolic via `diff`+`simplify`. |
| `nimblecas.forms` | [forms.md](reference/forms.md) | Exterior calculus over `Q`: p-forms, wedge `∧`, exterior derivative `d` (with `d²=0`), Hodge star `⋆` (Euclidean; `not_implemented` off it), interior product, closed/exact. |
| `nimblecas.reader` | [reader.md](reference/reader.md) | Text → `Expr` parser (the eval surface): precedence-climbing, exact-only (rationals never reals), round-trips `to_string`; malformed input → honest `MathError::syntax_error`. |
| `nimblecas.latex` | [latex.md](reference/latex.md) | Precedence-aware LaTeX math export: `to_latex(Expr)` rendering `\frac`/`\sqrt`, Greek letters, and function control words. |

The runtime and numeric chain (`core → simd → polynomial → {polyexpr, ratpoly → {pfd → ratint, resultant → rothstein} → integrate}}`; `parallel`):

| Module | Reference | Summary |
| :--- | :--- | :--- |
| `nimblecas.parallel` | [parallel.md](reference/parallel.md) | Deterministic fork–join over TBB/PPL/serial; order-preserving tree combinators. |
| `nimblecas.simd` | [simd.md](reference/simd.md) | Runtime-dispatched elementwise `float32` SIMD kernels (AVX-512 → AVX2 → scalar). |
| `nimblecas.gpu` | [gpu.md](reference/gpu.md) | Optional CUDA GPU acceleration (opt-in `-DNIMBLECAS_CUDA=ON`): batch polynomial evaluation on the device, plus a portable Triton kernel. |
| `nimblecas.polynomial` | [polynomial.md](reference/polynomial.md) | Dense univariate `int64` polynomials: ring ops, gcd, square-free factorization, SIMD batch eval. |
| `nimblecas.ratpoly` | [ratpoly.md](reference/ratpoly.md) | Exact `Rational` and dense polynomials over `Q[x]`: division-with-remainder, monic Euclidean gcd. |
| `nimblecas.polyexpr` | [polyexpr.md](reference/polyexpr.md) | Bridge between `Expr` and `Polynomial`; polynomial gcd / square-free factor at the `Expr` level. |
| `nimblecas.pfd` | [pfd.md](reference/pfd.md) | Square-free partial-fraction decomposition over `Q[x]`: Yun factorization, Bezout split, base-`b` power expansion. |
| `nimblecas.ratint` | [ratint.md](reference/ratint.md) | Hermite reduction of `int A/B dx` over `Q`: exact rational part plus a square-free-denominator logarithmic integrand. |
| `nimblecas.resultant` | [resultant.md](reference/resultant.md) | Resultant and discriminant over `Q[x]` via the Euclidean remainder sequence: common-factor / repeated-root detection. |
| `nimblecas.rothstein` | [rothstein.md](reference/rothstein.md) | Rothstein–Trager logarithmic integration over `Q(x)`: the residue resultant `R(t) = res_x(D, A − t·D')`, rational-residue logarithms of a square-free-denominator integrand. |
| `nimblecas.integrate` | [integrate.md](reference/integrate.md) | Rational-function integration capstone over `Q(x)`: Hermite reduction then Rothstein–Trager, assembling `int A/B dx = rational part + sum of residue-weighted logarithms`. |
| `nimblecas.symint` | [symint.md](reference/symint.md) | Expr-level symbolic integration (§7.19): linearity, power rule, an elementary table (exp/sin/cos/atan/asin/…) with linear substitution, and a bridge to the exact rational integrator; every antiderivative differentiates back to the integrand, else `not_implemented`. |
| `nimblecas.matrix` | [matrix.md](reference/matrix.md) | Dense matrices over exact `Rational`: add/multiply/transpose/trace, exact determinant, `A x = b` solve, inverse, and rank via Gaussian / Gauss-Jordan elimination over `Q`. |
| `nimblecas.kronprod` | [kronprod.md](reference/kronprod.md) | Structured matrix products over exact `Rational` (§7.2): Kronecker product / sum, direct sum, Hadamard product, and column-major `vec`/`unvec` — all exact, with the mixed-product and `vec(AXB)=(Bᵀ⊗A)vec(X)` identities. |
| `nimblecas.qrschur` | [qrschur.md](reference/qrschur.md) | QR decomposition and real Schur form (§7.2): exact Gram–Schmidt orthogonal factorization over `Q` (QᵀQ diagonal, unit-diagonal R), plus numeric Householder QR and Francis real-Schur (orthes+hqr2) tagged numeric; residual-checkable, non-convergence → honest error. |
| `nimblecas.cheigen` | [cheigen.md](reference/cheigen.md) | Eigenvalues of complex matrices (§7.2) via the real 2n embedding: Hermitian → exact-real spectrum, skew-Hermitian → exact purely-imaginary via the `iM`-Hermitian reduction, general non-conjugate-closed → numeric; a conjugate-closed spectrum (unrecoverable from the embedding) → honest `not_implemented`. |
| `nimblecas.combinatorics` | [combinatorics.md](reference/combinatorics.md) | Overflow-checked `int64` counting — factorial, binomial, permutations, Catalan, Fibonacci, Stirling numbers — plus exact-`Rational` Bernoulli numbers (Akiyama–Tanigawa, `B_1 = -1/2`). |
| `nimblecas.orthopoly` | [orthopoly.md](reference/orthopoly.md) | Classical orthogonal polynomials over `Q[x]` (Chebyshev T/U, Legendre, Laguerre, physicists'/probabilists' Hermite) via their three-term recurrences. |
| `nimblecas.roots` | [roots.md](reference/roots.md) | Rational roots of a polynomial over `Q[x]` with multiplicity, via the rational root theorem plus deflation (the exact/closed-form layer beneath `solve`). |
| `nimblecas.factor` | [factor.md](reference/factor.md) | Exact factorization of a polynomial over `Q` into irreducibles (§7.17): Yun square-free factorization then Kronecker's algorithm; feeds `solve` so reducible high-degree polynomials split into radical-solvable pieces. |
| `nimblecas.solve` | [solve.md](reference/solve.md) | Analytical equation solving (§7.21): exact closed-form roots by radicals for degree ≤ 4 (linear/quadratic/Cardano/Ferrari) after rational-root peeling and `factor` factorization over `Q`; only genuinely irreducible degree ≥ 5 factors resolve to companion-matrix eigenvalues via `numeigen`, each root tagged exact vs numeric. |
| `nimblecas.recurrence` | [recurrence.md](reference/recurrence.md) | Linear homogeneous constant-coefficient recurrences: characteristic polynomial over `Q[x]` and its rational roots via `roots` (rational-root case; irrational e.g. Fibonacci deferred). |
| `nimblecas.complex` | [complex.md](reference/complex.md) | Exact complex numbers over `Q` — the Gaussian rationals `Q + Qi`: overflow-checked add/subtract/multiply/divide/conjugate/reciprocal and the exact squared modulus (modulus and argument omitted as irrational). |
| `nimblecas.algnum` | [algnum.md](reference/algnum.md) | Exact arithmetic in a simple algebraic extension field `Q(α) = Q[x]/(m)` for monic irreducible `m` (§7.1): represents irrational/complex algebraic numbers (√2, i, ∛2) exactly; extended-Euclid inverse, `norm`/`trace` via the multiplication map, irreducibility enforced at construction. The substrate for Jordan form over an extension field. |
| `nimblecas.stats` | [stats.md](reference/stats.md) | Exact descriptive statistics over the rationals: mean, sample/population variance and covariance, and the symmetric covariance matrix `Σ` (returned as a `nimblecas.matrix` `Matrix`, its diagonal each variable's variance). |
| `nimblecas.lp` | [lp.md](reference/lp.md) | Exact-rational linear programming via single-phase Simplex: `maximize(A, b, c)` for `max c·x s.t. A x <= b, x >= 0` (`b >= 0`), Bland's rule anti-cycling, exact optimum / unbounded detection. |
| `nimblecas.ipm` | [ipm.md](reference/ipm.md) | Interior-point LP (§7.22): Mehrotra predictor–corrector primal–dual path-following for `min c·x s.t. A x = b, x >= 0` — the numerical (double-precision) companion to the exact Simplex, gap-certified to tolerance. |
| `nimblecas.probdist` | [probdist.md](reference/probdist.md) | Probability distribution catalog (§7.7): exact symbolic MGF/PGF, mean, variance, moments and cumulants (via symbolic differentiation) for the standard families, plus Markov/Chebyshev/Cantelli/Chernoff tail inequalities. |
| `nimblecas.statinfer` | [statinfer.md](reference/statinfer.md) | Inferential statistics & regression (§7.7.5): exact-rational OLS / ridge / weighted least squares via the normal equations, R², coefficient covariance, and a linear-fractional method of moments. |
| `nimblecas.hyptest` | [hyptest.md](reference/hyptest.md) | Hypothesis tests & MLE (§7.7): exact-rational χ²/F/ANOVA statistics and t²/z² exact squares (no fabricated p-values — critical-value comparison only), plus symbolic maximum-likelihood models (log-likelihood, score, Fisher information) and exact rational point estimates. |
| `nimblecas.probmethod` | [probmethod.md](reference/probmethod.md) | The Probabilistic Method (§7.18, Alon–Spencer): non-constructive existence proofs — first-moment / union-bound / second-moment arguments, the Lovász Local Lemma (symmetric via a rigorous rational enclosure `e_lo < e < e_hi` for a sound 3-way verdict; asymmetric exact), and BigInt Ramsey lower bounds. Sound certificates only; honest `not_certified`/`indeterminate`. |
| `nimblecas.numeric` | [numeric.md](reference/numeric.md) | Floating-point polynomial root-finders (Newton, bisection, secant) with Horner `eval` / `eval_derivative`; standalone numeric solver depending only on `core`. |

The wide-arithmetic tower — lifting the `int64` overflow ceiling (`int64 Rational → int128 → bigint → bigrational`; `bigfloat`/`doubledouble`; big-backed consumers):

| Module | Reference | Summary |
| :--- | :--- | :--- |
| `nimblecas.int128` | [int128.md](reference/int128.md) | Native `__int128` `Int128` + `Rational128` (~1.7·10³⁸), overflow-checked via `__builtin_*_overflow` — the tier between `int64` `Rational` and `BigInt`. |
| `nimblecas.bigint` | [bigint.md](reference/bigint.md) | Arbitrary-precision `BigInt` (sign-magnitude base-2³² limbs, Knuth Algorithm-D division) with `gcd`/`pow`/`modpow` for cryptography. |
| `nimblecas.bigrational` | [bigrational.md](reference/bigrational.md) | Exact unbounded rational `num`/`den` over `BigInt`, gcd-reduced with `den > 0`; infallible add/sub/mul, guarded divide/reciprocal/pow. |
| `nimblecas.bigdecimal` | [bigdecimal.md](reference/bigdecimal.md) | Exact base-10 scaled decimal (`BigInt` unscaled + `int32` scale) — the money type & boundary quantizer: semantic scale (`2.50` ≠ `2.5` by representation, `==` by value), seven explicit `Rounding` modes, no ambient context, `divide_exact` → `inexact` on non-terminating quotients. |
| `nimblecas.bigfloat` | [bigfloat.md](reference/bigfloat.md) | Arbitrary-precision software binary float on a `BigInt` mantissa; correctly-rounded add/mul/div/sqrt with guard/round/sticky ties-to-even. |
| `nimblecas.doubledouble` | [doubledouble.md](reference/doubledouble.md) | ~106-bit extended-precision float via error-free transforms; SIMD-batched `dd_sum`/`dd_dot`/`dd_poly` with bit-identical scalar == SIMD. |
| `nimblecas.bigcombinatorics` | [bigcombinatorics.md](reference/bigcombinatorics.md) | `BigInt`-backed factorial/binomial/multinomial/Catalan/Fibonacci/Stirling/Bell — the overflow-free counterpart to `combinatorics`. |
| `nimblecas.bigpowerseries` | [bigpowerseries.md](reference/bigpowerseries.md) | `Q[[x]]/(xᴺ)` truncated power series over `BigRational` — the exact/unbounded mirror of `powerseries`. |
| `nimblecas.bigmatrix` | [bigmatrix.md](reference/bigmatrix.md) | Dense matrix over `BigRational` with exact fraction-free **Bareiss** determinant + `from_matrix` promotion of an `int64` `Matrix`. |
| `nimblecas.bigeigen` | [bigeigen.md](reference/bigeigen.md) | Exact characteristic polynomial (Faddeev–LeVerrier) + rational eigenvalues over `BigRational` on `bigmatrix`; irrational/complex eigenvalues honestly omitted. |

Reasoning & algorithmics (search / logic / constraints on the `parallel` runtime; regular workloads GPU-offloadable, irregular ones CPU/distributed):

| Module | Reference | Summary |
| :--- | :--- | :--- |
| `nimblecas.search` | [search.md](reference/search.md) | Graph/tree search (BFS, DFS, IDDFS, Dijkstra, A*, tabu) + dynamic programming (edit-distance/LCS/knapsack) in recursive, iterative, and parallel forms. |
| `nimblecas.sat` | [sat.md](reference/sat.md) | Boolean SAT: DPLL, CDCL (1-UIP learning), WalkSAT, GSAT, `solve_portfolio`, and distributed `solve_shard`. Complete solvers are worst-case exponential. |
| `nimblecas.csp` | [csp.md](reference/csp.md) | Constraint satisfaction: AC-3 arc consistency, backtracking, forward checking, parallel. |
| `nimblecas.logic` | [logic.md](reference/logic.md) | Logic programming: unification + SLD resolution + OR-parallel search under a depth/step budget (semi-decidable). |
| `nimblecas.bitset` | [bitset.md](reference/bitset.md) | Word-parallel branchless fixed-capacity bitset — the CPU substrate for branchless/GPU-style regular workloads. |
| `nimblecas.bitcsp` | [bitcsp.md](reference/bitcsp.md) | Branchless bitset-domain AC-3 + three-bitmask N-queens (parallel result == serial). |

Differential equations (exact power-series over `Q`, except `sde` which is numerical):

| Module | Reference | Summary |
| :--- | :--- | :--- |
| `nimblecas.ode` | [ode.md](reference/ode.md) | Exact power-series solution of first-order ODE **systems** + higher-order via companion-system reduction. |
| `nimblecas.dde` | [dde.md](reference/dde.md) | Delay differential equations via the exact method of steps (piecewise polynomial per delay interval). |
| `nimblecas.sde` | [sde.md](reference/sde.md) | Stochastic DEs (Euler–Maruyama, Milstein, Heun, SRK, tamed Euler) on the counter-based RNG — **numerical**, per-path seeded for determinism. |
| `nimblecas.dae` | [dae.md](reference/dae.md) | Linear DAEs: index-1 semi-explicit + arbitrary-index via shuffle-algorithm index reduction, reduced onto `ode`. Exact over `Q`. |
| `nimblecas.pde` | [pde.md](reference/pde.md) | Linear (Cauchy–Kovalevskaya) + nonlinear (Adomian time-series) evolution PDEs + an exact 1-D Poisson/Dirichlet BVP. |
| `nimblecas.perturbation` | [perturbation.md](reference/perturbation.md) | ADM / HPM / HAM perturbation methods, exact in `Q[[x]]/(xᴺ)` on the `powerseries` substrate. |
| `nimblecas.pdeham` | [pdeham.md](reference/pdeham.md) | Homotopy Analysis Method with the ℏ convergence-control parameter for semilinear evolution PDEs `u_t = ν u_xx + c·u u_x + Σ a_p u^p` (§7): exact truncated `(Q[x])[[t]]` series; ℏ=−1 reduces to the ADM/HPM forward series, general ℏ selects a distinct exact deformation member. |
| `nimblecas.singpert` | [singpert.md](reference/singpert.md) | Singular perturbation (§7.4): leading-order matched asymptotic expansion (outer/inner/uniform composite) for the constant-coefficient boundary-layer problem `ε y'' + a y' + b y = 0`, `a > 0`. |

Additional linear algebra, numerics & simulation:

| Module | Reference | Summary |
| :--- | :--- | :--- |
| `nimblecas.cmatrix` | [cmatrix.md](reference/cmatrix.md) | Dense matrices over the exact Gaussian rationals (`Complex` entries). |
| `nimblecas.matdecomp` | [matdecomp.md](reference/matdecomp.md) | LU decomposition + matrix structure predicates. |
| `nimblecas.bandsolve` | [bandsolve.md](reference/bandsolve.md) | Thomas + banded-LU direct solvers for tridiagonal/banded systems + parallel multi-RHS batch. |
| `nimblecas.matexp` | [matexp.md](reference/matexp.md) | Matrix exponential via Taylor / Padé / scaling-and-squaring. |
| `nimblecas.eigen` | [eigen.md](reference/eigen.md) | Characteristic polynomial + rational eigenvalues/eigenvectors over `Q` (the `int64` tier `bigeigen` big-backs). |
| `nimblecas.frobenius` | [frobenius.md](reference/frobenius.md) | Rational (Frobenius) canonical form over `Q` (§7.2): invariant factors and minimal polynomial via the Smith normal form of `xI − A` over `Q[x]`, and the block-companion RCF — EXACT (needs no eigenvalues, unlike Jordan form); the transforming `P` is deliberately not returned. |
| `nimblecas.jordan` | [jordan.md](reference/jordan.md) | Jordan canonical form `A = P J P⁻¹` WITH the transforming `P` (§7.2): exact over `Q` when the characteristic polynomial splits (generalized-eigenvector chains), and over a quadratic extension `Q(α)` (via `algnum`) for a conjugate pair — e.g. `[[0,-1],[1,0]] → diag(i,-i)` over `Q(i)`; every `(J,P)` is verified `A·P == P·J` before return, degree-≥3 factors → honest `not_implemented`. |
| `nimblecas.numeigen` | [numeigen.md](reference/numeigen.md) | Numeric all-eigenvalue solver for real matrices: structure-aware dispatch (diagonal/triangular direct, Jacobi for symmetric, Francis double-shift real-Schur QR for general); `companion_eigenvalues` is the numeric polynomial root path under `solve`. |
| `nimblecas.dynamics` | [dynamics.md](reference/dynamics.md) | Equilibria, exact Routh–Hurwitz asymptotic stability, and rational equilibrium classification. |
| `nimblecas.powerseries` | [powerseries.md](reference/powerseries.md) | `Q[[x]]/(xᴺ)` truncated power series over `int64` `Rational`. |
| `nimblecas.laurent` | [laurent.md](reference/laurent.md) | Truncated Laurent series over `Q` (§7): a finite principal part plus a truncated Taylor tail with order-tracked arithmetic; exact `valuation`/`residue`, series inversion, and rational-function expansion about a pole. |
| `nimblecas.pade` | [pade.md](reference/pade.md) | Padé `[m/n]` rational approximant of a power series (exact-rational Toeplitz solve). |
| `nimblecas.contfrac` | [contfrac.md](reference/contfrac.md) | Continued fractions (§7): simple CF of a rational (Euclidean) with convergents, the eventually-periodic CF of √D, and the Viskovatov C-fraction of a power series — exact over `Q`/`ℤ`, honest `not_implemented` on a zero pivot. |
| `nimblecas.constants` | [constants.md](reference/constants.md) | Arbitrary-precision π/e/γ/ln2/ln10/√2/golden/Catalan as `BigFloat`. |
| `nimblecas.symconst` | [symconst.md](reference/symconst.md) | Named mathematical-constant `Expr` **leaves** (π, e, γ, φ) + a numeric-evaluation bridge to `constants`. |
| `nimblecas.numbertheory` | [numbertheory.md](reference/numbertheory.md) | Extended GCD, `mod_inverse`, Miller–Rabin, `next_prime`, CRT, Jacobi — crypto number-theory over `BigInt`. |
| `nimblecas.rng` | [rng.md](reference/rng.md) | Counter-based parallel RNG (Threefry) + splitmix64 seeding — partition-independent / thread-count-independent draws. |
| `nimblecas.mcmc` | [mcmc.md](reference/mcmc.md) | Metropolis–Hastings MCMC on the parallel RNG; per-chain seed `splitmix64(seed ^ chain)`. |
| `nimblecas.montecarlo` | [montecarlo.md](reference/montecarlo.md) | Monte Carlo integration / estimation + sample statistics — numerical. |
| `nimblecas.svgplot` | [svgplot.md](reference/svgplot.md) | In-core standalone SVG-string plotter (line/scatter/function; exact data → pixel mapping). |
| `nimblecas.quantum` | [quantum.md](reference/quantum.md) | Exact non-commutative operator algebra (ROADMAP §7.15): operators, commutators/Jacobi, Dirac bra/ket, dagger adjoint, `normal_form` canonicalizer. |

Applied linear algebra & operators (exact over `Q` unless noted):

| Module | Reference | Summary |
| :--- | :--- | :--- |
| `nimblecas.matstruct` | [matstruct.md](reference/matstruct.md) | Structured matrices + exact `LDL^T`/Cholesky/rational-Hessenberg factorizations; block builders & predicates. |
| `nimblecas.lie` | [lie.md](reference/lie.md) | Matrix Lie algebras: bracket, structure constants, Killing form, exponential map, truncated Lie transforms/series. |
| `nimblecas.krylov` | [krylov.md](reference/krylov.md) | Krylov methods: exact-over-`Q` conjugate gradient (SPD) + rational Arnoldi/Lanczos; numerical GMRES/MINRES/BiCGSTAB. |
| `nimblecas.semigroup` | [semigroup.md](reference/semigroup.md) | Functional analysis + C₀-semigroups (finite-dim): resolvent, spectrum, `e^{tA}`, Cauchy problem, Hille–Yosida, Sylvester/Lyapunov. |

Variational & analytical mechanics (exact symbolic):

| Module | Reference | Summary |
| :--- | :--- | :--- |
| `nimblecas.calcvar` | [calcvar.md](reference/calcvar.md) | Calculus of variations: Euler–Lagrange, Beltrami, holonomic/non-holonomic constraints, Lagrange multipliers. |
| `nimblecas.mechanics` | [mechanics.md](reference/mechanics.md) | Hamiltonian mechanics: Legendre transform, canonical equations, Poisson brackets, phase space, action-angle. |

Numerical methods & solvers:

| Module | Reference | Summary |
| :--- | :--- | :--- |
| `nimblecas.interpolation` | [interpolation.md](reference/interpolation.md) | Exact polynomial interpolation over `Q`: Lagrange, Newton divided differences, barycentric, Neville, Hermite. |
| `nimblecas.splines` | [splines.md](reference/splines.md) | Exact piecewise-polynomial geometry over `Q`: natural/clamped/periodic cubic splines, Hermite/PCHIP, Bézier (de Casteljau/Bernstein), B-splines (Cox–de Boor), rational-weight NURBS. |
| `nimblecas.optimize` | [optimize.md](reference/optimize.md) | Numerical unconstrained optimization: gradient descent, Newton, BFGS/L-BFGS, CG, Nelder–Mead, Kelley implicit filtering. |
| `nimblecas.nlsolve` | [nlsolve.md](reference/nlsolve.md) | Kelley iterative solvers for `F(x)=0`: Newton/Broyden/Newton–Krylov(JFNK)/Anderson/Levenberg–Marquardt (numerical). |
| `nimblecas.extrapolation` | [extrapolation.md](reference/extrapolation.md) | Sequence acceleration: Richardson, Romberg, Aitken Δ², Wynn ε — exact over `Q` for rational data, else numerical. |
| `nimblecas.pdenum` | [pdenum.md](reference/pdenum.md) | Numerical PDEs: FDM/FEM/FVM exact-over-`Q` discretizations + solves; method-of-lines; Crank–Nicolson (numerical). |
| `nimblecas.spectral` | [spectral.md](reference/spectral.md) | Spectral methods: exact Legendre/Chebyshev Galerkin + coefficient differentiation; numerical collocation/Fourier; DG/SEM. |
| `nimblecas.fft` | [fft.md](reference/fft.md) | Numeric O(n log n) FFT (§7.3): radix-2 Cooley–Tukey + Bluestein chirp-z for arbitrary lengths, inverse FFT, and FFT-based convolution — the numeric transform companion to the exact `spectral` tooling. |

Signal processing & uncertainty:

| Module | Reference | Summary |
| :--- | :--- | :--- |
| `nimblecas.wavelets` | [wavelets.md](reference/wavelets.md) | Wavelet transforms: Haar exact over `Q`; Daubechies/Symlets/Coiflets/biorthogonal/CWT numerical; DWT/SWT/packets/lifting. |
| `nimblecas.qmc` | [qmc.md](reference/qmc.md) | Quasi-Monte Carlo: exact low-discrepancy points (Van der Corput/Halton/Sobol/lattice) + RQMC + adaptive integration. |
| `nimblecas.compsense` | [compsense.md](reference/compsense.md) | Compressed sensing: exact-over-`Q` basis pursuit (via the LP dual), mutual coherence; numerical OMP/CoSaMP/IHT. |

Analysis, control & stochastic processes:

| Module | Reference | Summary |
| :--- | :--- | :--- |
| `nimblecas.analysis` | [analysis.md](reference/analysis.md) | Condition number (κ₁/κ_∞ exact), convergence tests, Lyapunov equation/stability (exact), Lyapunov exponent (numerical). |
| `nimblecas.control` | [control.md](reference/control.md) | Control systems: transfer-function/state-space, controllability/observability, Routh/Hurwitz/Kharitonov/Nyquist/Lyapunov stability, Bode. |
| `nimblecas.inteq` | [inteq.md](reference/inteq.md) | Integral equations: Fredholm/Volterra (linear + nonlinear), separable-kernel exact reduction, Neumann/Picard, ADM/HPM/HAM. |
| `nimblecas.stochastic` | [stochastic.md](reference/stochastic.md) | Stochastic processes: Markov-chain stationary distribution + hitting times (exact), WSS autocovariance, Yule–Walker, PSD. |
| `nimblecas.hmm` | [hmm.md](reference/hmm.md) | Discrete hidden Markov models exact over `Q`: Forward/Backward, posteriors (smoothing), Viterbi, Baum–Welch; `forward_scaled` numerical alternative; honest `overflow` at the int64 ceiling. |
| `nimblecas.smc` | [smc.md](reference/smc.md) | Sequential Monte Carlo (numerical): bootstrap particle filters, multinomial/systematic/stratified/residual resampling, ESS, variance reduction — deterministic via counter-based RNG, honest statistical estimates. |

Financial mathematics (exact-over-`Q` where closed-form, numerical/statistical where not):

| Module | Reference | Summary |
| :--- | :--- | :--- |
| `nimblecas.finance` | [finance.md](reference/finance.md) | The Excel/Mathematica/MATLAB/Maple financial suite with a **two-tier honesty contract**: Tier A exact over `Q` (PV/FV/PMT/IPMT/PPMT/CUMIPMT/CUMPRINC/NPV/FVSCHEDULE/depreciation/EFFECT/annuities) returning `Result<BigRational>`; Tier B numerical (NPER/RATE/IRR/XIRR/MIRR, bonds, rate roots) via a bracketed Brent solver returning `not_converged` not a bogus root — more accurate than Excel, never Excel-bit-identical; fluent `CashflowSchedule`/`TvmProblem` builders. |
| `nimblecas.currency` | [currency.md](reference/currency.md) | Exact FX: tagged `Money` (over `BigDecimal`), an exchange-rate **graph** with BFS cross-rate pathfinding, triangular-arbitrage detection (exact over `Q`), and covered-interest-parity forwards — a missing route → `not_implemented` (never a fabricated rate), cross-currency `Money` addition → `domain_error`. |
| `nimblecas.pricing` | [pricing.md](reference/pricing.md) | Derivatives pricing (numerical/statistical): Black-Scholes + full & extended Greeks (analytic oracle), Kamrad-Ritchken trinomial trees (European/American/Bermudan), reproducible partition-independent Monte Carlo (European + Asian with geometric control variate), Longstaff-Schwartz American MC, Black-76/digitals/barriers, composable `Portfolio`, and SVG plotting — MC returns estimate + standard error and is bit-reproducible under a fixed seed. |
| `nimblecas.analytics` | [analytics.md](reference/analytics.md) | Portfolio & risk analytics (numerical/statistical): returns, summary stats, Sharpe/Sortino/Treynor/information ratios, beta/alpha, drawdown, historical + Gaussian VaR/CVaR, and mean-variance optimization (min-variance/tangency/efficient frontier via Cholesky) — non-PD covariance → `domain_error`, division-by-zero-std refused, never `±inf`. |

Tooling, front-end & integration:

| Module | Reference | Summary |
| :--- | :--- | :--- |
| `nimblecas.testing` | [testing.md](reference/testing.md) | Internal, dependency-free test framework (`TestSuite`/`TestContext`) wired to ctest. |
| `nimblecas.webexport` | [webexport.md](reference/webexport.md) | JSON export bridge — serializes plot data + documents to the contract the `web/` front-end renders (the JSON analogue of `svgplot`). |
| `nimblecas.execdoc` | [execdoc.md](reference/execdoc.md) | Executable document engine (§7.13): runs ` ```nimblecas ` cells over `reader`+`simplify`+`diff`, incremental cache, HTML/MathJax render — a data bridge, honest exact results with captured cell errors. |
| `nimblecas.webkernel` | [webkernel.md](reference/webkernel.md) | Freestanding `wasm32` compute kernel (clang, no Emscripten) the browser front-end loads for live in-page sampling. See also [WASM build](architecture/wasm-build.md). |
| `nimblecas_ext` (Python) | [python-bindings.md](reference/python-bindings.md) | nanobind bindings: the `Expr` API, module functions, and `MathError`→exception translation. |

## Worked examples

Standalone executable documents (Markdown + ` ```nimblecas ` cells) that double as
integration tests — each is run and its rendered LaTeX asserted verbatim by a test file:

| Document | What it covers | Verified by |
| :--- | :--- | :--- |
| [PDE cross-method](examples/pde-multimethod.md) | `u_t = u_xx + u²`: exact closed form, FEM/FDM spatial cross-check, ADM == HPM. | `tests/execdoc_multimethod_tests.cpp`, `tests/pde_crossmethod_tests.cpp` |
| [Integral-equation cross-method](examples/inteq-multimethod.md) | Linear Volterra (Neumann == ADM == HPM == HAM) and a nonlinear Volterra ADM/HPM/HAM convergence check. | `tests/execdoc_multimethod_tests.cpp`, `tests/inteq_crossmethod_tests.cpp` |

## Testing

| Document | What it covers |
| :--- | :--- |
| [Sanitizers & memory-safety](testing/sanitizers.md) | ASan/UBSan/LSan/TSan/MSan + valgrind status and how to run each. |
| [Internal test framework](reference/testing.md) | The `nimblecas.testing` runner and ctest integration. |

## Module-dependency diagram

```
                nimblecas.core
                      │
      ┌───────────────┼───────────────────────────┐
      │               │                            │
nimblecas.simd  nimblecas.parallel          nimblecas.symbolic
      │               │        │                   │
      ▼               └────────┼──────────┐        │
nimblecas.polynomial           ▼          ▼        ▼
      │      │          nimblecas.cache  nimblecas.simplify
      ▼      ▼                 │                   │
nimblecas.  nimblecas.ratpoly  └────────┬──────────┤
 polyexpr                │              ▼          ▼
                         │       nimblecas.diff ◄──┘
                         │                 │
              ┌──────────┴──────────┐      ▼
              ▼                     ▼      nimblecas.vectorcalc
        nimblecas.pfd       nimblecas.resultant
              │                     │
              ▼                     ▼
        nimblecas.ratint    nimblecas.rothstein
              │                     │
              └──────────┬──────────┘
                         ▼
               nimblecas.integrate

nimblecas.matrix  nimblecas.combinatorics  nimblecas.orthopoly  nimblecas.roots
nimblecas.complex  nimblecas.lp   (ratpoly consumers; each depends on core + ratpoly)
nimblecas.stats     (matrix consumer; depends on core + ratpoly + matrix)
nimblecas.numeric   (standalone numeric root-finder; depends only on core)
nimblecas.testing   (stands alone)
nimblecas_ext       (nanobind: imports symbolic, simplify, diff, polyexpr)
nimblecas.gpu       (optional CUDA; depends on core — opt-in via -DNIMBLECAS_CUDA=ON)
```

The diagram above shows the original symbolic + numeric core. The later
subsystems layer on top of it along these roots:

- **Wide-arithmetic tower** — `int128 → bigint → bigrational → {bigfloat,
  bigcombinatorics, bigpowerseries, bigmatrix → bigeigen}`, each tier lifting the
  `int64` overflow ceiling of its `core`/`ratpoly`-based counterpart; `doubledouble`
  and `constants` sit alongside on `core`/`bigfloat`.
- **Differential equations** — `ode` builds on `powerseries`; `dde`/`dae`/`pde`/
  `perturbation` build on `ode`/`powerseries`/`ratpoly`; `sde`/`mcmc`/`montecarlo`
  build on the counter-based `rng`.
- **Reasoning & algorithmics** — `search`/`sat`/`csp`/`logic` and the branchless
  `bitset`/`bitcsp` build on `core` + the `parallel` fork–join runtime; each unit of
  work is a stateless pure function of `(problem, shard, seed)` for SGE/Ray/NCCL
  distribution.
- **Symbolic constants** — `symconst` bridges the `symbolic` `Expr` layer to the
  numeric `constants`.
- **Financial mathematics** — `bigdecimal` builds on `bigrational` as the exact
  money/quantizer type; `finance` and `currency` build on it (Tier-A exact over
  `Q`, quantizing to `BigDecimal` at the boundary; Tier-B numerical root-finding
  on `double`); `pricing` builds on the counter-based `rng` + `svgplot`
  (numerical/statistical), and `analytics` stands on `core` alone.

See the [architecture overview](architecture/overview.md) for the exact `import`
edges and the rationale.

## Build & test quickstart

```bash
# Linux/macOS — clang++-22 + libc++ + CMake >= 3.30 + Ninja
scripts/build.sh                         # configure, build, run tests (ctest)
NIMBLECAS_SANITIZE=ON scripts/build.sh   # ASan + UBSan + LSan build

# Windows — Visual Studio's bundled clang + MSVC STL (run in Git Bash)
scripts/build_win.sh

# Python bindings (Linux) — uv-managed dependencies
scripts/setup_python.sh                  # uv sync -> .venv + nanobind
```

Full details are in the [Quickstart](QUICKSTART.md). All code adheres to
`config/cpp_details.txt` (the authoritative code policy); every major change is
adversarially code-reviewed before it lands.
