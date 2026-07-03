# Product Requirement Document (PRD) — NimbleCAS

**Version:** 1.0.0  
**Date:** June 2026  
**Status:** Approved  
**Author:** Antigravity (Advanced Agentic Coding Team)

---

## 1. Executive Summary & Vision

**NimbleCAS** is a high-performance, modern Computer Algebra System (CAS) built for Windows using C++23, C++23 Modules, and the Parallel Patterns Library (PPL). It bridges the gap between pure symbolic mathematical manipulation and high-performance numeric execution by exploiting multi-core CPUs, multi-register SIMD vectorization, and GPU acceleration.

Unlike legacy systems which rely on heavyweight, single-threaded architectures and legacy C-style pointer structures, NimbleCAS uses **Copy-on-Write (COW) Expression Trees**, monadic error-handling via `std::expected`, and a compiler-optimized C++23 module dependency graph. The mathematical foundations of the system are based directly on the rigorous algebraic algorithms detailed in **Joel S. Cohen's** *Computer Algebra and Symbolic Computation: Elementary Algorithms* and *Mathematical Methods*.

Beyond deterministic symbolic algebra, NimbleCAS delivers a first-class **statistics, probability, and stochastic-processes engine** — from exact descriptive statistics and a symbolic distribution catalog through inferential modelling and high-performance Monte Carlo and stochastic-process simulation on the parallel CPU/GPU and distributed execution infrastructure.

---

## 2. Mathematical & Algorithmic Scope (Joel S. Cohen Guide)

NimbleCAS implements the core algebraic operations and representation styles defined by Joel S. Cohen. All expressions are represented in functional form (prefix representation) and manipulated via recursive mathematical algorithms.

### 2.1. Expression Representation
- **Tree-Structured Expressions**: All mathematical objects are represented as trees (or DAGs via Copy-on-Write sub-expressions).
  - **Kind**: Symbols (variables), Constants (integers, floats, rational numbers), and Operators/Functions.
  - **Functional Forms**: A sum $u + v$ is represented as `plus(u, v)`. A product $u \cdot v$ is `times(u, v)`. Powers $u^v$ are `power(u, v)`.
- **Properties of Expressions**:
  - `FreeOf(u, t)`: Check if expression $u$ does not contain sub-expression $t$.
  - `Substitute(u, t, r)`: Replace all instances of sub-expression $t$ with $r$ in $u$.
  - `Subterm(u, t)`: Check if $t$ is a subterm of $u$.
  - `Coeff(u, x, n)`: Extract the coefficient of $x^n$ in expression $u$.

### 2.2. Automatic Simplification
A core requirement of the symbolic engine is **Automatic Simplification**, which executes immediately when an expression is constructed. It transforms expressions into standard forms according to algebraic identities:
- **Sum Simplification**: Combining like terms, grouping constants, removing additive identities ($u + 0 \to u$).
- **Product Simplification**: Grouping factors, combining exponents, handling multiplicative identities ($u \cdot 1 \to u$) and absorbing elements ($u \cdot 0 \to 0$).
- **Power Simplification**: Handling base-exponent rules ($u^0 \to 1$, $u^1 \to u$, $(u^a)^b \to u^{a \cdot b}$).
- **Transcendental Simplification**: Basic trigonometric and logarithmic reductions ($\ln(e^x) \to x$, $\sin(0) \to 0$).
- **Fused Operators (Allocation Optimization)**: Fuses related operations (e.g., Fused Multiply-Add $a \cdot b + c \to \text{fma}(a, b, c)$) into a single AST node during automatic simplification. This minimizes recursive heap allocations and deallocations under RAII, improving AST traversal performance and keeping memory contiguous.

### 2.3. Polynomial and Rational Arithmetic
NimbleCAS supports rigorous polynomial computations in $\mathbb{Z}[x]$ and multivariate polynomial rings:
- **Expansion (`Expand(u)`)**: Recursively applies the distributive law to expand products and powers of sums.
- **Polynomial Division**: Division of polynomials with remainder ($A(x) = B(x)Q(x) + R(x)$).
- **Greatest Common Divisor (GCD)**: Implementation of the Euclidean algorithm and the Subresultant Polynomial Remainder Sequence (PRS) for multivariate polynomial GCD.
- **Square-Free Factorization**: Decomposing polynomials into square-free parts using Yun's algorithm.
- **Partial Fraction Decomposition**: Decomposing rational functions $\frac{P(x)}{Q(x)}$ (with $\deg(P) < \deg(Q)$) into a sum of simpler rational terms with denominators based on the irreducible factors (linear and quadratic) of $Q(x)$, supporting the Heaviside cover-up method and system solving of undetermined coefficients.

### 2.4. Symbolic Calculus
- **Symbolic Differentiation**: Implementation of the differentiation operator $\frac{d}{dx}$ using standard rules (sum, product, quotient, power, chain rule) for basic and transcendental functions.
- **Limits & L'Hôpital's Rule**: Symbolic evaluation of limits $\lim_{x \to c} f(x)$ for finite $c$, the infinities $\pm\infty$, and **one-sided** limits ($x \to c^{\pm}$). The engine classifies the seven **indeterminate forms** $\frac{0}{0}$, $\frac{\infty}{\infty}$, $0 \cdot \infty$, $\infty - \infty$, $1^{\infty}$, $0^{0}$, and $\infty^{0}$ and reduces them to the quotient forms (moving factors across the fraction bar, combining over a common denominator, or taking logarithms for the power forms). Indeterminate $\frac{0}{0}$ / $\frac{\infty}{\infty}$ limits are resolved by **L'Hôpital's rule** — $\lim \frac{f}{g} = \lim \frac{f'}{g'}$ via repeated differentiation of numerator and denominator (reusing the differentiation engine above) — cross-checked against **Taylor/series-expansion** limits (§2.9) and closed by the **squeeze theorem** for bounded$\times$vanishing patterns.
- **Partial Derivatives**: The product computes partial derivatives $\partial f / \partial x_i$ of multivariate expressions (differentiating with respect to one variable while holding the others fixed), including higher-order and mixed partials $\frac{\partial^{m+n} f}{\partial x^m \partial y^n}$ with automatic enforcement of Clairaut symmetry ($\partial_x \partial_y f = \partial_y \partial_x f$) for suitably smooth expressions.
- **Vector Calculus Operators**: First-class differential operators over scalar and vector fields — gradient $\nabla f$, divergence $\nabla \cdot \mathbf{F}$, curl $\nabla \times \mathbf{F}$, Laplacian $\nabla^2 f = \nabla \cdot \nabla f$, and directional derivatives $\nabla_{\mathbf{v}} f = \nabla f \cdot \mathbf{v}$ — together with the **Jacobian** matrix $J_{ij} = \partial f_i / \partial x_j$ and the **Hessian** matrix $H_{ij} = \partial^2 f / \partial x_i \partial x_j$ (reusing the linear-algebra engine of §2.7). Supports symbolic verification of the classical identities $\nabla \times \nabla f = \mathbf{0}$ and $\nabla \cdot (\nabla \times \mathbf{F}) = 0$.
- **Total Derivatives**: The total derivative / total differential via the multivariable chain rule $\frac{df}{dt} = \partial_t f + \sum_i \partial_{x_i} f \, \dot{x}_i$, including the fluid-dynamical **material derivative** $\frac{D}{Dt} = \partial_t + \mathbf{u} \cdot \nabla$, and **implicit differentiation** for relations $F(x, y) = 0$ (e.g. $\frac{dy}{dx} = -\,\partial_x F / \partial_y F$).
- **Symbolic Integration**:
  - Hermite Reduction for the rational part of integrals.
  - Rothstein-Trager algorithm for the logarithmic part of rational integrals.
  - Table-lookup and heuristic integration for transcendental functions.
- **Multiple Integrals**: Symbolic double, triple, and general $n$-fold iterated integrals $\int \cdots \int f \, dx_1 \cdots dx_n$ over both rectangular and general (variable-bound) regions, with **change of variables** via the Jacobian determinant $\left| \det \partial(x)/\partial(u) \right|$ (polar, cylindrical, and spherical coordinate systems built in) and **Fubini** reordering of the iterated limits. Includes line, surface, and volume integrals and the classical integral theorems — **Green's**, **Stokes'**, and the **divergence/Gauss** theorem — relating them to the vector-calculus operators above.

### 2.5. Analytical Equation Solving
- **Algebraic Solvers**: Solving linear and quadratic polynomial equations exactly.
- **Root Finding**: Analytical decomposition of higher-degree solvable polynomials.
- **Numerical Solvers**: High-performance, vectorized Newton-Raphson method for systems of non-linear equations.

### 2.6. Advanced Special Functions, Complex Numbers, and Combinatorics
- **Advanced Functions**: Implementation of special transcendental functions, specifically the **Lambert W function** ($\text{lambertW}(z)$), error functions ($\text{erf}$), and gamma/beta functions.
- **Complex Number Engine**: Native support for complex variables via a `ComplexNode` $z = a + b i$, carried **exactly** over the engine's `Rational` field where the parts are exact (Gaussian rationals $\mathbb{Q}(i)$) and symbolically otherwise. Provides the full **complex field arithmetic** (addition, multiplication $(a+bi)(c+di)=(ac-bd)+(ad+bc)i$, and conjugate-rationalized division), the **complex conjugate** $\bar z = a - bi$, **modulus** $|z| = \sqrt{a^2+b^2}$ and **argument** $\arg z$, and **polar / exponential form** $z = r e^{i\theta} = r(\cos\theta + i\sin\theta)$ via **Euler's formula** $e^{i\theta} = \cos\theta + i\sin\theta$ (with De Moivre powers). Computes **$n$-th roots and roots of unity** $\omega_k = e^{2\pi i k/n}$, and the **complex elementary functions** — $\exp$, $\log$ with explicit branch cuts (principal $\operatorname{Log} z = \ln|z| + i\arg z$), complex trigonometric/hyperbolic functions, and complex powers $z^w = e^{w\operatorname{Log} z}$ — with branch-cut tracking and complex residues.
- **Complex Analysis**: Holomorphic-function theory reusing the differentiation (§2.4) and series/residue (§2.9) engines — **Cauchy's integral theorem** ($\oint_\gamma f\,dz = 0$) and **integral formula** ($f^{(n)}(a) = \frac{n!}{2\pi i}\oint_\gamma \frac{f(z)}{(z-a)^{n+1}}\,dz$), **contour integration** over segments/arcs/keyhole contours, **Laurent series** about isolated singularities, and **residue calculus** ($\oint_\gamma f\,dz = 2\pi i \sum_k \operatorname{Res}(f,a_k)$) used as an automated engine for evaluating real definite integrals and infinite sums.
- **Blaschke Products**: Bounded analytic self-maps of the unit disk — the finite product $B(z) = \prod_k \frac{|a_k|}{a_k}\frac{a_k - z}{1 - \overline{a_k} z}$ with zeros $a_k$ in the disk and $|B(z)| = 1$ on $|z| = 1$, and infinite products converging under the **Blaschke condition** $\sum_k (1 - |a_k|) < \infty$. Represented as the canonical **inner functions** (bounded analytic functions on the disk) with their zero multisets.
- **Trigonometric Identities**: Symbolic recognition and application of the standard identities — **Pythagorean** ($\sin^2 + \cos^2 = 1$ and its $\tan/\sec$, $\cot/\csc$ forms), **angle-sum/difference**, **double-, half-, and multiple-angle**, **power-reduction**, **product-to-sum and sum-to-product**, and the **tangent half-angle (Weierstrass) substitution** $t = \tan(x/2)$. These extend the automatic-simplification engine (§2.2) to normalize/simplify trigonometric expressions and support the integration layer (§2.4) by rationalizing trig integrands. Known identities are stored for fast recall behind the XOR/Binary Fuse identity filter (§4.2).
- **Riemann Zeta Function**: The zeta function $\zeta(s) = \sum_{n\ge1} n^{-s}$ ($\Re(s) > 1$) with its **Euler product** $\zeta(s) = \prod_p (1 - p^{-s})^{-1}$ over primes, **analytic continuation** to $\mathbb{C}\setminus\{1\}$, and the **functional equation** $\zeta(s) = 2^s \pi^{s-1} \sin(\tfrac{\pi s}{2})\Gamma(1-s)\zeta(1-s)$. Closed forms at even integers via Bernoulli numbers $\zeta(2n) = \frac{(-1)^{n+1} B_{2n} (2\pi)^{2n}}{2\,(2n)!}$ (reusing the combinatorics/Bernoulli engine below) and at negative integers $\zeta(-n) = -\frac{B_{n+1}}{n+1}$, giving the **trivial zeros** at negative even integers; the critical strip/line and **Riemann Hypothesis** are carried as documented context. Includes the **Dirichlet eta** $\eta(s) = (1 - 2^{1-s})\zeta(s)$ and general Dirichlet $L$-function $L(s,\chi)$ connection.
- **Combinatorics**: Recursive and closed-form symbolic calculation of combinatorial functions. Native support for:
  - **Stirling Numbers**: Stirling numbers of the first kind $\left[ \begin{matrix} n \\ k \end{matrix} \right]$ (cycle-counting) and second kind $\left\{ \begin{matrix} n \\ k \end{matrix} \right\}$ (subset-partitioning).
  - **Catalan Numbers**: Catalan numbers $C_n = \frac{1}{n+1} \binom{2n}{n}$ with symbolic recurrence relations.
  - **Falling and Rising Powers**: Falling factorial powers $x^{\underline{n}} = \prod_{k=0}^{n-1} (x-k)$ and rising factorial powers $x^{\overline{n}} = \prod_{k=0}^{n-1} (x+k)$ from *Concrete Mathematics*, including algebraic transformations and conversions to/from ordinary powers using Stirling numbers.
  - **Partitions & Bell Numbers**: Integer partitions $P(n)$ and Bell numbers $B_n$ representing the total number of partitions of a set.
  - **Eulerian & Fibonacci Sequences**: Eulerian numbers $\left\langle \begin{matrix} n \\ k \end{matrix} \right\rangle$, Fibonacci numbers $F_n$, Lucas numbers $L_n$, and derangements (subfactorials $!n$).
  - **Bernoulli, Euler, and Multinomials**: Bernoulli numbers $B_n$, Euler numbers $E_n$, multinomial coefficients $\binom{n}{k_1, k_2, \dots, k_m}$, and Pochhammer symbol $(x)_n$.
  - **Harmonic Numbers**: Harmonic numbers $H_n = \sum_{i=1}^n \frac{1}{i}$ and generalized harmonic numbers $H_{n,r} = \sum_{i=1}^n \frac{1}{i^r}$.

### 2.7. Linear Algebra and Matrices
- **Symbolic Matrices**: Support for matrix symbols with non-numeric elements, symbolic determinants, matrix inverses, and matrix multiplications.
- **Matrix Decompositions**: Standard numeric and symbolic decompositions (LU, QR, SVD) and native high-speed **Cholesky decomposition** ($A = L L^T$) for symmetric positive-definite matrices.
- **Eigenvalue Solvers**: Standard and generalized eigenvalue problems ($A x = \lambda x$, $A x = \lambda B x$). Solvers include:
  - **Dense Matrices**: Power iteration and the QR algorithm.
  - **Sparse/Large Matrices**: Lanczos iteration (for symmetric matrices) and Arnoldi iteration (for general matrices) yielding Ritz eigenvalues.
- **Iterative Linear Solvers**: High-performance solvers for large systems ($A x = b$), including stationary methods (Jacobi, Gauss-Seidel, Successive Over-Relaxation (SOR)) and Krylov subspace methods (Conjugate Gradient (CG), generalized minimal residual (GMRES), and Bi-Conjugate Gradient Stabilized (BiCGSTAB)).

### 2.8. Differential Equations (ODEs and PDEs)
- **Analytical Solvers**: Automatic classification and exact solving of 1st and 2nd order linear ODEs and classical PDEs (heat, wave, Laplace equations).
- **Spectral and Pseudo-Spectral Methods**: Numerical solving of boundary value problems (BVPs) and non-linear PDEs. Evaluation of spatial derivatives in the spectral domain (using FFT or Chebyshev matrices) while non-linear multiplications are evaluated in the physical grid domain. Includes **dealiasing techniques** (Orszag 2/3-rule) to eliminate high-frequency aliasing errors.

### 2.9. Integral Transforms, Series, and Wavelets
- **Fourier Analysis**: Symbolic computation of Fourier series expansion, continuous Fourier transforms, and discrete/fast Fourier transforms.
- **Convolution & Spectral Relations**: The **Fourier convolution theorem** $\widehat{f * g} = \hat f\,\hat g$ and its dual $\widehat{f\cdot g} = \frac{1}{2\pi}\,\hat f * \hat g$, enabling **fast (linear and circular) convolution** in $O(N\log N)$ via the FFT (reusing the multi-dimensional FFT layer). The discrete **Cauchy product** (§2.9) is exactly this convolution in the series/polynomial setting. Includes the **power spectral density** via the **Wiener–Khinchin theorem** $S(\omega) = \mathcal{F}\{\gamma(h)\}$ (PSD as the Fourier transform of the autocovariance, §2.12.2 / §2.13.3), the **Parseval/Plancherel** energy identity, and **periodogram** spectrum estimation.
- **Multi-Dimensional Fourier Transforms**: $n$-dimensional continuous Fourier transforms $\hat{f}(\boldsymbol{\xi}) = \int_{\mathbb{R}^n} f(\mathbf{x})\, e^{-2\pi i\, \boldsymbol{\xi} \cdot \mathbf{x}}\, d\mathbf{x}$ and their inverses, and multi-dimensional DFT/FFT computed **separably** along each axis (reusing the 1-D FFT infrastructure of §2.9 and §3.x), for image/field data and the spectral solution of multi-dimensional PDEs (§2.8).
- **Wavelets**: Continuous Wavelet Transform (CWT) and Discrete Wavelet Transform (DWT) with support for Haar, Daubechies, and Morlet wavelet filters.
- **Series & Asymptotics**: Infinite sum evaluations, convergence tests, and **Taylor's series expansions** ($f(z) \approx \sum_{k=0}^n \frac{f^{(k)}(a)}{k!}(z-a)^k$) in both the **real** ($\mathbb{R}$) and **complex** ($\mathbb{C}$) domains. Supports Laurent and Puiseux series expansions of expressions around any point $z = a$ up to arbitrary order $n$, automatic differentiation, residue evaluation at singular poles, and computation of the radius of convergence.
- **Infinite Series & Convergence Tests**: Given a series $\sum_n a_n$, the engine decides convergence with a full battery of tests applied in a heuristic waterfall — **$n$-th-term/divergence**, **geometric**, **$p$-series**, **comparison** and **limit-comparison**, **ratio (d'Alembert)**, **root (Cauchy)**, **integral**, **alternating (Leibniz)**, **Dirichlet**, **Abel**, **Raabe**, and **Cauchy condensation** — and distinguishes **absolute vs conditional** convergence (with the **Riemann rearrangement** caveat blocking reordering of conditionally convergent series). For power series it computes the **radius and interval of convergence** ($R = 1/\limsup_n \sqrt[n]{|a_n|}$, Cauchy–Hadamard) with explicit endpoint testing.
- **Infinite Products**: Convergence of $\prod_n (1 + a_n)$ is decided by the **$\log$-to-series transform** ($\log\prod_n(1+a_n) = \sum_n \log(1+a_n)$), reducing to the series tests above; the product converges to a non-zero limit iff $\sum_n |a_n| < \infty$. A catalog of canonical **infinite-product identities** is recognized and rewritable in either direction: the **Euler product for sine** $\sin(\pi z) = \pi z \prod_{n\ge1}(1 - \frac{z^2}{n^2})$, the **Wallis product** $\frac{\pi}{2} = \prod_n \frac{4n^2}{4n^2-1}$, **Viète's formula** for $\frac{2}{\pi}$, the **Gamma reflection** $\Gamma(z)\Gamma(1-z) = \frac{\pi}{\sin\pi z}$ and Weierstrass product for $1/\Gamma$, and the **Euler product** for $\zeta$ (§2.6). Identities are stored for fast recall behind the XOR/Binary Fuse identity filter (§4.2).
- **Series & Sequence Manipulation**:
  - **Symbolic Summation**: Analytical evaluation of finite and infinite sums $\sum_{k=a}^b f(k)$ using **Gosper's algorithm** for hypergeometric terms and telescoping summation.
  - **Series Algebra**: Addition, subtraction, multiplication (Cauchy product), composition, reversion (Lagrange inversion formula), and differentiation/integration of series terms.
  - **Products of Series**: The **Cauchy product** $\left(\sum_n a_n\right)\left(\sum_n b_n\right) = \sum_n c_n$ with $c_n = \sum_{k=0}^n a_k b_{n-k}$ (a discrete convolution of the coefficient sequences, unifying it with the transform convolutions below), whose convergence is certified by **Mertens' theorem** (at least one factor absolutely convergent). This is the core of power-series multiplication and composition, with the product's coefficients captured as a **recurrence** (§2.14) for term-by-term generation.
  - **Sequence Operators**: Shift operators ($E$), difference operators ($\Delta$), and recurrence relation conversions.

### 2.10. Asymptotic Perturbation and Homotopy Methods (Esoterica)
NimbleCAS supports advanced symbolic solvers for highly non-linear or singular mathematical systems:
- **Singular Perturbation of ODEs**: Matched asymptotic expansions, boundary layer analysis, inner/outer expansions, and WKB approximation.
- **Homotopy Analysis Method (HAM)**: Constructing zero-order and high-order deformation equations, introducing auxiliary parameters $h$ and auxiliary functions $H(t)$ to control the convergence region and rate of series solutions.
- **Adomian Decomposition Method (ADM)**: Analytical approximation of non-linear differential equations by decomposing non-linear operators into Adomian Polynomials ($A_n$) for rapid series convergence.
- **Homotopy Perturbation Method (HPM) and Homotopy Analysis Protocol (HAP)**: Combines classical perturbation theory with homotopy to embed perturbation parameters $p \in [0, 1]$ into non-linear equations, solving the resulting sequence of linear equations.
- **Padé Approximations**: Construction of rational function approximations $[M/N]$ from Taylor series coefficients of a symbolic expression, solving the resulting Hankel system of linear equations to extend the domain of convergence and identify singular poles.
- **Continued Fractions**: Symbolic conversion of series expansions and functions into continued fraction representations (via Euler's continued fraction formula and Viskovatov's algorithm), and evaluating their convergents for high-speed numerical approximation.

### 2.11. Laplace Methods and Integral Transforms
- **Laplace Transforms**: Analytical computation of forward and inverse Laplace transforms ($\mathcal{L}\{f(t)\}$ and $\mathcal{L}^{-1}\{F(s)\}$), with automatic handling of Dirac delta, Heaviside step, and convolution integrals.
- **Laplace Convolution Theorem**: The convolution theorem $\mathcal{L}\{(f * g)(t)\} = F(s)\,G(s)$ with $(f * g)(t) = \int_0^t f(\tau) g(t-\tau)\,d\tau$, applied in both directions to invert products of transforms and to solve **Volterra integral** and **renewal** equations in transform space (cross-referencing the renewal-theory / stochastic Laplace analysis of §2.13.4).
- **Multi-Dimensional Laplace Transforms**: 2-D and $n$-D (multivariate) Laplace transforms $F(s_1, \dots, s_n) = \int_0^\infty \cdots \int_0^\infty f(t_1, \dots, t_n)\, e^{-(s_1 t_1 + \cdots + s_n t_n)}\, dt_1 \cdots dt_n$ and their inversion, computed by **iterated single-variable transforms** (reusing the 1-D machinery above), for multi-variable initial/boundary-value problems and the joint distributions of several non-negative random variables (§2.12.4).
- **Laplace's Approximation Method**: Asymptotic evaluation of high-dimensional integrals of the form $\int_a^b e^{M f(x)} g(x) dx$ for large $M$, locating the global maxima of $f(x)$ and using Taylor expansion to find the leading-order asymptotic behavior.

### 2.12. Statistics, Probability & Generating Functions
- **Probability Distributions**: Symbolic representation of continuous and discrete random variables, computing probability density functions (PDF), cumulative distribution functions (CDF), expected values, variance, higher-order moments, and moment-generating functions (MGF) for standard distributions.
- **Generating Functions**: Support for Ordinary Generating Functions (OGF) and Exponential Generating Functions (EGF). Ability to convert symbolic sequences to generating functions and vice versa.

#### 2.12.1. Descriptive Statistics
- **Measures of Location**: Arithmetic mean $\bar{x} = \frac{1}{n}\sum_i x_i$, weighted mean $\frac{\sum_i w_i x_i}{\sum_i w_i}$, geometric mean $\left(\prod_i x_i\right)^{1/n}$, and harmonic mean $n / \sum_i \tfrac{1}{x_i}$, together with the median, mode(s), and arbitrary quantiles/percentiles (with selectable interpolation conventions).
- **Measures of Dispersion**: Population variance $\sigma^2 = \frac{1}{n}\sum_i (x_i - \bar{x})^2$ and sample variance $s^2 = \frac{1}{n-1}\sum_i (x_i - \bar{x})^2$ (Bessel's correction), the corresponding **standard deviation** $\sigma$ / $s$, mean absolute deviation, coefficient of variation $\sigma/\bar{x}$, range, and interquartile range (IQR).
- **Measures of Shape**: Raw moments $m_k' = \frac{1}{n}\sum_i x_i^k$, central moments $m_k = \frac{1}{n}\sum_i (x_i - \bar{x})^k$, standardized moments, skewness $\gamma_1 = m_3/\sigma^3$, and excess kurtosis $\gamma_2 = m_4/\sigma^4 - 3$.
- **Exact & Numeric Dual Modes**: All descriptive statistics are computed both **exactly** (via the exact rational-arithmetic engine when given exact data, avoiding floating-point drift) and **numerically**. Numeric evaluation uses numerically stable, single-pass and parallelizable algorithms (**Welford's online recurrence** and **pairwise/parallel combination** of partial moments) that scale across the CPU/GPU parallel engine for large datasets.

#### 2.12.2. Correlation & Covariance
- **Covariance**: Sample and population covariance $\operatorname{cov}(X, Y)$ and the symmetric covariance matrix $\Sigma$ for multivariate data.
- **Correlation**: Pearson product-moment correlation $\rho_{XY} = \operatorname{cov}(X, Y)/(\sigma_X \sigma_Y)$ and the associated correlation matrix, plus rank correlations (**Spearman's $\rho$** and **Kendall's $\tau$**) for monotonic and ordinal association.
- **Time-Series Dependence**: Autocorrelation and partial-autocorrelation functions (ACF/PACF) and cross-correlation functions for sequential/time-series data, reusing the parallel FFT infrastructure (§2.9) for large lags.

#### 2.12.3. Statistical Distributions (Catalog)
- **Discrete Distributions**: Bernoulli, Binomial, Poisson, Geometric, Negative Binomial, Hypergeometric, Discrete Uniform, and Multinomial.
- **Continuous Distributions**: Uniform, Normal/Gaussian, Exponential, Gamma, Beta, Chi-squared, Student's $t$, F, Cauchy, Log-normal, Weibull, Pareto, and Multivariate Normal.
- **Per-Distribution Capabilities**: For each distribution the product exposes the symbolic PMF/PDF, CDF, quantile (inverse-CDF) function, MGF and characteristic function, closed-form mean/variance/skewness/kurtosis/entropy, documented **limiting relationships** between distributions (e.g. Binomial $\to$ Poisson, Poisson $\to$ Normal, Student's $t \to$ Normal), and random sampling for simulation.
- **Symbolic Reuse**: Symbolic moments and expected values are obtained by reusing the symbolic **integration** engine (§2.4), $\mathbb{E}[X^n] = \int x^n f(x)\,dx$, while moments derived from the MGF are obtained via the symbolic **differentiation** engine, $\mathbb{E}[X^n] = M_X^{(n)}(0)$.

#### 2.12.4. Moments, Cumulants & Generating Functions
- **Generating Functions**: Moment-generating function $M_X(t) = \mathbb{E}[e^{tX}]$, characteristic function $\varphi_X(t) = \mathbb{E}[e^{itX}]$, probability-generating function $G_X(z) = \mathbb{E}[z^X]$, and cumulant-generating function $K_X(t) = \ln M_X(t)$.
- **Laplace Transform of a Distribution**: The Laplace–Stieltjes transform $\tilde{f}_X(s) = \mathbb{E}[e^{-sX}]$ of a non-negative random variable $X$ — equivalently the MGF evaluated at $-s$, and closely related to the characteristic function $\varphi_X(is)$. The transform **uniquely characterises** the distribution, yields raw moments by differentiation at the origin ($\mathbb{E}[X^n] = (-1)^n \tilde{f}_X^{(n)}(0)$), and has documented closed forms for the standard distributions in the catalog. This capability reuses the existing symbolic Laplace-transform engine (§2.11), including its inversion machinery for recovering densities.
- **Moments & Cumulants**: Raw, central, and factorial moments and cumulants $\kappa_n$, with the full moment$\leftrightarrow$cumulant conversion relations (via Bell-polynomial expansions, connecting to the combinatorics engine of §2.6).
- **Series Expansions**: Edgeworth and Gram–Charlier A-series expansions for approximating densities from their moments/cumulants.

#### 2.12.5. Inferential Statistics & Regression
- **Parameter Estimation**: Method of moments and maximum-likelihood estimation (MLE), with symbolic score function $\partial_\theta \ln L$ and Fisher information $\mathcal{I}(\theta)$.
- **Confidence & Testing**: Confidence intervals and hypothesis testing, including $z$, $t$, $\chi^2$, and F tests with symbolic and numeric p-values.
- **Regression**: Least-squares regression (ordinary, weighted, and ridge/Tikhonov-regularized) solved via the normal equations $(X^\top X)\,\hat{\beta} = X^\top y$ on the linear-algebra engine (§2.7), with goodness-of-fit diagnostics ($R^2$, residual analysis).
- **Bayesian Inference**: Conjugate-prior posterior updates and maximum a posteriori (MAP) estimation for standard likelihood/prior families.

### 2.13. Stochastic Processes and Stochastic Differential Equations
- **Stochastic Calculus**: Support for Itô and Stratonovich calculus representations, implementing Itô's lemma for multidimensional stochastic variables.
- **SDE/SPDE Solvers**: Analytical solving of simple linear SDEs (e.g. Geometric Brownian Motion, Ornstein-Uhlenbeck process). High-performance numerical simulation using the parallelized **Euler-Maruyama** and **Milstein schemes** on CPU (PPL) and GPUs, and solving associated **Fokker-Planck equations** analytically or numerically.

#### 2.13.1. Discrete-Time Stochastic Processes
- **Markov Chains**: Symbolic and numeric transition matrices $P$, $n$-step transition probabilities $P^n$, stationary/limiting distributions $\pi P = \pi$, classification of states, absorption probabilities, and expected hitting/absorption times — solved via the linear-algebra engine (§2.7).
- **Random Walks & Branching Processes**: Simple and general random walks (including gambler's-ruin analysis) and Galton–Watson branching processes with extinction-probability computation via probability-generating functions (§2.12.4).
- **Martingales**: Symbolic representation and verification of the martingale property $\mathbb{E}[X_{n+1} \mid \mathcal{F}_n] = X_n$, sub-/super-martingales, and optional-stopping arguments.

#### 2.13.2. Continuous-Time Processes
- **Poisson & Counting Processes**: Homogeneous and inhomogeneous Poisson processes and compound-Poisson processes, with interarrival-time and jump-size distributions drawn from the catalog (§2.12.3).
- **Continuous-Time Markov Chains**: Generator (rate) matrices $Q$, Kolmogorov forward/backward equations, and stationary distributions solved via matrix exponentials $e^{Qt}$.
- **Gaussian & Diffusion Processes**: Brownian motion / Wiener processes, general Gaussian processes (specified by mean and covariance kernels), and the Ornstein–Uhlenbeck mean-reverting process.

#### 2.13.3. Time-Series Models
- **Model Families**: Autoregressive (AR), moving-average (MA), ARMA, and ARIMA models with symbolic coefficient handling.
- **Analysis & Estimation**: Stationarity and invertibility analysis, autocorrelation/partial-autocorrelation structure (§2.12.2), and parameter estimation via the **Yule–Walker equations** solved on the linear-algebra engine (§2.7).

#### 2.13.4. Laplace-Transform Methods for Stochastic Processes
- **First-Passage & Renewal Theory**: Laplace transforms of first-passage/hitting-time distributions, and renewal theory — solving the renewal equation and obtaining the renewal function $m(t)$ through its Laplace transform.
- **Queueing Analysis**: Transform-domain analysis of queues, including the M/G/1 **Pollaczek–Khinchine** waiting-time/queue-length transform.
- **Transform Solution of Evolution Equations**: Laplace transformation of the time variable in Fokker–Planck / Kolmogorov equations and in linear SDEs, reducing them to boundary-value problems in the transform domain; the **resolvent** / Laplace transform of a Markov process's transition density; and the **Laplace exponent** (Lévy–Khintchine representation) characterising compound-Poisson and general Lévy processes. All transform and inversion steps reuse the symbolic Laplace-transform capability of §2.11.

#### 2.13.5. Monte Carlo Simulation
- **High-Performance Sampling**: Massively parallel pseudo-random sampling of distributions and process paths, with **variance-reduction** techniques (antithetic variates, control variates, importance sampling) to accelerate convergence.
- **MCMC**: Markov chain Monte Carlo samplers, including **Metropolis–Hastings** and **Gibbs** sampling, for posterior and high-dimensional integration problems.
- **Execution Infrastructure**: Simulations execute on the parallel CPU/GPU engine (§3.1–§3.3) and are distributed across worker nodes via the **StochasticGraphExecutionEngine** (§3.4) for large-scale ensembles.

### 2.14. Difference Equations and Recurrence Relations
- **Recurrence Relations**: Solving linear and non-linear difference equations analytically using characteristic equations, generating functions, and Z-transforms.
- **Asymptotic Behavior**: Computing the asymptotic behavior of difference equation solutions for large $n$.

### 2.15. Dynamical Systems and Stability Analysis
- **Phase Space Analysis**: Analysis of dynamical systems $\dot{x} = f(x)$, identifying equilibrium points and computing the Jacobian matrix.
- **Stability and Bifurcation**: Determining stability via Lyapunov stability criteria, computing eigenvalues of the Jacobian at fixed points, and identifying bifurcation points (saddle-node, pitchfork, Hopf bifurcations) symbolically.
- **Chaos and Attractors**: High-speed numerical evaluation of chaotic trajectories (e.g., Lorenz attractor, Rössler attractor) and computation of Lyapunov exponents, Poincaré maps, and fractal dimensions using GPU vectorization.

### 2.16. Plotting and Visualization
- **2D Plotting Engine**: High-fidelity plotting of functions $y = f(x)$, parametric curves $(x(t), y(t))$, polar curves $r(\theta)$, implicit curves $f(x, y) = 0$, scatter plots, and 2D vector fields (e.g. phase portraits for 2D systems).
- **3D Plotting Engine**: Interactive rendering of surfaces $z = f(x, y)$, 3D parametric curves $(x(t), y(t), z(t))$, 3D implicit surfaces $f(x, y, z) = 0$, and 3D vector fields.
- **Interactive Visualizations**: Real-time rendering of parameter sweeps using sliders (e.g. observing the deformation of a bifurcation attractor or wave propagation over time), 3D camera rotation/zoom, and color mapping based on function value gradients or curvature.
- **GPU-Accelerated Graphics**: Integration with WebGPU and WebGL (for browser/document environments) and DirectX/Vulkan/OpenGL (for native GUI applications) to render millions of data points smoothly. Supporting automatic hardware detection and fallback.
- **LaTeX Math Exporter**: Built-in support to compile any symbolic expression tree into standard, syntax-conforming LaTeX strings (e.g. converting division to `\frac`, integrals to `\int`, special functions like `lambertW` to custom formats, and matrices to `\begin{pmatrix}`). Exposed directly in Python bindings (`expr.to_latex()`).

### 2.17. Executable Document Engine (Live Notebooks)
- **Live Markdown Integration**: Support for executing dynamic documents (`.ncmd` / `.casmd`) containing interspersed Markdown text and executable code blocks (fenced with ````{nimblecas}` or ````{python}`).
- **Inline Result Rendering**: The engine must execute code blocks in a sandboxed interpreter and render resulting mathematical evaluations (formatted as LaTeX equations), tables, stdout logs, and generated 2D/3D plots directly inline.
- **State Caching & Execution Modes**: Incremental execution support via cell-level hashing. If a code block's source and dependency cells are unchanged, the engine uses cached execution states to bypass recalculation.
- **Export Formats**: Ability to compile the executed live notebook into polished static HTML (containing embedded WebGL and WebGPU rendering scripts that dynamically fall back from WebGPU to WebGL, and finally to CPU/SVG depending on client hardware availability), standard Markdown, and production-ready PDF documents.

### 2.18. Numerical Solvers and Optimization Engine
- **Non-Linear Equations & Systems**: Solvers for root-finding of single univariate non-linear equations and multi-dimensional systems of non-linear equations ($F(x) = 0$).
- **Iterative Root Solvers**: Implementation of:
  - **Fixed-Point Iteration**: Solving $x = g(x)$ with user-defined convergence criteria.
  - **Newton-like & C.T. Kelley Iterative Methods**: High-speed Newton-Raphson, Secant, and Halley's methods for univariate equations. For multi-dimensional systems: multi-dimensional Newton's method, Broyden's updates, **Inexact Newton methods** (with linear solvers like GMRES/BiCGSTAB utilizing Kelley's $\eta$-forcing terms to prevent oversolving), **Jacobian-Free Newton-Krylov (JFNK)**, **backtracking line search** (Armijo rule for global convergence), and **Anderson Acceleration** for fixed-point iterations.
- **Linear & Non-Linear Optimization**:
  - **Linear Programming (LP)**: Solvers for bounded linear optimization systems using the Simplex and Interior-Point algorithms.
  - **Non-Linear Optimization**: Local and global minimizers including Gradient Descent, Conjugate Gradient, Nelder-Mead simplex, and L-BFGS algorithms.
  - **Non-Linear Least Squares**: Curve fitting and parameter estimation via the Levenberg-Marquardt and Gauss-Newton algorithms.
  - **Spline Interpolation**: Fitting and evaluation of 1D and 2D data points using linear, quadratic, and cubic splines, Hermite splines, B-splines (basis splines), and Non-Uniform Rational B-Splines (NURBS) with symbolic piecewise-expression exports.

### 2.19. Quantum Mechanics and Functional Analysis Engine
- **Abstract Algebras & Lie Algebras**: Symbolic representation of algebraic objects, non-commutative algebras, and Lie algebras. Native support for:
  - **Lie Brackets**: Symbolic brackets $[x, y]$ adhering to bilinear, skew-symmetric, and Jacobi identity ($[x, [y, z]] + [y, [z, x]] + [z, [x, y]] = 0$) rules.
  - **Structure Constants**: Resolving algebra expansions using structure constants: $[e_i, e_j] = \sum_k c_{i,j}^k e_k$.
- **Quantum State Representations (Dirac Notation)**:
  - **Bra and Ket Vector Spaces**: Symbolic representation of Kets $| \psi \rangle$ and Bras $\langle \phi |$, including scalar multiplications, linear combinations, and conjugate transpositions ($(| \psi \rangle)^\dagger = \langle \psi |$).
  - **Outer and Inner Products**: Support for inner products $\langle \phi | \psi \rangle$ (yielding scalars or expressions) and outer products $| \psi \rangle \langle \phi |$ (yielding projection operators).
  - **Hamiltonians & Operators**: Symbolic manipulation of self-adjoint operators (such as Hamiltonians $\hat{H}$, momentum $\hat{p}$, and position $\hat{x}$) acting on state vectors (eigenvalue equations: $\hat{H} | \psi \rangle = E | \psi \rangle$).
- **Abstract Spaces, Norms & Metrics**:
  - **Hilbert Spaces**: Symbolic definitions of infinite-dimensional inner-product spaces (such as $L^2$). Evaluates inner products $\langle x, y \rangle$, inducing the norm $\|x\|_H = \sqrt{\langle x, x \rangle}$, and inducing the metric $d(x, y) = \|x - y\|_H$. Includes completeness axioms, orthonormal basis expansions, and projections.
  - **Banach Spaces**: Symbolic definitions of complete normed vector spaces (such as $L^p$ spaces). Evaluates norms $\|x\|_B$ satisfying positive definiteness, absolute homogeneity, and the triangle inequality, inducing the metric $d(x, y) = \|x - y\|_B$.
  - **Metric Spaces**: Support for abstract distance metrics $d(x, y)$ satisfying positivity, symmetry ($d(x, y) = d(y, x)$), identity of indiscernibles ($d(x, y) = 0 \iff x = y$), and the triangle inequality ($d(x, z) \le d(x, y) + d(y, z)$).

### 2.20. Classical Orthogonal Polynomials
- **Symbolic Representation & Evaluation**: Dedicated AST nodes for families of univariate orthogonal polynomials, including symbolic expansions, evaluations, and derivations.
- **Orthogonal Families**: Native support for:
  - **Chebyshev Polynomials**: First kind $T_n(x)$ and second kind $U_n(x)$ with trigonometric conversions (e.g. $T_n(\cos\theta) = \cos(n\theta)$).
  - **Legendre Polynomials**: Standard Legendre polynomials $P_n(x)$ and associated Legendre polynomials $P_n^m(x)$.
  - **Laguerre Polynomials**: Laguerre polynomials $L_n(x)$ and generalized/associated Laguerre polynomials $L_n^\alpha(x)$.
  - **Hermite Polynomials**: Physicist's Hermite polynomials $H_n(x)$ and probabilist's Hermite polynomials $He_n(x)$.
  - **Jacobi & Gegenbauer Polynomials**: Jacobi polynomials $P_n^{(\alpha, \beta)}(x)$ and Gegenbauer (ultraspherical) polynomials $C_n^\alpha(x)$.
- **Recurrence Relations & Weight Functions**: Symbolic resolution of three-term recurrence relations (e.g., $T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)$), Rodrigues' formulas, and verification of orthogonality relations over respective domains against weight functions $w(x)$ (e.g. $w(x) = \frac{1}{\sqrt{1-x^2}}$ for $T_n(x)$).

---

## 3. High-Performance Architecture & Hardware Acceleration

To meet the speed demands of modern simulations and real-time physical modeling, NimbleCAS must fully exploit local hardware capabilities on Windows.

### 3.1. CPU Parallelism (Microsoft PPL / Intel oneTBB)
- **Cross-Platform Parallel Execution**: Concurrency must scale dynamically depending on the compile target:
  - **Windows**: Uses the Microsoft Concurrency Runtime's **Parallel Patterns Library (PPL)**.
  - **Linux & macOS**: Uses **Intel oneTBB (Threading Building Blocks)**.
  - Core math routines (parallel grids, polynomial arithmetic, numerical solvers) must use unified parallel wrappers that map to PPL or TBB at compile-time.
  - **Zero-Copy Ranges and Views**: To achieve maximum throughput, data pipeline transformations (such as array maps, matrix slicing, and polynomial reductions) must utilize C++20/C++23 ranges and views (`std::ranges` and `std::views`) composed lazily. These views must be processed directly within parallel loop constructs to avoid intermediate memory copies.
- **Concurrency Safeguards**: All parallel algorithms must be data-race free. Symbolic expression trees are immutable (COW), eliminating lock contention during concurrent reads.

### 3.2. Vectorization (Multiregister SIMD with Dynamic Dispatch)
- **Dynamic Dispatching**: Vectorized operations (such as matrix algebra, array-based numerical evaluations, and polynomial evaluation) must query the CPU capability at runtime and execute the most advanced instruction set available:
  - **AVX-512** (using localized function attributes `[[gnu::target("avx512f,avx512dq,fma")]]` per target policy).
  - **AVX2 & FMA** (Advanced Vector Extensions 2).
  - **SSE2** (Streaming SIMD Extensions 2).
  - **Scalar Fallback** (standard C++ loop structures).
- **Memory Alignment**: Vector arrays and matrices must use aligned memory allocations to avoid page-boundary faults.

### 3.3. GPU-Accelerated Symbolic Manipulation (JIT & Polish Notation)
- **Symbolic JIT Compilation (NVRTC)**: For array-based numerical evaluations, the symbolic engine must compile expression ASTs at runtime into high-performance CUDA/HLSL kernels using NVIDIA Runtime Compilation (NVRTC) to avoid recursive tree traversals on GPU registers.
- **Flattened Polish Stacks**: Pure symbolic processing on the GPU (e.g. parallel pattern matching or subterm searches) must flatten pointer-based trees into contiguous array buffers of prefix/postfix tokens, enabling GPU memory-coalesced parallel scans.
- **GPU Modular Arithmetic**: To parallelize multivariate polynomial expansions and high-degree GCD operations, computations are partitioned into modulo-prime images.
  - GPU threads execute modular multiplications and reductions in parallel using **Montgomery multiplication** or **Barrett reduction** to bypass division latency.
  - Final results are reconstructed on-device using a parallelized **Chinese Remainder Theorem (CRT)** or **Mixed-Radix Conversion (MRC)**.
- **Triton Kernel Acceleration**: Custom computation kernels (e.g. parallel SDE path simulations, chaotic attractor integrations, and high-dimensional grid evaluations) are compiled using **OpenAI's Triton compiler**. Triton abstracts block-level memory coalescing and shared memory management, allowing these high-performance kernels to run and scale seamlessly across single and multiple GPU nodes.

### 3.4. Distributed Symbolic Derivations (StochasticGraphExecutionEngine)
- **Stochastic Graph Scheduling**: Highly complex analytical expansions (e.g. high-order HAM deformation equations) are compiled into symbolic Directed Acyclic Graphs (DAGs) and executed on the distributed cluster.
- **Distributed Modular Division & Resultants**: High-degree polynomial GCDs are distributed to remote worker nodes, where each worker computes the GCD modulo a different prime $p_i$ concurrently, returning the modular image.
- **Distributed Sub-expression Memoization**: Distributed key-value tables cache simplified sub-expressions (hash-consing) across worker nodes to prevent redundant symbolic calculations across the network.

---

## 4. Software Engineering, C++23 Constraints & C++26 Readiness

NimbleCAS must adhere strictly to the rules of the official project code policy (`cpp_details.txt`).

### 4.1. Language Style & Safety
- **Toolchain**: Built using `clang++-22` on Windows.
- **C++26 Reflection Readiness**: The codebase must be structured to immediately transition to C++26 once a compiler toolchain supporting standard reflection (`std::meta` and the `^` operator) is stable. Symbolic visitor patterns, AST serialization, and JSON schema generations must be prepared to swap manual templates for compiler-driven reflection.
- **Default Parallel Library & Windows PPL**: Concurrency maps to **Intel oneTBB** as the default parallel backend for all non-Windows target platforms, while Windows compilation paths leverage the Microsoft Concurrency Runtime's **Parallel Patterns Library (PPL)**.
- **Modules**: Built using C++23 modules (`import std`). Header files are replaced with precompiled modules (`.pcm`) explicitly mapped in the build system.
- **Memory Management**: Zero raw pointers. Explicit usage of `std::unique_ptr` and `std::shared_ptr`.
- **Copy-on-Write (COW)**: Symbolic sub-expressions are wrapped in a COW smart pointer class (`CowPtr<T>`) to make copying expressions extremely cheap and threads safe.
- **Error Handling**: No C++ exceptions. Functions that can fail must return `std::expected<T, ErrorCode>` and utilize railway-oriented programming (monadic operations).
- **Trail Calling Return Syntax**: Trailing return types must be used for all function definitions:
  ```cpp
  auto evaluate(const Context& ctx) const -> std::expected<double, EvaluationError>;
  ```
- **Fluent & Composable APIs**: All key modules (symbolic engine, linear algebra matrices, homotopy/differential solvers, and plotting outputs) must expose a unified, fluent, and composable API utilizing method chaining. Configuration parameters and mathematical operations must link together seamlessly without manual temporary object declarations:
  ```cpp
  auto expr = x.add(y).mul(z).simplify().differentiate("x");
  auto mat = matrix.transpose().mul(other_matrix).inverse();
  auto solution = solver.with_equation(eq).with_order(6).solve();
  auto scene = plot.add_surface(f).with_resolution(256).with_colormap("viridis").render();
  ```

### 4.2. Third-Party Libraries & Dependencies
- **JSON Serialization**: Use **`fastestjsoninthewest`** for high-speed serialization of expression trees and context data.
- **Membership Filtering**: Use **XOR/Binary Fuse Filters** for rapid membership checks (e.g., matching function signatures, finding symbols in expressions).
- **Identity / Pattern Filter (fast identity recall)**: The same XOR/Binary Fuse mechanism is generalized into a reusable *identity/pattern filter* backing the engine's identity catalogs — **trigonometric identities** (§2.6), **infinite-product** and **series identities** (§2.9), and the integration table (§2.4). Each identity's left-hand-side pattern is reduced to a canonical **structural fingerprint** (reusing the engine's structural hashing / hash-consing) and inserted into the filter; a candidate expression's fingerprint is queried first — a near-$O(1)$, tiny-memory test with **no false negatives** (only rare false positives) — so the exact identity hash-table is probed only on a hit. This lets the simplification and integration layers apply identity matching **speculatively and cheaply** during a single AST traversal rather than scanning full identity tables per node.
- **Testing**: No external testing frameworks (like GTest or Catch2) are permitted. Testing must use the internal logger-based test runner or `sensen` library testing.

---

## 5. Non-Functional Requirements (NFRs)

| ID | Category | Requirement Description |
| :--- | :--- | :--- |
| **NFR-1** | **Precision** | All numerical calculations default to BF16 (Brain Floating Point 16) end-to-end to match target inference/training pipelines, with FP32 permitted only for specific solvers. |
| **NFR-2** | **Portability** | Must work on Windows, Linux, and macOS on day one. All build tools (Clang 22, CMake, Ninja, clang-tidy, clang-format), directory configurations, file I/O systems, and dependencies must be local and referenced relatively within the project directory, allowing relocatable out-of-the-box building. |
| **NFR-3** | **Reproducibility** | FP arithmetic must be bit-identical across hardware. No `-ffast-math` is allowed. Build flags must enforce `-march=x86-64-v3 -mtune=generic`. |
| **NFR-4** | **Sanitization** | The codebase must pass clean sweeps under AddressSanitizer (ASan), UndefinedBehaviorSanitizer (UBSan), and LeakSanitizer (LSan). |
