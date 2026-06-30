# Product Requirement Document (PRD) — NimbleCAS

**Version:** 1.0.0  
**Date:** June 2026  
**Status:** Approved  
**Author:** Antigravity (Advanced Agentic Coding Team)

---

## 1. Executive Summary & Vision

**NimbleCAS** is a high-performance, modern Computer Algebra System (CAS) built for Windows using C++23, C++23 Modules, and the Parallel Patterns Library (PPL). It bridges the gap between pure symbolic mathematical manipulation and high-performance numeric execution by exploiting multi-core CPUs, multi-register SIMD vectorization, and GPU acceleration.

Unlike legacy systems which rely on heavyweight, single-threaded architectures and legacy C-style pointer structures, NimbleCAS uses **Copy-on-Write (COW) Expression Trees**, monadic error-handling via `std::expected`, and a compiler-optimized C++23 module dependency graph. The mathematical foundations of the system are based directly on the rigorous algebraic algorithms detailed in **Joel S. Cohen's** *Computer Algebra and Symbolic Computation: Elementary Algorithms* and *Mathematical Methods*.

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

### 2.3. Polynomial and Rational Arithmetic
NimbleCAS supports rigorous polynomial computations in $\mathbb{Z}[x]$ and multivariate polynomial rings:
- **Expansion (`Expand(u)`)**: Recursively applies the distributive law to expand products and powers of sums.
- **Polynomial Division**: Division of polynomials with remainder ($A(x) = B(x)Q(x) + R(x)$).
- **Greatest Common Divisor (GCD)**: Implementation of the Euclidean algorithm and the Subresultant Polynomial Remainder Sequence (PRS) for multivariate polynomial GCD.
- **Square-Free Factorization**: Decomposing polynomials into square-free parts using Yun's algorithm.
- **Partial Fraction Decomposition**: Decomposing rational functions $\frac{P(x)}{Q(x)}$ (with $\deg(P) < \deg(Q)$) into a sum of simpler rational terms with denominators based on the irreducible factors (linear and quadratic) of $Q(x)$, supporting the Heaviside cover-up method and system solving of undetermined coefficients.

### 2.4. Symbolic Calculus
- **Symbolic Differentiation**: Implementation of the differentiation operator $\frac{d}{dx}$ using standard rules (sum, product, quotient, power, chain rule) for basic and transcendental functions.
- **Symbolic Integration**:
  - Hermite Reduction for the rational part of integrals.
  - Rothstein-Trager algorithm for the logarithmic part of rational integrals.
  - Table-lookup and heuristic integration for transcendental functions.

### 2.5. Analytical Equation Solving
- **Algebraic Solvers**: Solving linear and quadratic polynomial equations exactly.
- **Root Finding**: Analytical decomposition of higher-degree solvable polynomials.
- **Numerical Solvers**: High-performance, vectorized Newton-Raphson method for systems of non-linear equations.

### 2.6. Advanced Special Functions, Complex Numbers, and Combinatorics
- **Advanced Functions**: Implementation of special transcendental functions, specifically the **Lambert W function** ($\text{lambertW}(z)$), error functions ($\text{erf}$), and gamma/beta functions.
- **Complex Number Engine**: Native support for complex variables ($z = x + i y$) and Euler polar representations ($r e^{i\theta}$). Symbolic evaluation of complex functions, branch cut tracking, and complex residues.
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
- **Wavelets**: Continuous Wavelet Transform (CWT) and Discrete Wavelet Transform (DWT) with support for Haar, Daubechies, and Morlet wavelet filters.
- **Series & Asymptotics**: Infinite sum evaluations, convergence tests, and **Taylor's series expansions** ($f(z) \approx \sum_{k=0}^n \frac{f^{(k)}(a)}{k!}(z-a)^k$) in both the **real** ($\mathbb{R}$) and **complex** ($\mathbb{C}$) domains. Supports Laurent and Puiseux series expansions of expressions around any point $z = a$ up to arbitrary order $n$, automatic differentiation, residue evaluation at singular poles, and computation of the radius of convergence.
- **Series & Sequence Manipulation**:
  - **Symbolic Summation**: Analytical evaluation of finite and infinite sums $\sum_{k=a}^b f(k)$ using **Gosper's algorithm** for hypergeometric terms and telescoping summation.
  - **Series Algebra**: Addition, subtraction, multiplication (Cauchy product), composition, reversion (Lagrange inversion formula), and differentiation/integration of series terms.
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
- **Laplace's Approximation Method**: Asymptotic evaluation of high-dimensional integrals of the form $\int_a^b e^{M f(x)} g(x) dx$ for large $M$, locating the global maxima of $f(x)$ and using Taylor expansion to find the leading-order asymptotic behavior.

### 2.12. Probability and Generating Functions
- **Probability Distributions**: Symbolic representation of continuous and discrete random variables, computing probability density functions (PDF), cumulative distribution functions (CDF), expected values, variance, higher-order moments, and moment-generating functions (MGF) for standard distributions.
- **Generating Functions**: Support for Ordinary Generating Functions (OGF) and Exponential Generating Functions (EGF). Ability to convert symbolic sequences to generating functions and vice versa.

### 2.13. Stochastic Differential Equations (SDEs and SPDEs)
- **Stochastic Calculus**: Support for Itô and Stratonovich calculus representations, implementing Itô's lemma for multidimensional stochastic variables.
- **SDE/SPDE Solvers**: Analytical solving of simple linear SDEs (e.g. Geometric Brownian Motion, Ornstein-Uhlenbeck process). High-performance numerical simulation using the parallelized **Euler-Maruyama** and **Milstein schemes** on CPU (PPL) and GPUs, and solving associated **Fokker-Planck equations** analytically or numerically.

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
- **Testing**: No external testing frameworks (like GTest or Catch2) are permitted. Testing must use the internal logger-based test runner or `sensen` library testing.

---

## 5. Non-Functional Requirements (NFRs)

| ID | Category | Requirement Description |
| :--- | :--- | :--- |
| **NFR-1** | **Precision** | All numerical calculations default to BF16 (Brain Floating Point 16) end-to-end to match target inference/training pipelines, with FP32 permitted only for specific solvers. |
| **NFR-2** | **Portability** | Must work on Windows, Linux, and macOS on day one. All build tools (Clang 22, CMake, Ninja, clang-tidy, clang-format), directory configurations, file I/O systems, and dependencies must be local and referenced relatively within the project directory, allowing relocatable out-of-the-box building. |
| **NFR-3** | **Reproducibility** | FP arithmetic must be bit-identical across hardware. No `-ffast-math` is allowed. Build flags must enforce `-march=x86-64-v3 -mtune=generic`. |
| **NFR-4** | **Sanitization** | The codebase must pass clean sweeps under AddressSanitizer (ASan), UndefinedBehaviorSanitizer (UBSan), and LeakSanitizer (LSan). |
