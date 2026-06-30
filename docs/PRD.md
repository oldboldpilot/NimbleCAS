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
- **Combinatorics**: Recursive and closed-form symbolic calculation of factorials, binomial coefficients, combinations, permutations, Stirling numbers, and generating functions.

### 2.7. Linear Algebra and Matrices
- **Symbolic Matrices**: Support for matrix symbols with non-numeric elements, symbolic determinants, matrix inverses, and matrix multiplications.
- **Linear Algebra Routines**: Symbolic and high-performance numeric eigenvalues, eigenvectors, and matrix decompositions (LU, QR, SVD, Cholesky) accelerated via CPU SIMD and GPU co-processing.

### 2.8. Differential Equations (ODEs and PDEs)
- **Analytical ODE Solvers**: Automatic classification and exact solving of 1st and 2nd order linear ODEs (separation of variables, integrating factors, variation of parameters, Laplace transforms).
- **Analytical PDE Solvers**: Exact solutions for classical PDEs (heat, wave, Laplace equations) using separation of variables, Fourier transforms, and the method of characteristics.

### 2.9. Integral Transforms, Series, and Wavelets
- **Fourier Analysis**: Symbolic computation of Fourier series expansion, continuous Fourier transforms, and discrete/fast Fourier transforms.
- **Wavelets**: Continuous Wavelet Transform (CWT) and Discrete Wavelet Transform (DWT) with support for Haar, Daubechies, and Morlet wavelet filters.
- **Series & Asymptotics**: Infinite sum evaluations, convergence tests, and Taylor, Laurent, and Puiseux series expansions of expressions around any point $x = a$ up to arbitrary order $n$.

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

---

## 3. High-Performance Architecture & Hardware Acceleration

To meet the speed demands of modern simulations and real-time physical modeling, NimbleCAS must fully exploit local hardware capabilities on Windows.

### 3.1. CPU Parallelism (Microsoft PPL)
- **Parallel Patterns Library (PPL)**: Must be used for CPU-bound symbolic computations, such as:
  - Parallel evaluation of expressions over dense grids.
  - Parallel polynomial expansion and factorization using `ppl::parallel_invoke` and `ppl::task_group`.
  - Parallel numerical solvers and multi-start optimization searches.
- **Concurrency Safeguards**: All algorithms must be data-race free. Symbolic expression trees are immutable (COW), eliminating the need for expensive locking during parallel read sweeps.

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

## 4. Software Engineering & C++23 Constraints

NimbleCAS must adhere strictly to the rules of the official project code policy (`cpp_details.txt`).

### 4.1. Language Style & Safety
- **Toolchain**: Built using `clang++-22` on Windows.
- **Modules**: Built using C++23 modules (`import std`). Header files are replaced with precompiled modules (`.pcm`) explicitly mapped in the build system.
- **Memory Management**: Zero raw pointers. Explicit usage of `std::unique_ptr` and `std::shared_ptr`.
- **Copy-on-Write (COW)**: Symbolic sub-expressions are wrapped in a COW smart pointer class (`CowPtr<T>`) to make copying expressions extremely cheap and threads safe.
- **Error Handling**: No C++ exceptions. Functions that can fail must return `std::expected<T, ErrorCode>` and utilize railway-oriented programming (monadic operations).
- **Trail Calling Return Syntax**: Trailing return types must be used for all function definitions:
  ```cpp
  auto evaluate(const Context& ctx) const -> std::expected<double, EvaluationError>;
  ```
- **Fluent APIs**: Expression manipulation must support method chaining:
  ```cpp
  auto expr = x.add(y).mul(z).simplify();
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
| **NFR-2** | **Portability** | Build paths and module interfaces must be fully relative to the repository root. No absolute host-specific paths allowed. |
| **NFR-3** | **Reproducibility** | FP arithmetic must be bit-identical across hardware. No `-ffast-math` is allowed. Build flags must enforce `-march=x86-64-v3 -mtune=generic`. |
| **NFR-4** | **Sanitization** | The codebase must pass clean sweeps under AddressSanitizer (ASan), UndefinedBehaviorSanitizer (UBSan), and LeakSanitizer (LSan). |
