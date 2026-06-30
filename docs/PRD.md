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

### 3.3. GPU Acceleration (Co-processing Model)
- **GPU-CPU Co-processing**: The engine must support offloading heavy compute tasks to the GPU while the CPU manages expression tree transformations.
  - **Numerical Grids**: Evaluating a symbolic expression over millions of grid points must compile the expression into a vectorized kernel (CUDA/DirectX HLSL/SYCL) and run it on the GPU.
  - **Matrix and Linear Algebra**: Matrix exponentiation, symbolic/numerical determinants, and large linear solvers must utilize GPU-accelerated algorithms.

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
