# NimbleCAS — C++23 high-performance Computer Algebra System.
# @author Olumuyiwa Oluwasanmi
#
# Build (on the clang-22 server): scripts/build.sh
# See docs/QUICKSTART.md.

cmake_minimum_required(VERSION 3.30)
project(NimbleCAS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Emit compile_commands.json for clang-tidy / editor tooling (Code Policy Rule 34).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CanonicalFlags)   # global canonical compile/link flags (Rule 50)
include(StdModule)        # std_module target providing `import std` (Rules 12/41)

# ---- nimblecas.core : basic types, CowPtr, Result/std::expected error handling ----
add_library(nimblecas_core STATIC)
target_sources(nimblecas_core PUBLIC
  FILE_SET CXX_MODULES FILES src/core/core.cppm)
target_link_libraries(nimblecas_core PUBLIC std_module)

# ---- nimblecas.parallel : fork-join runtime & tree combinators (TBB / PPL) ----
add_library(nimblecas_parallel STATIC)
target_sources(nimblecas_parallel PUBLIC
  FILE_SET CXX_MODULES FILES src/parallel/parallel.cppm)
target_link_libraries(nimblecas_parallel PUBLIC std_module)
if(NOT WIN32)
  # Intel oneTBB backend on Linux/macOS (PPL is header/CRT-provided on Windows).
  # Linked when present; if absent the module compiles to its serial fallback.
  find_library(NIMBLECAS_TBB_LIB NAMES tbb)
  if(NIMBLECAS_TBB_LIB)
    target_link_libraries(nimblecas_parallel PUBLIC "${NIMBLECAS_TBB_LIB}")
  else()
    message(WARNING "oneTBB not found; nimblecas.parallel will use the serial fallback.")
  endif()
endif()

# ---- nimblecas.symbolic : expression trees, FreeOf, Substitute (Cohen) ----
add_library(nimblecas_symbolic STATIC)
target_sources(nimblecas_symbolic PUBLIC
  FILE_SET CXX_MODULES FILES src/symbolic/symbolic.cppm)
target_link_libraries(nimblecas_symbolic PUBLIC nimblecas_parallel nimblecas_core std_module)

# ---- nimblecas.cache : concurrent hash-consing / memoization ----
add_library(nimblecas_cache STATIC)
target_sources(nimblecas_cache PUBLIC
  FILE_SET CXX_MODULES FILES src/cache/cache.cppm)
target_link_libraries(nimblecas_cache PUBLIC nimblecas_symbolic nimblecas_core std_module)

# ---- nimblecas.simplify : Cohen automatic simplification ----
add_library(nimblecas_simplify STATIC)
target_sources(nimblecas_simplify PUBLIC
  FILE_SET CXX_MODULES FILES src/simplify/simplify.cppm)
target_link_libraries(nimblecas_simplify
  PUBLIC nimblecas_symbolic nimblecas_parallel nimblecas_cache nimblecas_core std_module)

# ---- nimblecas.simd : multi-register SIMD engine with dynamic dispatch ----
add_library(nimblecas_simd STATIC)
target_sources(nimblecas_simd PUBLIC
  FILE_SET CXX_MODULES FILES src/simd/simd.cppm)
target_link_libraries(nimblecas_simd PUBLIC std_module)

# ---- nimblecas.polynomial : dense univariate polynomial arithmetic (SIMD eval) ----
add_library(nimblecas_polynomial STATIC)
target_sources(nimblecas_polynomial PUBLIC
  FILE_SET CXX_MODULES FILES src/polynomial/polynomial.cppm)
target_link_libraries(nimblecas_polynomial PUBLIC nimblecas_simd nimblecas_core std_module)

# ---- nimblecas.polyexpr : Expr <-> Polynomial bridge ----
add_library(nimblecas_polyexpr STATIC)
target_sources(nimblecas_polyexpr PUBLIC
  FILE_SET CXX_MODULES FILES src/polyexpr/polyexpr.cppm)
target_link_libraries(nimblecas_polyexpr
  PUBLIC nimblecas_polynomial nimblecas_symbolic nimblecas_core std_module)

# ---- nimblecas.ratpoly : polynomials over the rationals Q[x] (division, gcd) ----
add_library(nimblecas_ratpoly STATIC)
target_sources(nimblecas_ratpoly PUBLIC
  FILE_SET CXX_MODULES FILES src/ratpoly/ratpoly.cppm)
target_link_libraries(nimblecas_ratpoly
  PUBLIC nimblecas_polynomial nimblecas_core std_module)

# ---- nimblecas.matrix : dense exact-rational matrices (§7.2) ----
add_library(nimblecas_matrix STATIC)
target_sources(nimblecas_matrix PUBLIC
  FILE_SET CXX_MODULES FILES src/matrix/matrix.cppm)
target_link_libraries(nimblecas_matrix
  PUBLIC nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.stats : descriptive statistics & covariance matrix (§7.7) ----
add_library(nimblecas_stats STATIC)
target_sources(nimblecas_stats PUBLIC
  FILE_SET CXX_MODULES FILES src/stats/stats.cppm)
target_link_libraries(nimblecas_stats
  PUBLIC nimblecas_matrix nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.combinatorics : factorial/binomial/Bernoulli/Stirling (§7.18) ----
add_library(nimblecas_combinatorics STATIC)
target_sources(nimblecas_combinatorics PUBLIC
  FILE_SET CXX_MODULES FILES src/combinatorics/combinatorics.cppm)
target_link_libraries(nimblecas_combinatorics
  PUBLIC nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.orthopoly : classical orthogonal polynomials (§7.16) ----
add_library(nimblecas_orthopoly STATIC)
target_sources(nimblecas_orthopoly PUBLIC
  FILE_SET CXX_MODULES FILES src/orthopoly/orthopoly.cppm)
target_link_libraries(nimblecas_orthopoly
  PUBLIC nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.roots : rational-root finding over Q[x] (§7.21) ----
add_library(nimblecas_roots STATIC)
target_sources(nimblecas_roots PUBLIC
  FILE_SET CXX_MODULES FILES src/roots/roots.cppm)
target_link_libraries(nimblecas_roots
  PUBLIC nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.complex : exact Gaussian-rational complex numbers (§7.1) ----
add_library(nimblecas_complex STATIC)
target_sources(nimblecas_complex PUBLIC
  FILE_SET CXX_MODULES FILES src/complex/complex.cppm)
target_link_libraries(nimblecas_complex
  PUBLIC nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.recurrence : linear constant-coefficient recurrences (§7.9) ----
add_library(nimblecas_recurrence STATIC)
target_sources(nimblecas_recurrence PUBLIC
  FILE_SET CXX_MODULES FILES src/recurrence/recurrence.cppm)
target_link_libraries(nimblecas_recurrence
  PUBLIC nimblecas_roots nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.numeric : numeric root-finders (Newton/bisection/secant) (§7.14) ----
add_library(nimblecas_numeric STATIC)
target_sources(nimblecas_numeric PUBLIC
  FILE_SET CXX_MODULES FILES src/numeric/numeric.cppm)
target_link_libraries(nimblecas_numeric
  PUBLIC nimblecas_core std_module)

# ---- nimblecas.lp : exact-rational linear programming (Simplex) (§7.22) ----
add_library(nimblecas_lp STATIC)
target_sources(nimblecas_lp PUBLIC
  FILE_SET CXX_MODULES FILES src/lp/lp.cppm)
target_link_libraries(nimblecas_lp
  PUBLIC nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.rng : parallelisable counter-based RNG (Threefry-2x64) (§7.8 dependency) ----
add_library(nimblecas_rng STATIC)
target_sources(nimblecas_rng PUBLIC
  FILE_SET CXX_MODULES FILES src/rng/rng.cppm)
target_link_libraries(nimblecas_rng
  PUBLIC nimblecas_core std_module)

# ---- nimblecas.eigen : exact spectral primitives (Faddeev-LeVerrier) (§7.10 dependency) ----
add_library(nimblecas_eigen STATIC)
target_sources(nimblecas_eigen PUBLIC
  FILE_SET CXX_MODULES FILES src/eigen/eigen.cppm)
target_link_libraries(nimblecas_eigen
  PUBLIC nimblecas_matrix nimblecas_roots nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.sde : stochastic ODE integrators (Euler-Maruyama, Milstein) on the RNG ----
add_library(nimblecas_sde STATIC)
target_sources(nimblecas_sde PUBLIC
  FILE_SET CXX_MODULES FILES src/sde/sde.cppm)
target_link_libraries(nimblecas_sde
  PUBLIC nimblecas_rng nimblecas_core std_module)

# ---- nimblecas.montecarlo : Monte Carlo integration & rejection sampling (§7.8) ----
add_library(nimblecas_montecarlo STATIC)
target_sources(nimblecas_montecarlo PUBLIC
  FILE_SET CXX_MODULES FILES src/montecarlo/montecarlo.cppm)
target_link_libraries(nimblecas_montecarlo
  PUBLIC nimblecas_rng nimblecas_core std_module)

# ---- nimblecas.powerseries : truncated formal power series over Q (§7.4/7.5 dependency) ----
add_library(nimblecas_powerseries STATIC)
target_sources(nimblecas_powerseries PUBLIC
  FILE_SET CXX_MODULES FILES src/powerseries/powerseries.cppm)
target_link_libraries(nimblecas_powerseries
  PUBLIC nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.dynamics : linear/affine stability via Routh-Hurwitz + eigen (§7.10) ----
add_library(nimblecas_dynamics STATIC)
target_sources(nimblecas_dynamics PUBLIC
  FILE_SET CXX_MODULES FILES src/dynamics/dynamics.cppm)
target_link_libraries(nimblecas_dynamics
  PUBLIC nimblecas_eigen nimblecas_matrix nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.pade : [m/n] Pade approximants for resummation (§7.4/7.5) ----
add_library(nimblecas_pade STATIC)
target_sources(nimblecas_pade PUBLIC
  FILE_SET CXX_MODULES FILES src/pade/pade.cppm)
target_link_libraries(nimblecas_pade
  PUBLIC nimblecas_powerseries nimblecas_matrix nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.matexp : exact-rational matrix exponentials (§7.2/7.19) ----
add_library(nimblecas_matexp STATIC)
target_sources(nimblecas_matexp PUBLIC
  FILE_SET CXX_MODULES FILES src/matexp/matexp.cppm)
target_link_libraries(nimblecas_matexp
  PUBLIC nimblecas_matrix nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.mcmc : Metropolis-Hastings MCMC on the parallel RNG (§7.8) ----
add_library(nimblecas_mcmc STATIC)
target_sources(nimblecas_mcmc PUBLIC
  FILE_SET CXX_MODULES FILES src/mcmc/mcmc.cppm)
target_link_libraries(nimblecas_mcmc
  PUBLIC nimblecas_rng nimblecas_core std_module)

# ---- nimblecas.matdecomp : exact LU decomposition + structure predicates (§7.2) ----
add_library(nimblecas_matdecomp STATIC)
target_sources(nimblecas_matdecomp PUBLIC
  FILE_SET CXX_MODULES FILES src/matdecomp/matdecomp.cppm)
target_link_libraries(nimblecas_matdecomp
  PUBLIC nimblecas_matrix nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.cmatrix : complex (Gaussian-rational) matrices (§7.2, §7.15 dep) ----
add_library(nimblecas_cmatrix STATIC)
target_sources(nimblecas_cmatrix PUBLIC
  FILE_SET CXX_MODULES FILES src/cmatrix/cmatrix.cppm)
target_link_libraries(nimblecas_cmatrix
  PUBLIC nimblecas_complex nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.search : graph/tree search & dynamic programming (parallel algorithmics) ----
add_library(nimblecas_search STATIC)
target_sources(nimblecas_search PUBLIC
  FILE_SET CXX_MODULES FILES src/search/search.cppm)
target_link_libraries(nimblecas_search
  PUBLIC nimblecas_parallel nimblecas_core std_module)

# ---- nimblecas.csp : finite-domain constraint satisfaction (AC-3, backtracking, FC) ----
add_library(nimblecas_csp STATIC)
target_sources(nimblecas_csp PUBLIC
  FILE_SET CXX_MODULES FILES src/csp/csp.cppm)
target_link_libraries(nimblecas_csp
  PUBLIC nimblecas_parallel nimblecas_core std_module)

# ---- nimblecas.doubledouble : ~106-bit extended-precision float (SIMD-batched) ----
add_library(nimblecas_doubledouble STATIC)
target_sources(nimblecas_doubledouble PUBLIC
  FILE_SET CXX_MODULES FILES src/doubledouble/doubledouble.cppm)
target_link_libraries(nimblecas_doubledouble
  PUBLIC nimblecas_simd nimblecas_core std_module)

# ---- nimblecas.bigint : arbitrary-precision signed integer (Knuth Algorithm D, modpow) ----
add_library(nimblecas_bigint STATIC)
target_sources(nimblecas_bigint PUBLIC
  FILE_SET CXX_MODULES FILES src/bigint/bigint.cppm)
target_link_libraries(nimblecas_bigint PUBLIC nimblecas_core std_module)

# ---- nimblecas.bigfloat : arbitrary-precision software binary float on the BigInt mantissa ----
add_library(nimblecas_bigfloat STATIC)
target_sources(nimblecas_bigfloat PUBLIC
  FILE_SET CXX_MODULES FILES src/bigfloat/bigfloat.cppm)
target_link_libraries(nimblecas_bigfloat
  PUBLIC nimblecas_bigint nimblecas_core std_module)

# ---- nimblecas.numbertheory : crypto primitives on BigInt (modinv, Miller-Rabin, CRT, RSA) ----
add_library(nimblecas_numbertheory STATIC)
target_sources(nimblecas_numbertheory PUBLIC
  FILE_SET CXX_MODULES FILES src/numbertheory/numbertheory.cppm)
target_link_libraries(nimblecas_numbertheory
  PUBLIC nimblecas_bigint nimblecas_rng nimblecas_core std_module)

# ---- nimblecas.constants : arbitrary-precision math constants (pi, e, gamma, ...) on BigFloat ----
add_library(nimblecas_constants STATIC)
target_sources(nimblecas_constants PUBLIC
  FILE_SET CXX_MODULES FILES src/constants/constants.cppm)
target_link_libraries(nimblecas_constants
  PUBLIC nimblecas_bigfloat nimblecas_bigint nimblecas_core std_module)

# ---- nimblecas.bigrational : exact unbounded rational on BigInt (no overflow ceiling) ----
add_library(nimblecas_bigrational STATIC)
target_sources(nimblecas_bigrational PUBLIC
  FILE_SET CXX_MODULES FILES src/bigrational/bigrational.cppm)
target_link_libraries(nimblecas_bigrational
  PUBLIC nimblecas_bigint nimblecas_core std_module)

# ---- nimblecas.bigpowerseries : big-backed truncated power series Q[[x]]/(x^N) over BigRational --
add_library(nimblecas_bigpowerseries STATIC)
target_sources(nimblecas_bigpowerseries PUBLIC
  FILE_SET CXX_MODULES FILES src/bigpowerseries/bigpowerseries.cppm)
target_link_libraries(nimblecas_bigpowerseries
  PUBLIC nimblecas_bigrational nimblecas_bigint nimblecas_core std_module)

# ---- nimblecas.bigmatrix : big-backed dense matrix over BigRational (exact Bareiss determinant) --
add_library(nimblecas_bigmatrix STATIC)
target_sources(nimblecas_bigmatrix PUBLIC
  FILE_SET CXX_MODULES FILES src/bigmatrix/bigmatrix.cppm)
target_link_libraries(nimblecas_bigmatrix
  PUBLIC nimblecas_bigrational nimblecas_bigint nimblecas_matrix nimblecas_ratpoly
         nimblecas_core std_module)

# ---- nimblecas.bigeigen : exact characteristic polynomials & rational eigenvalues over BigRational --
add_library(nimblecas_bigeigen STATIC)
target_sources(nimblecas_bigeigen PUBLIC
  FILE_SET CXX_MODULES FILES src/bigeigen/bigeigen.cppm)
target_link_libraries(nimblecas_bigeigen
  PUBLIC nimblecas_bigmatrix nimblecas_bigrational nimblecas_bigint nimblecas_matrix
         nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.symconst : symbolic math constants (pi, e, gamma, phi) leaves + numeric bridge ----
add_library(nimblecas_symconst STATIC)
target_sources(nimblecas_symconst PUBLIC
  FILE_SET CXX_MODULES FILES src/symconst/symconst.cppm)
target_link_libraries(nimblecas_symconst
  PUBLIC nimblecas_symbolic nimblecas_constants nimblecas_bigfloat nimblecas_core std_module)

# ---- nimblecas.bigcombinatorics : unbounded exact integer combinatorics (BigInt-backed) ----
add_library(nimblecas_bigcombinatorics STATIC)
target_sources(nimblecas_bigcombinatorics PUBLIC
  FILE_SET CXX_MODULES FILES src/bigcombinatorics/bigcombinatorics.cppm)
target_link_libraries(nimblecas_bigcombinatorics
  PUBLIC nimblecas_bigint nimblecas_core std_module)

# ---- nimblecas.int128 : native 128-bit integer & rational (overflow-checked middle tier) ----
add_library(nimblecas_int128 STATIC)
target_sources(nimblecas_int128 PUBLIC
  FILE_SET CXX_MODULES FILES src/int128/int128.cppm)
target_link_libraries(nimblecas_int128 PUBLIC nimblecas_core std_module)

# ---- nimblecas.bitset : branchless / bit-parallel fixed-capacity bit set ----
add_library(nimblecas_bitset STATIC)
target_sources(nimblecas_bitset PUBLIC
  FILE_SET CXX_MODULES FILES src/bitset/bitset.cppm)
target_link_libraries(nimblecas_bitset PUBLIC nimblecas_core std_module)

# ---- nimblecas.bitcsp : branchless bitset-domain CSP (AC-3, bitmask N-queens) ----
add_library(nimblecas_bitcsp STATIC)
target_sources(nimblecas_bitcsp PUBLIC
  FILE_SET CXX_MODULES FILES src/bitcsp/bitcsp.cppm)
target_link_libraries(nimblecas_bitcsp
  PUBLIC nimblecas_bitset nimblecas_parallel nimblecas_core std_module)

# ---- nimblecas.sat : internal parallel SAT solver (DPLL/CDCL/WalkSAT/GSAT, distributed) ----
add_library(nimblecas_sat STATIC)
target_sources(nimblecas_sat PUBLIC
  FILE_SET CXX_MODULES FILES src/sat/sat.cppm)
target_link_libraries(nimblecas_sat
  PUBLIC nimblecas_rng nimblecas_parallel nimblecas_core std_module)

# ---- nimblecas.logic : logic programming — unification + SLD resolution (OR-parallel) ----
add_library(nimblecas_logic STATIC)
target_sources(nimblecas_logic PUBLIC
  FILE_SET CXX_MODULES FILES src/logic/logic.cppm)
target_link_libraries(nimblecas_logic
  PUBLIC nimblecas_parallel nimblecas_core std_module)

# ---- nimblecas.bandsolve : exact banded & tridiagonal linear solvers (§7.2) ----
add_library(nimblecas_bandsolve STATIC)
target_sources(nimblecas_bandsolve PUBLIC
  FILE_SET CXX_MODULES FILES src/bandsolve/bandsolve.cppm)
target_link_libraries(nimblecas_bandsolve
  PUBLIC nimblecas_matrix nimblecas_ratpoly nimblecas_parallel nimblecas_core std_module)

# ---- nimblecas.perturbation : ADM/HPM/HAM ODE solvers via exact power series (§7) ----
add_library(nimblecas_perturbation STATIC)
target_sources(nimblecas_perturbation PUBLIC
  FILE_SET CXX_MODULES FILES src/perturbation/perturbation.cppm)
target_link_libraries(nimblecas_perturbation
  PUBLIC nimblecas_powerseries nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.ode : exact power-series ODE systems & higher-order solver ----
add_library(nimblecas_ode STATIC)
target_sources(nimblecas_ode PUBLIC
  FILE_SET CXX_MODULES FILES src/ode/ode.cppm)
target_link_libraries(nimblecas_ode
  PUBLIC nimblecas_powerseries nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.dde : exact delay differential equations via the method of steps ----
add_library(nimblecas_dde STATIC)
target_sources(nimblecas_dde PUBLIC
  FILE_SET CXX_MODULES FILES src/dde/dde.cppm)
target_link_libraries(nimblecas_dde
  PUBLIC nimblecas_powerseries nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.dae : exact linear index-1 differential-algebraic equations ----
add_library(nimblecas_dae STATIC)
target_sources(nimblecas_dae PUBLIC
  FILE_SET CXX_MODULES FILES src/dae/dae.cppm)
target_link_libraries(nimblecas_dae
  PUBLIC nimblecas_ode nimblecas_matrix nimblecas_powerseries nimblecas_ratpoly
         nimblecas_core std_module)

# ---- nimblecas.pde : exact linear evolution PDEs via power series in time ----
add_library(nimblecas_pde STATIC)
target_sources(nimblecas_pde PUBLIC
  FILE_SET CXX_MODULES FILES src/pde/pde.cppm)
target_link_libraries(nimblecas_pde
  PUBLIC nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.pfd : partial-fraction decomposition over Q[x] (Yun + Bezout) ----
add_library(nimblecas_pfd STATIC)
target_sources(nimblecas_pfd PUBLIC
  FILE_SET CXX_MODULES FILES src/pfd/pfd.cppm)
target_link_libraries(nimblecas_pfd
  PUBLIC nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.resultant : resultant & discriminant over Q[x] ----
add_library(nimblecas_resultant STATIC)
target_sources(nimblecas_resultant PUBLIC
  FILE_SET CXX_MODULES FILES src/resultant/resultant.cppm)
target_link_libraries(nimblecas_resultant
  PUBLIC nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.ratint : rational-function integration (Hermite reduction) ----
add_library(nimblecas_ratint STATIC)
target_sources(nimblecas_ratint PUBLIC
  FILE_SET CXX_MODULES FILES src/ratint/ratint.cppm)
target_link_libraries(nimblecas_ratint
  PUBLIC nimblecas_pfd nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.rothstein : Rothstein-Trager logarithmic integration ----
add_library(nimblecas_rothstein STATIC)
target_sources(nimblecas_rothstein PUBLIC
  FILE_SET CXX_MODULES FILES src/rothstein/rothstein.cppm)
target_link_libraries(nimblecas_rothstein
  PUBLIC nimblecas_resultant nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.integrate : rational-function integration capstone (Hermite + RT) ----
add_library(nimblecas_integrate STATIC)
target_sources(nimblecas_integrate PUBLIC
  FILE_SET CXX_MODULES FILES src/integrate/integrate.cppm)
target_link_libraries(nimblecas_integrate
  PUBLIC nimblecas_ratint nimblecas_rothstein nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.diff : symbolic differentiation ----
add_library(nimblecas_diff STATIC)
target_sources(nimblecas_diff PUBLIC
  FILE_SET CXX_MODULES FILES src/diff/diff.cppm)
target_link_libraries(nimblecas_diff
  PUBLIC nimblecas_simplify nimblecas_symbolic nimblecas_parallel nimblecas_cache
         nimblecas_core std_module)

# ---- nimblecas.vectorcalc : vector calculus (grad/div/curl/laplacian/jacobian/hessian) ----
add_library(nimblecas_vectorcalc STATIC)
target_sources(nimblecas_vectorcalc PUBLIC
  FILE_SET CXX_MODULES FILES src/vectorcalc/vectorcalc.cppm)
target_link_libraries(nimblecas_vectorcalc
  PUBLIC nimblecas_diff nimblecas_simplify nimblecas_symbolic nimblecas_core std_module)

# ---- nimblecas.svgplot : in-core standalone SVG-string plotter (visualization) ----
add_library(nimblecas_svgplot STATIC)
target_sources(nimblecas_svgplot PUBLIC
  FILE_SET CXX_MODULES FILES src/svgplot/svgplot.cppm)
target_link_libraries(nimblecas_svgplot PUBLIC nimblecas_core std_module)

# ---- nimblecas.latex : Expr -> LaTeX exporter (§7.12) ----
add_library(nimblecas_latex STATIC)
target_sources(nimblecas_latex PUBLIC
  FILE_SET CXX_MODULES FILES src/latex/latex.cppm)
target_link_libraries(nimblecas_latex
  PUBLIC nimblecas_symbolic nimblecas_core std_module)

# ---- nimblecas.series : Taylor series via the diff engine (§7.3) ----
add_library(nimblecas_series STATIC)
target_sources(nimblecas_series PUBLIC
  FILE_SET CXX_MODULES FILES src/series/series.cppm)
target_link_libraries(nimblecas_series
  PUBLIC nimblecas_diff nimblecas_simplify nimblecas_symbolic nimblecas_core std_module)

# ---- nimblecas.laplace : table-driven symbolic Laplace transform (§7.6) ----
add_library(nimblecas_laplace STATIC)
target_sources(nimblecas_laplace PUBLIC
  FILE_SET CXX_MODULES FILES src/laplace/laplace.cppm)
target_link_libraries(nimblecas_laplace
  PUBLIC nimblecas_simplify nimblecas_symbolic nimblecas_core std_module)

# ---- nimblecas.testing : internal test framework (Rule 39 — no GTest/Catch2) ----
add_library(nimblecas_testing STATIC)
target_sources(nimblecas_testing PUBLIC
  FILE_SET CXX_MODULES FILES src/testing/testing.cppm)
target_link_libraries(nimblecas_testing PUBLIC std_module)

# ---- tests ----
enable_testing()

add_executable(core_tests tests/core_tests.cpp)
target_link_libraries(core_tests PRIVATE nimblecas_core nimblecas_testing std_module)
add_test(NAME core_tests COMMAND core_tests)

add_executable(matrix_tests tests/matrix_tests.cpp)
target_link_libraries(matrix_tests
  PRIVATE nimblecas_matrix nimblecas_ratpoly nimblecas_polynomial nimblecas_core
          nimblecas_testing std_module)
add_test(NAME matrix_tests COMMAND matrix_tests)

add_executable(stats_tests tests/stats_tests.cpp)
target_link_libraries(stats_tests
  PRIVATE nimblecas_stats nimblecas_matrix nimblecas_ratpoly nimblecas_polynomial
          nimblecas_core nimblecas_testing std_module)
add_test(NAME stats_tests COMMAND stats_tests)

add_executable(combinatorics_tests tests/combinatorics_tests.cpp)
target_link_libraries(combinatorics_tests
  PRIVATE nimblecas_combinatorics nimblecas_ratpoly nimblecas_polynomial nimblecas_core
          nimblecas_testing std_module)
add_test(NAME combinatorics_tests COMMAND combinatorics_tests)

add_executable(orthopoly_tests tests/orthopoly_tests.cpp)
target_link_libraries(orthopoly_tests
  PRIVATE nimblecas_orthopoly nimblecas_ratpoly nimblecas_polynomial nimblecas_core
          nimblecas_testing std_module)
add_test(NAME orthopoly_tests COMMAND orthopoly_tests)

add_executable(roots_tests tests/roots_tests.cpp)
target_link_libraries(roots_tests
  PRIVATE nimblecas_roots nimblecas_ratpoly nimblecas_polynomial nimblecas_core
          nimblecas_testing std_module)
add_test(NAME roots_tests COMMAND roots_tests)

add_executable(complex_tests tests/complex_tests.cpp)
target_link_libraries(complex_tests
  PRIVATE nimblecas_complex nimblecas_ratpoly nimblecas_polynomial nimblecas_core
          nimblecas_testing std_module)
add_test(NAME complex_tests COMMAND complex_tests)

add_executable(recurrence_tests tests/recurrence_tests.cpp)
target_link_libraries(recurrence_tests
  PRIVATE nimblecas_recurrence nimblecas_roots nimblecas_ratpoly nimblecas_polynomial
          nimblecas_core nimblecas_testing std_module)
add_test(NAME recurrence_tests COMMAND recurrence_tests)

add_executable(numeric_tests tests/numeric_tests.cpp)
target_link_libraries(numeric_tests
  PRIVATE nimblecas_numeric nimblecas_core nimblecas_testing std_module)
add_test(NAME numeric_tests COMMAND numeric_tests)

add_executable(lp_tests tests/lp_tests.cpp)
target_link_libraries(lp_tests
  PRIVATE nimblecas_lp nimblecas_ratpoly nimblecas_polynomial nimblecas_core
          nimblecas_testing std_module)
add_test(NAME lp_tests COMMAND lp_tests)

add_executable(rng_tests tests/rng_tests.cpp)
target_link_libraries(rng_tests
  PRIVATE nimblecas_rng nimblecas_core nimblecas_testing std_module)
add_test(NAME rng_tests COMMAND rng_tests)

add_executable(eigen_tests tests/eigen_tests.cpp)
target_link_libraries(eigen_tests
  PRIVATE nimblecas_eigen nimblecas_matrix nimblecas_roots nimblecas_ratpoly
          nimblecas_core nimblecas_testing std_module)
add_test(NAME eigen_tests COMMAND eigen_tests)

add_executable(powerseries_tests tests/powerseries_tests.cpp)
target_link_libraries(powerseries_tests
  PRIVATE nimblecas_powerseries nimblecas_ratpoly nimblecas_polynomial nimblecas_core
          nimblecas_testing std_module)
add_test(NAME powerseries_tests COMMAND powerseries_tests)

add_executable(montecarlo_tests tests/montecarlo_tests.cpp)
target_link_libraries(montecarlo_tests
  PRIVATE nimblecas_montecarlo nimblecas_rng nimblecas_core nimblecas_testing std_module)
add_test(NAME montecarlo_tests COMMAND montecarlo_tests)

add_executable(dynamics_tests tests/dynamics_tests.cpp)
target_link_libraries(dynamics_tests
  PRIVATE nimblecas_dynamics nimblecas_eigen nimblecas_matrix nimblecas_ratpoly
          nimblecas_core nimblecas_testing std_module)
add_test(NAME dynamics_tests COMMAND dynamics_tests)

add_executable(pade_tests tests/pade_tests.cpp)
target_link_libraries(pade_tests
  PRIVATE nimblecas_pade nimblecas_powerseries nimblecas_matrix nimblecas_ratpoly
          nimblecas_polynomial nimblecas_core nimblecas_testing std_module)
add_test(NAME pade_tests COMMAND pade_tests)

add_executable(matexp_tests tests/matexp_tests.cpp)
target_link_libraries(matexp_tests
  PRIVATE nimblecas_matexp nimblecas_matrix nimblecas_ratpoly nimblecas_polynomial
          nimblecas_core nimblecas_testing std_module)
add_test(NAME matexp_tests COMMAND matexp_tests)

add_executable(mcmc_tests tests/mcmc_tests.cpp)
target_link_libraries(mcmc_tests
  PRIVATE nimblecas_mcmc nimblecas_rng nimblecas_core nimblecas_testing std_module)
add_test(NAME mcmc_tests COMMAND mcmc_tests)

add_executable(matdecomp_tests tests/matdecomp_tests.cpp)
target_link_libraries(matdecomp_tests
  PRIVATE nimblecas_matdecomp nimblecas_matrix nimblecas_ratpoly nimblecas_polynomial
          nimblecas_core nimblecas_testing std_module)
add_test(NAME matdecomp_tests COMMAND matdecomp_tests)

add_executable(cmatrix_tests tests/cmatrix_tests.cpp)
target_link_libraries(cmatrix_tests
  PRIVATE nimblecas_cmatrix nimblecas_complex nimblecas_ratpoly nimblecas_polynomial
          nimblecas_core nimblecas_testing std_module)
add_test(NAME cmatrix_tests COMMAND cmatrix_tests)

add_executable(logic_tests tests/logic_tests.cpp)
target_link_libraries(logic_tests
  PRIVATE nimblecas_logic nimblecas_parallel nimblecas_core nimblecas_testing std_module)
add_test(NAME logic_tests COMMAND logic_tests)

add_executable(doubledouble_tests tests/doubledouble_tests.cpp)
target_link_libraries(doubledouble_tests
  PRIVATE nimblecas_doubledouble nimblecas_simd nimblecas_core nimblecas_testing std_module)
add_test(NAME doubledouble_tests COMMAND doubledouble_tests)

add_executable(bigint_tests tests/bigint_tests.cpp)
target_link_libraries(bigint_tests
  PRIVATE nimblecas_bigint nimblecas_core nimblecas_testing std_module)
add_test(NAME bigint_tests COMMAND bigint_tests)

add_executable(bigcombinatorics_tests tests/bigcombinatorics_tests.cpp)
target_link_libraries(bigcombinatorics_tests
  PRIVATE nimblecas_bigcombinatorics nimblecas_combinatorics nimblecas_bigint nimblecas_ratpoly
          nimblecas_polynomial nimblecas_core nimblecas_testing std_module)
add_test(NAME bigcombinatorics_tests COMMAND bigcombinatorics_tests)

add_executable(bigmatrix_tests tests/bigmatrix_tests.cpp)
target_link_libraries(bigmatrix_tests
  PRIVATE nimblecas_bigmatrix nimblecas_matrix nimblecas_bigrational nimblecas_bigint
          nimblecas_ratpoly nimblecas_polynomial nimblecas_core nimblecas_testing std_module)
add_test(NAME bigmatrix_tests COMMAND bigmatrix_tests)

add_executable(bigeigen_tests tests/bigeigen_tests.cpp)
target_link_libraries(bigeigen_tests
  PRIVATE nimblecas_bigeigen nimblecas_bigmatrix nimblecas_bigrational nimblecas_bigint
          nimblecas_matrix nimblecas_ratpoly nimblecas_polynomial nimblecas_core
          nimblecas_testing std_module)
add_test(NAME bigeigen_tests COMMAND bigeigen_tests)

add_executable(symconst_tests tests/symconst_tests.cpp)
target_link_libraries(symconst_tests
  PRIVATE nimblecas_symconst nimblecas_constants nimblecas_bigfloat nimblecas_bigint
          nimblecas_diff nimblecas_simplify nimblecas_symbolic nimblecas_cache
          nimblecas_parallel nimblecas_core nimblecas_testing std_module)
add_test(NAME symconst_tests COMMAND symconst_tests)

add_executable(bigpowerseries_tests tests/bigpowerseries_tests.cpp)
target_link_libraries(bigpowerseries_tests
  PRIVATE nimblecas_bigpowerseries nimblecas_powerseries nimblecas_bigrational nimblecas_bigint
          nimblecas_ratpoly nimblecas_polynomial nimblecas_core nimblecas_testing std_module)
add_test(NAME bigpowerseries_tests COMMAND bigpowerseries_tests)

add_executable(bigrational_tests tests/bigrational_tests.cpp)
target_link_libraries(bigrational_tests
  PRIVATE nimblecas_bigrational nimblecas_bigint nimblecas_ratpoly nimblecas_polynomial
          nimblecas_core nimblecas_testing std_module)
add_test(NAME bigrational_tests COMMAND bigrational_tests)

add_executable(numbertheory_tests tests/numbertheory_tests.cpp)
target_link_libraries(numbertheory_tests
  PRIVATE nimblecas_numbertheory nimblecas_bigint nimblecas_rng nimblecas_core
          nimblecas_testing std_module)
add_test(NAME numbertheory_tests COMMAND numbertheory_tests)

add_executable(bigfloat_tests tests/bigfloat_tests.cpp)
target_link_libraries(bigfloat_tests
  PRIVATE nimblecas_bigfloat nimblecas_bigint nimblecas_core nimblecas_testing std_module)
add_test(NAME bigfloat_tests COMMAND bigfloat_tests)

add_executable(constants_tests tests/constants_tests.cpp)
target_link_libraries(constants_tests
  PRIVATE nimblecas_constants nimblecas_bigfloat nimblecas_bigint nimblecas_core
          nimblecas_testing std_module)
add_test(NAME constants_tests COMMAND constants_tests)

add_executable(int128_tests tests/int128_tests.cpp)
target_link_libraries(int128_tests
  PRIVATE nimblecas_int128 nimblecas_ratpoly nimblecas_polynomial nimblecas_core
          nimblecas_testing std_module)
add_test(NAME int128_tests COMMAND int128_tests)

add_executable(bitset_tests tests/bitset_tests.cpp)
target_link_libraries(bitset_tests
  PRIVATE nimblecas_bitset nimblecas_core nimblecas_testing std_module)
add_test(NAME bitset_tests COMMAND bitset_tests)

add_executable(bitcsp_tests tests/bitcsp_tests.cpp)
target_link_libraries(bitcsp_tests
  PRIVATE nimblecas_bitcsp nimblecas_bitset nimblecas_csp nimblecas_parallel nimblecas_core
          nimblecas_testing std_module)
add_test(NAME bitcsp_tests COMMAND bitcsp_tests)

add_executable(sat_tests tests/sat_tests.cpp)
target_link_libraries(sat_tests
  PRIVATE nimblecas_sat nimblecas_rng nimblecas_parallel nimblecas_core nimblecas_testing
          std_module)
add_test(NAME sat_tests COMMAND sat_tests)

add_executable(search_tests tests/search_tests.cpp)
target_link_libraries(search_tests
  PRIVATE nimblecas_search nimblecas_parallel nimblecas_core nimblecas_testing std_module)
add_test(NAME search_tests COMMAND search_tests)

add_executable(csp_tests tests/csp_tests.cpp)
target_link_libraries(csp_tests
  PRIVATE nimblecas_csp nimblecas_parallel nimblecas_core nimblecas_testing std_module)
add_test(NAME csp_tests COMMAND csp_tests)

add_executable(bandsolve_tests tests/bandsolve_tests.cpp)
target_link_libraries(bandsolve_tests
  PRIVATE nimblecas_bandsolve nimblecas_matrix nimblecas_ratpoly nimblecas_polynomial
          nimblecas_parallel nimblecas_core nimblecas_testing std_module)
add_test(NAME bandsolve_tests COMMAND bandsolve_tests)

add_executable(sde_tests tests/sde_tests.cpp)
target_link_libraries(sde_tests
  PRIVATE nimblecas_sde nimblecas_rng nimblecas_core nimblecas_testing std_module)
add_test(NAME sde_tests COMMAND sde_tests)

add_executable(dae_tests tests/dae_tests.cpp)
target_link_libraries(dae_tests
  PRIVATE nimblecas_dae nimblecas_ode nimblecas_matrix nimblecas_powerseries nimblecas_ratpoly
          nimblecas_polynomial nimblecas_core nimblecas_testing std_module)
add_test(NAME dae_tests COMMAND dae_tests)

add_executable(pde_tests tests/pde_tests.cpp)
target_link_libraries(pde_tests
  PRIVATE nimblecas_pde nimblecas_ratpoly nimblecas_polynomial nimblecas_core
          nimblecas_testing std_module)
add_test(NAME pde_tests COMMAND pde_tests)

add_executable(dde_tests tests/dde_tests.cpp)
target_link_libraries(dde_tests
  PRIVATE nimblecas_dde nimblecas_powerseries nimblecas_ratpoly nimblecas_polynomial
          nimblecas_core nimblecas_testing std_module)
add_test(NAME dde_tests COMMAND dde_tests)

add_executable(ode_tests tests/ode_tests.cpp)
target_link_libraries(ode_tests
  PRIVATE nimblecas_ode nimblecas_perturbation nimblecas_powerseries nimblecas_ratpoly
          nimblecas_polynomial nimblecas_core nimblecas_testing std_module)
add_test(NAME ode_tests COMMAND ode_tests)

add_executable(perturbation_tests tests/perturbation_tests.cpp)
target_link_libraries(perturbation_tests
  PRIVATE nimblecas_perturbation nimblecas_powerseries nimblecas_ratpoly nimblecas_polynomial
          nimblecas_core nimblecas_testing std_module)
add_test(NAME perturbation_tests COMMAND perturbation_tests)

add_executable(svgplot_tests tests/svgplot_tests.cpp)
target_link_libraries(svgplot_tests
  PRIVATE nimblecas_svgplot nimblecas_core nimblecas_testing std_module)
add_test(NAME svgplot_tests COMMAND svgplot_tests)

add_executable(latex_tests tests/latex_tests.cpp)
target_link_libraries(latex_tests
  PRIVATE nimblecas_latex nimblecas_symbolic nimblecas_core nimblecas_testing std_module)
add_test(NAME latex_tests COMMAND latex_tests)

add_executable(series_tests tests/series_tests.cpp)
target_link_libraries(series_tests
  PRIVATE nimblecas_series nimblecas_diff nimblecas_simplify nimblecas_symbolic
          nimblecas_cache nimblecas_parallel nimblecas_core nimblecas_testing std_module)
add_test(NAME series_tests COMMAND series_tests)

add_executable(laplace_tests tests/laplace_tests.cpp)
target_link_libraries(laplace_tests
  PRIVATE nimblecas_laplace nimblecas_simplify nimblecas_symbolic nimblecas_cache
          nimblecas_parallel nimblecas_core nimblecas_testing std_module)
add_test(NAME laplace_tests COMMAND laplace_tests)

add_executable(symbolic_tests tests/symbolic_tests.cpp)
target_link_libraries(symbolic_tests
  PRIVATE nimblecas_symbolic nimblecas_core nimblecas_testing std_module)
add_test(NAME symbolic_tests COMMAND symbolic_tests)

add_executable(simplify_tests tests/simplify_tests.cpp)
target_link_libraries(simplify_tests
  PRIVATE nimblecas_simplify nimblecas_symbolic nimblecas_core nimblecas_testing std_module)
add_test(NAME simplify_tests COMMAND simplify_tests)

add_executable(cache_tests tests/cache_tests.cpp)
target_link_libraries(cache_tests
  PRIVATE nimblecas_cache nimblecas_symbolic nimblecas_core nimblecas_testing std_module)
add_test(NAME cache_tests COMMAND cache_tests)

add_executable(diff_tests tests/diff_tests.cpp)
target_link_libraries(diff_tests
  PRIVATE nimblecas_diff nimblecas_simplify nimblecas_symbolic nimblecas_core
          nimblecas_testing std_module)
add_test(NAME diff_tests COMMAND diff_tests)

add_executable(vectorcalc_tests tests/vectorcalc_tests.cpp)
target_link_libraries(vectorcalc_tests
  PRIVATE nimblecas_vectorcalc nimblecas_diff nimblecas_simplify nimblecas_symbolic
          nimblecas_cache nimblecas_parallel nimblecas_core nimblecas_testing std_module)
add_test(NAME vectorcalc_tests COMMAND vectorcalc_tests)

add_executable(simd_tests tests/simd_tests.cpp)
target_link_libraries(simd_tests PRIVATE nimblecas_simd nimblecas_testing std_module)
add_test(NAME simd_tests COMMAND simd_tests)

add_executable(polynomial_tests tests/polynomial_tests.cpp)
target_link_libraries(polynomial_tests
  PRIVATE nimblecas_polynomial nimblecas_simd nimblecas_core nimblecas_testing std_module)
add_test(NAME polynomial_tests COMMAND polynomial_tests)

add_executable(polyexpr_tests tests/polyexpr_tests.cpp)
target_link_libraries(polyexpr_tests
  PRIVATE nimblecas_polyexpr nimblecas_polynomial nimblecas_symbolic nimblecas_core
          nimblecas_testing std_module)
add_test(NAME polyexpr_tests COMMAND polyexpr_tests)

add_executable(ratpoly_tests tests/ratpoly_tests.cpp)
target_link_libraries(ratpoly_tests
  PRIVATE nimblecas_ratpoly nimblecas_polynomial nimblecas_core nimblecas_testing std_module)
add_test(NAME ratpoly_tests COMMAND ratpoly_tests)

add_executable(pfd_tests tests/pfd_tests.cpp)
target_link_libraries(pfd_tests
  PRIVATE nimblecas_pfd nimblecas_ratpoly nimblecas_polynomial nimblecas_core
          nimblecas_testing std_module)
add_test(NAME pfd_tests COMMAND pfd_tests)

add_executable(resultant_tests tests/resultant_tests.cpp)
target_link_libraries(resultant_tests
  PRIVATE nimblecas_resultant nimblecas_ratpoly nimblecas_polynomial nimblecas_core
          nimblecas_testing std_module)
add_test(NAME resultant_tests COMMAND resultant_tests)

add_executable(ratint_tests tests/ratint_tests.cpp)
target_link_libraries(ratint_tests
  PRIVATE nimblecas_ratint nimblecas_pfd nimblecas_ratpoly nimblecas_polynomial
          nimblecas_core nimblecas_testing std_module)
add_test(NAME ratint_tests COMMAND ratint_tests)

add_executable(rothstein_tests tests/rothstein_tests.cpp)
target_link_libraries(rothstein_tests
  PRIVATE nimblecas_rothstein nimblecas_resultant nimblecas_ratpoly nimblecas_polynomial
          nimblecas_core nimblecas_testing std_module)
add_test(NAME rothstein_tests COMMAND rothstein_tests)

add_executable(integrate_tests tests/integrate_tests.cpp)
target_link_libraries(integrate_tests
  PRIVATE nimblecas_integrate nimblecas_ratint nimblecas_rothstein nimblecas_pfd
          nimblecas_resultant nimblecas_ratpoly nimblecas_polynomial nimblecas_core
          nimblecas_testing std_module)
add_test(NAME integrate_tests COMMAND integrate_tests)

# ---- cross-module feature / integration tests ----
add_executable(feature_algebra_tests tests/feature_algebra_tests.cpp)
target_link_libraries(feature_algebra_tests
  PRIVATE nimblecas_complex nimblecas_ratpoly nimblecas_polynomial nimblecas_core
          nimblecas_testing std_module)
add_test(NAME feature_algebra_tests COMMAND feature_algebra_tests)

add_executable(feature_symbolic_tests tests/feature_symbolic_tests.cpp)
target_link_libraries(feature_symbolic_tests
  PRIVATE nimblecas_diff nimblecas_series nimblecas_laplace nimblecas_vectorcalc
          nimblecas_latex nimblecas_polyexpr nimblecas_simplify nimblecas_symbolic
          nimblecas_polynomial nimblecas_cache nimblecas_parallel nimblecas_core
          nimblecas_testing std_module)
add_test(NAME feature_symbolic_tests COMMAND feature_symbolic_tests)

add_executable(feature_linalg_tests tests/feature_linalg_tests.cpp)
target_link_libraries(feature_linalg_tests
  PRIVATE nimblecas_eigen nimblecas_matexp nimblecas_matdecomp nimblecas_bandsolve
          nimblecas_cmatrix nimblecas_lp nimblecas_stats nimblecas_matrix nimblecas_complex
          nimblecas_roots nimblecas_ratpoly nimblecas_polynomial nimblecas_parallel
          nimblecas_core nimblecas_testing std_module)
add_test(NAME feature_linalg_tests COMMAND feature_linalg_tests)

add_executable(feature_polycalc_tests tests/feature_polycalc_tests.cpp)
target_link_libraries(feature_polycalc_tests
  PRIVATE nimblecas_integrate nimblecas_ratint nimblecas_rothstein nimblecas_pfd
          nimblecas_resultant nimblecas_roots nimblecas_recurrence nimblecas_orthopoly
          nimblecas_combinatorics nimblecas_ratpoly nimblecas_polynomial nimblecas_core
          nimblecas_testing std_module)
add_test(NAME feature_polycalc_tests COMMAND feature_polycalc_tests)

add_executable(feature_numprob_tests tests/feature_numprob_tests.cpp)
target_link_libraries(feature_numprob_tests
  PRIVATE nimblecas_numeric nimblecas_rng nimblecas_montecarlo nimblecas_mcmc nimblecas_pade
          nimblecas_powerseries nimblecas_dynamics nimblecas_eigen nimblecas_matrix
          nimblecas_ratpoly nimblecas_polynomial nimblecas_parallel nimblecas_core
          nimblecas_testing std_module)
add_test(NAME feature_numprob_tests COMMAND feature_numprob_tests)

# CPU profiling harness (perf / VTune) — a plain executable, not a ctest target.
add_executable(cpu_bench tools/cpu_bench.cpp)
target_link_libraries(cpu_bench
  PRIVATE nimblecas_polynomial nimblecas_simd nimblecas_core std_module)

add_executable(parallel_tests tests/parallel_tests.cpp)
target_link_libraries(parallel_tests
  PRIVATE nimblecas_parallel nimblecas_testing std_module)
add_test(NAME parallel_tests COMMAND parallel_tests)

# ---- nimblecas.gpu : CUDA GPU acceleration (ROADMAP 5) — opt-in via -DNIMBLECAS_CUDA=ON --
# The .cu is compiled by nvcc INDEPENDENTLY (a custom command -> static archive), never
# through CMake's CUDA language, so the global clang/libc++ canonical flags (which nvcc
# cannot parse) never reach it. The kernels are reached over the plain C ABI of
# src/gpu/gpu_bridge.h; only POD crosses the nvcc <-> clang boundary.
option(NIMBLECAS_CUDA "Build the CUDA GPU kernels (requires nvcc)" OFF)
if(NIMBLECAS_CUDA AND NOT WIN32)
  set(NIMBLECAS_NVCC "" CACHE FILEPATH "Path to nvcc (auto-detected if empty)")
  if(NOT NIMBLECAS_NVCC)
    find_program(NIMBLECAS_NVCC nvcc
      HINTS /usr/local/cuda/bin /usr/local/cuda-13.2/bin ENV CUDA_HOME ENV PATH REQUIRED)
  endif()
  get_filename_component(_cuda_bin "${NIMBLECAS_NVCC}" DIRECTORY)
  find_library(NIMBLECAS_CUDART cudart
    HINTS "${_cuda_bin}/../lib64" /usr/local/cuda/lib64 /usr/local/cuda-13.2/lib64 REQUIRED)
  get_filename_component(_cuda_libdir "${NIMBLECAS_CUDART}" DIRECTORY)

  set(_gpu_src "${CMAKE_SOURCE_DIR}/src/gpu/gpu_kernels.cu")
  set(_gpu_lib "${CMAKE_BINARY_DIR}/libnimblecas_gpu_kernels.a")
  add_custom_command(
    OUTPUT "${_gpu_lib}"
    COMMAND "${NIMBLECAS_NVCC}" -O3 -std=c++17 -arch=native -Xcompiler -fPIC
            -I "${CMAKE_SOURCE_DIR}/src/gpu" -lib "${_gpu_src}" -o "${_gpu_lib}"
    DEPENDS "${_gpu_src}" "${CMAKE_SOURCE_DIR}/src/gpu/gpu_bridge.h"
    VERBATIM
    COMMENT "nvcc: building CUDA GPU kernels (-arch=native)")
  add_custom_target(nimblecas_gpu_kernels DEPENDS "${_gpu_lib}")

  add_library(nimblecas_gpu STATIC)
  target_sources(nimblecas_gpu PUBLIC FILE_SET CXX_MODULES FILES src/gpu/gpu.cppm)
  target_include_directories(nimblecas_gpu PRIVATE "${CMAKE_SOURCE_DIR}/src/gpu")
  add_dependencies(nimblecas_gpu nimblecas_gpu_kernels)
  target_link_libraries(nimblecas_gpu
    PUBLIC nimblecas_core std_module "${_gpu_lib}" "${NIMBLECAS_CUDART}")

  add_executable(gpu_tests tests/gpu_tests.cpp)
  add_dependencies(gpu_tests nimblecas_gpu_kernels)
  target_link_libraries(gpu_tests
    PRIVATE nimblecas_gpu nimblecas_core nimblecas_testing std_module)
  set_target_properties(gpu_tests PROPERTIES BUILD_RPATH "${_cuda_libdir}")
  add_test(NAME gpu_tests COMMAND gpu_tests)
endif()

# ---- Python bindings (nanobind, ROADMAP 10) — opt-in via -DNIMBLECAS_PYTHON=ON ----
option(NIMBLECAS_PYTHON "Build the nanobind Python bindings" OFF)
if(NIMBLECAS_PYTHON)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(nimblecas_ext src/bindings/bindings.cpp)
  target_link_libraries(nimblecas_ext
    PRIVATE nimblecas_diff nimblecas_simplify nimblecas_polyexpr nimblecas_polynomial
            nimblecas_symbolic nimblecas_simd nimblecas_core std_module)

  add_test(NAME python_bindings
           COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_bindings.py)
  set_tests_properties(python_bindings PROPERTIES
    ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:nimblecas_ext>")
endif()
