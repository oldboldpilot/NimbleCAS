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

# ---- nimblecas.wavelets : wavelet transforms (Haar exact over Q; db/sym/coif/bior numerical) ----
add_library(nimblecas_wavelets STATIC)
target_sources(nimblecas_wavelets PUBLIC
  FILE_SET CXX_MODULES FILES src/wavelets/wavelets.cppm)
target_link_libraries(nimblecas_wavelets
  PUBLIC nimblecas_ratpoly nimblecas_polynomial nimblecas_parallel nimblecas_core std_module)

# ---- nimblecas.matrix : dense exact-rational matrices (§7.2) ----
add_library(nimblecas_matrix STATIC)
target_sources(nimblecas_matrix PUBLIC
  FILE_SET CXX_MODULES FILES src/matrix/matrix.cppm)
target_link_libraries(nimblecas_matrix
  PUBLIC nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.krylov : Krylov subspace methods (exact CG/Arnoldi/Lanczos over Q + numerical GMRES/etc.) ----
add_library(nimblecas_krylov STATIC)
target_sources(nimblecas_krylov PUBLIC
  FILE_SET CXX_MODULES FILES src/krylov/krylov.cppm)
target_link_libraries(nimblecas_krylov
  PUBLIC nimblecas_matrix nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.stats : descriptive statistics & covariance matrix (§7.7) ----
add_library(nimblecas_stats STATIC)
target_sources(nimblecas_stats PUBLIC
  FILE_SET CXX_MODULES FILES src/stats/stats.cppm)
target_link_libraries(nimblecas_stats
  PUBLIC nimblecas_matrix nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.stochastic : stochastic-process analysis (Markov/WSS/ARMA/PSD) ----
add_library(nimblecas_stochastic STATIC)
target_sources(nimblecas_stochastic PUBLIC
  FILE_SET CXX_MODULES FILES src/stochastic/stochastic.cppm)
target_link_libraries(nimblecas_stochastic
  PUBLIC nimblecas_stats nimblecas_matrix nimblecas_roots nimblecas_ratpoly
         nimblecas_polynomial nimblecas_core std_module)

# ---- nimblecas.hmm : discrete hidden Markov models, exact over Q ----
add_library(nimblecas_hmm STATIC)
target_sources(nimblecas_hmm PUBLIC
  FILE_SET CXX_MODULES FILES src/hmm/hmm.cppm)
target_link_libraries(nimblecas_hmm
  PUBLIC nimblecas_matrix nimblecas_ratpoly nimblecas_polynomial nimblecas_core std_module)

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

# ---- nimblecas.spectral : spectral methods (Galerkin/collocation/Fourier/SEM/DG) ----
add_library(nimblecas_spectral STATIC)
target_sources(nimblecas_spectral PUBLIC
  FILE_SET CXX_MODULES FILES src/spectral/spectral.cppm)
target_link_libraries(nimblecas_spectral
  PUBLIC nimblecas_orthopoly nimblecas_matrix nimblecas_ratpoly nimblecas_polynomial
         nimblecas_core std_module)

# ---- nimblecas.fft : fast Fourier transform (numeric radix-2 CT + Bluestein) (§7.3) ----
add_library(nimblecas_fft STATIC)
target_sources(nimblecas_fft PUBLIC
  FILE_SET CXX_MODULES FILES src/fft/fft.cppm)
target_link_libraries(nimblecas_fft
  PUBLIC nimblecas_core std_module)

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

# ---- nimblecas.quantum : exact non-commutative operator algebra (§7.15) ----
add_library(nimblecas_quantum STATIC)
target_sources(nimblecas_quantum PUBLIC
  FILE_SET CXX_MODULES FILES src/quantum/quantum.cppm)
target_link_libraries(nimblecas_quantum
  PUBLIC nimblecas_complex nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.recurrence : linear constant-coefficient recurrences (§7.9) ----
add_library(nimblecas_recurrence STATIC)
target_sources(nimblecas_recurrence PUBLIC
  FILE_SET CXX_MODULES FILES src/recurrence/recurrence.cppm)
target_link_libraries(nimblecas_recurrence
  PUBLIC nimblecas_symbolic nimblecas_matrix nimblecas_roots nimblecas_ratpoly
         nimblecas_core std_module)

# ---- nimblecas.numeric : numeric root-finders (Newton/bisection/secant) (§7.14) ----
add_library(nimblecas_numeric STATIC)
target_sources(nimblecas_numeric PUBLIC
  FILE_SET CXX_MODULES FILES src/numeric/numeric.cppm)
target_link_libraries(nimblecas_numeric
  PUBLIC nimblecas_core std_module)

# ---- nimblecas.optimize : unconstrained numerical optimization over R^n (+ implicit filtering) ----
add_library(nimblecas_optimize STATIC)
target_sources(nimblecas_optimize PUBLIC
  FILE_SET CXX_MODULES FILES src/optimize/optimize.cppm)
target_link_libraries(nimblecas_optimize
  PUBLIC nimblecas_parallel nimblecas_core std_module)

# ---- nimblecas.nlsolve : iterative solvers for F(x)=0 in R^n (Kelley) (§7.14) ----
add_library(nimblecas_nlsolve STATIC)
target_sources(nimblecas_nlsolve PUBLIC
  FILE_SET CXX_MODULES FILES src/nlsolve/nlsolve.cppm)
target_link_libraries(nimblecas_nlsolve
  PUBLIC nimblecas_parallel nimblecas_core std_module)

# ---- nimblecas.extrapolation : sequence acceleration (Richardson/Romberg/Aitken/Wynn) ----
add_library(nimblecas_extrapolation STATIC)
target_sources(nimblecas_extrapolation PUBLIC
  FILE_SET CXX_MODULES FILES src/extrapolation/extrapolation.cppm)
target_link_libraries(nimblecas_extrapolation
  PUBLIC nimblecas_ratpoly nimblecas_polynomial nimblecas_core std_module)

# ---- nimblecas.lp : exact-rational linear programming (Simplex) (§7.22) ----
add_library(nimblecas_lp STATIC)
target_sources(nimblecas_lp PUBLIC
  FILE_SET CXX_MODULES FILES src/lp/lp.cppm)
target_link_libraries(nimblecas_lp
  PUBLIC nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.compsense : compressed sensing / sparse recovery (exact BP via LP dual) ----
add_library(nimblecas_compsense STATIC)
target_sources(nimblecas_compsense PUBLIC
  FILE_SET CXX_MODULES FILES src/compsense/compsense.cppm)
target_link_libraries(nimblecas_compsense
  PUBLIC nimblecas_lp nimblecas_matrix nimblecas_ratpoly nimblecas_parallel nimblecas_core std_module)

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

# ---- nimblecas.qmc : quasi-Monte Carlo low-discrepancy sequences & adaptive integration (§7.8) ----
add_library(nimblecas_qmc STATIC)
target_sources(nimblecas_qmc PUBLIC
  FILE_SET CXX_MODULES FILES src/qmc/qmc.cppm)
target_link_libraries(nimblecas_qmc
  PUBLIC nimblecas_rng nimblecas_ratpoly nimblecas_parallel nimblecas_core std_module)

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

# ---- nimblecas.analysis : conditioning, convergence & Lyapunov stability (§7.10) ----
add_library(nimblecas_analysis STATIC)
target_sources(nimblecas_analysis PUBLIC
  FILE_SET CXX_MODULES FILES src/analysis/analysis.cppm)
target_link_libraries(nimblecas_analysis
  PUBLIC nimblecas_dynamics nimblecas_eigen nimblecas_matrix nimblecas_ratpoly
         nimblecas_polynomial nimblecas_core std_module)

# ---- nimblecas.semigroup : operator theory & C0-semigroups (finite-dim realization) ----
add_library(nimblecas_semigroup STATIC)
target_sources(nimblecas_semigroup PUBLIC
  FILE_SET CXX_MODULES FILES src/semigroup/semigroup.cppm)
target_link_libraries(nimblecas_semigroup
  PUBLIC nimblecas_matexp nimblecas_dynamics nimblecas_eigen nimblecas_matrix
         nimblecas_roots nimblecas_ratpoly nimblecas_polynomial nimblecas_core std_module)

# ---- nimblecas.control : linear control systems (TF/SS, stability criteria, Bode) ----
add_library(nimblecas_control STATIC)
target_sources(nimblecas_control PUBLIC
  FILE_SET CXX_MODULES FILES src/control/control.cppm)
target_link_libraries(nimblecas_control
  PUBLIC nimblecas_dynamics nimblecas_roots nimblecas_complex nimblecas_matrix
         nimblecas_ratpoly nimblecas_polynomial nimblecas_parallel nimblecas_core std_module)

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

# ---- nimblecas.lie : matrix Lie algebras & Lie transforms over Q ----
add_library(nimblecas_lie STATIC)
target_sources(nimblecas_lie PUBLIC
  FILE_SET CXX_MODULES FILES src/lie/lie.cppm)
target_link_libraries(nimblecas_lie
  PUBLIC nimblecas_matexp nimblecas_matrix nimblecas_ratpoly nimblecas_core std_module)

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

# ---- nimblecas.matstruct : structured matrices + exact LDL^T/Cholesky/Hessenberg (§7.2) ----
add_library(nimblecas_matstruct STATIC)
target_sources(nimblecas_matstruct PUBLIC
  FILE_SET CXX_MODULES FILES src/matstruct/matstruct.cppm)
target_link_libraries(nimblecas_matstruct
  PUBLIC nimblecas_matdecomp nimblecas_matrix nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.interpolation : exact polynomial interpolation over Q ----
add_library(nimblecas_interpolation STATIC)
target_sources(nimblecas_interpolation PUBLIC
  FILE_SET CXX_MODULES FILES src/interpolation/interpolation.cppm)
target_link_libraries(nimblecas_interpolation
  PUBLIC nimblecas_ratpoly nimblecas_polynomial nimblecas_core std_module)

# ---- nimblecas.splines : exact splines / Bezier / B-spline / NURBS over Q (§7) ----
add_library(nimblecas_splines STATIC)
target_sources(nimblecas_splines PUBLIC
  FILE_SET CXX_MODULES FILES src/splines/splines.cppm)
target_link_libraries(nimblecas_splines
  PUBLIC nimblecas_bandsolve nimblecas_matrix nimblecas_ratpoly nimblecas_polynomial
         nimblecas_core std_module)

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

# ---- nimblecas.pdenum : numerical PDE methods — FDM/FEM/FVM/method-of-lines (§7.2) ----
add_library(nimblecas_pdenum STATIC)
target_sources(nimblecas_pdenum PUBLIC
  FILE_SET CXX_MODULES FILES src/pdenum/pdenum.cppm)
target_link_libraries(nimblecas_pdenum
  PUBLIC nimblecas_bandsolve nimblecas_matrix nimblecas_ratpoly nimblecas_parallel
         nimblecas_core std_module)

# ---- nimblecas.inteq : integral equations (Fredholm/Volterra; ADM/HPM/HAM) ----
add_library(nimblecas_inteq STATIC)
target_sources(nimblecas_inteq PUBLIC
  FILE_SET CXX_MODULES FILES src/inteq/inteq.cppm)
target_link_libraries(nimblecas_inteq
  PUBLIC nimblecas_integrate nimblecas_laplace nimblecas_simplify nimblecas_matrix
         nimblecas_symbolic nimblecas_ratpoly nimblecas_core std_module)

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

# ---- nimblecas.limits : symbolic limits / L'Hopital (§7) ----
add_library(nimblecas_limits STATIC)
target_sources(nimblecas_limits PUBLIC
  FILE_SET CXX_MODULES FILES src/limits/limits.cppm)
target_link_libraries(nimblecas_limits
  PUBLIC nimblecas_diff nimblecas_simplify nimblecas_symbolic nimblecas_core std_module)

# ---- nimblecas.tensor : tensor calculus / differential geometry (§7, extends 7.19) ----
add_library(nimblecas_tensor STATIC)
target_sources(nimblecas_tensor PUBLIC
  FILE_SET CXX_MODULES FILES src/tensor/tensor.cppm)
target_link_libraries(nimblecas_tensor
  PUBLIC nimblecas_diff nimblecas_simplify nimblecas_symbolic nimblecas_core std_module)

# ---- nimblecas.forms : differential forms / exterior calculus (wedge/d/hodge) ----
add_library(nimblecas_forms STATIC)
target_sources(nimblecas_forms PUBLIC
  FILE_SET CXX_MODULES FILES src/forms/forms.cppm)
target_link_libraries(nimblecas_forms
  PUBLIC nimblecas_diff nimblecas_simplify nimblecas_symbolic nimblecas_core std_module)

# ---- nimblecas.smc : Sequential Monte Carlo / particle filtering (§7.8, numerical) ----
add_library(nimblecas_smc STATIC)
target_sources(nimblecas_smc PUBLIC
  FILE_SET CXX_MODULES FILES src/smc/smc.cppm)
target_link_libraries(nimblecas_smc
  PUBLIC nimblecas_rng nimblecas_core std_module)

# ---- nimblecas.reader : text -> Expr parser (eval surface for REPL / execdoc) ----
add_library(nimblecas_reader STATIC)
target_sources(nimblecas_reader PUBLIC
  FILE_SET CXX_MODULES FILES src/reader/reader.cppm)
target_link_libraries(nimblecas_reader
  PUBLIC nimblecas_symbolic nimblecas_core std_module)

# ---- nimblecas.execdoc : executable document engine / live notebooks (§7.13) ----
add_library(nimblecas_execdoc STATIC)
target_sources(nimblecas_execdoc PUBLIC
  FILE_SET CXX_MODULES FILES src/execdoc/execdoc.cppm)
target_link_libraries(nimblecas_execdoc
  PUBLIC nimblecas_reader nimblecas_latex nimblecas_diff nimblecas_simplify
         nimblecas_symbolic nimblecas_core std_module)

# ---- nimblecas.mechanics : Hamiltonian mechanics, phase space, action-angle ----
add_library(nimblecas_mechanics STATIC)
target_sources(nimblecas_mechanics PUBLIC
  FILE_SET CXX_MODULES FILES src/mechanics/mechanics.cppm)
target_link_libraries(nimblecas_mechanics
  PUBLIC nimblecas_diff nimblecas_simplify nimblecas_symbolic nimblecas_core std_module)

# ---- nimblecas.calcvar : calculus of variations (Euler-Lagrange, Beltrami, constraints) ----
add_library(nimblecas_calcvar STATIC)
target_sources(nimblecas_calcvar PUBLIC
  FILE_SET CXX_MODULES FILES src/calcvar/calcvar.cppm)
target_link_libraries(nimblecas_calcvar
  PUBLIC nimblecas_diff nimblecas_simplify nimblecas_symbolic nimblecas_cache
         nimblecas_parallel nimblecas_core std_module)

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

# ---- nimblecas.webexport : JSON export bridge for the WebGPU/WASM front-end ----
add_library(nimblecas_webexport STATIC)
target_sources(nimblecas_webexport PUBLIC
  FILE_SET CXX_MODULES FILES src/webexport/webexport.cppm)
target_link_libraries(nimblecas_webexport
  PUBLIC nimblecas_latex nimblecas_symbolic nimblecas_core std_module)

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
  PUBLIC nimblecas_pfd nimblecas_ratpoly nimblecas_simplify nimblecas_symbolic
         nimblecas_core std_module)

# ---- nimblecas.testing : internal test framework (Rule 39 — no GTest/Catch2) ----
add_library(nimblecas_testing STATIC)
target_sources(nimblecas_testing PUBLIC
  FILE_SET CXX_MODULES FILES src/testing/testing.cppm)
target_link_libraries(nimblecas_testing PUBLIC std_module)

# ---- nimblecas.numeigen : numeric all-eigenvalue solver (structure-aware QR/Jacobi) (§7.2) ----
add_library(nimblecas_numeigen STATIC)
target_sources(nimblecas_numeigen PUBLIC
  FILE_SET CXX_MODULES FILES src/numeigen/numeigen.cppm)
target_link_libraries(nimblecas_numeigen
  PUBLIC nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.solve : exact closed-form roots (radicals ≤ quartic) + companion eigenvalues (§7.21) ----
add_library(nimblecas_solve STATIC)
target_sources(nimblecas_solve PUBLIC
  FILE_SET CXX_MODULES FILES src/solve/solve.cppm)
target_link_libraries(nimblecas_solve
  PUBLIC nimblecas_factor nimblecas_numeigen nimblecas_roots nimblecas_symbolic
         nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.ipm : primal–dual interior-point LP (Mehrotra, numerical) (§7.22) ----
add_library(nimblecas_ipm STATIC)
target_sources(nimblecas_ipm PUBLIC
  FILE_SET CXX_MODULES FILES src/ipm/ipm.cppm)
target_link_libraries(nimblecas_ipm
  PUBLIC nimblecas_core std_module)

# ---- nimblecas.singpert : singular perturbation, matched asymptotic expansions (§7.4) ----
add_library(nimblecas_singpert STATIC)
target_sources(nimblecas_singpert PUBLIC
  FILE_SET CXX_MODULES FILES src/singpert/singpert.cppm)
target_link_libraries(nimblecas_singpert
  PUBLIC nimblecas_symbolic nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.probdist : distribution catalog, MGF/PGF, moments, cumulants, tail bounds (§7.7) ----
add_library(nimblecas_probdist STATIC)
target_sources(nimblecas_probdist PUBLIC
  FILE_SET CXX_MODULES FILES src/probdist/probdist.cppm)
target_link_libraries(nimblecas_probdist
  PUBLIC nimblecas_simplify nimblecas_diff nimblecas_symbolic nimblecas_core std_module)

# ---- nimblecas.expand : algebraic expansion of an Expr (distribute + multinomial) (§7.20) ----
add_library(nimblecas_expand STATIC)
target_sources(nimblecas_expand PUBLIC
  FILE_SET CXX_MODULES FILES src/expand/expand.cppm)
target_link_libraries(nimblecas_expand
  PUBLIC nimblecas_simplify nimblecas_combinatorics nimblecas_symbolic nimblecas_core std_module)

# ---- nimblecas.factor : exact factorization over Q (Yun + Kronecker) (§7.17/§7.21) ----
add_library(nimblecas_factor STATIC)
target_sources(nimblecas_factor PUBLIC
  FILE_SET CXX_MODULES FILES src/factor/factor.cppm)
target_link_libraries(nimblecas_factor
  PUBLIC nimblecas_polynomial nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.statinfer : inferential statistics & regression (OLS/ridge/WLS, R²) (§7.7.5) ----
add_library(nimblecas_statinfer STATIC)
target_sources(nimblecas_statinfer PUBLIC
  FILE_SET CXX_MODULES FILES src/statinfer/statinfer.cppm)
target_link_libraries(nimblecas_statinfer
  PUBLIC nimblecas_stats nimblecas_matrix nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.symint : Expr-level symbolic integration (§7.19) ----
add_library(nimblecas_symint STATIC)
target_sources(nimblecas_symint PUBLIC
  FILE_SET CXX_MODULES FILES src/symint/symint.cppm)
target_link_libraries(nimblecas_symint
  PUBLIC nimblecas_integrate nimblecas_rothstein nimblecas_polyexpr nimblecas_polynomial
         nimblecas_diff nimblecas_simplify nimblecas_symbolic nimblecas_ratpoly
         nimblecas_core std_module)

# ---- nimblecas.kronprod : Kronecker/direct/Hadamard products, vec/unvec (§7.2) ----
add_library(nimblecas_kronprod STATIC)
target_sources(nimblecas_kronprod PUBLIC
  FILE_SET CXX_MODULES FILES src/kronprod/kronprod.cppm)
target_link_libraries(nimblecas_kronprod
  PUBLIC nimblecas_matrix nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.qrschur : QR decomposition + real Schur form (§7.2) ----
add_library(nimblecas_qrschur STATIC)
target_sources(nimblecas_qrschur PUBLIC
  FILE_SET CXX_MODULES FILES src/qrschur/qrschur.cppm)
target_link_libraries(nimblecas_qrschur
  PUBLIC nimblecas_numeigen nimblecas_matrix nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.cheigen : numeric eigenvalues of complex matrices (§7.2) ----
add_library(nimblecas_cheigen STATIC)
target_sources(nimblecas_cheigen PUBLIC
  FILE_SET CXX_MODULES FILES src/cheigen/cheigen.cppm)
target_link_libraries(nimblecas_cheigen
  PUBLIC nimblecas_cmatrix nimblecas_complex nimblecas_numeigen nimblecas_ratpoly
         nimblecas_core std_module)

# ---- nimblecas.laurent : truncated Laurent series over Q (§7) ----
add_library(nimblecas_laurent STATIC)
target_sources(nimblecas_laurent PUBLIC
  FILE_SET CXX_MODULES FILES src/laurent/laurent.cppm)
target_link_libraries(nimblecas_laurent
  PUBLIC nimblecas_powerseries nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.contfrac : continued fractions + Viskovatov (§7) ----
add_library(nimblecas_contfrac STATIC)
target_sources(nimblecas_contfrac PUBLIC
  FILE_SET CXX_MODULES FILES src/contfrac/contfrac.cppm)
target_link_libraries(nimblecas_contfrac
  PUBLIC nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.hyptest : hypothesis tests + MLE (score/Fisher) (§7.7) ----
add_library(nimblecas_hyptest STATIC)
target_sources(nimblecas_hyptest PUBLIC
  FILE_SET CXX_MODULES FILES src/hyptest/hyptest.cppm)
target_link_libraries(nimblecas_hyptest
  PUBLIC nimblecas_statinfer nimblecas_probdist nimblecas_diff nimblecas_simplify
         nimblecas_stats nimblecas_matrix nimblecas_symbolic nimblecas_ratpoly
         nimblecas_core std_module)

# ---- nimblecas.pdeham : PDE homotopy analysis method, h-bar deformation (§7) ----
add_library(nimblecas_pdeham STATIC)
target_sources(nimblecas_pdeham PUBLIC
  FILE_SET CXX_MODULES FILES src/pdeham/pdeham.cppm)
target_link_libraries(nimblecas_pdeham
  PUBLIC nimblecas_expand nimblecas_diff nimblecas_simplify nimblecas_symbolic
         nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.frobenius : rational (Frobenius) canonical form over Q (§7.2) ----
add_library(nimblecas_frobenius STATIC)
target_sources(nimblecas_frobenius PUBLIC
  FILE_SET CXX_MODULES FILES src/frobenius/frobenius.cppm)
target_link_libraries(nimblecas_frobenius
  PUBLIC nimblecas_matrix nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.algnum : simple algebraic extension field Q(alpha)=Q[x]/(m) (§7.1/§7.2) ----
add_library(nimblecas_algnum STATIC)
target_sources(nimblecas_algnum PUBLIC
  FILE_SET CXX_MODULES FILES src/algnum/algnum.cppm)
target_link_libraries(nimblecas_algnum
  PUBLIC nimblecas_factor nimblecas_matrix nimblecas_ratpoly nimblecas_polynomial
         nimblecas_core std_module)

# ---- nimblecas.jordan : Jordan canonical form A = P J P^-1, over Q and over Q(alpha) (§7.2) ----
add_library(nimblecas_jordan STATIC)
target_sources(nimblecas_jordan PUBLIC
  FILE_SET CXX_MODULES FILES src/jordan/jordan.cppm)
target_link_libraries(nimblecas_jordan
  PUBLIC nimblecas_algnum nimblecas_eigen nimblecas_factor nimblecas_roots
         nimblecas_matrix nimblecas_ratpoly nimblecas_polynomial nimblecas_core std_module)

# ---- nimblecas.probmethod : the Probabilistic Method — existence proofs, LLL, Ramsey (§7.18) ----
add_library(nimblecas_probmethod STATIC)
target_sources(nimblecas_probmethod PUBLIC
  FILE_SET CXX_MODULES FILES src/probmethod/probmethod.cppm)
target_link_libraries(nimblecas_probmethod
  PUBLIC nimblecas_bigcombinatorics nimblecas_bigint nimblecas_ratpoly nimblecas_core std_module)

# ---- nimblecas.bigdecimal : exact base-10 scaled decimal (money type + boundary quantizer) ----
add_library(nimblecas_bigdecimal STATIC)
target_sources(nimblecas_bigdecimal PUBLIC
  FILE_SET CXX_MODULES FILES src/bigdecimal/bigdecimal.cppm)
target_link_libraries(nimblecas_bigdecimal
  PUBLIC nimblecas_bigrational nimblecas_bigint nimblecas_core std_module)

# ---- nimblecas.finance : Excel/Mathematica/MATLAB/Maple financial suite (TVM, cashflow, bonds) ----
add_library(nimblecas_finance STATIC)
target_sources(nimblecas_finance PUBLIC
  FILE_SET CXX_MODULES FILES src/finance/finance.cppm)
target_link_libraries(nimblecas_finance
  PUBLIC nimblecas_bigdecimal nimblecas_bigrational nimblecas_bigint nimblecas_core std_module)

# ---- nimblecas.currency : exact FX — money, rate graph, cross rates, arbitrage, forwards ----
add_library(nimblecas_currency STATIC)
target_sources(nimblecas_currency PUBLIC
  FILE_SET CXX_MODULES FILES src/currency/currency.cppm)
target_link_libraries(nimblecas_currency
  PUBLIC nimblecas_bigdecimal nimblecas_bigrational nimblecas_bigint nimblecas_core std_module)

# ---- nimblecas.pricing : options/instrument pricing (trinomial trees, MC, LSM, plotting) ----
add_library(nimblecas_pricing STATIC)
target_sources(nimblecas_pricing PUBLIC
  FILE_SET CXX_MODULES FILES src/pricing/pricing.cppm)
target_link_libraries(nimblecas_pricing
  PUBLIC nimblecas_svgplot nimblecas_rng nimblecas_core std_module)

# ---- tests ----
enable_testing()

add_executable(bigdecimal_tests tests/bigdecimal_tests.cpp)
target_link_libraries(bigdecimal_tests
  PRIVATE nimblecas_bigdecimal nimblecas_bigrational nimblecas_bigint nimblecas_core
          nimblecas_testing std_module)
add_test(NAME bigdecimal_tests COMMAND bigdecimal_tests)

add_executable(finance_tests tests/finance_tests.cpp)
target_link_libraries(finance_tests
  PRIVATE nimblecas_finance nimblecas_bigdecimal nimblecas_bigrational nimblecas_bigint
          nimblecas_core nimblecas_testing std_module)
add_test(NAME finance_tests COMMAND finance_tests)

add_executable(currency_tests tests/currency_tests.cpp)
target_link_libraries(currency_tests
  PRIVATE nimblecas_currency nimblecas_bigdecimal nimblecas_bigrational nimblecas_bigint
          nimblecas_core nimblecas_testing std_module)
add_test(NAME currency_tests COMMAND currency_tests)

add_executable(pricing_tests tests/pricing_tests.cpp)
target_link_libraries(pricing_tests
  PRIVATE nimblecas_pricing nimblecas_svgplot nimblecas_rng nimblecas_core
          nimblecas_testing std_module)
add_test(NAME pricing_tests COMMAND pricing_tests)

add_executable(numeigen_tests tests/numeigen_tests.cpp)
target_link_libraries(numeigen_tests
  PRIVATE nimblecas_numeigen nimblecas_ratpoly nimblecas_core nimblecas_testing std_module)
add_test(NAME numeigen_tests COMMAND numeigen_tests)

add_executable(solve_tests tests/solve_tests.cpp)
target_link_libraries(solve_tests
  PRIVATE nimblecas_solve nimblecas_factor nimblecas_numeigen nimblecas_roots nimblecas_symbolic
          nimblecas_polynomial nimblecas_ratpoly nimblecas_core nimblecas_testing std_module)
add_test(NAME solve_tests COMMAND solve_tests)

add_executable(ipm_tests tests/ipm_tests.cpp)
target_link_libraries(ipm_tests
  PRIVATE nimblecas_ipm nimblecas_core nimblecas_testing std_module)
add_test(NAME ipm_tests COMMAND ipm_tests)

add_executable(singpert_tests tests/singpert_tests.cpp)
target_link_libraries(singpert_tests
  PRIVATE nimblecas_singpert nimblecas_symbolic nimblecas_ratpoly nimblecas_core
          nimblecas_testing std_module)
add_test(NAME singpert_tests COMMAND singpert_tests)

add_executable(probdist_tests tests/probdist_tests.cpp)
target_link_libraries(probdist_tests
  PRIVATE nimblecas_probdist nimblecas_simplify nimblecas_diff nimblecas_symbolic
          nimblecas_core nimblecas_testing std_module)
add_test(NAME probdist_tests COMMAND probdist_tests)

add_executable(expand_tests tests/expand_tests.cpp)
target_link_libraries(expand_tests
  PRIVATE nimblecas_expand nimblecas_simplify nimblecas_combinatorics nimblecas_symbolic
          nimblecas_ratpoly nimblecas_core nimblecas_testing std_module)
add_test(NAME expand_tests COMMAND expand_tests)

add_executable(factor_tests tests/factor_tests.cpp)
target_link_libraries(factor_tests
  PRIVATE nimblecas_factor nimblecas_polynomial nimblecas_ratpoly nimblecas_core
          nimblecas_testing std_module)
add_test(NAME factor_tests COMMAND factor_tests)

add_executable(statinfer_tests tests/statinfer_tests.cpp)
target_link_libraries(statinfer_tests
  PRIVATE nimblecas_statinfer nimblecas_stats nimblecas_matrix nimblecas_ratpoly
          nimblecas_polynomial nimblecas_core nimblecas_testing std_module)
add_test(NAME statinfer_tests COMMAND statinfer_tests)

add_executable(symint_tests tests/symint_tests.cpp)
target_link_libraries(symint_tests
  PRIVATE nimblecas_symint nimblecas_integrate nimblecas_rothstein nimblecas_polyexpr
          nimblecas_polynomial nimblecas_diff nimblecas_simplify nimblecas_symbolic
          nimblecas_ratpoly nimblecas_core nimblecas_testing std_module)
add_test(NAME symint_tests COMMAND symint_tests)

add_executable(kronprod_tests tests/kronprod_tests.cpp)
target_link_libraries(kronprod_tests
  PRIVATE nimblecas_kronprod nimblecas_matrix nimblecas_ratpoly
          nimblecas_core nimblecas_testing std_module)
add_test(NAME kronprod_tests COMMAND kronprod_tests)

add_executable(qrschur_tests tests/qrschur_tests.cpp)
target_link_libraries(qrschur_tests
  PRIVATE nimblecas_qrschur nimblecas_numeigen nimblecas_matrix nimblecas_ratpoly
          nimblecas_core nimblecas_testing std_module)
add_test(NAME qrschur_tests COMMAND qrschur_tests)

add_executable(cheigen_tests tests/cheigen_tests.cpp)
target_link_libraries(cheigen_tests
  PRIVATE nimblecas_cheigen nimblecas_cmatrix nimblecas_complex nimblecas_numeigen
          nimblecas_ratpoly nimblecas_core nimblecas_testing std_module)
add_test(NAME cheigen_tests COMMAND cheigen_tests)

add_executable(laurent_tests tests/laurent_tests.cpp)
target_link_libraries(laurent_tests
  PRIVATE nimblecas_laurent nimblecas_powerseries nimblecas_ratpoly
          nimblecas_core nimblecas_testing std_module)
add_test(NAME laurent_tests COMMAND laurent_tests)

add_executable(contfrac_tests tests/contfrac_tests.cpp)
target_link_libraries(contfrac_tests
  PRIVATE nimblecas_contfrac nimblecas_ratpoly nimblecas_core nimblecas_testing std_module)
add_test(NAME contfrac_tests COMMAND contfrac_tests)

add_executable(hyptest_tests tests/hyptest_tests.cpp)
target_link_libraries(hyptest_tests
  PRIVATE nimblecas_hyptest nimblecas_statinfer nimblecas_probdist nimblecas_stats
          nimblecas_simplify nimblecas_diff nimblecas_symbolic nimblecas_matrix
          nimblecas_ratpoly nimblecas_core nimblecas_testing std_module)
add_test(NAME hyptest_tests COMMAND hyptest_tests)

add_executable(pdeham_tests tests/pdeham_tests.cpp)
target_link_libraries(pdeham_tests
  PRIVATE nimblecas_pdeham nimblecas_pde nimblecas_expand nimblecas_diff nimblecas_simplify
          nimblecas_symbolic nimblecas_ratpoly nimblecas_core nimblecas_testing std_module)
add_test(NAME pdeham_tests COMMAND pdeham_tests)

add_executable(frobenius_tests tests/frobenius_tests.cpp)
target_link_libraries(frobenius_tests
  PRIVATE nimblecas_frobenius nimblecas_eigen nimblecas_matrix nimblecas_roots
          nimblecas_ratpoly nimblecas_core nimblecas_testing std_module)
add_test(NAME frobenius_tests COMMAND frobenius_tests)

add_executable(algnum_tests tests/algnum_tests.cpp)
target_link_libraries(algnum_tests
  PRIVATE nimblecas_algnum nimblecas_factor nimblecas_matrix nimblecas_ratpoly
          nimblecas_polynomial nimblecas_core nimblecas_testing std_module)
add_test(NAME algnum_tests COMMAND algnum_tests)

add_executable(jordan_tests tests/jordan_tests.cpp)
target_link_libraries(jordan_tests
  PRIVATE nimblecas_jordan nimblecas_algnum nimblecas_eigen nimblecas_factor
          nimblecas_roots nimblecas_matrix nimblecas_ratpoly nimblecas_polynomial
          nimblecas_core nimblecas_testing std_module)
add_test(NAME jordan_tests COMMAND jordan_tests)

add_executable(probmethod_tests tests/probmethod_tests.cpp)
target_link_libraries(probmethod_tests
  PRIVATE nimblecas_probmethod nimblecas_bigcombinatorics nimblecas_bigint
          nimblecas_ratpoly nimblecas_core nimblecas_testing std_module)
add_test(NAME probmethod_tests COMMAND probmethod_tests)

add_executable(core_tests tests/core_tests.cpp)
target_link_libraries(core_tests PRIVATE nimblecas_core nimblecas_testing std_module)
add_test(NAME core_tests COMMAND core_tests)

add_executable(matrix_tests tests/matrix_tests.cpp)
target_link_libraries(matrix_tests
  PRIVATE nimblecas_matrix nimblecas_ratpoly nimblecas_polynomial nimblecas_core
          nimblecas_testing std_module)
add_test(NAME matrix_tests COMMAND matrix_tests)

add_executable(krylov_tests tests/krylov_tests.cpp)
target_link_libraries(krylov_tests
  PRIVATE nimblecas_krylov nimblecas_matrix nimblecas_ratpoly nimblecas_polynomial
          nimblecas_core nimblecas_testing std_module)
add_test(NAME krylov_tests COMMAND krylov_tests)

add_executable(stats_tests tests/stats_tests.cpp)
target_link_libraries(stats_tests
  PRIVATE nimblecas_stats nimblecas_matrix nimblecas_ratpoly nimblecas_polynomial
          nimblecas_core nimblecas_testing std_module)
add_test(NAME stats_tests COMMAND stats_tests)

add_executable(stochastic_tests tests/stochastic_tests.cpp)
target_link_libraries(stochastic_tests
  PRIVATE nimblecas_stochastic nimblecas_stats nimblecas_matrix nimblecas_roots
          nimblecas_ratpoly nimblecas_polynomial nimblecas_core nimblecas_testing std_module)
add_test(NAME stochastic_tests COMMAND stochastic_tests)

add_executable(hmm_tests tests/hmm_tests.cpp)
target_link_libraries(hmm_tests
  PRIVATE nimblecas_hmm nimblecas_matrix nimblecas_ratpoly nimblecas_polynomial
          nimblecas_core nimblecas_testing std_module)
add_test(NAME hmm_tests COMMAND hmm_tests)

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

add_executable(spectral_tests tests/spectral_tests.cpp)
target_link_libraries(spectral_tests
  PRIVATE nimblecas_spectral nimblecas_orthopoly nimblecas_matrix nimblecas_ratpoly
          nimblecas_polynomial nimblecas_core nimblecas_testing std_module)
add_test(NAME spectral_tests COMMAND spectral_tests)

add_executable(fft_tests tests/fft_tests.cpp)
target_link_libraries(fft_tests
  PRIVATE nimblecas_fft nimblecas_core nimblecas_testing std_module)
add_test(NAME fft_tests COMMAND fft_tests)

add_executable(semigroup_tests tests/semigroup_tests.cpp)
target_link_libraries(semigroup_tests
  PRIVATE nimblecas_semigroup nimblecas_matexp nimblecas_dynamics nimblecas_eigen
          nimblecas_matrix nimblecas_roots nimblecas_ratpoly nimblecas_polynomial
          nimblecas_core nimblecas_testing std_module)
add_test(NAME semigroup_tests COMMAND semigroup_tests)

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

add_executable(quantum_tests tests/quantum_tests.cpp)
target_link_libraries(quantum_tests
  PRIVATE nimblecas_quantum nimblecas_complex nimblecas_ratpoly nimblecas_polynomial
          nimblecas_core nimblecas_testing std_module)
add_test(NAME quantum_tests COMMAND quantum_tests)

add_executable(recurrence_tests tests/recurrence_tests.cpp)
target_link_libraries(recurrence_tests
  PRIVATE nimblecas_recurrence nimblecas_symbolic nimblecas_simplify nimblecas_matrix
          nimblecas_roots nimblecas_ratpoly nimblecas_polynomial
          nimblecas_core nimblecas_testing std_module)
add_test(NAME recurrence_tests COMMAND recurrence_tests)

add_executable(numeric_tests tests/numeric_tests.cpp)
target_link_libraries(numeric_tests
  PRIVATE nimblecas_numeric nimblecas_core nimblecas_testing std_module)
add_test(NAME numeric_tests COMMAND numeric_tests)

add_executable(compsense_tests tests/compsense_tests.cpp)
target_link_libraries(compsense_tests
  PRIVATE nimblecas_compsense nimblecas_lp nimblecas_matrix nimblecas_ratpoly
          nimblecas_polynomial nimblecas_core nimblecas_testing std_module)
add_test(NAME compsense_tests COMMAND compsense_tests)

add_executable(optimize_tests tests/optimize_tests.cpp)
target_link_libraries(optimize_tests
  PRIVATE nimblecas_optimize nimblecas_core nimblecas_testing std_module)
add_test(NAME optimize_tests COMMAND optimize_tests)

add_executable(nlsolve_tests tests/nlsolve_tests.cpp)
target_link_libraries(nlsolve_tests
  PRIVATE nimblecas_nlsolve nimblecas_core nimblecas_testing std_module)
add_test(NAME nlsolve_tests COMMAND nlsolve_tests)

add_executable(extrapolation_tests tests/extrapolation_tests.cpp)
target_link_libraries(extrapolation_tests
  PRIVATE nimblecas_extrapolation nimblecas_ratpoly nimblecas_polynomial nimblecas_core
          nimblecas_testing std_module)
add_test(NAME extrapolation_tests COMMAND extrapolation_tests)

add_executable(wavelets_tests tests/wavelets_tests.cpp)
target_link_libraries(wavelets_tests
  PRIVATE nimblecas_wavelets nimblecas_ratpoly nimblecas_polynomial nimblecas_core
          nimblecas_testing std_module)
add_test(NAME wavelets_tests COMMAND wavelets_tests)

add_executable(qmc_tests tests/qmc_tests.cpp)
target_link_libraries(qmc_tests
  PRIVATE nimblecas_qmc nimblecas_montecarlo nimblecas_rng nimblecas_ratpoly
          nimblecas_polynomial nimblecas_core nimblecas_testing std_module)
add_test(NAME qmc_tests COMMAND qmc_tests)

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

add_executable(lie_tests tests/lie_tests.cpp)
target_link_libraries(lie_tests
  PRIVATE nimblecas_lie nimblecas_matexp nimblecas_matrix nimblecas_ratpoly
          nimblecas_polynomial nimblecas_core nimblecas_testing std_module)
add_test(NAME lie_tests COMMAND lie_tests)

add_executable(matstruct_tests tests/matstruct_tests.cpp)
target_link_libraries(matstruct_tests
  PRIVATE nimblecas_matstruct nimblecas_matdecomp nimblecas_eigen nimblecas_roots
          nimblecas_matrix nimblecas_ratpoly nimblecas_polynomial nimblecas_core
          nimblecas_testing std_module)
add_test(NAME matstruct_tests COMMAND matstruct_tests)

add_executable(interpolation_tests tests/interpolation_tests.cpp)
target_link_libraries(interpolation_tests
  PRIVATE nimblecas_interpolation nimblecas_ratpoly nimblecas_polynomial nimblecas_core
          nimblecas_testing std_module)
add_test(NAME interpolation_tests COMMAND interpolation_tests)

add_executable(splines_tests tests/splines_tests.cpp)
target_link_libraries(splines_tests
  PRIVATE nimblecas_splines nimblecas_bandsolve nimblecas_matrix nimblecas_ratpoly
          nimblecas_polynomial nimblecas_core nimblecas_testing std_module)
add_test(NAME splines_tests COMMAND splines_tests)

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

add_executable(webexport_tests tests/webexport_tests.cpp)
target_link_libraries(webexport_tests
  PRIVATE nimblecas_webexport nimblecas_latex nimblecas_symbolic nimblecas_core
          nimblecas_testing std_module)
add_test(NAME webexport_tests COMMAND webexport_tests)

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

add_executable(limits_tests tests/limits_tests.cpp)
target_link_libraries(limits_tests
  PRIVATE nimblecas_limits nimblecas_diff nimblecas_simplify nimblecas_symbolic
          nimblecas_cache nimblecas_parallel nimblecas_core nimblecas_testing std_module)
add_test(NAME limits_tests COMMAND limits_tests)

add_executable(tensor_tests tests/tensor_tests.cpp)
target_link_libraries(tensor_tests
  PRIVATE nimblecas_tensor nimblecas_diff nimblecas_simplify nimblecas_symbolic
          nimblecas_cache nimblecas_parallel nimblecas_core nimblecas_testing std_module)
add_test(NAME tensor_tests COMMAND tensor_tests)

add_executable(forms_tests tests/forms_tests.cpp)
target_link_libraries(forms_tests
  PRIVATE nimblecas_forms nimblecas_diff nimblecas_simplify nimblecas_symbolic
          nimblecas_cache nimblecas_parallel nimblecas_core nimblecas_testing std_module)
add_test(NAME forms_tests COMMAND forms_tests)

add_executable(smc_tests tests/smc_tests.cpp)
target_link_libraries(smc_tests
  PRIVATE nimblecas_smc nimblecas_rng nimblecas_core nimblecas_testing std_module)
add_test(NAME smc_tests COMMAND smc_tests)

add_executable(reader_tests tests/reader_tests.cpp)
target_link_libraries(reader_tests
  PRIVATE nimblecas_reader nimblecas_symbolic nimblecas_core nimblecas_testing std_module)
add_test(NAME reader_tests COMMAND reader_tests)

add_executable(execdoc_tests tests/execdoc_tests.cpp)
target_link_libraries(execdoc_tests
  PRIVATE nimblecas_execdoc nimblecas_reader nimblecas_latex nimblecas_diff
          nimblecas_simplify nimblecas_symbolic nimblecas_cache nimblecas_parallel
          nimblecas_core nimblecas_testing std_module)
add_test(NAME execdoc_tests COMMAND execdoc_tests)

add_executable(execdoc_multimethod_tests tests/execdoc_multimethod_tests.cpp)
target_link_libraries(execdoc_multimethod_tests
  PRIVATE nimblecas_execdoc nimblecas_reader nimblecas_latex nimblecas_diff
          nimblecas_simplify nimblecas_symbolic nimblecas_cache nimblecas_parallel
          nimblecas_core nimblecas_testing std_module)
add_test(NAME execdoc_multimethod_tests COMMAND execdoc_multimethod_tests)

add_executable(mechanics_tests tests/mechanics_tests.cpp)
target_link_libraries(mechanics_tests
  PRIVATE nimblecas_mechanics nimblecas_diff nimblecas_simplify nimblecas_symbolic
          nimblecas_cache nimblecas_parallel nimblecas_core nimblecas_testing std_module)
add_test(NAME mechanics_tests COMMAND mechanics_tests)

add_executable(calcvar_tests tests/calcvar_tests.cpp)
target_link_libraries(calcvar_tests
  PRIVATE nimblecas_calcvar nimblecas_diff nimblecas_simplify nimblecas_symbolic
          nimblecas_cache nimblecas_parallel nimblecas_core nimblecas_testing std_module)
add_test(NAME calcvar_tests COMMAND calcvar_tests)

add_executable(analysis_tests tests/analysis_tests.cpp)
target_link_libraries(analysis_tests
  PRIVATE nimblecas_analysis nimblecas_dynamics nimblecas_eigen nimblecas_matrix
          nimblecas_ratpoly nimblecas_polynomial nimblecas_core nimblecas_testing std_module)
add_test(NAME analysis_tests COMMAND analysis_tests)

add_executable(control_tests tests/control_tests.cpp)
target_link_libraries(control_tests
  PRIVATE nimblecas_control nimblecas_dynamics nimblecas_eigen nimblecas_matrix
          nimblecas_complex nimblecas_roots nimblecas_ratpoly nimblecas_polynomial
          nimblecas_core nimblecas_testing std_module)
add_test(NAME control_tests COMMAND control_tests)

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

add_executable(pdenum_tests tests/pdenum_tests.cpp)
target_link_libraries(pdenum_tests
  PRIVATE nimblecas_pdenum nimblecas_bandsolve nimblecas_matrix nimblecas_ratpoly
          nimblecas_polynomial nimblecas_parallel nimblecas_core nimblecas_testing std_module)
add_test(NAME pdenum_tests COMMAND pdenum_tests)

add_executable(inteq_tests tests/inteq_tests.cpp)
target_link_libraries(inteq_tests
  PRIVATE nimblecas_inteq nimblecas_integrate nimblecas_ratint nimblecas_rothstein
          nimblecas_pfd nimblecas_resultant nimblecas_laplace nimblecas_simplify
          nimblecas_cache nimblecas_matrix nimblecas_symbolic nimblecas_parallel
          nimblecas_ratpoly nimblecas_polynomial nimblecas_simd nimblecas_core
          nimblecas_testing std_module)
add_test(NAME inteq_tests COMMAND inteq_tests)

# ---- cross-module feature / integration tests ----
add_executable(pde_crossmethod_tests tests/pde_crossmethod_tests.cpp)
target_link_libraries(pde_crossmethod_tests
  PRIVATE nimblecas_pde nimblecas_pdenum nimblecas_bandsolve nimblecas_matrix
          nimblecas_ratpoly nimblecas_polynomial nimblecas_parallel nimblecas_core
          nimblecas_testing std_module)
add_test(NAME pde_crossmethod_tests COMMAND pde_crossmethod_tests)

add_executable(inteq_crossmethod_tests tests/inteq_crossmethod_tests.cpp)
target_link_libraries(inteq_crossmethod_tests
  PRIVATE nimblecas_inteq nimblecas_integrate nimblecas_ratint nimblecas_rothstein
          nimblecas_pfd nimblecas_resultant nimblecas_laplace nimblecas_simplify
          nimblecas_cache nimblecas_matrix nimblecas_symbolic nimblecas_parallel
          nimblecas_ratpoly nimblecas_polynomial nimblecas_simd nimblecas_core
          nimblecas_testing std_module)
add_test(NAME inteq_crossmethod_tests COMMAND inteq_crossmethod_tests)

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

add_executable(feature_reasoning_tests tests/feature_reasoning_tests.cpp)
target_link_libraries(feature_reasoning_tests
  PRIVATE nimblecas_search nimblecas_sat nimblecas_csp nimblecas_logic
          nimblecas_bitcsp nimblecas_bitset nimblecas_rng nimblecas_parallel
          nimblecas_core nimblecas_testing std_module)
add_test(NAME feature_reasoning_tests COMMAND feature_reasoning_tests)

add_executable(feature_diffeq_tests tests/feature_diffeq_tests.cpp)
target_link_libraries(feature_diffeq_tests
  PRIVATE nimblecas_ode nimblecas_dde nimblecas_dae nimblecas_perturbation nimblecas_sde
          nimblecas_powerseries nimblecas_matrix nimblecas_rng nimblecas_ratpoly
          nimblecas_core nimblecas_testing std_module)
add_test(NAME feature_diffeq_tests COMMAND feature_diffeq_tests)

add_executable(feature_arithmetic_tests tests/feature_arithmetic_tests.cpp)
target_link_libraries(feature_arithmetic_tests
  PRIVATE nimblecas_bigeigen nimblecas_bigmatrix nimblecas_bigpowerseries
          nimblecas_bigcombinatorics nimblecas_bigrational nimblecas_bigfloat
          nimblecas_bigint nimblecas_int128 nimblecas_numbertheory nimblecas_constants
          nimblecas_doubledouble nimblecas_matrix nimblecas_ratpoly nimblecas_polynomial
          nimblecas_simd nimblecas_rng nimblecas_core nimblecas_testing std_module)
add_test(NAME feature_arithmetic_tests COMMAND feature_arithmetic_tests)

add_executable(feature_applied_numerics_tests tests/feature_applied_numerics_tests.cpp)
target_link_libraries(feature_applied_numerics_tests
  PRIVATE nimblecas_interpolation nimblecas_spectral nimblecas_pdenum nimblecas_qmc
          nimblecas_extrapolation nimblecas_optimize nimblecas_nlsolve nimblecas_control
          nimblecas_stochastic nimblecas_dynamics nimblecas_eigen nimblecas_orthopoly
          nimblecas_bandsolve nimblecas_roots nimblecas_complex nimblecas_stats
          nimblecas_matrix nimblecas_rng nimblecas_ratpoly nimblecas_polynomial
          nimblecas_parallel nimblecas_simd nimblecas_core nimblecas_testing std_module)
add_test(NAME feature_applied_numerics_tests COMMAND feature_applied_numerics_tests)

add_executable(feature_applied_linalg_tests tests/feature_applied_linalg_tests.cpp)
target_link_libraries(feature_applied_linalg_tests
  PRIVATE nimblecas_matstruct nimblecas_lie nimblecas_krylov nimblecas_semigroup
          nimblecas_control nimblecas_analysis nimblecas_matexp nimblecas_matdecomp
          nimblecas_dynamics nimblecas_eigen nimblecas_roots nimblecas_matrix
          nimblecas_ratpoly nimblecas_polynomial nimblecas_complex nimblecas_core
          nimblecas_testing std_module)
add_test(NAME feature_applied_linalg_tests COMMAND feature_applied_linalg_tests)

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
  # Allow user override via -DNIMBLECAS_NVCC=/path/to/nvcc; otherwise auto-detect.
  # NOTE: do NOT pre-seed NIMBLECAS_NVCC with an empty CACHE entry -- find_program
  # skips the search when its result variable already holds any value (including ""),
  # which silently leaves nvcc empty and collapses the custom command to a bare `cd`.
  if(NOT NIMBLECAS_NVCC)
    find_program(NIMBLECAS_NVCC_EXE nvcc
      HINTS /usr/local/cuda/bin /usr/local/cuda-13.2/bin ENV CUDA_HOME ENV PATH REQUIRED)
    set(NIMBLECAS_NVCC "${NIMBLECAS_NVCC_EXE}")
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
    PRIVATE nimblecas_solve nimblecas_factor nimblecas_numeigen nimblecas_symint
            nimblecas_expand nimblecas_hyptest nimblecas_stats nimblecas_kronprod
            nimblecas_contfrac nimblecas_probdist nimblecas_qrschur nimblecas_cheigen
            nimblecas_cmatrix nimblecas_laurent nimblecas_frobenius nimblecas_dynamics
            nimblecas_algnum nimblecas_jordan nimblecas_matrix nimblecas_complex
            nimblecas_ratpoly nimblecas_diff nimblecas_simplify nimblecas_polyexpr
            nimblecas_polynomial nimblecas_symbolic nimblecas_simd nimblecas_core std_module)

  # Expose the GPU submodule to Python only when the CUDA kernels are built.
  if(NIMBLECAS_CUDA AND NOT WIN32)
    add_dependencies(nimblecas_ext nimblecas_gpu_kernels)
    target_link_libraries(nimblecas_ext PRIVATE nimblecas_gpu)
    target_compile_definitions(nimblecas_ext PRIVATE NIMBLECAS_EXT_WITH_GPU)
  endif()

  add_test(NAME python_bindings
           COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_bindings.py)
  set_tests_properties(python_bindings PROPERTIES
    ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:nimblecas_ext>")
endif()
