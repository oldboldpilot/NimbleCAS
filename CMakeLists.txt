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
