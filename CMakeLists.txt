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

# ---- nimblecas.diff : symbolic differentiation ----
add_library(nimblecas_diff STATIC)
target_sources(nimblecas_diff PUBLIC
  FILE_SET CXX_MODULES FILES src/diff/diff.cppm)
target_link_libraries(nimblecas_diff
  PUBLIC nimblecas_simplify nimblecas_symbolic nimblecas_parallel nimblecas_cache
         nimblecas_core std_module)

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

add_executable(parallel_tests tests/parallel_tests.cpp)
target_link_libraries(parallel_tests
  PRIVATE nimblecas_parallel nimblecas_testing std_module)
add_test(NAME parallel_tests COMMAND parallel_tests)

# ---- Python bindings (nanobind, ROADMAP 10) — opt-in via -DNIMBLECAS_PYTHON=ON ----
option(NIMBLECAS_PYTHON "Build the nanobind Python bindings" OFF)
if(NIMBLECAS_PYTHON)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(nimblecas_ext src/bindings/bindings.cpp)
  target_link_libraries(nimblecas_ext
    PRIVATE nimblecas_diff nimblecas_simplify nimblecas_symbolic nimblecas_core std_module)

  add_test(NAME python_bindings
           COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_bindings.py)
  set_tests_properties(python_bindings PROPERTIES
    ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:nimblecas_ext>")
endif()
