#!/usr/bin/env bash
# NimbleCAS canonical build configuration.
# @author Olumuyiwa Oluwasanmi
#
# This file holds the AUTHORITATIVE CANONICAL_FLAGS array (Code Policy Rule 50).
# cmake/CanonicalFlags.cmake MUST mirror this list exactly; any divergence causes
# BMI (Binary Module Interface) validation to reject a PCM with a
# module-file-config-mismatch error.
#
# All build scripts source this file. REPO_ROOT is resolved via git so the tree
# is relocatable (Code Policy Rule 51).

set -euo pipefail

REPO_ROOT="$(git -C "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" rev-parse --show-toplevel)"
export REPO_ROOT

# Mandated toolchain: clang++-23 only (Code Policy Rule 50). Overridable for probing.
#
# The C compiler is named too, and not optionally. CMake defaults CC to the system cc
# (gcc), which does not understand -stdlib=libc++, so any dependency that enables C
# fails at the compiler test with an error naming a flag rather than the compiler.
#
# clang-scan-deps must match the compiler exactly: CMake cannot discover the C++20
# module import graph without a scanner of the same release, and a version-mismatched
# one is worse than none.
export NIMBLECAS_CLANGXX="${NIMBLECAS_CLANGXX:-clang++-23}"
export NIMBLECAS_CLANGC="${NIMBLECAS_CLANGC:-clang-23}"
export NIMBLECAS_SCAN_DEPS="${NIMBLECAS_SCAN_DEPS:-clang-scan-deps-23}"
export NIMBLECAS_CLANG_TIDY="${NIMBLECAS_CLANG_TIDY:-clang-tidy-23}"

# Canonical, host-portable flags. NOTE: -march=x86-64-v3 (NOT -march=native) and NO
# -ffast-math, so results are bit-identical across hosts (Code Policy Rules 50/55).
# AVX-512 is engaged per-function via [[gnu::target(...)]], never as a global flag.
#
# DEVIATION (documented): the policy's `-nostdinc++ -isystem external/libcxx-v1/include`
# assumes a vendored libc++ tree. Phase 1 builds on the fixed server against the system
# libc++-23 via -stdlib=libc++. Vendoring external/libcxx-v1 is a follow-up required
# before multi-host / cloud builds (Rule 51 rationale: differing cloud LLVM versions).
CANONICAL_FLAGS=(
  -std=c++23 -stdlib=libc++ -fPIC -O3
  -march=x86-64-v3 -mtune=generic
  -mavx -mavx2 -mfma
  -pthread -fstack-protector-strong -DNDEBUG
  -D_LIBCPP_ENABLE_EXPERIMENTAL -fexperimental-library
)
export CANONICAL_FLAGS
