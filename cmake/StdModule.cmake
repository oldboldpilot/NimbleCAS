# NimbleCAS `import std` support (Code Policy Rules 12 / 41).
# @author Olumuyiwa Oluwasanmi
#
# CMake's built-in `import std` support is gated behind a version-locked experimental
# UUID and is discouraged by the policy's explicit-module discipline (Rules 50-52).
# Instead we compile the standard library's std module source as an ordinary
# CXX_MODULES library target, so CMake's module scanner resolves `import std;` in our
# translation units normally and with explicit, reproducible dependency edges.
#
# Cross-platform: libc++'s std.cppm on Linux/macOS, the MSVC toolset's std.ixx on
# Windows (clang targeting windows-msvc uses the Microsoft STL). Verified under both.

if(WIN32)
  # Locate the MSVC toolset's std.ixx. Prefer the VCToolsInstallDir env var (set in a
  # VS developer shell); otherwise glob the newest Visual Studio MSVC toolset.
  if(DEFINED ENV{VCToolsInstallDir})
    file(TO_CMAKE_PATH "$ENV{VCToolsInstallDir}/modules/std.ixx" _ncas_std_default)
  else()
    file(GLOB _ncas_std_candidates
      "C:/Program Files*/Microsoft Visual Studio/*/*/VC/Tools/MSVC/*/modules/std.ixx")
    list(SORT _ncas_std_candidates)
    list(REVERSE _ncas_std_candidates)  # newest toolset first
    list(GET _ncas_std_candidates 0 _ncas_std_default)
  endif()

  set(NIMBLECAS_STD_MODULE_SRC "${_ncas_std_default}"
      CACHE FILEPATH "Path to the standard library std module source (MSVC std.ixx)")
else()
  # libc++ std module source (share/libc++/v1/std.cppm).
  # DEVIATION (documented): uses the system libc++ tree rather than a vendored
  # external/libcxx-v1 (Rule 51). Fine on the fixed build server; vendor for cloud.
  #
  # The location comes from cmake/LibcxxLocate.cmake, which asks clang++ where its OWN
  # libc++.so.1 is and takes the prefix from that. std.cppm must come from the same libc++
  # release as the clang++ compiling it: a mismatched pair either fails to build or, worse,
  # builds against a different standard library than everything else links to.
  #
  # Deriving `/usr/lib/llvm-<major>` from the version number instead would still be a guess --
  # it fixes the version but hardcodes a Debian layout, so a source-built clang on a host that
  # also has a packaged LLVM of the same major would silently take the packaged one.
  if(NIMBLECAS_LIBCXX_PREFIX)
    set(_ncas_std_share "${NIMBLECAS_LIBCXX_PREFIX}/share/libc++/v1")
  else()
    set(_ncas_std_share "")
  endif()

  if(NOT NIMBLECAS_LIBCXX_SHARE AND NOT NIMBLECAS_STD_MODULE_SRC AND NOT _ncas_std_share)
    # Do not invent a path. An unconditional FATAL_ERROR here would also make the escape hatch
    # this message names unreachable, so it is guarded on the caller not having answered.
    message(FATAL_ERROR
      "Could not locate this compiler's libc++ (asked "
      "'${CMAKE_CXX_COMPILER} -print-file-name=libc++.so.1'). Pass "
      "-DNIMBLECAS_LIBCXX_SHARE=<dir containing std.cppm> or "
      "-DNIMBLECAS_STD_MODULE_SRC=<path to std.cppm> explicitly.")
  endif()

  set(NIMBLECAS_LIBCXX_SHARE "${_ncas_std_share}"
      CACHE PATH "Directory containing libc++'s std.cppm / std.compat.cppm")

  # A CACHE variable survives a toolchain change in an existing build dir. If the cached path
  # is not the one this compiler reports, say so rather than silently compiling one release's
  # std.cppm with another release's clang -- the exact failure this derivation prevents, and it
  # leaves no other trace.
  if(_ncas_std_share AND NOT NIMBLECAS_LIBCXX_SHARE STREQUAL _ncas_std_share)
    message(WARNING
      "NIMBLECAS_LIBCXX_SHARE ('${NIMBLECAS_LIBCXX_SHARE}') is not what this compiler reports "
      "('${_ncas_std_share}'). This is usually a STALE CMake cache from a previous toolchain. "
      "Delete the build directory, or set it deliberately.")
  endif()

  set(NIMBLECAS_STD_MODULE_SRC "${NIMBLECAS_LIBCXX_SHARE}/std.cppm"
      CACHE FILEPATH "Path to the standard library std module source (libc++ std.cppm)")
endif()

if(NOT EXISTS "${NIMBLECAS_STD_MODULE_SRC}")
  message(FATAL_ERROR
    "std module source not found at ${NIMBLECAS_STD_MODULE_SRC} "
    "(derived from compiler version ${CMAKE_CXX_COMPILER_VERSION}). "
    "Install the matching libc++ development package, or set "
    "-DNIMBLECAS_STD_MODULE_SRC=<path> to std.cppm (libc++) or std.ixx (MSVC).")
endif()

cmake_path(GET NIMBLECAS_STD_MODULE_SRC PARENT_PATH _ncas_std_dir)

add_library(std_module STATIC)
target_sources(std_module PUBLIC
  FILE_SET CXX_MODULES
  BASE_DIRS "${_ncas_std_dir}"
  FILES "${NIMBLECAS_STD_MODULE_SRC}")
# The std module exports the reserved name `std`; MSVC's std.ixx also uses angled
# includes inside the module purview. Silence both expected diagnostics.
target_compile_options(std_module PRIVATE
  -Wno-reserved-module-identifier
  -Wno-include-angled-in-module-purview)
