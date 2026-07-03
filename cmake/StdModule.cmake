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
  # DEVIATION (documented): uses the system libc++-22 tree rather than a vendored
  # external/libcxx-v1 (Rule 51). Fine on the fixed build server; vendor for cloud.
  set(NIMBLECAS_LIBCXX_SHARE "/usr/lib/llvm-22/share/libc++/v1"
      CACHE PATH "Directory containing libc++'s std.cppm / std.compat.cppm")
  set(NIMBLECAS_STD_MODULE_SRC "${NIMBLECAS_LIBCXX_SHARE}/std.cppm"
      CACHE FILEPATH "Path to the standard library std module source (libc++ std.cppm)")
endif()

if(NOT EXISTS "${NIMBLECAS_STD_MODULE_SRC}")
  message(FATAL_ERROR
    "std module source not found at ${NIMBLECAS_STD_MODULE_SRC}. "
    "Set -DNIMBLECAS_STD_MODULE_SRC=<path> to std.cppm (libc++) or std.ixx (MSVC).")
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
