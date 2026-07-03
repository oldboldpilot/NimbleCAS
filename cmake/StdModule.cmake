# NimbleCAS `import std` support (Code Policy Rules 12 / 41).
# @author Olumuyiwa Oluwasanmi
#
# CMake's built-in `import std` support is gated behind a version-locked experimental
# UUID and is discouraged by the policy's explicit-module discipline (Rules 50-52).
# Instead we compile libc++'s std.cppm as an ordinary CXX_MODULES library target, so
# CMake's module scanner resolves `import std;` in our translation units normally and
# with explicit, reproducible dependency edges.
#
# DEVIATION (documented): uses the system libc++-22 std module source rather than a
# vendored external/libcxx-v1 tree (Rule 51). Acceptable on the single fixed build
# server; vendor before multi-host / cloud builds.

set(NIMBLECAS_LIBCXX_SHARE "/usr/lib/llvm-22/share/libc++/v1"
    CACHE PATH "Directory containing libc++'s std.cppm / std.compat.cppm")

if(NOT EXISTS "${NIMBLECAS_LIBCXX_SHARE}/std.cppm")
  message(FATAL_ERROR
    "libc++ std module source not found at ${NIMBLECAS_LIBCXX_SHARE}/std.cppm. "
    "Set -DNIMBLECAS_LIBCXX_SHARE=<dir> to the libc++-22 share/libc++/v1 directory.")
endif()

add_library(std_module STATIC)
target_sources(std_module PUBLIC
  FILE_SET CXX_MODULES
  BASE_DIRS "${NIMBLECAS_LIBCXX_SHARE}"
  FILES "${NIMBLECAS_LIBCXX_SHARE}/std.cppm")
# std.cppm exports the reserved module name `std`; silence the expected diagnostic.
target_compile_options(std_module PRIVATE -Wno-reserved-module-identifier)
