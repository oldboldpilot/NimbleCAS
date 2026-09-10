# NimbleCAS canonical compile/link flags (Code Policy Rule 50).
# @author Olumuyiwa Oluwasanmi
#
# The Linux list MUST stay in sync with the CANONICAL_FLAGS array in
# scripts/build_common.sh (divergence causes BMI module-file-config-mismatch errors).
#
# -std=c++23 is supplied by CMAKE_CXX_STANDARD; the rest are applied globally so that
# EVERY translation unit and module (including the std module) is compiled identically.
#
# Two supported toolchains (Code Policy: clang + libc++ on Linux, clang + MSVC STL on
# Windows). The engine is portable C++23 — verified to build and run under both.

if(WIN32)
  # clang targeting x86_64-pc-windows-msvc uses the Microsoft STL (no libc++, no
  # -fPIC/-pthread). The two -Wno flags silence expected diagnostics from Microsoft's
  # std.ixx module (reserved 'std' name; angled includes in the module purview).
  add_compile_options(
    -O3
    -march=x86-64-v3 -mtune=generic
    -mavx -mavx2 -mfma
    -DNDEBUG
    -Wno-reserved-module-identifier
    -Wno-include-angled-in-module-purview)
  # __builtin_cpu_supports (SIMD dynamic dispatch) needs __cpu_model from clang's
  # compiler-rt builtins, which the MSVC-style link does not pull in by default.
  # Link the builtins archive explicitly (located relative to the clang compiler).
  get_filename_component(_ncas_clang_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
  file(GLOB _ncas_builtins
    "${_ncas_clang_bin}/../lib/clang/*/lib/windows/clang_rt.builtins-x86_64.lib")
  if(_ncas_builtins)
    list(SORT _ncas_builtins)
    list(GET _ncas_builtins -1 _ncas_builtins_lib)  # newest clang toolset
    add_link_options("${_ncas_builtins_lib}")
  else()
    message(WARNING "clang_rt.builtins-x86_64.lib not found; SIMD __builtin_cpu_supports "
                    "may fail to link. Set it via add_link_options manually.")
  endif()
  # Reserve an 8 MB stack, matching the Linux and macOS default. Windows reserves only 1 MB.
  #
  # `nimblecas.logic` is a recursive-descent interpreter whose `max_derivation_depth` of 1000
  # exists precisely so that a non-terminating program cannot overflow the native stack — and
  # that guarantee is a statement about the stack it runs on. Leaving Windows at an eighth of
  # what every other platform provides would make a DOCUMENTED INVARIANT quietly false on one
  # platform, which is worse than not claiming it at all. Matching the others means the depth
  # budget means the same thing everywhere, and one number can be reasoned about instead of
  # three. Reserve is ADDRESS SPACE, not memory: the pages are committed lazily, so a query
  # that nests three deep pays for three pages.
  add_link_options(-Wl,/STACK:8388608)
else()
  # clang + vendored/system libc++ on Linux/macOS.
  add_compile_options(
    -stdlib=libc++
    -fexperimental-library
    -D_LIBCPP_ENABLE_EXPERIMENTAL
    -fPIC
    -O3
    -march=x86-64-v3 -mtune=generic
    -mavx -mavx2 -mfma
    -pthread
    -fstack-protector-strong
    -DNDEBUG)
  add_link_options(-stdlib=libc++ -fexperimental-library -pthread)
endif()

# Sanitizer builds (Code Policy Rules 36 / 56). Select one via
#   -DNIMBLECAS_SANITIZER=address|thread|memory|undefined
# (empty = none). NIMBLECAS_SANITIZE=ON is a back-compat alias for "address".
# Applied last so -O1/-g/-UNDEBUG win over the release defaults above. Linux only
# (the Windows path uses the release flags). ASan implies LeakSanitizer.
option(NIMBLECAS_SANITIZE "Alias for -DNIMBLECAS_SANITIZER=address" OFF)
set(NIMBLECAS_SANITIZER "" CACHE STRING
    "Sanitizer: address | thread | memory | undefined (empty = none)")
if(NIMBLECAS_SANITIZE AND NIMBLECAS_SANITIZER STREQUAL "")
  set(NIMBLECAS_SANITIZER "address")
endif()

if(NIMBLECAS_SANITIZER AND NOT WIN32)
  add_compile_options(-fno-omit-frame-pointer -fno-sanitize-recover=all -O1 -g -UNDEBUG)
  if(NIMBLECAS_SANITIZER STREQUAL "address")
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
  elseif(NIMBLECAS_SANITIZER STREQUAL "thread")
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
  elseif(NIMBLECAS_SANITIZER STREQUAL "memory")
    add_compile_options(-fsanitize=memory -fsanitize-memory-track-origins=2)
    add_link_options(-fsanitize=memory)
  elseif(NIMBLECAS_SANITIZER STREQUAL "undefined")
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
  else()
    message(FATAL_ERROR "Unknown NIMBLECAS_SANITIZER='${NIMBLECAS_SANITIZER}'")
  endif()
endif()

# Pin the RUNTIME libc++ to the one belonging to THIS compiler.
#
# The directory comes from cmake/LibcxxLocate.cmake, which asks clang++ itself rather than
# templating a distro path from a version number: same-major but different-installation is a
# real configuration, and it produces exactly the partial, one-binary-only load failure this
# pin exists to remove.
#
# Skipped for MemorySanitizer, which needs an INSTRUMENTED libc++ (scripts/build_msan.sh sets
# its own rpath to the MSan-built tree). Pinning the stock library there would be backwards and
# would bury the run in false positives from uninstrumented library code.
if(NOT WIN32 AND NOT NIMBLECAS_SANITIZER STREQUAL "memory")
  if(NIMBLECAS_LIBCXX_RUNTIME_DIR)
    add_link_options("-Wl,-rpath,${NIMBLECAS_LIBCXX_RUNTIME_DIR}")
  else()
    # Say it out loud. A guard against loading the wrong libc++ that silently does not apply is
    # worse than no guard, because the build looks identical either way.
    message(WARNING
      "Could not locate this compiler's libc++.so.1, so no runtime rpath was pinned. Binaries "
      "will load whichever libc++ the loader finds first, which on a host with more than one "
      "LLVM may not be the one they were compiled against.")
  endif()
endif()
