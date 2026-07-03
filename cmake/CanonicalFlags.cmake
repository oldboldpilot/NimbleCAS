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
