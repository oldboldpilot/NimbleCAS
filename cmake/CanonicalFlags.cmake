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

# AddressSanitizer + UndefinedBehaviorSanitizer + LeakSanitizer build
# (Code Policy Rules 36 / 56). Applied last so -O1/-UNDEBUG win over the release
# defaults above. (Linux toolchain; the Windows path uses the release flags only.)
option(NIMBLECAS_SANITIZE "Build with ASan+UBSan+LSan" OFF)
if(NIMBLECAS_SANITIZE AND NOT WIN32)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all -O1 -UNDEBUG)
  add_link_options(-fsanitize=address,undefined)
endif()
