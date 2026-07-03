# NimbleCAS canonical compile/link flags (Code Policy Rule 50).
# @author Olumuyiwa Oluwasanmi
#
# This list MUST stay in sync with the CANONICAL_FLAGS array in
# scripts/build_common.sh. Divergence causes BMI module-file-config-mismatch errors.
#
# -std=c++23 is supplied by CMAKE_CXX_STANDARD; the rest are applied globally so that
# EVERY translation unit and module (including the std module) is compiled identically.

# Applied to all targets. -O3/-DNDEBUG are the release default; the sanitizer build
# below overrides optimisation and re-enables assertions.
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

# AddressSanitizer + UndefinedBehaviorSanitizer + LeakSanitizer build
# (Code Policy Rules 36 / 56). Applied last so -O1/-UNDEBUG win over the release
# defaults above.
option(NIMBLECAS_SANITIZE "Build with ASan+UBSan+LSan" OFF)
if(NIMBLECAS_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all -O1 -UNDEBUG)
  add_link_options(-fsanitize=address,undefined)
endif()
