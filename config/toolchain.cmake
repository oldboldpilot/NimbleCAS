# NimbleCAS toolchain file — clang++-23 (Code Policy Rule 50).
# @author Olumuyiwa Oluwasanmi
#
# Usage: cmake -DCMAKE_TOOLCHAIN_FILE=config/toolchain.cmake ...
# Locates clang++-23; override with -DNIMBLECAS_CLANGXX=/path/to/clang++-23.
# Also pins the C compiler and the module scanner: CMake would otherwise default CC to
# the system cc (gcc), which rejects -stdlib=libc++, and cannot discover the module
# import graph without a clang-scan-deps of the SAME release.

set(CMAKE_SYSTEM_NAME ${CMAKE_HOST_SYSTEM_NAME})

if(NOT DEFINED NIMBLECAS_CLANGXX)
  find_program(NIMBLECAS_CLANGXX NAMES clang++-23 clang++ REQUIRED)
endif()

set(CMAKE_CXX_COMPILER "${NIMBLECAS_CLANGXX}" CACHE FILEPATH "C++ compiler (clang++-23)")

if(NOT DEFINED NIMBLECAS_CLANGC)
  find_program(NIMBLECAS_CLANGC NAMES clang-23 clang REQUIRED)
endif()
set(CMAKE_C_COMPILER "${NIMBLECAS_CLANGC}" CACHE FILEPATH "C compiler (clang-23)")

if(NOT DEFINED NIMBLECAS_SCAN_DEPS)
  find_program(NIMBLECAS_SCAN_DEPS NAMES clang-scan-deps-23 clang-scan-deps REQUIRED)
endif()
set(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS "${NIMBLECAS_SCAN_DEPS}"
    CACHE FILEPATH "C++20 module dependency scanner (clang-scan-deps-23)")
