# NimbleCAS toolchain file — clang++-22 (Code Policy Rule 50).
# @author Olumuyiwa Oluwasanmi
#
# Usage: cmake -DCMAKE_TOOLCHAIN_FILE=config/toolchain.cmake ...
# Locates clang++-22; override with -DNIMBLECAS_CLANGXX=/path/to/clang++-22.

set(CMAKE_SYSTEM_NAME ${CMAKE_HOST_SYSTEM_NAME})

if(NOT DEFINED NIMBLECAS_CLANGXX)
  find_program(NIMBLECAS_CLANGXX NAMES clang++-22 clang++ REQUIRED)
endif()

set(CMAKE_CXX_COMPILER "${NIMBLECAS_CLANGXX}" CACHE FILEPATH "C++ compiler (clang++-22)")
