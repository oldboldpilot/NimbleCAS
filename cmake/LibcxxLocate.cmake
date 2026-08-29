# NimbleCAS libc++ location — ASK THE COMPILER, do not template a distro path.
# @author Olumuyiwa Oluwasanmi
#
# Two things must come from the SAME libc++ as the clang++ compiling the code: the `std`
# module source (std.cppm) and the libc++ the binaries load at runtime. Getting either from a
# different release is the failure this file exists to prevent, and it is a nasty one — the
# build succeeds, almost everything runs, and then a single binary that happens to touch a
# symbol new in one release dies at load with an undefined symbol, which reads as a broken
# test rather than a broken library path.
#
# An earlier version derived `/usr/lib/llvm-<major>/...` from CMAKE_CXX_COMPILER_VERSION. That
# beats a hardcoded `llvm-22`, but it is still a guess: it derives only the version NUMBER and
# templates a Debian layout around it, so a source-built clang reporting 23.x on a host that
# also has a packaged llvm-23 would silently be pointed at the packaged one.
#
# `-print-resource-dir` is the authoritative answer. It names THIS driver's own resource
# directory, `<llvm-prefix>/lib/clang/<major>`, from which the installation prefix is three
# levels up. `-print-file-name=libc++.so.1` is deliberately NOT used for the prefix: it
# resolves through the ordinary library search path and commonly lands in a multiarch system
# directory (`/lib/x86_64-linux-gnu/...`) whose parent holds no `share/`.
#
# Sets, in the including scope:
#   NIMBLECAS_LIBCXX_PREFIX       the LLVM installation prefix ("" if unknown)
#   NIMBLECAS_LIBCXX_RUNTIME_DIR  directory to rpath for libc++.so.1 ("" if unknown)

set(NIMBLECAS_LIBCXX_PREFIX "")
set(NIMBLECAS_LIBCXX_RUNTIME_DIR "")

if(NOT WIN32 AND CMAKE_CXX_COMPILER)
  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -print-resource-dir
    OUTPUT_VARIABLE _ncas_resource_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _ncas_resource_rc)

  if(_ncas_resource_rc EQUAL 0 AND IS_ABSOLUTE "${_ncas_resource_dir}")
    # <prefix>/lib/clang/<major>  ->  <prefix>
    get_filename_component(_ncas_p "${_ncas_resource_dir}" DIRECTORY)   # .../lib/clang
    get_filename_component(_ncas_p "${_ncas_p}" DIRECTORY)              # .../lib
    get_filename_component(_ncas_p "${_ncas_p}" DIRECTORY)              # <prefix>
    if(EXISTS "${_ncas_p}")
      set(NIMBLECAS_LIBCXX_PREFIX "${_ncas_p}")
    endif()
  endif()

  # Runtime directory: prefer the libc++ shipped INSIDE this LLVM installation, which is
  # unambiguously the matching one. Fall back to whatever the driver would actually link
  # against, which is still far better than letting the loader choose.
  if(NIMBLECAS_LIBCXX_PREFIX AND EXISTS "${NIMBLECAS_LIBCXX_PREFIX}/lib/libc++.so.1")
    set(NIMBLECAS_LIBCXX_RUNTIME_DIR "${NIMBLECAS_LIBCXX_PREFIX}/lib")
  else()
    execute_process(
      COMMAND "${CMAKE_CXX_COMPILER}" -stdlib=libc++ -print-file-name=libc++.so.1
      OUTPUT_VARIABLE _ncas_libcxx_so
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE _ncas_libcxx_rc)
    # -print-file-name echoes its argument back when it cannot resolve it, so an absolute
    # path that exists is the success signal.
    if(_ncas_libcxx_rc EQUAL 0 AND IS_ABSOLUTE "${_ncas_libcxx_so}"
       AND EXISTS "${_ncas_libcxx_so}")
      get_filename_component(NIMBLECAS_LIBCXX_RUNTIME_DIR "${_ncas_libcxx_so}" DIRECTORY)
    endif()
  endif()
endif()
