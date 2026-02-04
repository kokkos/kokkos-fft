# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_subdirectory(src)
if(KokkosFFT_ENABLE_TESTS AND NOT(KokkosFFT_ENABLE_TESTS_DISTRIBUTED_ONLY))
  add_subdirectory(unit_test)
endif()

if(KokkosFFT_ENABLE_BENCHMARK)
  add_subdirectory(perf_test)
endif()
