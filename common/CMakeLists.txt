# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_subdirectory(src)
if(KokkosFFT_ENABLE_TESTS)
  add_subdirectory(unit_test)
endif()
