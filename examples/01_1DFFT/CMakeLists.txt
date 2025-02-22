# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_executable(01_1DFFT 01_1DFFT.cpp)
target_link_libraries(01_1DFFT PUBLIC KokkosFFT::fft)
