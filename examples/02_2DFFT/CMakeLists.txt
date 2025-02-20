# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_executable(02_2DFFT 02_2DFFT.cpp)
target_link_libraries(02_2DFFT PUBLIC KokkosFFT::fft)
