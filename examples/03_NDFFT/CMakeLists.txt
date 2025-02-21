# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_executable(03_NDFFT 03_NDFFT.cpp)
target_link_libraries(03_NDFFT PUBLIC KokkosFFT::fft)
