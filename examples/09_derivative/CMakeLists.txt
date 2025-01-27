# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_executable(09_derivative 09_derivative.cpp)
target_link_libraries(09_derivative PUBLIC KokkosFFT::fft)
