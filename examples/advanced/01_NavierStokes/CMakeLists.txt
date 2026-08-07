# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_executable(01_navier_stokes 01_navier_stokes.cpp)
target_link_libraries(01_navier_stokes PUBLIC KokkosFFT::fft)
