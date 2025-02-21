# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_executable(04_batchedFFT 04_batchedFFT.cpp)
target_link_libraries(04_batchedFFT PUBLIC KokkosFFT::fft)
