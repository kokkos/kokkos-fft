# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_executable(06_1DFFT_reuse_plans 06_1DFFT_reuse_plans.cpp)
target_link_libraries(06_1DFFT_reuse_plans PUBLIC KokkosFFT::fft)
