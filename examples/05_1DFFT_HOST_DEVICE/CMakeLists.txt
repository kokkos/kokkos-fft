# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_executable(05_1DFFT_HOST_DEVICE 05_1DFFT_HOST_DEVICE.cpp)
target_link_libraries(05_1DFFT_HOST_DEVICE PUBLIC KokkosFFT::fft)
