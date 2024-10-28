# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_executable(07_unmanaged_views 07_unmanaged_views.cpp)
target_link_libraries(07_unmanaged_views PUBLIC KokkosFFT::fft)
