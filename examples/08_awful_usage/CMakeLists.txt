# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_executable(08_awful_usage 08_awful_usage.cpp)
target_link_libraries(08_awful_usage PUBLIC KokkosFFT::fft)
