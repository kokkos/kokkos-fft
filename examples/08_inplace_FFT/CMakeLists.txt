# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_executable(08_inplace_FFT 08_inplace_FFT.cpp)
target_link_libraries(08_inplace_FFT PUBLIC KokkosFFT::fft)
