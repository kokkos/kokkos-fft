# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_executable(docs_fft docs_fft.cpp)
target_link_libraries(docs_fft PUBLIC KokkosFFT::fft)
