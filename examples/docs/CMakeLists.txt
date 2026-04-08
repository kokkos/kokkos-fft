# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_executable(docs_fft docs_fft.cpp)
target_link_libraries(docs_fft PUBLIC KokkosFFT::fft)

add_executable(docs_ifft docs_ifft.cpp)
target_link_libraries(docs_ifft PUBLIC KokkosFFT::fft)

add_executable(docs_fft2 docs_fft2.cpp)
target_link_libraries(docs_fft2 PUBLIC KokkosFFT::fft)

add_executable(docs_ifft2 docs_ifft2.cpp)
target_link_libraries(docs_ifft2 PUBLIC KokkosFFT::fft)

add_executable(docs_fftn docs_fftn.cpp)
target_link_libraries(docs_fftn PUBLIC KokkosFFT::fft)

add_executable(docs_ifftn docs_ifftn.cpp)
target_link_libraries(docs_ifftn PUBLIC KokkosFFT::fft)
