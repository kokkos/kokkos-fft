# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

# Standard FFT examples
add_executable(docs_fft standard/docs_fft.cpp)
target_link_libraries(docs_fft PUBLIC KokkosFFT::fft)

add_executable(docs_ifft standard/docs_ifft.cpp)
target_link_libraries(docs_ifft PUBLIC KokkosFFT::fft)

add_executable(docs_fft2 standard/docs_fft2.cpp)
target_link_libraries(docs_fft2 PUBLIC KokkosFFT::fft)

add_executable(docs_ifft2 standard/docs_ifft2.cpp)
target_link_libraries(docs_ifft2 PUBLIC KokkosFFT::fft)

add_executable(docs_fftn standard/docs_fftn.cpp)
target_link_libraries(docs_fftn PUBLIC KokkosFFT::fft)

add_executable(docs_ifftn standard/docs_ifftn.cpp)
target_link_libraries(docs_ifftn PUBLIC KokkosFFT::fft)

# Real FFT examples
add_executable(docs_rfft real/docs_rfft.cpp)
target_link_libraries(docs_rfft PUBLIC KokkosFFT::fft)

add_executable(docs_irfft real/docs_irfft.cpp)
target_link_libraries(docs_irfft PUBLIC KokkosFFT::fft)
