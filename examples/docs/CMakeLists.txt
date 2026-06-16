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

add_executable(docs_rfft2 real/docs_rfft2.cpp)
target_link_libraries(docs_rfft2 PUBLIC KokkosFFT::fft)

add_executable(docs_irfft2 real/docs_irfft2.cpp)
target_link_libraries(docs_irfft2 PUBLIC KokkosFFT::fft)

add_executable(docs_rfftn real/docs_rfftn.cpp)
target_link_libraries(docs_rfftn PUBLIC KokkosFFT::fft)

add_executable(docs_irfftn real/docs_irfftn.cpp)
target_link_libraries(docs_irfftn PUBLIC KokkosFFT::fft)

# Hermitian FFT examples
add_executable(docs_hfft hermitian/docs_hfft.cpp)
target_link_libraries(docs_hfft PUBLIC KokkosFFT::fft)

add_executable(docs_ihfft hermitian/docs_ihfft.cpp)
target_link_libraries(docs_ihfft PUBLIC KokkosFFT::fft)

# Helper routines
add_executable(docs_fftfreq helper/docs_fftfreq.cpp)
target_link_libraries(docs_fftfreq PUBLIC KokkosFFT::fft)

add_executable(docs_rfftfreq helper/docs_rfftfreq.cpp)
target_link_libraries(docs_rfftfreq PUBLIC KokkosFFT::fft)
