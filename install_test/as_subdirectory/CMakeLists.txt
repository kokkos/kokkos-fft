# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.22)
project(Kokkos-FFT-as-subdirectory LANGUAGES CXX)

add_subdirectory(tpls/kokkos)
add_subdirectory(tpls/Kokkos-FFT)

add_executable(hello-Kokkos-FFT hello.cpp)
target_link_libraries(hello-Kokkos-FFT PUBLIC Kokkos::kokkos KokkosFFT::fft)
