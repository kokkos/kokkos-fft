# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.22)
project(Kokkos-FFT-as-library LANGUAGES CXX)

find_package(Kokkos CONFIG REQUIRED)
find_package(KokkosFFT CONFIG REQUIRED)

add_executable(hello-Kokkos-FFT hello.cpp)
target_link_libraries(hello-Kokkos-FFT PUBLIC Kokkos::kokkos KokkosFFT::fft)
