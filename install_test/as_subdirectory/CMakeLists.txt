# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.22)
project(kokkos-fft-as-subdirectory LANGUAGES CXX)

add_subdirectory(tpls/kokkos)
add_subdirectory(tpls/kokkos-fft)

add_executable(hello-kokkos-fft hello.cpp)
target_link_libraries(hello-kokkos-fft PUBLIC Kokkos::kokkos KokkosFFT::fft)
