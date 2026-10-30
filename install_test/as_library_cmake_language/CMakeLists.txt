# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.25.2)
project(kokkos-fft-as-library LANGUAGES CXX)

find_package(Kokkos CONFIG REQUIRED)
find_package(KokkosFFT CONFIG REQUIRED)

enable_language(${Kokkos_COMPILE_LANGUAGE})

add_executable(hello-kokkos-fft hello.cpp)
set_source_files_properties(hello.cpp cmake_example.cpp PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
set_target_properties(
  hello-kokkos-fft PROPERTIES ${Kokkos_COMPILE_LANGUAGE}_ARCHITECTURES
                              ${Kokkos_${Kokkos_COMPILE_LANGUAGE}_ARCHITECTURES}
)
target_link_libraries(hello-kokkos-fft PUBLIC Kokkos::kokkos KokkosFFT::fft)
