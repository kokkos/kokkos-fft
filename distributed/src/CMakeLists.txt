# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_library(distributed INTERFACE)
find_package(MPI REQUIRED COMPONENTS CXX)
target_link_libraries(distributed INTERFACE Kokkos::kokkos KokkosFFT::fft MPI::MPI_CXX)

target_include_directories(
  distributed INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:${INSTALL_INCLUDEDIR}>
)
add_library(KokkosFFT::distributed ALIAS distributed)
