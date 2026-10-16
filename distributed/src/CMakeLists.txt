# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_library(distributed INTERFACE)
find_package(MPI REQUIRED COMPONENTS CXX)
if(KOKKOSFFT_COMPILE_LANGUAGE STREQUAL CUDA)
  target_compile_features(distributed INTERFACE cuda_std_20)
elseif(KOKKOSFFT_COMPILE_LANGUAGE STREQUAL HIP)
  target_compile_features(distributed INTERFACE hip_std_20)
else()
  target_compile_features(distributed INTERFACE cxx_std_20)
endif()
target_link_libraries(distributed INTERFACE Kokkos::kokkos KokkosFFT::fft MPI::MPI_CXX)

target_include_directories(
  distributed INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:${INSTALL_INCLUDEDIR}>
)
add_library(KokkosFFT::distributed ALIAS distributed)
