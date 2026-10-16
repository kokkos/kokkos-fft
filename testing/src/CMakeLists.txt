# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_library(testing INTERFACE)
if(KOKKOSFFT_COMPILE_LANGUAGE STREQUAL CUDA)
  target_compile_features(testing INTERFACE cuda_std_20)
elseif(KOKKOSFFT_COMPILE_LANGUAGE STREQUAL HIP)
  target_compile_features(testing INTERFACE hip_std_20)
else()
  target_compile_features(testing INTERFACE cxx_std_20)
endif()
target_link_libraries(testing INTERFACE Kokkos::kokkos GTest::gtest GTest::gmock)

target_include_directories(
  testing INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:${INSTALL_INCLUDEDIR}>
)
add_library(KokkosFFT::testing ALIAS testing)
