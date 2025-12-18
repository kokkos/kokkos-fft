# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_library(testing INTERFACE)

target_link_libraries(testing INTERFACE Kokkos::kokkos GTest::gtest GTest::gmock)

target_include_directories(
    testing INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:${INSTALL_INCLUDEDIR}>
)
add_library(KokkosFFT::testing ALIAS testing)
