# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_executable(
  unit-tests-Kokkos-FFT-common
  Test_Main.cpp
  Test_Common_Utils.cpp
  Test_Traits.cpp
  Test_Layout.cpp
  Test_Normalization.cpp
  Test_Mapping.cpp
  Test_Transpose.cpp
  Test_Extents.cpp
  Test_Padding.cpp
  Test_Helpers.cpp
)

target_compile_features(unit-tests-Kokkos-FFT-common PUBLIC cxx_std_17)

target_link_libraries(unit-tests-Kokkos-FFT-common PUBLIC common GTest::gtest)

# Enable GoogleTest
include(GoogleTest)
gtest_discover_tests(unit-tests-Kokkos-FFT-common PROPERTIES DISCOVERY_TIMEOUT 600 DISCOVERY_MODE PRE_TEST)
