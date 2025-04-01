# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_executable(
  unit-tests-kokkos-fft-common
  Test_Main.cpp
  Test_Common_Utils.cpp
  Test_Traits.cpp
  Test_Normalization.cpp
  Test_Transpose.cpp
  Test_Extents.cpp
  Test_Padding.cpp
  Test_Helpers.cpp
)

target_compile_features(unit-tests-kokkos-fft-common PUBLIC cxx_std_17)

target_link_libraries(unit-tests-kokkos-fft-common PUBLIC common GTest::gtest)

# Enable GoogleTest
include(GoogleTest)
gtest_discover_tests(unit-tests-kokkos-fft-common PROPERTIES DISCOVERY_TIMEOUT 600 DISCOVERY_MODE PRE_TEST)
