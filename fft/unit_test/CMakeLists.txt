# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_executable(unit-tests-kokkos-fft-core Test_Main.cpp Test_Plans.cpp Test_Transform.cpp)

add_executable(unit-tests-kokkos-fft-core-no-init
    Test_MainNoInit.cpp
    Test_InitializeFinalize.cpp
)

target_compile_features(unit-tests-kokkos-fft-core PUBLIC cxx_std_17)
target_link_libraries(unit-tests-kokkos-fft-core PUBLIC KokkosFFT::fft GTest::gtest)

target_compile_features(unit-tests-kokkos-fft-core-no-init PUBLIC cxx_std_17)
target_link_libraries(unit-tests-kokkos-fft-core-no-init PUBLIC KokkosFFT::fft GTest::gtest)

# Enable GoogleTest
include(GoogleTest)
gtest_discover_tests(unit-tests-kokkos-fft-core PROPERTIES DISCOVERY_TIMEOUT 600 DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(unit-tests-kokkos-fft-core-no-init PROPERTIES DISCOVERY_TIMEOUT 600 DISCOVERY_MODE PRE_TEST)
