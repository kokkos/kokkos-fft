# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

# Enable GoogleTest
include(GoogleTest)

set(UNIT_TEST_TESTING_SOURCES
    Test_Main.cpp
    Test_AreClose.cpp
    Test_AlmostEqualUlps.cpp
    Test_CountErrors.cpp
    Test_FindErrors.cpp
    Test_PrintErrors.cpp
    Test_Allclose.cpp
    Test_TypeCartesianProduct.cpp
    Test_AlmostEqualNulp.cpp
)

add_executable(unit-tests-kokkos-fft-testing ${UNIT_TEST_TESTING_SOURCES})
kokkosfft_set_language(${UNIT_TEST_TESTING_SOURCES})
kokkosfft_set_architectures(unit-tests-kokkos-fft-testing)
target_link_libraries(unit-tests-kokkos-fft-testing PUBLIC KokkosFFT::testing)

gtest_discover_tests(unit-tests-kokkos-fft-testing PROPERTIES DISCOVERY_TIMEOUT 600 DISCOVERY_MODE PRE_TEST)
