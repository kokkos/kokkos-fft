# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_executable(unit-tests-Kokkos-FFT-testing Test_Main.cpp Test_AreClose.cpp Test_AlmostEqualUlps.cpp Test_CountErrors.cpp Test_FindErrors.cpp Test_PrintErrors.cpp Test_Allclose.cpp
Test_TypeCartesianProduct.cpp Test_AlmostEqualNulp.cpp)

target_compile_features(unit-tests-Kokkos-FFT-testing PUBLIC cxx_std_20)
target_link_libraries(unit-tests-Kokkos-FFT-testing PUBLIC KokkosFFT::testing)

# Enable GoogleTest
include(GoogleTest)
gtest_discover_tests(unit-tests-Kokkos-FFT-testing PROPERTIES DISCOVERY_TIMEOUT 600 DISCOVERY_MODE PRE_TEST)
