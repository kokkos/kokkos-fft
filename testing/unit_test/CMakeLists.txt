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
if(Kokkos_COMPILE_LANGUAGE)
  set_source_files_properties(${UNIT_TEST_TESTING_SOURCES} PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
  set_property(
    TARGET unit-tests-kokkos-fft-testing PROPERTY ${Kokkos_COMPILE_LANGUAGE}_ARCHITECTURES
                                                  ${Kokkos_${Kokkos_COMPILE_LANGUAGE}_ARCHITECTURES}
  )
  set_property(
    TARGET unit-tests-kokkos-fft-testing PROPERTY ${Kokkos_COMPILE_LANGUAGE}_STANDARD
                                                  ${Kokkos_${Kokkos_COMPILE_LANGUAGE}_STANDARD}
  )
else()
  target_compile_features(unit-tests-kokkos-fft-testing PUBLIC cxx_std_20)
endif()

target_link_libraries(unit-tests-kokkos-fft-testing PUBLIC KokkosFFT::testing)
gtest_discover_tests(unit-tests-kokkos-fft-testing PROPERTIES DISCOVERY_TIMEOUT 600 DISCOVERY_MODE PRE_TEST)
