# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

# Enable GoogleTest
include(GoogleTest)

set(UNIT_TEST_COMMON_SOURCES
    Test_Main.cpp
    Test_Assert.cpp
    Test_CheckConditions.cpp
    Test_Container_Helpers.cpp
    Test_Convert_Types.cpp
    Test_Traits.cpp
    Test_Layout.cpp
    Test_Normalization.cpp
    Test_MDOperations.cpp
    Test_Mapping.cpp
    Test_Extents.cpp
    Test_Padding.cpp
    Test_Helpers.cpp
)

add_executable(unit-tests-kokkos-fft-common ${UNIT_TEST_COMMON_SOURCES})
if(Kokkos_COMPILE_LANGUAGE)
  set_source_files_properties(${UNIT_TEST_COMMON_SOURCES} PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
  set_property(
    TARGET unit-tests-kokkos-fft-common PROPERTY ${Kokkos_COMPILE_LANGUAGE}_ARCHITECTURES
                                                 ${Kokkos_${Kokkos_COMPILE_LANGUAGE}_ARCHITECTURES}
  )
  set_property(
    TARGET unit-tests-kokkos-fft-common PROPERTY ${Kokkos_COMPILE_LANGUAGE}_STANDARD
                                                 ${Kokkos_${Kokkos_COMPILE_LANGUAGE}_STANDARD}
  )
else()
  target_compile_features(unit-tests-kokkos-fft-common PUBLIC cxx_std_20)
endif()

target_link_libraries(unit-tests-kokkos-fft-common PUBLIC common KokkosFFT::testing GTest::gtest GTest::gmock)

set(UNIT_TEST_TRANSPOSE_SOURCES Test_Main.cpp Test_Transpose.cpp)
add_executable(unit-tests-kokkos-transpose ${UNIT_TEST_TRANSPOSE_SOURCES})
if(Kokkos_COMPILE_LANGUAGE)
  set_source_files_properties(${UNIT_TEST_TRANSPOSE_SOURCES} PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
  set_property(
    TARGET unit-tests-kokkos-transpose PROPERTY ${Kokkos_COMPILE_LANGUAGE}_ARCHITECTURES
                                                ${Kokkos_${Kokkos_COMPILE_LANGUAGE}_ARCHITECTURES}
  )
  set_property(
    TARGET unit-tests-kokkos-transpose PROPERTY ${Kokkos_COMPILE_LANGUAGE}_STANDARD
                                                ${Kokkos_${Kokkos_COMPILE_LANGUAGE}_STANDARD}
  )
else()
  target_compile_features(unit-tests-kokkos-transpose PUBLIC cxx_std_20)
endif()

target_link_libraries(unit-tests-kokkos-transpose PUBLIC common KokkosFFT::testing GTest::gtest GTest::gmock)

gtest_discover_tests(unit-tests-kokkos-fft-common PROPERTIES DISCOVERY_TIMEOUT 600 DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(unit-tests-kokkos-transpose PROPERTIES DISCOVERY_TIMEOUT 600 DISCOVERY_MODE PRE_TEST)
