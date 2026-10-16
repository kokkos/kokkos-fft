# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

# Enable GoogleTest
include(GoogleTest)

set(UNIT_TEST_CORE_SOURCES Test_Main.cpp Test_Plans.cpp Test_Transform.cpp)
add_executable(unit-tests-kokkos-fft-core ${UNIT_TEST_CORE_SOURCES})
if(KOKKOSFFT_COMPILE_LANGUAGE)
  set_source_files_properties(${UNIT_TEST_CORE_SOURCES} PROPERTIES LANGUAGE ${KOKKOSFFT_COMPILE_LANGUAGE})
  set_property(
    TARGET unit-tests-kokkos-fft-core PROPERTY ${KOKKOSFFT_COMPILE_LANGUAGE}_ARCHITECTURES
                                               ${KOKKOSFFT_${KOKKOSFFT_COMPILE_LANGUAGE}_ARCHITECTURES}
  )
endif()

target_link_libraries(unit-tests-kokkos-fft-core PUBLIC KokkosFFT::fft GTest::gtest)

set(UNIT_TEST_DYN_SOURCES Test_Main.cpp Test_DynPlans.cpp)
add_executable(unit-tests-kokkos-dynfft ${UNIT_TEST_DYN_SOURCES})
if(KOKKOSFFT_COMPILE_LANGUAGE)
  set_source_files_properties(${UNIT_TEST_DYN_SOURCES} PROPERTIES LANGUAGE ${KOKKOSFFT_COMPILE_LANGUAGE})
  set_property(
    TARGET unit-tests-kokkos-dynfft PROPERTY ${KOKKOSFFT_COMPILE_LANGUAGE}_ARCHITECTURES
                                             ${KOKKOSFFT_${KOKKOSFFT_COMPILE_LANGUAGE}_ARCHITECTURES}
  )
endif()

target_link_libraries(unit-tests-kokkos-dynfft PUBLIC KokkosFFT::fft GTest::gtest)

gtest_discover_tests(unit-tests-kokkos-fft-core PROPERTIES DISCOVERY_TIMEOUT 600 DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(unit-tests-kokkos-dynfft PROPERTIES DISCOVERY_TIMEOUT 600 DISCOVERY_MODE PRE_TEST)

if(KokkosFFT_ENABLE_CALLBACK)
  add_executable(unit-tests-kokkos-fft-callback Test_Main.cpp Test_Callback.cpp)
  target_link_libraries(
    unit-tests-kokkos-fft-callback PUBLIC KokkosFFT::fft KokkosFFT::testing GTest::gtest GTest::gmock
  )
  gtest_discover_tests(unit-tests-kokkos-fft-callback PROPERTIES DISCOVERY_TIMEOUT 600 DISCOVERY_MODE PRE_TEST)
endif()
