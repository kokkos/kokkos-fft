# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.22)

set(KOKKOSFFT_TOP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Set project
project(KokkosFFT VERSION 0.2.1 LANGUAGES CXX)

# Add cmake helpers
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Options
option(KokkosFFT_ENABLE_HOST_AND_DEVICE "Enable FFT on both host and device" OFF)
option(KokkosFFT_ENABLE_INTERNAL_KOKKOS "Build internal Kokkos instead of relying on external one" OFF)
option(KokkosFFT_ENABLE_EXAMPLES "Build KokkosFFT examples" OFF)
option(KokkosFFT_ENABLE_TESTS "Build KokkosFFT tests" OFF)
option(KokkosFFT_ENABLE_BENCHMARK "Build benchmarks for KokkosFFT" OFF)
option(KokkosFFT_ENABLE_DOCS "Build KokkosFFT documentaion/website" OFF)

# Version information
set(KOKKOSFFT_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(KOKKOSFFT_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(KOKKOSFFT_VERSION_PATCH ${PROJECT_VERSION_PATCH})

set(KOKKOS_REQUIRED_VERSION 4.4.0)

if(NOT KokkosFFT_ENABLE_INTERNAL_KOKKOS)
  # First check, Kokkos is added as subdirectory or not
  if(NOT TARGET Kokkos::kokkos)
    find_package(Kokkos ${KOKKOS_REQUIRED_VERSION} REQUIRED)
    # Check the alignment of complex numbers
    kokkos_check(OPTIONS COMPLEX_ALIGN)
  endif()
else()
  add_subdirectory(tpls/kokkos)
endif()

# Device specific options
set(KokkosFFT_TO_ENABLE_FFTW ON)
if(Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_SYCL)
  if(KokkosFFT_ENABLE_HOST_AND_DEVICE)
    message(WARNING "KokkosFFT_ENABLE_HOST_AND_DEVICE is deprecated. Please enable FFTW explicitly with -DKokkosFFT_ENABLE_FFTW=ON")
  else()
    set(KokkosFFT_TO_ENABLE_FFTW OFF)
  endif()
endif()

option(KokkosFFT_ENABLE_FFTW "Enable fftw as a KokkosFFT backend on CPUs" ${KokkosFFT_TO_ENABLE_FFTW})
option(KokkosFFT_ENABLE_CUFFT "Enable cufft as a KokkosFFT backend on NVIDIA GPUs" ${Kokkos_ENABLE_CUDA})
option(KokkosFFT_ENABLE_ROCFFT "Enable rocfft as a KokkosFFT backend on AMD GPUs" OFF)
option(KokkosFFT_ENABLE_ONEMKL "Enable onemkl as a KokkosFFT backend on Intel GPUs" ${Kokkos_ENABLE_SYCL})

set(KokkosFFT_TO_ENABLE_HIPFFT ${Kokkos_ENABLE_HIP})
if(KokkosFFT_ENABLE_ROCFFT)
  set(KokkosFFT_TO_ENABLE_HIPFFT OFF)
endif()
option(KokkosFFT_ENABLE_HIPFFT "Enable hipfft as a KokkosFFT backend on AMD GPUs" ${KokkosFFT_TO_ENABLE_HIPFFT})

# Check if multiple device tpls are not enabled
set(ENABLED_DEVICE_TPLS "")
set(KOKKOSFFT_DEVICE_TPL_LIST "CUFFT" "ROCFFT" "HIPFFT" "ONEMKL")
foreach(DEVICE_TPL IN LISTS KOKKOSFFT_DEVICE_TPL_LIST)
  if(KokkosFFT_ENABLE_${DEVICE_TPL})
    list(APPEND ENABLED_DEVICE_TPLS ${DEVICE_TPL})
  endif()
endforeach()

# Get the number of enabled tpls
list(LENGTH ENABLED_DEVICE_TPLS COUNT)
if(COUNT GREATER 1)
  message(FATAL_ERROR "Error: Only one device tpls may be enabled. The following options are enabled: ${ENABLED_DEVICE_TPLS}")
endif()

# Set tpls
set(KOKKOSFFT_HAS_DEFAULT_BACKEND)
set(KOKKOSFFT_TPL_LIST)
set(KOKKOSFFT_BACKEND_LIST)
include(KokkosFFT_tpls)
get_tpls_list(KOKKOSFFT_TPL_LIST KOKKOSFFT_BACKEND_LIST KOKKOSFFT_HAS_DEFAULT_BACKEND)

# ==================================================================
# CMake Summary
# ==================================================================

message("")
message(STATUS "KokkosFFT version: ${KOKKOSFFT_VERSION_MAJOR}.${KOKKOSFFT_VERSION_MINOR}.${KOKKOSFFT_VERSION_PATCH}")
message(STATUS "KokkosFFT TPLs:")
if(KOKKOSFFT_TPL_LIST)
  include(KokkosFFT_utils)
  foreach(BACKEND TPL IN ZIP_LISTS KOKKOSFFT_BACKEND_LIST KOKKOSFFT_TPL_LIST)
    # [TO DO] show more information about the library (like location)
    pad_string(BACKEND_PADDED "${BACKEND}" 13)
    message(STATUS "    ${BACKEND_PADDED}: ${TPL}")
  endforeach()
else()
  message(STATUS " (None)")
endif()
message("")

# Googletest
if(KokkosFFT_ENABLE_TESTS)
  include(CTest)
  find_package(GTest CONFIG)
  if(NOT GTest_FOUND)
    add_subdirectory(tpls/googletest)
  endif()
endif()

# Build documentation with Doxygen and Sphinx
if(KokkosFFT_ENABLE_DOCS)
  find_package(Doxygen REQUIRED)
  find_package(Sphinx REQUIRED)
  add_subdirectory(docs)
endif()

# Benchmark
if(KokkosFFT_ENABLE_BENCHMARK)
  option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
  add_subdirectory(tpls/benchmark)
  include(KokkosFFT_Git_Hash)

  # [TO DO] Fix this, it detects benchmark not a googlebench
  #find_package(benchmark QUIET)
  #if(NOT benchmark_FOUND)
  #    add_subdirectory(tpls/benchmark)
  #endif()
endif()

# Configure files to display configuration
# Configure the library
set(PACKAGE_NAME_CONFIG_FILES KokkosFFT_config.h KokkosFFT_Version_Info.hpp)

foreach(CONFIG_FILE ${PACKAGE_NAME_CONFIG_FILES})
  configure_file(cmake/${CONFIG_FILE}.in ${CMAKE_BINARY_DIR}/${CONFIG_FILE})
endforeach()

# Set directories used for install
include(GNUInstallDirs)
set(LIBDIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
set(INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR})
set(INSTALL_LIBDIR ${LIBDIR})
set(KokkosFFT_EXPORT_TARGET "${PROJECT_NAME}-Targets")

add_subdirectory(common)
add_subdirectory(fft)
if(KokkosFFT_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

# Installation
include(CMakePackageConfigHelpers)

install(TARGETS common fft EXPORT ${KokkosFFT_EXPORT_TARGET})
install(EXPORT ${KokkosFFT_EXPORT_TARGET} NAMESPACE KokkosFFT:: DESTINATION ${INSTALL_LIBDIR})

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/common/src/ ${CMAKE_CURRENT_SOURCE_DIR}/fft/src/
  DESTINATION ${INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.hpp"
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindFFTW.cmake DESTINATION ${INSTALL_LIBDIR})

configure_package_config_file(
  cmake/KokkosFFTConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/KokkosFFTConfig.cmake
  INSTALL_DESTINATION ${INSTALL_LIBDIR}
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/KokkosFFTConfigVersion.cmake VERSION ${KokkosFFT_VERSION} COMPATIBILITY SameMajorVersion
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/KokkosFFTConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/KokkosFFTConfigVersion.cmake
        DESTINATION ${INSTALL_LIBDIR}
)
