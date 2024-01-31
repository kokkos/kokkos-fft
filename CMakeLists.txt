cmake_minimum_required(VERSION 3.22)
project(KokkosFFT LANGUAGES CXX)

# Add cmake helpers for FFTW
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Options
option(BUILD_EXAMPLES "Build KokkosFFT examples" ON)
option(KokkosFFT_ENABLE_HOST_AND_DEVICE "Enable FFT on both host and device" OFF)
option(KokkosFFT_INTERNAL_Kokkos "Build internal Kokkos instead of relying on external one" OFF)
option(KokkosFFT_ENABLE_BENCHMARK "Build benchmarks for KokkosFFT" OFF)

# Version information
set(KokkosFFT_VERSION_MAJOR 0)
set(KokkosFFT_VERSION_MINOR 0)
set(KokkosFFT_VERSION_PATCH 00)
set(KokkosFFT_VERSION "${KokkosFFT_VERSION_MAJOR}.${KokkosFFT_VERSION_MINOR}.${KokkosFFT_VERSION_PATCH}")

#Set variables for config file
math(EXPR KOKKOSFFT_VERSION "${KokkosFFT_VERSION_MAJOR} * 10000 + ${KokkosFFT_VERSION_MINOR} * 100 + ${KokkosFFT_VERSION_PATCH}")
math(EXPR KOKKOSFFT_VERSION_MAJOR "${KOKKOSFFT_VERSION} / 10000")
math(EXPR KOKKOSFFT_VERSION_MINOR "${KOKKOSFFT_VERSION} / 100 % 100")
math(EXPR KOKKOSFFT_VERSION_PATCH "${KOKKOSFFT_VERSION} % 100")

if (NOT KokkosFFT_INTERNAL_Kokkos)
    # First check, Kokkos is added as subdirectory or not
    if(NOT TARGET Kokkos::kokkos)
        find_package(Kokkos REQUIRED)
    endif()
else ()
    add_subdirectory(tpls/kokkos)
endif ()

# Googletest
include(CTest)
if(BUILD_TESTING)
    find_package(GTest CONFIG)
    if(NOT GTest_FOUND)
        add_subdirectory(tpls/googletest)
    endif()
endif()

# Benchmark
if(KokkosFFT_ENABLE_BENCHMARK)
    option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
    add_subdirectory(tpls/benchmark)

    # [TO DO] Fix this, it detects benchmark not a googlebench
    #find_package(benchmark QUIET)
    #if(NOT benchmark_FOUND)
    #    add_subdirectory(tpls/benchmark)
    #endif()
endif()

# Configure files to display configuration
# Configure the library
set(
    PACKAGE_NAME_CONFIG_FILES
    KokkosFFT_config.h
    KokkosFFT_Version_Info.hpp
)

foreach(CONFIG_FILE ${PACKAGE_NAME_CONFIG_FILES})
    configure_file(
        cmake/${CONFIG_FILE}.in
        ${CMAKE_BINARY_DIR}/${CONFIG_FILE}
    )
endforeach()

# Set directories used for install
include(GNUInstallDirs)
set(LIBDIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
set(INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
set(INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/${LIBDIR})
set(KokkosFFT_EXPORT_TARGET "${PROJECT_NAME}-Targets")

add_subdirectory(common)
add_subdirectory(fft)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation
include(CMakePackageConfigHelpers)

install(
    TARGETS common fft
    EXPORT ${KokkosFFT_EXPORT_TARGET}
)

install(
    EXPORT ${KokkosFFT_EXPORT_TARGET}
    NAMESPACE KokkosFFT::
    DESTINATION ${INSTALL_LIBDIR}
)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/common/src/ ${CMAKE_CURRENT_SOURCE_DIR}/fft/src/
    DESTINATION ${INSTALL_INCLUDEDIR}
)

install(
    FILES ${CMAKE_SOURCE_DIR}/cmake/FindFFTW.cmake
    DESTINATION ${INSTALL_LIBDIR}
)

configure_package_config_file(cmake/KokkosFFTConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/KokkosFFTConfig.cmake
    INSTALL_DESTINATION ${INSTALL_LIBDIR}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/KokkosFFTConfigVersion.cmake
    VERSION ${KokkosFFT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/KokkosFFTConfig.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/KokkosFFTConfigVersion.cmake
    DESTINATION ${INSTALL_LIBDIR}
)