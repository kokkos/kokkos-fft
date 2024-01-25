cmake_minimum_required(VERSION 3.23)
project(kokkos-fft LANGUAGES CXX)

# Add cmake helpers for FFTW
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Options
option(BUILD_EXAMPLES "Build kokkos-fft examples" ON)
option(KokkosFFT_ENABLE_HOST_AND_DEVICE "Enable fft on both host and device" OFF)
option(KokkosFFT_INTERNAL_Kokkos "Build internal Kokkos instead of relying on external one" OFF)

if (NOT KokkosFFT_INTERNAL_Kokkos)
    # First check, Kokkos is added as subdirectory or not
    if ("${KOKKOS_PATH}" STREQUAL "")
        find_package(Kokkos REQUIRED)
    else()
        message(STATUS "Using Kokkos at ${KOKKOS_PATH}")
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

# Set directories used for install
include(GNUInstallDirs)
set(LIBDIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
set(INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
set(INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/${LIBDIR})
set(KokkosFFT_EXPORT_TARGET "KokkosFFT-Targets")

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
    VERSION 0.0.0
    COMPATIBILITY SameMajorVersion
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/KokkosFFTConfig.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/KokkosFFTConfigVersion.cmake
    DESTINATION ${INSTALL_LIBDIR}
)