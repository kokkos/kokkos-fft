cmake_minimum_required(VERSION 3.23)
project(kokkos-fft LANGUAGES CXX)

# Add cmake helpers for FFTW
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/cmake")

# Options
option(BUILD_EXAMPLES "Build kokkos-fft examples" ON)
option(KokkosFFT_ENABLE_HOST_AND_DEVICE "Enable fft on both host and device" OFF)
option(KokkosFFT_ENABLE_INSTALL_TEST "Enable install test" ON)

find_package(Kokkos CONFIG)
if(NOT Kokkos_FOUND)
    add_subdirectory(tpls/kokkos)
endif()

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
set(INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
set(INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

add_subdirectory(common)
add_subdirectory(fft)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation
install(
    TARGETS common fft
    EXPORT KokkosFFT-targets
)

install(
    EXPORT KokkosFFT-targets
    DESTINATION ${INSTALL_INCLUDEDIR}
)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/common/
    DESTINATION ${INSTALL_INCLUDEDIR}
)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/fft/
    DESTINATION ${INSTALL_INCLUDEDIR}
)

configure_package_config_file(cmake/KokkosFFTConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/KokkosFFTConfig.cmake
    INSTALL_DESTINATION ${INSTALL_INCLUDEDIR}
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/KokkosFFTConfig.cmake
    DESTINATION ${INSTALL_INCLUDEDIR}
)