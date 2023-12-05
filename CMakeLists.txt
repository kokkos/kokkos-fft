cmake_minimum_required(VERSION 3.23)
project(kokkos-fft LANGUAGES CXX)

# Add cmake helpers for FFTW
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/cmake")

find_package(Kokkos CONFIG)
if(NOT kokkos_FOUND)
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

add_subdirectory(common)
add_subdirectory(fft)