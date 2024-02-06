cmake_minimum_required(VERSION 3.22)
project(install-test LANGUAGES CXX)

find_package(Kokkos CONFIG REQUIRED)
find_package(KokkosFFT CONFIG REQUIRED)

add_executable(install-test main.cpp)
target_compile_features(install-test PUBLIC cxx_std_17)
target_link_libraries(install-test PUBLIC Kokkos::kokkos KokkosFFT::fft)