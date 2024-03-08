cmake_minimum_required(VERSION 3.22)
project(kokkos-fft-as-library LANGUAGES CXX)

find_package(Kokkos CONFIG REQUIRED)
find_package(KokkosFFT CONFIG REQUIRED)

add_executable(hello-kokkos-fft hello.cpp)
target_link_libraries(hello-kokkos-fft PUBLIC Kokkos::kokkos KokkosFFT::fft)
