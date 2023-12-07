add_executable(01_1DFFT 01_1DFFT.cpp)
target_link_libraries(01_1DFFT PUBLIC Kokkos::fft)