add_executable(04_batchedFFT 04_batchedFFT.cpp)
target_link_libraries(04_batchedFFT PUBLIC Kokkos::fft)