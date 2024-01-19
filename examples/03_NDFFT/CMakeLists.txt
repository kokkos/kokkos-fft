add_executable(03_NDFFT 03_NDFFT.cpp)
target_link_libraries(03_NDFFT PUBLIC KokkosFFT::fft)