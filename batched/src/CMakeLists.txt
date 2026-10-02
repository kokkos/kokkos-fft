add_library(batched INTERFACE)

target_link_libraries(batched INTERFACE Kokkos::kokkos KokkosFFT::fft)

target_compile_features(batched INTERFACE cxx_std_20)

target_include_directories(
  batched
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:${INSTALL_INCLUDEDIR}>)
