add_library(fft INTERFACE)

target_link_libraries(fft
    INTERFACE
    common
    Kokkos::kokkos
)

target_include_directories(fft INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           $<INSTALL_INTERFACE:${INSTALL_INCLUDEDIR}>
)
add_library(Kokkos::fft ALIAS fft)