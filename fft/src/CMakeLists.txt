add_library(fft INTERFACE)

target_link_libraries(fft
    INTERFACE
    common
    Kokkos::kokkos
)

target_include_directories(fft INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
add_library(Kokkos::fft ALIAS fft)