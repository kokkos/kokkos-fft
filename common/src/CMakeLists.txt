add_library(common INTERFACE)

target_link_libraries(common
    INTERFACE
    Kokkos::kokkos
)

if(Kokkos_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED COMPONENTS cufft)
  target_link_libraries(common INTERFACE CUDA::cufft)
  if(KokkosFFT_ENABLE_HOST_AND_DEVICE)
    find_package(FFTW MODULE REQUIRED)
    target_link_libraries(common INTERFACE FFTW::Float FFTW::Double FFTW::FloatOpenMP FFTW::DoubleOpenMP)
    target_compile_definitions(common INTERFACE ENABLE_HOST_AND_DEVICE)
  endif()
elseif(Kokkos_ENABLE_HIP)
  find_package(hipfft REQUIRED)
  target_link_libraries(common INTERFACE hip::hipfft)
  if(KokkosFFT_ENABLE_HOST_AND_DEVICE)
    find_package(FFTW MODULE REQUIRED)
    target_link_libraries(common INTERFACE FFTW::Float FFTW::Double FFTW::FloatOpenMP FFTW::DoubleOpenMP)
    target_compile_definitions(common INTERFACE ENABLE_HOST_AND_DEVICE)
  endif()
elseif(Kokkos_ENABLE_OPENMP)
  find_package(FFTW MODULE REQUIRED)
  target_link_libraries(common INTERFACE FFTW::Float FFTW::Double FFTW::FloatOpenMP FFTW::DoubleOpenMP)
elseif(Kokkos_ENABLE_SERIAL)
  find_package(FFTW MODULE REQUIRED)
  target_link_libraries(common INTERFACE FFTW::Float FFTW::Double)
endif()

target_compile_features(common INTERFACE cxx_std_17)
target_include_directories(common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})