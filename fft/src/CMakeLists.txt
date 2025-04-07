# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_library(fft INTERFACE)

target_link_libraries(fft INTERFACE common Kokkos::kokkos)

# find and link libraries based on KOKKOSFFT_TPL_LIST
if("CUFFT" IN_LIST KOKKOSFFT_TPL_LIST)
  find_package(CUDAToolkit REQUIRED COMPONENTS cufft)
  target_link_libraries(fft INTERFACE CUDA::cufft)
  target_compile_definitions(fft INTERFACE KOKKOSFFT_ENABLE_TPL_CUFFT)
endif()

if("ROCFFT" IN_LIST KOKKOSFFT_TPL_LIST)
  find_package(rocfft REQUIRED)
  target_link_libraries(fft INTERFACE roc::rocfft)
  target_compile_definitions(fft INTERFACE KOKKOSFFT_ENABLE_TPL_ROCFFT)
endif()

if("HIPFFT" IN_LIST KOKKOSFFT_TPL_LIST)
  find_package(hipfft REQUIRED)
  target_link_libraries(fft INTERFACE hip::hipfft)
  target_compile_definitions(fft INTERFACE KOKKOSFFT_ENABLE_TPL_HIPFFT)
endif()

if("ONEMKL" IN_LIST KOKKOSFFT_TPL_LIST)
  find_package(MKL REQUIRED COMPONENTS SYCL)
  target_link_libraries(fft INTERFACE MKL::MKL_SYCL)
  target_compile_definitions(fft INTERFACE KOKKOSFFT_ENABLE_TPL_ONEMKL)
endif()

if("FFTW_OPENMP" IN_LIST KOKKOSFFT_TPL_LIST)
  find_package(FFTW MODULE REQUIRED)
  target_link_libraries(fft INTERFACE FFTW::Float FFTW::Double FFTW::FloatOpenMP FFTW::DoubleOpenMP)
  target_compile_definitions(fft INTERFACE KOKKOSFFT_ENABLE_TPL_FFTW)
elseif("FFTW_THREADS" IN_LIST KOKKOSFFT_TPL_LIST)
  find_package(FFTW MODULE REQUIRED)
  target_link_libraries(fft INTERFACE FFTW::Float FFTW::Double FFTW::FloatThreads FFTW::DoubleThreads)
  target_compile_definitions(fft INTERFACE KOKKOSFFT_ENABLE_TPL_FFTW)
elseif("FFTW_SERIAL" IN_LIST KOKKOSFFT_TPL_LIST)
  find_package(FFTW MODULE REQUIRED)
  target_link_libraries(fft INTERFACE FFTW::Float FFTW::Double)
  target_compile_definitions(fft INTERFACE KOKKOSFFT_ENABLE_TPL_FFTW)
endif()

if(KokkosFFT_ENABLE_HOST_AND_DEVICE)
  target_compile_definitions(fft INTERFACE ENABLE_HOST_AND_DEVICE)
endif()

target_include_directories(
  fft INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:${INSTALL_INCLUDEDIR}>
)
add_library(KokkosFFT::fft ALIAS fft)
