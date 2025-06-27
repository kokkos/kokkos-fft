# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_library(fft INTERFACE)

target_link_libraries(fft INTERFACE common Kokkos::kokkos)

# find and link libraries based on KOKKOSFFT_TPL_LIST
if("CUFFT" IN_LIST KOKKOSFFT_TPL_LIST)
  find_package(CUDAToolkit REQUIRED COMPONENTS cufft)
  target_link_libraries(fft INTERFACE CUDA::cufft)
endif()

if("ROCFFT" IN_LIST KOKKOSFFT_TPL_LIST)
  find_package(rocfft REQUIRED)
  target_link_libraries(fft INTERFACE roc::rocfft)
endif()

if("HIPFFT" IN_LIST KOKKOSFFT_TPL_LIST)
  find_package(hipfft REQUIRED)
  target_link_libraries(fft INTERFACE hip::hipfft)
endif()

if("ONEMKL" IN_LIST KOKKOSFFT_TPL_LIST)
  find_package(MKL REQUIRED COMPONENTS SYCL)
  target_link_libraries(fft INTERFACE MKL::MKL_SYCL)
endif()

if("FFTW_OPENMP" IN_LIST KOKKOSFFT_TPL_LIST)
  find_package(FFTW MODULE REQUIRED COMPONENTS FLOAT_LIB DOUBLE_LIB FLOAT_OPENMP_LIB DOUBLE_OPENMP_LIB)
  target_link_libraries(fft INTERFACE FFTW::Float FFTW::Double FFTW::FloatOpenMP FFTW::DoubleOpenMP)
elseif("FFTW_THREADS" IN_LIST KOKKOSFFT_TPL_LIST)
  find_package(FFTW MODULE REQUIRED COMPONENTS FLOAT_LIB DOUBLE_LIB FLOAT_THREADS_LIB DOUBLE_THREADS_LIB)
  target_link_libraries(fft INTERFACE FFTW::Float FFTW::Double FFTW::FloatThreads FFTW::DoubleThreads)
elseif("FFTW_SERIAL" IN_LIST KOKKOSFFT_TPL_LIST)
  find_package(FFTW MODULE REQUIRED COMPONENTS FLOAT_LIB DOUBLE_LIB)
  target_link_libraries(fft INTERFACE FFTW::Float FFTW::Double)
endif()

message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(
  fft INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:${INSTALL_INCLUDEDIR}>
)
add_library(KokkosFFT::fft ALIAS fft)
