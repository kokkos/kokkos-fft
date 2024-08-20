# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

add_library(common INTERFACE)

target_link_libraries(common
    INTERFACE
    Kokkos::kokkos
)

# Check which tpls are needed
list(FIND tpls_list "CUFFT" idx_CUFFT)
list(FIND tpls_list "ROCFFT" idx_ROCFFT)
list(FIND tpls_list "HIPFFT" idx_HIPFFT)
list(FIND tpls_list "ONEMKL" idx_ONEMKL)
list(FIND tpls_list "FFTW_OPENMP" idx_FFTW_OPENMP)
list(FIND tpls_list "FFTW_THREADS" idx_FFTW_THREADS)
list(FIND tpls_list "FFTW_SERIAL" idx_FFTW_SERIAL)

if(idx_CUFFT GREATER -1)
    find_package(CUDAToolkit REQUIRED COMPONENTS cufft)
    target_link_libraries(common INTERFACE CUDA::cufft)
    target_compile_definitions(common INTERFACE KOKKOSFFT_ENABLE_TPL_CUFFT)
endif()

if(idx_ROCFFT GREATER -1)
    find_package(rocfft REQUIRED)
    target_link_libraries(common INTERFACE roc::rocfft)
    target_compile_definitions(common INTERFACE KOKKOSFFT_ENABLE_TPL_ROCFFT)
endif()

if(idx_HIPFFT GREATER -1)
    find_package(hipfft REQUIRED)
    target_link_libraries(common INTERFACE hip::hipfft)
    target_compile_definitions(common INTERFACE KOKKOSFFT_ENABLE_TPL_HIPFFT)
endif()

if(idx_ONEMKL GREATER -1)
    find_package(MKL REQUIRED COMPONENTS SYCL)
    target_link_libraries(common INTERFACE MKL::MKL_SYCL)
    target_compile_definitions(common INTERFACE KOKKOSFFT_ENABLE_TPL_ONEMKL)
endif()

if(idx_FFTW_OPENMP GREATER -1)
    find_package(FFTW MODULE REQUIRED)
    target_link_libraries(common INTERFACE FFTW::Float FFTW::Double FFTW::FloatOpenMP FFTW::DoubleOpenMP)
    target_compile_definitions(common INTERFACE KOKKOSFFT_ENABLE_TPL_FFTW)
elseif(idx_FFTW_THREADS GREATER -1)
    find_package(FFTW MODULE REQUIRED)
    target_link_libraries(common INTERFACE FFTW::Float FFTW::Double FFTW::FloatThreads FFTW::DoubleThreads)
    target_compile_definitions(common INTERFACE KOKKOSFFT_ENABLE_TPL_FFTW)
elseif(idx_FFTW_SERIAL GREATER -1)
    find_package(FFTW MODULE REQUIRED)
    target_link_libraries(common INTERFACE FFTW::Float FFTW::Double)
    target_compile_definitions(common INTERFACE KOKKOSFFT_ENABLE_TPL_FFTW)
endif()

if(KokkosFFT_ENABLE_HOST_AND_DEVICE)
    target_compile_definitions(common INTERFACE ENABLE_HOST_AND_DEVICE)
endif()

target_compile_features(common INTERFACE cxx_std_17)
target_include_directories(common INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           $<INSTALL_INTERFACE:${INSTALL_INCLUDEDIR}>
)

