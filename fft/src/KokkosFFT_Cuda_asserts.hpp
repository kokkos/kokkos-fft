// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_CUDA_ASSERTS_HPP
#define KOKKOSFFT_CUDA_ASSERTS_HPP

#include <stdexcept>
#include <string_view>
#include <cufft.h>
#include "KokkosFFT_Asserts.hpp"

#if defined(__cpp_lib_source_location) && __cpp_lib_source_location >= 201907L
#include <source_location>
#define KOKKOSFFT_CHECK_CUFFT_CALL(call)                                   \
  KokkosFFT::Impl::check_fft_call(                                         \
      call, #call, CUFFT_SUCCESS, KokkosFFT::Impl::cufft_result_to_string, \
      std::source_location::current().file_name(),                         \
      std::source_location::current().line(),                              \
      std::source_location::current().function_name(),                     \
      std::source_location::current().column())

#define KOKKOSFFT_CHECK_CUDA_CALL(call)                                 \
  KokkosFFT::Impl::check_fft_call(                                      \
      call, #call, cudaSuccess, KokkosFFT::Impl::cuda_result_to_string, \
      std::source_location::current().file_name(),                      \
      std::source_location::current().line(),                           \
      std::source_location::current().function_name(),                  \
      std::source_location::current().column())
#else
#define KOKKOSFFT_CHECK_CUFFT_CALL(call)                                   \
  KokkosFFT::Impl::check_fft_call(call, #call, CUFFT_SUCCESS,              \
                                  KokkosFFT::Impl::cufft_result_to_string, \
                                  __FILE__, __LINE__, __FUNCTION__)
#define KOKKOSFFT_CHECK_CUDA_CALL(call)                                   \
  KokkosFFT::Impl::check_fft_call(call, #call, cudaSuccess,               \
                                  KokkosFFT::Impl::cuda_result_to_string, \
                                  __FILE__, __LINE__, __FUNCTION__)
#endif

namespace KokkosFFT {
namespace Impl {

inline std::string_view cufft_result_to_string(cufftResult result) {
  switch (result) {
    case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
    case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
    // case CUFFT_MISSING_DEPENDENCY: return "CUFFT_MISSING_DEPENDENCY";
    // case CUFFT_NVRTC_FAILURE:      return "CUFFT_NVRTC_FAILURE";
    // case CUFFT_NVJITLINK_FAILURE:  return "CUFFT_NVJITLINK_FAILURE";
    // case CUFFT_NVSHMEM_FAILURE:    return "CUFFT_NVSHMEM_FAILURE";
    default: return "UNKNOWN_CUFFT_ERROR";
  }
}

inline std::string_view cuda_result_to_string(cudaError_t result) {
  switch (result) {
    case cudaSuccess: return "cudaSuccess";
    case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
    case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
    case cudaErrorInitializationError: return "cudaErrorInitializationError";
    case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
    case cudaErrorLaunchTimeout: return "cudaErrorLaunchTimeout";
    case cudaErrorLaunchOutOfResources: return "cudaErrorLaunchOutOfResources";
    case cudaErrorInvalidDeviceFunction:
      return "cudaErrorInvalidDeviceFunction";
    case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration";
    case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
    case cudaErrorInvalidMemcpyDirection:
      return "cudaErrorInvalidMemcpyDirection";
    case cudaErrorUnknown: return "cudaErrorUnknown";
    default: return "UNKNOWN_CUDA_ERROR";
  }
}

}  // namespace Impl
}  // namespace KokkosFFT

#endif
