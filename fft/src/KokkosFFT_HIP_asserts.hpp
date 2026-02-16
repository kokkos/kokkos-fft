// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_HIP_ASSERTS_HPP
#define KOKKOSFFT_HIP_ASSERTS_HPP

#include <stdexcept>
#include <string_view>
#include <hipfft/hipfft.h>
#include "KokkosFFT_asserts.hpp"

#if defined(__cpp_lib_source_location) && __cpp_lib_source_location >= 201907L
#include <source_location>
#define KOKKOSFFT_CHECK_HIPFFT_CALL(call)                                    \
  KokkosFFT::Impl::check_fft_call(                                           \
      call, #call, HIPFFT_SUCCESS, KokkosFFT::Impl::hipfft_result_to_string, \
      std::source_location::current().file_name(),                           \
      std::source_location::current().line(),                                \
      std::source_location::current().function_name(),                       \
      std::source_location::current().column())
#else
#define KOKKOSFFT_CHECK_HIPFFT_CALL(call)                                   \
  KokkosFFT::Impl::check_fft_call(call, #call, HIPFFT_SUCCESS,              \
                                  KokkosFFT::Impl::hipfft_result_to_string, \
                                  __FILE__, __LINE__, __FUNCTION__)
#endif

namespace KokkosFFT {
namespace Impl {

inline std::string_view hipfft_result_to_string(hipfftResult result) {
  switch (result) {
    case HIPFFT_SUCCESS: return "HIPFFT_SUCCESS";
    case HIPFFT_INVALID_PLAN: return "HIPFFT_INVALID_PLAN";
    case HIPFFT_ALLOC_FAILED: return "HIPFFT_ALLOC_FAILED";
    case HIPFFT_INVALID_TYPE: return "HIPFFT_INVALID_TYPE";
    case HIPFFT_INVALID_VALUE: return "HIPFFT_INVALID_VALUE";
    case HIPFFT_INTERNAL_ERROR: return "HIPFFT_INTERNAL_ERROR";
    case HIPFFT_EXEC_FAILED: return "HIPFFT_EXEC_FAILED";
    case HIPFFT_SETUP_FAILED: return "HIPFFT_SETUP_FAILED";
    case HIPFFT_INVALID_SIZE: return "HIPFFT_INVALID_SIZE";
    case HIPFFT_UNALIGNED_DATA: return "HIPFFT_UNALIGNED_DATA";
    case HIPFFT_INCOMPLETE_PARAMETER_LIST:
      return "HIPFFT_INCOMPLETE_PARAMETER_LIST";
    case HIPFFT_INVALID_DEVICE: return "HIPFFT_INVALID_DEVICE";
    case HIPFFT_PARSE_ERROR: return "HIPFFT_PARSE_ERROR";
    case HIPFFT_NO_WORKSPACE: return "HIPFFT_NO_WORKSPACE";
    case HIPFFT_NOT_IMPLEMENTED: return "HIPFFT_NOT_IMPLEMENTED";
    case HIPFFT_NOT_SUPPORTED: return "HIPFFT_NOT_SUPPORTED";
    default: return "UNKNOWN_HIPFFT_ERROR";
  }
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
