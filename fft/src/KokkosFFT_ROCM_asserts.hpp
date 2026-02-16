// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_ROCM_ASSERTS_HPP
#define KOKKOSFFT_ROCM_ASSERTS_HPP

#include <stdexcept>
#include <string_view>
#include <rocfft/rocfft.h>
#include "KokkosFFT_asserts.hpp"

#if defined(__cpp_lib_source_location) && __cpp_lib_source_location >= 201907L
#include <source_location>
#define KOKKOSFFT_CHECK_ROCFFT_CALL(call)              \
  KokkosFFT::Impl::check_fft_call(                     \
      call, #call, rocfft_status_success,              \
      KokkosFFT::Impl::rocfft_result_to_string,        \
      std::source_location::current().file_name(),     \
      std::source_location::current().line(),          \
      std::source_location::current().function_name(), \
      std::source_location::current().column())
#else
#define KOKKOSFFT_CHECK_ROCFFT_CALL(call)                                   \
  KokkosFFT::Impl::check_fft_call(call, #call, rocfft_status_success,       \
                                  KokkosFFT::Impl::rocfft_result_to_string, \
                                  __FILE__, __LINE__, __FUNCTION__)
#endif

namespace KokkosFFT {
namespace Impl {

inline std::string_view rocfft_result_to_string(rocfft_status result) {
  switch (result) {
    case rocfft_status_success: return "rocfft_status_success";
    case rocfft_status_failure: return "rocfft_status_failure";
    case rocfft_status_invalid_arg_value:
      return "rocfft_status_invalid_arg_value";
    case rocfft_status_invalid_dimensions:
      return "rocfft_status_invalid_dimensions";
    case rocfft_status_invalid_array_type:
      return "rocfft_status_invalid_array_type";
    case rocfft_status_invalid_strides: return "rocfft_status_invalid_strides";
    case rocfft_status_invalid_distance:
      return "rocfft_status_invalid_distance";
    case rocfft_status_invalid_offset: return "rocfft_status_invalid_offset";
    case rocfft_status_invalid_work_buffer:
      return "rocfft_status_invalid_work_buffer";
    default: return "UNKNOWN_ROCFFT_ERROR";
  }
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
