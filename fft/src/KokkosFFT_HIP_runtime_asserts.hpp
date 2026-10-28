// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_HIP_RUNTIME_ASSERTS_HPP
#define KOKKOSFFT_HIP_RUNTIME_ASSERTS_HPP

// HIP runtime error checking shared by the hipFFT and the native rocFFT
// backends (e.g. for the device allocations owned by callback-enabled plans).

#include <hip/hip_runtime_api.h>
#include "KokkosFFT_Asserts.hpp"

#if defined(__cpp_lib_source_location) && __cpp_lib_source_location >= 201907L
#include <source_location>
#define KOKKOSFFT_CHECK_HIP_CALL(call)                 \
  KokkosFFT::Impl::check_fft_call(                     \
      call, #call, hipSuccess, hipGetErrorString,      \
      std::source_location::current().file_name(),     \
      std::source_location::current().line(),          \
      std::source_location::current().function_name(), \
      std::source_location::current().column())
#else
#define KOKKOSFFT_CHECK_HIP_CALL(call)                                        \
  KokkosFFT::Impl::check_fft_call(call, #call, hipSuccess, hipGetErrorString, \
                                  __FILE__, __LINE__, __FUNCTION__)
#endif

#endif
