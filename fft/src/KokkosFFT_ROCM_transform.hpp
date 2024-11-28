// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_ROCM_TRANSFORM_HPP
#define KOKKOSFFT_ROCM_TRANSFORM_HPP

#include <complex>
#include <rocfft/rocfft.h>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_asserts.hpp"

namespace KokkosFFT {
namespace Impl {
inline void exec_plan(rocfft_plan& plan, float* idata,
                      std::complex<float>* odata, int /*direction*/,
                      const rocfft_execution_info& execution_info) {
  Kokkos::Profiling::ScopedRegion region("KokkosFFT::exec_plan[TPL_rocfft]");
  rocfft_status status =
      rocfft_execute(plan, (void**)&idata, (void**)&odata, execution_info);
  KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                     "rocfft_execute for R2C failed");
}

inline void exec_plan(rocfft_plan& plan, double* idata,
                      std::complex<double>* odata, int /*direction*/,
                      const rocfft_execution_info& execution_info) {
  Kokkos::Profiling::ScopedRegion region("KokkosFFT::exec_plan[TPL_rocfft]");
  rocfft_status status =
      rocfft_execute(plan, (void**)&idata, (void**)&odata, execution_info);
  KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                     "rocfft_execute for D2Z failed");
}

inline void exec_plan(rocfft_plan& plan, std::complex<float>* idata,
                      float* odata, int /*direction*/,
                      const rocfft_execution_info& execution_info) {
  Kokkos::Profiling::ScopedRegion region("KokkosFFT::exec_plan[TPL_rocfft]");
  rocfft_status status =
      rocfft_execute(plan, (void**)&idata, (void**)&odata, execution_info);
  KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                     "rocfft_execute for C2R failed");
}

inline void exec_plan(rocfft_plan& plan, std::complex<double>* idata,
                      double* odata, int /*direction*/,
                      const rocfft_execution_info& execution_info) {
  Kokkos::Profiling::ScopedRegion region("KokkosFFT::exec_plan[TPL_rocfft]");
  rocfft_status status =
      rocfft_execute(plan, (void**)&idata, (void**)&odata, execution_info);
  KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                     "rocfft_execute for Z2D failed");
}

inline void exec_plan(rocfft_plan& plan, std::complex<float>* idata,
                      std::complex<float>* odata, int /*direction*/,
                      const rocfft_execution_info& execution_info) {
  Kokkos::Profiling::ScopedRegion region("KokkosFFT::exec_plan[TPL_rocfft]");
  rocfft_status status =
      rocfft_execute(plan, (void**)&idata, (void**)&odata, execution_info);
  KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                     "rocfft_execute for C2C failed");
}

inline void exec_plan(rocfft_plan& plan, std::complex<double>* idata,
                      std::complex<double>* odata, int /*direction*/,
                      const rocfft_execution_info& execution_info) {
  Kokkos::Profiling::ScopedRegion region("KokkosFFT::exec_plan[TPL_rocfft]");
  rocfft_status status =
      rocfft_execute(plan, (void**)&idata, (void**)&odata, execution_info);
  KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                     "rocfft_execute for Z2Z failed");
}

}  // namespace Impl
}  // namespace KokkosFFT

#endif
