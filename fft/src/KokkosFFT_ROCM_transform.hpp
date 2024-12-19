// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_ROCM_TRANSFORM_HPP
#define KOKKOSFFT_ROCM_TRANSFORM_HPP

#include <complex>
#include <rocfft/rocfft.h>
#include "KokkosFFT_ROCM_types.hpp"

namespace KokkosFFT {
namespace Impl {
void exec_plan(ScopedRocfftPlan<float>& scoped_plan, float* idata,
               std::complex<float>* odata, int /*direction*/) {
  rocfft_status status =
      rocfft_execute(scoped_plan.plan(), (void**)&idata, (void**)&odata,
                     scoped_plan.execution_info());
  KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                     "rocfft_execute for R2C failed");
}

void exec_plan(ScopedRocfftPlan<double>& scoped_plan, double* idata,
               std::complex<double>* odata, int /*direction*/) {
  rocfft_status status =
      rocfft_execute(scoped_plan.plan(), (void**)&idata, (void**)&odata,
                     scoped_plan.execution_info());
  KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                     "rocfft_execute for D2Z failed");
}

void exec_plan(ScopedRocfftPlan<float>& scoped_plan, std::complex<float>* idata,
               float* odata, int /*direction*/) {
  rocfft_status status =
      rocfft_execute(scoped_plan.plan(), (void**)&idata, (void**)&odata,
                     scoped_plan.execution_info());
  KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                     "rocfft_execute for C2R failed");
}

void exec_plan(ScopedRocfftPlan<double>& scoped_plan,
               std::complex<double>* idata, double* odata, int /*direction*/) {
  rocfft_status status =
      rocfft_execute(scoped_plan.plan(), (void**)&idata, (void**)&odata,
                     scoped_plan.execution_info());
  KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                     "rocfft_execute for Z2D failed");
}

void exec_plan(ScopedRocfftPlan<float>& scoped_plan, std::complex<float>* idata,
               std::complex<float>* odata, int /*direction*/) {
  rocfft_status status =
      rocfft_execute(scoped_plan.plan(), (void**)&idata, (void**)&odata,
                     scoped_plan.execution_info());
  KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                     "rocfft_execute for C2C failed");
}

void exec_plan(ScopedRocfftPlan<double>& scoped_plan,
               std::complex<double>* idata, std::complex<double>* odata,
               int /*direction*/) {
  rocfft_status status =
      rocfft_execute(scoped_plan.plan(), (void**)&idata, (void**)&odata,
                     scoped_plan.execution_info());
  KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                     "rocfft_execute for Z2Z failed");
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
