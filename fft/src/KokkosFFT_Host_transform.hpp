// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_HOST_TRANSFORM_HPP
#define KOKKOSFFT_HOST_TRANSFORM_HPP

#include <fftw3.h>

namespace KokkosFFT {
namespace Impl {

template <typename ScopedPlanType>
void exec_plan(const ScopedPlanType& scoped_plan, float* idata,
               fftwf_complex* odata, int /*direction*/) {
  fftwf_execute_dft_r2c(scoped_plan.plan(), idata, odata);
}

template <typename ScopedPlanType>
void exec_plan(const ScopedPlanType& scoped_plan, double* idata,
               fftw_complex* odata, int /*direction*/) {
  fftw_execute_dft_r2c(scoped_plan.plan(), idata, odata);
}

template <typename ScopedPlanType>
void exec_plan(const ScopedPlanType& scoped_plan, fftwf_complex* idata,
               float* odata, int /*direction*/) {
  fftwf_execute_dft_c2r(scoped_plan.plan(), idata, odata);
}

template <typename ScopedPlanType>
void exec_plan(const ScopedPlanType& scoped_plan, fftw_complex* idata,
               double* odata, int /*direction*/) {
  fftw_execute_dft_c2r(scoped_plan.plan(), idata, odata);
}

template <typename ScopedPlanType>
void exec_plan(const ScopedPlanType& scoped_plan, fftwf_complex* idata,
               fftwf_complex* odata, int /*direction*/) {
  fftwf_execute_dft(scoped_plan.plan(), idata, odata);
}

template <typename ScopedPlanType>
void exec_plan(const ScopedPlanType& scoped_plan, fftw_complex* idata,
               fftw_complex* odata, int /*direction*/) {
  fftw_execute_dft(scoped_plan.plan(), idata, odata);
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
