// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_HOST_TRANSFORM_HPP
#define KOKKOSFFT_HOST_TRANSFORM_HPP

#include <fftw3.h>

namespace KokkosFFT {
namespace Impl {
template <typename PlanType, typename... Args>
void _exec(PlanType& plan, float* idata, fftwf_complex* odata,
           int /*direction*/, Args...) {
  fftwf_execute_dft_r2c(plan, idata, odata);
}

template <typename PlanType, typename... Args>
void _exec(PlanType& plan, double* idata, fftw_complex* odata,
           int /*direction*/, Args...) {
  fftw_execute_dft_r2c(plan, idata, odata);
}

template <typename PlanType, typename... Args>
void _exec(PlanType& plan, fftwf_complex* idata, float* odata,
           int /*direction*/, Args...) {
  fftwf_execute_dft_c2r(plan, idata, odata);
}

template <typename PlanType, typename... Args>
void _exec(PlanType& plan, fftw_complex* idata, double* odata,
           int /*direction*/, Args...) {
  fftw_execute_dft_c2r(plan, idata, odata);
}

template <typename PlanType, typename... Args>
void _exec(PlanType& plan, fftwf_complex* idata, fftwf_complex* odata,
           int /*direction*/, Args...) {
  fftwf_execute_dft(plan, idata, odata);
}

template <typename PlanType, typename... Args>
void _exec(PlanType plan, fftw_complex* idata, fftw_complex* odata,
           int /*direction*/, Args...) {
  fftw_execute_dft(plan, idata, odata);
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif