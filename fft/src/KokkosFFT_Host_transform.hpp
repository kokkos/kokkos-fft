// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_HOST_TRANSFORM_HPP
#define KOKKOSFFT_HOST_TRANSFORM_HPP

#include <fftw3.h>
#include <Kokkos_Profiling_ScopedRegion.hpp>

namespace KokkosFFT {
namespace Impl {

template <typename ScopedPlanType>
void exec_plan(const ScopedPlanType& scoped_plan, float* idata,
               fftwf_complex* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_fftwExecR2C]");
  fftwf_execute_dft_r2c(scoped_plan.plan(), idata, odata);
}

template <typename ScopedPlanType>
void exec_plan(const ScopedPlanType& scoped_plan, double* idata,
               fftw_complex* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_fftwExecD2Z]");
  fftw_execute_dft_r2c(scoped_plan.plan(), idata, odata);
}

template <typename ScopedPlanType>
void exec_plan(const ScopedPlanType& scoped_plan, fftwf_complex* idata,
               float* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_fftwExecC2R]");
  fftwf_execute_dft_c2r(scoped_plan.plan(), idata, odata);
}

template <typename ScopedPlanType>
void exec_plan(const ScopedPlanType& scoped_plan, fftw_complex* idata,
               double* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_fftwExecZ2D]");
  fftw_execute_dft_c2r(scoped_plan.plan(), idata, odata);
}

template <typename ScopedPlanType>
void exec_plan(const ScopedPlanType& scoped_plan, fftwf_complex* idata,
               fftwf_complex* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_fftwExecC2C]");
  fftwf_execute_dft(scoped_plan.plan(), idata, odata);
}

template <typename ScopedPlanType>
void exec_plan(const ScopedPlanType& scoped_plan, fftw_complex* idata,
               fftw_complex* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_fftwExecZ2Z]");
  fftw_execute_dft(scoped_plan.plan(), idata, odata);
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
