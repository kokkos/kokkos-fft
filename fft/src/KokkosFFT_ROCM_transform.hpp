// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_ROCM_TRANSFORM_HPP
#define KOKKOSFFT_ROCM_TRANSFORM_HPP

#include <complex>
#include <rocfft/rocfft.h>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_asserts.hpp"
#include "KokkosFFT_ROCM_types.hpp"

namespace KokkosFFT {
namespace Impl {
inline void exec_plan(const ScopedRocfftPlan<float>& scoped_plan, float* idata,
                      std::complex<float>* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_rocfftExecR2C]");
  rocfft_status status =
      rocfft_execute(scoped_plan.plan(), (void**)&idata, (void**)&odata,
                     scoped_plan.execution_info());
  KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                     "rocfft_execute for R2C failed");
}

inline void exec_plan(const ScopedRocfftPlan<double>& scoped_plan,
                      double* idata, std::complex<double>* odata,
                      int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_rocfftExecD2Z]");
  rocfft_status status =
      rocfft_execute(scoped_plan.plan(), (void**)&idata, (void**)&odata,
                     scoped_plan.execution_info());
  KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                     "rocfft_execute for D2Z failed");
}

inline void exec_plan(
    const ScopedRocfftPlan<Kokkos::complex<float>>& scoped_plan,
    std::complex<float>* idata, float* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_rocfftExecC2R]");
  rocfft_status status =
      rocfft_execute(scoped_plan.plan(), (void**)&idata, (void**)&odata,
                     scoped_plan.execution_info());
  KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                     "rocfft_execute for C2R failed");
}

inline void exec_plan(
    const ScopedRocfftPlan<Kokkos::complex<double>>& scoped_plan,
    std::complex<double>* idata, double* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_rocfftExecZ2D]");
  rocfft_status status =
      rocfft_execute(scoped_plan.plan(), (void**)&idata, (void**)&odata,
                     scoped_plan.execution_info());
  KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                     "rocfft_execute for Z2D failed");
}

inline void exec_plan(
    const ScopedRocfftPlan<Kokkos::complex<float>>& scoped_plan,
    std::complex<float>* idata, std::complex<float>* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_rocfftExecC2C]");
  rocfft_status status =
      rocfft_execute(scoped_plan.plan(), (void**)&idata, (void**)&odata,
                     scoped_plan.execution_info());
  KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                     "rocfft_execute for C2C failed");
}

inline void exec_plan(
    const ScopedRocfftPlan<Kokkos::complex<double>>& scoped_plan,
    std::complex<double>* idata, std::complex<double>* odata,
    int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_rocfftExecZ2Z]");
  rocfft_status status =
      rocfft_execute(scoped_plan.plan(), (void**)&idata, (void**)&odata,
                     scoped_plan.execution_info());
  KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                     "rocfft_execute for Z2Z failed");
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
