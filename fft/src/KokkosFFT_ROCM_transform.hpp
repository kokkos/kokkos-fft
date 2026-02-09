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
#include "KokkosFFT_ROCM_asserts.hpp"

namespace KokkosFFT {
namespace Impl {
inline void exec_plan(const ScopedRocfftPlan<float>& scoped_plan, float* idata,
                      std::complex<float>* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_rocfftExecR2C]");
  KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_execute(scoped_plan.plan(), (void**)&idata,
                                             (void**)&odata,
                                             scoped_plan.execution_info()));
}

inline void exec_plan(const ScopedRocfftPlan<double>& scoped_plan,
                      double* idata, std::complex<double>* odata,
                      int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_rocfftExecD2Z]");
  KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_execute(scoped_plan.plan(), (void**)&idata,
                                             (void**)&odata,
                                             scoped_plan.execution_info()));
}

inline void exec_plan(
    const ScopedRocfftPlan<Kokkos::complex<float>>& scoped_plan,
    std::complex<float>* idata, float* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_rocfftExecC2R]");
  KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_execute(scoped_plan.plan(), (void**)&idata,
                                             (void**)&odata,
                                             scoped_plan.execution_info()));
}

inline void exec_plan(
    const ScopedRocfftPlan<Kokkos::complex<double>>& scoped_plan,
    std::complex<double>* idata, double* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_rocfftExecZ2D]");
  KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_execute(scoped_plan.plan(), (void**)&idata,
                                             (void**)&odata,
                                             scoped_plan.execution_info()));
}

inline void exec_plan(
    const ScopedRocfftPlan<Kokkos::complex<float>>& scoped_plan,
    std::complex<float>* idata, std::complex<float>* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_rocfftExecC2C]");
  KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_execute(scoped_plan.plan(), (void**)&idata,
                                             (void**)&odata,
                                             scoped_plan.execution_info()));
}

inline void exec_plan(
    const ScopedRocfftPlan<Kokkos::complex<double>>& scoped_plan,
    std::complex<double>* idata, std::complex<double>* odata,
    int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_rocfftExecZ2Z]");
  KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_execute(scoped_plan.plan(), (void**)&idata,
                                             (void**)&odata,
                                             scoped_plan.execution_info()));
}

inline void exec_plan(
    const ScopedRocfftPlan<Kokkos::complex<double>>& scoped_plan,
    std::complex<double>* idata, std::complex<double>* odata,
    int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_rocfftExecZ2Z]");
  KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_execute(scoped_plan.plan(), (void**)&idata,
                                             (void**)&odata,
                                             scoped_plan.execution_info()));
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
