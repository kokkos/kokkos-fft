// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_HIP_TRANSFORM_HPP
#define KOKKOSFFT_HIP_TRANSFORM_HPP

#include <hipfft/hipfft.h>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_asserts.hpp"
#include "KokkosFFT_HIP_types.hpp"

namespace KokkosFFT {
namespace Impl {

template <typename PlanType>
inline void exec_plan(const PlanType& scoped_plan, hipfftReal* idata,
                      hipfftComplex* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_hipfftExecR2C]");
  hipfftResult hipfft_rt = hipfftExecR2C(scoped_plan.plan(), idata, odata);
  KOKKOSFFT_THROW_IF(hipfft_rt != HIPFFT_SUCCESS, "hipfftExecR2C failed");
}

template <typename PlanType>
inline void exec_plan(const PlanType& scoped_plan, hipfftDoubleReal* idata,
                      hipfftDoubleComplex* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_hipfftExecD2Z]");
  hipfftResult hipfft_rt = hipfftExecD2Z(scoped_plan.plan(), idata, odata);
  KOKKOSFFT_THROW_IF(hipfft_rt != HIPFFT_SUCCESS, "hipfftExecD2Z failed");
}

template <typename PlanType>
inline void exec_plan(const PlanType& scoped_plan, hipfftComplex* idata,
                      hipfftReal* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_hipfftExecC2R]");
  hipfftResult hipfft_rt = hipfftExecC2R(scoped_plan.plan(), idata, odata);
  KOKKOSFFT_THROW_IF(hipfft_rt != HIPFFT_SUCCESS, "hipfftExecC2R failed");
}

template <typename PlanType>
inline void exec_plan(const PlanType& scoped_plan, hipfftDoubleComplex* idata,
                      hipfftDoubleReal* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_hipfftExecZ2D]");
  hipfftResult hipfft_rt = hipfftExecZ2D(scoped_plan.plan(), idata, odata);
  KOKKOSFFT_THROW_IF(hipfft_rt != HIPFFT_SUCCESS, "hipfftExecZ2D failed");
}

template <typename PlanType>
inline void exec_plan(const PlanType& scoped_plan, hipfftComplex* idata,
                      hipfftComplex* odata, int direction) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_hipfftExecC2C]");
  hipfftResult hipfft_rt =
      hipfftExecC2C(scoped_plan.plan(), idata, odata, direction);
  KOKKOSFFT_THROW_IF(hipfft_rt != HIPFFT_SUCCESS, "hipfftExecC2C failed");
}

template <typename PlanType>
inline void exec_plan(const PlanType& scoped_plan, hipfftDoubleComplex* idata,
                      hipfftDoubleComplex* odata, int direction) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_hipfftExecZ2Z]");
  hipfftResult hipfft_rt =
      hipfftExecZ2Z(scoped_plan.plan(), idata, odata, direction);
  KOKKOSFFT_THROW_IF(hipfft_rt != HIPFFT_SUCCESS, "hipfftExecZ2Z failed");
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
