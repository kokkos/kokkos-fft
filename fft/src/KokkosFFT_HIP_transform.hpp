// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_HIP_TRANSFORM_HPP
#define KOKKOSFFT_HIP_TRANSFORM_HPP

#include <hipfft/hipfft.h>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_asserts.hpp"
#include "KokkosFFT_HIP_types.hpp"
#include "KokkosFFT_HIP_asserts.hpp"

namespace KokkosFFT {
namespace Impl {

template <typename PlanType>
inline void exec_plan(const PlanType& scoped_plan, hipfftReal* idata,
                      hipfftComplex* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_hipfftExecR2C]");
  KOKKOSFFT_CHECK_HIPFFT_CALL(hipfftExecR2C(scoped_plan.plan(), idata, odata));
}

template <typename PlanType>
inline void exec_plan(const PlanType& scoped_plan, hipfftDoubleReal* idata,
                      hipfftDoubleComplex* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_hipfftExecD2Z]");
  KOKKOSFFT_CHECK_HIPFFT_CALL(hipfftExecD2Z(scoped_plan.plan(), idata, odata));
}

template <typename PlanType>
inline void exec_plan(const PlanType& scoped_plan, hipfftComplex* idata,
                      hipfftReal* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_hipfftExecC2R]");
  KOKKOSFFT_CHECK_HIPFFT_CALL(hipfftExecC2R(scoped_plan.plan(), idata, odata));
}

template <typename PlanType>
inline void exec_plan(const PlanType& scoped_plan, hipfftDoubleComplex* idata,
                      hipfftDoubleReal* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_hipfftExecZ2D]");
  KOKKOSFFT_CHECK_HIPFFT_CALL(hipfftExecZ2D(scoped_plan.plan(), idata, odata));
}

template <typename PlanType>
inline void exec_plan(const PlanType& scoped_plan, hipfftComplex* idata,
                      hipfftComplex* odata, int direction) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_hipfftExecC2C]");
  KOKKOSFFT_CHECK_HIPFFT_CALL(
      hipfftExecC2C(scoped_plan.plan(), idata, odata, direction));
}

template <typename PlanType>
inline void exec_plan(const PlanType& scoped_plan, hipfftDoubleComplex* idata,
                      hipfftDoubleComplex* odata, int direction) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_hipfftExecZ2Z]");
  KOKKOSFFT_CHECK_HIPFFT_CALL(
      hipfftExecZ2Z(scoped_plan.plan(), idata, odata, direction));
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
