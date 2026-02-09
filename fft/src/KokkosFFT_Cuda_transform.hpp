// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_CUDA_TRANSFORM_HPP
#define KOKKOSFFT_CUDA_TRANSFORM_HPP

#include <cufft.h>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_asserts.hpp"
#include "KokkosFFT_Cuda_types.hpp"
#include "KokkosFFT_Cuda_asserts.hpp"

namespace KokkosFFT {
namespace Impl {

template <typename PlanType>
inline void exec_plan(const PlanType& scoped_plan, cufftReal* idata,
                      cufftComplex* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_cufftExecR2C]");
  KOKKOSFFT_CHECK_CUFFT_CALL(cufftExecR2C(scoped_plan.plan(), idata, odata));
}

template <typename PlanType>
inline void exec_plan(const PlanType& scoped_plan, cufftDoubleReal* idata,
                      cufftDoubleComplex* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_cufftExecD2Z]");
  KOKKOSFFT_CHECK_CUFFT_CALL(cufftExecD2Z(scoped_plan.plan(), idata, odata));
}

template <typename PlanType>
inline void exec_plan(const PlanType& scoped_plan, cufftComplex* idata,
                      cufftReal* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_cufftExecC2R]");
  KOKKOSFFT_CHECK_CUFFT_CALL(cufftExecC2R(scoped_plan.plan(), idata, odata));
}

template <typename PlanType>
inline void exec_plan(const PlanType& scoped_plan, cufftDoubleComplex* idata,
                      cufftDoubleReal* odata, int /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_cufftExecZ2D]");
  KOKKOSFFT_CHECK_CUFFT_CALL(cufftExecZ2D(scoped_plan.plan(), idata, odata));
}

template <typename PlanType>
inline void exec_plan(const PlanType& scoped_plan, cufftComplex* idata,
                      cufftComplex* odata, int direction) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_cufftExecC2C]");
  KOKKOSFFT_CHECK_CUFFT_CALL(
      cufftExecC2C(scoped_plan.plan(), idata, odata, direction));
}

template <typename PlanType>
inline void exec_plan(const PlanType& scoped_plan, cufftDoubleComplex* idata,
                      cufftDoubleComplex* odata, int direction) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_cufftExecZ2Z]");
  KOKKOSFFT_CHECK_CUFFT_CALL(
      cufftExecZ2Z(scoped_plan.plan(), idata, odata, direction));
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
