// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_HIP_TRANSFORM_HPP
#define KOKKOSFFT_HIP_TRANSFORM_HPP

#include <hipfft/hipfft.h>
#include "KokkosFFT_asserts.hpp"

namespace KokkosFFT {
namespace Impl {
template <typename ScopedPlanType>
inline void exec_plan(ScopedPlanType& scoped_plan, hipfftReal* idata,
                      hipfftComplex* odata, int /*direction*/) {
  hipfftResult hipfft_rt = hipfftExecR2C(scoped_plan.plan(), idata, odata);
  KOKKOSFFT_THROW_IF(hipfft_rt != HIPFFT_SUCCESS, "hipfftExecR2C failed");
}

template <typename ScopedPlanType>
inline void exec_plan(ScopedPlanType& scoped_plan, hipfftDoubleReal* idata,
                      hipfftDoubleComplex* odata, int /*direction*/) {
  hipfftResult hipfft_rt = hipfftExecD2Z(scoped_plan.plan(), idata, odata);
  KOKKOSFFT_THROW_IF(hipfft_rt != HIPFFT_SUCCESS, "hipfftExecD2Z failed");
}

template <typename ScopedPlanType>
inline void exec_plan(ScopedPlanType& scoped_plan, hipfftComplex* idata,
                      hipfftReal* odata, int /*direction*/) {
  hipfftResult hipfft_rt = hipfftExecC2R(scoped_plan.plan(), idata, odata);
  KOKKOSFFT_THROW_IF(hipfft_rt != HIPFFT_SUCCESS, "hipfftExecC2R failed");
}

template <typename ScopedPlanType>
inline void exec_plan(ScopedPlanType& scoped_plan, hipfftDoubleComplex* idata,
                      hipfftDoubleReal* odata, int /*direction*/) {
  hipfftResult hipfft_rt = hipfftExecZ2D(scoped_plan.plan(), idata, odata);
  KOKKOSFFT_THROW_IF(hipfft_rt != HIPFFT_SUCCESS, "hipfftExecZ2D failed");
}

template <typename ScopedPlanType>
inline void exec_plan(ScopedPlanType& scoped_plan, hipfftComplex* idata,
                      hipfftComplex* odata, int direction) {
  hipfftResult hipfft_rt =
      hipfftExecC2C(scoped_plan.plan(), idata, odata, direction);
  KOKKOSFFT_THROW_IF(hipfft_rt != HIPFFT_SUCCESS, "hipfftExecC2C failed");
}

template <typename ScopedPlanType>
inline void exec_plan(ScopedPlanType& scoped_plan, hipfftDoubleComplex* idata,
                      hipfftDoubleComplex* odata, int direction) {
  hipfftResult hipfft_rt =
      hipfftExecZ2Z(scoped_plan.plan(), idata, odata, direction);
  KOKKOSFFT_THROW_IF(hipfft_rt != HIPFFT_SUCCESS, "hipfftExecZ2Z failed");
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
