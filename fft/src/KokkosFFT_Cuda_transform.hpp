// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_CUDA_TRANSFORM_HPP
#define KOKKOSFFT_CUDA_TRANSFORM_HPP

#include <cufft.h>
#include "KokkosFFT_Cuda_types.hpp"

namespace KokkosFFT {
namespace Impl {

inline void exec_plan(const ScopedCufftPlan& scoped_plan, cufftReal* idata,
                      cufftComplex* odata, int /*direction*/) {
  cufftResult cufft_rt = cufftExecR2C(scoped_plan.plan(), idata, odata);
  KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftExecR2C failed");
}

inline void exec_plan(const ScopedCufftPlan& scoped_plan,
                      cufftDoubleReal* idata, cufftDoubleComplex* odata,
                      int /*direction*/) {
  cufftResult cufft_rt = cufftExecD2Z(scoped_plan.plan(), idata, odata);
  KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftExecD2Z failed");
}

inline void exec_plan(const ScopedCufftPlan& scoped_plan, cufftComplex* idata,
                      cufftReal* odata, int /*direction*/) {
  cufftResult cufft_rt = cufftExecC2R(scoped_plan.plan(), idata, odata);
  KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftExecC2R failed");
}

inline void exec_plan(const ScopedCufftPlan& scoped_plan,
                      cufftDoubleComplex* idata, cufftDoubleReal* odata,
                      int /*direction*/) {
  cufftResult cufft_rt = cufftExecZ2D(scoped_plan.plan(), idata, odata);
  KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftExecZ2D failed");
}

inline void exec_plan(const ScopedCufftPlan& scoped_plan, cufftComplex* idata,
                      cufftComplex* odata, int direction) {
  cufftResult cufft_rt =
      cufftExecC2C(scoped_plan.plan(), idata, odata, direction);
  KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftExecC2C failed");
}

inline void exec_plan(const ScopedCufftPlan& scoped_plan,
                      cufftDoubleComplex* idata, cufftDoubleComplex* odata,
                      int direction) {
  cufftResult cufft_rt =
      cufftExecZ2Z(scoped_plan.plan(), idata, odata, direction);
  KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftExecZ2Z failed");
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
