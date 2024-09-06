// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_CUDA_TRANSFORM_HPP
#define KOKKOSFFT_CUDA_TRANSFORM_HPP

#include <cufft.h>
#include "KokkosFFT_asserts.hpp"

namespace KokkosFFT {
namespace Impl {
template <typename... Args>
inline void exec_plan(cufftHandle& plan, cufftReal* idata, cufftComplex* odata,
                      int /*direction*/, Args...) {
  cufftResult cufft_rt = cufftExecR2C(plan, idata, odata);
  KOKKOSFFT_EXPECTS(cufft_rt == CUFFT_SUCCESS, "cufftExecR2C failed");
}

template <typename... Args>
inline void exec_plan(cufftHandle& plan, cufftDoubleReal* idata,
                      cufftDoubleComplex* odata, int /*direction*/, Args...) {
  cufftResult cufft_rt = cufftExecD2Z(plan, idata, odata);
  KOKKOSFFT_EXPECTS(cufft_rt == CUFFT_SUCCESS, "cufftExecD2Z failed");
}

template <typename... Args>
inline void exec_plan(cufftHandle& plan, cufftComplex* idata, cufftReal* odata,
                      int /*direction*/, Args...) {
  cufftResult cufft_rt = cufftExecC2R(plan, idata, odata);
  KOKKOSFFT_EXPECTS(cufft_rt == CUFFT_SUCCESS, "cufftExecC2R failed");
}

template <typename... Args>
inline void exec_plan(cufftHandle& plan, cufftDoubleComplex* idata,
                      cufftDoubleReal* odata, int /*direction*/, Args...) {
  cufftResult cufft_rt = cufftExecZ2D(plan, idata, odata);
  KOKKOSFFT_EXPECTS(cufft_rt == CUFFT_SUCCESS, "cufftExecZ2D failed");
}

template <typename... Args>
inline void exec_plan(cufftHandle& plan, cufftComplex* idata,
                      cufftComplex* odata, int direction, Args...) {
  cufftResult cufft_rt = cufftExecC2C(plan, idata, odata, direction);
  KOKKOSFFT_EXPECTS(cufft_rt == CUFFT_SUCCESS, "cufftExecC2C failed");
}

template <typename... Args>
inline void exec_plan(cufftHandle& plan, cufftDoubleComplex* idata,
                      cufftDoubleComplex* odata, int direction, Args...) {
  cufftResult cufft_rt = cufftExecZ2Z(plan, idata, odata, direction);
  KOKKOSFFT_EXPECTS(cufft_rt == CUFFT_SUCCESS, "cufftExecZ2Z failed");
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
