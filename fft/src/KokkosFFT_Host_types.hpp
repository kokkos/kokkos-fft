// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_HOST_TYPES_HPP
#define KOKKOSFFT_HOST_TYPES_HPP

#include <fftw3.h>
#include "KokkosFFT_FFTW_Types.hpp"

namespace KokkosFFT {
namespace Impl {
template <typename ExecutionSpace>
struct FFTDataType {
  using float32    = float;
  using float64    = double;
  using complex64  = fftwf_complex;
  using complex128 = fftw_complex;
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType {
  using type = ScopedFFTWPlan<ExecutionSpace, T1, T2>;
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTDynPlanType {
  using type = ScopedFFTWPlan<ExecutionSpace, T1, T2>;
};

}  // namespace Impl
}  // namespace KokkosFFT

#endif
