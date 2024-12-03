// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_HOST_TYPES_HPP
#define KOKKOSFFT_HOST_TYPES_HPP

#include "KokkosFFT_FFTW_Types.hpp"

namespace KokkosFFT {
namespace Impl {
using FFTDirectionType = int;

template <typename ExecutionSpace>
struct FFTDataType {
  using float32    = float;
  using float64    = double;
  using complex64  = fftwf_complex;
  using complex128 = fftw_complex;
};

template <typename ExecutionSpace>
using TransformType = FFTWTransformType;

template <typename ExecutionSpace, typename T1, typename T2>
using transform_type = fftw_transform_type<ExecutionSpace, T1, T2>;

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType {
  using type = ScopedFFTWPlanType<ExecutionSpace, T1, T2>;
};

template <typename ExecutionSpace>
auto direction_type(Direction direction) {
  return direction == Direction::forward ? FFTW_FORWARD : FFTW_BACKWARD;
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
