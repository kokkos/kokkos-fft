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

/*
// Define fft transform types
template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type {
  static_assert(std::is_same_v<T1, T2>,
                "Real to real transform is unavailable");
};

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type<ExecutionSpace, T1, Kokkos::complex<T2>> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  static constexpr FFTWTransformType m_type = std::is_same_v<T1, float>
                                                  ? FFTWTransformType::R2C
                                                  : FFTWTransformType::D2Z;
  static constexpr FFTWTransformType type() { return m_type; };
};

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type<ExecutionSpace, Kokkos::complex<T1>, T2> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  static constexpr FFTWTransformType m_type = std::is_same_v<T2, float>
                                                  ? FFTWTransformType::C2R
                                                  : FFTWTransformType::Z2D;
  static constexpr FFTWTransformType type() { return m_type; };
};

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type<ExecutionSpace, Kokkos::complex<T1>,
                      Kokkos::complex<T2>> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  static constexpr FFTWTransformType m_type = std::is_same_v<T1, float>
                                                  ? FFTWTransformType::C2C
                                                  : FFTWTransformType::Z2Z;
  static constexpr FFTWTransformType type() { return m_type; };
};
*/

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
