// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_ROCM_TYPES_HPP
#define KOKKOSFFT_ROCM_TYPES_HPP

#include <complex>
#include <rocfft/rocfft.h>
#include "KokkosFFT_common_types.hpp"

// Check the size of complex type
static_assert(sizeof(std::complex<float>) == sizeof(Kokkos::complex<float>));
static_assert(alignof(std::complex<float>) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(std::complex<double>) == sizeof(Kokkos::complex<double>));
static_assert(alignof(std::complex<double>) <=
              alignof(Kokkos::complex<double>));

#ifdef ENABLE_HOST_AND_DEVICE
#include <fftw3.h>
#include "KokkosFFT_utils.hpp"
static_assert(sizeof(fftwf_complex) == sizeof(Kokkos::complex<float>));
static_assert(alignof(fftwf_complex) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(fftw_complex) == sizeof(Kokkos::complex<double>));
static_assert(alignof(fftw_complex) <= alignof(Kokkos::complex<double>));
#endif

namespace KokkosFFT {
namespace Impl {
using FFTDirectionType                     = int;
constexpr FFTDirectionType ROCFFT_FORWARD  = 1;
constexpr FFTDirectionType ROCFFT_BACKWARD = -1;

enum class FFTWTransformType { R2C, D2Z, C2R, Z2D, C2C, Z2Z };

template <typename ExecutionSpace>
using TransformType = FFTWTransformType;

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
  using _TransformType = TransformType<ExecutionSpace>;

  static constexpr _TransformType m_type = std::is_same_v<T1, float>
                                               ? FFTWTransformType::R2C
                                               : FFTWTransformType::D2Z;
  static constexpr _TransformType type() { return m_type; };
};

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type<ExecutionSpace, Kokkos::complex<T1>, T2> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  using _TransformType = TransformType<ExecutionSpace>;

  static constexpr _TransformType m_type = std::is_same_v<T2, float>
                                               ? FFTWTransformType::C2R
                                               : FFTWTransformType::Z2D;
  static constexpr _TransformType type() { return m_type; };
};

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type<ExecutionSpace, Kokkos::complex<T1>,
                      Kokkos::complex<T2>> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  using _TransformType = TransformType<ExecutionSpace>;

  static constexpr _TransformType m_type = std::is_same_v<T1, float>
                                               ? FFTWTransformType::C2C
                                               : FFTWTransformType::Z2Z;
  static constexpr _TransformType type() { return m_type; };
};

#ifdef ENABLE_HOST_AND_DEVICE

template <typename ExecutionSpace>
struct FFTDataType {
  using float32 = float;
  using float64 = double;
  using complex64 =
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                         std::complex<float>, fftwf_complex>;
  using complex128 =
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                         std::complex<double>, fftw_complex>;
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType {
  using fftwHandle = std::conditional_t<
      std::is_same_v<KokkosFFT::Impl::real_type_t<T1>, float>, fftwf_plan,
      fftw_plan>;
  using type = std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                                  rocfft_plan, fftwHandle>;
};

template <typename ExecutionSpace>
using FFTInfoType =
    std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                       rocfft_execution_info, int>;

template <typename ExecutionSpace>
auto direction_type(Direction direction) {
  static constexpr FFTDirectionType _FORWARD =
      std::is_same_v<ExecutionSpace, Kokkos::HIP> ? ROCFFT_FORWARD
                                                  : FFTW_FORWARD;

  static constexpr FFTDirectionType _BACKWARD =
      std::is_same_v<ExecutionSpace, Kokkos::HIP> ? ROCFFT_BACKWARD
                                                  : FFTW_BACKWARD;
  return direction == Direction::forward ? _FORWARD : _BACKWARD;
}
#else
template <typename ExecutionSpace>
struct FFTDataType {
  using float32    = float;
  using float64    = double;
  using complex64  = std::complex<float>;
  using complex128 = std::complex<double>;
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType {
  using type = rocfft_plan;
};

template <typename ExecutionSpace>
using FFTInfoType = rocfft_execution_info;

template <typename ExecutionSpace>
auto direction_type(Direction direction) {
  return direction == Direction::forward ? ROCFFT_FORWARD : ROCFFT_BACKWARD;
}
#endif
}  // namespace Impl
}  // namespace KokkosFFT

#endif