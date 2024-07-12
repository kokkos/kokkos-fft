// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_CUDA_TYPES_HPP
#define KOKKOSFFT_CUDA_TYPES_HPP

#include <cufft.h>
#include "KokkosFFT_common_types.hpp"

// Check the size of complex type
static_assert(sizeof(cufftComplex) == sizeof(Kokkos::complex<float>));
static_assert(alignof(cufftComplex) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(cufftDoubleComplex) == sizeof(Kokkos::complex<double>));
static_assert(alignof(cufftDoubleComplex) <= alignof(Kokkos::complex<double>));

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
using FFTDirectionType = int;

// Unused
template <typename ExecutionSpace>
using FFTInfoType = int;

#ifdef ENABLE_HOST_AND_DEVICE
enum class FFTWTransformType { R2C, D2Z, C2R, Z2D, C2C, Z2Z };

template <typename ExecutionSpace>
struct FFTDataType {
  using float32 =
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::Cuda>,
                         cufftReal, float>;
  using float64 =
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::Cuda>,
                         cufftDoubleReal, double>;
  using complex64 =
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::Cuda>,
                         cufftComplex, fftwf_complex>;
  using complex128 =
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::Cuda>,
                         cufftDoubleComplex, fftw_complex>;
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType {
  using fftwHandle = std::conditional_t<
      std::is_same_v<KokkosFFT::Impl::base_floating_point_type<T1>, float>,
      fftwf_plan, fftw_plan>;
  using type = std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::Cuda>,
                                  cufftHandle, fftwHandle>;
};

template <typename ExecutionSpace>
using TransformType =
    std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::Cuda>, cufftType,
                       FFTWTransformType>;

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

  static constexpr _TransformType m_cuda_type =
      std::is_same_v<T1, float> ? CUFFT_R2C : CUFFT_D2Z;
  static constexpr _TransformType m_cpu_type = std::is_same_v<T1, float>
                                                   ? FFTWTransformType::R2C
                                                   : FFTWTransformType::D2Z;

  static constexpr _TransformType type() {
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>) {
      return m_cuda_type;
    } else {
      return m_cpu_type;
    }
  }
};

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type<ExecutionSpace, Kokkos::complex<T1>, T2> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  using _TransformType = TransformType<ExecutionSpace>;

  static constexpr _TransformType m_cuda_type =
      std::is_same_v<T1, float> ? CUFFT_C2R : CUFFT_Z2D;
  static constexpr _TransformType m_cpu_type = std::is_same_v<T1, float>
                                                   ? FFTWTransformType::C2R
                                                   : FFTWTransformType::Z2D;

  static constexpr _TransformType type() {
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>) {
      return m_cuda_type;
    } else {
      return m_cpu_type;
    }
  }
};

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type<ExecutionSpace, Kokkos::complex<T1>,
                      Kokkos::complex<T2>> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  using _TransformType = TransformType<ExecutionSpace>;

  static constexpr _TransformType m_cuda_type =
      std::is_same_v<T1, float> ? CUFFT_C2C : CUFFT_Z2Z;
  static constexpr _TransformType m_cpu_type = std::is_same_v<T1, float>
                                                   ? FFTWTransformType::C2C
                                                   : FFTWTransformType::Z2Z;

  static constexpr _TransformType type() {
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>) {
      return m_cuda_type;
    } else {
      return m_cpu_type;
    }
  }
};

template <typename ExecutionSpace>
auto direction_type(Direction direction) {
  static constexpr FFTDirectionType _FORWARD =
      std::is_same_v<ExecutionSpace, Kokkos::Cuda> ? CUFFT_FORWARD
                                                   : FFTW_FORWARD;
  static constexpr FFTDirectionType _BACKWARD =
      std::is_same_v<ExecutionSpace, Kokkos::Cuda> ? CUFFT_INVERSE
                                                   : FFTW_BACKWARD;
  return direction == Direction::forward ? _FORWARD : _BACKWARD;
}
#else
template <typename ExecutionSpace>
struct FFTDataType {
  using float32    = cufftReal;
  using float64    = cufftDoubleReal;
  using complex64  = cufftComplex;
  using complex128 = cufftDoubleComplex;
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType {
  using type = cufftHandle;
};

template <typename ExecutionSpace>
using TransformType = cufftType;

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
  static constexpr _TransformType m_type =
      std::is_same_v<T1, float> ? CUFFT_R2C : CUFFT_D2Z;
  static constexpr _TransformType type() { return m_type; };
};

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type<ExecutionSpace, Kokkos::complex<T1>, T2> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  using _TransformType = TransformType<ExecutionSpace>;
  static constexpr _TransformType m_type =
      std::is_same_v<T2, float> ? CUFFT_C2R : CUFFT_Z2D;
  static constexpr _TransformType type() { return m_type; };
};

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type<ExecutionSpace, Kokkos::complex<T1>,
                      Kokkos::complex<T2>> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  using _TransformType = TransformType<ExecutionSpace>;
  static constexpr _TransformType m_type =
      std::is_same_v<T1, float> ? CUFFT_C2C : CUFFT_Z2Z;
  static constexpr _TransformType type() { return m_type; };
};

template <typename ExecutionSpace>
auto direction_type(Direction direction) {
  return direction == Direction::forward ? CUFFT_FORWARD : CUFFT_INVERSE;
}
#endif
}  // namespace Impl
}  // namespace KokkosFFT

#endif