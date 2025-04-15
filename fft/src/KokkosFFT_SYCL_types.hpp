// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_SYCL_TYPES_HPP
#define KOKKOSFFT_SYCL_TYPES_HPP

#include <complex>
#include <sycl/sycl.hpp>
#include <mkl.h>
#include <oneapi/mkl/dfti.hpp>
#include "KokkosFFT_common_types.hpp"
#include "KokkosFFT_utils.hpp"

#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)
#include "KokkosFFT_FFTW_Types.hpp"
#endif

// Check the size of complex type
// [TO DO] I guess this kind of test is already made by Kokkos itself
static_assert(sizeof(std::complex<float>) == sizeof(Kokkos::complex<float>));
static_assert(alignof(std::complex<float>) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(std::complex<double>) == sizeof(Kokkos::complex<double>));
static_assert(alignof(std::complex<double>) <=
              alignof(Kokkos::complex<double>));

namespace KokkosFFT {
namespace Impl {
using FFTDirectionType                      = int;
constexpr FFTDirectionType MKL_FFT_FORWARD  = 1;
constexpr FFTDirectionType MKL_FFT_BACKWARD = -1;

#if !defined(KOKKOSFFT_ENABLE_TPL_FFTW)
enum class FFTWTransformType { R2C, D2Z, C2R, Z2D, C2C, Z2Z };
#endif

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

#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)

template <typename ExecutionSpace>
struct FFTDataType {
  using float32 = float;
  using float64 = double;

  using complex64 = std::conditional_t<
      std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL>,
      std::complex<float>, fftwf_complex>;
  using complex128 = std::conditional_t<
      std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL>,
      std::complex<double>, fftw_complex>;
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType {
  static_assert(std::is_same_v<T1, T2>,
                "Real to real transform is unavailable");
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType<ExecutionSpace, T1, Kokkos::complex<T2>> {
  using float_type = T1;
  static constexpr oneapi::mkl::dft::precision prec =
      std::is_same_v<KokkosFFT::Impl::base_floating_point_type<float_type>,
                     float>
          ? oneapi::mkl::dft::precision::SINGLE
          : oneapi::mkl::dft::precision::DOUBLE;
  static constexpr oneapi::mkl::dft::domain dom =
      oneapi::mkl::dft::domain::REAL;

  using fftwHandle   = ScopedFFTWPlan<ExecutionSpace, T1, Kokkos::complex<T2>>;
  using onemklHandle = oneapi::mkl::dft::descriptor<prec, dom>;
  using type         = std::conditional_t<
      std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL>, onemklHandle,
      fftwHandle>;
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType<ExecutionSpace, Kokkos::complex<T1>, T2> {
  using float_type = T2;
  static constexpr oneapi::mkl::dft::precision prec =
      std::is_same_v<KokkosFFT::Impl::base_floating_point_type<float_type>,
                     float>
          ? oneapi::mkl::dft::precision::SINGLE
          : oneapi::mkl::dft::precision::DOUBLE;
  static constexpr oneapi::mkl::dft::domain dom =
      oneapi::mkl::dft::domain::REAL;

  using fftwHandle   = ScopedFFTWPlan<ExecutionSpace, Kokkos::complex<T1>, T2>;
  using onemklHandle = oneapi::mkl::dft::descriptor<prec, dom>;
  using type         = std::conditional_t<
      std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL>, onemklHandle,
      fftwHandle>;
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType<ExecutionSpace, Kokkos::complex<T1>, Kokkos::complex<T2>> {
  using float_type = KokkosFFT::Impl::base_floating_point_type<T1>;
  static constexpr oneapi::mkl::dft::precision prec =
      std::is_same_v<KokkosFFT::Impl::base_floating_point_type<float_type>,
                     float>
          ? oneapi::mkl::dft::precision::SINGLE
          : oneapi::mkl::dft::precision::DOUBLE;
  static constexpr oneapi::mkl::dft::domain dom =
      oneapi::mkl::dft::domain::COMPLEX;

  using fftwHandle =
      ScopedFFTWPlan<ExecutionSpace, Kokkos::complex<T1>, Kokkos::complex<T2>>;
  using onemklHandle = oneapi::mkl::dft::descriptor<prec, dom>;
  using type         = std::conditional_t<
      std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL>, onemklHandle,
      fftwHandle>;
};

template <typename ExecutionSpace>
auto direction_type(Direction direction) {
  static constexpr FFTDirectionType FORWARD =
      std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL>
          ? MKL_FFT_FORWARD
          : FFTW_FORWARD;
  static constexpr FFTDirectionType BACKWARD =
      std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL>
          ? MKL_FFT_BACKWARD
          : FFTW_BACKWARD;
  return direction == Direction::forward ? FORWARD : BACKWARD;
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
  static_assert(std::is_same_v<T1, T2>,
                "Real to real transform is unavailable");
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType<ExecutionSpace, T1, Kokkos::complex<T2>> {
  using float_type = T1;
  static constexpr oneapi::mkl::dft::precision prec =
      std::is_same_v<KokkosFFT::Impl::base_floating_point_type<float_type>,
                     float>
          ? oneapi::mkl::dft::precision::SINGLE
          : oneapi::mkl::dft::precision::DOUBLE;
  static constexpr oneapi::mkl::dft::domain dom =
      oneapi::mkl::dft::domain::REAL;

  using type = oneapi::mkl::dft::descriptor<prec, dom>;
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType<ExecutionSpace, Kokkos::complex<T1>, T2> {
  using float_type = T2;
  static constexpr oneapi::mkl::dft::precision prec =
      std::is_same_v<KokkosFFT::Impl::base_floating_point_type<float_type>,
                     float>
          ? oneapi::mkl::dft::precision::SINGLE
          : oneapi::mkl::dft::precision::DOUBLE;
  static constexpr oneapi::mkl::dft::domain dom =
      oneapi::mkl::dft::domain::REAL;

  using type = oneapi::mkl::dft::descriptor<prec, dom>;
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType<ExecutionSpace, Kokkos::complex<T1>, Kokkos::complex<T2>> {
  using float_type = KokkosFFT::Impl::base_floating_point_type<T1>;
  static constexpr oneapi::mkl::dft::precision prec =
      std::is_same_v<KokkosFFT::Impl::base_floating_point_type<float_type>,
                     float>
          ? oneapi::mkl::dft::precision::SINGLE
          : oneapi::mkl::dft::precision::DOUBLE;
  static constexpr oneapi::mkl::dft::domain dom =
      oneapi::mkl::dft::domain::COMPLEX;

  using type = oneapi::mkl::dft::descriptor<prec, dom>;
};

template <typename ExecutionSpace>
auto direction_type(Direction direction) {
  return direction == Direction::forward ? MKL_FFT_FORWARD : MKL_FFT_BACKWARD;
}
#endif
}  // namespace Impl
}  // namespace KokkosFFT

#endif
