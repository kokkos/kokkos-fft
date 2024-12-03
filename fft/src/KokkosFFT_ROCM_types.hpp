// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_ROCM_TYPES_HPP
#define KOKKOSFFT_ROCM_TYPES_HPP

#include <complex>
#include <rocfft/rocfft.h>
#include "KokkosFFT_common_types.hpp"
#if defined(ENABLE_HOST_AND_DEVICE)
#include "KokkosFFT_FFTW_Types.hpp"
#endif

// Check the size of complex type
static_assert(sizeof(std::complex<float>) == sizeof(Kokkos::complex<float>));
static_assert(alignof(std::complex<float>) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(std::complex<double>) == sizeof(Kokkos::complex<double>));
static_assert(alignof(std::complex<double>) <=
              alignof(Kokkos::complex<double>));

namespace KokkosFFT {
namespace Impl {
using FFTDirectionType                     = int;
constexpr FFTDirectionType ROCFFT_FORWARD  = 1;
constexpr FFTDirectionType ROCFFT_BACKWARD = -1;

#if !defined(ENABLE_HOST_AND_DEVICE)
enum class FFTWTransformType { R2C, D2Z, C2R, Z2D, C2C, Z2Z };
#endif

template <typename ExecutionSpace>
using TransformType = FFTWTransformType;

/// \brief A class that wraps rocfft for RAII
template <typename ExecutionSpace, typename T>
struct ScopedRocfftPlanType {
  using floating_point_type = KokkosFFT::Impl::base_floating_point_type<T>;
  rocfft_plan m_plan;
  rocfft_execution_info m_execution_info;

  using BufferViewType =
      Kokkos::View<Kokkos::complex<floating_point_type> *, ExecutionSpace>;

  bool m_is_info_created = false;
  bool m_is_plan_created = false;

  //! Internal work buffer
  BufferViewType m_buffer;

  ScopedRocfftPlanType() {}
  ~ScopedRocfftPlanType() {
    if (m_is_info_created) rocfft_execution_info_destroy(m_execution_info);
    if (m_is_plan_created) rocfft_plan_destroy(m_plan);
  }

  void allocate_work_buffer(std::size_t workbuffersize) {
    m_buffer = BufferViewType("work buffer", workbuffersize);
  }
  rocfft_plan &plan() { return m_plan; }
  rocfft_execution_info &execution_info() { return m_execution_info; }
};

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

#if defined(ENABLE_HOST_AND_DEVICE)

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
  using fftw_plan_type   = ScopedFFTWPlanType<ExecutionSpace, T1, T2>;
  using rocfft_plan_type = ScopedRocfftPlanType<ExecutionSpace, T1>;
  using type = std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                                  rocfft_plan_type, fftw_plan_type>;
};

template <typename ExecutionSpace>
auto direction_type(Direction direction) {
  static constexpr FFTDirectionType FORWARD =
      std::is_same_v<ExecutionSpace, Kokkos::HIP> ? ROCFFT_FORWARD
                                                  : FFTW_FORWARD;

  static constexpr FFTDirectionType BACKWARD =
      std::is_same_v<ExecutionSpace, Kokkos::HIP> ? ROCFFT_BACKWARD
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
  using type = ScopedRocfftPlanType<ExecutionSpace, T1>;
};

template <typename ExecutionSpace>
auto direction_type(Direction direction) {
  return direction == Direction::forward ? ROCFFT_FORWARD : ROCFFT_BACKWARD;
}
#endif
}  // namespace Impl
}  // namespace KokkosFFT

#endif
