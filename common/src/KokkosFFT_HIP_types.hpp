#ifndef KOKKOSFFT_HIP_TYPES_HPP
#define KOKKOSFFT_HIP_TYPES_HPP

#include <hipfft/hipfft.h>

// Check the size of complex type
static_assert(sizeof(hipfftComplex) == sizeof(Kokkos::complex<float>));
static_assert(alignof(hipfftComplex) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(hipfftDoubleComplex) == sizeof(Kokkos::complex<double>));
static_assert(alignof(hipfftDoubleComplex) <= alignof(Kokkos::complex<double>));

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
enum class Direction {
  Forward,
  Backward,
};

using FFTDirectionType = int;

#ifdef ENABLE_HOST_AND_DEVICE
enum class FFTWTransformType { R2C, D2Z, C2R, Z2D, C2C, Z2Z };

template <typename ExecutionSpace>
struct FFTDataType {
  using float32 =
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                         hipfftReal, float>;
  using float64 =
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                         hipfftDoubleReal, double>;
  using complex64 =
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                         hipfftComplex, fftwf_complex>;
  using complex128 =
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                         hipfftDoubleComplex, fftw_complex>;
};

template <typename ExecutionSpace, typename T>
struct FFTPlanType {
  using fftwHandle =
      std::conditional_t<std::is_same_v<KokkosFFT::Impl::real_type_t<T>, float>,
                         fftwf_plan, fftw_plan>;
  using type = std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                                  hipfftHandle, fftwHandle>;
};

template <typename ExecutionSpace>
using TransformType =
    std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>, hipfftType,
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
      std::is_same_v<T1, float> ? HIPFFT_R2C : HIPFFT_D2Z;
  static constexpr _TransformType m_cpu_type = std::is_same_v<T1, float>
                                                   ? FFTWTransformType::R2C
                                                   : FFTWTransformType::D2Z;

  static constexpr _TransformType type() {
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::HIP>) {
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
      std::is_same_v<T1, float> ? HIPFFT_C2R : HIPFFT_Z2D;
  static constexpr _TransformType m_cpu_type = std::is_same_v<T1, float>
                                                   ? FFTWTransformType::C2R
                                                   : FFTWTransformType::Z2D;

  static constexpr _TransformType type() {
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::HIP>) {
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
      std::is_same_v<T1, float> ? HIPFFT_C2C : HIPFFT_Z2Z;
  static constexpr _TransformType m_cpu_type = std::is_same_v<T1, float>
                                                   ? FFTWTransformType::C2C
                                                   : FFTWTransformType::Z2Z;

  static constexpr _TransformType type() {
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::HIP>) {
      return m_cuda_type;
    } else {
      return m_cpu_type;
    }
  }
};

template <typename ExecutionSpace>
auto direction_type(Direction direction) {
  static constexpr FFTDirectionType _FORWARD =
      std::is_same_v<ExecutionSpace, Kokkos::HIP> ? HIPFFT_FORWARD
                                                  : FFTW_FORWARD;
  static constexpr FFTDirectionType _BACKWARD =
      std::is_same_v<ExecutionSpace, Kokkos::HIP> ? HIPFFT_BACKWARD
                                                  : FFTW_BACKWARD;
  return direction == Direction::Forward ? _FORWARD : _BACKWARD;
}
#else
template <typename ExecutionSpace>
struct FFTDataType {
  using float32    = hipfftReal;
  using float64    = hipfftDoubleReal;
  using complex64  = hipfftComplex;
  using complex128 = hipfftDoubleComplex;
};

template <typename ExecutionSpace, typename T>
struct FFTPlanType {
  using type = hipfftHandle;
};

template <typename ExecutionSpace>
using TransformType = hipfftType;

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
      std::is_same_v<T1, float> ? HIPFFT_R2C : HIPFFT_D2Z;
  static constexpr _TransformType type() { return m_type; };
};

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type<ExecutionSpace, Kokkos::complex<T1>, T2> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  using _TransformType = TransformType<ExecutionSpace>;
  static constexpr _TransformType m_type =
      std::is_same_v<T2, float> ? HIPFFT_C2R : HIPFFT_Z2D;
  static constexpr _TransformType type() { return m_type; };
};

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type<ExecutionSpace, Kokkos::complex<T1>,
                      Kokkos::complex<T2>> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  using _TransformType = TransformType<ExecutionSpace>;
  static constexpr _TransformType m_type =
      std::is_same_v<T1, float> ? HIPFFT_C2C : HIPFFT_Z2Z;
  static constexpr _TransformType type() { return m_type; };
};

template <typename ExecutionSpace>
auto direction_type(Direction direction) {
  return direction == Direction::Forward ? HIPFFT_FORWARD : HIPFFT_BACKWARD;
}
#endif
}  // namespace Impl
}  // namespace KokkosFFT

#endif