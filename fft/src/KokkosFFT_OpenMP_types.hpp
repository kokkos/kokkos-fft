#ifndef KOKKOSFFT_OPENMP_TYPES_HPP
#define KOKKOSFFT_OPENMP_TYPES_HPP

#include <fftw3.h>
#include "KokkosFFT_common_types.hpp"
#include "KokkosFFT_utils.hpp"

// Check the size of complex type
static_assert(sizeof(fftwf_complex) == sizeof(Kokkos::complex<float>));
static_assert(alignof(fftwf_complex) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(fftw_complex) == sizeof(Kokkos::complex<double>));
static_assert(alignof(fftw_complex) <= alignof(Kokkos::complex<double>));

namespace KokkosFFT {
namespace Impl {
using FFTDirectionType = int;

// Unused
template <typename ExecutionSpace>
using FFTInfoType = int;

enum class FFTWTransformType { R2C, D2Z, C2R, Z2D, C2C, Z2Z };

template <typename ExecutionSpace>
struct FFTDataType {
  using float32    = float;
  using float64    = double;
  using complex64  = fftwf_complex;
  using complex128 = fftw_complex;
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType {
  using type = std::conditional_t<
      std::is_same_v<KokkosFFT::Impl::real_type_t<T1>, float>, fftwf_plan,
      fftw_plan>;
};

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

template <typename ExecutionSpace>
auto direction_type(Direction direction) {
  return direction == Direction::forward ? FFTW_FORWARD : FFTW_BACKWARD;
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif