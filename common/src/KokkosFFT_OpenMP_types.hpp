#ifndef KOKKOSFFT_OPENMP_TYPES_HPP
#define KOKKOSFFT_OPENMP_TYPES_HPP

#include <fftw3.h>
#include "KokkosFFT_utils.hpp"

namespace KokkosFFT {
namespace Impl {
  enum class TransformType {
    R2C,
    D2Z,
    C2R,
    Z2D,
    C2C,
    Z2Z
  };

  #define KOKKOS_FFT_FORWARD FFTW_FORWARD
  #define KOKKOS_FFT_BACKWARD FFTW_BACKWARD
  #define KOKKOS_FFT_R2C TransformType::R2C
  #define KOKKOS_FFT_D2Z TransformType::D2Z
  #define KOKKOS_FFT_C2R TransformType::C2R
  #define KOKKOS_FFT_Z2D TransformType::Z2D
  #define KOKKOS_FFT_C2C TransformType::C2C
  #define KOKKOS_FFT_Z2Z TransformType::Z2Z

  struct FFTDataType {
    using float32    = float;
    using float64    = double;
    using complex64  = fftwf_complex;
    using complex128 = fftw_complex;
  };

  template <typename T>
  struct FFTPlanType {
    using type = std::conditional_t<std::is_same_v<KokkosFFT::Impl::real_type_t<T>, float>, fftwf_plan, fftw_plan>;
  };

  using FFTResultType = int;
  using FFTDirectionType = int;

  template <typename T1, typename T2>
  struct transform_type {
    static_assert(std::is_same_v<T1, T2>, "Real to real transform is unavailable");
  };

  template <typename T1, typename T2>
  struct transform_type<T1, Kokkos::complex<T2>> {
    static_assert(std::is_same_v<T1, T2>, "T1 and T2 should have the same precision");
    static constexpr TransformType m_type = std::is_same_v<T1, float> ? KOKKOS_FFT_R2C : KOKKOS_FFT_D2Z;
    static constexpr TransformType type() { return m_type; };
  };

  template <typename T1, typename T2>
  struct transform_type<Kokkos::complex<T1>, T2> {
    static_assert(std::is_same_v<T1, T2>, "T1 and T2 should have the same precision");
    static constexpr TransformType m_type = std::is_same_v<T2, float> ? KOKKOS_FFT_C2R : KOKKOS_FFT_Z2D;
    static constexpr TransformType type() { return m_type; };
  };

  template <typename T1, typename T2>
  struct transform_type<Kokkos::complex<T1>, Kokkos::complex<T2>> {
    static_assert(std::is_same_v<T1, T2>, "T1 and T2 should have the same precision");
    static constexpr TransformType m_type = std::is_same_v<T1, float> ? KOKKOS_FFT_C2C : KOKKOS_FFT_Z2Z;
    static constexpr TransformType type() { return m_type; };
  };
} // namespace Impl
}; // namespace KokkosFFT

#endif