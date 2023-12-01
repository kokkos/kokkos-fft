#ifndef __KOKKOSFFT_CUDA_TYPES_HPP__
#define __KOKKOSFFT_CUDA_TYPES_HPP__

#include <cufft.h>

namespace KokkosFFT {
  #define KOKKOS_FFT_FORWARD CUFFT_FORWARD
  #define KOKKOS_FFT_BACKWARD CUFFT_INVERSE
  #define KOKKOS_FFT_R2C CUFFT_R2C
  #define KOKKOS_FFT_D2Z CUFFT_D2Z
  #define KOKKOS_FFT_C2R CUFFT_C2R
  #define KOKKOS_FFT_Z2D CUFFT_Z2D
  #define KOKKOS_FFT_C2C CUFFT_C2C
  #define KOKKOS_FFT_Z2Z CUFFT_Z2Z

  struct FFTDataType {
    using float32    = cufftReal;
    using float64    = cufftDoubleReal;
    using complex64  = cufftComplex;
    using complex128 = cufftDoubleComplex;
  };

  template <typename T>
  struct FFTPlanType {
    using type = cufftHandle;
  };

  using FFTResultType = cufftResult;
  using TransformType = cufftType;
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
};

#endif