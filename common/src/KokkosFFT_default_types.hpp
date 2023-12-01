#ifndef __KOKKOSFFT_DEFAULT_TYPES_HPP__
#define __KOKKOSFFT_DEFAULT_TYPES_HPP__

#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
  using default_device = Kokkos::Cuda;
  #include "KokkosFFT_Cuda_types.hpp"
#elif defined(KOKKOS_ENABLE_HIP)
  using default_device = Kokkos::HIP;
  #include "KokkosFFT_HIP_types.hpp"
#elif defined(KOKKOS_ENABLE_OPENMP)
  using default_device = Kokkos::OpenMP;
  #include "KokkosFFT_OpenMP_types.hpp"
#elif defined(KOKKOS_ENABLE_THREADS)
  using default_device = Kokkos::Threads;
  #include "KokkosFFT_OpenMP_types.hpp"
#else
  using default_device = Kokkos::Serial;
  #include "KokkosFFT_OpenMP_types.hpp"
#endif

#include "KokkosFFT_utils.hpp"

// Check the size of complex type
static_assert(sizeof(KokkosFFT::FFTDataType::complex64) == sizeof(Kokkos::complex<float>));
static_assert(alignof(KokkosFFT::FFTDataType::complex64) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(KokkosFFT::FFTDataType::complex128) == sizeof(Kokkos::complex<double>));
static_assert(alignof(KokkosFFT::FFTDataType::complex128) <= alignof(Kokkos::complex<double>));

namespace KokkosFFT {
  // Define fft data types
  template <typename T>
  struct fft_data_type {
    using type = std::conditional_t<std::is_same_v<T, float>, KokkosFFT::FFTDataType::float32, KokkosFFT::FFTDataType::float64>;
  };

  template <typename T>
  struct fft_data_type<Kokkos::complex<T>> {
    using type = std::conditional_t<std::is_same_v<T, float>, KokkosFFT::FFTDataType::complex64, KokkosFFT::FFTDataType::complex128>;
  };

  // Define type to specify transform axis
  template <std::size_t DIM>
  using axis_type = std::array<int, DIM>;
}

#endif