#ifndef KOKKOSFFT_DEFAULT_TYPES_HPP
#define KOKKOSFFT_DEFAULT_TYPES_HPP

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

namespace KokkosFFT {
  // Define type to specify transform axis
  template <std::size_t DIM>
  using axis_type = std::array<int, DIM>;

  // Define type to specify new shape
  template <std::size_t DIM>
  using shape_type = std::array<std::size_t, DIM>;

  enum class Normalization {
    FORWARD,
    BACKWARD,
    ORTHO
  };
} // namespace KokkosFFT

namespace KokkosFFT {
namespace Impl {
  // Define fft data types
  template <typename ExecutionSpace, typename T>
  struct fft_data_type {
    using type = std::conditional_t<std::is_same_v<T, float>, typename KokkosFFT::Impl::FFTDataType<ExecutionSpace>::float32, typename KokkosFFT::Impl::FFTDataType<ExecutionSpace>::float64>;
  };

  template <typename ExecutionSpace, typename T>
  struct fft_data_type<ExecutionSpace, Kokkos::complex<T>> {
    using type = std::conditional_t<std::is_same_v<T, float>, typename KokkosFFT::Impl::FFTDataType<ExecutionSpace>::complex64, typename KokkosFFT::Impl::FFTDataType<ExecutionSpace>::complex128>;
  };
} // namespace Impl
} // namespace KokkosFFT

#endif