// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_DEFAULT_TYPES_HPP
#define KOKKOSFFT_DEFAULT_TYPES_HPP

#include <Kokkos_Core.hpp>

#if !defined(KOKKOS_ENABLE_COMPLEX_ALIGN)
static_assert(false,
              "KokkosFFT requires option -DKokkos_ENABLE_COMPLEX_ALIGN=ON to "
              "build Kokkos");
#endif

#include "KokkosFFT_config.hpp"

#if defined(KOKKOSFFT_ENABLE_TPL_CUFFT)
#define KOKKOSFFT_HAS_DEVICE_TPL
#include "KokkosFFT_Cuda_types.hpp"
#elif defined(KOKKOSFFT_ENABLE_TPL_ROCFFT)
#define KOKKOSFFT_HAS_DEVICE_TPL
#include "KokkosFFT_ROCM_types.hpp"
#elif defined(KOKKOSFFT_ENABLE_TPL_HIPFFT)
#define KOKKOSFFT_HAS_DEVICE_TPL
#include "KokkosFFT_HIP_types.hpp"
#elif defined(KOKKOSFFT_ENABLE_TPL_ONEMKL)
#define KOKKOSFFT_HAS_DEVICE_TPL
#include "KokkosFFT_SYCL_types.hpp"
#elif defined(KOKKOSFFT_ENABLE_TPL_FFTW)
#include "KokkosFFT_Host_types.hpp"
#else
static_assert(false, "KokkosFFT requires at least one backend library");
#endif

namespace KokkosFFT {
namespace Impl {
// Define fft data types
template <typename ExecutionSpace, typename T>
struct fft_data_type {
  using type = std::conditional_t<
      std::is_same_v<T, float>,
      typename KokkosFFT::Impl::FFTDataType<ExecutionSpace>::float32,
      typename KokkosFFT::Impl::FFTDataType<ExecutionSpace>::float64>;
};

template <typename ExecutionSpace, typename T>
struct fft_data_type<ExecutionSpace, Kokkos::complex<T>> {
  using type = std::conditional_t<
      std::is_same_v<T, float>,
      typename KokkosFFT::Impl::FFTDataType<ExecutionSpace>::complex64,
      typename KokkosFFT::Impl::FFTDataType<ExecutionSpace>::complex128>;
};

#if defined(KOKKOSFFT_HAS_DEVICE_TPL)
#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)
// Backend libraries are available from all the execution spaces
template <typename ExecutionSpace>
struct is_allowed_space : std::true_type {};
#else
// Only device backend library is available
template <typename ExecutionSpace>
struct is_allowed_space
    : std::is_same<ExecutionSpace, Kokkos::DefaultExecutionSpace> {};
#endif
#else
// Only host backend library is available
template <typename ExecutionSpace>
struct is_allowed_space : is_AnyHostSpace<ExecutionSpace> {};
#endif

/// \brief Helper to check if the backend FFT library corresponding to
/// ExecutionSpace is available or not
template <typename ExecutionSpace>
inline constexpr bool is_allowed_space_v =
    is_allowed_space<ExecutionSpace>::value;

}  // namespace Impl
}  // namespace KokkosFFT

#endif
