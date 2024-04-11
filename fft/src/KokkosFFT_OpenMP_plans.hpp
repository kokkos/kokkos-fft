// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_OPENMP_PLANS_HPP
#define KOKKOSFFT_OPENMP_PLANS_HPP

#include <numeric>
#include "KokkosFFT_default_types.hpp"
#include "KokkosFFT_layouts.hpp"

namespace KokkosFFT {
namespace Impl {
template <typename ExecutionSpace, typename T>
void _init_threads(const ExecutionSpace& exec_space) {
#if defined(KOKKOS_ENABLE_OPENMP) || defined(KOKKOS_ENABLE_THREADS)
  int nthreads = exec_space.concurrency();

  if constexpr (std::is_same_v<T, float>) {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(nthreads);
  } else {
    fftw_init_threads();
    fftw_plan_with_nthreads(nthreads);
  }
#endif
}

// batched transform, over ND Views
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType, typename BufferViewType, typename InfoType,
          std::size_t fft_rank = 1,
          std::enable_if_t<
              std::is_same_v<ExecutionSpace, Kokkos::DefaultHostExecutionSpace>,
              std::nullptr_t> = nullptr>
auto _create(const ExecutionSpace& exec_space, std::unique_ptr<PlanType>& plan,
             const InViewType& in, const OutViewType& out,
             [[maybe_unused]] BufferViewType& buffer,
             [[maybe_unused]] InfoType& execution_info,
             [[maybe_unused]] Direction direction, axis_type<fft_rank> axes,
             shape_type<fft_rank> s) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::_create: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::_create: OutViewType is not a Kokkos::View.");
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(
      InViewType::rank() >= fft_rank,
      "KokkosFFT::_create: Rank of View must be larger than Rank of FFT.");
  const int rank = fft_rank;

  _init_threads<ExecutionSpace, KokkosFFT::Impl::real_type_t<in_value_type>>(
      exec_space);

  constexpr auto type =
      KokkosFFT::Impl::transform_type<ExecutionSpace, in_value_type,
                                      out_value_type>::type();
  auto [in_extents, out_extents, fft_extents, howmany] =
      KokkosFFT::Impl::get_extents(in, out, axes, s);
  int idist    = std::accumulate(in_extents.begin(), in_extents.end(), 1,
                              std::multiplies<>());
  int odist    = std::accumulate(out_extents.begin(), out_extents.end(), 1,
                              std::multiplies<>());
  int fft_size = std::accumulate(fft_extents.begin(), fft_extents.end(), 1,
                                 std::multiplies<>());

  auto* idata = reinterpret_cast<typename KokkosFFT::Impl::fft_data_type<
      ExecutionSpace, in_value_type>::type*>(in.data());
  auto* odata = reinterpret_cast<typename KokkosFFT::Impl::fft_data_type<
      ExecutionSpace, out_value_type>::type*>(out.data());

  // For the moment, considering the contiguous layout only
  int istride = 1, ostride = 1;
  auto sign = KokkosFFT::Impl::direction_type<ExecutionSpace>(direction);

  plan = std::make_unique<PlanType>();
  if constexpr (type == KokkosFFT::Impl::FFTWTransformType::R2C) {
    *plan = fftwf_plan_many_dft_r2c(
        rank, fft_extents.data(), howmany, idata, in_extents.data(), istride,
        idist, odata, out_extents.data(), ostride, odist, FFTW_ESTIMATE);
  } else if constexpr (type == KokkosFFT::Impl::FFTWTransformType::D2Z) {
    *plan = fftw_plan_many_dft_r2c(
        rank, fft_extents.data(), howmany, idata, in_extents.data(), istride,
        idist, odata, out_extents.data(), ostride, odist, FFTW_ESTIMATE);
  } else if constexpr (type == KokkosFFT::Impl::FFTWTransformType::C2R) {
    *plan = fftwf_plan_many_dft_c2r(
        rank, fft_extents.data(), howmany, idata, in_extents.data(), istride,
        idist, odata, out_extents.data(), ostride, odist, FFTW_ESTIMATE);
  } else if constexpr (type == KokkosFFT::Impl::FFTWTransformType::Z2D) {
    *plan = fftw_plan_many_dft_c2r(
        rank, fft_extents.data(), howmany, idata, in_extents.data(), istride,
        idist, odata, out_extents.data(), ostride, odist, FFTW_ESTIMATE);
  } else if constexpr (type == KokkosFFT::Impl::FFTWTransformType::C2C) {
    *plan = fftwf_plan_many_dft(
        rank, fft_extents.data(), howmany, idata, in_extents.data(), istride,
        idist, odata, out_extents.data(), ostride, odist, sign, FFTW_ESTIMATE);
  } else if constexpr (type == KokkosFFT::Impl::FFTWTransformType::Z2Z) {
    *plan = fftw_plan_many_dft(
        rank, fft_extents.data(), howmany, idata, in_extents.data(), istride,
        idist, odata, out_extents.data(), ostride, odist, sign, FFTW_ESTIMATE);
  }

  return fft_size;
}

template <typename ExecutionSpace, typename PlanType,
          std::enable_if_t<
              std::is_same_v<ExecutionSpace, Kokkos::DefaultHostExecutionSpace>,
              std::nullptr_t> = nullptr>
void _destroy_plan(std::unique_ptr<PlanType>& plan) {
  if constexpr (std::is_same_v<PlanType, fftwf_plan>) {
    fftwf_destroy_plan(*plan);
  } else {
    fftw_destroy_plan(*plan);
  }
}

template <typename ExecutionSpace, typename InfoType,
          std::enable_if_t<
              std::is_same_v<ExecutionSpace, Kokkos::DefaultHostExecutionSpace>,
              std::nullptr_t> = nullptr>
void _destroy_info(InfoType& plan) {}
}  // namespace Impl
}  // namespace KokkosFFT

#endif