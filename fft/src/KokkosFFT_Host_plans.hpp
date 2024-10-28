// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_HOST_PLANS_HPP
#define KOKKOSFFT_HOST_PLANS_HPP

#include <numeric>
#include "KokkosFFT_default_types.hpp"
#include "KokkosFFT_layouts.hpp"
#include "KokkosFFT_traits.hpp"

namespace KokkosFFT {
namespace Impl {

template <typename ExecutionSpace, typename T>
void init_threads([[maybe_unused]] const ExecutionSpace& exec_space) {
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
          std::enable_if_t<is_AnyHostSpace_v<ExecutionSpace>, std::nullptr_t> =
              nullptr>
auto create_plan(const ExecutionSpace& exec_space,
                 std::unique_ptr<PlanType>& plan, const InViewType& in,
                 const OutViewType& out, BufferViewType&, InfoType&,
                 Direction direction, axis_type<fft_rank> axes,
                 shape_type<fft_rank> s, bool is_inplace) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "create_plan: InViewType and OutViewType must have the same base "
      "floating point type (float/double), the same layout "
      "(LayoutLeft/LayoutRight), "
      "and the same rank. ExecutionSpace must be accessible to the data in "
      "InViewType and OutViewType.");

  static_assert(
      InViewType::rank() >= fft_rank,
      "KokkosFFT::create_plan: Rank of View must be larger than Rank of FFT.");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  const int rank       = fft_rank;

  init_threads<ExecutionSpace,
               KokkosFFT::Impl::base_floating_point_type<in_value_type>>(
      exec_space);

  constexpr auto type =
      KokkosFFT::Impl::transform_type<ExecutionSpace, in_value_type,
                                      out_value_type>::type();
  auto [in_extents, out_extents, fft_extents, howmany] =
      KokkosFFT::Impl::get_extents(in, out, axes, s, is_inplace);
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
  [[maybe_unused]] auto sign =
      KokkosFFT::Impl::direction_type<ExecutionSpace>(direction);

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

template <typename ExecutionSpace, typename PlanType, typename InfoType,
          std::enable_if_t<is_AnyHostSpace_v<ExecutionSpace>, std::nullptr_t> =
              nullptr>
void destroy_plan_and_info(std::unique_ptr<PlanType>& plan, InfoType&) {
  if constexpr (std::is_same_v<PlanType, fftwf_plan>) {
    fftwf_destroy_plan(*plan);
  } else {
    fftw_destroy_plan(*plan);
  }
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
