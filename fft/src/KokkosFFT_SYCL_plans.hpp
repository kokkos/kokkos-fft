// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_SYCL_PLANS_HPP
#define KOKKOSFFT_SYCL_PLANS_HPP

#include <numeric>
#include <algorithm>
#include "KokkosFFT_SYCL_types.hpp"
#include "KokkosFFT_layouts.hpp"

namespace KokkosFFT {
namespace Impl {
// Helper to convert the integer type of vectors
template <typename InType, typename OutType>
auto convert_int_type(std::vector<InType>& in) -> std::vector<OutType> {
  std::vector<OutType> out(in.size());
  std::transform(
      in.begin(), in.end(), out.begin(),
      [](const InType v) -> OutType { return static_cast<OutType>(v); });

  return out;
}

// Helper to compute strides from extents
// (n0, n1) -> (0, n1, 1)
// (n0) -> (0, 1)
template <typename InType, typename OutType>
auto compute_strides(std::vector<InType>& extents) -> std::vector<OutType> {
  std::vector<OutType> out;

  OutType stride = 1;
  for (auto it = extents.rbegin(); it != extents.rend(); ++it) {
    out.push_back(stride);
    stride *= static_cast<OutType>(*it);
  }
  out.push_back(0);
  std::reverse(out.begin(), out.end());

  return out;
}

// batched transform, over ND Views
template <
    typename ExecutionSpace, typename PlanType, typename InViewType,
    typename OutViewType, typename BufferViewType, typename InfoType,
    std::size_t fft_rank             = 1,
    std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL>,
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

  // For the moment, considering the contiguous layout only
  auto sign = KokkosFFT::Impl::direction_type<ExecutionSpace>(direction);

  // Create plan
  auto in_strides   = compute_strides<int, std::int64_t>(in_extents);
  auto out_strides  = compute_strides<int, std::int64_t>(out_extents);
  auto _fft_extents = convert_int_type<int, std::int64_t>(fft_extents);

  // In oneMKL, the distance is always defined based on R2C transform
  std::int64_t _idist = static_cast<std::int64_t>(std::max(idist, odist));
  std::int64_t _odist = static_cast<std::int64_t>(std::min(idist, odist));

  plan = std::make_unique<PlanType>(_fft_extents);
  plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                  in_strides.data());
  plan->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                  out_strides.data());

  // Configuration for batched plan
  plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, _idist);
  plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, _odist);
  plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                  static_cast<std::int64_t>(howmany));

  // Data layout in conjugate-even domain
  plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
  plan->set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
                  DFTI_COMPLEX_COMPLEX);

  sycl::queue q = exec_space.sycl_queue();
  plan->commit(q);

  return fft_size;
}

template <
    typename ExecutionSpace, typename PlanType,
    std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL>,
                     std::nullptr_t> = nullptr>
void _destroy_plan(std::unique_ptr<PlanType>&) {
  // In oneMKL, plans are destroybed by destructor
}

template <
    typename ExecutionSpace, typename InfoType,
    std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL>,
                     std::nullptr_t> = nullptr>
void _destroy_info(InfoType) {
  // not used, no finalization is required
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif