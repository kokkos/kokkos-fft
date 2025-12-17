// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_SYCL_PLANS_HPP
#define KOKKOSFFT_SYCL_PLANS_HPP

#include <numeric>
#include <algorithm>
#if defined(INTEL_MKL_VERSION) && INTEL_MKL_VERSION >= 20250100
#include <oneapi/mkl/dft.hpp>
#else
#include <oneapi/mkl/dfti.hpp>
#endif
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_SYCL_types.hpp"
#include "KokkosFFT_Extents.hpp"
#include "KokkosFFT_traits.hpp"
#include "KokkosFFT_utils.hpp"

namespace KokkosFFT {
namespace Impl {

template <
    typename ExecutionSpace, typename T,
    std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL>,
                     std::nullptr_t> = nullptr>
void setup() {
  [[maybe_unused]] static bool once = [] {
    if (!(Kokkos::is_initialized() || Kokkos::is_finalized())) {
      Kokkos::abort(
          "Error: KokkosFFT APIs must not be called before initializing "
          "Kokkos.\n");
    }
    if (Kokkos::is_finalized()) {
      Kokkos::abort(
          "Error: KokkosFFT APIs must not be called after finalizing "
          "Kokkos.\n");
    }
    return true;
  }();
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
    typename OutViewType, std::size_t fft_rank = 1,
    std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL>,
                     std::nullptr_t> = nullptr>
auto create_plan(const ExecutionSpace& exec_space,
                 std::unique_ptr<PlanType>& plan, const InViewType& in,
                 const OutViewType& out, Direction direction,
                 axis_type<fft_rank> axes, shape_type<fft_rank> s,
                 bool is_inplace) {
  KOKKOSFFT_STATIC_ASSERT_VIEWS_ARE_OPERATABLE(
      (KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                               OutViewType>),
      "create_plan");
  static_assert(InViewType::rank() >= fft_rank,
                "create_plan: Rank of View must be larger than Rank of FFT.");

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::create_plan[TPL_oneMKL]");
  auto [in_extents, out_extents, fft_extents, howmany] =
      KokkosFFT::Impl::get_extents(in, out, axes, s, is_inplace);

  using index_type = FFTIndexType<Kokkos::Experimental::SYCL>;
  index_type idist = total_size(in_extents);
  index_type odist = total_size(out_extents);

  // Create plan
  auto in_strides  = compute_strides<std::size_t, index_type>(in_extents);
  auto out_strides = compute_strides<std::size_t, index_type>(out_extents);
  auto int64_fft_extents = convert_base_int_type<index_type>(fft_extents);

  // In oneMKL, the distance is always defined based on R2C transform
  // idist is the larger one, and odist is the smaller one
  auto [max_odist, max_idist] = std::minmax({idist, odist});

  plan = std::make_unique<PlanType>(
      int64_fft_extents, in_strides, out_strides, max_idist, max_odist,
      static_cast<index_type>(howmany), direction, is_inplace);
  plan->commit(exec_space);

  return fft_extents;
}

// batched transform, over ND Views
template <
    typename ExecutionSpace, typename PlanType, typename InViewType,
    typename OutViewType,
    std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL>,
                     std::nullptr_t> = nullptr>
auto create_dynplan(const ExecutionSpace& exec_space,
                    std::unique_ptr<PlanType>& plan, const InViewType& in,
                    const OutViewType& out, Direction direction,
                    std::size_t dim, bool is_inplace) {
  KOKKOSFFT_STATIC_ASSERT_VIEWS_ARE_OPERATABLE(
      (KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                               OutViewType>),
      "create_dynplan");
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::create_dynplan[TPL_oneMKL]");
  auto [in_extents, out_extents, fft_extents, howmany] =
      KokkosFFT::Impl::get_extents(in, out, dim, is_inplace);

  using index_type = FFTIndexType<Kokkos::Experimental::SYCL>;
  index_type idist = total_size(in_extents);
  index_type odist = total_size(out_extents);

  // Create plan
  auto in_strides  = compute_strides<std::size_t, index_type>(in_extents);
  auto out_strides = compute_strides<std::size_t, index_type>(out_extents);
  auto int64_fft_extents = convert_base_int_type<index_type>(fft_extents);

  // In oneMKL, the distance is always defined based on R2C transform
  // idist is the larger one, and odist is the smaller one
  auto [max_odist, max_idist] = std::minmax({idist, odist});

  plan = std::make_unique<PlanType>(
      int64_fft_extents, in_strides, out_strides, max_idist, max_odist,
      static_cast<index_type>(howmany), direction, is_inplace);
  plan->use_external_workspace();
  plan->commit(exec_space);

  // We can get the workspace size after commit
  plan->set_workspace_size();

  return fft_extents;
}

}  // namespace Impl
}  // namespace KokkosFFT

#endif
