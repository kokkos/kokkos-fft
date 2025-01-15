// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_ROCM_PLANS_HPP
#define KOKKOSFFT_ROCM_PLANS_HPP

#include <numeric>
#include <algorithm>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_ROCM_types.hpp"
#include "KokkosFFT_Extents.hpp"
#include "KokkosFFT_traits.hpp"
#include "KokkosFFT_asserts.hpp"
#include "KokkosFFT_utils.hpp"

namespace KokkosFFT {
namespace Impl {

template <typename ExecutionSpace, typename T,
          std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
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

    rocfft_status status = rocfft_setup();
    if (status != rocfft_status_success) Kokkos::abort("rocfft_setup failed");

    // Register cleanup function as a hook in Kokkos::finalize
    Kokkos::push_finalize_hook([]() {
      rocfft_status status = rocfft_cleanup();
      if (status != rocfft_status_success)
        Kokkos::abort("rocfft_cleanup failed");
    });
    return true;
  }();
}

// batched transform, over ND Views
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType, std::size_t fft_rank = 1,
          std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                           std::nullptr_t> = nullptr>
auto create_plan(const ExecutionSpace& exec_space,
                 std::unique_ptr<PlanType>& plan, const InViewType& in,
                 const OutViewType& out, Direction direction,
                 axis_type<fft_rank> axes, shape_type<fft_rank> s,
                 bool is_inplace) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "create_plan: InViewType and OutViewType must have the same base "
      "floating point type (float/double), the same layout "
      "(LayoutLeft/LayoutRight), "
      "and the same rank. ExecutionSpace must be accessible to the data in "
      "InViewType and OutViewType.");

  static_assert(InViewType::rank() >= fft_rank,
                "KokkosFFT::create_plan: Rank of View must be larger than "
                "Rank of FFT.");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::create_plan[TPL_rocfft]");
  constexpr auto type =
      KokkosFFT::Impl::transform_type<ExecutionSpace, in_value_type,
                                      out_value_type>::type();
  auto [in_extents, out_extents, fft_extents, howmany] =
      KokkosFFT::Impl::get_extents(in, out, axes, s, is_inplace);

  // Create a plan
  plan = std::make_unique<PlanType>(type, in_extents, out_extents, fft_extents,
                                    howmany, direction, is_inplace);
  plan->commit(exec_space);

  // Calculate the total size of the FFT
  int fft_size = std::accumulate(fft_extents.begin(), fft_extents.end(), 1,
                                 std::multiplies<>());

  return fft_size;
}

}  // namespace Impl
}  // namespace KokkosFFT

#endif
