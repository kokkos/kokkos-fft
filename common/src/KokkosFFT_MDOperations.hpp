// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_MDOPERATIONS_HPP
#define KOKKOSFFT_MDOPERATIONS_HPP

#include <Kokkos_Core.hpp>
#include "KokkosFFT_Layout.hpp"

namespace KokkosFFT {
namespace Impl {
/// \brief Retrieves the policy for the parallel execution.
/// If the view is 1D, a Kokkos::RangePolicy is used. For higher dimensions up
/// to 6D, a Kokkos::MDRangePolicy is used. For 7D and 8D views, we use 6D
/// MDRangePolicy
///
/// \tparam IndexType The type of index to be used in the policy (e.g., int,
/// std::size_t)
/// \tparam ExecutionSpace The Kokkos execution space to be used for the policy
/// \tparam ViewType The type of the Kokkos view for which the policy is being
/// created
/// \param[in] space The Kokkos execution space used to launch the parallel
/// reduction.
/// \param[in] x The Kokkos view to be used for determining the policy.
/// \return A Kokkos execution policy (either RangePolicy or MDRangePolicy) to
/// loop over the elements of the view (up to 6D).
template <typename IndexType, typename ExecutionSpace, typename ViewType>
auto get_mdpolicy(const ExecutionSpace& space, const ViewType& x) {
  constexpr std::size_t rank_truncated =
      std::min(ViewType::rank(), std::size_t(6));
  if constexpr (ViewType::rank() == 1) {
    using range_policy_type =
        Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<IndexType>>;
    return range_policy_type(space, 0, x.extent(0));
  } else {
    using LayoutType = typename ViewType::array_layout;
    static const Kokkos::Iterate outer_iteration_pattern =
        layout_iterate_type_selector<LayoutType>::outer_iteration_pattern;
    static const Kokkos::Iterate inner_iteration_pattern =
        layout_iterate_type_selector<LayoutType>::inner_iteration_pattern;
    using iterate_type = Kokkos::Rank<rank_truncated, outer_iteration_pattern,
                                      inner_iteration_pattern>;
    using mdrange_policy_type =
        Kokkos::MDRangePolicy<ExecutionSpace, iterate_type,
                              Kokkos::IndexType<IndexType>>;
    Kokkos::Array<std::size_t, rank_truncated> begins{}, ends{};
    for (std::size_t i = 0; i < rank_truncated; ++i) {
      ends[i] = x.extent(i);
    }
    return mdrange_policy_type(space, begins, ends);
  }
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
