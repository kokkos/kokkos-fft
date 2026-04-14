// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_PADDING_HPP
#define KOKKOSFFT_PADDING_HPP

#include <algorithm>
#include <array>
#include <utility>
#include <tuple>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_common_types.hpp"
#include "KokkosFFT_asserts.hpp"

namespace KokkosFFT {
namespace Impl {
/// \brief Partially copy the input view to the output view to pad or crop the
/// input view to the output view.
/// \tparam ExecutionSpace The type of Kokkos execution space.
/// \tparam InViewType The input view type
/// \tparam OutViewType The output view type
/// \tparam Is The index sequence for the dimensions of the view
///
/// \param[in] exec_space execution space instance
/// \param[in] in The input view
/// \param[in,out] out The output view
/// \param[in] Is The index sequence for the dimensions of the view
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t... Is>
void crop_or_pad_impl(const ExecutionSpace& exec_space, const InViewType& in,
                      const OutViewType& out, std::index_sequence<Is...>) {
  constexpr std::size_t rank = InViewType::rank();
  using extents_type         = std::array<std::size_t, rank>;

  extents_type extents{};
  for (std::size_t i = 0; i < rank; i++) {
    extents.at(i) = std::min(in.extent(i), out.extent(i));
  }

  auto sub_in = Kokkos::subview(
      in, std::make_pair(std::size_t(0), std::get<Is>(extents))...);
  auto sub_out = Kokkos::subview(
      out, std::make_pair(std::size_t(0), std::get<Is>(extents))...);
  Kokkos::deep_copy(exec_space, sub_out, sub_in);
}

/// \brief Crop or pad the input view to the output view.
/// This function partially copies the input view to the output view to pad or
/// crop the input view to the output view. The extents of the copied region are
/// determined by the minimum of the extents of the input and output views.
/// \tparam ExecutionSpace The type of Kokkos execution space.
/// \tparam InViewType The input view type
/// \tparam OutViewType The output view type
///
/// \param[in] exec_space execution space instance
/// \param[in] in The input view
/// \param[in,out] out The output view
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void crop_or_pad(const ExecutionSpace& exec_space, const InViewType& in,
                 const OutViewType& out) {
  KOKKOSFFT_STATIC_ASSERT_VIEWS_ARE_OPERATABLE(
      (KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                               OutViewType>),
      "crop_or_pad");
  crop_or_pad_impl(exec_space, in, out,
                   std::make_index_sequence<InViewType::rank()>{});
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
