// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_PADDING_HPP
#define KOKKOSFFT_PADDING_HPP

#include <tuple>
#include "KokkosFFT_common_types.hpp"
#include "KokkosFFT_utils.hpp"

namespace KokkosFFT {
namespace Impl {

template <typename ViewType, std::size_t DIM>
auto is_crop_or_pad_needed(const ViewType& view,
                           const shape_type<DIM>& modified_shape) {
  static_assert(ViewType::rank() == DIM,
                "is_crop_or_pad_needed: Rank of View must be equal to Rank "
                "of extended shape.");

  constexpr int rank = static_cast<int>(ViewType::rank());
  bool not_same      = false;
  for (int i = 0; i < rank; i++) {
    if (modified_shape.at(i) != view.extent(i)) {
      not_same = true;
      break;
    }
  }

  return not_same;
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t... Is>
void crop_or_pad_impl(const ExecutionSpace& exec_space, const InViewType& in,
                      const OutViewType& out, std::index_sequence<Is...>) {
  constexpr std::size_t rank = InViewType::rank();
  using extents_type         = std::array<std::size_t, rank>;

  extents_type extents;
  for (std::size_t i = 0; i < rank; i++) {
    extents.at(i) = std::min(in.extent(i), out.extent(i));
  }

  auto sub_in = Kokkos::subview(
      in, std::make_pair(std::size_t(0), std::get<Is>(extents))...);
  auto sub_out = Kokkos::subview(
      out, std::make_pair(std::size_t(0), std::get<Is>(extents))...);
  Kokkos::deep_copy(exec_space, sub_out, sub_in);
}

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
