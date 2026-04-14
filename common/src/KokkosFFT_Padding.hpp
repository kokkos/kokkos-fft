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
#include "KokkosFFT_traits.hpp"

namespace KokkosFFT {
namespace Impl {
/// \brief Crop or pad the input view into the output view by copying the
/// overlapping region (the per-dimension minimum of both extents).
///
/// For every dimension \c i the copied range is
/// <tt>[0, min(in.extent(i), out.extent(i)))</tt>.
/// If \p out is larger than \p in in some dimension the extra elements retain
/// their previous values (typically zero after construction).
///
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam InViewType     Type of the source Kokkos::View
/// \tparam OutViewType    Type of the destination Kokkos::View
/// \tparam Is             Non-type parameter pack of \c std::size_t indices
/// (<tt>0, 1, …, rank-1</tt>), deduced from the \c std::index_sequence tag
/// argument and used to expand per-dimension subview ranges at compile time
///
/// \param[in]     exec_space Kokkos execution space instance
/// \param[in]     in         Source view to copy from
/// \param[in,out] out        Destination view to copy into
/// \param[in]     (unnamed)  \c std::index_sequence<Is...> tag that carries the
/// index pack; pass \c std::make_index_sequence<rank>{}
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

/// \brief Crop or pad the input view into the output view.
///
/// Delegates to \c crop_or_pad_impl by automatically generating the index
/// sequence from the view rank. For every dimension \c i the copied range is
/// <tt>[0, min(in.extent(i), out.extent(i)))</tt>.
///
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam InViewType     Type of the source Kokkos::View
/// \tparam OutViewType    Type of the destination Kokkos::View
///
/// \param[in]     exec_space Kokkos execution space instance
/// \param[in]     in         Source view to copy from
/// \param[in,out] out        Destination view to copy into
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
