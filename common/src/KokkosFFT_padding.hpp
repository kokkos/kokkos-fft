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

/// \brief Return a new shape of the input view based on the
/// specified input shape and axes.
///
/// \tparam InViewType The input view type
/// \tparam OutViewType The output view type
/// \tparam DIM         The dimensionality of the shape and axes
///
/// \param in [in] Input view from which to derive the new shape
/// \param out [in] Output view (unused but necessary for type deduction)
/// \param shape [in] The new shape of the input view. If the shape is zero,
/// no modifications are made.
/// \param axes [in] Axes over which the shape modification is applied.
template <typename InViewType, typename OutViewType, std::size_t DIM>
auto get_modified_shape(const InViewType in, const OutViewType /* out */,
                        shape_type<DIM> shape, axis_type<DIM> axes) {
  static_assert(
      KokkosFFT::Impl::have_same_rank_v<InViewType, OutViewType>,
      "get_modified_shape: Input View and Output View must have the same rank");
  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, axes),
                     "input axes are not valid for the view");

  shape_type<DIM> zeros = {};  // default shape means no crop or pad
  if (shape == zeros) {
    return KokkosFFT::Impl::extract_extents(in);
  }

  // Convert the input axes to be in the range of [0, rank-1]
  constexpr std::size_t rank = InViewType::rank();
  auto non_negative_axes     = convert_negative_axes<int, DIM, rank>(axes);

  using full_shape_type = shape_type<rank>;
  full_shape_type modified_shape;
  for (std::size_t i = 0; i < rank; i++) {
    modified_shape.at(i) = in.extent(i);
  }

  // Update shapes based on newly given shape
  for (std::size_t i = 0; i < DIM; i++) {
    auto non_negative_axis = non_negative_axes.at(i);
    KOKKOSFFT_THROW_IF(shape.at(i) <= 0,
                       "get_modified_shape: shape must be greater than 0");
    modified_shape.at(non_negative_axis) = shape.at(i);
  }

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  bool is_C2R = is_complex_v<in_value_type> && is_real_v<out_value_type>;

  if (is_C2R) {
    auto reshaped_axis               = non_negative_axes.back();
    modified_shape.at(reshaped_axis) = modified_shape.at(reshaped_axis) / 2 + 1;
  }

  return modified_shape;
}

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
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "crop_or_pad: InViewType and OutViewType must have the same base "
      "floating point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");
  crop_or_pad_impl(exec_space, in, out,
                   std::make_index_sequence<InViewType::rank()>{});
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
