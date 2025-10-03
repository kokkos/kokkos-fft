// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_EXTENTS_HPP
#define KOKKOSFFT_EXTENTS_HPP

#include <vector>
#include <tuple>
#include <iostream>
#include <numeric>
#include "KokkosFFT_common_types.hpp"
#include "KokkosFFT_utils.hpp"
#include "KokkosFFT_transpose.hpp"
#include "KokkosFFT_padding.hpp"

namespace KokkosFFT {
namespace Impl {
/// \brief Compute input, output and fft extents required for FFT
/// libraries based on the input view, output view, axes and shape.
/// Extents are converted into Layout Right
///
/// \tparam InViewType The input view type
/// \tparam OutViewType The output view type
/// \tparam DIM         The dimensionality of the axes
///
/// \param in [in] Input view
/// \param out [in] Output view
/// \param axes [in] Axes over which the FFT operations are performed.
/// \param shape [in] The new shape of the input view. If the shape is zero,
/// no modifications are made.
/// \param is_inplace [in] Whether the FFT is inplace or not
template <typename InViewType, typename OutViewType, std::size_t DIM = 1>
auto get_extents(const InViewType& in, const OutViewType& out,
                 axis_type<DIM> axes, shape_type<DIM> shape = {},
                 [[maybe_unused]] bool is_inplace = false) {
  using in_value_type     = typename InViewType::non_const_value_type;
  using out_value_type    = typename OutViewType::non_const_value_type;
  using array_layout_type = typename InViewType::array_layout;

  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, axes),
                     "input axes are not valid for the view");

  constexpr std::size_t rank = InViewType::rank;
  [[maybe_unused]] int inner_most_axis =
      std::is_same_v<array_layout_type, typename Kokkos::LayoutLeft>
          ? 0
          : (rank - 1);

  // index map after transpose over axis
  auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(in, axes);

  // Get new shape based on shape parameter
  auto modified_in_shape =
      KokkosFFT::Impl::get_modified_shape(in, out, shape, axes);

  // Get extents for the inner most axes in LayoutRight
  // If we allow the FFT on the layoutLeft, this part should be modified
  std::vector<int> in_extents_full, out_extents_full, fft_extents_full;
  for (std::size_t i = 0; i < rank; i++) {
    auto idx        = map.at(i);
    auto in_extent  = modified_in_shape.at(idx);
    auto out_extent = out.extent(idx);
    in_extents_full.push_back(in_extent);
    out_extents_full.push_back(out_extent);

    // The extent for transform is always equal to the extent
    // of the extent of real type (R2C or C2R)
    // For C2C, the in and out extents are the same.
    // In the end, we can just use the largest extent among in and out extents.
    auto fft_extent = std::max(in_extent, out_extent);
    fft_extents_full.push_back(fft_extent);
  }

  static_assert(!(is_real_v<in_value_type> && is_real_v<out_value_type>),
                "get_extents: real to real transform is not supported");

  auto mismatched_extents = [&in, &out, &axes]() -> std::string {
    std::string message;
    message += in.label();
    message += "(";
    message += std::to_string(in.extent(0));
    for (std::size_t r = 1; r < rank; r++) {
      message += ",";
      message += std::to_string(in.extent(r));
    }
    message += "), ";
    message += out.label();
    message += "(";
    message += std::to_string(out.extent(0));
    for (std::size_t r = 1; r < rank; r++) {
      message += ",";
      message += std::to_string(out.extent(r));
    }
    message += "), with axes (";
    message += std::to_string(axes.at(0));
    for (std::size_t i = 1; i < axes.size(); i++) {
      message += ",";
      message += std::to_string(axes.at(i));
    }
    message += ")";
    return message;
  };

  for (std::size_t i = 0; i < rank; i++) {
    // The requirement for inner_most_axis is different for transform type
    if (static_cast<int>(i) == inner_most_axis) continue;
    KOKKOSFFT_THROW_IF(in_extents_full.at(i) != out_extents_full.at(i),
                       "input and output extents must be the same except for "
                       "the transform axis: " +
                           mismatched_extents());
  }

  if constexpr (is_complex_v<in_value_type> && is_complex_v<out_value_type>) {
    // Then C2C
    KOKKOSFFT_THROW_IF(
        in_extents_full.at(inner_most_axis) !=
            out_extents_full.at(inner_most_axis),
        "input and output extents must be the same for C2C transform: " +
            mismatched_extents());
  }

  if constexpr (is_real_v<in_value_type>) {
    // Then R2C
    if (is_inplace) {
      in_extents_full.at(inner_most_axis) =
          out_extents_full.at(inner_most_axis) * 2;
    } else {
      KOKKOSFFT_THROW_IF(
          out_extents_full.at(inner_most_axis) !=
              in_extents_full.at(inner_most_axis) / 2 + 1,
          "For R2C, the 'output extent' of transform must be equal to "
          "'input extent'/2 + 1: " +
              mismatched_extents());
    }
  }

  if constexpr (is_real_v<out_value_type>) {
    // Then C2R
    if (is_inplace) {
      out_extents_full.at(inner_most_axis) =
          in_extents_full.at(inner_most_axis) * 2;
    } else {
      KOKKOSFFT_THROW_IF(
          in_extents_full.at(inner_most_axis) !=
              out_extents_full.at(inner_most_axis) / 2 + 1,
          "For C2R, the 'input extent' of transform must be equal to "
          "'output extent' / 2 + 1: " +
              mismatched_extents());
    }
  }

  if constexpr (std::is_same_v<array_layout_type, Kokkos::LayoutLeft>) {
    std::reverse(in_extents_full.begin(), in_extents_full.end());
    std::reverse(out_extents_full.begin(), out_extents_full.end());
    std::reverse(fft_extents_full.begin(), fft_extents_full.end());
  }

  // Define subvectors starting from last - DIM
  // Dimensions relevant to FFTs
  std::vector<int> in_extents(in_extents_full.end() - DIM,
                              in_extents_full.end());
  std::vector<int> out_extents(out_extents_full.end() - DIM,
                               out_extents_full.end());
  std::vector<int> fft_extents(fft_extents_full.end() - DIM,
                               fft_extents_full.end());

  int total_fft_size = total_size(fft_extents_full);
  int fft_size       = total_size(fft_extents);
  int howmany        = total_fft_size / fft_size;

  return std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, int>(
      in_extents, out_extents, fft_extents, howmany);
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
