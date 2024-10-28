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

  shape_type<DIM> zeros = {0};  // default shape means no crop or pad
  if (shape == zeros) {
    return KokkosFFT::Impl::extract_extents(in);
  }

  // Convert the input axes to be in the range of [0, rank-1]
  std::vector<int> positive_axes;
  for (std::size_t i = 0; i < DIM; i++) {
    int axis = KokkosFFT::Impl::convert_negative_axis(in, axes.at(i));
    positive_axes.push_back(axis);
  }

  constexpr int rank    = static_cast<int>(InViewType::rank());
  using full_shape_type = shape_type<rank>;
  full_shape_type modified_shape;
  for (int i = 0; i < rank; i++) {
    modified_shape.at(i) = in.extent(i);
  }

  // Update shapes based on newly given shape
  for (int i = 0; i < static_cast<int>(DIM); i++) {
    int positive_axis = positive_axes.at(i);
    assert(shape.at(i) > 0);
    modified_shape.at(positive_axis) = shape.at(i);
  }

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  bool is_C2R = is_complex_v<in_value_type> && is_real_v<out_value_type>;

  if (is_C2R) {
    int reshaped_axis                = positive_axes.back();
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

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void crop_or_pad_impl(const ExecutionSpace& exec_space, const InViewType& in,
                      OutViewType& out, shape_type<1> s) {
  auto s0 = s.at(0);
  out     = OutViewType("out", s0);

  auto n0 = std::min(s0, in.extent(0));

  Kokkos::parallel_for(
      "KokkosFFT::crop_or_pad",
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>(
          exec_space, 0, n0),
      KOKKOS_LAMBDA(std::size_t i0) { out(i0) = in(i0); });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void crop_or_pad_impl(const ExecutionSpace& exec_space, const InViewType& in,
                      OutViewType& out, shape_type<2> s) {
  constexpr std::size_t DIM = 2;

  auto [s0, s1] = s;
  out           = OutViewType("out", s0, s1);

  int n0 = std::min(s0, in.extent(0));
  int n1 = std::min(s1, in.extent(1));

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default>>;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  range_type range(
      exec_space, point_type{{0, 0}}, point_type{{n0, n1}}, tile_type{{4, 4}}
      // [TO DO] Choose optimal tile sizes for each device
  );

  Kokkos::parallel_for(
      "KokkosFFT::crop_or_pad", range,
      KOKKOS_LAMBDA(int i0, int i1) { out(i0, i1) = in(i0, i1); });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void crop_or_pad_impl(const ExecutionSpace& exec_space, const InViewType& in,
                      OutViewType& out, shape_type<3> s) {
  constexpr std::size_t DIM = 3;

  auto [s0, s1, s2] = s;
  out               = OutViewType("out", s0, s1, s2);

  int n0 = std::min(s0, in.extent(0));
  int n1 = std::min(s1, in.extent(1));
  int n2 = std::min(s2, in.extent(2));

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default>>;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  range_type range(
      exec_space, point_type{{0, 0, 0}}, point_type{{n0, n1, n2}},
      tile_type{{4, 4, 4}}  // [TO DO] Choose optimal tile sizes for each device
  );

  Kokkos::parallel_for(
      "KokkosFFT::crop_or_pad", range, KOKKOS_LAMBDA(int i0, int i1, int i2) {
        out(i0, i1, i2) = in(i0, i1, i2);
      });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void crop_or_pad_impl(const ExecutionSpace& exec_space, const InViewType& in,
                      OutViewType& out, shape_type<4> s) {
  constexpr std::size_t DIM = 4;

  auto [s0, s1, s2, s3] = s;
  out                   = OutViewType("out", s0, s1, s2, s3);

  int n0 = std::min(s0, in.extent(0));
  int n1 = std::min(s1, in.extent(1));
  int n2 = std::min(s2, in.extent(2));
  int n3 = std::min(s3, in.extent(3));

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default>>;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  range_type range(exec_space, point_type{{0, 0, 0, 0}},
                   point_type{{n0, n1, n2, n3}}, tile_type{{4, 4, 4, 4}}
                   // [TO DO] Choose optimal tile sizes for each device
  );

  Kokkos::parallel_for(
      "KokkosFFT::crop_or_pad", range,
      KOKKOS_LAMBDA(int i0, int i1, int i2, int i3) {
        out(i0, i1, i2, i3) = in(i0, i1, i2, i3);
      });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void crop_or_pad_impl(const ExecutionSpace& exec_space, const InViewType& in,
                      OutViewType& out, shape_type<5> s) {
  constexpr std::size_t DIM = 5;

  auto [s0, s1, s2, s3, s4] = s;
  out                       = OutViewType("out", s0, s1, s2, s3, s4);

  int n0 = std::min(s0, in.extent(0));
  int n1 = std::min(s1, in.extent(1));
  int n2 = std::min(s2, in.extent(2));
  int n3 = std::min(s3, in.extent(3));
  int n4 = std::min(s4, in.extent(4));

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default>>;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  range_type range(exec_space, point_type{{0, 0, 0, 0, 0}},
                   point_type{{n0, n1, n2, n3, n4}}, tile_type{{4, 4, 4, 4, 1}}
                   // [TO DO] Choose optimal tile sizes for each device
  );

  Kokkos::parallel_for(
      "KokkosFFT::crop_or_pad", range,
      KOKKOS_LAMBDA(int i0, int i1, int i2, int i3, int i4) {
        out(i0, i1, i2, i3, i4) = in(i0, i1, i2, i3, i4);
      });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void crop_or_pad_impl(const ExecutionSpace& exec_space, const InViewType& in,
                      OutViewType& out, shape_type<6> s) {
  constexpr std::size_t DIM = 6;

  auto [s0, s1, s2, s3, s4, s5] = s;
  out                           = OutViewType("out", s0, s1, s2, s3, s4, s5);

  int n0 = std::min(s0, in.extent(0));
  int n1 = std::min(s1, in.extent(1));
  int n2 = std::min(s2, in.extent(2));
  int n3 = std::min(s3, in.extent(3));
  int n4 = std::min(s4, in.extent(4));
  int n5 = std::min(s5, in.extent(5));

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default>>;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  range_type range(exec_space, point_type{{0, 0, 0, 0, 0, 0}},
                   point_type{{n0, n1, n2, n3, n4, n5}},
                   tile_type{{4, 4, 4, 4, 1, 1}}
                   // [TO DO] Choose optimal tile sizes for each device
  );

  Kokkos::parallel_for(
      "KokkosFFT::crop_or_pad", range,
      KOKKOS_LAMBDA(int i0, int i1, int i2, int i3, int i4, int i5) {
        out(i0, i1, i2, i3, i4, i5) = in(i0, i1, i2, i3, i4, i5);
      });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void crop_or_pad_impl(const ExecutionSpace& exec_space, const InViewType& in,
                      OutViewType& out, shape_type<7> s) {
  constexpr std::size_t DIM = 6;

  auto [s0, s1, s2, s3, s4, s5, s6] = s;
  out = OutViewType("out", s0, s1, s2, s3, s4, s5, s6);

  int n0 = std::min(s0, in.extent(0));
  int n1 = std::min(s1, in.extent(1));
  int n2 = std::min(s2, in.extent(2));
  int n3 = std::min(s3, in.extent(3));
  int n4 = std::min(s4, in.extent(4));
  int n5 = std::min(s5, in.extent(5));
  int n6 = std::min(s6, in.extent(6));

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default>>;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  range_type range(exec_space, point_type{{0, 0, 0, 0, 0, 0}},
                   point_type{{n0, n1, n2, n3, n4, n5}},
                   tile_type{{4, 4, 4, 4, 1, 1}}
                   // [TO DO] Choose optimal tile sizes for each device
  );

  Kokkos::parallel_for(
      "KokkosFFT::crop_or_pad", range,
      KOKKOS_LAMBDA(int i0, int i1, int i2, int i3, int i4, int i5) {
        for (int i6 = 0; i6 < n6; i6++) {
          out(i0, i1, i2, i3, i4, i5, i6) = in(i0, i1, i2, i3, i4, i5, i6);
        }
      });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void crop_or_pad_impl(const ExecutionSpace& exec_space, const InViewType& in,
                      OutViewType& out, shape_type<8> s) {
  constexpr std::size_t DIM = 6;

  auto [s0, s1, s2, s3, s4, s5, s6, s7] = s;
  out = OutViewType("out", s0, s1, s2, s3, s4, s5, s6, s7);

  int n0 = std::min(s0, in.extent(0));
  int n1 = std::min(s1, in.extent(1));
  int n2 = std::min(s2, in.extent(2));
  int n3 = std::min(s3, in.extent(3));
  int n4 = std::min(s4, in.extent(4));
  int n5 = std::min(s5, in.extent(5));
  int n6 = std::min(s6, in.extent(6));
  int n7 = std::min(s7, in.extent(7));

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default>>;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  range_type range(exec_space, point_type{{0, 0, 0, 0, 0, 0}},
                   point_type{{n0, n1, n2, n3, n4, n5}},
                   tile_type{{4, 4, 4, 4, 1, 1}}
                   // [TO DO] Choose optimal tile sizes for each device
  );

  Kokkos::parallel_for(
      "KokkosFFT::crop_or_pad", range,
      KOKKOS_LAMBDA(int i0, int i1, int i2, int i3, int i4, int i5) {
        for (int i6 = 0; i6 < n6; i6++) {
          for (int i7 = 0; i7 < n7; i7++) {
            out(i0, i1, i2, i3, i4, i5, i6, i7) =
                in(i0, i1, i2, i3, i4, i5, i6, i7);
          }
        }
      });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
void crop_or_pad(const ExecutionSpace& exec_space, const InViewType& in,
                 OutViewType& out, shape_type<DIM> s) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "crop_or_pad: InViewType and OutViewType must have the same base "
      "floating point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");
  crop_or_pad_impl(exec_space, in, out, s);
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
