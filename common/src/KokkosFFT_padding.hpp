// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
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
  static_assert(InViewType::rank() >= DIM,
                "get_modified_shape: Rank of Input View must be larger "
                "than or equal to the Rank of new shape");
  static_assert(OutViewType::rank() >= DIM,
                "get_modified_shape: Rank of Output View must be larger "
                "than or equal to the Rank of new shape");
  static_assert(DIM > 0,
                "get_modified_shape: Rank of FFT axes must be "
                "larger than or equal to 1");
  constexpr int rank = static_cast<int>(InViewType::rank());

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

  // Assert if the elements are overlapped
  assert(!KokkosFFT::Impl::has_duplicate_values(positive_axes));
  assert(!KokkosFFT::Impl::is_out_of_range_value_included(positive_axes, rank));

  using full_shape_type = shape_type<rank>;
  full_shape_type modified_shape;
  for (int i = 0; i < rank; i++) {
    modified_shape.at(i) = in.extent(i);
  }

  // Update shapes based on newly given shape
  for (int i = 0; i < DIM; i++) {
    int positive_axis = positive_axes.at(i);
    assert(shape.at(i) > 0);
    modified_shape.at(positive_axis) = shape.at(i);
  }

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  bool is_C2R = is_complex<in_value_type>::value &&
                std::is_floating_point_v<out_value_type>;

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

  // [TO DO] Add a is_C2R arg. If is_C2R is true, then shape should be shape/2+1
  constexpr int rank = static_cast<int>(ViewType::rank());

  bool not_same = false;
  for (int i = 0; i < rank; i++) {
    if (modified_shape.at(i) != view.extent(i)) {
      not_same = true;
      break;
    }
  }

  return not_same;
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void _crop_or_pad(const ExecutionSpace& exec_space, const InViewType& in,
                  OutViewType& out, shape_type<1> s) {
  auto _n0 = s.at(0);
  out      = OutViewType("out", _n0);

  auto n0 = std::min(_n0, in.extent(0));

  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>(
          exec_space, 0, n0),
      KOKKOS_LAMBDA(int i0) { out(i0) = in(i0); });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void _crop_or_pad(const ExecutionSpace& exec_space, const InViewType& in,
                  OutViewType& out, shape_type<2> s) {
  constexpr std::size_t DIM = 2;

  auto [_n0, _n1] = s;
  out             = OutViewType("out", _n0, _n1);

  int n0 = std::min(_n0, in.extent(0));
  int n1 = std::min(_n1, in.extent(1));

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default>>;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  range_type range(point_type{{0, 0}}, point_type{{n0, n1}}, tile_type{{4, 4}}
                   // [TO DO] Choose optimal tile sizes for each device
  );

  Kokkos::parallel_for(
      range, KOKKOS_LAMBDA(int i0, int i1) { out(i0, i1) = in(i0, i1); });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void _crop_or_pad(const ExecutionSpace& exec_space, const InViewType& in,
                  OutViewType& out, shape_type<3> s) {
  constexpr std::size_t DIM = 3;

  auto [_n0, _n1, _n2] = s;
  out                  = OutViewType("out", _n0, _n1, _n2);

  int n0 = std::min(_n0, in.extent(0));
  int n1 = std::min(_n1, in.extent(1));
  int n2 = std::min(_n2, in.extent(2));

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default>>;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  range_type range(
      point_type{{0, 0, 0}}, point_type{{n0, n1, n2}}, tile_type{{4, 4, 4}}
      // [TO DO] Choose optimal tile sizes for each device
  );

  Kokkos::parallel_for(
      range, KOKKOS_LAMBDA(int i0, int i1, int i2) {
        out(i0, i1, i2) = in(i0, i1, i2);
      });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void _crop_or_pad(const ExecutionSpace& exec_space, const InViewType& in,
                  OutViewType& out, shape_type<4> s) {
  constexpr std::size_t DIM = 4;

  auto [_n0, _n1, _n2, _n3] = s;
  out                       = OutViewType("out", _n0, _n1, _n2, _n3);

  int n0 = std::min(_n0, in.extent(0));
  int n1 = std::min(_n1, in.extent(1));
  int n2 = std::min(_n2, in.extent(2));
  int n3 = std::min(_n3, in.extent(3));

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default>>;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  range_type range(point_type{{0, 0, 0, 0}}, point_type{{n0, n1, n2, n3}},
                   tile_type{{4, 4, 4, 4}}
                   // [TO DO] Choose optimal tile sizes for each device
  );

  Kokkos::parallel_for(
      range, KOKKOS_LAMBDA(int i0, int i1, int i2, int i3) {
        out(i0, i1, i2, i3) = in(i0, i1, i2, i3);
      });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void _crop_or_pad(const ExecutionSpace& exec_space, const InViewType& in,
                  OutViewType& out, shape_type<5> s) {
  constexpr std::size_t DIM = 5;

  auto [_n0, _n1, _n2, _n3, _n4] = s;
  out                            = OutViewType("out", _n0, _n1, _n2, _n3, _n4);

  int n0 = std::min(_n0, in.extent(0));
  int n1 = std::min(_n1, in.extent(1));
  int n2 = std::min(_n2, in.extent(2));
  int n3 = std::min(_n3, in.extent(3));
  int n4 = std::min(_n4, in.extent(4));

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default>>;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  range_type range(point_type{{0, 0, 0, 0, 0}},
                   point_type{{n0, n1, n2, n3, n4}}, tile_type{{4, 4, 4, 4, 1}}
                   // [TO DO] Choose optimal tile sizes for each device
  );

  Kokkos::parallel_for(
      range, KOKKOS_LAMBDA(int i0, int i1, int i2, int i3, int i4) {
        out(i0, i1, i2, i3, i4) = in(i0, i1, i2, i3, i4);
      });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void _crop_or_pad(const ExecutionSpace& exec_space, const InViewType& in,
                  OutViewType& out, shape_type<6> s) {
  constexpr std::size_t DIM = 6;

  auto [_n0, _n1, _n2, _n3, _n4, _n5] = s;
  out = OutViewType("out", _n0, _n1, _n2, _n3, _n4, _n5);

  int n0 = std::min(_n0, in.extent(0));
  int n1 = std::min(_n1, in.extent(1));
  int n2 = std::min(_n2, in.extent(2));
  int n3 = std::min(_n3, in.extent(3));
  int n4 = std::min(_n4, in.extent(4));
  int n5 = std::min(_n5, in.extent(5));

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default>>;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  range_type range(point_type{{0, 0, 0, 0, 0, 0}},
                   point_type{{n0, n1, n2, n3, n4, n5}},
                   tile_type{{4, 4, 4, 4, 1, 1}}
                   // [TO DO] Choose optimal tile sizes for each device
  );

  Kokkos::parallel_for(
      range, KOKKOS_LAMBDA(int i0, int i1, int i2, int i3, int i4, int i5) {
        out(i0, i1, i2, i3, i4, i5) = in(i0, i1, i2, i3, i4, i5);
      });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void _crop_or_pad(const ExecutionSpace& exec_space, const InViewType& in,
                  OutViewType& out, shape_type<7> s) {
  constexpr std::size_t DIM = 6;

  auto [_n0, _n1, _n2, _n3, _n4, _n5, _n6] = s;
  out = OutViewType("out", _n0, _n1, _n2, _n3, _n4, _n5, _n6);

  int n0 = std::min(_n0, in.extent(0));
  int n1 = std::min(_n1, in.extent(1));
  int n2 = std::min(_n2, in.extent(2));
  int n3 = std::min(_n3, in.extent(3));
  int n4 = std::min(_n4, in.extent(4));
  int n5 = std::min(_n5, in.extent(5));
  int n6 = std::min(_n6, in.extent(6));

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default>>;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  range_type range(point_type{{0, 0, 0, 0, 0, 0}},
                   point_type{{n0, n1, n2, n3, n4, n5}},
                   tile_type{{4, 4, 4, 4, 1, 1}}
                   // [TO DO] Choose optimal tile sizes for each device
  );

  Kokkos::parallel_for(
      range, KOKKOS_LAMBDA(int i0, int i1, int i2, int i3, int i4, int i5) {
        for (int i6 = 0; i6 < n6; i6++) {
          out(i0, i1, i2, i3, i4, i5, i6) = in(i0, i1, i2, i3, i4, i5, i6);
        }
      });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void _crop_or_pad(const ExecutionSpace& exec_space, const InViewType& in,
                  OutViewType& out, shape_type<8> s) {
  constexpr std::size_t DIM = 6;

  auto [_n0, _n1, _n2, _n3, _n4, _n5, _n6, _n7] = s;
  out = OutViewType("out", _n0, _n1, _n2, _n3, _n4, _n5, _n6, _n7);

  int n0 = std::min(_n0, in.extent(0));
  int n1 = std::min(_n1, in.extent(1));
  int n2 = std::min(_n2, in.extent(2));
  int n3 = std::min(_n3, in.extent(3));
  int n4 = std::min(_n4, in.extent(4));
  int n5 = std::min(_n5, in.extent(5));
  int n6 = std::min(_n6, in.extent(6));
  int n7 = std::min(_n7, in.extent(7));

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default>>;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  range_type range(point_type{{0, 0, 0, 0, 0, 0}},
                   point_type{{n0, n1, n2, n3, n4, n5}},
                   tile_type{{4, 4, 4, 4, 1, 1}}
                   // [TO DO] Choose optimal tile sizes for each device
  );

  Kokkos::parallel_for(
      range, KOKKOS_LAMBDA(int i0, int i1, int i2, int i3, int i4, int i5) {
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
  static_assert(InViewType::rank() == DIM,
                "crop_or_pad: Rank of View must be equal to Rank "
                "of extended shape.");
  static_assert(OutViewType::rank() == DIM,
                "crop_or_pad: Rank of View must be equal to Rank "
                "of extended shape.");
  _crop_or_pad(exec_space, in, out, s);
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
