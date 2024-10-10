// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_TRANSPOSE_HPP
#define KOKKOSFFT_TRANSPOSE_HPP

#include <numeric>
#include <tuple>
#include "KokkosFFT_common_types.hpp"
#include "KokkosFFT_utils.hpp"

namespace KokkosFFT {
namespace Impl {
template <typename ViewType, std::size_t DIM>
auto get_map_axes(const ViewType& view, axis_type<DIM> axes) {
  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(view, axes),
                     "get_map_axes: input axes are not valid for the view");

  // Convert the input axes to be in the range of [0, rank-1]
  axis_type<DIM> non_negative_axes = {};
  for (std::size_t i = 0; i < DIM; i++) {
    non_negative_axes.at(i) =
        KokkosFFT::Impl::convert_negative_axis(view, axes.at(i));
  }

  // how indices are map
  // For 5D View and axes are (2,3), map would be (0, 1, 4, 2, 3)
  constexpr int rank = static_cast<int>(ViewType::rank());
  std::vector<int> map;
  map.reserve(rank);

  if (std::is_same_v<typename ViewType::array_layout, Kokkos::LayoutRight>) {
    // Stack axes not specified by axes (0, 1, 4)
    for (int i = 0; i < rank; i++) {
      if (!is_found(non_negative_axes, i)) {
        map.push_back(i);
      }
    }

    // Stack axes on the map (For layout Right)
    // Then stack (2, 3) to have (0, 1, 4, 2, 3)
    for (auto axis : non_negative_axes) {
      map.push_back(axis);
    }
  } else {
    // For layout Left, stack innermost axes first
    std::reverse(non_negative_axes.begin(), non_negative_axes.end());
    for (auto axis : non_negative_axes) {
      map.push_back(axis);
    }

    // Then stack remaining axes
    for (int i = 0; i < rank; i++) {
      if (!is_found(non_negative_axes, i)) {
        map.push_back(i);
      }
    }
  }

  using full_axis_type     = axis_type<rank>;
  full_axis_type array_map = {}, array_map_inv = {};
  std::copy_n(map.begin(), rank, array_map.begin());

  // Construct inverse map
  for (int i = 0; i < rank; i++) {
    array_map_inv.at(i) = get_index(array_map, i);
  }

  return std::tuple<full_axis_type, full_axis_type>({array_map, array_map_inv});
}

template <typename ViewType>
auto get_map_axes(const ViewType& view, int axis) {
  return get_map_axes(view, axis_type<1>({axis}));
}

template <class ViewType>
axis_type<ViewType::rank()> compute_transpose_extents(
    ViewType const& view, axis_type<ViewType::rank()> const& map) {
  static_assert(Kokkos::is_view_v<ViewType>,
                "compute_transpose_extents: ViewType must be a Kokkos::View.");
  constexpr std::size_t rank = ViewType::rank();

  axis_type<rank> out_extents;
  for (std::size_t i = 0; i < rank; ++i) {
    out_extents.at(i) = view.extent(map.at(i));
  }

  return out_extents;
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void transpose_impl(const ExecutionSpace& exec_space, const InViewType& in,
                    const OutViewType& out, axis_type<2> /*_map*/) {
  constexpr std::size_t DIM = 2;

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default> >;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  int n0 = in.extent(0), n1 = in.extent(1);

  range_type range(
      exec_space, point_type{{0, 0}}, point_type{{n0, n1}}, tile_type{{4, 4}}
      // [TO DO] Choose optimal tile sizes for each device
  );

  Kokkos::parallel_for(
      "KokkosFFT::transpose", range,
      KOKKOS_LAMBDA(int i0, int i1) { out(i1, i0) = in(i0, i1); });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void transpose_impl(const ExecutionSpace& exec_space, const InViewType& in,
                    const OutViewType& out, axis_type<3> _map) {
  constexpr std::size_t DIM  = 3;
  constexpr std::size_t rank = InViewType::rank();

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default> >;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  int n0 = in.extent(0), n1 = in.extent(1), n2 = in.extent(2);

  range_type range(
      exec_space, point_type{{0, 0, 0}}, point_type{{n0, n1, n2}},
      tile_type{{4, 4, 4}}  // [TO DO] Choose optimal tile sizes for each device
  );

  Kokkos::Array<int, 3> map = {_map[0], _map[1], _map[2]};
  Kokkos::parallel_for(
      "KokkosFFT::transpose", range, KOKKOS_LAMBDA(int i0, int i1, int i2) {
        int _indices[rank] = {i0, i1, i2};
        int _i0            = _indices[map[0]];
        int _i1            = _indices[map[1]];
        int _i2            = _indices[map[2]];

        out(_i0, _i1, _i2) = in(i0, i1, i2);
      });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void transpose_impl(const ExecutionSpace& exec_space, const InViewType& in,
                    const OutViewType& out, axis_type<4> _map) {
  constexpr std::size_t DIM  = 4;
  constexpr std::size_t rank = InViewType::rank();

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default> >;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  int n0 = in.extent(0), n1 = in.extent(1), n2 = in.extent(2),
      n3 = in.extent(3);

  range_type range(exec_space, point_type{{0, 0, 0, 0}},
                   point_type{{n0, n1, n2, n3}}, tile_type{{4, 4, 4, 4}}
                   // [TO DO] Choose optimal tile sizes for each device
  );

  Kokkos::Array<int, rank> map = {_map[0], _map[1], _map[2], _map[3]};
  Kokkos::parallel_for(
      "KokkosFFT::transpose", range,
      KOKKOS_LAMBDA(int i0, int i1, int i2, int i3) {
        int _indices[rank] = {i0, i1, i2, i3};
        int _i0            = _indices[map[0]];
        int _i1            = _indices[map[1]];
        int _i2            = _indices[map[2]];
        int _i3            = _indices[map[3]];

        out(_i0, _i1, _i2, _i3) = in(i0, i1, i2, i3);
      });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void transpose_impl(const ExecutionSpace& exec_space, const InViewType& in,
                    const OutViewType& out, axis_type<5> _map) {
  constexpr std::size_t DIM  = 5;
  constexpr std::size_t rank = InViewType::rank();

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default> >;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  int n0 = in.extent(0), n1 = in.extent(1), n2 = in.extent(2),
      n3 = in.extent(3);
  int n4 = in.extent(4);

  range_type range(exec_space, point_type{{0, 0, 0, 0, 0}},
                   point_type{{n0, n1, n2, n3, n4}}, tile_type{{4, 4, 4, 4, 1}}
                   // [TO DO] Choose optimal tile sizes for each device
  );

  Kokkos::Array<int, rank> map = {_map[0], _map[1], _map[2], _map[3], _map[4]};
  Kokkos::parallel_for(
      "KokkosFFT::transpose", range,
      KOKKOS_LAMBDA(int i0, int i1, int i2, int i3, int i4) {
        int _indices[rank] = {i0, i1, i2, i3, i4};
        int _i0            = _indices[map[0]];
        int _i1            = _indices[map[1]];
        int _i2            = _indices[map[2]];
        int _i3            = _indices[map[3]];
        int _i4            = _indices[map[4]];

        out(_i0, _i1, _i2, _i3, _i4) = in(i0, i1, i2, i3, i4);
      });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void transpose_impl(const ExecutionSpace& exec_space, const InViewType& in,
                    const OutViewType& out, axis_type<6> _map) {
  constexpr std::size_t DIM  = 6;
  constexpr std::size_t rank = InViewType::rank();

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default> >;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  int n0 = in.extent(0), n1 = in.extent(1), n2 = in.extent(2),
      n3 = in.extent(3);
  int n4 = in.extent(4), n5 = in.extent(5);

  range_type range(exec_space, point_type{{0, 0, 0, 0, 0, 0}},
                   point_type{{n0, n1, n2, n3, n4, n5}},
                   tile_type{{4, 4, 4, 4, 1, 1}}
                   // [TO DO] Choose optimal tile sizes for each device
  );

  Kokkos::Array<int, rank> map = {_map[0], _map[1], _map[2],
                                  _map[3], _map[4], _map[5]};
  Kokkos::parallel_for(
      "KokkosFFT::transpose", range,
      KOKKOS_LAMBDA(int i0, int i1, int i2, int i3, int i4, int i5) {
        int _indices[rank] = {i0, i1, i2, i3, i4, i5};
        int _i0            = _indices[map[0]];
        int _i1            = _indices[map[1]];
        int _i2            = _indices[map[2]];
        int _i3            = _indices[map[3]];
        int _i4            = _indices[map[4]];
        int _i5            = _indices[map[5]];

        out(_i0, _i1, _i2, _i3, _i4, _i5) = in(i0, i1, i2, i3, i4, i5);
      });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void transpose_impl(const ExecutionSpace& exec_space, const InViewType& in,
                    const OutViewType& out, axis_type<7> _map) {
  constexpr std::size_t DIM  = 6;
  constexpr std::size_t rank = InViewType::rank();

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default> >;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  int n0 = in.extent(0), n1 = in.extent(1), n2 = in.extent(2),
      n3 = in.extent(3);
  int n4 = in.extent(4), n5 = in.extent(5), n6 = in.extent(6);

  range_type range(exec_space, point_type{{0, 0, 0, 0, 0, 0}},
                   point_type{{n0, n1, n2, n3, n4, n5}},
                   tile_type{{4, 4, 4, 4, 1, 1}}
                   // [TO DO] Choose optimal tile sizes for each device
  );

  Kokkos::Array<int, rank> map = {_map[0], _map[1], _map[2], _map[3],
                                  _map[4], _map[5], _map[6]};
  Kokkos::parallel_for(
      "KokkosFFT::transpose", range,
      KOKKOS_LAMBDA(int i0, int i1, int i2, int i3, int i4, int i5) {
        for (int i6 = 0; i6 < n6; i6++) {
          int _indices[rank] = {i0, i1, i2, i3, i4, i5, i6};
          int _i0            = _indices[map[0]];
          int _i1            = _indices[map[1]];
          int _i2            = _indices[map[2]];
          int _i3            = _indices[map[3]];
          int _i4            = _indices[map[4]];
          int _i5            = _indices[map[5]];
          int _i6            = _indices[map[6]];

          out(_i0, _i1, _i2, _i3, _i4, _i5, _i6) =
              in(i0, i1, i2, i3, i4, i5, i6);
        }
      });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void transpose_impl(const ExecutionSpace& exec_space, const InViewType& in,
                    const OutViewType& out, axis_type<8> _map) {
  constexpr std::size_t DIM = 6;

  constexpr std::size_t rank = InViewType::rank();

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default> >;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  int n0 = in.extent(0), n1 = in.extent(1), n2 = in.extent(2),
      n3 = in.extent(3);
  int n4 = in.extent(4), n5 = in.extent(5), n6 = in.extent(6),
      n7 = in.extent(7);

  range_type range(exec_space, point_type{{0, 0, 0, 0, 0, 0}},
                   point_type{{n0, n1, n2, n3, n4, n5}},
                   tile_type{{4, 4, 4, 4, 1, 1}}
                   // [TO DO] Choose optimal tile sizes for each device
  );

  Kokkos::Array<int, rank> map = {_map[0], _map[1], _map[2], _map[3],
                                  _map[4], _map[5], _map[6], _map[7]};
  Kokkos::parallel_for(
      "KokkosFFT::transpose", range,
      KOKKOS_LAMBDA(int i0, int i1, int i2, int i3, int i4, int i5) {
        for (int i6 = 0; i6 < n6; i6++) {
          for (int i7 = 0; i7 < n7; i7++) {
            int _indices[rank] = {i0, i1, i2, i3, i4, i5, i6, i7};
            int _i0            = _indices[map[0]];
            int _i1            = _indices[map[1]];
            int _i2            = _indices[map[2]];
            int _i3            = _indices[map[3]];
            int _i4            = _indices[map[4]];
            int _i5            = _indices[map[5]];
            int _i6            = _indices[map[6]];
            int _i7            = _indices[map[7]];

            out(_i0, _i1, _i2, _i3, _i4, _i5, _i6, _i7) =
                in(i0, i1, i2, i3, i4, i5, i6, i7);
          }
        }
      });
}

/* Make the axis direction to the inner most direction
   axis should be the range in [-(rank-1), rank-1], where
   negative number is interpreted as rank + axis.
   E.g. axis = -1 for rank 3 matrix is interpreted as axis = 2

 * E.g.
        Layout Left
        A (3, 4, 2) and axis = 1 -> A' (4, 3, 2)
        B (2, 4, 3, 5) and axis = 2 -> B' (3, 2, 4, 5)
        C (8, 6, 3) and axis = 0 -> C' (8, 6, 3)
        D (7, 5) and axis = -1 -> D' (5, 7)

        Layout Right
        A (3, 4, 2) and axis = 1 -> A' (3, 2, 4)
        B (2, 4, 3, 5) and axis = 2 -> B' (2, 4, 5, 3)
        C (8, 6, 3) and axis = 0 -> C' (6, 3, 8)
        D (5, 7) and axis = -1 -> D' (5, 7)
 *
*/
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
void transpose(const ExecutionSpace& exec_space, const InViewType& in,
               const OutViewType& out, axis_type<DIM> map) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "transpose: InViewType and OutViewType must have the same base floating "
      "point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");

  static_assert(InViewType::rank() == DIM,
                "transpose: Rank of View must be equal to Rank of "
                "transpose axes.");

  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::is_transpose_needed(map),
                     "transpose: transpose not necessary");

  // in order not to call transpose_impl for 1D case
  if constexpr (DIM > 1) {
    transpose_impl(exec_space, in, out, map);
  }
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
