#ifndef KOKKOSFFT_TRANSPOSE_HPP
#define KOKKOSFFT_TRANSPOSE_HPP

#include <numeric>
#include <tuple>
#include "KokkosFFT_default_types.hpp"
#include "KokkosFFT_utils.hpp"

namespace KokkosFFT {
namespace Impl {
template <typename ViewType, std::size_t DIM>
auto get_map_axes(const ViewType& view, axis_type<DIM> _axes) {
  static_assert(ViewType::rank() >= DIM,
                "KokkosFFT::get_map_axes: Rank of View must be larger thane or "
                "equal to the Rank of FFT axes.");
  static_assert(DIM > 0,
                "KokkosFFT::get_map_axes: Rank of FFT axes must be larger than "
                "or equal to 1.");

  constexpr int rank      = static_cast<int>(ViewType::rank());
  using array_layout_type = typename ViewType::array_layout;

  // Convert the input axes to be in the range of [0, rank-1]
  std::vector<int> axes;
  for (std::size_t i = 0; i < DIM; i++) {
    int axis = KokkosFFT::Impl::convert_negative_axis(view, _axes.at(i));
    axes.push_back(axis);
  }

  // Assert if the elements are overlapped
  assert(!KokkosFFT::Impl::has_duplicate_values(axes));

  // how indices are map
  // For 5D View and axes are (2,3), map would be (0, 1, 4, 2, 3)
  std::vector<int> map, map_inv;
  map.reserve(rank);
  map_inv.reserve(rank);

  if (std::is_same<array_layout_type, Kokkos::LayoutRight>::value) {
    // Stack axes not specified by axes (0, 1, 4)
    for (int i = 0; i < rank; i++) {
      if (!is_found(axes, i)) {
        map.push_back(i);
      }
    }

    // Stack axes on the map (For layout Right)
    // Then stack (2, 3) to have (0, 1, 4, 2, 3)
    for (auto axis : axes) {
      map.push_back(axis);
    }
  } else {
    // For layout Left, stack innermost axes first
    std::reverse(axes.begin(), axes.end());
    for (auto axis : axes) {
      map.push_back(axis);
    }

    // Then stack remaining axes
    for (int i = 0; i < rank; i++) {
      if (!is_found(axes, i)) {
        map.push_back(i);
      }
    }
  }

  // Construct inverse map
  for (int i = 0; i < rank; i++) {
    map_inv.push_back(get_index(map, i));
  }

  using full_axis_type = axis_type<rank>;
  full_axis_type array_map, array_map_inv;
  std::copy(map.begin(), map.end(), array_map.begin());
  std::copy(map_inv.begin(), map_inv.end(), array_map_inv.begin());

  return std::tuple<full_axis_type, full_axis_type>({array_map, array_map_inv});
}

template <typename ViewType>
auto get_map_axes(const ViewType& view, int axis) {
  return get_map_axes(view, axis_type<1>({axis}));
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::enable_if_t<InViewType::rank() == 1, std::nullptr_t> = nullptr>
void _transpose(const ExecutionSpace& exec_space, InViewType& in,
                OutViewType& out, [[maybe_unused]] axis_type<1> _map) {}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void _transpose(const ExecutionSpace& exec_space, InViewType& in,
                OutViewType& out, axis_type<2> _map) {
  constexpr std::size_t DIM = 2;
  static_assert(InViewType::rank() == DIM,
                "KokkosFFT::_transpose: Rank of View must be equal to Rank of "
                "transpose axes.");

  constexpr std::size_t rank = InViewType::rank();
  using array_layout_type    = typename InViewType::array_layout;

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default> >;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  int n0 = in.extent(0), n1 = in.extent(1);

  range_type range(point_type{{0, 0}}, point_type{{n0, n1}}, tile_type{{4, 4}}
                   // [TO DO] Choose optimal tile sizes for each device
  );

  bool is_out_view_ready = true;
  std::array<int, rank> out_extents;
  for (int i = 0; i < rank; i++) {
    out_extents.at(i) = in.extent(_map.at(i));
    if (out_extents.at(i) != out.extent(i)) {
      is_out_view_ready = false;
    }
  }

  if (!is_out_view_ready) {
    auto [_n0, _n1] = out_extents;
    out             = OutViewType("out", _n0, _n1);
  }

  Kokkos::parallel_for(
      range, KOKKOS_LAMBDA(int i0, int i1) { out(i1, i0) = in(i0, i1); });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void _transpose(const ExecutionSpace& exec_space, InViewType& in,
                OutViewType& out, axis_type<3> _map) {
  constexpr std::size_t DIM = 3;
  static_assert(InViewType::rank() == DIM,
                "KokkosFFT::_transpose: Rank of View must be equal to Rank of "
                "transpose axes.");
  constexpr std::size_t rank = InViewType::rank();
  using array_layout_type    = typename InViewType::array_layout;

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default> >;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  int n0 = in.extent(0), n1 = in.extent(1), n2 = in.extent(2);

  range_type range(
      point_type{{0, 0, 0}}, point_type{{n0, n1, n2}}, tile_type{{4, 4, 4}}
      // [TO DO] Choose optimal tile sizes for each device
  );

  // Assign a View if not a shallow copy
  bool is_out_view_ready = true;
  std::array<int, rank> out_extents;
  for (int i = 0; i < rank; i++) {
    out_extents.at(i) = in.extent(_map.at(i));
    if (out_extents.at(i) != out.extent(i)) {
      is_out_view_ready = false;
    }
  }

  if (!is_out_view_ready) {
    auto [_n0, _n1, _n2] = out_extents;
    out                  = OutViewType("out", _n0, _n1, _n2);
  }
  Kokkos::parallel_for(
      range, KOKKOS_LAMBDA(int i0, int i1, int i2) {
        int _i0 = i0, _i1 = i1, _i2 = i2;
        if (_map[0] == 0 && _map[1] == 2 && _map[2] == 1) {
          _i1 = i2;
          _i2 = i1;
        } else if (_map[0] == 1 && _map[1] == 0 && _map[2] == 2) {
          _i0 = i1;
          _i1 = i0;
        } else if (_map[0] == 1 && _map[1] == 2 && _map[2] == 0) {
          _i0 = i1;
          _i1 = i2;
          _i2 = i0;
        } else if (_map[0] == 2 && _map[1] == 1 && _map[2] == 0) {
          _i0 = i2;
          _i2 = i0;
        } else if (_map[0] == 2 && _map[1] == 0 && _map[2] == 1) {
          _i0 = i2;
          _i1 = i0;
          _i2 = i1;
        }
        out(_i0, _i1, _i2) = in(i0, i1, i2);
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
void transpose(const ExecutionSpace& exec_space, InViewType& in,
               OutViewType& out, axis_type<DIM> _map) {
  using in_value_type     = typename InViewType::non_const_value_type;
  using out_value_type    = typename OutViewType::non_const_value_type;
  using array_layout_type = typename InViewType::array_layout;

  static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::transpose: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::transpose: OutViewType is not a Kokkos::View.");

  static_assert(InViewType::rank() == OutViewType::rank(),
                "KokkosFFT::transpose: InViewType and OutViewType must have "
                "the same rank.");

  if (!KokkosFFT::Impl::is_transpose_needed(_map)) {
    throw std::runtime_error("KokkosFFT::transpose: transpose not necessary");
  }

  _transpose(exec_space, in, out, _map);
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif