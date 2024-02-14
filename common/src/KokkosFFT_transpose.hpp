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
                "get_map_axes: Rank of View must be larger thane or "
                "equal to the Rank of FFT axes.");
  static_assert(DIM > 0,
                "get_map_axes: Rank of FFT axes must be larger than "
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

  Kokkos::Array<int, 3> map = {_map[0], _map[1], _map[2]};
  Kokkos::parallel_for(
      range, KOKKOS_LAMBDA(int i0, int i1, int i2) {
        int _indices[rank] = {i0, i1, i2};
        int _i0 = _indices[map[0]];
        int _i1 = _indices[map[1]];
        int _i2 = _indices[map[2]];

        out(_i0, _i1, _i2) = in(i0, i1, i2);
      });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void _transpose(const ExecutionSpace& exec_space, InViewType& in,
                OutViewType& out, axis_type<4> _map) {
  constexpr std::size_t DIM = 4;
  constexpr std::size_t rank = InViewType::rank();
  using array_layout_type    = typename InViewType::array_layout;

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default> >;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  int n0 = in.extent(0), n1 = in.extent(1), n2 = in.extent(2), n3 = in.extent(3);

  range_type range(
      point_type{{0, 0, 0, 0}}, point_type{{n0, n1, n2, n3}}, tile_type{{4, 4, 4, 4}}
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
    auto [_n0, _n1, _n2, _n3] = out_extents;
    out                       = OutViewType("out", _n0, _n1, _n2, _n3);
  }

  Kokkos::Array<int, rank> map = {_map[0], _map[1], _map[2], _map[3]};
  Kokkos::parallel_for(
      range, KOKKOS_LAMBDA(int i0, int i1, int i2, int i3) {
        int _indices[rank] = {i0, i1, i2, i3};
        int _i0 = _indices[map[0]];
        int _i1 = _indices[map[1]];
        int _i2 = _indices[map[2]];
        int _i3 = _indices[map[3]];

        out(_i0, _i1, _i2, _i3) = in(i0, i1, i2, i3);
      });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void _transpose(const ExecutionSpace& exec_space, InViewType& in,
                OutViewType& out, axis_type<5> _map) {
  constexpr std::size_t DIM = 5;
  constexpr std::size_t rank = InViewType::rank();
  using array_layout_type    = typename InViewType::array_layout;

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default> >;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  int n0 = in.extent(0), n1 = in.extent(1), n2 = in.extent(2), n3 = in.extent(3);
  int n4 = in.extent(4);

  range_type range(
      point_type{{0, 0, 0, 0, 0}}, point_type{{n0, n1, n2, n3, n4}}, tile_type{{4, 4, 4, 4, 1}}
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
    auto [_n0, _n1, _n2, _n3, _n4] = out_extents;
    out                            = OutViewType("out", _n0, _n1, _n2, _n3, _n4);
  }

  Kokkos::Array<int, rank> map = {_map[0], _map[1], _map[2], _map[3], _map[4]};
  Kokkos::parallel_for(
      range, KOKKOS_LAMBDA(int i0, int i1, int i2, int i3, int i4) {
        int _indices[rank] = {i0, i1, i2, i3, i4};
        int _i0 = _indices[map[0]];
        int _i1 = _indices[map[1]];
        int _i2 = _indices[map[2]];
        int _i3 = _indices[map[3]];
        int _i4 = _indices[map[4]];

        out(_i0, _i1, _i2, _i3, _i4) = in(i0, i1, i2, i3, i4);
      });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void _transpose(const ExecutionSpace& exec_space, InViewType& in,
                OutViewType& out, axis_type<6> _map) {
  constexpr std::size_t DIM = 6;
  constexpr std::size_t rank = InViewType::rank();
  using array_layout_type    = typename InViewType::array_layout;

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default> >;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  int n0 = in.extent(0), n1 = in.extent(1), n2 = in.extent(2), n3 = in.extent(3);
  int n4 = in.extent(4), n5 = in.extent(5);

  range_type range(
      point_type{{0, 0, 0, 0, 0, 0}}, point_type{{n0, n1, n2, n3, n4, n5}}, tile_type{{4, 4, 4, 4, 1, 1}}
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
    auto [_n0, _n1, _n2, _n3, _n4, _n5] = out_extents;
    out                                 = OutViewType("out", _n0, _n1, _n2, _n3, _n4, _n5);
  }

  Kokkos::Array<int, rank> map = {_map[0], _map[1], _map[2], _map[3], _map[4], _map[5]};
  Kokkos::parallel_for(
      range, KOKKOS_LAMBDA(int i0, int i1, int i2, int i3, int i4, int i5) {
        int _indices[rank] = {i0, i1, i2, i3, i4, i5};
        int _i0 = _indices[map[0]];
        int _i1 = _indices[map[1]];
        int _i2 = _indices[map[2]];
        int _i3 = _indices[map[3]];
        int _i4 = _indices[map[4]];
        int _i5 = _indices[map[5]];

        out(_i0, _i1, _i2, _i3, _i4, _i5) = in(i0, i1, i2, i3, i4, i5);
      });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void _transpose(const ExecutionSpace& exec_space, InViewType& in,
                OutViewType& out, axis_type<7> _map) {
  constexpr std::size_t DIM = 6;
  constexpr std::size_t rank = InViewType::rank();
  using array_layout_type    = typename InViewType::array_layout;

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default> >;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  int n0 = in.extent(0), n1 = in.extent(1), n2 = in.extent(2), n3 = in.extent(3);
  int n4 = in.extent(4), n5 = in.extent(5), n6 = in.extent(6);

  range_type range(
      point_type{{0, 0, 0, 0, 0, 0}}, point_type{{n0, n1, n2, n3, n4, n5}}, tile_type{{4, 4, 4, 4, 1, 1}}
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
    auto [_n0, _n1, _n2, _n3, _n4, _n5, _n6] = out_extents;
    out                                      = OutViewType("out", _n0, _n1, _n2, _n3, _n4, _n5, _n6);
  }

  Kokkos::Array<int, rank> map = {_map[0], _map[1], _map[2], _map[3], _map[4], _map[5], _map[6]};
  Kokkos::parallel_for(
      range, KOKKOS_LAMBDA(int i0, int i1, int i2, int i3, int i4, int i5) {
        for(int i6=0; i6<n6; i6++) {
          int _indices[rank] = {i0, i1, i2, i3, i4, i5, i6};
          int _i0 = _indices[map[0]];
          int _i1 = _indices[map[1]];
          int _i2 = _indices[map[2]];
          int _i3 = _indices[map[3]];
          int _i4 = _indices[map[4]];
          int _i5 = _indices[map[5]];
          int _i6 = _indices[map[6]];

          out(_i0, _i1, _i2, _i3, _i4, _i5, _i6) = in(i0, i1, i2, i3, i4, i5, i6);
        }
      });
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void _transpose(const ExecutionSpace& exec_space, InViewType& in,
                OutViewType& out, axis_type<8> _map) {
  constexpr std::size_t DIM = 6;

  constexpr std::size_t rank = InViewType::rank();
  using array_layout_type    = typename InViewType::array_layout;

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default> >;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  int n0 = in.extent(0), n1 = in.extent(1), n2 = in.extent(2), n3 = in.extent(3);
  int n4 = in.extent(4), n5 = in.extent(5), n6 = in.extent(6), n7 = in.extent(7);

  range_type range(
      point_type{{0, 0, 0, 0, 0, 0}}, point_type{{n0, n1, n2, n3, n4, n5}}, tile_type{{4, 4, 4, 4, 1, 1}}
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
    auto [_n0, _n1, _n2, _n3, _n4, _n5, _n6, _n7] = out_extents;
    out                                           = OutViewType("out", _n0, _n1, _n2, _n3, _n4, _n5, _n6, _n7);
  }

  Kokkos::Array<int, rank> map = {_map[0], _map[1], _map[2], _map[3], _map[4], _map[5], _map[6], _map[7]};
  Kokkos::parallel_for(
      range, KOKKOS_LAMBDA(int i0, int i1, int i2, int i3, int i4, int i5) {
        for(int i6=0; i6<n6; i6++) {
          for(int i7=0; i7<n7; i7++) {
            int _indices[rank] = {i0, i1, i2, i3, i4, i5, i6, i7};
            int _i0 = _indices[map[0]];
            int _i1 = _indices[map[1]];
            int _i2 = _indices[map[2]];
            int _i3 = _indices[map[3]];
            int _i4 = _indices[map[4]];
            int _i5 = _indices[map[5]];
            int _i6 = _indices[map[6]];
            int _i7 = _indices[map[7]];

            out(_i0, _i1, _i2, _i3, _i4, _i5, _i6, _i7) = in(i0, i1, i2, i3, i4, i5, i6, i7);
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
void transpose(const ExecutionSpace& exec_space, InViewType& in,
               OutViewType& out, axis_type<DIM> _map) {
  using in_value_type     = typename InViewType::non_const_value_type;
  using out_value_type    = typename OutViewType::non_const_value_type;
  using array_layout_type = typename InViewType::array_layout;

  static_assert(Kokkos::is_view<InViewType>::value,
                "transpose: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<InViewType>::value,
                "transpose: OutViewType is not a Kokkos::View.");

  static_assert(InViewType::rank() == OutViewType::rank(),
                "transpose: InViewType and OutViewType must have "
                "the same rank.");

  static_assert(InViewType::rank() == DIM,
                "transpose: Rank of View must be equal to Rank of "
                "transpose axes.");

  if (!KokkosFFT::Impl::is_transpose_needed(_map)) {
    throw std::runtime_error("transpose: transpose not necessary");
  }

  _transpose(exec_space, in, out, _map);
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif