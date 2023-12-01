#ifndef __KOKKOSFFT_TRANSPOSE_HPP__
#define __KOKKOSFFT_TRANSPOSE_HPP__

#include "KokkosFFT_default_types.hpp"
#include "KokkosFFT_utils.hpp"

namespace KokkosFFT {
  template < std::size_t DIM >
  using MDPolicy = Kokkos::MDRangePolicy< Kokkos::Rank<DIM, Kokkos::Iterate::Default, Kokkos::Iterate::Default> >;

  template <typename ViewType, std::size_t DIM>
  auto get_map_axes(const ViewType& view, std::array<int, DIM> _axes) {
    static_assert(ViewType::rank() >= DIM,
                  "KokkosFFT::get_map_axes: Rank of View must be larger thane or equal to the Rank of FFT axes.");
    static_assert(DIM > 0,
                  "KokkosFFT::get_map_axes: Rank of FFT axes must be larger than or equal to 1.");

    int rank = static_cast<int>( ViewType::rank() );
    using array_layout_type = typename ViewType::array_layout;

    std::vector<int> axes;
    for(std::size_t i=0; i<DIM; i++) {
      int _axis = _axes.at(i);
      assert( abs(_axis) < rank ); // axis should be in [-(rank-1), rank-1]
      int axis = _axis < 0 ? rank + _axis : _axis;
      axes.push_back(axis);
    }

    std::vector<int> inner_most_axes(DIM);
    if(std::is_same_v<array_layout_type, typename Kokkos::LayoutLeft>) {
      for(std::size_t i=0; i<DIM; i++) { inner_most_axes.at(i) = i; }
    } else {
      for(std::size_t i=0; i<DIM; i++) { inner_most_axes.at(i) = rank-1-i; }
    }

    std::vector<int> map(rank);
    for(int i=0; i<map.size(); i++) { map.at(i) = i; }

    // Make axes to the inner_most axes
    // E.g (0, 1, 2) -> (1, 2, 0) if axes=(1, 2) for layout left
    for(std::size_t i=0; i<DIM; i++) {
      auto inner_most_axis = inner_most_axes.at(i);
      auto axis = axes.at(i);

      map.at(inner_most_axis) = axis;
      map.at(axis) = inner_most_axis;
    }

    return map;
  }

  template <typename ViewType>
  auto get_map_axes(const ViewType& view, int axis) {
    return get_map_axes(view, std::array<int, 1>({axis}));
  }

  template <typename InViewType, typename OutViewType,
            std::enable_if_t<InViewType::rank()==1, std::nullptr_t> = nullptr>
  void _transpose(InViewType& in, OutViewType& out, int axis=0) {
    out = in;
  }

  template <typename InViewType, typename OutViewType,
            std::enable_if_t<InViewType::rank()==2, std::nullptr_t> = nullptr>
  void _transpose(InViewType& in, OutViewType& out, int axis=0) {
    std::size_t rank = InViewType::rank();
    using array_layout_type = typename InViewType::array_layout;

    using range_type = MDPolicy<2>;
    using tile_type = typename range_type::tile_type;
    using point_type = typename range_type::point_type;

    int n0 = in.extent(0), n1 = in.extent(1);

    range_type range(
      point_type{{0, 0}},
      point_type{{n0, n1}},
      tile_type{{4, 4}} // [TO DO] Choose optimal tile sizes for each device
    );

    int inner_most_axis = std::is_same_v<array_layout_type, typename Kokkos::LayoutLeft> ? 0 : rank - 1;
    if(axis == inner_most_axis) {
      out = in;
    } else {
      // Assign a View if not a shallow copy
      out = OutViewType("out", n1, n0);
      
      /*
      Kokkos::parallel_for(range,
        KOKKOS_LAMBDA (int i0, int i1) {
          out(i1, i0) = in(i0, i1);
        }
      );
      */
      
    }
  }

  template <typename InViewType, typename OutViewType,
            std::enable_if_t<InViewType::rank()==3, std::nullptr_t> = nullptr>
  void _transpose(InViewType& in, OutViewType& out, int axis=0) {
    std::size_t rank = InViewType::rank();
    using array_layout_type = typename InViewType::array_layout;

    using range_type = MDPolicy<3>;
    using tile_type = typename range_type::tile_type;
    using point_type = typename range_type::point_type;

    int n0 = in.extent(0), n1 = in.extent(1), n2 = in.extent(2);

    range_type range(
      point_type{{0, 0, 0}},
      point_type{{n0, n1, n2}},
      tile_type{{4, 4, 4}} // [TO DO] Choose optimal tile sizes for each device
    );

    int inner_most_axis = std::is_same_v<array_layout_type, typename Kokkos::LayoutLeft> ? 0 : rank - 1;

    if(axis == inner_most_axis) {
      out = in;
    } else {
      using MapView = Kokkos::View<int[3], typename InViewType::memory_space>;
      MapView map("map");
      auto h_map = Kokkos::create_mirror_view(map);

      // Original order 0, 1, 2
      for(int i=0; i<3; i++) {h_map(i) = i;}

      // Make axis to the inner_most_axis, say (0, 1, 2) -> (1, 0, 2) if axis=1 for layout left
      h_map(inner_most_axis) = axis;
      h_map(axis) = inner_most_axis;

      Kokkos::deep_copy(map, h_map);

      int _n0 = in.extent(h_map(0)), _n1 = in.extent(h_map(1)), _n2 = in.extent(h_map(2));

      // Assign a View if not a shallow copy
      out = OutViewType("out", _n0, _n1, _n2);
      /*
      Kokkos::parallel_for(
        range,
        KOKKOS_LAMBDA (const int i0, const int i1, const int i2) {
          int _i0 = i0, _i1 = i1, _i2 = i2;
          if(map(0) == 1) {
            _i0 = i1; _i1 = i0;
          } else if(map(0) == 2) {
            _i0 = i2; _i2 = i0;
          } else if(map(1) == 2) {
            _i1 = i2; _i2 = i1;
          }
          out(_i0, _i1, _i2) = in(i0, i1, i2);
        }
      );
      */
    }
  }

  /* Make the axis direction to the inner most direction 
     axis should be the range in [-(rank-1), rank-1], where 
     negative number is interpreted as rank + axis. 
     E.g. axis = -1 for rank 3 matrix is interpreted as axis = 2

   * E.g. 
          Layout Left
          A (3, 4, 2) and axis = 1 -> A' (4, 3, 2)
          B (2, 4, 3, 5) and axis = 2 -> B' (3, 4, 2, 5)
          C (8, 6, 3) and axis = 0 -> C' (8, 6, 3)
          D (7, 5) and axis = -1 -> D' (5, 7)

          Layout Right
          A (3, 4, 2) and axis = 1 -> A' (3, 2, 4)
          B (2, 4, 3, 5) and axis = 2 -> B' (2, 4, 5, 3)
          C (8, 6, 3) and axis = 0 -> C' (3, 6, 8)
          D (5, 7) and axis = -1 -> D' (5, 7)
   *
  */
  template <typename InViewType, typename OutViewType>
  void transpose(InViewType& in, OutViewType& out, int _axis=0) {
    using in_value_type = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;
    using array_layout_type = typename InViewType::array_layout;

    static_assert(Kokkos::is_view<InViewType>::value,
                  "KokkosFFT::create: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<InViewType>::value,
                  "KokkosFFT::create: OutViewType is not a Kokkos::View.");

    static_assert(InViewType::rank() == OutViewType::rank(),
                  "KokkosFFT::create: InViewType and OutViewType must have the same rank.");
      
    // [TO DO]
    // check the layouts are contiguous
    std::size_t rank = InViewType::rank();
    assert( abs(_axis) < rank ); // axis should be in [-(rank-1), rank-1]
    int axis = _axis < 0 ? static_cast<int>(rank) + _axis : _axis;
    _transpose(in, out, axis);
  }


};

#endif