#ifndef KOKKOSFFT_PADDING_HPP
#define KOKKOSFFT_PADDING_HPP

#include <tuple>
#include "KokkosFFT_default_types.hpp"
#include "KokkosFFT_utils.hpp"

namespace KokkosFFT {
namespace Impl {
template <typename ViewType, std::size_t DIM>
auto get_modified_shape(const ViewType& view, shape_type<DIM> shape) {
  static_assert(ViewType::rank() >= DIM,
                "KokkosFFT::get_modified_shape: Rank of View must be larger "
                "than or equal to the Rank of new shape");
  static_assert(DIM > 0,
                "KokkosFFT::get_modified_shape: Rank of FFT axes must be "
                "larger than or equal to 1");

  // [TO DO] Add a is_C2R arg. If is_C2R is true, then shape should be shape/2+1
  constexpr int rank = static_cast<int>(ViewType::rank());

  using full_shape_type = shape_type<rank>;
  full_shape_type modified_shape;
  for (int i = 0; i < rank; i++) {
    modified_shape.at(i) = view.extent(i);
  }

  // Update shapes based on newly given shape
  for (int i = 0; i < DIM; i++) {
    assert(shape.at(i) > 0);
    modified_shape.at(i) = shape.at(i);
  }

  // [TO DO] may return, is_modification_needed if modified_shape is not equal
  // view.extents() May be implement other function. is_crop_or_pad_needed(view,
  // shape);
  return modified_shape;
}

template <typename ViewType, std::size_t DIM>
auto is_crop_or_pad_needed(const ViewType& view,
                           const shape_type<DIM>& modified_shape) {
  static_assert(ViewType::rank() == DIM,
                "KokkosFFT::_crop_or_pad: Rank of View must be equal to Rank "
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

template <typename ExecutionSpace, typename ViewType>
void _crop_or_pad(const ExecutionSpace& exec_space, const ViewType& in,
                  ViewType& out, shape_type<1> s) {
  constexpr std::size_t DIM = 1;
  static_assert(ViewType::rank() == DIM,
                "KokkosFFT::_crop_or_pad: Rank of View must be equal to Rank "
                "of extended shape.");

  auto _n0 = s.at(0);
  out      = ViewType("out", _n0);

  auto n0 = std::min(_n0, in.extent(0));

  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>(
          exec_space, 0, n0),
      KOKKOS_LAMBDA(int i0) { out(i0) = in(i0); });
}

template <typename ExecutionSpace, typename ViewType>
void _crop_or_pad(const ExecutionSpace& exec_space, const ViewType& in,
                  ViewType& out, shape_type<2> s) {
  constexpr std::size_t DIM = 2;
  static_assert(ViewType::rank() == DIM,
                "KokkosFFT::_crop_or_pad: Rank of View must be equal to Rank "
                "of extended shape.");

  auto [_n0, _n1] = s;
  out             = ViewType("out", _n0, _n1);

  int n0 = std::min(_n0, in.extent(0));
  int n1 = std::min(_n1, in.extent(1));

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<2, Kokkos::Iterate::Default, Kokkos::Iterate::Default>>;
  using tile_type  = typename range_type::tile_type;
  using point_type = typename range_type::point_type;

  range_type range(point_type{{0, 0}}, point_type{{n0, n1}}, tile_type{{4, 4}}
                   // [TO DO] Choose optimal tile sizes for each device
  );

  Kokkos::parallel_for(
      range, KOKKOS_LAMBDA(int i0, int i1) { out(i0, i1) = in(i0, i1); });
}

template <typename ExecutionSpace, typename ViewType>
void _crop_or_pad(const ExecutionSpace& exec_space, const ViewType& in,
                  ViewType& out, shape_type<3> s) {
  constexpr std::size_t DIM = 3;
  static_assert(ViewType::rank() == DIM,
                "KokkosFFT::_crop_or_pad: Rank of View must be equal to Rank "
                "of extended shape.");

  auto [_n0, _n1, _n2] = s;
  out                  = ViewType("out", _n0, _n1, _n2);

  int n0 = std::min(_n0, in.extent(0));
  int n1 = std::min(_n1, in.extent(1));
  int n2 = std::min(_n2, in.extent(2));

  using range_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<3, Kokkos::Iterate::Default, Kokkos::Iterate::Default>>;
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

template <typename ExecutionSpace, typename ViewType, std::size_t DIM = 1>
void crop_or_pad(const ExecutionSpace& exec_space, const ViewType& in,
                 ViewType& out, shape_type<DIM> s) {
  _crop_or_pad(exec_space, in, out, s);
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif