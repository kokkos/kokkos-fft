// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_UTILS_HPP
#define KOKKOSFFT_UTILS_HPP

#include <Kokkos_Core.hpp>
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
#include "KokkosFFT_traits.hpp"

#if defined(KOKKOS_ENABLE_CXX17)
#include <cstdlib>
#else
#include <source_location>
#endif

namespace KokkosFFT {
namespace Impl {

template <typename ViewType>
auto convert_negative_axis(ViewType, int _axis = -1) {
  static_assert(Kokkos::is_view<ViewType>::value,
                "convert_negative_axis: ViewType is not a Kokkos::View.");
  int rank = static_cast<int>(ViewType::rank());
  if (_axis < -rank || _axis >= rank) {
    throw std::runtime_error("axis should be in [-rank, rank-1]");
  }

  int axis = _axis < 0 ? rank + _axis : _axis;
  return axis;
}

template <typename ViewType>
auto convert_negative_shift(const ViewType& view, int _shift, int _axis) {
  static_assert(Kokkos::is_view<ViewType>::value,
                "convert_negative_shift: ViewType is not a Kokkos::View.");
  int axis                    = convert_negative_axis(view, _axis);
  int extent                  = view.extent(axis);
  [[maybe_unused]] int shift0 = 0, shift1 = 0, shift2 = extent / 2;

  if (_shift < 0) {
    shift0 = -_shift + extent % 2;  // add 1 for odd case
    shift1 = -_shift;
    shift2 = 0;
  } else if (_shift > 0) {
    shift0 = _shift;
    shift1 = _shift + extent % 2;  // add 1 for odd case
    shift2 = 0;
  }

  return std::tuple<int, int, int>({shift0, shift1, shift2});
}

template <typename T>
bool is_found(std::vector<T>& values, const T& key) {
  return std::find(values.begin(), values.end(), key) != values.end();
}

template <typename T>
bool has_duplicate_values(const std::vector<T>& values) {
  std::set<T> set_values(values.begin(), values.end());
  return set_values.size() < values.size();
}

template <typename IntType, std::enable_if_t<std::is_integral_v<IntType>,
                                             std::nullptr_t> = nullptr>
bool is_out_of_range_value_included(const std::vector<IntType>& values,
                                    IntType max) {
  bool is_included = false;
  for (auto value : values) {
    is_included = value >= max;
  }
  return is_included;
}

template <std::size_t DIM = 1>
bool is_transpose_needed(std::array<int, DIM> map) {
  std::array<int, DIM> contiguous_map;
  std::iota(contiguous_map.begin(), contiguous_map.end(), 0);
  return map != contiguous_map;
}

template <typename T>
std::size_t get_index(std::vector<T>& values, const T& key) {
  auto it           = find(values.begin(), values.end(), key);
  std::size_t index = 0;
  if (it != values.end()) {
    index = it - values.begin();
  } else {
    throw std::runtime_error("key is not included in values");
  }

  return index;
}

template <typename T, std::size_t... I>
std::array<T, sizeof...(I)> make_sequence_array(std::index_sequence<I...>) {
  return std::array<T, sizeof...(I)>{{I...}};
}

template <int N, typename T>
std::array<T, N> index_sequence(T const& start) {
  auto sequence = make_sequence_array<T>(std::make_index_sequence<N>());
  std::transform(sequence.begin(), sequence.end(), sequence.begin(),
                 [=](const T sequence) -> T { return start + sequence; });
  return sequence;
}

template <typename ElementType>
inline std::vector<ElementType> arange(const ElementType start,
                                       const ElementType stop,
                                       const ElementType step = 1) {
  const size_t length = ceil((stop - start) / step);
  std::vector<ElementType> result;
  ElementType delta = (stop - start) / length;

  // thrust::sequence
  for (std::size_t i = 0; i < length; i++) {
    ElementType value = start + delta * i;
    result.push_back(value);
  }
  return result;
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void conjugate(const ExecutionSpace& exec_space, const InViewType& in,
               OutViewType& out) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "conjugate: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "conjugate: OutViewType is not a Kokkos::View.");

  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(KokkosFFT::Impl::is_complex<out_value_type>::value,
                "conjugate: OutViewType must be complex");
  std::size_t size = in.size();
  out              = OutViewType("out", in.layout());

  auto* in_data  = in.data();
  auto* out_data = out.data();

  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>(
          exec_space, 0, size),
      KOKKOS_LAMBDA(std::size_t i) { out_data[i] = Kokkos::conj(in_data[i]); });
}

template <typename ViewType>
auto extract_extents(const ViewType& view) {
  static_assert(Kokkos::is_view<ViewType>::value,
                "extract_extents: ViewType is not a Kokkos::View.");
  constexpr std::size_t rank = ViewType::rank();
  std::array<std::size_t, rank> extents;
  for (std::size_t i = 0; i < rank; i++) {
    extents.at(i) = view.extent(i);
  }
  return extents;
}

template <typename ViewType, typename Label>
void create_view(ViewType& out, const Label& label,
                 const std::array<int, 1>& extents) {
  out = ViewType(label, extents[0]);
}

template <typename ViewType, typename Label>
void create_view(ViewType& out, const Label& label,
                 const std::array<int, 2>& extents) {
  out = ViewType(label, extents[0], extents[1]);
}

template <typename ViewType, typename Label>
void create_view(ViewType& out, const Label& label,
                 const std::array<int, 3>& extents) {
  out = ViewType(label, extents[0], extents[1], extents[2]);
}

template <typename ViewType, typename Label>
void create_view(ViewType& out, const Label& label,
                 const std::array<int, 4>& extents) {
  out = ViewType(label, extents[0], extents[1], extents[2], extents[3]);
}

template <typename ViewType, typename Label>
void create_view(ViewType& out, const Label& label,
                 const std::array<int, 5>& extents) {
  out = ViewType(label, extents[0], extents[1], extents[2], extents[3],
                 extents[4]);
}

template <typename ViewType, typename Label>
void create_view(ViewType& out, const Label& label,
                 const std::array<int, 6>& extents) {
  out = ViewType(label, extents[0], extents[1], extents[2], extents[3],
                 extents[4], extents[5]);
}

template <typename ViewType, typename Label>
void create_view(ViewType& out, const Label& label,
                 const std::array<int, 7>& extents) {
  out = ViewType(label, extents[0], extents[1], extents[2], extents[3],
                 extents[4], extents[5], extents[6]);
}

template <typename ViewType, typename Label>
void create_view(ViewType& out, const Label& label,
                 const std::array<int, 8>& extents) {
  out = ViewType(label, extents[0], extents[1], extents[2], extents[3],
                 extents[4], extents[5], extents[6], extents[7]);
}

template <typename ViewType>
void reshape_view(ViewType& out, const std::array<int, 1>& extents) {
  if (ViewType::required_allocation_size(out.layout()) >=
      ViewType::required_allocation_size(extents[0])) {
    out = ViewType(out.data(), extents[0]);
  } else {
    throw std::runtime_error("reshape_view: insufficient memory");
  }
}

template <typename ViewType>
void reshape_view(ViewType& out, const std::array<int, 2>& extents) {
  if (ViewType::required_allocation_size(out.layout()) >=
      ViewType::required_allocation_size(extents[0], extents[1])) {
    out = ViewType(out.data(), extents[0], extents[1]);
  } else {
    throw std::runtime_error("reshape_view: insufficient memory");
  }
}

template <typename ViewType>
void reshape_view(ViewType& out, const std::array<int, 3>& extents) {
  if (ViewType::required_allocation_size(out.layout()) >=
      ViewType::required_allocation_size(extents[0], extents[1], extents[2])) {
    out = ViewType(out.data(), extents[0], extents[1], extents[2]);
  } else {
    throw std::runtime_error("reshape_view: insufficient memory");
  }
}

template <typename ViewType>
void reshape_view(ViewType& out, const std::array<int, 4>& extents) {
  if (ViewType::required_allocation_size(out.layout()) >=
      ViewType::required_allocation_size(extents[0], extents[1], extents[2],
                                         extents[3])) {
    out = ViewType(out.data(), extents[0], extents[1], extents[2], extents[3]);
  } else {
    throw std::runtime_error("reshape_view: insufficient memory");
  }
}

template <typename ViewType>
void reshape_view(ViewType& out, const std::array<int, 5>& extents) {
  if (ViewType::required_allocation_size(out.layout()) >=
      ViewType::required_allocation_size(extents[0], extents[1], extents[2],
                                         extents[3], extents[4])) {
    out = ViewType(out.data(), extents[0], extents[1], extents[2], extents[3],
                   extents[4]);
  } else {
    throw std::runtime_error("reshape_view: insufficient memory");
  }
}

template <typename ViewType>
void reshape_view(ViewType& out, const std::array<int, 6>& extents) {
  if (ViewType::required_allocation_size(out.layout()) >=
      ViewType::required_allocation_size(extents[0], extents[1], extents[2],
                                         extents[3], extents[4], extents[5])) {
    out = ViewType(out.data(), extents[0], extents[1], extents[2], extents[3],
                   extents[4], extents[5]);
  } else {
    throw std::runtime_error("reshape_view: insufficient memory");
  }
}

template <typename ViewType>
void reshape_view(ViewType& out, const std::array<int, 7>& extents) {
  if (ViewType::required_allocation_size(out.layout()) >=
      ViewType::required_allocation_size(extents[0], extents[1], extents[2],
                                         extents[3], extents[4], extents[5],
                                         extents[6])) {
    out = ViewType(out.data(), extents[0], extents[1], extents[2], extents[3],
                   extents[4], extents[5], extents[6]);
  } else {
    throw std::runtime_error("reshape_view: insufficient memory");
  }
}

template <typename ViewType>
void reshape_view(ViewType& out, const std::array<int, 8>& extents) {
  if (ViewType::required_allocation_size(out.layout()) >=
      ViewType::required_allocation_size(extents[0], extents[1], extents[2],
                                         extents[3], extents[4], extents[5],
                                         extents[6], extents[7])) {
    out = ViewType(out.data(), extents[0], extents[1], extents[2], extents[3],
                   extents[4], extents[5], extents[6], extents[7]);
  } else {
    throw std::runtime_error("reshape_view: insufficient memory");
  }
}

}  // namespace Impl
}  // namespace KokkosFFT

#endif
