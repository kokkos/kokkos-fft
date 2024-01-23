#ifndef KOKKOSFFT_UTILS_HPP
#define KOKKOSFFT_UTILS_HPP

#include <Kokkos_Core.hpp>
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>

namespace KokkosFFT {
namespace Impl {
template <typename T>
struct real_type {
  using type = T;
};

template <typename T>
struct real_type<Kokkos::complex<T>> {
  using type = T;
};

template <typename T>
using real_type_t = typename real_type<T>::type;

template <typename T>
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<Kokkos::complex<T>> : std::true_type {};

template <typename ViewType, typename Enable = void>
struct is_layout_left_or_right : std::false_type {};

template <typename ViewType>
struct is_layout_left_or_right<
    ViewType,
    std::enable_if_t<
        std::is_same_v<typename ViewType::array_layout, Kokkos::LayoutLeft> ||
        std::is_same_v<typename ViewType::array_layout, Kokkos::LayoutRight>>>
    : std::true_type {};

template <typename ViewType>
inline constexpr bool is_layout_left_or_right_v =
    is_layout_left_or_right<ViewType>::value;

template <typename ExecutionSpace, typename ViewType,
          std::enable_if_t<ViewType::rank() == 1, std::nullptr_t> = nullptr>
struct complex_view_type {
  using value_type        = typename ViewType::non_const_value_type;
  using float_type        = KokkosFFT::Impl::real_type_t<value_type>;
  using complex_type      = Kokkos::complex<float_type>;
  using array_layout_type = typename ViewType::array_layout;
  using type = Kokkos::View<complex_type*, array_layout_type, ExecutionSpace>;
};

template <typename ViewType>
auto convert_negative_axis(const ViewType& view, int _axis = -1) {
  static_assert(Kokkos::is_view<ViewType>::value,
                "convert_negative_axis: ViewType is not a Kokkos::View.");
  int rank = static_cast<int>(ViewType::rank());
  assert(abs(_axis) < rank);  // axis should be in [-(rank-1), rank-1]
  int axis = _axis < 0 ? rank + _axis : _axis;
  return axis;
}

template <typename ViewType>
auto convert_negative_shift(const ViewType& view, int _shift, int _axis) {
  static_assert(Kokkos::is_view<ViewType>::value,
                "convert_negative_shift: ViewType is not a Kokkos::View.");
  int axis   = convert_negative_axis(view, _axis);
  int extent = view.extent(axis);
  int shift0 = 0, shift1 = 0, shift2 = extent / 2;

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
void permute(std::vector<T>& values, const std::vector<std::size_t>& indices) {
  std::vector<T> out;
  out.reserve(indices.size());
  for (auto index : indices) {
    out.push_back(values.at(index));
  }
  values = std::move(out);
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

template <
    typename IntType, std::size_t DIM = 1,
    std::enable_if_t<std::is_integral_v<IntType>, std::nullptr_t> = nullptr>
bool is_out_of_range_value_included(const std::array<IntType, DIM>& values,
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
  for (auto i = 0; i < length; i++) {
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

  using in_value_type  = typename InViewType::non_const_value_type;
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
      KOKKOS_LAMBDA(const int& i) {
        out_value_type tmp = in_data[i];
        out_data[i]        = Kokkos::conj(in_data[i]);
      });
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

}  // namespace Impl
}  // namespace KokkosFFT

#endif