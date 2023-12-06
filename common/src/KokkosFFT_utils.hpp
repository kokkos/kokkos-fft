#ifndef __KOKKOSFFT_UTILS_HPP__
#define __KOKKOSFFT_UTILS_HPP__

#include <Kokkos_Core.hpp>
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>

namespace KokkosFFT {
  template <typename T>
  struct real_type {
    using type = T;
  };

  template <typename T>
  struct real_type <Kokkos::complex<T>>{
    using type = T;
  };

  template <typename T>
  using real_type_t = typename real_type<T>::type;

  template <typename T>
  struct is_complex : std::false_type {};

  template <typename T>
  struct is_complex<Kokkos::complex<T>> : std::true_type {};

  template <typename ViewType>
  auto convert_negative_axis(const ViewType& view, int _axis=-1) {
    static_assert(Kokkos::is_view<ViewType>::value,
                  "KokkosFFT::convert_negative_axis: ViewType is not a Kokkos::View.");
    int rank = static_cast<int>( ViewType::rank() );
    assert( abs(_axis) < rank ); // axis should be in [-(rank-1), rank-1]
    int axis = _axis < 0 ? rank + _axis : _axis;
    return axis;
  }

  template <typename T>
  void permute(std::vector<T>& values, const std::vector<std::size_t>& indices) {
    std::vector<T> out;
    out.reserve(indices.size());
    for(auto index: indices) {
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

  template <std::size_t DIM=1>
  bool is_transpose_needed(std::array<int, DIM> map) {
    std::array<int, DIM> contiguous_map;
    std::iota(contiguous_map.begin(), contiguous_map.end(), 0);
    return map != contiguous_map;
  }

  template <typename T>
  std::size_t get_index(std::vector<T>& values, const T& key) {
    auto it = find(values.begin(), values.end(), key);
    std::size_t index = 0;
    if(it != values.end()) {
      index = it - values.begin();
    } else {
      throw std::runtime_error("key is not included in values");
    }

    return index;
  }

  template <typename T, std::size_t... I>
  std::array<T, sizeof...(I)> make_sequence_array(std::index_sequence<I...>) {
    return std::array<T, sizeof...(I)>{ {I...} };
  }

  template <int N, typename T>
  std::array<T, N> index_sequence(T const& start) {
    auto sequence = make_sequence_array<T>(std::make_index_sequence<N>());
    std::transform(sequence.begin(), sequence.end(), sequence.begin(),
                   [=](const T sequence) -> T {return start + sequence;});
    return sequence;
  }


};

#endif