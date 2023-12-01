#ifndef __KOKKOSFFT_UTILS_HPP__
#define __KOKKOSFFT_UTILS_HPP__

#include <Kokkos_Core.hpp>
#include <vector>

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
};

#endif