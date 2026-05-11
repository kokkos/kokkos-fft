// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_UTILS_HPP
#define KOKKOSFFT_UTILS_HPP

#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <limits>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Asserts.hpp"
#include "KokkosFFT_Common_Types.hpp"
#include "KokkosFFT_Traits.hpp"

namespace KokkosFFT {
namespace Impl {
template <typename T, std::size_t N, std::size_t... Is>
constexpr Kokkos::Array<std::remove_cv_t<T>, N> to_array_lvalue(
    std::array<T, N>& a, std::index_sequence<Is...>) {
  return {{a[Is]...}};
}
template <typename T, std::size_t N, std::size_t... Is>
constexpr Kokkos::Array<std::remove_cv_t<T>, N> to_array_rvalue(
    std::array<T, N>&& a, std::index_sequence<Is...>) {
  return {{std::move(a[Is])...}};
}

template <typename T, std::size_t N>
constexpr Kokkos::Array<T, N> to_array(std::array<T, N>& a) {
  return to_array_lvalue(a, std::make_index_sequence<N>());
}
template <typename T, std::size_t N>
constexpr Kokkos::Array<T, N> to_array(std::array<T, N>&& a) {
  return to_array_rvalue(std::move(a), std::make_index_sequence<N>());
}

/// \brief Convert a std::array to std::vector
/// \tparam ArrayType The type of the std::array
/// \param[in, out] arr The input std::array
/// \return A std::vector containing the elements of the input array
template <typename ArrayType>
auto to_vector(ArrayType&& arr) {
  using array_type = std::decay_t<ArrayType>;
  static_assert(KokkosFFT::Impl::is_std_array_v<array_type>,
                "to_vector: Input type must be a std::array");

  using value_type        = typename array_type::value_type;
  constexpr std::size_t N = std::tuple_size_v<array_type>;

  if constexpr (std::is_rvalue_reference_v<ArrayType&&>) {
    // Move elements from the rvalue array
    std::vector<value_type> vec;
    vec.reserve(N);
    std::move(arr.begin(), arr.end(), std::back_inserter(vec));
    return vec;
  } else {
    // Copy elements from the lvalue array
    std::vector<value_type> vec(arr.begin(), arr.end());
    return vec;
  }
}

}  // namespace Impl
}  // namespace KokkosFFT

#endif
