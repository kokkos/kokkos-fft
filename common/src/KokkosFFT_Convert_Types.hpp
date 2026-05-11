// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_CONVERT_TYPES_HPP
#define KOKKOSFFT_CONVERT_TYPES_HPP

#include <algorithm>
#include <array>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include "KokkosFFT_Asserts.hpp"
#include "KokkosFFT_Common_Types.hpp"
#include "KokkosFFT_Traits.hpp"

namespace KokkosFFT {
namespace Impl {
/// \brief Convert a lvalue std::array to Kokkos::Array
/// \tparam To The target integral type
/// \tparam N The number of elements in the array
///
/// \tparam Is A parameter pack of indices for the array elements
/// \param[in] a The input array to be converted
/// \return A new Kokkos::Array with the same number of elements as the input
/// array
template <typename T, std::size_t N, std::size_t... Is>
constexpr Kokkos::Array<std::remove_cv_t<T>, N> to_array_lvalue(
    std::array<T, N>& a, std::index_sequence<Is...>) {
  return {{a[Is]...}};
}

/// \brief Convert a std::array to Kokkos::Array by moving elements from an
/// rvalue std::array \tparam To The target integral type \tparam N The number
/// of elements in the array \tparam Is A parameter pack of indices for the
/// array elements
///
/// \param[in, out] a The input array to be converted, which will be moved from
/// \return A new Kokkos::Array with the same number of elements as the input
/// array
template <typename T, std::size_t N, std::size_t... Is>
constexpr Kokkos::Array<std::remove_cv_t<T>, N> to_array_rvalue(
    std::array<T, N>&& a, std::index_sequence<Is...>) {
  return {{std::move(a[Is])...}};
}

/// \brief Convert a std::array to Kokkos::Array
/// \tparam T The type of the elements in the array
/// \tparam N The number of elements in the array
///
/// \param[in] a The input lvalue std::array
/// \return A Kokkos::Array containing the elements of the input array
template <typename T, std::size_t N>
constexpr Kokkos::Array<T, N> to_array(std::array<T, N>& a) {
  return to_array_lvalue(a, std::make_index_sequence<N>());
}

/// \brief Convert a std::array to Kokkos::Array by moving elements from an
/// rvalue std::array \tparam T The type of the elements in the array \tparam N
/// The number of elements in the array
///
/// \param[in, out] a The input rvalue std::array
/// \return A Kokkos::Array containing the elements of the input array
template <typename T, std::size_t N>
constexpr Kokkos::Array<T, N> to_array(std::array<T, N>&& a) {
  return to_array_rvalue(std::move(a), std::make_index_sequence<N>());
}

/// \brief Convert a std::array to std::vector
/// \tparam ArrayType The type of the std::array
///
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
