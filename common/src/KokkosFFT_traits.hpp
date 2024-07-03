// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_TRAITS_HPP
#define KOKKOSFFT_TRAITS_HPP

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

template <typename T, typename Enable = void>
struct is_real : std::false_type {};

template <typename T>
struct is_real<
    T, std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_real_v = is_real<T>::value;

template <typename T, typename Enable = void>
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<
    Kokkos::complex<T>,
    std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

// is value type admissible for KokkosFFT
template <typename T, typename Enable = void>
struct is_admissible_value_type : std::false_type {};

template <typename T>
struct is_admissible_value_type<
    T, std::enable_if_t<is_real_v<T> || is_complex_v<T>>> : std::true_type {};

template <typename T>
struct is_admissible_value_type<
    T, std::enable_if_t<Kokkos::is_view<T>::value &&
                        (is_real_v<typename T::non_const_value_type> ||
                         is_complex_v<typename T::non_const_value_type>)>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_admissible_value_type_v =
    is_admissible_value_type<T>::value;

// is layout admissible for KokkosFFT
template <typename ViewType, typename Enable = void>
struct is_layout_left_or_right : std::false_type {};

template <typename ViewType>
struct is_layout_left_or_right<
    ViewType,
    std::enable_if_t<
        Kokkos::is_view<ViewType>::value &&
        (std::is_same_v<typename ViewType::array_layout, Kokkos::LayoutLeft> ||
         std::is_same_v<typename ViewType::array_layout, Kokkos::LayoutRight>)>>
    : std::true_type {};

template <typename ViewType>
inline constexpr bool is_layout_left_or_right_v =
    is_layout_left_or_right<ViewType>::value;

// is view admissible for KokkosFFT
template <typename ViewType, typename Enable = void>
struct is_admissible_view : std::false_type {};

template <typename ViewType>
struct is_admissible_view<
    ViewType, std::enable_if_t<Kokkos::is_view<ViewType>::value &&
                               is_layout_left_or_right_v<ViewType> &&
                               is_admissible_value_type_v<ViewType>>>
    : std::true_type {};

template <typename ViewType>
inline constexpr bool is_admissible_view_v =
    is_admissible_view<ViewType>::value;

template <typename T>
struct managable_view_type {
  using type = Kokkos::View<typename T::data_type, typename T::array_layout,
                            typename T::memory_space,
                            Kokkos::MemoryTraits<T::memory_traits::impl_value &
                                                 ~unsigned(Kokkos::Unmanaged)>>;
};

template <typename ExecutionSpace, typename ViewType,
          std::enable_if_t<ViewType::rank() == 1, std::nullptr_t> = nullptr>
struct complex_view_type {
  using value_type        = typename ViewType::non_const_value_type;
  using float_type        = KokkosFFT::Impl::real_type_t<value_type>;
  using complex_type      = Kokkos::complex<float_type>;
  using array_layout_type = typename ViewType::array_layout;
  using type = Kokkos::View<complex_type*, array_layout_type, ExecutionSpace>;
};

}  // namespace Impl
}  // namespace KokkosFFT

#endif
