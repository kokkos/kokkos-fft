// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_TRAITS_HPP
#define KOKKOSFFT_TRAITS_HPP

#include <Kokkos_Core.hpp>

namespace KokkosFFT {
namespace Impl {

// Traits for unary operation

template <typename T>
struct base_floating_point {
  using value_type = T;
};

template <typename T>
struct base_floating_point<Kokkos::complex<T>> {
  using value_type = T;
};

/// \brief Helper to extract the base floating point type from a complex type
template <typename T>
using base_floating_point_type = typename base_floating_point<T>::value_type;

template <typename T, typename Enable = void>
struct is_real : std::false_type {};

template <typename T>
struct is_real<
    T, std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>>
    : std::true_type {};

/// \brief Helper to check if a type is an acceptable real type (float/double)
/// for Kokkos-FFT
template <typename T>
inline constexpr bool is_real_v = is_real<T>::value;

template <typename T, typename Enable = void>
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<
    Kokkos::complex<T>,
    std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>>
    : std::true_type {};

/// \brief Helper to check if a type is an acceptable complex type
/// (Kokkos::complex<float>/Kokkos::complex<double>) for Kokkos-FFT
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
    T, std::enable_if_t<Kokkos::is_view_v<T> &&
                        (is_real_v<typename T::non_const_value_type> ||
                         is_complex_v<typename T::non_const_value_type>)>>
    : std::true_type {};

/// \brief Helper to check if a type is an acceptable value type
/// (float/double/Kokkos::complex<float>/Kokkos::complex<double>) for Kokkos-FFT
/// When applied to Kokkos::View, then check if a value type is an acceptable
/// real/complex type.
template <typename T>
inline constexpr bool is_admissible_value_type_v =
    is_admissible_value_type<T>::value;

template <typename ViewType, typename Enable = void>
struct is_layout_left_or_right : std::false_type {};

template <typename ViewType>
struct is_layout_left_or_right<
    ViewType,
    std::enable_if_t<
        Kokkos::is_view_v<ViewType> &&
        (std::is_same_v<typename ViewType::array_layout, Kokkos::LayoutLeft> ||
         std::is_same_v<typename ViewType::array_layout, Kokkos::LayoutRight>)>>
    : std::true_type {};

/// \brief Helper to check if a View layout is an acceptable layout type
/// (Kokkos::LayoutLeft/Kokkos::LayoutRight) for Kokkos-FFT
template <typename ViewType>
inline constexpr bool is_layout_left_or_right_v =
    is_layout_left_or_right<ViewType>::value;

template <typename ViewType, typename Enable = void>
struct is_admissible_view : std::false_type {};

template <typename ViewType>
struct is_admissible_view<
    ViewType, std::enable_if_t<Kokkos::is_view_v<ViewType> &&
                               is_layout_left_or_right_v<ViewType> &&
                               is_admissible_value_type_v<ViewType>>>
    : std::true_type {};

/// \brief Helper to check if a View is an acceptable for Kokkos-FFT. Values and
/// layout are checked
template <typename ViewType>
inline constexpr bool is_admissible_view_v =
    is_admissible_view<ViewType>::value;

template <typename ExecutionSpace, typename ViewType, typename Enable = void>
struct is_operatable_view : std::false_type {};

template <typename ExecutionSpace, typename ViewType>
struct is_operatable_view<
    ExecutionSpace, ViewType,
    std::enable_if_t<
        Kokkos::is_execution_space_v<ExecutionSpace> &&
        is_admissible_view_v<ViewType> &&
        Kokkos::SpaceAccessibility<
            ExecutionSpace, typename ViewType::memory_space>::accessible>>
    : std::true_type {};

/// \brief Helper to check if a View is an acceptable View for Kokkos-FFT and
/// memory space is accessible from the ExecutionSpace
template <typename ExecutionSpace, typename ViewType>
inline constexpr bool is_operatable_view_v =
    is_operatable_view<ExecutionSpace, ViewType>::value;

// Traits for binary operations
template <typename T1, typename T2, typename Enable = void>
struct have_same_precision : std::false_type {};

template <typename T1, typename T2>
struct have_same_precision<
    T1, T2,
    std::enable_if_t<!Kokkos::is_view_v<T1> && !Kokkos::is_view_v<T2> &&
                     std::is_same_v<base_floating_point_type<T1>,
                                    base_floating_point_type<T2>>>>
    : std::true_type {};

template <typename InViewType, typename OutViewType>
struct have_same_precision<
    InViewType, OutViewType,
    std::enable_if_t<
        Kokkos::is_view_v<InViewType> && Kokkos::is_view_v<OutViewType> &&
        std::is_same_v<
            base_floating_point_type<typename InViewType::non_const_value_type>,
            base_floating_point_type<
                typename OutViewType::non_const_value_type>>>>
    : std::true_type {};

/// \brief Helper to check if two value have the same base precision type.
/// When applied to Kokkos::View, then check if values of views have the same
/// base precision type.
template <typename T1, typename T2>
inline constexpr bool have_same_precision_v =
    have_same_precision<T1, T2>::value;

template <typename InViewType, typename OutViewType, typename Enable = void>
struct have_same_layout : std::false_type {};

template <typename InViewType, typename OutViewType>
struct have_same_layout<
    InViewType, OutViewType,
    std::enable_if_t<Kokkos::is_view_v<InViewType> &&
                     Kokkos::is_view_v<OutViewType> &&
                     std::is_same_v<typename InViewType::array_layout,
                                    typename OutViewType::array_layout>>>
    : std::true_type {};

/// \brief Helper to check if two views have the same layout type.
template <typename InViewType, typename OutViewType>
inline constexpr bool have_same_layout_v =
    have_same_layout<InViewType, OutViewType>::value;

template <typename InViewType, typename OutViewType, typename Enable = void>
struct have_same_rank : std::false_type {};

template <typename InViewType, typename OutViewType>
struct have_same_rank<
    InViewType, OutViewType,
    std::enable_if_t<Kokkos::is_view_v<InViewType> &&
                     Kokkos::is_view_v<OutViewType> &&
                     InViewType::rank() == OutViewType::rank()>>
    : std::true_type {};

/// \brief Helper to check if two views have the same rank.
template <typename InViewType, typename OutViewType>
inline constexpr bool have_same_rank_v =
    have_same_rank<InViewType, OutViewType>::value;

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename Enable = void>
struct are_operatable_views : std::false_type {};

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
struct are_operatable_views<
    ExecutionSpace, InViewType, OutViewType,
    std::enable_if_t<is_operatable_view_v<ExecutionSpace, InViewType> &&
                     is_operatable_view_v<ExecutionSpace, OutViewType> &&
                     have_same_precision_v<InViewType, OutViewType> &&
                     have_same_layout_v<InViewType, OutViewType> &&
                     have_same_rank_v<InViewType, OutViewType>>>
    : std::true_type {};

/// \brief Helper to check if Views are acceptable View for Kokkos-FFT and
/// memory space are accessible from the ExecutionSpace.
/// In addition, precisions, layout and rank are checked to be identical.
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
inline constexpr bool are_operatable_views_v =
    are_operatable_views<ExecutionSpace, InViewType, OutViewType>::value;

// Other traits

/// \brief Helper to define a managable View type from the original view type
template <typename T>
struct managable_view_type {
  using type = Kokkos::View<typename T::data_type, typename T::array_layout,
                            typename T::memory_space,
                            Kokkos::MemoryTraits<T::memory_traits::impl_value &
                                                 ~unsigned(Kokkos::Unmanaged)>>;
};

/// \brief Helper to define a complex 1D View type from a real/complex 1D View
/// type, while keeping other properties
template <typename ExecutionSpace, typename ViewType,
          std::enable_if_t<ViewType::rank() == 1, std::nullptr_t> = nullptr>
struct complex_view_type {
  using value_type   = typename ViewType::non_const_value_type;
  using float_type   = KokkosFFT::Impl::base_floating_point_type<value_type>;
  using complex_type = Kokkos::complex<float_type>;
  using array_layout_type = typename ViewType::array_layout;
  using type = Kokkos::View<complex_type*, array_layout_type, ExecutionSpace>;
};

}  // namespace Impl
}  // namespace KokkosFFT

#endif
